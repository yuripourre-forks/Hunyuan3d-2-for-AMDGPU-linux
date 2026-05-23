# Hunyuan 3D service layer for REST API (no Gradio).
# Licensed under the same terms as gradio_app.py / upstream Hunyuan3D-2.

from __future__ import annotations

import json
import os
import random
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import trimesh
from PIL import Image

from hy3dgen.shapegen.pipelines import export_to_trimesh
from hy3dgen.shapegen.utils import logger

MAX_SEED = int(1e7)
SUPPORTED_FORMATS = ("glb", "obj", "ply", "stl")
MV_VIEW_NAMES = ("front", "back", "left", "right")


class TextureUnavailableError(RuntimeError):
    """Texture pipeline is disabled or failed to load."""


class MultiviewUnavailableError(RuntimeError):
    """Multiview pipeline is disabled or not loaded."""


class SingleViewUnavailableError(RuntimeError):
    """Single-view pipeline is disabled or not loaded."""


@dataclass
class GenerationOptions:
    steps: int = 5
    guidance_scale: float = 5.0
    seed: int = 1234
    randomize_seed: bool = False
    octree_resolution: int = 256
    remove_background: bool = True
    num_chunks: int = 8000
    output_format: str = "glb"

    def resolved_seed(self) -> int:
        if self.randomize_seed:
            return random.randint(0, MAX_SEED)
        return int(self.seed)


@dataclass
class ServiceResult:
    output_path: str
    stats: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def stats_header_value(self) -> str:
        return json.dumps(self.stats, default=str)


@dataclass
class ServiceConfig:
    shape_model_path: str = "tencent/Hunyuan3D-2"
    shape_subfolder: str = "hunyuan3d-dit-v2-0-turbo"
    mv_model_path: str = "tencent/Hunyuan3D-2mv"
    mv_subfolder: str = "hunyuan3d-dit-v2-mv-turbo"
    texgen_model_path: str = "tencent/Hunyuan3D-2"
    cache_path: str = "api_cache"
    device: str = "cuda"
    mc_algo: str = "mc"
    low_vram_mode: bool = False
    no_texture: bool = False
    enable_flashvdm: bool = False
    compile_models: bool = False
    disable_multiview: bool = False
    disable_single_view: bool = False


def gen_save_folder(cache_dir: str, max_size: int = 200) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    dirs = [f for f in Path(cache_dir).iterdir() if f.is_dir()]
    if len(dirs) >= max_size:
        oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
        shutil.rmtree(oldest_dir)
        logger.info("Removed the oldest folder: %s", oldest_dir)
    new_folder = os.path.join(cache_dir, str(uuid.uuid4()))
    os.makedirs(new_folder, exist_ok=True)
    return new_folder


def export_mesh(
    mesh: trimesh.Trimesh,
    save_folder: str,
    *,
    textured: bool = False,
    mesh_type: str = "glb",
) -> str:
    if mesh_type not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported output format: {mesh_type}")
    name = "textured_mesh" if textured else "white_mesh"
    path = os.path.join(save_folder, f"{name}.{mesh_type}")
    if mesh_type in ("glb", "obj"):
        mesh.export(path, include_normals=textured)
    else:
        mesh.export(path)
    return path


def _load_image_upload(data: bytes) -> Image.Image:
    from io import BytesIO

    return Image.open(BytesIO(data))


class Hunyuan3DService:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.cache_dir = config.cache_path
        os.makedirs(self.cache_dir, exist_ok=True)
        if config.disable_multiview and config.disable_single_view:
            raise ValueError(
                "At least one shape pipeline must be enabled "
                "(do not use --no-multiview and --no-single-view together)."
            )

        self._lock = threading.Lock()
        self._sv_init_lock = threading.Lock()
        self._mv_init_lock = threading.Lock()
        self.has_texturegen = False
        self.single_view_enabled = not config.disable_single_view
        self.multiview_enabled = not config.disable_multiview
        self.i23d_worker = None
        self.i23d_mv_worker = None

        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import (
            DegenerateFaceRemover,
            FaceReducer,
            FloaterRemover,
            Hunyuan3DDiTFlowMatchingPipeline,
        )

        self._Hunyuan3DDiTFlowMatchingPipeline = Hunyuan3DDiTFlowMatchingPipeline
        self.rmbg_worker = BackgroundRemover()
        self.face_reduce_worker = FaceReducer()
        self.floater_remove_worker = FloaterRemover()
        self.degenerate_face_remove_worker = DegenerateFaceRemover()

        if self.single_view_enabled:
            logger.info("Loading single-view shape model...")
            self.i23d_worker = self._build_shape_worker(
                config.shape_model_path,
                config.shape_subfolder,
            )
        else:
            logger.info("Single-view shape model disabled (--no-single-view)")

        if self.multiview_enabled:
            if config.disable_single_view:
                logger.info("Loading multiview shape model (multiview-only mode)...")
                self._get_mv_worker()
            else:
                logger.info(
                    "Multiview shape model will load on first multiview request "
                    "(run ./scripts/download-models.sh to prefetch weights)"
                )
        else:
            logger.info("Multiview shape model disabled (--no-multiview)")

        self.texgen_worker = None
        if not config.no_texture:
            try:
                from hy3dgen.texgen import Hunyuan3DPaintPipeline

                self.texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(
                    config.texgen_model_path
                )
                if config.low_vram_mode:
                    self.texgen_worker.enable_model_cpu_offload()
                self.has_texturegen = True
            except Exception as exc:
                logger.warning("Failed to load texture generator: %s", exc)

        if config.low_vram_mode:
            torch.cuda.empty_cache()

    def _require_texture(self) -> None:
        if not self.has_texturegen or self.texgen_worker is None:
            raise TextureUnavailableError(
                "Texture pipeline is not available. Install texture requirements "
                "and run without --no-texture."
            )

    def _require_multiview(self) -> None:
        if self.config.disable_multiview:
            raise MultiviewUnavailableError(
                "Multiview is disabled on this server. Restart without --no-multiview."
            )

    def _require_single_view(self) -> None:
        if self.config.disable_single_view:
            raise SingleViewUnavailableError(
                "Single-view is disabled on this server. Restart without --no-single-view "
                "or use multiview endpoints."
            )

    def _build_shape_worker(self, model_path: str, subfolder: str):
        worker = self._Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=True,
            device=self.config.device,
        )
        if self.config.enable_flashvdm:
            mc_algo = (
                "mc" if self.config.device in ("cpu", "mps") else self.config.mc_algo
            )
            worker.enable_flashvdm(mc_algo=mc_algo)
        if self.config.compile_models:
            worker.compile()
        return worker

    def _get_sv_worker(self):
        self._require_single_view()
        if self.i23d_worker is not None:
            return self.i23d_worker
        with self._sv_init_lock:
            if self.i23d_worker is not None:
                return self.i23d_worker
            logger.info(
                "Loading single-view shape model %s/%s (may download weights)...",
                self.config.shape_model_path,
                self.config.shape_subfolder,
            )
            self.i23d_worker = self._build_shape_worker(
                self.config.shape_model_path,
                self.config.shape_subfolder,
            )
            if self.config.low_vram_mode:
                torch.cuda.empty_cache()
            return self.i23d_worker

    def _get_mv_worker(self):
        self._require_multiview()
        if self.i23d_mv_worker is not None:
            return self.i23d_mv_worker
        with self._mv_init_lock:
            if self.i23d_mv_worker is not None:
                return self.i23d_mv_worker
            logger.info(
                "Loading multiview shape model %s/%s (may download weights)...",
                self.config.mv_model_path,
                self.config.mv_subfolder,
            )
            self.i23d_mv_worker = self._build_shape_worker(
                self.config.mv_model_path,
                self.config.mv_subfolder,
            )
            if self.config.low_vram_mode:
                torch.cuda.empty_cache()
            return self.i23d_mv_worker

    def _maybe_clear_vram(self) -> None:
        if self.config.low_vram_mode:
            torch.cuda.empty_cache()

    def _prepare_single_image(
        self, image: Image.Image, remove_background: bool
    ) -> tuple[Image.Image, dict[str, float]]:
        time_meta: dict[str, float] = {}
        if remove_background or image.mode == "RGB":
            start = time.time()
            image = self.rmbg_worker(image.convert("RGB"))
            time_meta["remove_background"] = time.time() - start
        return image, time_meta

    def _prepare_multiview(
        self, views: dict[str, Image.Image], remove_background: bool
    ) -> tuple[dict[str, Image.Image], dict[str, float]]:
        time_meta: dict[str, float] = {}
        if remove_background or any(v.mode == "RGB" for v in views.values()):
            start = time.time()
            for key, img in views.items():
                if remove_background or img.mode == "RGB":
                    views[key] = self.rmbg_worker(img.convert("RGB"))
            time_meta["remove_background"] = time.time() - start
        return views, time_meta

    def _run_shape_pipeline(
        self,
        *,
        pipeline,
        image_input,
        opts: GenerationOptions,
        model_label: str,
    ) -> tuple[trimesh.Trimesh, Image.Image | None, dict[str, Any]]:
        seed = opts.resolved_seed()
        save_folder = gen_save_folder(self.cache_dir)
        stats: dict[str, Any] = {
            "model": {"shapegen": model_label, "texgen": self.config.texgen_model_path},
            "params": {
                "steps": opts.steps,
                "guidance_scale": opts.guidance_scale,
                "seed": seed,
                "octree_resolution": opts.octree_resolution,
                "remove_background": opts.remove_background,
                "num_chunks": opts.num_chunks,
                "output_format": opts.output_format,
            },
        }
        time_meta: dict[str, float] = {}

        is_multiview = isinstance(image_input, dict)
        if is_multiview:
            image_input, rembg_time = self._prepare_multiview(
                image_input, opts.remove_background
            )
            time_meta.update(rembg_time)
            ref_image = image_input.get("front")
            if ref_image is None:
                ref_image = next(iter(image_input.values()))
        else:
            image_input, rembg_time = self._prepare_single_image(
                image_input, opts.remove_background
            )
            time_meta.update(rembg_time)
            ref_image = image_input

        start = time.time()
        generator = torch.Generator().manual_seed(seed)
        outputs = pipeline(
            image=image_input,
            num_inference_steps=opts.steps,
            guidance_scale=opts.guidance_scale,
            generator=generator,
            octree_resolution=opts.octree_resolution,
            num_chunks=opts.num_chunks,
            output_type="mesh",
        )
        time_meta["shape_generation"] = time.time() - start
        logger.info("Shape generation took %.2f seconds", time_meta["shape_generation"])

        tmp_start = time.time()
        mesh = export_to_trimesh(outputs)[0]
        time_meta["export_to_trimesh"] = time.time() - tmp_start

        stats["number_of_faces"] = int(mesh.faces.shape[0])
        stats["number_of_vertices"] = int(mesh.vertices.shape[0])
        stats["time"] = time_meta
        stats["save_folder"] = save_folder

        return mesh, ref_image, stats

    def _finalize_mesh_result(
        self,
        mesh: trimesh.Trimesh,
        stats: dict[str, Any],
        opts: GenerationOptions,
        *,
        textured: bool = False,
        seed: int | None = None,
    ) -> ServiceResult:
        save_folder = stats["save_folder"]
        path = export_mesh(
            mesh,
            save_folder,
            textured=textured,
            mesh_type=opts.output_format,
        )
        resolved_seed = seed if seed is not None else stats["params"]["seed"]
        return ServiceResult(output_path=path, stats=stats, seed=resolved_seed)

    def generate_mesh_from_image(
        self, image: Image.Image, opts: GenerationOptions | None = None
    ) -> ServiceResult:
        opts = opts or GenerationOptions()
        model_label = f"{self.config.shape_model_path}/{self.config.shape_subfolder}"

        sv_worker = self._get_sv_worker()
        with self._lock:
            mesh, _, stats = self._run_shape_pipeline(
                pipeline=sv_worker,
                image_input=image,
                opts=opts,
                model_label=model_label,
            )
            stats["time"]["total"] = sum(stats["time"].values())
            result = self._finalize_mesh_result(mesh, stats, opts)
            self._maybe_clear_vram()
            return result

    def generate_mesh_from_multiview(
        self, views: dict[str, Image.Image], opts: GenerationOptions | None = None
    ) -> ServiceResult:
        if not views:
            raise ValueError("At least one view image is required.")
        opts = opts or GenerationOptions()
        model_label = f"{self.config.mv_model_path}/{self.config.mv_subfolder}"
        mv_worker = self._get_mv_worker()

        with self._lock:
            mesh, _, stats = self._run_shape_pipeline(
                pipeline=mv_worker,
                image_input=views,
                opts=opts,
                model_label=model_label,
            )
            stats["time"]["total"] = sum(stats["time"].values())
            result = self._finalize_mesh_result(mesh, stats, opts)
            self._maybe_clear_vram()
            return result

    def generate_texture(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image | list[Image.Image],
        opts: GenerationOptions | None = None,
    ) -> ServiceResult:
        self._require_texture()
        opts = opts or GenerationOptions()
        save_folder = gen_save_folder(self.cache_dir)
        stats: dict[str, Any] = {
            "model": {"texgen": self.config.texgen_model_path},
            "params": {"output_format": opts.output_format},
            "save_folder": save_folder,
        }
        time_meta: dict[str, float] = {}

        with self._lock:
            start = time.time()
            reduced = self.face_reduce_worker(mesh)
            time_meta["face_reduction"] = time.time() - start

            start = time.time()
            textured_mesh = self.texgen_worker(reduced, image)
            time_meta["texture_generation"] = time.time() - start
            logger.info(
                "Texture generation took %.2f seconds", time_meta["texture_generation"]
            )

            stats["number_of_faces"] = int(textured_mesh.faces.shape[0])
            stats["number_of_vertices"] = int(textured_mesh.vertices.shape[0])
            time_meta["total"] = sum(time_meta.values())
            stats["time"] = time_meta

            result = self._finalize_mesh_result(
                textured_mesh, stats, opts, textured=True
            )
            self._maybe_clear_vram()
            return result

    def generate_textured_from_image(
        self, image: Image.Image, opts: GenerationOptions | None = None
    ) -> ServiceResult:
        self._require_texture()
        opts = opts or GenerationOptions()
        start_total = time.time()
        model_label = f"{self.config.shape_model_path}/{self.config.shape_subfolder}"

        sv_worker = self._get_sv_worker()
        with self._lock:
            mesh, ref_image, stats = self._run_shape_pipeline(
                pipeline=sv_worker,
                image_input=image,
                opts=opts,
                model_label=model_label,
            )
            export_mesh(mesh, stats["save_folder"], textured=False, mesh_type=opts.output_format)

            start = time.time()
            mesh = self.face_reduce_worker(mesh)
            stats["time"]["face_reduction"] = time.time() - start

            start = time.time()
            textured_mesh = self.texgen_worker(mesh, ref_image)
            stats["time"]["texture_generation"] = time.time() - start

            stats["time"]["total"] = time.time() - start_total
            result = self._finalize_mesh_result(
                textured_mesh,
                stats,
                opts,
                textured=True,
                seed=stats["params"]["seed"],
            )
            self._maybe_clear_vram()
            return result

    def generate_textured_from_multiview(
        self, views: dict[str, Image.Image], opts: GenerationOptions | None = None
    ) -> ServiceResult:
        self._require_texture()
        if not views:
            raise ValueError("At least one view image is required.")
        opts = opts or GenerationOptions()
        start_total = time.time()
        model_label = f"{self.config.mv_model_path}/{self.config.mv_subfolder}"

        mv_worker = self._get_mv_worker()
        with self._lock:
            mesh, ref_image, stats = self._run_shape_pipeline(
                pipeline=mv_worker,
                image_input=views,
                opts=opts,
                model_label=model_label,
            )
            export_mesh(mesh, stats["save_folder"], textured=False, mesh_type=opts.output_format)

            start = time.time()
            mesh = self.face_reduce_worker(mesh)
            stats["time"]["face_reduction"] = time.time() - start

            start = time.time()
            textured_mesh = self.texgen_worker(mesh, ref_image)
            stats["time"]["texture_generation"] = time.time() - start

            stats["time"]["total"] = time.time() - start_total
            result = self._finalize_mesh_result(
                textured_mesh,
                stats,
                opts,
                textured=True,
                seed=stats["params"]["seed"],
            )
            self._maybe_clear_vram()
            return result

    @staticmethod
    def load_mesh_from_bytes(data: bytes, filename: str | None = None) -> trimesh.Trimesh:
        from io import BytesIO

        suffix = ""
        if filename:
            suffix = Path(filename).suffix.lower()
        if suffix in (".glb", ".gltf"):
            mesh = trimesh.load(BytesIO(data), file_type="glb")
        elif suffix == ".obj":
            mesh = trimesh.load(BytesIO(data), file_type="obj")
        else:
            mesh = trimesh.load(BytesIO(data), file_type="glb")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        return mesh

    @staticmethod
    def load_image_from_bytes(data: bytes) -> Image.Image:
        return _load_image_upload(data)
