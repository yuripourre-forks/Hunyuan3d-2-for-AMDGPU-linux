# Hunyuan3D REST API server (no Gradio UI).

from __future__ import annotations

import argparse
import logging
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from hunyuan3d_service import (
    MV_VIEW_NAMES,
    GenerationOptions,
    Hunyuan3DService,
    MultiviewUnavailableError,
    SingleViewUnavailableError,
    ServiceConfig,
    SUPPORTED_FORMATS,
    TextureUnavailableError,
)

logger = logging.getLogger("hunyuan3d.api")

app = FastAPI(
    title="Hunyuan3D API",
    description="Synchronous mesh and texture generation for Hunyuan3D-2 on ROCm.",
    version="1.0.0",
)

_service: Hunyuan3DService | None = None


def get_service() -> Hunyuan3DService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    return _service


def _parse_generation_options(
    steps: int = Form(5),
    guidance_scale: float = Form(5.0),
    seed: int = Form(1234),
    randomize_seed: bool = Form(False),
    octree_resolution: int = Form(256),
    remove_background: bool = Form(True),
    num_chunks: int = Form(8000),
    output_format: str = Form("glb"),
) -> GenerationOptions:
    fmt = output_format.lower().strip()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"output_format must be one of: {', '.join(SUPPORTED_FORMATS)}",
        )
    return GenerationOptions(
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        randomize_seed=randomize_seed,
        octree_resolution=octree_resolution,
        remove_background=remove_background,
        num_chunks=num_chunks,
        output_format=fmt,
    )


def _file_response(result, download_name: str) -> FileResponse:
    return FileResponse(
        result.output_path,
        media_type="application/octet-stream",
        filename=download_name,
        headers={"X-Hunyuan-Stats": result.stats_header_value()},
    )


async def _read_upload(upload: UploadFile) -> bytes:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=422, detail=f"Empty upload: {upload.filename}")
    return data


async def _load_views_from_form(
    front: UploadFile | None = File(None),
    back: UploadFile | None = File(None),
    left: UploadFile | None = File(None),
    right: UploadFile | None = File(None),
) -> dict:
    uploads = {
        "front": front,
        "back": back,
        "left": left,
        "right": right,
    }
    views = {}
    svc = get_service()
    for name, upload in uploads.items():
        if upload is None or not upload.filename:
            continue
        data = await _read_upload(upload)
        views[name] = svc.load_image_from_bytes(data)
    if not views:
        raise HTTPException(
            status_code=422,
            detail=f"Provide at least one view ({', '.join(MV_VIEW_NAMES)}).",
        )
    return views


@app.get("/health")
def health() -> dict:
    svc = get_service()
    return {
        "status": "ok",
        "texture_available": svc.has_texturegen,
        "single_view_enabled": svc.single_view_enabled,
        "single_view_loaded": svc.i23d_worker is not None,
        "multiview_enabled": svc.multiview_enabled,
        "multiview_loaded": svc.i23d_mv_worker is not None,
    }


@app.post("/v1/mesh/from-image")
async def mesh_from_image(
    image: Annotated[UploadFile, File(...)],
    steps: int = Form(5),
    guidance_scale: float = Form(5.0),
    seed: int = Form(1234),
    randomize_seed: bool = Form(False),
    octree_resolution: int = Form(256),
    remove_background: bool = Form(True),
    num_chunks: int = Form(8000),
    output_format: str = Form("glb"),
):
    gen_opts = _parse_generation_options(
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        randomize_seed=randomize_seed,
        octree_resolution=octree_resolution,
        remove_background=remove_background,
        num_chunks=num_chunks,
        output_format=output_format,
    )
    svc = get_service()
    data = await _read_upload(image)
    pil_image = svc.load_image_from_bytes(data)
    try:
        result = svc.generate_mesh_from_image(pil_image, gen_opts)
    except SingleViewUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("mesh/from-image failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _file_response(result, f"white_mesh.{gen_opts.output_format}")


@app.post("/v1/mesh/from-multiview")
async def mesh_from_multiview(
    front: UploadFile | None = File(None),
    back: UploadFile | None = File(None),
    left: UploadFile | None = File(None),
    right: UploadFile | None = File(None),
    steps: int = Form(5),
    guidance_scale: float = Form(5.0),
    seed: int = Form(1234),
    randomize_seed: bool = Form(False),
    octree_resolution: int = Form(256),
    remove_background: bool = Form(True),
    num_chunks: int = Form(8000),
    output_format: str = Form("glb"),
):
    gen_opts = _parse_generation_options(
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        randomize_seed=randomize_seed,
        octree_resolution=octree_resolution,
        remove_background=remove_background,
        num_chunks=num_chunks,
        output_format=output_format,
    )
    views = await _load_views_from_form(front, back, left, right)
    svc = get_service()
    try:
        result = svc.generate_mesh_from_multiview(views, gen_opts)
    except MultiviewUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("mesh/from-multiview failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _file_response(result, f"white_mesh.{gen_opts.output_format}")


@app.post("/v1/texture")
async def texture_mesh(
    mesh: Annotated[UploadFile, File(...)],
    image: UploadFile | None = File(None),
    images: list[UploadFile] | None = File(None),
    steps: int = Form(5),
    guidance_scale: float = Form(5.0),
    seed: int = Form(1234),
    randomize_seed: bool = Form(False),
    octree_resolution: int = Form(256),
    remove_background: bool = Form(True),
    num_chunks: int = Form(8000),
    output_format: str = Form("glb"),
):
    gen_opts = _parse_generation_options(
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        randomize_seed=randomize_seed,
        octree_resolution=octree_resolution,
        remove_background=remove_background,
        num_chunks=num_chunks,
        output_format=output_format,
    )
    svc = get_service()
    mesh_data = await _read_upload(mesh)
    loaded_mesh = svc.load_mesh_from_bytes(mesh_data, mesh.filename)

    ref_images = []
    if images:
        for upload in images:
            if upload.filename:
                ref_images.append(svc.load_image_from_bytes(await _read_upload(upload)))
    if image and image.filename:
        ref_images.append(svc.load_image_from_bytes(await _read_upload(image)))
    if not ref_images:
        raise HTTPException(
            status_code=422,
            detail="Provide image or images (reference for texturing).",
        )
    ref = ref_images[0] if len(ref_images) == 1 else ref_images

    try:
        result = svc.generate_texture(loaded_mesh, ref, gen_opts)
    except TextureUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("texture failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _file_response(result, f"textured_mesh.{gen_opts.output_format}")


@app.post("/v1/mesh/textured")
async def mesh_textured(
    image: UploadFile | None = File(None),
    front: UploadFile | None = File(None),
    back: UploadFile | None = File(None),
    left: UploadFile | None = File(None),
    right: UploadFile | None = File(None),
    steps: int = Form(5),
    guidance_scale: float = Form(5.0),
    seed: int = Form(1234),
    randomize_seed: bool = Form(False),
    octree_resolution: int = Form(256),
    remove_background: bool = Form(True),
    num_chunks: int = Form(8000),
    output_format: str = Form("glb"),
):
    gen_opts = _parse_generation_options(
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        randomize_seed=randomize_seed,
        octree_resolution=octree_resolution,
        remove_background=remove_background,
        num_chunks=num_chunks,
        output_format=output_format,
    )
    svc = get_service()
    has_mv = any(
        u is not None and u.filename
        for u in (front, back, left, right)
    )
    has_single = image is not None and image.filename

    if has_single and has_mv:
        raise HTTPException(
            status_code=422,
            detail="Provide either image or multiview fields, not both.",
        )
    if not has_single and not has_mv:
        raise HTTPException(
            status_code=422,
            detail="Provide image (single-view) or front/back/left/right (multiview).",
        )

    try:
        if has_single:
            data = await _read_upload(image)
            pil_image = svc.load_image_from_bytes(data)
            result = svc.generate_textured_from_image(pil_image, gen_opts)
        else:
            views = await _load_views_from_form(front, back, left, right)
            result = svc.generate_textured_from_multiview(views, gen_opts)
    except SingleViewUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MultiviewUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TextureUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("mesh/textured failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _file_response(result, f"textured_mesh.{gen_opts.output_format}")


def build_service_from_args(args: argparse.Namespace) -> Hunyuan3DService:
    config = ServiceConfig(
        shape_model_path=args.shape_model_path,
        shape_subfolder=args.shape_subfolder,
        mv_model_path=args.mv_model_path,
        mv_subfolder=args.mv_subfolder,
        texgen_model_path=args.texgen_model_path,
        cache_path=args.cache_path,
        device=args.device,
        mc_algo=args.mc_algo,
        low_vram_mode=args.low_vram_mode,
        no_texture=args.no_texture,
        enable_flashvdm=args.enable_flashvdm,
        compile_models=args.compile,
        disable_multiview=args.no_multiview,
        disable_single_view=args.no_single_view,
    )
    return Hunyuan3DService(config)


def main() -> None:
    global _service

    parser = argparse.ArgumentParser(description="Hunyuan3D REST API server")
    parser.add_argument("--shape_model_path", type=str, default="tencent/Hunyuan3D-2")
    parser.add_argument("--shape_subfolder", type=str, default="hunyuan3d-dit-v2-0-turbo")
    parser.add_argument("--mv_model_path", type=str, default="tencent/Hunyuan3D-2mv")
    parser.add_argument("--mv_subfolder", type=str, default="hunyuan3d-dit-v2-mv-turbo")
    parser.add_argument("--texgen_model_path", type=str, default="tencent/Hunyuan3D-2")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mc_algo", type=str, default="mc")
    parser.add_argument("--cache-path", type=str, default="api_cache")
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="Do not load texture pipeline (texture endpoints return 503)",
    )
    parser.add_argument("--enable_flashvdm", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--low_vram_mode", action="store_true")
    parser.add_argument(
        "--no-multiview",
        action="store_true",
        help="Do not load multiview shape model (multiview endpoints return 503)",
    )
    parser.add_argument(
        "--no-single-view",
        action="store_true",
        help="Do not load single-view shape model; loads multiview at startup (multiview-only)",
    )
    args = parser.parse_args()

    if args.no_multiview and args.no_single_view:
        parser.error("Use at most one of --no-multiview and --no-single-view.")

    logging.basicConfig(level=logging.INFO)
    logger.info("Loading Hunyuan3D models (this may take several minutes)...")
    _service = build_service_from_args(args)
    logger.info(
        "Models ready (texture=%s, single_view=%s, multiview=%s). Starting API on %s:%s",
        _service.has_texturegen,
        _service.single_view_enabled,
        _service.multiview_enabled,
        args.host,
        args.port,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
