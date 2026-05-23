[Español-Spanish](README-ES.md)

# Hunyuan3D-2 for AMDGPU on Linux

Scripts to install and run [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) on Linux with an AMD GPU and ROCm. Texture generation requires building `custom_rasterizer` for HIP; this repo builds it from source during install (no prebuilt wheels).

Tested with ROCm 6.3–6.4 on RX 7900 XTX (`gfx1100`) and RDNA4-class GPUs (`gfx1201`). `GPU_ARCHS` defaults to `auto` (detected from `rocminfo`).

## Prerequisites

1. **ROCm** — [Install ROCm on Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). Verify with `rocminfo` or `/opt/rocm`.

2. **System packages** (Fedora example; adjust for your distro):
   ```bash
   sudo dnf install git python3 python3-pip patch gcc-c++ ninja-build rocm-dev
   ```

3. **Python** — 3.10 or newer (`python3 --version`).

4. **Path** — Do not install or run from a directory path that contains spaces ([ROCm #4329](https://github.com/ROCm/ROCm/issues/4329)).

5. **Versions** — PyTorch 2.8+ is not supported here. Defaults use torch 2.7.1 from the ROCm 6.3 wheel index (see `config/defaults.env`).

## Quick install

```bash
git clone https://github.com/dgarcia1985/Hunyuan3d-2-for-AMDGPU-linux.git
cd Hunyuan3d-2-for-AMDGPU-linux
./scripts/install.sh
```

Options:

```bash
./scripts/install.sh --port 8080
./scripts/install.sh --gpu-arch gfx1030    # e.g. RX 6000 series
./scripts/install.sh --skip-flash-attention  # skip long flash-attention build
```

Override defaults via environment variables:

```bash
GPU_ARCHS=gfx1030 GRADIO_PORT=9000 ./scripts/install.sh
```

## Run Gradio

### Full install (fresh)

```bash
./scripts/install.sh
./scripts/run.sh              # http://127.0.0.1:8080
./scripts/run-multiview.sh    # multiview mode
```

### After rasterizer bootstrap

If you already ran `./scripts/bootstrap-rasterizer.sh`, complete the app stack then launch:

```bash
./scripts/install-app.sh
# faster (skips long flash-attention build; use --no-flashvdm when running):
# ./scripts/install-app.sh --skip-flash-attention

./scripts/run.sh
```

`install.sh` and `install-app.sh` **prefetch public Hugging Face weights without login** (same as the original installer: models are public Tencent repos). No `huggingface-cli login` is required. Re-running install **skips download** when weights are already in the Hugging Face cache (`~/.cache/huggingface/hub` by default); only missing subfolders are fetched.

To download or refresh models only:

```bash
./scripts/download-models.sh
./scripts/download-models.sh --no-multiview   # skip multiview checkpoint (~smaller download)
```

If Gradio warns about missing `diffusion_pytorch_model.safetensors` under `hunyuan3d-paint-v2-0-turbo/vae`, re-run `download-models.sh` (older installs used a shallow pattern and skipped nested weight files). The paint VAE may use `.bin` weights; diffusers will load those automatically.

Skip prefetch during install (models download on first Gradio run instead):

```bash
./scripts/install-app.sh --skip-model-download
```

Options:

```bash
GRADIO_PORT=9000 ./scripts/run.sh
./scripts/run.sh --no-flashvdm          # if install-app skipped flash-attention
./scripts/run.sh --no-texture         # shape only, no texture pipeline
```

## REST API

API-only FastAPI server (no Gradio UI). Loads single-view, multiview, and texture pipelines in one process. Requests are **synchronous** (the connection stays open until the mesh file is ready; generation can take several minutes).

```bash
./scripts/install.sh          # or install-app.sh after bootstrap
./scripts/run-api.sh          # http://127.0.0.1:8081
```

OpenAPI docs: `http://127.0.0.1:8081/docs`

Options:

```bash
API_PORT=9001 ./scripts/run-api.sh
./scripts/run-api.sh --no-flashvdm
./scripts/run-api.sh --no-multiview      # single-view only (MV loads on first multiview call)
./scripts/run-api.sh --no-single-view    # multiview only (loads MV at startup, skips single-view)
./scripts/run-api.sh --no-texture        # mesh endpoints only; texture returns 503
```

Shape model loading (pick one mode; do not combine `--no-multiview` and `--no-single-view`):

| Flag | Startup loads | Endpoints |
|------|----------------|-----------|
| (default) | Single-view + texture; multiview on first use | All |
| `--no-multiview` | Single-view + texture only | No multiview routes (503) |
| `--no-single-view` | Multiview + texture only | No single-image shape routes (503) |

Multiview weights (`tencent/Hunyuan3D-2mv`, ~5GB) load at startup in multiview-only mode, or on first multiview request otherwise. Prefetch with `./scripts/download-models.sh`. If install reported models cached but multiview still downloads, re-run `download-models.sh` — only `config.yaml` in cache means weights were incomplete.

Prefetch models before first API call (recommended):

```bash
./scripts/download-models.sh
```

### Endpoints

All endpoints use `POST` with `multipart/form-data`. The response body is the mesh file (`glb` by default). Stats (timings, face count, seed) are in the `X-Hunyuan-Stats` JSON header.

| Endpoint | Description |
|----------|-------------|
| `POST /v1/mesh/from-image` | Untextured mesh from one image (`image` field) |
| `POST /v1/mesh/from-multiview` | Untextured mesh from 1–4 views (`front`, `back`, `left`, `right`) |
| `POST /v1/texture` | Texture an existing mesh (`mesh` + `image` or repeated `images`) |
| `POST /v1/mesh/textured` | Shape + texture from `image` or multiview fields |

Optional form fields (all endpoints): `steps` (default `5`), `guidance_scale` (`5.0`), `seed`, `randomize_seed`, `octree_resolution` (`256`), `remove_background` (`true`), `num_chunks` (`8000`), `output_format` (`glb`, `obj`, `ply`, `stl`).

### Examples

Mesh from a single image:

```bash
curl -X POST http://127.0.0.1:8081/v1/mesh/from-image \
  -F "image=@input.png" \
  -F "remove_background=true" \
  -o white_mesh.glb
```

Mesh from multiview images:

```bash
curl -X POST http://127.0.0.1:8081/v1/mesh/from-multiview \
  -F "front=@front.png" \
  -F "back=@back.png" \
  -F "left=@left.png" \
  -F "right=@right.png" \
  -o white_mesh.glb
```

Texture an existing mesh:

```bash
curl -X POST http://127.0.0.1:8081/v1/texture \
  -F "mesh=@model.glb" \
  -F "image=@reference.png" \
  -o textured_mesh.glb
```

Full textured model from one image:

```bash
curl -X POST http://127.0.0.1:8081/v1/mesh/textured \
  -F "image=@input.png" \
  -o textured_mesh.glb
```

**Notes:** Only one GPU job runs at a time (`workers=1`). Loading all pipelines needs substantial VRAM; `--low_vram_mode` is enabled by default in `run-api.sh`.

## Rasterizer-only build (fast)

To build only `custom_rasterizer` (venv + ROCm PyTorch + clone upstream, no flash-attention or Gradio):

```bash
./scripts/bootstrap-rasterizer.sh
```

Options: `--gpu-arch gfx1201`, `--force-hipify` (re-hipify after upstream updates).

## Repository layout

```
config/defaults.env          # pinned versions, GPU arch, ports
scripts/
  install.sh                 # full installer (venv + everything)
  install-app.sh             # complete app after bootstrap-rasterizer
  download-models.sh         # prefetch HF weights (no login)
  bootstrap-rasterizer.sh    # minimal venv + rasterizer only
  build-custom-rasterizer.sh # hipify + build texture rasterizer
  build-flash-attention.sh   # ROCm flash-attention
  run.sh / run-multiview.sh  # launch Gradio (with preflight checks)
  run-api.sh                 # launch REST API
patches/                     # setup.py patch for HIP sources
gradio_app.py                # copied into vendor/Hunyuan3D-2 on install
api_server.py                # REST API (copied on install)
hunyuan3d_service.py         # pipeline service layer (copied on install)
vendor/Hunyuan3D-2/          # cloned upstream (gitignored)
.venv/                       # Python virtualenv (gitignored)
```

## Building custom_rasterizer manually

If install fails at the rasterizer step, ensure the venv has ROCm PyTorch, then:

```bash
./scripts/build-custom-rasterizer.sh
```

Override arch or force a clean hipify:

```bash
./scripts/build-custom-rasterizer.sh --gpu-arch gfx1201 --force-hipify
```

Build scripts auto-detect `GPU_ARCHS` and set `HIP_VISIBLE_DEVICES` to the primary dGPU (useful when an iGPU is also present).

What the script does:

1. Runs PyTorch hipify on `rasterizer.cpp`, `grid_neighbor.cpp`, `rasterizer_gpu.cu` under `vendor/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer/`
2. Applies `patches/custom_rasterizer-setup-rocm.patch` so `setup.py` compiles `*_hip.*` sources
3. `pip install -e . --no-build-isolation` and runs a minimal GPU `rasterize_image()` smoke test

Some users report the extension build succeeds on Arch but fails on Ubuntu with the same Python/ROCm; if that happens, try building on Arch (or a container) and reusing the same venv layout, or adjust compiler/ROCm dev packages on your distro.

## Known issues

- **Spaces in path** — Model generation can fail if the project path contains spaces (ROCm limitation).
- **Torch 2.8** — Not working with this setup; use 2.7.x as in `config/defaults.env`.
- **Dual GPU (iGPU + dGPU)** — Build scripts set `HIP_VISIBLE_DEVICES` to the detected dGPU. Texture generation at runtime may still need the iGPU disabled in BIOS on some systems.
- **VRAM** — Texture pipeline is heavy; `--low_vram_mode` is enabled in the run scripts by default.

## Configuration

Edit [`config/defaults.env`](config/defaults.env) or export variables before `install.sh`, `install-app.sh`, `run.sh`, or `run-api.sh`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GPU_ARCHS` | `auto` | ROCm offload arch (`auto` uses `rocminfo`; e.g. `gfx1201`, `gfx1100`) |
| `TORCH_VERSION` | `2.7.1` | PyTorch version |
| `PYTORCH_ROCM_INDEX` | rocm6.3 index URL | pip index for ROCm wheels |
| `GRADIO_PORT` | `8080` | Web UI port |
| `API_PORT` | `8081` | REST API port |
