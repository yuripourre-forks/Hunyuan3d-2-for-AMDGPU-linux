[Español-Spanish](README-ES.md)

# Hunyuan3D-2 for AMDGPU on Linux

Scripts to install and run [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) on Linux with an AMD GPU and ROCm. Texture generation requires building `custom_rasterizer` for HIP; this repo builds it from source during install (no prebuilt wheels).

Tested with ROCm 6.3–6.4 and RX 7900 XTX (`gfx1100`). Other GPUs may need a different `GPU_ARCHS` value.

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

## Run

After install:

```bash
./scripts/run.sh              # single-view + text-to-3D
./scripts/run-multiview.sh    # multiview mode
```

Change port: `GRADIO_PORT=9000 ./scripts/run.sh`

## Repository layout

```
config/defaults.env          # pinned versions, GPU arch, ports
scripts/
  install.sh                 # main installer
  build-custom-rasterizer.sh # hipify + build texture rasterizer
  build-flash-attention.sh   # ROCm flash-attention
  run.sh / run-multiview.sh  # launch Gradio
patches/                     # setup.py patch for HIP sources
gradio_app.py                # copied into vendor/Hunyuan3D-2 on install
vendor/Hunyuan3D-2/          # cloned upstream (gitignored)
.venv/                       # Python virtualenv (gitignored)
```

## Building custom_rasterizer manually

If install fails at the rasterizer step, ensure the venv has ROCm PyTorch, then:

```bash
source .venv/bin/activate
export GPU_ARCHS=gfx1100   # match your GPU
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
./scripts/build-custom-rasterizer.sh
```

What the script does:

1. Runs PyTorch hipify on `rasterizer.cpp`, `grid_neighbor.cpp`, `rasterizer_gpu.cu` under `vendor/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer/`
2. Applies `patches/custom_rasterizer-setup-rocm.patch` so `setup.py` compiles `*_hip.*` sources
3. `pip install -e .` and checks `custom_rasterizer_kernel.rasterize_image` exists

Some users report the extension build succeeds on Arch but fails on Ubuntu with the same Python/ROCm; if that happens, try building on Arch (or a container) and reusing the same venv layout, or adjust compiler/ROCm dev packages on your distro.

## Known issues

- **Spaces in path** — Model generation can fail if the project path contains spaces (ROCm limitation).
- **Torch 2.8** — Not working with this setup; use 2.7.x as in `config/defaults.env`.
- **Dual GPU (iGPU + dGPU)** — Texture generation may fail; disabling the integrated GPU in BIOS has helped some users.
- **VRAM** — Texture pipeline is heavy; `--low_vram_mode` is enabled in the run scripts by default.

## Configuration

Edit [`config/defaults.env`](config/defaults.env) or export variables before `install.sh` / `run.sh`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GPU_ARCHS` | `gfx1100` | ROCm offload arch for extension builds |
| `TORCH_VERSION` | `2.7.1` | PyTorch version |
| `PYTORCH_ROCM_INDEX` | rocm6.3 index URL | pip index for ROCm wheels |
| `GRADIO_PORT` | `8080` | Web UI port |
