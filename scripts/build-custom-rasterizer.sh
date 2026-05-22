#!/usr/bin/env bash
# Hipify and build custom_rasterizer for ROCm from Hunyuan3D-2 sources.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_venv
export_rocm_build_env

HUNYUAN_DIR="$(hunyuan3d_dir)"
RASTERIZER_DIR="${HUNYUAN_DIR}/hy3dgen/texgen/custom_rasterizer"
PATCH_FILE="${REPO_ROOT}/patches/custom_rasterizer-setup-rocm.patch"

[[ -d "${RASTERIZER_DIR}" ]] || die "custom_rasterizer not found. Run install.sh first (clones Hunyuan3D-2)."

python -c "import torch; assert getattr(torch.version, 'hip', None), 'PyTorch must be ROCm/HIP build'" \
    || die "Install ROCm PyTorch first: ./scripts/install.sh (step before custom_rasterizer)"

log "Building custom_rasterizer at ${RASTERIZER_DIR}"
cd "${RASTERIZER_DIR}"

KERNEL_DIR="lib/custom_rasterizer_kernel"
hipify_sources() {
    log "Hipifying CUDA sources..."
    python -m torch.utils.hipify.hipify_python \
        "${KERNEL_DIR}/rasterizer.cpp" \
        "${KERNEL_DIR}/grid_neighbor.cpp" \
        "${KERNEL_DIR}/rasterizer_gpu.cu"
}

if [[ ! -f "${KERNEL_DIR}/rasterizer_hip.cpp" ]] \
    || [[ ! -f "${KERNEL_DIR}/grid_neighbor_hip.cpp" ]] \
    || [[ ! -f "${KERNEL_DIR}/rasterizer_gpu.hip" ]]; then
    hipify_sources
fi

for f in "${KERNEL_DIR}/rasterizer_hip.cpp" \
         "${KERNEL_DIR}/grid_neighbor_hip.cpp" \
         "${KERNEL_DIR}/rasterizer_gpu.hip"; do
    [[ -f "${f}" ]] || die "Hipify failed: missing ${f}"
done

if ! grep -q 'rasterizer_hip.cpp' setup.py 2>/dev/null; then
    log "Applying ROCm setup.py patch..."
    patch -p1 -N < "${PATCH_FILE}" || die "Failed to apply ${PATCH_FILE}"
fi

log "Installing custom_rasterizer (editable)..."
pip install -e .

log "Smoke test: custom_rasterizer_kernel.rasterize_image"
python -c "
import custom_rasterizer_kernel
assert hasattr(custom_rasterizer_kernel, 'rasterize_image'), \
    'custom_rasterizer_kernel missing rasterize_image (wrong or incomplete build)'
print('custom_rasterizer OK')
"

log "custom_rasterizer build complete."
