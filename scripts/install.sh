#!/usr/bin/env bash
# Install Hunyuan3D-2 with ROCm: venv, PyTorch, extensions, and dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SKIP_FLASH_ATTENTION=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            GRADIO_PORT="$2"
            shift 2
            ;;
        --skip-flash-attention)
            SKIP_FLASH_ATTENTION=1
            shift
            ;;
        --gpu-arch)
            GPU_ARCHS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--port PORT] [--gpu-arch GFX] [--skip-flash-attention]"
            exit 0
            ;;
        *)
            die "Unknown option: $1 (try --help)"
            ;;
    esac
done

require_cmd python3
require_cmd git
require_cmd patch
check_python_version
check_rocm
warn_path_spaces

export_rocm_build_env

ensure_venv

TORCH_VER="${TORCH_VERSION:-2.7.1}"
TV_VER="${TORCHVISION_VERSION:-0.22.1}"
TA_VER="${TORCHAUDIO_VERSION:-2.7.1}"
INDEX="${PYTORCH_ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.3}"

log "Installing PyTorch ${TORCH_VER} (ROCm)..."
pip install --upgrade pip wheel setuptools
pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" "torchaudio==${TA_VER}" \
    --index-url "${INDEX}"

log "Installing Python dependencies..."
pip install -r "${REPO_ROOT}/requirements.txt"

HUNYUAN_DIR="$(hunyuan3d_dir)"
REPO_URL="${HUNYUAN3D_REPO:-https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git}"

if [[ ! -d "${HUNYUAN_DIR}/.git" ]]; then
    log "Cloning Hunyuan3D-2..."
    mkdir -p "$(dirname "${HUNYUAN_DIR}")"
    git clone "${REPO_URL}" "${HUNYUAN_DIR}"
else
    log "Using existing Hunyuan3D-2 at ${HUNYUAN_DIR}"
fi

log "Installing hy3dgen (editable)..."
pip install -e "${HUNYUAN_DIR}"

"${SCRIPT_DIR}/build-custom-rasterizer.sh"

DIFF_DIR="${HUNYUAN_DIR}/hy3dgen/texgen/differentiable_renderer"
log "Building differentiable_renderer..."
pip install -e "${DIFF_DIR}"

if [[ "${SKIP_FLASH_ATTENTION}" -eq 0 ]]; then
    "${SCRIPT_DIR}/build-flash-attention.sh"
else
    log "Skipping flash-attention (--skip-flash-attention)"
fi

GRADIO_SRC="${REPO_ROOT}/gradio_app.py"
GRADIO_DST="${HUNYUAN_DIR}/gradio_app.py"
if [[ -f "${GRADIO_SRC}" ]]; then
    log "Installing gradio_app.py into Hunyuan3D-2..."
    cp -f "${GRADIO_SRC}" "${GRADIO_DST}"
fi

# Persist chosen port for run scripts
echo "${GRADIO_PORT:-8080}" > "${REPO_ROOT}/config/port"

log ""
log "Installation complete."
log "  Run single-view:  ./scripts/run.sh"
log "  Run multiview:    ./scripts/run-multiview.sh"
log "  Port (default):   ${GRADIO_PORT:-8080} (override: GRADIO_PORT=9000 ./scripts/run.sh)"
