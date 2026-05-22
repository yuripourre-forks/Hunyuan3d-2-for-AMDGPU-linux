#!/usr/bin/env bash
# Launch Hunyuan3D-2 Gradio (single-view + text-to-3D).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_venv
export_rocm_build_env

PORT="${GRADIO_PORT:-8080}"
if [[ -f "${REPO_ROOT}/config/port" ]]; then
    PORT="$(tr -d '[:space:]' < "${REPO_ROOT}/config/port")"
fi

HUNYUAN_DIR="$(hunyuan3d_dir)"
cd "${HUNYUAN_DIR}"

exec python gradio_app.py \
    --model_path tencent/Hunyuan3D-2 \
    --subfolder hunyuan3d-dit-v2-0-turbo \
    --texgen_model_path tencent/Hunyuan3D-2 \
    --low_vram_mode \
    --enable_flashvdm \
    --enable_t23d \
    --port "${PORT}" \
    "$@"
