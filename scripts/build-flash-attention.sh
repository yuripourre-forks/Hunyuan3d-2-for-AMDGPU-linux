#!/usr/bin/env bash
# Build ROCm flash-attention (main_perf branch) into the active venv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_venv
export_rocm_build_env

FLASH_DIR="${REPO_ROOT}/vendor/flash-attention"
REPO_URL="${FLASH_ATTENTION_REPO:-https://github.com/ROCm/flash-attention.git}"
BRANCH="${FLASH_ATTENTION_BRANCH:-main_perf}"

if [[ ! -d "${FLASH_DIR}/.git" ]]; then
    log "Cloning flash-attention (${BRANCH})..."
    mkdir -p "${REPO_ROOT}/vendor"
    git clone --single-branch --branch "${BRANCH}" "${REPO_URL}" "${FLASH_DIR}"
else
    log "Using existing flash-attention at ${FLASH_DIR}"
fi

cd "${FLASH_DIR}"
pip install sentencepiece packaging
log "Building flash-attention (this may take several minutes)..."
python setup.py install

log "flash-attention build complete."
