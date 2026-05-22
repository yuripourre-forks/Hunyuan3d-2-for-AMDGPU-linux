#!/usr/bin/env bash
# Shared helpers for Hunyuan3D-2 ROCm install and run scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${REPO_ROOT}/config/defaults.env"
VENV_DIR="${REPO_ROOT}/.venv"

if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
fi

log() { echo "[hunyuan3d] $*"; }
die() { echo "[hunyuan3d] ERROR: $*" >&2; exit 1; }

require_cmd() {
    local cmd="$1"
    command -v "${cmd}" >/dev/null 2>&1 || die "Required command not found: ${cmd}"
}

check_python_version() {
    local min_major min_minor
    min_major="${PYTHON_MIN%%.*}"
    min_minor="${PYTHON_MIN#*.}"
    min_minor="${min_minor%%.*}"

    local ver major minor
    ver="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
    major="${ver%%.*}"
    minor="${ver#*.}"

    if [[ "${major}" -lt "${min_major}" ]] \
        || { [[ "${major}" -eq "${min_major}" ]] && [[ "${minor}" -lt "${min_minor}" ]]; }; then
        die "Python ${ver} found; need >= ${PYTHON_MIN}"
    fi
    log "Using Python ${ver}"
}

check_rocm() {
    if command -v rocminfo >/dev/null 2>&1; then
        log "ROCm detected (rocminfo available)"
        return 0
    fi
    if [[ -d /opt/rocm ]]; then
        log "ROCm detected (/opt/rocm)"
        return 0
    fi
    die "ROCm not found. Install ROCm and ensure rocminfo or /opt/rocm exists."
}

warn_path_spaces() {
    case "${REPO_ROOT}" in
        *" "*)
            log "WARNING: Install path contains spaces; ROCm builds may fail."
            log "  See https://github.com/ROCm/ROCm/issues/4329"
            ;;
    esac
}

activate_venv() {
    if [[ ! -d "${VENV_DIR}" ]]; then
        die "Virtualenv not found at ${VENV_DIR}. Run: ./scripts/install.sh"
    fi
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
}

ensure_venv() {
    if [[ ! -d "${VENV_DIR}" ]]; then
        log "Creating virtualenv at ${VENV_DIR}"
        python3 -m venv "${VENV_DIR}"
    fi
    activate_venv
}

hunyuan3d_dir() {
    echo "${REPO_ROOT}/${HUNYUAN3D_DIR:-vendor/Hunyuan3D-2}"
}

export_rocm_build_env() {
    export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}"
    export GPU_ARCHS="${GPU_ARCHS:-gfx1100}"
}
