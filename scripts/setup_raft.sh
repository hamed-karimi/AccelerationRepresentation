#!/usr/bin/env bash
# Clone RAFT and download pretrained weights for optical-flow feature extraction.
#
# Usage (from repository root):
#   bash scripts/setup_raft.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAFT_DIR="${RAFT_DIR:-${REPO_ROOT}/RAFT}"
RAFT_URL="https://github.com/princeton-vl/RAFT.git"

log() { echo "[setup_raft] $*"; }

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

require_cmd git
require_cmd bash

if [[ ! -d "$RAFT_DIR/.git" ]]; then
    log "cloning ${RAFT_URL} -> ${RAFT_DIR}"
    git clone "$RAFT_URL" "$RAFT_DIR"
else
    log "using existing clone: ${RAFT_DIR}"
fi

cd "$RAFT_DIR"

if [[ ! -f download_models.sh ]]; then
    echo "ERROR: download_models.sh not found in ${RAFT_DIR}" >&2
    exit 1
fi

log "downloading pretrained models (raft-things.pth, ...)"
bash download_models.sh

if [[ ! -f models/raft-things.pth ]]; then
    echo "ERROR: models/raft-things.pth not found after download" >&2
    exit 1
fi

log "done."
log "RAFT is ready at ${RAFT_DIR}"
