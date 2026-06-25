#!/usr/bin/env bash
# Prepare StudyForrest fMRI data in the layout expected by this repository.
#
# Prerequisites: datalad, git-annex
#   https://handbook.datalad.org/en/latest/intro/installation.html
#
# Usage (from repository root):
#   bash scripts/prepare_studyforrest_data.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLONE_DIR="${STUDYFORREST_CLONE_DIR:-${REPO_ROOT}/studyforrest-data}"
TARGET_DIR="${FORREST_DATA_DIR:-${REPO_ROOT}/forrest_study_fmri}"
DATASET_URL="https://github.com/psychoinformatics-de/studyforrest-data"

# Subjects used in the analysis scripts
SUBJECTS=(1 2 3 4 5 9 10 14 15 16 17 18 19 20)
RUNS=(1 2 3 4 5 6 7 8)

log() { echo "[prepare_studyforrest_data] $*"; }

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

find_source_file() {
    local pattern="$1"
    local result
    result="$(find "$CLONE_DIR" -type f -name "$pattern" 2>/dev/null | head -n 1 || true)"
    if [[ -z "$result" ]]; then
        return 1
    fi
    printf '%s' "$result"
}

link_file() {
    local src="$1"
    local dst="$2"
    mkdir -p "$(dirname "$dst")"
    if [[ -e "$dst" ]]; then
        log "already exists: $dst"
        return 0
    fi
    ln -s "$(realpath "$src")" "$dst"
    log "linked: $dst"
}

require_cmd datalad
require_cmd git-annex

if [[ ! -d "$CLONE_DIR/.git" ]]; then
    log "cloning ${DATASET_URL} -> ${CLONE_DIR}"
    datalad clone "$DATASET_URL" "$CLONE_DIR"
else
    log "using existing clone: ${CLONE_DIR}"
fi

cd "$CLONE_DIR"

log "fetching dataset metadata (this may take a while)"
datalad get -n .

if [[ -d derivative ]]; then
    log "installing derivative subdataset metadata"
    datalad get -n derivative || true
fi

log "arranging files under ${TARGET_DIR}"

for subject in "${SUBJECTS[@]}"; do
    sub_id="$(printf '%02d' "$subject")"
    func_dir="${TARGET_DIR}/sub-${sub_id}_complete/func"
    anat_dir="${TARGET_DIR}/sub-${sub_id}_complete/anat"
    mkdir -p "$func_dir" "$anat_dir"

    for run in "${RUNS[@]}"; do
        func_name="sub-${sub_id}_ses-movie_task-movie_run-${run}_bold_space-T1w_preproc.nii.gz"
        src="$(find_source_file "$func_name")" || {
            echo "ERROR: could not find ${func_name} under ${CLONE_DIR}" >&2
            echo "Try retrieving derivative data, e.g.:" >&2
            echo "  cd ${CLONE_DIR} && datalad get -r derivative" >&2
            exit 1
        }
        log "retrieving ${src}"
        datalad get "$src"
        link_file "$src" "${func_dir}/${func_name}"
    done

    for tissue in GM WM CSF; do
        anat_name="sub-${sub_id}_T1w_class-${tissue}_probtissue.nii.gz"
        src="$(find_source_file "$anat_name")" || {
            echo "ERROR: could not find ${anat_name} under ${CLONE_DIR}" >&2
            exit 1
        }
        log "retrieving ${src}"
        datalad get "$src"
        link_file "$src" "${anat_dir}/${anat_name}"
    done

    # Optional but required by write_mni_tmaps.py for SPM normalization
    t1_name="sub-${sub_id}_T1w_preproc.nii.gz"
    if src="$(find_source_file "$t1_name")"; then
        log "retrieving ${src}"
        datalad get "$src"
        link_file "$src" "${anat_dir}/${t1_name}"
    else
        log "warning: ${t1_name} not found (needed for write_mni_tmaps.py)"
    fi
done

log "done."
log "Next step: run CompCorr preprocessing"
log "  cd preprocessing && python CompCorr.py"
