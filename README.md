# Motion Encoding fMRI Analysis

Accompanying code for encoding-model analyses of optical-flow motion features (velocity and acceleration) against fMRI data from the [StudyForrest](https://studyforrest.org/) project.

## Requirements

Install Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Additional external tools:

- **DataLad** and **git-annex** — for downloading StudyForrest data ([installation guide](https://handbook.datalad.org/en/latest/intro/installation.html))
- **SPM12** — for MNI normalization in `write_mni_tmaps.py` (passed via `--spm12-path`)
- **RAFT** — cloned into `./RAFT` (see [RAFT setup](#raft-setup) below)

---

## Data Preparation

Functional and anatomical MRI files must be organized under `forrest_study_fmri/` at the repository root (sibling to `analyses/` and `preprocessing/`).

### Subjects

The analysis scripts use these subject IDs:

`1, 2, 3, 4, 5, 9, 10, 14, 15, 16, 17, 18, 19, 20`

### Expected folder structure

```
forrest_study_fmri/
├── sub-01_complete/
│   ├── func/
│   │   ├── sub-01_ses-movie_task-movie_run-1_bold_space-T1w_preproc.nii.gz
│   │   ├── sub-01_ses-movie_task-movie_run-2_bold_space-T1w_preproc.nii.gz
│   │   ├── sub-01_ses-movie_task-movie_run-3_bold_space-T1w_preproc.nii.gz
│   │   ├── sub-01_ses-movie_task-movie_run-4_bold_space-T1w_preproc.nii.gz
│   │   ├── sub-01_ses-movie_task-movie_run-5_bold_space-T1w_preproc.nii.gz
│   │   ├── sub-01_ses-movie_task-movie_run-6_bold_space-T1w_preproc.nii.gz
│   │   ├── sub-01_ses-movie_task-movie_run-7_bold_space-T1w_preproc.nii.gz
│   │   ├── sub-01_ses-movie_task-movie_run-8_bold_space-T1w_preproc.nii.gz
│   │   └── compcorr/                          # created by preprocessing/CompCorr.py
│   │       ├── func_run-01.nii.gz
│   │       ├── func_run-02.nii.gz
│   │       └── ...
│   └── anat/
│       ├── sub-01_T1w_class-GM_probtissue.nii.gz
│       ├── sub-01_T1w_class-WM_probtissue.nii.gz
│       ├── sub-01_T1w_class-CSF_probtissue.nii.gz
│       └── sub-01_T1w_preproc.nii.gz          # needed for write_mni_tmaps.py
├── sub-02_complete/
│   ├── func/
│   └── anat/
├── ...
└── MNI/                                       # needed for write_mni_tmaps.py
    └── mni_icbm152_nlin_asym_09c/
        └── 1mm_T1.nii.gz
```

The same `sub-{XX}_complete/{func,anat}` layout is repeated for every subject listed above.

### Download and arrange StudyForrest data

Data are retrieved from the StudyForrest DataLad superdataset:

[https://github.com/psychoinformatics-de/studyforrest-data](https://github.com/psychoinformatics-de/studyforrest-data)

From the repository root, run:

```bash
bash scripts/prepare_studyforrest_data.sh
```

This script will:

1. Clone the StudyForrest DataLad superdataset into `./studyforrest-data` (or `$STUDYFORREST_CLONE_DIR` if set)
2. Fetch dataset metadata and retrieve required preprocessed files with `datalad get`
3. Create symbolic links under `./forrest_study_fmri/` using the filenames expected by the analysis code

Optional environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `STUDYFORREST_CLONE_DIR` | `./studyforrest-data` | Where to clone the DataLad dataset |
| `FORREST_DATA_DIR` | `./forrest_study_fmri` | Output directory for the organized files |

If file retrieval fails, you may need to download derivative data manually first:

```bash
cd studyforrest-data
datalad get -r derivative
```

See the [StudyForrest data access page](https://studyforrest.org/access.html) for more detail.

### fMRI denoising (CompCor)

After the raw/preprocessed files are in place, run:

```bash
cd preprocessing
python CompCorr.py
```

This writes CompCor-denoised functional runs to `forrest_study_fmri/sub-{XX}_complete/func/compcorr/func_run-{RR}.nii.gz`.

---

## RAFT setup

Optical-flow features are extracted with [RAFT](https://github.com/princeton-vl/RAFT) (ECCV 2020). Clone the repository and download pretrained weights into `./RAFT`:

```bash
bash scripts/setup_raft.sh
```

This script will:

1. Clone [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) into `./RAFT` (or `$RAFT_DIR` if set)
2. Run RAFT's `download_models.sh` to fetch `models/raft-things.pth`

Expected layout:

```
RAFT/
├── core/                    # imported by raft_feature_extraction.py
├── models/
│   └── raft-things.pth      # pretrained weights
└── download_models.sh
```

Optional environment variable:

| Variable | Default | Purpose |
|----------|---------|---------|
| `RAFT_DIR` | `./RAFT` | Where to clone RAFT |

Feature extraction then writes outputs to:

```
features/RAFT/FlyingThings3D-trained-cropped/seg_{0-7}/flow.pt
```

Run from `analyses/`:

```bash
python raft_feature_extraction.py /path/to/stimulus/frames --segment 0
```

Stimulus frames should be organized as `seg_{N}/f_000000.jpg`, ... under the path you pass to the script.

---

## Reproducing the t-maps

All analysis commands below are run from the `analyses/` directory. They assume CompCor-preprocessed fMRI data and RAFT flow features (`features/RAFT/.../flow.pt`) are already in place.

### Overview

| Map | Script | Output folder (per subject) |
|-----|--------|-----------------------------|
| Velocity | `encoding_model.py --module velocity` | `results/EncodingModel/subject-{XX}/velocity/` |
| Acceleration | `encoding_model.py --module acceleration` | `results/EncodingModel/subject-{XX}/acceleration/` |
| Unique acceleration (velocity controlled) | `control_variance_pred_fmri.py` | `results/EncodingModel/subject-{XX}/control velocity predict acceleration/` |
| Unique velocity (acceleration controlled) | `control_variance_pred_fmri.py` | `results/EncodingModel/subject-{XX}/control acceleration predict velocity/` |

Each first-level script writes segment-level results under `segments/GLMResult/`, including `t_map.npy` and ridge coefficients (`seg_{0-7}_model_coef.npy`).

### Step 1 — Velocity and acceleration maps

Fit the encoding model separately for each subject and each motion module:

```bash
cd analyses

python encoding_model.py 1 --module velocity
python encoding_model.py 1 --module acceleration
```

Repeat for all subjects: `1, 2, 3, 4, 5, 9, 10, 14, 15, 16, 17, 18, 19, 20`.

Example loop:

```bash
for sub in 1 2 3 4 5 9 10 14 15 16 17 18 19 20; do
  python encoding_model.py "$sub" --module velocity
  python encoding_model.py "$sub" --module acceleration
done
```

**Outputs (per subject and module):**

```
results/EncodingModel/subject-01/velocity/segments/GLMResult/t_map.npy
results/EncodingModel/subject-01/acceleration/segments/GLMResult/t_map.npy
```

Run both velocity and acceleration models before the unique-variance step — `control_variance_pred_fmri.py` needs ridge coefficients from the encoding models.

### Step 2 — Unique acceleration and unique velocity maps

These analyses partial out one motion module and test whether the other explains remaining fMRI variance.

**Unique acceleration** (control for velocity, test acceleration):

```bash
python control_variance_pred_fmri.py 1 --control-module velocity --predictor-module acceleration
```

**Unique velocity** (control for acceleration, test velocity):

```bash
python control_variance_pred_fmri.py 1 --control-module acceleration --predictor-module velocity
```

Example loop for both unique maps:

```bash
for sub in 1 2 3 4 5 9 10 14 15 16 17 18 19 20; do
  python control_variance_pred_fmri.py "$sub" --control-module velocity --predictor-module acceleration
  python control_variance_pred_fmri.py "$sub" --control-module acceleration --predictor-module velocity
done
```

**Outputs:**

```
results/EncodingModel/subject-01/control velocity predict acceleration/segments/GLMResult/t_map.npy
results/EncodingModel/subject-01/control acceleration predict velocity/segments/GLMResult/t_map.npy
```

### Step 3 — Group-level permutation test (FWE)

`second_level_permutation_test.py` runs non-parametric permutation inference across subjects with cluster-level correction. It loads subject-level NIfTI maps from `results/EncodingModel/subject-{XX}/{module}/` and writes group results to `results/EncodingModel/Mean/MNI/{module}/`.

Run once per map type:

```bash
python second_level_permutation_test.py --module acceleration
python second_level_permutation_test.py --module velocity
python second_level_permutation_test.py --module "control velocity predict acceleration"
python second_level_permutation_test.py --module "control acceleration predict velocity"
```

Group-level outputs include cluster log-p maps and significance masks, e.g.:

```
results/EncodingModel/Mean/MNI/acceleration/logp_max_size wt_map 0.0002 6.nii
results/EncodingModel/Mean/MNI/acceleration/logp_max_size 10000 wt_map 0.0002 6 sig mask.nii
```

**Note:** Second-level inference expects subject-level NIfTI maps (`wt_map.nii`) in each subject's module directory. Convert first-level `t_map.npy` results to MNI space with `write_mni_tmaps.py` (requires SPM12) before running the permutation test:

```bash
python write_mni_tmaps.py 1 --module velocity --spm12-path /path/to/spm12
python write_mni_tmaps.py 1 --module acceleration --spm12-path /path/to/spm12
```

### Full first-level workflow (summary)

```
raft_feature_extraction.py  →  features/RAFT/.../flow.pt
         ↓
encoding_model.py (--module velocity | acceleration)  →  t_map.npy
         ↓
control_variance_pred_fmri.py  →  unique t_map.npy
         ↓
write_mni_tmaps.py  →  subject-level NIfTI maps
         ↓
second_level_permutation_test.py  →  group maps (FWE-corrected)
```

---

## Contact

If you have questions about this code, or notice or run into any problems while reproducing the analyses, please contact **Hamed Karimi** at [hamedkk72@gmail.com](mailto:hamedkk72@gmail.com).

