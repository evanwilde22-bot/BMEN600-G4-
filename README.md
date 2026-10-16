# BMEN600BTTB
BMEN600 final project repository (Group 4)

* 1. We want to investigate the radiomic features in liposarcomas using CT/MR with specific mutations/expression clusters. Our biggest uncertainty is the high heterogeneity when comparing samples to...

revise - the overall project works, but we need to change - Sarcoma radiogenomic datasets are extremely limited (5 verified matched datasets). Non-sarcoma cancer types have much more accessible mat...

## Repository structure

```
BMEN600-G4-/
├── MNet/                         MedicalNet (Med3D) 3D-ResNet code, vendored from Tencent/MedicalNet (with local changes, see below)
│   ├── datasets/sarcoma.py       STS dataset loader (added for this project)
│   ├── model.py, setting.py      Model builder + CLI arguments
│   └── pretrain/                 (you create this) put downloaded pre-trained weights here
├── scripts/
│   ├── generate_file_lists.py    Build train.txt / val.txt from the STS data folder
│   ├── check_mask_values.py      Sanity-check a mask file (data validation)
│   ├── train_sarcoma.py          Training entry point for STS data
│   └── test_sarcoma.py           Evaluation entry point (mean Dice) for STS data
├── DATASETS.md                   Notes on datasets used for the project
├── TCGA-SARC-cohort-summary.csv  Cohort summary data
└── Section 6 - Teamwork Project plan.md
```

See [`MNet/README.md`](MNet/README.md) for the upstream MedicalNet documentation and [`scripts/README.md`](scripts/README.md) for the dataset scripts.

## Quick start

1. [Install](#1-install) (one-time)
2. [Download pre-trained weights](#2-download-pre-trained-weights-recommended) (recommended)
3. [Organize + validate your STS data](#3-prepare-and-validate-the-sts-data)
4. [Train](#4-train)
5. [Evaluate](#5-evaluate)

---

## 1. Install

Requires **Python 3.9+** (PyTorch >= 2.6 is required by `MNet/requirements.txt`) and `git`.

### Windows (PowerShell)

```powershell
git clone https://github.com/evanwilde22-bot/BMEN600-G4-.git
cd BMEN600-G4-
py -3 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r MNet/requirements.txt
```

- If activation is blocked, run `Set-ExecutionPolicy -Scope Process RemoteSigned` first, then activate again.
- **NVIDIA GPU users:** the default PyPI `torch` wheel on Windows is CPU-only. Install a CUDA build from <https://pytorch.org/get-started/locally/> *after* the step above, e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`

### Linux (bash)

```bash
git clone https://github.com/evanwilde22-bot/BMEN600-G4-.git
cd BMEN600-G4-
python3 -m venv venv
source venv/bin/activate
pip install -r MNet/requirements.txt
```

The PyPI `torch` wheel on Linux includes CUDA support. For a specific CUDA version, use the selector at <https://pytorch.org/get-started/locally/>.

### macOS (zsh/bash)

```bash
git clone https://github.com/evanwilde22-bot/BMEN600-G4-.git
cd BMEN600-G4-
python3 -m venv venv
source venv/bin/activate
pip install -r MNet/requirements.txt
```

macOS has no CUDA. **Add `--no_cuda` to every train/test command** (training will be slow on CPU).

### Verify the install

```bash
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

If CUDA is `False` and you expected a GPU, fix the torch install above, or run with `--no_cuda`.

### Command-line notes

Examples below use `\` for line continuation (Linux/macOS). In **PowerShell** use a backtick `` ` `` instead; in **cmd.exe** use `^`. Or just put each command on a single line.

---

## 2. Download pre-trained weights (recommended)

The STS training set is small (~40 training cases). Training from scratch overfits heavily (we observed near-perfect training Dice but ~0 validation Dice), so **starting from MedicalNet pre-trained weights is strongly recommended.**

1. Download the pre-trained models from the links in [`MNet/README.md`](MNet/README.md#demo) (Google Drive / Tencent Weiyun).
2. Create `MNet/pretrain/` and place the `.pth` files inside it.

The weights file **must match** the architecture flags you pass:

| Weights file | `--model_depth` | `--resnet_shortcut` |
|---|---|---|
| `resnet_10.pth` / `resnet_10_23dataset.pth` | `10` | `B` |
| `resnet_18.pth` / `resnet_18_23dataset.pth` | `18` | `A` |
| `resnet_34.pth` / `resnet_34_23dataset.pth` | `34` | `A` |
| `resnet_50.pth` / `resnet_50_23dataset.pth` | `50` | `B` |
| `resnet_101.pth` | `101` | `B` |
| `resnet_152.pth` | `152` | `B` |
| `resnet_200.pth` | `200` | `B` |

How the weights are used on the command line:

| Flag | Used by | Purpose |
|---|---|---|
| `--pretrain_path <file.pth>` | **training** | Initialize the backbone from MedicalNet weights (fine-tuning). Pass `--pretrain_path=""` to train from scratch. |
| `--resume_path <checkpoint.pth.tar>` | **evaluation** (and optionally training) | Load *your own* trained checkpoint. Evaluation uses this only; `--pretrain_path` is ignored when evaluating. |

> **Pitfall:** `--pretrain_path` defaults to `pretrain/resnet_50.pth`. If you don't pass it, training looks for that file and fails if it is missing. Always pass `--pretrain_path` explicitly (a real file, or `""` for scratch), and make sure it matches `--model_depth` / `--resnet_shortcut`.

---

## 3. Prepare and validate the STS data

### Folder and file names matter

The scripts locate files **by name**, so the STS data must be stored with exactly this naming and layout:

```
<DATA_ROOT>/                       <- the folder you pass as --data_root
    STS_001/                       <- case folders must start with "STS_"
        RTstruct_T1/               <- modality folder name must match --modality
            image.nii.gz           <- must match --image_file
            mask_GTV_Mass.nii.gz   <- must match --mask
            mask_GTV_Edema.nii.gz
        RTstruct_T2FS/
        RTstruct_CT/
        ...
    STS_002/
    ...
    STS_051/
    train.txt                      <- generated below
    val.txt                        <- generated below
```

Things that will silently break the pipeline:

- **Case folder names** must start with `STS_` (or pass `--case_prefix`). Folders that don't match are ignored.
- **Modality folder and file names** (`RTstruct_T1`, `image.nii.gz`, `mask_GTV_Mass.nii.gz`, ...) must match the `--modality`, `--image_file`, and `--mask` arguments exactly. Cases missing them are skipped (the script prints which ones).
- **`train.txt` / `val.txt` store paths relative to `--data_root`.** Use the *same* `--data_root` for generating the lists, training, and evaluation. If you move or rename the data folder, pass the new path as `--data_root` (the relative entries in the lists stay valid as long as the inside of the folder is unchanged).
- **No spaces inside case/modality/file names.** Each line of the list is `<image> <mask>` split on a single space.
- Keep the quotes around Windows paths that contain spaces.

### 3a. Validate a mask (data validation)

Confirms the mask contains the expected integer labels (STS masks store foreground as `255`; `SarcomaDataset` remaps it to class `1`):

```bash
python scripts/check_mask_values.py --path "/path/to/STS-nifti/nifti/STS_001/RTstruct_T1/mask_GTV_Mass.nii.gz"
```

```powershell
python scripts\check_mask_values.py --path "C:\path\to\STS-nifti\nifti\STS_001\RTstruct_T1\mask_GTV_Mass.nii.gz"
```

### 3b. Generate `train.txt` / `val.txt`

```bash
python scripts/generate_file_lists.py \
  --data_root "/path/to/STS-nifti/nifti" \
  --modality RTstruct_T1 \
  --mask mask_GTV_Mass.nii.gz
```

```powershell
python scripts\generate_file_lists.py `
  --data_root "C:\path\to\STS-nifti\nifti" `
  --modality RTstruct_T1 `
  --mask mask_GTV_Mass.nii.gz
```

| Flag | Default | Description |
|---|---|---|
| `--data_root` | (required) | Root directory containing `STS_XXX` case folders |
| `--modality` | `RTstruct_T1` | Modality subfolder (`RTstruct_T1`, `RTstruct_T2FS`, `RTstruct_CT`, `RTstruct_PET`, `RTstruct_AlignedT1toPET`, `RTstruct_AlignedT2FStoPET`) |
| `--image_file` | `image.nii.gz` | Image filename within the modality folder |
| `--mask` | `mask_GTV_Mass.nii.gz` | Mask filename (or `mask_GTV_Edema.nii.gz`) |
| `--val_fraction` | `0.2` | Fraction of cases reserved for validation |
| `--seed` | `1` | Random seed for the shuffle/split |
| `--case_prefix` | `STS_` | Prefix used to identify case folders |

This writes `train.txt` and `val.txt` into `--data_root` and prints how many cases were used/skipped.

---

## 4. Train

Run from the `MNet/` directory (so `pretrain/...` and `trails/...` paths resolve):

**Linux / macOS**

```bash
cd MNet
python ../scripts/train_sarcoma.py --gpu_id 0 --model_depth 18 --resnet_shortcut A \
  --input_D 32 --input_H 160 --input_W 160 --batch_size 1 --n_epochs 30 \
  --pretrain_path "pretrain/resnet_18.pth" \
  --data_root "/path/to/STS-nifti/nifti" \
  --img_list "/path/to/STS-nifti/nifti/train.txt"
```

**Windows (PowerShell)**

```powershell
cd MNet
python ..\scripts\train_sarcoma.py --gpu_id 0 --model_depth 18 --resnet_shortcut A `
  --input_D 32 --input_H 160 --input_W 160 --batch_size 1 --n_epochs 30 `
  --pretrain_path "pretrain/resnet_18.pth" `
  --data_root "C:\path\to\STS-nifti\nifti" `
  --img_list "C:\path\to\STS-nifti\nifti\train.txt"
```

Variations:

- **From scratch:** replace the pretrain flag with `--pretrain_path=""`.
- **CPU only / macOS:** add `--no_cuda` (and you can drop `--gpu_id`).
- **Multi-GPU:** `--gpu_id 0 1`.
- **Quick smoke test:** `--n_epochs 1`.
- **Resume from your own checkpoint:** add `--resume_path trails/models/sarcoma_resnet_18_epoch_<N>_final.pth.tar`.
- Other depths: change `--model_depth`/`--resnet_shortcut` **and** the weights file (see the table in step 2).

Outputs:

- Checkpoints are written after **every epoch** to `MNet/trails/models/sarcoma_resnet_<depth>_epoch_<N>_final.pth.tar`.
- **The filename depends only on depth and epoch, so a new run overwrites the previous run's checkpoints.** Rename/move `trails/` between experiments, and check `LastWriteTime` / file date before evaluating.
- Logs print `lr`, `loss`, `loss_seg`, and `loss_dice` per batch.

---

## 5. Evaluate

Computes mean Dice for the tumor class on the validation list. `--resume_path` is your trained checkpoint (pre-trained MedicalNet weights are not used here). The architecture flags must match the training run.

**Linux / macOS**

```bash
cd MNet
python ../scripts/test_sarcoma.py --gpu_id 0 --model_depth 18 --resnet_shortcut A \
  --input_D 32 --input_H 160 --input_W 160 \
  --resume_path trails/models/sarcoma_resnet_18_epoch_29_final.pth.tar \
  --data_root "/path/to/STS-nifti/nifti" \
  --img_list "/path/to/STS-nifti/nifti/val.txt"
```

**Windows (PowerShell)**

```powershell
cd MNet
python ..\scripts\test_sarcoma.py --gpu_id 0 --model_depth 18 --resnet_shortcut A `
  --input_D 32 --input_H 160 --input_W 160 `
  --resume_path trails/models/sarcoma_resnet_18_epoch_29_final.pth.tar `
  --data_root "C:\path\to\STS-nifti\nifti" `
  --img_list "C:\path\to\STS-nifti\nifti\val.txt"
```

Output: `mean dice for class-1 is <value>`. (Add `--no_cuda` on CPU-only machines.) A `FutureWarning` about `torch.load` is expected and harmless.

---

## Changes made to the vendored MNet code

`MNet/` started as an unmodified copy of [Tencent/MedicalNet](https://github.com/Tencent/MedicalNet). Project-specific changes:

| Area | Change |
|---|---|
| `MNet/datasets/sarcoma.py` (new) | `SarcomaDataset` for the STS NIfTI data. Remaps mask foreground (`255`) to class `1`. **Training uses the same preprocessing as testing** (trim empty background, resize, intensity-normalize) with **no tumor-centered random crop**, so tumor scale matches between training and inference. Training-only augmentation: random per-axis flips and mild intensity scale/shift. |
| `MNet/model.py` | Pre-trained weights are loaded with `torch.load(..., weights_only=True, map_location='cpu')` (PyTorch >= 2.6 safe loading) and are only loaded when `--phase` is not `test`; with `--pretrain_path` set, the new `conv_seg` head gets a 100x learning rate vs. the backbone. |
| `MNet/requirements.txt` | `torch>=2.6.0`, `numpy>=1.22.0`, `scipy>=1.8.0` (security advisories in the original pins). |
| `scripts/train_sarcoma.py` | Uses `SarcomaDataset`. Loss = class-weighted cross-entropy (tumor weight `FOREGROUND_CE_WEIGHT = 20`) **+ soft Dice loss** to counter extreme tumor/background class imbalance. `ExponentialLR` (gamma 0.99) now steps **once per epoch** (it was per batch, which collapsed the learning rate). Saves a checkpoint every epoch. |
| `scripts/test_sarcoma.py` | Evaluates mean Dice with the same `255 -> 1` mask remap as training. |

The original `MNet/train.py`, `MNet/test.py`, and `MNet/datasets/brains18.py` are left untouched and are for the MRBrainS18 demo, **not** the STS data. Use the `scripts/*_sarcoma.py` entry points for this project.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `FileNotFoundError` for `pretrain/resnet_50.pth` | `--pretrain_path` wasn't passed (default is `resnet_50.pth`). Pass a real file or `--pretrain_path=""`. |
| Shape/key mismatch when loading pre-trained weights | Weights file doesn't match `--model_depth` / `--resnet_shortcut`. |
| `Skipped N cases missing ...` when generating lists | Folder/file names don't match `--modality`, `--image_file`, `--mask`, or case prefix. |
| `AssertionError` on an image/mask path | `--data_root` differs from the one used to generate the lists, or the data folder was moved/renamed. |
| `TypeError` about `gpu_id` | Pass `--gpu_id 0` (or `--no_cuda`). |
| `torch.cuda.is_available()` is `False` | CPU-only torch installed (common on Windows) or no GPU; install a CUDA build or use `--no_cuda`. |
| Evaluation looks unchanged after retraining | Old checkpoint with the same filename; check its modification date. |
| Train Dice ~0.99 but validation Dice ~0 | Overfitting on a small dataset; use pre-trained weights and the augmentation already in `SarcomaDataset`. |
