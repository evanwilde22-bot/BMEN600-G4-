# BMEN600BTTB
BMEN600 final project repository (Group 4)

* 1. We want to investigate the radiomic features in liposarcomas using CT/MR with specific mutations/expression clusters. Our biggest uncertainty is the high heterogeneity when comparing samples to...

revise - the overall project works, but we need to change - Sarcoma radiogenomic datasets are extremely limited (5 verified matched datasets). Non-sarcoma cancer types have much more accessible mat...

## Repository structure

```
BMEN600-G4-/
├── MNet/                         MedicalNet (Med3D) model code, vendored from Tencent/MedicalNet
├── scripts/                      Utility scripts for preparing the STS dataset for training/testing
├── DATASETS.md                   Notes on datasets used for the project
├── TCGA-SARC-cohort-summary.csv  Cohort summary data
└── Section 6 - Teamwork Project plan.md
```

See [`MNet/README.md`](MNet/README.md) for full MedicalNet documentation, and [`scripts/README.md`](scripts/README.md) for details on the dataset preparation scripts.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/evanwilde22-bot/BMEN600-G4-.git
   cd BMEN600-G4-
   ```

2. **Set up a Python environment** (Python 3.7+ recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate    # on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   The MedicalNet model (`MNet/`) has its own requirements file:
   ```bash
   pip install -r MNet/requirements.txt
   ```

   If you need the vendored MedicalNet pre-trained weights, download them from the links in [`MNet/README.md`](MNet/README.md#demo) and place them under `MNet/pretrain/`.

## Preparing the dataset

Training and testing expect data in the following layout (see [`scripts/README.md`](scripts/README.md) for full details):

```
STS_001/
    RTstruct_T1/
        image.nii.gz
        mask_GTV_Edema.nii.gz
        mask_GTV_Mass.nii.gz
    RTstruct_T2FS/
        ...
    RTstruct_CT/
        ...
...
STS_051/
    ...
```

1. (Optional) Sanity-check a mask file before training:
   ```bash
   python scripts/check_mask_values.py --path "/path/to/STS_001/RTstruct_T1/mask_GTV_Mass.nii.gz"
   ```

2. Generate the `train.txt` / `val.txt` file lists expected by `MNet/datasets/brains18.py`:
   ```bash
   python scripts/generate_file_lists.py \
     --data_root "/path/to/STS-nifti/nifti" \
     --modality RTstruct_T1 \
     --mask mask_GTV_Mass.nii.gz
   ```

   Key arguments:

   | Flag | Default | Description |
   |---|---|---|
   | `--data_root` | (required) | Root directory containing `STS_XXX` case folders |
   | `--modality` | `RTstruct_T1` | Modality subfolder (`RTstruct_T1`, `RTstruct_T2FS`, `RTstruct_CT`, `RTstruct_PET`, `RTstruct_AlignedT1toPET`, `RTstruct_AlignedT2FStoPET`) |
   | `--image_file` | `image.nii.gz` | Image filename within the modality folder |
   | `--mask` | `mask_GTV_Mass.nii.gz` | Mask filename within the modality folder (or `mask_GTV_Edema.nii.gz`) |
   | `--val_fraction` | `0.2` | Fraction of cases reserved for validation |
   | `--seed` | `1` | Random seed for the shuffle/split |
   | `--case_prefix` | `STS_` | Prefix used to identify case folders |

## Training

Train using the generated file lists and the MedicalNet 3D-ResNet backbone:

```bash
python MNet/train.py --gpu_id 0 --model_depth 18 --resnet_shortcut A \
  --input_D 32 --input_H 160 --input_W 160 --batch_size 1 --n_epochs 1 \
  --data_root "/path/to/STS-nifti/nifti" \
  --img_list "/path/to/STS-nifti/nifti/train.txt" \
  --pretrain_path=""
```

- Use `--gpu_id 0 1` for multi-GPU training, or a single ID (e.g. `--gpu_id 0`) for single-GPU training.
- Set `--pretrain_path` to a MedicalNet pre-trained checkpoint (e.g. `resnet_18.pth`) to fine-tune instead of training from scratch.
- There is also a project-specific training script, `scripts/train_sarcoma.py`, tailored to the sarcoma dataset:
  ```bash
  python scripts/train_sarcoma.py
  ```
  (see the script for configurable arguments/paths).

## Testing / Evaluation

Run inference/evaluation against the validation set with a trained checkpoint:

```bash
python MNet/test.py --gpu_id 0 \
  --resume_path trails/models/resnet_18_epoch_<N>_batch_<M>.pth.tar \
  --img_list "/path/to/STS-nifti/nifti/val.txt"
```

A project-specific testing script is also available:

```bash
python scripts/test_sarcoma.py
```

For additional model details, pre-trained weights, and benchmark results, see [`MNet/README.md`](MNet/README.md).
