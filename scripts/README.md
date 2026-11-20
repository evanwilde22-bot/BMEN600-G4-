# Scripts

Utility scripts for preparing the Soft-tissue-Sarcoma (STS) NIfTI dataset for
training with the MNet model in `MNet/`.

## Expected data layout

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
    RTstruct_PET/
        ...
    RTstruct_AlignedT1toPET/
        ...
    RTstruct_AlignedT2FStoPET/
        ...
STS_002/
    ...
...
STS_051/
    ...
```

## check_mask_values.py

Sanity-checks a single mask file before training — confirms it contains the
expected integer label values (e.g. `[0, 1]` for binary tumor segmentation).

```powershell
python check_mask_values.py --path "C:\path\to\STS_001\RTstruct_T1\mask_GTV_Mass.nii.gz"
```

## generate_file_lists.py

Scans all `STS_XXX` case folders under `--data_root`, pairs up the image and
mask file for the chosen modality, shuffles, splits into train/val, and
writes `train.txt` / `val.txt` directly into `--data_root` in the format
expected by `MNet/datasets/brains18.py`:

```
<relative image path> <relative mask path>
```

```powershell
python generate_file_lists.py `
  --data_root "C:\Users\Evan\Documents\ManskeLab\BMEN600\NIFTI\STS-nifti\nifti" `
  --modality RTstruct_T1 `
  --mask mask_GTV_Mass.nii.gz
```

Arguments:

| Flag | Default | Description |
|---|---|---|
| `--data_root` | (required) | Root directory containing `STS_XXX` case folders |
| `--modality` | `RTstruct_T1` | One or more modality subfolders (`RTstruct_T1`, `RTstruct_T2FS`, `RTstruct_CT`, `RTstruct_PET`, `RTstruct_AlignedT1toPET`, `RTstruct_AlignedT2FStoPET`) |
| `--image_file` | `image.nii.gz` | Image filename within the modality folder |
| `--mask` | `mask_GTV_Mass.nii.gz` | Mask filename within the modality folder (or `mask_GTV_Edema.nii.gz`) |
| `--val_fraction` | `0.2` | Fraction of cases reserved for validation |
| `--seed` | `1` | Random seed for the shuffle/split |
| `--case_prefix` | `STS_` | Prefix used to identify case folders |

### Multi-modality trial (T1, T2FS, and both PET-aligned)

Pass several modalities to pool them into a single train/val split. Each
available modality of a case becomes its own single-channel sample, and the
split is done per case so no patient appears in both train and val.

```bash
python scripts/generate_file_lists.py \
  --data_root "/path/to/STS-nifti" --mask mask_GTV_Mass.nii.gz \
  --modality RTstruct_T1 RTstruct_T2FS RTstruct_AlignedT1toPET RTstruct_AlignedT2FStoPET

cd MNet
python ../scripts/train_sarcoma.py --gpu_id 0 --model_depth 18 --resnet_shortcut A \
  --input_D 32 --input_H 160 --input_W 160 --batch_size 1 --n_epochs 50 \
  --dropout_prob 0.2 --pretrain_path "" \
  --data_root "/path/to/STS-nifti" \
  --img_list "/path/to/STS-nifti/train.txt" \
  --val_img_list "/path/to/STS-nifti/val.txt"
```

Add `--resume_path <checkpoint>` to continue from an existing checkpoint
(the input layer is unchanged, so checkpoints stay compatible).

After running, point MNet's training/testing at the generated lists:

```powershell
python train.py --gpu_id 0 --model_depth 18 --resnet_shortcut A `
  --input_D 32 --input_H 160 --input_W 160 --batch_size 1 --n_epochs 1 `
  --data_root "C:\Users\Evan\Documents\ManskeLab\BMEN600\NIFTI\STS-nifti\nifti" `
  --img_list "C:\Users\Evan\Documents\ManskeLab\BMEN600\NIFTI\STS-nifti\nifti\train.txt" `
  --pretrain_path=""
```

## Resuming from the latest checkpoint (`resume_latest.ps1`)

`resume_latest.ps1` picks a checkpoint for `train_sarcoma.py --resume_path`. It prefers `MNet/trails/models/sarcoma_resnet_<depth>_best.pth.tar` (best validation Dice, saved when `--val_img_list` is set), falls back to the most recently modified `sarcoma_resnet_<depth>_epoch_*_final.pth.tar`, and otherwise returns an empty string so training starts from `--pretrain_path`. `-ModelDepth` defaults to 18.

```powershell
$resume = & ./scripts/resume_latest.ps1 -ModelDepth 18
cd MNet
python ..\scripts\train_sarcoma.py --gpu_id 0 --model_depth 18 --resnet_shortcut A `
  --input_D 32 --input_H 160 --input_W 160 --batch_size 1 --n_epochs 30 `
  --dropout_prob 0.2 --val_interval 2 `
  --pretrain_path "pretrain/resnet_18.pth" `
  --resume_path $resume `
  --data_root "C:\path\to\STS-nifti\nifti" `
  --img_list "C:\path\to\STS-nifti\nifti\train.txt" `
  --val_img_list "C:\path\to\STS-nifti\nifti\val.txt"
```

`--dropout_prob 0.2` counters overfitting on the small STS dataset, and a lower `--val_interval` gives best-checkpoint tracking more samples in short resumed runs (at the cost of extra evaluation time).

## Late fusion of four native modalities

This experimental workflow trains four **separate** single-channel 3D models:
CT, T1, T2FS/STIR and PET. It combines their foreground probability maps,
not the input scans, with a weighted average. It uses only native modalities,
not the `RTstruct_Aligned*` images. T2FS and STIR share one model; the
generator rejects cases containing both rather than silently choosing one.

The split is by patient and requires all four images and their mass masks.
With the current local collection, 50 complete patients yield 30 training,
10 validation and 10 test patients at the default fractions. The small test
set limits statistical confidence. Training and validation use the same
full-volume preprocessing; the loss is computed at input resolution to avoid
erasing small tumors when downsampling masks to the model output grid.
This still does not guarantee that the model resolves small lesions.

Install the updated requirements first (`pip install -r MNet/requirements.txt`).
The following macOS/Linux commands run from the repository root, with the
virtual environment activated. Set `DATA_ROOT` to the actual STS directory.
The output folders are separate from the dataset. On Windows, adapt the
variable assignments and loop to PowerShell.

```bash
ROOT="$PWD"
DATA_ROOT="$ROOT/STS-nifti"
LISTS="$ROOT/fusion_lists"
python scripts/generate_fusion_lists.py --data_root "$DATA_ROOT" --output_dir "$LISTS"

cd "$ROOT/MNet"
for mod in ct t1 t2 pet; do
  python ../scripts/train_sarcoma.py --no_cuda \
    --model_depth 18 --resnet_shortcut A \
    --input_D 32 --input_H 160 --input_W 160 --batch_size 1 \
    --n_epochs 30 --val_interval 2 --dropout_prob 0.2 \
    --pretrain_path "pretrain/resnet_18.pth" \
    --consistent_preprocessing --modality_name "$mod" \
    --data_root "$DATA_ROOT" \
    --img_list "$LISTS/train_${mod}.txt" \
    --val_img_list "$LISTS/val_${mod}.txt" \
    --output_prefix "$ROOT/fusion_models/$mod"
done
cd "$ROOT"
```

For CUDA replace `--no_cuda` with `--gpu_id 0`. Keep output prefixes distinct:
rerunning an experiment with the same prefix overwrites its checkpoints.
Use validation patients to choose checkpoints and fusion weights. Do not tune
on the test patients. These new models must be trained before fusion; older
checkpoints do not contain the required preprocessing and modality metadata.

For each test patient, align the native images without using tumor masks, then
resample predictions into the chosen reference image's grid. The preferred
input is a **visually verified** SimpleITK `.tfm` transform for each moving
image, named `<case>_<reference>_to_<moving>.tfm`, e.g.
`STS_001_ct_to_t1.tfm`. Despite the filename, the transform must map
**reference/fixed physical coordinates to moving physical coordinates**, as
required by SimpleITK resampling. Use `--transform_dir` to supply these files.

Alternatively, the following prepares *candidate* rigid mutual-information
transforms and registered images; it does not run inference or evaluation:

```bash
python scripts/fuse_sarcoma.py \
  --data_root "$DATA_ROOT" --cases "$LISTS/test_cases.txt" \
  --output_dir "$ROOT/registration_candidates" \
  --modalities ct t1 t2 pet --reference ct --auto_register
```

Registration is not guaranteed to align different fields of view or patient
positions. **Inspect every registered image against the reference**, correct
failed registrations with suitable tools, and save the verified transforms
before fusion. There is no automatic quality validation. In one local
MRI/PET case, automatic registration converged but the masks did not overlap;
convergence is not evidence of correct alignment. Do not use tumor masks to
derive registration transforms.

For example, after training, this runs equal-weight four-modality inference
in CT space on the held-out test patients:

```bash
python scripts/fuse_sarcoma.py \
  --data_root "$DATA_ROOT" --cases "$LISTS/test_cases.txt" \
  --output_dir "$ROOT/fusion_predictions/all" \
  --modalities ct t1 t2 pet --reference ct \
  --ct_checkpoint "$ROOT/fusion_models/ct_best.pth.tar" \
  --t1_checkpoint "$ROOT/fusion_models/t1_best.pth.tar" \
  --t2_checkpoint "$ROOT/fusion_models/t2_best.pth.tar" \
  --pet_checkpoint "$ROOT/fusion_models/pet_best.pth.tar" \
  --transform_dir "$ROOT/verified_transforms" --evaluate
```

Weights default to 1; set, for example, `--ct_weight 2` to
change them. At each voxel only modalities covering that voxel contribute to
the normalized average. `--threshold` defaults to 0.5. Output includes native
and registered probability maps, registered images, the fused probability
and mask, and per-case/mean Dice when `--evaluate` is set. The reference mask
is used only for evaluation. Single-modality inference needs no registration:

```bash
python scripts/fuse_sarcoma.py \
  --data_root "$DATA_ROOT" --cases "$LISTS/test_cases.txt" \
  --output_dir "$ROOT/fusion_predictions/ct_only" \
  --modalities ct --ct_checkpoint "$ROOT/fusion_models/ct_best.pth.tar" \
  --evaluate
```

Repeat with `--modalities t1`, `t2` or `pet` and the corresponding checkpoint.
Each single-modality score is measured in that modality's native mask space;
fused scores use the reference modality's mask. Check annotation and
registration differences when comparing scores. `--device cuda:0` enables
GPU inference; all models must use the specified common architecture and
`--input_size` (defaults match the training example). The workflow is for
research, not clinical use.
