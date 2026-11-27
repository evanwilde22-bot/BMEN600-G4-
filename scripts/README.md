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

## Experimental T2 patch model

This separate T2FS/STIR experiment retains the same `full-02` patient split
for comparison. Images are reoriented to RAS and resampled to 1.5 x 1.5 x
7 mm. Training samples patches centered on tumor voxels and image voxels;
inference uses overlapping patches across the entire image without the
reference mask. Patches are averaged before predictions are returned to the
original image grid. This approach is experimental and does not guarantee
better Dice or eliminate boundary artifacts.

The command runs on CPU and has no hard time limit. It uses MedicalNet's
pretrained ResNet-18 backbone, not the previous T2 checkpoint, because the
preprocessing and training task have changed. The default split requires the
lists from `full-02`; use `--lists` to supply another directory containing
`train_t2.txt`, `val_t2.txt`, and `test_t2.txt`. Patients must not overlap.
Each run snapshots those lists and requires a new output directory. It saves
only the latest and best checkpoints, so it uses much less disk than saving
every epoch. The latest checkpoint is saved for recovery, but automatic resume
is not implemented for this experimental runner.

From the repository root or `MNet/`:

```bash
"$HOME/Documents/BMEN600-G4-/venv/bin/python" "$HOME/Documents/BMEN600-G4-/scripts/t2_patches.py" --run-dir "$HOME/Documents/BMEN600-G4-/fusion-runs/t2-patches-01" --epochs 30 --patches-per-case 2 --val-interval 5
```

Validation uses the full image and native reference grid every five epochs
and on the last epoch. It takes significantly longer than training one patch;
actual runtime depends on image sizes and hardware. To create native-grid
validation masks for visual inspection:

```bash
"$HOME/Documents/BMEN600-G4-/venv/bin/python" "$HOME/Documents/BMEN600-G4-/scripts/t2_patches.py" --run-dir "$HOME/Documents/BMEN600-G4-/fusion-runs/t2-patches-01" --evaluate val
```

The command prints the output directory and per-patient Dice. Each case
contains `t2_mask.nii.gz` and `t2_probability.nii.gz`. Use `--evaluate test`
only after making modeling choices on the validation cohort. The original
test cases were inspected during registration development; use an untouched
external cohort for final assessment. `--data-root` can override the
recorded data location if the dataset is moved.

## Late fusion of four native modalities

This experimental workflow trains four **separate** single-channel 3D models:
CT, T1, T2FS/STIR and PET. It combines their foreground probability maps,
not the input scans, with a weighted average. Models use native modalities;
automatic transform recovery uses the dataset's `RTstruct_Aligned*` MRI
copies as registration references only, never as model inputs. T2FS and STIR
share one model; the generator rejects cases containing both rather than
silently choosing one.

The split is by patient and requires all four images and their mass masks.
With the current local collection, 50 complete patients yield 30 training,
10 validation and 10 test patients at the default fractions. The small test
set limits statistical confidence. Training and validation use the same
full-volume preprocessing; the loss is computed at input resolution to avoid
erasing small tumors when downsampling masks to the model output grid.
This still does not guarantee that the model resolves small lesions.

Install the updated requirements first (`pip install -r MNet/requirements.txt`).
For a one-epoch smoke test of all four models, run from the repository root:

```bash
python scripts/smoke_fusion.py
```

By default this uses `STS-nifti`, the pretrained
`MNet/pretrain/resnet_18.pth`, 10 training patients and 3 validation patients
per model, and CPU. It creates a new run under `fusion-smoke-runs/` and prints
its location. Preparation and training stop after 25 minutes; if the limit is
reached, the command fails and the last model may be incomplete. Installation
is separate and not included in the limit. The limit is a maximum, not a
target runtime, and smoke-test results are not a measure of model accuracy.

Use `--data-root /path/to/STS-nifti` for another dataset location, `--gpu-id 0`
for CUDA, or `--help` for other options. The four models are independent; this
command does not register or fuse their predictions.

For adjustable runs from **either** the repository root or `MNet/`, the
following examples use the repository's absolute macOS path. Replace it if
the repository is elsewhere:

```bash
# Short smoke test: two training patients, one validation patient per model.
"$HOME/Documents/BMEN600-G4-/venv/bin/python" "$HOME/Documents/BMEN600-G4-/scripts/smoke_fusion.py" --train-cases 2 --val-cases 1 --epochs 1 --time-limit 5

# Full split, one epoch per model, with a 30-minute maximum.
"$HOME/Documents/BMEN600-G4-/venv/bin/python" "$HOME/Documents/BMEN600-G4-/scripts/smoke_fusion.py" --train-cases all --val-cases all --epochs 1 --time-limit 30 --run-dir "$HOME/Documents/BMEN600-G4-/fusion-runs/full-01"
```

The run directory must not already exist. Change `full-01` for another run.
`--epochs`, `--train-cases`, `--val-cases`, `--val-interval` and `--time-limit`
are adjustable; `--train-cases all` uses the full *training split*, not the
held-out patients. One epoch is a limited full-data run, not a guarantee of
convergence. A time limit can interrupt any model, leaving later models
untrained or the current checkpoint incomplete; the command then exits with
an error. A complete four-model session cannot be guaranteed to fit 30
minutes on every machine. Installation is not included in the limit.

After training, first generate candidate registrations for the test patients:

```bash
"$HOME/Documents/BMEN600-G4-/venv/bin/python" "$HOME/Documents/BMEN600-G4-/scripts/fuse_sarcoma.py" --run_dir "$HOME/Documents/BMEN600-G4-/fusion-runs/full-01" --auto_register --output_dir "$HOME/Documents/BMEN600-G4-/fusion-runs/full-01/registration"
```

Inspect and correct the registrations before evaluating; automatic
registration is not reliable without review. After verification, evaluate the
weighted four-model fusion on held-out patients:

```bash
"$HOME/Documents/BMEN600-G4-/venv/bin/python" "$HOME/Documents/BMEN600-G4-/scripts/fuse_sarcoma.py" --run_dir "$HOME/Documents/BMEN600-G4-/fusion-runs/full-01" --transform_dir "$HOME/Documents/BMEN600-G4-/fusion-runs/full-01/registration/transforms" --evaluate
```

This prints per-patient and mean Dice and writes a uniquely named evaluation
directory inside the run directory. `--run_dir` defaults to the run's four
best checkpoints and its held-out test list; use `--data_root` if the dataset
is not in the repository's `STS-nifti/`. For native single-modality Dice
without registration, replace `--transform_dir ...` with `--modalities ct`
(or `t1`, `t2`, `pet`). Do not use the test set for model or weight selection.

### Full training

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

#### Resuming a modality model

Resume an epoch-final or best checkpoint with `train_sarcoma.py` (the
four-model `smoke_fusion.py` runner does not resume). Use the same modality,
architecture, preprocessing, optimizer parameter groups, training list and
validation list, and a **new** output prefix so the earlier run is preserved.
`--n_epochs` is the total target, not the number of additional epochs.
For example, to continue T2 from the end of `full-02` through 40 total
epochs, from either the repository root or `MNet/`:

```bash
"$HOME/Documents/BMEN600-G4-/venv/bin/python" "$HOME/Documents/BMEN600-G4-/scripts/train_sarcoma.py" --no_cuda --model_depth 18 --resnet_shortcut A --input_D 32 --input_H 160 --input_W 160 --batch_size 1 --num_workers 0 --n_epochs 40 --val_interval 1 --dropout_prob 0.2 --pretrain_path "$HOME/Documents/BMEN600-G4-/MNet/pretrain/resnet_18.pth" --consistent_preprocessing --modality_name t2 --data_root "$HOME/Documents/BMEN600-G4-/STS-nifti" --img_list "$HOME/Documents/BMEN600-G4-/fusion-runs/full-02/lists/smoke_train_t2.txt" --val_img_list "$HOME/Documents/BMEN600-G4-/fusion-runs/full-02/lists/smoke_val_t2.txt" --resume_path "$HOME/Documents/BMEN600-G4-/fusion-runs/full-02/models/t2_epoch_29_final.pth.tar" --output_prefix "$HOME/Documents/BMEN600-G4-/fusion-runs/t2-continued/models/t2"
```

The optimizer and scheduler resume from the checkpoint. Older epoch-end
checkpoints without scheduler state restore the exponential schedule from
the saved learning rates and completed epoch. When available, the original
best checkpoint beside the resumed checkpoint is copied and retained unless
validation improves; keep these files together. If no earlier best is
available, a warning says that best selection starts with resumed epochs.
Mid-epoch checkpoints are not supported. Random generator and data-loader
states are not restored, so continuation is not bit-for-bit reproducible.

For each test patient, align the native images without using tumor masks, then
resample predictions into the chosen reference image's grid. The preferred
input is a **visually verified** SimpleITK `.tfm` transform for each moving
image, named `<case>_<reference>_to_<moving>.tfm`, e.g.
`STS_001_ct_to_t1.tfm`. Despite the filename, the transform must map
**reference/fixed physical coordinates to moving physical coordinates**, as
required by SimpleITK resampling. Use `--transform_dir` to supply these files.

Alternatively, the following prepares *candidate* rigid mutual-information
and correlation transforms and registered images; it does not run inference
or evaluation:

```bash
python scripts/fuse_sarcoma.py \
  --data_root "$DATA_ROOT" --cases "$LISTS/test_cases.txt" \
  --output_dir "$ROOT/registration_candidates" \
  --modalities ct t1 t2 pet --reference ct --auto_register
```

For this STS collection, paired CT and PET share a physical coordinate system
and are not re-optimized. For MRI, each aligned MRI copy is compared to its
native scan to recover a transform from PET space to native MRI space; the
aligned image is not fed into a model. Recovery checks image correlation and
coverage and stops if they are too low. Cases missing the required aligned MRI
copy also stop and require manual transforms. The CT/PET assumption and
similarity checks are not a guarantee of correct alignment.
The supplied MRI references may be cropped around the lesion, so using them
for registration can reveal approximate lesion location and bias evaluation.
This workflow also requires those reference images at inference and is not
validated for native scans alone. Use a separate untouched external cohort
for final assessment after registration development.

**Inspect every registered image against the reference**, correct failed
registrations with suitable tools, and save verified transforms before fusion.
Different limb positions can leave anatomy misaligned outside the MRI reference
crop even when image-similarity checks pass; a single rigid transform cannot
correct those differences. Review multiple planes and the region being fused.
Use a new output directory; existing case outputs are not overwritten. Do not
use tumor masks to derive registration transforms.

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
