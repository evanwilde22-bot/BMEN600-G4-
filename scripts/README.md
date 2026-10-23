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
| `--modality` | `RTstruct_T1` | Modality subfolder (`RTstruct_T1`, `RTstruct_T2FS`, `RTstruct_CT`, `RTstruct_PET`, `RTstruct_AlignedT1toPET`, `RTstruct_AlignedT2FStoPET`) |
| `--image_file` | `image.nii.gz` | Image filename within the modality folder |
| `--mask` | `mask_GTV_Mass.nii.gz` | Mask filename within the modality folder (or `mask_GTV_Edema.nii.gz`) |
| `--val_fraction` | `0.2` | Fraction of cases reserved for validation |
| `--seed` | `1` | Random seed for the shuffle/split |
| `--case_prefix` | `STS_` | Prefix used to identify case folders |

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
