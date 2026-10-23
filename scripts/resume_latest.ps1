<#
.SYNOPSIS
Selects a checkpoint to pass to train_sarcoma.py --resume_path.

.DESCRIPTION
Looks in MNet/trails/models/ for:
  1. sarcoma_resnet_<depth>_best.pth.tar (best validation Dice; preferred)
  2. the most recently modified sarcoma_resnet_<depth>_epoch_*_final.pth.tar
  3. otherwise returns an empty string (training starts from --pretrain_path)
Prints the selection and returns the path.

.EXAMPLE
$resume = & ./scripts/resume_latest.ps1 -ModelDepth 18
#>
param(
    [int]$ModelDepth = 18
)

$modelsDir = Join-Path (Join-Path $PSScriptRoot "..") "MNet/trails/models"
$best = Join-Path $modelsDir "sarcoma_resnet_${ModelDepth}_best.pth.tar"

$selected = ""
if (Test-Path $best) {
    $selected = (Resolve-Path $best).Path
    Write-Host "Resuming from best checkpoint: $selected"
} else {
    $latest = Get-ChildItem -Path $modelsDir -Filter "sarcoma_resnet_${ModelDepth}_epoch_*_final.pth.tar" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) {
        $selected = $latest.FullName
        Write-Host "No best checkpoint found; resuming from latest epoch checkpoint: $selected"
    } else {
        Write-Host "No checkpoint found; training will start from --pretrain_path."
    }
}

return $selected
