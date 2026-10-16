# Vendored copy of Tencent MedicalNet

This directory is a vendored copy of https://github.com/Tencent/MedicalNet
(commit 20f76aaab5cac8056eaf50b79ed97c09dbfbd3bd), vendored on 2026-10-09 for use in the
BMEN600 Group 4 final project. The original LICENSE and README.md are preserved.

## Local modifications

- `requirements.txt`: `numpy==1.15.4` -> `numpy>=1.22.0`, `scipy==1.1.0` -> `scipy>=1.8.0`, and
  `torch>=2.6.0`, to resolve known security advisories in the pinned versions (NumPy NULL pointer
  dereference / unsafe deserialization, SciPy use-after-free) and support safe checkpoint loading.
- `model.py`: pre-trained weights are loaded with `torch.load(..., weights_only=True, map_location='cpu')`
  and only when `--phase` is not `test`.
- `datasets/sarcoma.py` (new file): dataset loader for the Soft-tissue-Sarcoma (STS) NIfTI data. Remaps
  mask foreground (255) to class 1; training preprocessing matches testing (no tumor-centered crop) with
  random flip + intensity jitter augmentation.

Project-specific training/evaluation entry points live outside this directory in
`scripts/train_sarcoma.py` and `scripts/test_sarcoma.py`. The upstream `train.py`, `test.py`, and
`datasets/brains18.py` are unchanged.
