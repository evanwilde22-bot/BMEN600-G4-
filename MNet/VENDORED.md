# Vendored copy of Tencent MedicalNet

This directory is a vendored, unmodified copy of https://github.com/Tencent/MedicalNet
(commit 20f76aaab5cac8056eaf50b79ed97c09dbfbd3bd), vendored on 2026-10-09 for use in the
BMEN600 Group 4 final project. The original LICENSE and README.md are preserved.

## Local modifications

The only change from upstream is in `requirements.txt`: `numpy==1.15.4` -> `numpy>=1.22.0`
and `scipy==1.1.0` -> `scipy>=1.8.0`, to resolve known security advisories in the pinned versions
(NumPy NULL pointer dereference / unsafe deserialization, SciPy use-after-free).
