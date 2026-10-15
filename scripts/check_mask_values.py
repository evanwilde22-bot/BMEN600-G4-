"""
Quick sanity-check utility: inspect unique values, dtype, and shape of a
NIfTI mask file to confirm it is a binary (or expected multi-class)
segmentation label volume before training.

Usage:
    python check_mask_values.py --path "C:\path\to\STS_001\RTstruct_T1\mask_GTV_Mass.nii.gz"
"""

import argparse
import nibabel as nib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Check NIfTI mask values")
    parser.add_argument("--path", type=str, required=True, help="Path to a NIfTI mask file")
    return parser.parse_args()


def main():
    args = parse_args()
    mask = nib.load(args.path)
    data = np.asanyarray(mask.dataobj)

    print(f"File: {args.path}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Unique values: {np.unique(data)}")
    print(f"Voxel count per label:")
    values, counts = np.unique(data, return_counts=True)
    for v, c in zip(values, counts):
        print(f"  {v}: {c}")


if __name__ == "__main__":
    main()
