r"""
Generates train.txt / val.txt file lists for MNet training on the
Soft-tissue-Sarcoma (STS) NIfTI dataset.

Expected directory layout (per case):
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

Usage:
    python generate_file_lists.py --data_root "C:\path\to\nifti" --modality RTstruct_T1 --mask mask_GTV_Mass.nii.gz

    # T1, T2FS and both PET-aligned volumes in one trial:
    python generate_file_lists.py --data_root "C:\\path\\to\\nifti" --mask mask_GTV_Mass.nii.gz \\
        --modality RTstruct_T1 RTstruct_T2FS RTstruct_AlignedT1toPET RTstruct_AlignedT2FStoPET
"""

import os
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate train/val file lists for STS NIfTI data")
    parser.add_argument("--data_root", type=str, required=True,
                         help="Root directory containing STS_XXX case folders")
    parser.add_argument("--modality", type=str, nargs="+", default=["RTstruct_T1"],
                         help="One or more modality subfolder names (e.g. RTstruct_T1 RTstruct_T2FS "
                              "RTstruct_AlignedT1toPET RTstruct_AlignedT2FStoPET). With several, every "
                              "available modality of a case becomes a training sample, and cases are "
                              "split train/val as a unit (no leakage across modalities).")
    parser.add_argument("--image_file", type=str, default="image.nii.gz",
                         help="Image filename within each modality folder")
    parser.add_argument("--mask", type=str, default="mask_GTV_Mass.nii.gz",
                         help="Mask filename within each modality folder "
                              "(e.g. mask_GTV_Mass.nii.gz or mask_GTV_Edema.nii.gz)")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                         help="Fraction of cases reserved for validation")
    parser.add_argument("--seed", type=int, default=1,
                         help="Random seed for shuffling/splitting")
    parser.add_argument("--case_prefix", type=str, default="STS_",
                         help="Prefix used to identify case folders")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    cases = sorted([d for d in os.listdir(args.data_root)
                     if d.startswith(args.case_prefix) and os.path.isdir(os.path.join(args.data_root, d))])

    case_pairs = {}
    skipped = []

    for case in cases:
        found = []
        for modality in args.modality:
            img_rel = os.path.join(case, modality, args.image_file)
            mask_rel = os.path.join(case, modality, args.mask)
            if (os.path.isfile(os.path.join(args.data_root, img_rel))
                    and os.path.isfile(os.path.join(args.data_root, mask_rel))):
                # normalize to forward slashes for cross-platform consistency
                found.append((img_rel.replace("\\", "/"), mask_rel.replace("\\", "/")))
            else:
                skipped.append(f"{case}/{modality}")
        if found:
            case_pairs[case] = found

    if skipped:
        print(f"Skipped {len(skipped)} missing {args.image_file} or {args.mask}:")
        for s in skipped:
            print(f"  - {s}")

    if not case_pairs:
        print("No valid image/mask pairs found. Check --data_root, --modality, and --mask arguments.")
        return

    # split by case so all modalities of one patient land on the same side
    case_ids = list(case_pairs)
    random.shuffle(case_ids)
    n_val = max(1, int(len(case_ids) * args.val_fraction))
    val_pairs = [p for c in case_ids[:n_val] for p in case_pairs[c]]
    train_pairs = [p for c in case_ids[n_val:] for p in case_pairs[c]]
    random.shuffle(train_pairs)
    pairs = val_pairs + train_pairs

    train_path = os.path.join(args.data_root, "train.txt")
    val_path = os.path.join(args.data_root, "val.txt")

    with open(train_path, "w") as f:
        for img, mask in train_pairs:
            f.write(f"{img} {mask}\n")

    with open(val_path, "w") as f:
        for img, mask in val_pairs:
            f.write(f"{img} {mask}\n")

    print(f"\nCases: {len(case_ids)} | Total samples: {len(pairs)} | Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    print(f"Wrote {train_path}")
    print(f"Wrote {val_path}")


if __name__ == "__main__":
    main()
