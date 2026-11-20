"""Create a shared patient split and single-modality lists for late fusion."""

import argparse
import random
from pathlib import Path


MODALITIES = {
    "ct": ("RTstruct_CT",),
    "t1": ("RTstruct_T1",),
    "t2": ("RTstruct_T2FS", "RTstruct_STIR"),
    "pet": ("RTstruct_PET",),
}


def modality_folder(case, modality):
    found = [case / name for name in MODALITIES[modality] if (case / name / "image.nii.gz").is_file()]
    if len(found) > 1:
        raise ValueError(f"{case.name}: multiple {modality} images; select one explicitly")
    return found[0] if found else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mask", default="mask_GTV_Mass.nii.gz")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if not (0 < args.val_fraction < 1 and 0 < args.test_fraction < 1
            and args.val_fraction + args.test_fraction < 1):
        parser.error("validation and test fractions must be positive and sum to less than 1")
    cases = {}
    for case in sorted(args.data_root.glob("STS_*")):
        if not case.is_dir():
            continue
        folders = {name: modality_folder(case, name) for name in MODALITIES}
        missing = [name for name, folder in folders.items()
                   if folder is None or not (folder / args.mask).is_file()]
        if missing:
            print(f"Skipping {case.name}: missing image or mask for {', '.join(missing)}")
            continue
        cases[case.name] = folders
    if len(cases) < 3:
        raise ValueError("At least three complete cases with all four image/mask pairs are required")

    ids = list(cases)
    random.Random(args.seed).shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_fraction))
    n_test = max(1, int(len(ids) * args.test_fraction))
    if n_val + n_test >= len(ids):
        raise ValueError("Split fractions leave no training patients")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_ids in (("train", ids[n_val + n_test:]),
                             ("val", ids[:n_val]),
                             ("test", ids[n_val:n_val + n_test])):
        (args.output_dir / f"{split}_cases.txt").write_text(
            "".join(f"{case_id}\n" for case_id in split_ids))
        for modality in MODALITIES:
            lines = []
            for case_id in split_ids:
                folder = cases[case_id][modality]
                image = (folder / "image.nii.gz").relative_to(args.data_root).as_posix()
                mask = (folder / args.mask).relative_to(args.data_root).as_posix()
                if any(c.isspace() for c in image + mask):
                    raise ValueError(f"Whitespace in image/mask path: {image}, {mask}")
                lines.append(f"{image} {mask}\n")
            (args.output_dir / f"{split}_{modality}.txt").write_text("".join(lines))
        print(f"{split}: {len(split_ids)} patients")


if __name__ == "__main__":
    main()
