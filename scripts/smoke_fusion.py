"""Run a time-limited, four-modality training smoke test."""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from generate_fusion_lists import MODALITIES


ROOT = Path(__file__).resolve().parents[1]
MNET = ROOT / "MNet"


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def run_command(command, *, cwd, deadline, env):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Smoke test time limit reached before the next step.")
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True, timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            "Smoke test time limit reached. The current model may be incomplete."
        ) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "STS-nifti",
                        help="STS patient directory (default: repository's STS-nifti)")
    parser.add_argument("--pretrained", type=Path, default=MNET / "pretrain" / "resnet_18.pth",
                        help="MedicalNet ResNet-18 pretrained checkpoint")
    parser.add_argument("--output-root", type=Path, default=ROOT / "fusion-smoke-runs",
                        help="Parent for a new unique run directory")
    parser.add_argument("--train-cases", type=positive_int, default=10)
    parser.add_argument("--val-cases", type=positive_int, default=3)
    parser.add_argument("--time-limit", type=positive_int, default=25,
                        help="Maximum preparation and training time, in minutes (default: 25)")
    parser.add_argument("--gpu-id", type=int, help="CUDA GPU index; omit for CPU")
    args = parser.parse_args(argv)
    data_root = args.data_root.resolve()
    pretrained = args.pretrained.resolve()
    if not data_root.is_dir():
        parser.error(f"data directory does not exist: {data_root}")
    if not pretrained.is_file():
        parser.error(f"pretrained checkpoint does not exist: {pretrained}")
    if args.gpu_id is not None and args.gpu_id < 0:
        parser.error("--gpu-id must be nonnegative")

    deadline = time.monotonic() + args.time_limit * 60
    args.output_root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_root.resolve()))
    lists = run / "lists"
    print(f"Smoke test output: {run}", flush=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    generator = [sys.executable, str(ROOT / "scripts" / "generate_fusion_lists.py"),
                 "--data_root", str(data_root), "--output_dir", str(lists)]

    try:
        run_command(generator, cwd=ROOT, deadline=deadline, env=env)
        for split, count in (("train", args.train_cases), ("val", args.val_cases)):
            for modality in MODALITIES:
                source = lists / f"{split}_{modality}.txt"
                lines = source.read_text().splitlines()
                if len(lines) < count:
                    raise ValueError(f"{source} has {len(lines)} cases; requested {count}")
                (lists / f"smoke_{split}_{modality}.txt").write_text(
                    "".join(f"{line}\n" for line in lines[:count]))

        for modality in MODALITIES:
            print(f"\nTraining {modality} model (one epoch)", flush=True)
            command = [
                sys.executable, str(ROOT / "scripts" / "train_sarcoma.py"),
                "--model_depth", "18", "--resnet_shortcut", "A",
                "--input_D", "32", "--input_H", "160", "--input_W", "160",
                "--batch_size", "1", "--num_workers", "0", "--n_epochs", "1",
                "--dropout_prob", "0.2", "--pretrain_path", str(pretrained),
                "--consistent_preprocessing", "--modality_name", modality,
                "--data_root", str(data_root),
                "--img_list", str(lists / f"smoke_train_{modality}.txt"),
                "--val_img_list", str(lists / f"smoke_val_{modality}.txt"),
                "--output_prefix", str(run / "models" / modality),
            ]
            command.extend(["--no_cuda"] if args.gpu_id is None else ["--gpu_id", str(args.gpu_id)])
            run_command(command, cwd=MNET, deadline=deadline, env=env)
    except (TimeoutError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Smoke test failed: {exc}\nPartial output: {run}", file=sys.stderr, flush=True)
        return 1

    print(f"\nSmoke test completed. Checkpoints and lists: {run}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
