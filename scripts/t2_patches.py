"""Train or evaluate a T2/STIR patch model on a fixed patient split."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MNet"))
from datasets.sarcoma import normalize_consistent_volume
from model import generate_model
from train_sarcoma import dice_loss

SPACING = (1.5, 1.5, 7.0)


def load_cases(data_root, list_path):
    cases = []
    for line in list_path.read_text().splitlines():
        image, mask = line.split()
        cases.append((data_root / image, data_root / mask))
    if not cases:
        raise ValueError(f"No cases in {list_path}")
    ids = [image.parent.parent.name for image, _ in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate patients in {list_path}")
    if any(image.parent != mask.parent or image.parent.name not in ("RTstruct_T2FS", "RTstruct_STIR")
           for image, mask in cases):
        raise ValueError(f"Expected matching native T2FS/STIR image and mask paths in {list_path}")
    return cases


def prepare(image_path, mask_path):
    image = nib.load(image_path)
    mask = nib.load(mask_path)
    if image.shape != mask.shape or not np.allclose(image.affine, mask.affine, atol=1e-3):
        raise ValueError(f"Image and mask grids differ: {image_path}")
    canonical = nib.as_closest_canonical(image)
    canonical = nib.Nifti1Image(np.asarray(canonical.dataobj, dtype=np.float32), canonical.affine)
    sampled = resample_to_output(canonical, voxel_sizes=SPACING, order=1)
    label = resample_from_to(mask, sampled, order=0)
    volume = normalize_consistent_volume(np.asarray(sampled.dataobj, dtype=np.float32))
    truth = np.asarray(label.dataobj) > 0
    if not truth.any():
        raise ValueError(f"Mask has no foreground after resampling: {mask_path}")
    return image, sampled, volume.transpose(2, 1, 0), truth.transpose(2, 1, 0)


def padded(array, patch):
    return np.pad(array, [(0, max(0, size - dim)) for dim, size in zip(array.shape, patch)])


def patch_at(array, center, patch):
    array = padded(array, patch)
    starts = [min(max(0, int(c) - size // 2), dim - size)
              for c, size, dim in zip(center, patch, array.shape)]
    return array[tuple(slice(start, start + size) for start, size in zip(starts, patch))]


def image_center(image, rng):
    for _ in range(100):
        center = tuple(rng.integers(size) for size in image.shape)
        if image[center] != 0:
            return center
    locations = np.flatnonzero(image)
    if not locations.size:
        raise ValueError("Image has no nonzero voxels for patch sampling")
    return np.unravel_index(locations[rng.integers(len(locations))], image.shape)


def starts(length, patch):
    if length <= patch:
        return [0]
    points = list(range(0, length - patch + 1, max(1, patch // 2)))
    if points[-1] != length - patch:
        points.append(length - patch)
    return points


def predict(model, volume, patch, device):
    image = padded(volume, patch)
    total = np.zeros(image.shape, dtype=np.float32)
    count = np.zeros(image.shape, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for z in starts(image.shape[0], patch[0]):
            for y in starts(image.shape[1], patch[1]):
                for x in starts(image.shape[2], patch[2]):
                    region = (slice(z, z + patch[0]), slice(y, y + patch[1]),
                              slice(x, x + patch[2]))
                    tensor = torch.from_numpy(image[region].copy()[None, None]).to(device)
                    logits = F.interpolate(model(tensor), size=patch, mode="trilinear",
                                           align_corners=False)
                    probability = F.softmax(logits, dim=1)[0, 1].cpu().numpy()
                    total[region] += probability
                    count[region] += 1
    if np.any(count == 0):
        raise RuntimeError("Sliding-window inference did not cover the image")
    return (total / count)[tuple(slice(0, size) for size in volume.shape)]


def native_prediction(model, case, patch, device):
    original, sampled, volume, truth = case
    probability = predict(model, volume, patch, device).transpose(2, 1, 0)
    prob_image = nib.Nifti1Image(probability, sampled.affine)
    native = resample_from_to(prob_image, original, order=1)
    return np.asarray(native.dataobj, dtype=np.float32)


def evaluate(model, cases, patch, device, output=None):
    scores = []
    for image_path, mask_path, case in cases:
        probability = native_prediction(model, case, patch, device)
        prediction = probability >= 0.5
        reference = np.asarray(nib.load(mask_path).dataobj) > 0
        total = prediction.sum() + reference.sum()
        score = float(2 * np.logical_and(prediction, reference).sum() / total) if total else 1.0
        name = image_path.parent.parent.name
        print(f"{name}: Dice = {score:.4f}", flush=True)
        scores.append(score)
        if output is not None:
            target = output / name
            target.mkdir(parents=True, exist_ok=False)
            original = case[0]
            nib.save(nib.Nifti1Image(probability, original.affine),
                     target / "t2_probability.nii.gz")
            nib.save(nib.Nifti1Image(prediction.astype(np.uint8), original.affine),
                     target / "t2_mask.nii.gz")
    return float(np.mean(scores))


def model_for(patch, pretrained):
    options = SimpleNamespace(model="resnet", model_depth=18, input_D=patch[0],
                              input_H=patch[1], input_W=patch[2], resnet_shortcut="A",
                              no_cuda=True, n_seg_classes=2, dropout_prob=0.2,
                              phase="train" if pretrained else "test",
                              pretrain_path=str(pretrained) if pretrained else "",
                              new_layer_names=["conv_seg"])
    return generate_model(options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--lists", type=Path, default=ROOT / "fusion-runs/full-02/lists")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patches-per-case", type=int, default=2)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(24, 128, 128),
                        metavar=("D", "H", "W"))
    parser.add_argument("--pretrained", type=Path, default=ROOT / "MNet/pretrain/resnet_18.pth")
    parser.add_argument("--evaluate", choices=("val", "test"))
    args = parser.parse_args()
    if args.epochs < 1 or args.patches_per_case < 2 or args.val_interval < 1:
        parser.error("epochs and val-interval must be positive; patches-per-case must be at least 2")
    if any(n < 8 or n % 8 for n in args.patch_size):
        parser.error("patch dimensions must be multiples of 8 and at least 8")
    torch.set_num_threads(2)
    device = torch.device("cpu")
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    patch = tuple(args.patch_size)

    if args.evaluate:
        config = json.loads((args.run_dir / "run.json").read_text())
        args.data_root = args.data_root or Path(config["data_root"])
        lists = args.run_dir / "lists"
        checkpoint = torch.load(args.run_dir / "t2_best.pth.tar", map_location="cpu", weights_only=True)
        patch = tuple(checkpoint["patch_size"])
        if tuple(checkpoint["spacing"]) != SPACING:
            raise ValueError("Checkpoint spacing is incompatible with this workflow")
        model, _ = model_for(patch, None)
        model.load_state_dict(checkpoint["state_dict"])
        split = args.evaluate
    else:
        args.data_root = args.data_root or ROOT / "STS-nifti"
        splits = {name: load_cases(args.data_root, args.lists / f"{name}_t2.txt")
                  for name in ("train", "val", "test")}
        patient_sets = [{image.parent.parent.name for image, _ in cases}
                        for cases in splits.values()]
        if any(patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Train, validation and test patient lists must not overlap")
        args.run_dir.mkdir(parents=True, exist_ok=False)
        lists = args.run_dir / "lists"
        lists.mkdir()
        for name in splits:
            shutil.copyfile(args.lists / f"{name}_t2.txt", lists / f"{name}_t2.txt")
        (args.run_dir / "run.json").write_text(json.dumps({
            "data_root": str(args.data_root.resolve()), "epochs": args.epochs,
            "patch_size": patch, "spacing": SPACING, "patches_per_case": args.patches_per_case,
        }, indent=2) + "\n")
        model, parameters = model_for(patch, args.pretrained)
        optimizer = torch.optim.SGD([
            {"params": parameters["base_parameters"], "lr": 0.001},
            {"params": parameters["new_parameters"], "lr": 0.1},
        ], momentum=0.9, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        split = "val"

    val_cases = [(image, mask, prepare(image, mask))
                 for image, mask in load_cases(args.data_root, lists / f"{split}_t2.txt")]
    if args.evaluate:
        import tempfile
        output = Path(tempfile.mkdtemp(prefix=f"evaluation-{split}-", dir=args.run_dir))
        score = evaluate(model, val_cases, patch, device, output)
        print(f"Mean Dice: {score:.4f}\nPredictions: {output}", flush=True)
        return

    train_cases = [prepare(image, mask)
                   for image, mask in load_cases(args.data_root, lists / "train_t2.txt")]
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for index in rng.permutation(len(train_cases)):
            _, _, image, label = train_cases[index]
            foreground = np.argwhere(label)
            for sample in range(args.patches_per_case):
                center = (foreground[rng.integers(len(foreground))] if sample % 2 == 0
                          else image_center(image, rng))
                x = patch_at(image, center, patch)
                y = patch_at(label, center, patch)
                for axis in range(3):
                    if rng.random() < 0.5:
                        x, y = np.flip(x, axis), np.flip(y, axis)
                x = (x * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)).copy()
                tensor = torch.from_numpy(x[None, None].astype(np.float32)).to(device)
                target = torch.from_numpy(y.copy()[None].astype(np.int64)).to(device)
                optimizer.zero_grad()
                logits = F.interpolate(model(tensor), size=patch, mode="trilinear",
                                       align_corners=False)
                loss = F.cross_entropy(logits, target) + dice_loss(logits, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        scheduler.step()
        print(f"Epoch {epoch + 1}/{args.epochs}: mean loss {np.mean(losses):.4f}", flush=True)
        if (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.epochs:
            score = evaluate(model, val_cases, patch, device)
            print(f"Validation mean Dice: {score:.4f}", flush=True)
            if score > best:
                best = score
                torch.save({"state_dict": model.state_dict(), "patch_size": patch,
                            "spacing": SPACING, "epoch": epoch + 1, "val_dice": score},
                           args.run_dir / "t2_best.pth.tar")
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "patch_size": patch,
                    "spacing": SPACING, "epoch": epoch + 1, "best_val_dice": best},
                   args.run_dir / "t2_latest.pth.tar")
    print(f"Best validation Dice: {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
