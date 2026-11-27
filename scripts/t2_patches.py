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
from scipy import ndimage
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


def robust_normalize(volume):
    volume = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError("Image contains non-finite intensities")
    foreground = volume != 0
    pixels = volume[foreground]
    if pixels.size < 2:
        raise ValueError("Image has insufficient nonzero intensities")
    lower, upper = np.percentile(pixels, (0.5, 99.5))
    clipped = np.clip(pixels, lower, upper)
    std = float(clipped.std())
    if std <= 0:
        raise ValueError("Image has no intensity variation after clipping")
    result = np.zeros_like(volume)
    result[foreground] = (clipped - clipped.mean()) / std
    return result


def prepare(image_path, mask_path, refined=False):
    image = nib.load(image_path)
    mask = nib.load(mask_path)
    if image.shape != mask.shape or not np.allclose(image.affine, mask.affine, atol=1e-3):
        raise ValueError(f"Image and mask grids differ: {image_path}")
    canonical = nib.as_closest_canonical(image)
    canonical = nib.Nifti1Image(np.asarray(canonical.dataobj, dtype=np.float32), canonical.affine)
    sampled = resample_to_output(canonical, voxel_sizes=SPACING, order=1)
    label = resample_from_to(mask, sampled, order=0)
    array = np.asarray(sampled.dataobj, dtype=np.float32)
    volume = robust_normalize(array) if refined else normalize_consistent_volume(array)
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


def sampling_locations(image, label):
    foreground = np.flatnonzero(label)
    boundary = ndimage.binary_dilation(label, iterations=4) & ~label & (image != 0)
    nonzero = image[image != 0]
    if not nonzero.size:
        raise ValueError("Image has no nonzero voxels for patch sampling")
    bright = (image >= np.percentile(nonzero, 80)) & ~label & (image != 0)
    locations = (foreground, np.flatnonzero(boundary), np.flatnonzero(bright))
    if any(not len(indices) for indices in locations):
        raise ValueError("Could not sample tumor, boundary and bright background patch centers")
    return locations


def augment_refined(image, rng):
    result = image.copy()
    foreground = result != 0
    values = result[foreground]
    gamma = rng.uniform(0.8, 1.2)
    values = np.sign(values) * np.abs(values) ** gamma
    values = values * rng.uniform(0.85, 1.15) + rng.uniform(-0.1, 0.1)
    values += rng.normal(0, 0.03, size=values.shape)
    result[foreground] = values
    return result


def starts(length, patch):
    if length <= patch:
        return [0]
    points = list(range(0, length - patch + 1, max(1, patch // 2)))
    if points[-1] != length - patch:
        points.append(length - patch)
    return points


def blending_weights(patch):
    axes = [np.exp(-0.5 * (np.linspace(-1, 1, size) / 0.5) ** 2) for size in patch]
    weights = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return np.maximum(weights, 0.01).astype(np.float32)


def predict(model, volume, patch, device, refined=False):
    image = padded(volume, patch)
    total = np.zeros(image.shape, dtype=np.float32)
    count = np.zeros(image.shape, dtype=np.float32)
    weights = blending_weights(patch) if refined else np.ones(patch, dtype=np.float32)
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
                    total[region] += probability * weights
                    count[region] += weights
    if np.any(count == 0):
        raise RuntimeError("Sliding-window inference did not cover the image")
    return (total / count)[tuple(slice(0, size) for size in volume.shape)]


def native_prediction(model, case, patch, device, refined=False):
    original, sampled, volume, truth = case
    probability = predict(model, volume, patch, device, refined).transpose(2, 1, 0)
    prob_image = nib.Nifti1Image(probability, sampled.affine)
    native = resample_from_to(prob_image, original, order=1)
    return np.asarray(native.dataobj, dtype=np.float32)


def evaluate(model, cases, patch, device, output=None, refined=False, threshold=0.5):
    scores = []
    for image_path, mask_path, case in cases:
        probability = native_prediction(model, case, patch, device, refined)
        prediction = probability >= threshold
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


class SkipDecoder(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        del backbone.conv_seg
        self.backbone = backbone
        self.bottleneck = torch.nn.Sequential(torch.nn.Conv3d(512, 64, 1), torch.nn.ReLU())
        self.decode_quarter = torch.nn.Sequential(
            torch.nn.Conv3d(128, 64, 3, padding=1), torch.nn.BatchNorm3d(64), torch.nn.ReLU())
        self.decode_half = torch.nn.Sequential(
            torch.nn.Conv3d(128, 32, 3, padding=1), torch.nn.BatchNorm3d(32), torch.nn.ReLU())
        self.classifier = torch.nn.Conv3d(32, 2, 1)

    def forward(self, image):
        b = self.backbone
        stem = b.relu(b.bn1(b.conv1(image)))
        skip = b.layer1(b.maxpool(stem))
        features = b.feature_dropout(b.layer4(b.layer3(b.layer2(skip))))
        features = self.bottleneck(features)
        features = F.interpolate(features, size=skip.shape[2:], mode="trilinear", align_corners=False)
        features = self.decode_quarter(torch.cat((features, skip), dim=1))
        features = F.interpolate(features, size=stem.shape[2:], mode="trilinear", align_corners=False)
        features = self.decode_half(torch.cat((features, stem), dim=1))
        return F.interpolate(self.classifier(features), size=image.shape[2:],
                             mode="trilinear", align_corners=False)


def model_for(patch, pretrained, refined=False):
    options = SimpleNamespace(model="resnet", model_depth=18, input_D=patch[0],
                              input_H=patch[1], input_W=patch[2], resnet_shortcut="A",
                              no_cuda=True, n_seg_classes=2, dropout_prob=0.2,
                              phase="train" if pretrained else "test",
                              pretrain_path=str(pretrained) if pretrained else "",
                              new_layer_names=["conv_seg"])
    model, parameters = generate_model(options)
    if not refined:
        return model, parameters
    model = SkipDecoder(model)
    return model, {"base_parameters": list(model.backbone.parameters()),
                   "new_parameters": list(model.bottleneck.parameters())
                   + list(model.decode_quarter.parameters())
                   + list(model.decode_half.parameters()) + list(model.classifier.parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--lists", type=Path, default=ROOT / "fusion-runs/full-02/lists")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patches-per-case", type=int, default=2)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(24, 128, 128),
                        metavar=("D", "H", "W"))
    parser.add_argument("--pretrained", type=Path, default=ROOT / "MNet/pretrain/resnet_18.pth")
    parser.add_argument("--evaluate", choices=("val", "test"))
    parser.add_argument("--refined", action="store_true",
                        help="Train the skip-decoder, robust-normalization experiment")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Inference threshold for evaluation (training selection uses 0.5)")
    args = parser.parse_args()
    if args.epochs < 1 or args.patches_per_case < 2 or args.val_interval < 1:
        parser.error("epochs and val-interval must be positive; patches-per-case must be at least 2")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if any(n < 8 or n % 8 for n in args.patch_size):
        parser.error("patch dimensions must be multiples of 8 and at least 8")
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    if not args.evaluate and args.threshold != 0.5:
        parser.error("--threshold is only configurable for evaluation")
    if not args.evaluate and args.refined and args.patches_per_case < 3:
        parser.error("the refined experiment requires at least 3 patches per case")
    torch.set_num_threads(2)
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    patch = tuple(args.patch_size)

    if args.evaluate:
        config = json.loads((args.run_dir / "run.json").read_text())
        args.data_root = args.data_root or Path(config["data_root"])
        lists = args.run_dir / "lists"
        checkpoint = torch.load(args.run_dir / "t2_best.pth.tar", map_location="cpu", weights_only=True)
        patch = tuple(checkpoint["patch_size"])
        if tuple(checkpoint["spacing"]) != SPACING:
            raise ValueError("Checkpoint spacing is incompatible with this workflow")
        refined = checkpoint.get("variant", "baseline") == "refined"
        if checkpoint.get("variant", "baseline") not in ("baseline", "refined"):
            raise ValueError("Checkpoint model variant is not supported")
        if args.refined and not refined:
            parser.error("--refined does not match the checkpoint")
        model, _ = model_for(patch, None, refined)
        model.load_state_dict(checkpoint["state_dict"])
        split = args.evaluate
    else:
        refined = args.refined
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
            "variant": "refined" if refined else "baseline",
            "seed": args.seed,
        }, indent=2) + "\n")
        model, parameters = model_for(patch, args.pretrained, refined)
        optimizer = torch.optim.SGD([
            {"params": parameters["base_parameters"], "lr": 0.001},
            {"params": parameters["new_parameters"], "lr": 0.01 if refined else 0.1},
        ], momentum=0.9, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        split = "val"

    val_cases = [(image, mask, prepare(image, mask, refined))
                 for image, mask in load_cases(args.data_root, lists / f"{split}_t2.txt")]
    if args.evaluate:
        import tempfile
        output = Path(tempfile.mkdtemp(prefix=f"evaluation-{split}-", dir=args.run_dir))
        score = evaluate(model, val_cases, patch, device, output, refined, args.threshold)
        (output / "run.json").write_text(json.dumps({
            "split": split, "threshold": args.threshold, "mean_dice": score,
            "checkpoint": str(args.run_dir / "t2_best.pth.tar"),
        }, indent=2) + "\n")
        print(f"Mean Dice: {score:.4f}\nPredictions: {output}", flush=True)
        return

    train_cases = [prepare(image, mask, refined)
                   for image, mask in load_cases(args.data_root, lists / "train_t2.txt")]
    locations = [sampling_locations(case[2], case[3]) for case in train_cases] if refined else None
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for index in rng.permutation(len(train_cases)):
            _, _, image, label = train_cases[index]
            foreground = np.argwhere(label) if not refined else None
            for sample in range(args.patches_per_case):
                if refined:
                    candidates = locations[index][sample % 3]
                    center = np.unravel_index(candidates[rng.integers(len(candidates))], image.shape)
                else:
                    center = (foreground[rng.integers(len(foreground))] if sample % 2 == 0
                              else image_center(image, rng))
                x = patch_at(image, center, patch)
                y = patch_at(label, center, patch)
                for axis in range(3):
                    if rng.random() < 0.5:
                        x, y = np.flip(x, axis), np.flip(y, axis)
                x = (augment_refined(x, rng) if refined else
                     (x * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)).copy())
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
            score = evaluate(model, val_cases, patch, device, refined=refined)
            print(f"Validation mean Dice: {score:.4f}", flush=True)
            if score > best:
                best = score
                torch.save({"state_dict": model.state_dict(), "patch_size": patch,
                            "spacing": SPACING, "epoch": epoch + 1, "val_dice": score,
                            "variant": "refined" if refined else "baseline"},
                           args.run_dir / "t2_best.pth.tar")
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "patch_size": patch,
                    "spacing": SPACING, "epoch": epoch + 1, "best_val_dice": best,
                    "variant": "refined" if refined else "baseline"},
                   args.run_dir / "t2_latest.pth.tar")
    print(f"Best validation Dice: {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
