"""Run independent modality models and fuse their registered probabilities."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy import ndimage

MNET_DIR = Path(__file__).resolve().parents[1] / "MNet"
sys.path.insert(0, str(MNET_DIR))

from datasets.sarcoma import normalize_consistent_volume, resize_volume
from model import generate_model
from generate_fusion_lists import MODALITIES, modality_folder


def register_images(fixed, moving):
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1, seed=1)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration.SetInitialTransform(initial, inPlace=False)
    transform = registration.Execute(fixed, moving)
    print(f"Registration metric: {registration.GetMetricValue():.6g}; "
          f"{registration.GetOptimizerStopConditionDescription()}")
    return transform


def resample(image, reference, transform, interpolator, default=0.0):
    return sitk.Resample(image, reference, transform, interpolator, default, sitk.sitkFloat32)


def load_model(path, modality, args, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "consistent": True,
        "input_size": list(args.input_size),
        "model_depth": args.model_depth,
        "resnet_shortcut": args.resnet_shortcut,
        "modality": modality,
    }
    if checkpoint.get("preprocessing") != expected:
        raise ValueError(f"{path}: checkpoint preprocessing/architecture metadata does not match {expected}")
    opts = SimpleNamespace(model="resnet", model_depth=args.model_depth,
                           input_D=args.input_size[0], input_H=args.input_size[1],
                           input_W=args.input_size[2], resnet_shortcut=args.resnet_shortcut,
                           no_cuda=True, n_seg_classes=2, dropout_prob=0.0,
                           phase="test", pretrain_path="")
    model, _ = generate_model(opts)
    state = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def predict(model, image_path, input_size, device):
    image = nib.load(str(image_path))
    array = np.asanyarray(image.dataobj)
    volume = normalize_consistent_volume(resize_volume(array, input_size))
    tensor = torch.from_numpy(volume[None, None]).to(device)
    with torch.no_grad():
        probability = F.softmax(model(tensor), dim=1)[0, 1].cpu().numpy()
    scale = [target / actual for target, actual in zip(array.shape, probability.shape)]
    probability = ndimage.zoom(probability, scale, order=1).astype(np.float32)
    if probability.shape != array.shape:
        raise ValueError(f"Prediction grid mismatch for {image_path}")
    result = sitk.GetImageFromArray(probability.transpose(2, 1, 0))
    source = sitk.ReadImage(str(image_path))
    result.CopyInformation(source)
    return source, result


def dice(pred, truth):
    a = np.asarray(pred, dtype=bool)
    b = np.asarray(truth, dtype=bool)
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else float(2 * np.logical_and(a, b).sum() / total)


def fuse(predictions, shape):
    numerator = np.zeros(shape, dtype=np.float32)
    denominator = np.zeros(shape, dtype=np.float32)
    for probability, coverage, weight in predictions:
        numerator += probability * coverage * weight
        denominator += coverage * weight
    result = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True, help="One patient ID per line")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", choices=list(MODALITIES), required=True)
    for name in MODALITIES:
        parser.add_argument(f"--{name}_checkpoint", type=Path)
        parser.add_argument(f"--{name}_weight", type=float, default=1.0)
    parser.add_argument("--reference", choices=list(MODALITIES),
                        help="Output grid; defaults to the first selected modality")
    parser.add_argument("--transform_dir", type=Path,
                        help="Directory of fixed-to-moving SimpleITK .tfm transforms")
    parser.add_argument("--auto_register", action="store_true",
                        help="Prepare candidate transforms and registered images for inspection; do not fuse")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--mask", default="mask_GTV_Mass.nii.gz")
    parser.add_argument("--input_size", type=int, nargs=3, default=(32, 160, 160))
    parser.add_argument("--model_depth", type=int, default=18)
    parser.add_argument("--resnet_shortcut", choices=["A", "B"], default="A")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if len(set(args.modalities)) != len(args.modalities):
        parser.error("modalities must not be repeated")
    if args.reference is None:
        args.reference = args.modalities[0]
    if args.reference not in args.modalities:
        parser.error("reference must be one of the selected modalities")
    if len(args.modalities) > 1 and not (args.transform_dir or args.auto_register):
        parser.error("fusion requires --transform_dir or --auto_register")
    if args.transform_dir and args.auto_register:
        parser.error("choose either --transform_dir or --auto_register")
    if args.auto_register and (len(args.modalities) < 2 or args.evaluate):
        parser.error("--auto_register requires multiple modalities and cannot be used with --evaluate")
    if not 0 <= args.threshold <= 1 or any(n <= 0 for n in args.input_size):
        parser.error("threshold must be in [0,1] and input dimensions must be positive")
    for name in args.modalities:
        if not args.auto_register and getattr(args, f"{name}_checkpoint") is None:
            parser.error(f"--{name}_checkpoint is required")
        weight = getattr(args, f"{name}_weight")
        if not np.isfinite(weight) or weight <= 0:
            parser.error(f"--{name}_weight must be finite and positive")
    case_ids = [line.strip() for line in args.cases.read_text().splitlines() if line.strip()]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        parser.error("case list must be nonempty and contain unique patient IDs")
    if any(Path(case_id).name != case_id or not case_id.startswith("STS_") for case_id in case_ids):
        parser.error("case list must contain STS_ patient directory names only")

    if args.auto_register:
        transform_dir = args.output_dir / "transforms"
        transform_dir.mkdir(parents=True, exist_ok=True)
        for case_id in case_ids:
            case = args.data_root / case_id
            folders = {name: modality_folder(case, name) for name in args.modalities}
            missing = [name for name, folder in folders.items() if folder is None]
            if missing:
                raise FileNotFoundError(f"{case_id}: missing images for {', '.join(missing)}")
            case_out = args.output_dir / case_id
            case_out.mkdir(parents=True, exist_ok=True)
            reference = sitk.ReadImage(str(folders[args.reference] / "image.nii.gz"))
            sitk.WriteImage(reference, str(case_out / f"{args.reference}_reference.nii.gz"))
            for name in args.modalities:
                if name == args.reference:
                    continue
                image = sitk.ReadImage(str(folders[name] / "image.nii.gz"))
                print(f"{case_id}: registering {name} to {args.reference}")
                transform = register_images(reference, image)
                sitk.WriteTransform(transform, str(transform_dir / f"{case_id}_{args.reference}_to_{name}.tfm"))
                aligned = resample(image, reference, transform, sitk.sitkLinear)
                sitk.WriteImage(aligned, str(case_out / f"{name}_registered.nii.gz"))
        print(f"Candidate transforms saved to {transform_dir}. Inspect every registration before fusion.")
        return

    device = torch.device(args.device)
    models = {name: load_model(getattr(args, f"{name}_checkpoint"), name, args, device)
              for name in args.modalities}
    print("Inspect registered images before interpreting fused predictions; rigid registration is not guaranteed.")
    scores = []
    for case_id in case_ids:
        case = args.data_root / case_id
        folders = {name: modality_folder(case, name) for name in args.modalities}
        missing = [name for name, folder in folders.items() if folder is None]
        if missing:
            raise FileNotFoundError(f"{case_id}: missing images for {', '.join(missing)}")
        case_out = args.output_dir / case_id
        case_out.mkdir(parents=True, exist_ok=True)
        reference = sitk.ReadImage(str(folders[args.reference] / "image.nii.gz"))
        weights = [getattr(args, f"{name}_weight") for name in args.modalities]

        def aligned_predictions():
            for name, weight in zip(args.modalities, weights):
                image, probability = predict(models[name], folders[name] / "image.nii.gz",
                                             args.input_size, device)
                sitk.WriteImage(probability, str(case_out / f"{name}_probability_native.nii.gz"))
                if name == args.reference:
                    transform = sitk.Transform(3, sitk.sitkIdentity)
                else:
                    transform_path = args.transform_dir / f"{case_id}_{args.reference}_to_{name}.tfm"
                    if not transform_path.is_file():
                        raise FileNotFoundError(f"Missing fixed-to-moving transform: {transform_path}")
                    transform = sitk.ReadTransform(str(transform_path))
                    sitk.WriteTransform(transform, str(case_out / f"{args.reference}_to_{name}.tfm"))
                    aligned_image = resample(image, reference, transform, sitk.sitkLinear)
                    sitk.WriteImage(aligned_image, str(case_out / f"{name}_registered.nii.gz"))
                aligned = resample(probability, reference, transform, sitk.sitkLinear)
                coverage = sitk.Image(image.GetSize(), sitk.sitkFloat32) + 1
                coverage.CopyInformation(image)
                coverage = resample(coverage, reference, transform, sitk.sitkNearestNeighbor)
                coverage_array = sitk.GetArrayFromImage(coverage)
                if not np.any(coverage_array):
                    raise ValueError(f"{case_id}: {name} has no overlap with {args.reference}")
                sitk.WriteImage(aligned, str(case_out / f"{name}_probability_registered.nii.gz"))
                yield sitk.GetArrayFromImage(aligned), coverage_array, weight

        combined = fuse(aligned_predictions(), tuple(reversed(reference.GetSize())))
        probability_image = sitk.GetImageFromArray(combined)
        probability_image.CopyInformation(reference)
        sitk.WriteImage(probability_image, str(case_out / "fused_probability.nii.gz"))
        binary = sitk.Cast(probability_image >= args.threshold, sitk.sitkUInt8)
        sitk.WriteImage(binary, str(case_out / "fused_mask.nii.gz"))
        if args.evaluate:
            mask = sitk.ReadImage(str(folders[args.reference] / args.mask))
            if (mask.GetSize() != reference.GetSize()
                    or not np.allclose(mask.GetOrigin(), reference.GetOrigin(), atol=1e-3)
                    or not np.allclose(mask.GetSpacing(), reference.GetSpacing(), atol=1e-3)
                    or not np.allclose(mask.GetDirection(), reference.GetDirection(), atol=1e-3)):
                raise ValueError(f"{case_id}: reference mask and image grids differ")
            score = dice(sitk.GetArrayFromImage(binary), sitk.GetArrayFromImage(mask) > 0)
            scores.append(score)
            print(f"{case_id}: fused Dice = {score:.4f}")
    if scores:
        print(f"Mean Dice: {np.mean(scores):.4f} over {len(scores)} cases")
    (args.output_dir / "run.json").write_text(json.dumps({
        "cases": case_ids, "modalities": args.modalities, "reference": args.reference,
        "weights": dict(zip(args.modalities, weights)), "threshold": args.threshold,
        "dice": dict(zip(case_ids, scores)) if scores else None,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
