"""Run independent modality models and fuse their registered probabilities."""

import argparse
import itertools
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy import ndimage, signal

MNET_DIR = Path(__file__).resolve().parents[1] / "MNet"
sys.path.insert(0, str(MNET_DIR))

from datasets.sarcoma import normalize_consistent_volume, resize_volume
from model import generate_model
from generate_fusion_lists import MODALITIES, modality_folder


def registration_similarity(fixed, moving, transform):
    warped = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0, sitk.sitkFloat32)
    fixed_array = sitk.GetArrayViewFromImage(fixed)
    moving_array = sitk.GetArrayViewFromImage(warped)
    fixed_foreground = fixed_array > 0
    valid = fixed_foreground & (moving_array > 0)
    if not fixed_foreground.any() or valid.sum() < 2:
        raise ValueError("MRI reference images have insufficient nonzero overlap")
    coverage = float(valid.sum() / fixed_foreground.sum())
    correlation = float(np.corrcoef(fixed_array[valid], moving_array[valid])[0, 1])
    if not np.isfinite(correlation):
        raise ValueError("MRI reference images have insufficient intensity variation")
    return correlation, coverage


def image_bounds(image):
    corners = np.array([
        image.TransformContinuousIndexToPhysicalPoint(index)
        for index in itertools.product(*[(0, size - 1) for size in image.GetSize()])
    ])
    return corners.min(axis=0), corners.max(axis=0)


def canonical_coarse_image(image, spacing=4.0):
    lower, upper = image_bounds(image)
    size = [int(np.ceil((end - start) / spacing)) + 1 for start, end in zip(lower, upper)]
    return sitk.Resample(image, size, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear,
                         lower.tolist(), [spacing] * 3, [1, 0, 0, 0, 1, 0, 0, 0, 1],
                         0, sitk.sitkFloat32)


def template_initialization(fixed, moving):
    fixed_coarse = canonical_coarse_image(fixed)
    moving_coarse = canonical_coarse_image(moving)
    template = sitk.GetArrayFromImage(fixed_coarse)
    image = sitk.GetArrayFromImage(moving_coarse)
    if any(a > b for a, b in zip(template.shape, image.shape)):
        raise ValueError("Aligned MRI reference exceeds the native image; manual registration required")
    mask = (template > 0).astype(np.float32)
    count = float(mask.sum())
    if count < 2:
        raise ValueError("Aligned MRI reference has insufficient nonzero intensities")
    centered = (template - (template * mask).sum() / count) * mask
    reverse = (slice(None, None, -1),) * 3
    numerator = signal.fftconvolve(image, centered[reverse], mode="valid")
    local_sum = signal.fftconvolve(image, mask[reverse], mode="valid")
    local_squared_sum = signal.fftconvolve(image * image, mask[reverse], mode="valid")
    variance = np.maximum(local_squared_sum - local_sum * local_sum / count, 0)
    denominator = np.sqrt((centered * centered).sum() * variance)
    coverage = signal.fftconvolve((image > 0).astype(np.float32), mask[reverse], mode="valid") / count
    valid = (denominator > 1e-4) & (coverage >= 0.9)
    if not valid.any():
        raise ValueError("No adequately covered native MRI region matches the aligned reference")
    scores = np.full_like(numerator, -np.inf)
    np.divide(numerator, denominator, out=scores, where=valid)
    z, y, x = np.unravel_index(np.argmax(scores), scores.shape)
    moving_center = moving_coarse.TransformContinuousIndexToPhysicalPoint(
        (float(x + (template.shape[2] - 1) / 2),
         float(y + (template.shape[1] - 1) / 2),
         float(z + (template.shape[0] - 1) / 2)))
    fixed_center = fixed_coarse.TransformContinuousIndexToPhysicalPoint(
        [(size - 1) / 2 for size in fixed_coarse.GetSize()])
    initial = sitk.Euler3DTransform()
    initial.SetCenter(fixed_center)
    initial.SetTranslation((np.array(moving_center) - np.array(fixed_center)).tolist())
    print(f"Coarse image template correlation: {scores[z, y, x]:.4f}", flush=True)
    return initial


def optimize_mri_registration(fixed, moving, metric, initial):
    registration = sitk.ImageRegistrationMethod()
    if metric == "correlation":
        registration.SetMetricAsCorrelation()
    else:
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricFixedMask(sitk.Cast(fixed > 0, sitk.sitkUInt8))
        registration.SetMetricMovingMask(sitk.Cast(moving > 0, sitk.sitkUInt8))
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.4, seed=23)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=0.001, numberOfIterations=300,
        relaxationFactor=0.5, gradientMagnitudeTolerance=1e-6)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=False)
    return registration.Execute(fixed, moving)


def register_images(fixed, moving):
    """Recover a PET-space-to-native MRI transform from two copies of the same MRI."""
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)
    candidates = []
    for metric in ("correlation", "masked mutual information"):
        initial = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
        try:
            transform = optimize_mri_registration(fixed, moving, metric, initial)
            correlation, coverage = registration_similarity(fixed, moving, transform)
        except (RuntimeError, ValueError) as exc:
            print(f"{metric} registration failed: {exc}", file=sys.stderr, flush=True)
            continue
        print(f"{metric}: image correlation {correlation:.4f}, coverage {coverage:.4f}", flush=True)
        if correlation >= 0.95 and coverage >= 0.9:
            candidates.append((correlation, transform))
    if not candidates:
        initial = template_initialization(fixed, moving)
        transform = optimize_mri_registration(fixed, moving, "correlation", initial)
        correlation, coverage = registration_similarity(fixed, moving, transform)
        print(f"Template refinement: image correlation {correlation:.4f}, coverage {coverage:.4f}",
              flush=True)
        if correlation < 0.95 or coverage < 0.9:
            raise ValueError("MRI transform recovery failed its image similarity check; "
                             "register manually and visually verify the transform")
        candidates.append((correlation, transform))
    return max(candidates, key=lambda candidate: candidate[0])[1]


def aligned_mri_path(case, native_folder):
    names = {
        "RTstruct_T1": ("RTstruct_AlignedT1toPET", "RTstruct_AlignedT1toPET_BOX"),
        "RTstruct_T2FS": ("RTstruct_AlignedT2FStoPET", "RTstruct_AlignedT2FStoPET_BOX"),
        "RTstruct_STIR": ("RTstruct_AlignedSTIRtoPET", "RTstruct_AlignedSTIRtoPET_BOX"),
    }
    matches = [case / name / "image.nii.gz" for name in names[native_folder.name]
               if (case / name / "image.nii.gz").is_file()]
    if len(matches) != 1:
        raise ValueError(f"{case.name}: expected one PET-aligned reference for "
                         f"{native_folder.name}, found {len(matches)}; provide a "
                         "manually verified transform")
    return matches[0]


def relative_transform(common_to_reference, common_to_moving):
    transform = sitk.CompositeTransform(3)
    transform.AddTransform(common_to_moving)
    transform.AddTransform(common_to_reference.GetInverse())
    transform.FlattenTransform()
    return transform


def check_ct_pet_geometry(ct, pet):
    if not np.allclose(ct.GetDirection(), pet.GetDirection(), atol=1e-4):
        raise ValueError("CT/PET directions differ; provide a manually verified transform")
    if abs(ct.GetOrigin()[2] - pet.GetOrigin()[2]) > max(ct.GetSpacing()[2], pet.GetSpacing()[2]):
        raise ValueError("CT/PET axial origins differ; provide a manually verified transform")
    centers = [np.array(image.TransformContinuousIndexToPhysicalPoint(
        [(size - 1) / 2 for size in image.GetSize()])) for image in (ct, pet)]
    if np.linalg.norm((centers[0] - centers[1])[:2]) > 3:
        raise ValueError("CT/PET in-plane centers differ; provide a manually verified transform")


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
    parser.add_argument("--run_dir", type=Path,
                        help="Training run containing lists/ and models/; supplies default paths")
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--cases", type=Path, help="One patient ID per line")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--modalities", nargs="+", choices=list(MODALITIES))
    for name in MODALITIES:
        parser.add_argument(f"--{name}_checkpoint", type=Path)
        parser.add_argument(f"--{name}_weight", type=float, default=1.0)
    parser.add_argument("--reference", choices=list(MODALITIES),
                        help="Output grid; defaults to the first selected modality")
    parser.add_argument("--transform_dir", type=Path,
                        help="Directory of fixed-to-moving SimpleITK .tfm transforms")
    parser.add_argument("--auto_register", action="store_true",
                        help="Recover candidate transforms using PET-aligned MRI copies for inspection; do not fuse")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--mask", default="mask_GTV_Mass.nii.gz")
    parser.add_argument("--input_size", type=int, nargs=3, default=(32, 160, 160))
    parser.add_argument("--model_depth", type=int, default=18)
    parser.add_argument("--resnet_shortcut", choices=["A", "B"], default="A")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.run_dir is not None:
        if not args.run_dir.is_dir():
            parser.error(f"training run directory does not exist: {args.run_dir}")
        args.data_root = args.data_root or MNET_DIR.parent / "STS-nifti"
        args.cases = args.cases or args.run_dir / "lists" / "test_cases.txt"
        args.modalities = args.modalities or list(MODALITIES)
        for name in args.modalities:
            if getattr(args, f"{name}_checkpoint") is None:
                setattr(args, f"{name}_checkpoint", args.run_dir / "models" / f"{name}_best.pth.tar")
    for name in ("data_root", "cases", "modalities"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required without --run_dir")
    if args.output_dir is None and args.run_dir is None:
        parser.error("--output_dir is required without --run_dir")
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

    if args.output_dir is None:
        prefix = "registration-" if args.auto_register else "evaluation-"
        args.output_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=args.run_dir))
    print(f"Output directory: {args.output_dir}", flush=True)

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
            if case_out.exists():
                raise FileExistsError(f"Registration output already exists: {case_out}")
            images = {name: sitk.ReadImage(str(folder / "image.nii.gz"))
                      for name, folder in folders.items()}
            if "ct" in images and "pet" in images:
                check_ct_pet_geometry(images["ct"], images["pet"])
            if "ct" in images or "pet" in images:
                print(f"{case_id}: CT/PET identity assumes the paired STS acquisition; "
                      "visually inspect the result.", flush=True)
            common_to_native = {}
            common_image = images.get("ct", images.get("pet"))
            for name in args.modalities:
                if name in ("ct", "pet"):
                    common_to_native[name] = sitk.Transform(3, sitk.sitkIdentity)
                else:
                    aligned_path = aligned_mri_path(case, folders[name])
                    print(f"{case_id}: recovering {name} transform using {aligned_path}", flush=True)
                    aligned = sitk.ReadImage(str(aligned_path))
                    if common_image is not None:
                        ref_min, ref_max = image_bounds(common_image)
                        aligned_min, aligned_max = image_bounds(aligned)
                        if np.any(np.minimum(ref_max, aligned_max) <= np.maximum(ref_min, aligned_min)):
                            raise ValueError(f"{case_id}: {aligned_path} does not overlap the "
                                             "common PET/CT space; manually verify registration")
                    else:
                        common_image = aligned
                    common_to_native[name] = register_images(aligned, images[name])
            reference = images[args.reference]
            case_out.mkdir(parents=True)
            sitk.WriteImage(reference, str(case_out / f"{args.reference}_reference.nii.gz"))
            for name in args.modalities:
                if name == args.reference:
                    continue
                transform = relative_transform(common_to_native[args.reference], common_to_native[name])
                transform_path = transform_dir / f"{case_id}_{args.reference}_to_{name}.tfm"
                if transform_path.exists():
                    raise FileExistsError(f"Registration transform already exists: {transform_path}")
                sitk.WriteTransform(transform, str(transform_path))
                aligned = resample(images[name], reference, transform, sitk.sitkLinear)
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
