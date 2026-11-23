import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MNet"))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets.sarcoma import SarcomaDataset
import fuse_sarcoma
from fuse_sarcoma import fuse, register_images, resample
from generate_fusion_lists import main as generate_lists
from model import generate_model
from train_sarcoma import loss_inputs_at_input_resolution


class FusionTests(unittest.TestCase):
    def test_run_directory_defaults_evaluate_all_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            (run / "lists").mkdir(parents=True)
            (run / "lists" / "test_cases.txt").write_text("STS_001\n")
            data = root / "data"
            array = np.zeros((8, 8, 8), dtype=np.uint8)
            array[3:5, 3:5, 3:5] = 1
            image = sitk.GetImageFromArray(array)
            for name in ("CT", "T1", "STIR", "PET"):
                folder = data / "STS_001" / f"RTstruct_{name}"
                folder.mkdir(parents=True)
                sitk.WriteImage(image, str(folder / "image.nii.gz"))
                sitk.WriteImage(image, str(folder / "mask_GTV_Mass.nii.gz"))
            transforms = root / "transforms"
            transforms.mkdir()
            for name in ("t1", "t2", "pet"):
                sitk.WriteTransform(sitk.Transform(3, sitk.sitkIdentity),
                                    str(transforms / f"STS_001_ct_to_{name}.tfm"))
            args = ["fuse_sarcoma", "--run_dir", str(run), "--data_root", str(data),
                    "--transform_dir", str(transforms), "--evaluate"]
            probability = sitk.Cast(image, sitk.sitkFloat32)
            with patch.object(sys, "argv", args), \
                 patch.object(fuse_sarcoma, "load_model", return_value=None) as load, \
                 patch.object(fuse_sarcoma, "predict", return_value=(image, probability)):
                fuse_sarcoma.main()
            self.assertEqual(load.call_count, 4)
            evaluation = next(run.glob("evaluation-*"))
            self.assertTrue((evaluation / "STS_001" / "fused_mask.nii.gz").is_file())
            import json
            self.assertEqual(json.loads((evaluation / "run.json").read_text())["dice"]["STS_001"], 1.0)

    def test_loss_keeps_small_input_resolution_tumor(self):
        labels = torch.zeros((1, 1, 8, 8, 8))
        labels[0, 0, 3, 3, 3] = 1
        logits = torch.zeros((1, 2, 2, 2, 2), requires_grad=True)
        resized_logits, target = loss_inputs_at_input_resolution(logits, labels)
        self.assertEqual(resized_logits.shape, (1, 2, 8, 8, 8))
        self.assertEqual(target.sum().item(), 1)
        resized_logits.sum().backward()
        self.assertIsNotNone(logits.grad)

    def test_weighted_average_ignores_outside_field_of_view(self):
        first = np.array([0.2, 0.4, 0.8], dtype=np.float32)
        second = np.array([0.8, 0.9, 0.1], dtype=np.float32)
        result = fuse([(first, np.ones(3), 1), (second, np.array([1, 0, 0]), 3)], first.shape)
        np.testing.assert_allclose(result, [0.65, 0.4, 0.8])

    def test_fixed_to_moving_transform_resamples_prediction(self):
        array = np.zeros((8, 8, 8), dtype=np.float32)
        array[3, 3, 4] = 1
        image = sitk.GetImageFromArray(array)
        transform = sitk.TranslationTransform(3, (1, 0, 0))
        aligned = resample(image, image, transform, sitk.sitkNearestNeighbor)
        self.assertEqual(sitk.GetArrayFromImage(aligned)[3, 3, 3], 1)

    def test_registration_accepts_different_image_pixel_types(self):
        coords = np.indices((24, 24, 24))
        data = (1000 * np.exp(-((coords - 11.5) ** 2).sum(axis=0) / 40)).astype(np.int16)
        fixed = sitk.GetImageFromArray(data)
        moving = sitk.GetImageFromArray(data.astype(np.float32))
        result = register_images(fixed, moving)
        self.assertEqual(result.GetDimension(), 3)

    def test_training_and_inference_use_same_volume_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.txt"
            path.write_text("")
            settings = SimpleNamespace(input_D=8, input_H=8, input_W=8,
                                       phase="train", consistent_preprocessing=True)
            dataset = SarcomaDataset(tmp, path, settings)
            data = np.arange(512, dtype=np.float32).reshape(8, 8, 8)
            image = nib.Nifti1Image(data, np.eye(4))
            label = nib.Nifti1Image((data > 200).astype(np.uint8), np.eye(4))
            with patch.object(dataset, "__random_flip__", side_effect=lambda a, b: (a, b)), \
                 patch.object(dataset, "__random_intensity_jitter__", side_effect=lambda a: a):
                trained, _ = dataset.__training_data_process__(image, label)
            inference = dataset.__testing_data_process__(image)
            np.testing.assert_array_equal(trained, inference)

    def test_patient_split_groups_all_modalities_and_excludes_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "lists"
            for index in range(5):
                case = root / f"STS_{index:03d}"
                for modality in ("CT", "T1", "STIR", "PET", "AlignedT1toPET"):
                    folder = case / f"RTstruct_{modality}"
                    folder.mkdir(parents=True)
                    (folder / "image.nii.gz").touch()
                    (folder / "mask_GTV_Mass.nii.gz").touch()
            with patch.object(sys, "argv", ["generate_fusion_lists", "--data_root", str(root),
                                            "--output_dir", str(output)]):
                generate_lists()
            train = set((output / "train_cases.txt").read_text().splitlines())
            val = set((output / "val_cases.txt").read_text().splitlines())
            test = set((output / "test_cases.txt").read_text().splitlines())
            self.assertFalse((train & val) | (train & test) | (val & test))
            self.assertEqual(len(train | val | test), 5)
            for split, expected in (("train", train), ("val", val), ("test", test)):
                for name in ("ct", "t1", "t2", "pet"):
                    lines = (output / f"{split}_{name}.txt").read_text().splitlines()
                    self.assertEqual({line.split("/")[0] for line in lines}, expected)
                    self.assertNotIn("Aligned", "\n".join(lines))

    def test_cpu_loads_module_prefixed_pretrained_weights(self):
        opts = SimpleNamespace(model="resnet", model_depth=10, input_W=16, input_H=16,
                               input_D=16, resnet_shortcut="B", no_cuda=True,
                               n_seg_classes=2, dropout_prob=0, phase="test",
                               pretrain_path="")
        original, _ = generate_model(opts)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pth"
            expected = torch.full_like(original.conv1.weight, 0.125)
            state = {"module." + key: value for key, value in original.state_dict().items()
                     if not key.startswith("conv_seg.")}
            state["module.conv1.weight"] = expected
            torch.save({"state_dict": state}, path)
            opts.phase = "train"
            opts.pretrain_path = str(path)
            opts.new_layer_names = ["conv_seg"]
            loaded, _ = generate_model(opts)
        self.assertTrue(torch.equal(loaded.conv1.weight, expected))


if __name__ == "__main__":
    unittest.main()
