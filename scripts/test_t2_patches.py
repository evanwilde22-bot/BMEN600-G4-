import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_patches


class PatchTests(unittest.TestCase):
    def test_sliding_window_covers_volume_without_reference_mask(self):
        class Pointwise(torch.nn.Module):
            def forward(self, image):
                return torch.cat((torch.zeros_like(image), image), dim=1)

        volume = np.linspace(-2, 2, 11 * 19 * 23, dtype=np.float32).reshape(11, 19, 23)
        result = t2_patches.predict(Pointwise(), volume, (8, 16, 16), torch.device("cpu"))
        self.assertEqual(result.shape, volume.shape)
        np.testing.assert_allclose(result, torch.sigmoid(torch.from_numpy(volume)), atol=1e-6)
        weighted = t2_patches.predict(Pointwise(), volume, (8, 16, 16), torch.device("cpu"), True)
        np.testing.assert_allclose(weighted, result, atol=1e-6)
        weights = t2_patches.blending_weights((8, 16, 16))
        self.assertGreater(weights[4, 8, 8], weights[0, 0, 0])
        self.assertGreater(weights.min(), 0)

    def test_refined_sampling_and_normalization(self):
        image = np.arange(1, 1001, dtype=np.float32).reshape(10, 10, 10)
        image[0, 0, 0] = 0
        image[-1, -1, -1] = 1e9
        normalized = t2_patches.robust_normalize(image)
        self.assertEqual(normalized[0, 0, 0], 0)
        self.assertGreater(float(normalized.std()), 0.9)
        self.assertLess(float(normalized.max()), 3)
        label = np.zeros_like(image, dtype=bool)
        label[4:6, 4:6, 4:6] = True
        foreground, boundary, bright = t2_patches.sampling_locations(normalized, label)
        self.assertTrue(label.ravel()[foreground].all())
        self.assertFalse(label.ravel()[boundary].any())
        self.assertFalse(label.ravel()[bright].any())
        augmented = t2_patches.augment_refined(normalized, np.random.default_rng(42))
        self.assertEqual(augmented[0, 0, 0], 0)
        self.assertFalse(np.array_equal(augmented, normalized))
        with self.assertRaises(ValueError):
            t2_patches.robust_normalize(np.ones((4, 4, 4), dtype=np.float32))

    def test_skip_decoder_has_full_resolution_and_backpropagates(self):
        torch.set_num_threads(2)
        model, parameters = t2_patches.model_for((8, 16, 16), None, True)
        model.train()
        logits = model(torch.randn(1, 1, 8, 16, 16))
        self.assertEqual(tuple(logits.shape), (1, 2, 8, 16, 16))
        logits.mean().backward()
        self.assertIsNotNone(model.backbone.conv1.weight.grad)
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertTrue(parameters["base_parameters"])
        self.assertTrue(parameters["new_parameters"])

    def test_training_and_evaluation_preserve_native_grid_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, lists, run = root / "data", root / "lists", root / "run"
            lists.mkdir()
            affine = np.diag([-1.5, -1.5, 7.0, 1.0])
            affine[:2, 3] = 22.5
            for number, split in enumerate(("train", "val", "test"), 1):
                folder = data / f"STS_{number:03d}" / "RTstruct_STIR"
                folder.mkdir(parents=True)
                image = np.arange(16 * 16 * 8, dtype=np.float32).reshape(16, 16, 8)
                mask = np.zeros(image.shape, dtype=np.uint8)
                mask[5:10, 5:10, 2:6] = 1
                nib.save(nib.Nifti1Image(image, affine), folder / "image.nii.gz")
                nib.save(nib.Nifti1Image(mask, affine), folder / "mask_GTV_Mass.nii.gz")
                relative = folder.relative_to(data)
                (lists / f"{split}_t2.txt").write_text(
                    f"{relative}/image.nii.gz {relative}/mask_GTV_Mass.nii.gz\n")

            case = t2_patches.prepare(data / "STS_002/RTstruct_STIR/image.nii.gz",
                                      data / "STS_002/RTstruct_STIR/mask_GTV_Mass.nii.gz")
            restored = t2_patches.resample_from_to(
                nib.Nifti1Image(case[3].transpose(2, 1, 0).astype(np.uint8), case[1].affine),
                case[0], order=0)
            np.testing.assert_array_equal(np.asarray(restored.dataobj), mask)

            def small_model(patch_size, pretrained, refined=False):
                model = torch.nn.Conv3d(1, 2, 1)
                return model, {"base_parameters": [], "new_parameters": list(model.parameters())}

            args = ["t2_patches", "--run-dir", str(run), "--data-root", str(data),
                    "--lists", str(lists), "--epochs", "1", "--val-interval", "1",
                    "--patch-size", "8", "16", "16"]
            with patch.object(t2_patches, "model_for", side_effect=small_model), \
                 patch.object(sys, "argv", args):
                t2_patches.main()
            self.assertTrue((run / "t2_best.pth.tar").is_file())
            with patch.object(t2_patches, "model_for", side_effect=small_model), \
                 patch.object(sys, "argv", ["t2_patches", "--run-dir", str(run), "--evaluate", "val"]):
                t2_patches.main()
            result = next(run.glob("evaluation-val-*/STS_002"))
            prediction = nib.load(result / "t2_mask.nii.gz")
            probability = nib.load(result / "t2_probability.nii.gz")
            self.assertEqual(prediction.shape, (16, 16, 8))
            np.testing.assert_allclose(prediction.affine, affine)
            self.assertEqual(probability.get_data_dtype(), np.dtype("float32"))
            self.assertEqual(len(list(result.glob("*.nii.gz"))), 2)

            refined_run = root / "refined"
            args = ["t2_patches", "--run-dir", str(refined_run), "--data-root", str(data),
                    "--lists", str(lists), "--epochs", "1", "--val-interval", "1",
                    "--patch-size", "8", "16", "16", "--refined", "--patches-per-case", "3",
                    "--seed", "42"]
            with patch.object(t2_patches, "model_for", side_effect=small_model), \
                 patch.object(sys, "argv", args):
                t2_patches.main()
            with patch.object(t2_patches, "model_for", side_effect=small_model), \
                 patch.object(sys, "argv", ["t2_patches", "--run-dir", str(refined_run),
                                            "--evaluate", "val", "--threshold", "0.7"]):
                t2_patches.main()
            import json
            config = json.loads((refined_run / "run.json").read_text())
            self.assertEqual(config["seed"], 42)
            metadata = json.loads(next(refined_run.glob("evaluation-val-*/run.json")).read_text())
            self.assertEqual(metadata["threshold"], 0.7)


if __name__ == "__main__":
    unittest.main()
