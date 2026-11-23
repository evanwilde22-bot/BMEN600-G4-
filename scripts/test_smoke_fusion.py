import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import smoke_fusion


class SmokeFusionTests(unittest.TestCase):
    def test_runs_four_models_on_matching_patient_subsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            weights = root / "weights.pth"
            weights.touch()
            commands = []

            def fake_run(command, **kwargs):
                commands.append((command, kwargs))
                if "generate_fusion_lists.py" in command[1]:
                    lists = Path(command[command.index("--output_dir") + 1])
                    lists.mkdir()
                    for split in ("train", "val"):
                        for mod in smoke_fusion.MODALITIES:
                            (lists / f"{split}_{mod}.txt").write_text(
                                f"STS_001/{mod}/image STS_001/{mod}/mask\n"
                                f"STS_002/{mod}/image STS_002/{mod}/mask\n")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(smoke_fusion.subprocess, "run", side_effect=fake_run):
                result = smoke_fusion.main([
                    "--data-root", str(data), "--pretrained", str(weights),
                    "--run-dir", str(root / "output"),
                    "--train-cases", "all", "--val-cases", "1",
                    "--epochs", "3", "--val-interval", "2",
                ])
            self.assertEqual(result, 0)
            self.assertEqual(len(commands), 5)
            for (command, kwargs), modality in zip(commands[1:], smoke_fusion.MODALITIES):
                self.assertEqual(command[command.index("--modality_name") + 1], modality)
                self.assertEqual(command[command.index("--n_epochs") + 1], "3")
                self.assertEqual(command[command.index("--val_interval") + 1], "2")
                self.assertIn("--no_cuda", command)
                self.assertGreater(kwargs["timeout"], 0)
                for flag, expected in (("--img_list", 2), ("--val_img_list", 1)):
                    lines = Path(command[command.index(flag) + 1]).read_text().splitlines()
                    self.assertEqual(len(lines), expected)
                    self.assertTrue(lines[0].startswith("STS_001/"))

    def test_existing_run_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights.pth"
            weights.touch()
            with self.assertRaises(SystemExit) as exc:
                smoke_fusion.main([
                    "--data-root", str(root), "--pretrained", str(weights),
                    "--run-dir", str(root),
                ])
            self.assertNotEqual(exc.exception.code, 0)

    def test_timeout_returns_failure_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights.pth"
            weights.touch()
            with patch.object(smoke_fusion.subprocess, "run",
                              side_effect=subprocess.TimeoutExpired("generator", 1)) as run:
                result = smoke_fusion.main([
                    "--data-root", str(root), "--pretrained", str(weights),
                    "--output-root", str(root / "output"),
                ])
            self.assertEqual(result, 1)
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
