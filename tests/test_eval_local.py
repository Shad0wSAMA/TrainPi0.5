import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.eval_local import quantile_diagnostics, validate_local_inputs


class EvalLocalTest(unittest.TestCase):
    def test_reports_checkpoint_quantiles_that_differ_from_dataset(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            dataset = root / "dataset"
            (dataset / "meta").mkdir(parents=True)
            local = {"action": {"q01": [0.0, 1.0], "q99": [2.0, 3.0]},
                     "observation.state": {"q01": [0.0, 1.0], "q99": [2.0, 3.0]}}
            (dataset / "meta" / "stats.json").write_text(json.dumps(local))
            arrays = {f"{key}.{quantile}": np.array(values, dtype="<f4")
                      for key, value in local.items() for quantile, values in value.items()}
            arrays["action.q99"] = np.array([2.0, 9.0], dtype="<f4")
            header, body = {}, bytearray()
            for name, array in arrays.items():
                start = len(body)
                body.extend(array.tobytes())
                header[name] = {"dtype": "F32", "shape": list(array.shape),
                                "data_offsets": [start, len(body)]}
            encoded = json.dumps(header).encode()
            path = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)
            result = quantile_diagnostics(checkpoint, dataset)
            self.assertEqual(result["action"]["q99"]["max_abs_diff"], 6.0)
            self.assertEqual(result["observation.state"]["q99"]["max_abs_diff"], 0.0)

    def test_v21_dataset_is_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            for name in (
                "config.json", "model.safetensors", "policy_preprocessor.json",
                "policy_postprocessor.json",
                "policy_preprocessor_step_3_normalizer_processor.safetensors",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            ):
                (checkpoint / name).touch()
            dataset = root / "dataset"
            (dataset / "meta").mkdir(parents=True)
            (dataset / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v2.1"}))
            with self.assertRaisesRegex(RuntimeError, "v2.1.*v3.0"):
                validate_local_inputs(checkpoint, dataset)

    def test_missing_checkpoint_processor_is_reported(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").touch()
            (checkpoint / "model.safetensors").touch()
            with self.assertRaisesRegex(FileNotFoundError, "policy_preprocessor.json"):
                validate_local_inputs(checkpoint, root / "dataset")


if __name__ == "__main__":
    unittest.main()
