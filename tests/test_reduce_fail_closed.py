"""Host-side and static regressions for fail-closed CUDA reduce examples."""

from __future__ import annotations

import re
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
REDUCE_DIR = REPO_ROOT / "1_cuda_reduce_study"
HOST_TEST_SOURCE = REPO_ROOT / "tests/reduce_fail_closed_host.cu"
HELPER = REDUCE_DIR / "reduce_runtime_checks.cuh"


class ReduceCallerContractTests(unittest.TestCase):
    def test_all_nine_callers_use_every_fail_closed_gate(self):
        sources = sorted(REDUCE_DIR.glob("my_reduce_v*.cu"))
        self.assertEqual(len(sources), 9)

        for source_path in sources:
            source = source_path.read_text(encoding="utf-8")
            version = int(re.search(r"my_reduce_v(\d+)_", source_path.name).group(1))
            tolerance = "5e-3" if version >= 7 else "1e-3"
            with self.subTest(source=source_path.name):
                self.assertIn('#include "reduce_runtime_checks.cuh"', source)
                self.assertNotIn("bool check(", source)
                self.assertEqual(source.count("REDUCE_CUDA_LAUNCH_CHECK();"), 1)
                self.assertEqual(source.count("REDUCE_CUDA_SYNC_CHECK();"), 1)
                self.assertRegex(
                    source,
                    rf"reduce_checks::results_match\(\s*output,\s*result,\s*block_num,\s*{tolerance}\s*\)",
                )
                self.assertIn("return EXIT_FAILURE;", source)
                self.assertIn("return EXIT_SUCCESS;", source)

                for api, expected_count in (
                    ("cudaMalloc", 2),
                    ("cudaMemcpy", 2),
                    ("cudaFree", 2),
                ):
                    all_calls = re.findall(rf"\b{api}\s*\(", source)
                    guarded_calls = re.findall(
                        rf"REDUCE_CUDA_CHECK\s*\(\s*{api}\s*\(", source
                    )
                    self.assertEqual(len(all_calls), expected_count, api)
                    self.assertEqual(len(guarded_calls), expected_count, api)


class ReduceHostFailureTests(unittest.TestCase):
    def test_correct_result_is_zero_and_injected_failures_are_nonzero(self):
        self.assertTrue(HELPER.is_file(), "missing shared fail-closed helper")
        with tempfile.TemporaryDirectory() as tmp_dir:
            binary = Path(tmp_dir) / "reduce-fail-closed-host-test"
            compile_result = subprocess.run(
                [
                    "nvcc",
                    "-std=c++11",
                    "-I",
                    str(REDUCE_DIR),
                    str(HOST_TEST_SOURCE),
                    "-o",
                    str(binary),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)

            correct = subprocess.run(
                [str(binary), "correct"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(correct.returncode, 0, correct.stderr)

            for mode, token in (
                ("mismatch", "RESULT_MISMATCH"),
                ("nan", "RESULT_MISMATCH"),
                ("cuda-api", "CUDA_API_ERROR"),
                ("launch", "CUDA_LAUNCH_ERROR"),
                ("sync", "CUDA_SYNC_ERROR"),
            ):
                with self.subTest(mode=mode):
                    failed = subprocess.run(
                        [str(binary), mode],
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    self.assertNotEqual(failed.returncode, 0)
                    self.assertIn(token, failed.stderr)


if __name__ == "__main__":
    unittest.main()
