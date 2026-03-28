# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for reproducer functionality (CPU-only, no kernel execution)."""

import argparse
import os
import unittest
from pathlib import Path

import tritonparse.reproducer.orchestrator
from tests.test_utils import (
    cleanup_temp_dir,
    get_test_ndjson_file,
    setup_temp_reproduce_dir,
)
from tritonparse.reproducer.cli import _add_reproducer_args
from tritonparse.reproducer.ingestion.ndjson import get_kernel_info
from tritonparse.reproducer.utils import _normalize_device, determine_output_paths


class TestReproducer(unittest.TestCase):
    """Tests for reproducer generation."""

    def test_get_kernel_info_extracts_function_name_from_source(self):
        comp_event = {
            "payload": {
                "metadata": {
                    "name": "triton_add aiv",
                },
                "python_source": {
                    "file_path": "/tmp/test_tritonparse.py",
                    "code": "@triton.jit\ndef triton_add(x, y):\n    return x\n",
                },
            },
            "stack": [],
        }

        kernel_info = get_kernel_info(comp_event)

        self.assertEqual(kernel_info.function_name, "triton_add")

    def test_determine_output_paths_sanitizes_kernel_directory_name(self):
        temp_dir, out_dir = setup_temp_reproduce_dir()

        try:
            out_py_path, temp_json_path = determine_output_paths(
                out_dir,
                "triton_add aiv",
                "example",
                7,
            )

            self.assertEqual(out_py_path.parent.name, "triton_add_aiv")
            self.assertEqual(temp_json_path.parent.name, "triton_add_aiv")
            self.assertTrue(out_py_path.parent.exists())
        finally:
            cleanup_temp_dir(temp_dir)

    def test_reproduce_mutual_exclusivity(self):
        """Test that --line and --kernel/--launch-id are mutually exclusive."""
        parser = argparse.ArgumentParser()
        _add_reproducer_args(parser)

        # Test: both --line and --kernel provided should raise error
        # Create a mock parser with error method
        mock_parser = argparse.ArgumentParser()
        _add_reproducer_args(mock_parser)
        args = mock_parser.parse_args(
            ["test.ndjson", "--line", "5", "--kernel", "matmul_kernel"]
        )

        # The mutual exclusivity check happens in cli.py main()
        # We test that args are parsed correctly, and the check will happen there
        self.assertEqual(args.kernel, "matmul_kernel")
        self.assertEqual(args.line, 5)

        # Test: only --kernel should work (line defaults to 0, which is allowed)
        args = parser.parse_args(["test.ndjson", "--kernel", "matmul_kernel"])
        self.assertEqual(args.kernel, "matmul_kernel")
        self.assertEqual(args.line, 0)  # default value, allowed with --kernel

        # Test: only --line should work
        args = parser.parse_args(["test.ndjson", "--line", "5"])
        self.assertEqual(args.line, 5)
        self.assertIsNone(args.kernel)

    def test_reproduce_kernel_launch_id(self):
        """End-to-end test: reproduce using --kernel and --launch-id."""
        gz_file = get_test_ndjson_file()
        temp_dir, out_dir = setup_temp_reproduce_dir()

        try:
            # Test reproducing fused_op_kernel launch_id=0
            result = tritonparse.reproducer.orchestrator.reproduce(
                input_path=str(gz_file),
                line_index=0,  # Placeholder, will be recalculated from kernel_name
                out_dir=out_dir,
                template="example",
                kernel_name="fused_op_kernel",
                launch_id=0,
            )

            # Verify output structure
            self.assertIn("kernel", result)
            self.assertIn("repro_script", result)
            self.assertIn("repro_context", result)
            self.assertTrue(os.path.exists(result["repro_script"]))
            self.assertTrue(os.path.exists(result["repro_context"]))

            # Verify the script contains kernel name
            script_content = Path(result["repro_script"]).read_text()
            self.assertIn("fused_op_kernel", script_content)

        finally:
            cleanup_temp_dir(temp_dir)

    def test_reproduce_kernel_not_found(self):
        """Test that proper error is raised when kernel not found."""
        gz_file = get_test_ndjson_file()
        temp_dir, out_dir = setup_temp_reproduce_dir()

        try:
            with self.assertRaises(ValueError) as cm:
                tritonparse.reproducer.orchestrator.reproduce(
                    input_path=str(gz_file),
                    line_index=0,  # Placeholder, will be recalculated from kernel_name
                    out_dir=out_dir,
                    template="example",
                    kernel_name="nonexistent_kernel",
                    launch_id=0,
                )

            error_msg = str(cm.exception)
            self.assertIn("not found", error_msg)
            self.assertIn("nonexistent_kernel", error_msg)

        finally:
            cleanup_temp_dir(temp_dir)

    def test_reproduce_launch_id_out_of_range(self):
        """Test that proper error is raised when launch_id is out of range."""
        gz_file = get_test_ndjson_file()
        temp_dir, out_dir = setup_temp_reproduce_dir()

        try:
            # fused_op_kernel has only 4 launches (0-3), test with launch_id=10
            with self.assertRaises(ValueError) as cm:
                tritonparse.reproducer.orchestrator.reproduce(
                    input_path=str(gz_file),
                    line_index=0,  # Placeholder, will be recalculated from kernel_name
                    out_dir=out_dir,
                    template="example",
                    kernel_name="fused_op_kernel",
                    launch_id=10,
                )

            error_msg = str(cm.exception)
            self.assertIn("has only 4 launches", error_msg)
            self.assertIn("--launch-id 10", error_msg)
            self.assertIn("Valid range: 0 to 3", error_msg)

        finally:
            cleanup_temp_dir(temp_dir)


class TestNormalizeDevice(unittest.TestCase):
    """Tests for preserving explicit device selection in reproducers."""

    def test_preserves_cuda_without_index(self):
        self.assertEqual(_normalize_device("cuda"), "cuda")

    def test_preserves_cuda_with_index(self):
        self.assertEqual(_normalize_device("cuda:2"), "cuda:2")

    def test_preserves_npu_without_index(self):
        self.assertEqual(_normalize_device("npu"), "npu")

    def test_preserves_npu_with_index(self):
        self.assertEqual(_normalize_device("npu:3"), "npu:3")

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(_normalize_device(" npu:1 "), "npu:1")

    def test_non_string_passthrough(self):
        self.assertIsNone(_normalize_device(None))


if __name__ == "__main__":
    unittest.main()
