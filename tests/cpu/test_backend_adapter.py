# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

from tritonparse.backend import (
    build_backend_trace_contract,
    build_stage_descriptor_trace_contract,
    get_backend_registry,
    get_present_stage_descriptors_from_event,
)
from tritonparse.parse.trace_processor import parse_single_trace_content
from tritonparse.reproducer.ingestion.ndjson import build_context_bundle


class TestBackendAdapter(unittest.TestCase):
    def test_registry_resolves_from_trace_contract(self):
        adapter = get_backend_registry().resolve_from_runtime(
            adapter_name="nvidia",
        )
        contract = build_backend_trace_contract(adapter)

        resolved = get_backend_registry().resolve_from_trace(
            compilation_metadata=contract,
            launch_metadata=None,
        )
        self.assertEqual(resolved.adapter_name, "nvidia")

    def test_stage_descriptors_are_serialized_into_trace_contract(self):
        adapter = get_backend_registry().resolve_from_runtime(
            adapter_name="nvidia",
        )

        stage_descriptors = build_stage_descriptor_trace_contract(
            adapter,
            ["kernel.ttir", "kernel.ptx", "metadata.json"],
        )

        self.assertEqual([stage["file_name"] for stage in stage_descriptors], ["kernel.ptx", "kernel.ttir"])
        self.assertEqual([stage["name"] for stage in stage_descriptors], ["ptx", "ttir"])

        event = {
            "event_type": "compilation",
            "payload": {
                "metadata": {
                    "adapter_name": "nvidia",
                    "stage_descriptors": stage_descriptors,
                },
                "file_content": {
                    "kernel.ttir": "module {}",
                    "kernel.ptx": "// ptx",
                },
                "file_path": {},
            },
        }
        self.assertEqual(
            [stage.name for stage in get_present_stage_descriptors_from_event(event)],
            ["ttir", "ptx"],
        )

    def test_parse_single_trace_content_requires_backend_contract(self):
        raw_trace = {
            "event_type": "compilation",
            "payload": {
                "metadata": {"hash": "abc", "name": "kernel"},
                "file_content": {
                    "kernel.ttir": '#loc = loc("/tmp/test.py":1:1)\nmodule {\n  %0 = arith.constant 0 : i32 loc(#loc)\n}'
                },
                "file_path": {"kernel.ttir": "/tmp/kernel.ttir"},
            },
        }

        with self.assertRaises(ValueError):
            parse_single_trace_content(__import__("json").dumps(raw_trace))

    def test_parse_single_trace_content_uses_artifact_presence(self):
        adapter = get_backend_registry().resolve_from_runtime(
            adapter_name="nvidia",
        )
        contract = build_backend_trace_contract(adapter)
        raw_trace = {
            "event_type": "compilation",
            "payload": {
                "metadata": {
                    "hash": "abc",
                    "name": "kernel",
                    **contract,
                },
                "python_source": {
                    "file_path": "/tmp/test.py",
                    "start_line": 1,
                    "end_line": 3,
                    "code": "@triton.jit\ndef kernel():\n    pass\n",
                },
                "file_content": {
                    "kernel.ttir": '#loc = loc("/tmp/test.py":2:5)\nmodule {\n  %0 = arith.constant 0 : i32 loc(#loc)\n}'
                },
                "file_path": {"kernel.ttir": "/tmp/kernel.ttir"},
            },
        }

        parsed = __import__("json").loads(
            parse_single_trace_content(__import__("json").dumps(raw_trace))
        )
        self.assertIn("source_mappings", parsed["payload"])
        self.assertIn("ttir", parsed["payload"]["source_mappings"])
        self.assertIn("python", parsed["payload"]["source_mappings"])

    def test_reproducer_context_uses_adapter_device_and_sync(self):
        adapter = get_backend_registry().resolve_from_runtime(
            adapter_name="ascend",
        )
        contract = build_backend_trace_contract(adapter)

        comp_event = {
            "event_type": "compilation",
            "payload": {
                "metadata": {
                    "hash": "kernel_hash",
                    "name": "kernel",
                    **contract,
                },
                "python_source": {
                    "file_path": "/tmp/test_kernel.py",
                    "start_line": 1,
                    "end_line": 3,
                    "code": "@triton.jit\ndef kernel(x):\n    return x\n",
                },
                "file_content": {},
                "file_path": {},
            },
            "stack": [],
        }
        launch_event = {
            "event_type": "launch",
            "name": "kernel",
            "grid": [1, 1, 1],
            "stack": [],
            "compilation_metadata": {
                "hash": "kernel_hash",
                "name": "kernel",
                **contract,
            },
            "extracted_args": {
                "x": {
                    "type": "tensor",
                    "shape": [4],
                    "dtype": "torch.float32",
                    "device": "npu",
                    "stride": [1],
                    "is_contiguous": True,
                    "numel": 4,
                }
            },
        }

        bundle = build_context_bundle([comp_event, launch_event], 1)
        self.assertEqual(bundle.tensor_args["x"]["device"], "npu:0")
        self.assertEqual(bundle.synchronize_snippet, "torch.npu.synchronize()")


if __name__ == "__main__":
    unittest.main()