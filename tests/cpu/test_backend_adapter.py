# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import unittest

from tritonparse.backend import get_backend_registry, get_present_stage_descriptors_from_event
from tritonparse.parse.trace_processor import parse_single_trace_content


class TestBackendAdapter(unittest.TestCase):
    def test_registry_resolves_built_in_adapter_by_name(self):
        adapter = get_backend_registry().resolve(adapter_name="nvidia")
        self.assertEqual(adapter.adapter_name, "nvidia")
        self.assertEqual(adapter.pipeline_kind, "triton")
        self.assertEqual(adapter.runtime_backend, "cuda")

    def test_registry_validates_pipeline_kind_when_resolving_adapter(self):
        adapter = get_backend_registry().resolve(
            adapter_name="amd",
            pipeline_kind="triton",
        )
        self.assertEqual(adapter.adapter_name, "amd")

    def test_adapter_reports_known_stage_extensions(self):
        adapter = get_backend_registry().resolve(adapter_name="nvidia")
        self.assertEqual(
            adapter.known_stage_extensions,
            {".ttir", ".ttgir", ".llir", ".ptx", ".cubin", ".sass"},
        )

    def test_present_stage_descriptors_fall_back_to_artifact_suffixes(self):
        event = {
            "event_type": "compilation",
            "payload": {
                "metadata": {},
                "file_content": {
                    "kernel.ttir": "module {}",
                    "kernel.ptx": "// ptx",
                },
                "file_path": {},
            },
        }

        stages = get_present_stage_descriptors_from_event(event)
        self.assertEqual([stage.name for stage in stages], ["ttir", "ptx"])

    def test_present_stage_descriptors_use_metadata_when_available(self):
        event = {
            "event_type": "compilation",
            "payload": {
                "metadata": {
                    "stage_descriptors": [
                        {
                            "name": "ptx",
                            "extension": ".ptx",
                            "display_name": "PTX",
                            "display_order": 40,
                            "is_text": True,
                            "supports_source_mapping": True,
                            "parser_id": "ptx_loc",
                            "syntax_id": "ptx",
                            "file_name": "kernel.ptx",
                        },
                        {
                            "name": "ttgir",
                            "extension": ".ttgir",
                            "display_name": "TTGIR",
                            "display_order": 20,
                            "is_text": True,
                            "supports_source_mapping": True,
                            "parser_id": "generic_loc",
                            "syntax_id": "mlir",
                            "file_name": "kernel.ttgir",
                        },
                    ]
                },
                "file_content": {
                    "kernel.ttgir": "module {}",
                    "kernel.ptx": "// ptx",
                },
                "file_path": {},
            },
        }

        stages = get_present_stage_descriptors_from_event(event)
        self.assertEqual([stage.name for stage in stages], ["ttgir", "ptx"])

    def test_parse_single_trace_content_keeps_legacy_fallback(self):
        raw_trace = {
            "event_type": "compilation",
            "payload": {
                "metadata": {"hash": "abc", "name": "kernel"},
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

        parsed = json.loads(parse_single_trace_content(json.dumps(raw_trace)))
        self.assertIn("source_mappings", parsed["payload"])
        self.assertIn("ttir", parsed["payload"]["source_mappings"])

    def test_parse_single_trace_content_prefers_stage_metadata_when_available(self):
        raw_trace = {
            "event_type": "compilation",
            "payload": {
                "metadata": {
                    "hash": "abc",
                    "name": "kernel",
                    "adapter_name": "nvidia",
                    "pipeline_kind": "triton",
                    "stage_descriptors": [
                        {
                            "name": "ttir",
                            "extension": ".ttir",
                            "display_name": "TTIR",
                            "display_order": 10,
                            "is_text": True,
                            "supports_source_mapping": True,
                            "parser_id": "generic_loc",
                            "syntax_id": "mlir",
                            "file_name": "kernel.ttir",
                        }
                    ],
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

        parsed = json.loads(parse_single_trace_content(json.dumps(raw_trace)))
        self.assertIn("source_mappings", parsed["payload"])
        self.assertIn("ttir", parsed["payload"]["source_mappings"])


if __name__ == "__main__":
    unittest.main()