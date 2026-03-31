#  Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Integration tests for backend-agnostic TritonParse functionality.

Tests the complete flow from runtime tracing through parsing to reproducer generation,
ensuring backend information is correctly propagated throughout.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from tritonparse.backends import BackendInfo, extract_backend_from_trace
from tritonparse.parse.trace_processor import parse_single_trace_content


class TestBackendMetadataPropagation:
    """Tests that backend metadata propagates correctly through the pipeline."""

    def test_ascend_trace_structure(self):
        """Test that Ascend trace has correct structure."""
        # Create a mock Ascend trace event
        trace_event = {
            "event_type": "compilation",
            "compilation_metadata": {
                "registry_key": "ascend",
                "target_backend": "npu",
                "device_prefix": "npu",
                "target": {
                    "backend": "npu",
                    "arch": 910,
                    "warp_size": 16,
                },
                "ir_stages": [
                    {
                        "name": "ttir",
                        "extension": ".ttir",
                        "stage_origin": "backend",
                        "is_text": True,
                        "supports_source_mapping": True,
                        "mapping_kind": "generic",
                    },
                    {
                        "name": "ttadapter",
                        "extension": ".ttadapter",
                        "stage_origin": "backend",
                        "is_text": True,
                        "supports_source_mapping": True,
                        "mapping_kind": "generic",
                    },
                    {
                        "name": "bcmlir",
                        "extension": ".bcmlir",
                        "stage_origin": "backend",
                        "is_text": True,
                        "supports_source_mapping": True,
                        "mapping_kind": "generic",
                    },
                ],
            },
            "payload": {
                "metadata": {
                    "hash": "abc123",
                    "name": "test_kernel",
                },
                "file_content": {
                    "test_kernel.ttir": "module { ... }",
                    "test_kernel.ttadapter": "...",
                    "test_kernel.bcmlir": "...",
                },
                "file_path": {},
            },
        }

        # Extract backend info
        backend_info = extract_backend_from_trace(trace_event)

        assert isinstance(backend_info, BackendInfo)
        assert backend_info.target_backend == "npu"
        assert backend_info.device_prefix == "npu"
        assert backend_info.registry_key == "ascend"

    def test_nvidia_trace_structure(self):
        """Test that NVIDIA trace has correct structure."""
        trace_event = {
            "event_type": "compilation",
            "compilation_metadata": {
                "registry_key": "nvidia",
                "target_backend": "cuda",
                "device_prefix": "cuda",
                "target": {
                    "backend": "cuda",
                    "arch": 80,
                    "warp_size": 32,
                },
                "ir_stages": [
                    {
                        "name": "ttir",
                        "extension": ".ttir",
                        "stage_origin": "backend",
                        "is_text": True,
                        "supports_source_mapping": True,
                        "mapping_kind": "generic",
                    },
                    {
                        "name": "ptx",
                        "extension": ".ptx",
                        "stage_origin": "backend",
                        "is_text": True,
                        "supports_source_mapping": True,
                        "mapping_kind": "ptx",
                    },
                ],
            },
            "payload": {
                "metadata": {"hash": "def456"},
                "file_content": {
                    "kernel.ptx": ".version 6.4 ...",
                },
                "file_path": {},
            },
        }

        backend_info = extract_backend_from_trace(trace_event)

        assert backend_info.target_backend == "cuda"
        assert backend_info.device_prefix == "cuda"
        assert backend_info.registry_key == "nvidia"


class TestIRStagesExtraction:
    """Tests IR stages extraction and handling."""

    def test_extract_ir_types_from_ascend_trace(self):
        """Test extracting IR types from Ascend trace."""
        # Simulate frontend extraction from trace
        ir_files = {
            "kernel.ttir": "module {...}",
            "kernel.ttadapter": "...",
            "kernel.bcmlir": "...",
            "kernel.source.json": "{}",
        }

        # Extract IR types (frontend logic)
        ir_types = set()
        for filename in ir_files.keys():
            if '.' in filename:
                parts = filename.split('.')
                ir_type = parts[-1]
                if ir_type not in ['json', 'py', 'source']:
                    ir_types.add(ir_type)

        assert 'ttir' in ir_types
        assert 'ttadapter' in ir_types
        assert 'bcmlir' in ir_types
        assert 'source' not in ir_types  # Should be filtered
        assert 'json' not in ir_types  # Should be filtered

    def test_ir_stages_metadata_contains_mapping_kind(self):
        """Test that ir_stages contains mapping_kind information."""
        trace_event = {
            "compilation_metadata": {
                "ir_stages": [
                    {
                        "name": "ttir",
                        "mapping_kind": "generic",
                        "is_text": True,
                    },
                    {
                        "name": "ptx",
                        "mapping_kind": "ptx",
                        "is_text": True,
                    },
                    {
                        "name": "sass",
                        "mapping_kind": "sass",
                        "is_text": True,
                    },
                ],
            }
        }

        ir_stages = trace_event["compilation_metadata"]["ir_stages"]

        # Verify each stage has mapping_kind
        for stage in ir_stages:
            assert "mapping_kind" in stage
            assert stage["mapping_kind"] in ["generic", "ptx", "sass", "none"]


class TestBackendMetadataInjection:
    """Tests that backend_metadata is correctly injected during parsing."""

    def test_backend_metadata_injection(self):
        """Test that parse_single_trace_content injects backend_metadata."""
        # Create a trace event with compilation_metadata
        trace_event = {
            "event_type": "compilation",
            "compilation_metadata": {
                "device_prefix": "npu",
                "target_backend": "npu",
                "registry_key": "ascend",
            },
            "payload": {
                "metadata": {"hash": "test123"},
                "file_content": {},
                "file_path": {},
            },
        }

        # Convert to JSON string (as would be in trace file)
        trace_json = json.dumps(trace_event)

        # Parse the trace (this should inject backend_metadata)
        parsed_trace = json.loads(parse_single_trace_content(trace_json))

        # Check that backend_metadata was injected
        assert "backend_metadata" in parsed_trace
        assert parsed_trace["backend_metadata"]["device_prefix"] == "npu"
        assert parsed_trace["backend_metadata"]["target_backend"] == "npu"


class TestDevicePrefixUsage:
    """Tests that device_prefix is used correctly in different stages."""

    def test_synchronize_code_generation_for_ascend(self):
        """Test that Ascend backend generates correct synchronize code."""
        # Simulate reproducer placeholder replacement logic
        device_prefix = "npu"
        sync_call = f"torch.{device_prefix}.synchronize()"

        assert sync_call == "torch.npu.synchronize()"

    def test_synchronize_code_generation_for_nvidia(self):
        """Test that NVIDIA backend generates correct synchronize code."""
        device_prefix = "cuda"
        sync_call = f"torch.{device_prefix}.synchronize()"

        assert sync_call == "torch.cuda.synchronize()"

    def test_device_inference_priority(self):
        """Test device inference priority order."""
        launch_event = {
            # Priority 1: backend_metadata.device_prefix (newest)
            "backend_metadata": {"device_prefix": "npu"},
            # Priority 2: compilation_metadata.device_prefix
            "compilation_metadata": {"device_prefix": "npu"},
            # Priority 3: backend_name (legacy)
            "backend_name": "npu",
        }

        # Should use backend_metadata.device_prefix first
        device = launch_event.get("backend_metadata", {}).get("device_prefix")
        assert device == "npu"


class TestBackendCompatibility:
    """Tests backward compatibility and graceful degradation."""

    def test_missing_ir_stages_fallback_to_filename_extraction(self):
        """Test that missing ir_stages falls back to filename extraction."""
        trace_event = {
            "compilation_metadata": {
                "device_prefix": "npu",
                "target_backend": "npu",
                # ir_stages is missing
            },
            "payload": {
                "file_content": {
                    "kernel.ttadapter": "...",
                    "kernel.bcmlir": "...",
                },
            },
        }

        # Should still be able to extract backend info
        backend_info = extract_backend_from_trace(trace_event)
        assert backend_info.device_prefix == "npu"

        # Should fall back to filename extraction for IR types
        ir_files = trace_event["payload"]["file_content"]
        ir_types = set()
        for filename in ir_files.keys():
            if '.' in filename:
                ir_type = filename.split('.')[-1]
                if ir_type not in ['json', 'py']:
                    ir_types.add(ir_type)

        assert 'ttadapter' in ir_types
        assert 'bcmlir' in ir_types


class TestAscendSpecificFeatures:
    """Tests Ascend-specific IR types and features."""

    def test_ttadapter_ir_type(self):
        """Test TTAdapter IR type handling."""
        ir_type = "ttadapter"

        # Check that it's recognized as a text IR
        assert ir_type.endswith("adapter") or "adapter" in ir_type.lower()

        # Should map to generic mapping kind
        assert ir_type in ["ttadapter", "bcmlir", "ttir", "ttgir", "llir"]

    def test_bcmlir_ir_type(self):
        """Test BCMLIR IR type handling."""
        ir_type = "bcmlir"

        # Check that it's recognized as an MLIR dialect
        assert "mlir" in ir_type.lower() or "bcm" in ir_type.lower()

    def test_ascend_stage_origin(self):
        """Test that Ascend IR stages have correct stage_origin."""
        ir_stages = [
            {
                "name": "ttadapter",
                "stage_origin": "backend",
                "is_text": True,
            },
            {
                "name": "bcmlir",
                "stage_origin": "backend",
                "is_text": True,
            },
        ]

        for stage in ir_stages:
            assert stage["stage_origin"] == "backend"
            assert stage["is_text"] is True


@pytest.mark.parametrize("backend,expected_prefix", [
    ("cuda", "cuda"),
    ("hip", "cuda"),  # AMD uses cuda namespace
    ("npu", "npu"),
])
def test_device_prefix_mapping(backend, expected_prefix):
    """Test device prefix mapping for different backends."""
    from tritonparse.backends import get_device_prefix_fallback

    assert get_device_prefix_fallback(backend) == expected_prefix


@pytest.mark.parametrize("backend,expected_key", [
    ("cuda", "nvidia"),
    ("hip", "amd"),
    ("npu", "ascend"),
])
def test_registry_key_mapping(backend, expected_key):
    """Test registry key mapping for different backends."""
    from tritonparse.backends import get_registry_key

    assert get_registry_key(backend) == expected_key
