#  Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Unit tests for the backends module.

Tests the backend information extraction and management utilities.
"""

import pytest

from tritonparse.backends import (
    BackendInfo,
    extract_backend_from_trace,
    get_device_prefix_fallback,
    get_registry_key,
)


class TestGetRegistryKey:
    """Tests for get_registry_key function."""

    def test_nvidia_backend(self):
        """Test registry key mapping for NVIDIA CUDA backend."""
        assert get_registry_key("cuda") == "nvidia"

    def test_amd_backend(self):
        """Test registry key mapping for AMD ROCm backend."""
        assert get_registry_key("hip") == "amd"

    def test_ascend_backend(self):
        """Test registry key mapping for Ascend NPU backend."""
        assert get_registry_key("npu") == "ascend"

    def test_unknown_backend(self):
        """Test registry key mapping for unknown backends."""
        # Unknown backends should map to themselves
        assert get_registry_key("xpu") == "xpu"
        assert get_registry_key("custom") == "custom"


class TestGetDevicePrefixFallback:
    """Tests for get_device_prefix_fallback function."""

    def test_cuda_backend(self):
        """Test device prefix for NVIDIA CUDA backend."""
        assert get_device_prefix_fallback("cuda") == "cuda"

    def test_hip_backend(self):
        """Test device prefix for AMD ROCm backend (special case)."""
        # HIP backend uses cuda namespace in PyTorch
        assert get_device_prefix_fallback("hip") == "cuda"

    def test_npu_backend(self):
        """Test device prefix for Ascend NPU backend."""
        assert get_device_prefix_fallback("npu") == "npu"

    def test_unknown_backend(self):
        """Test device prefix for unknown backends."""
        # Unknown backends should default to backend name
        assert get_device_prefix_fallback("xpu") == "xpu"


class TestExtractBackendFromTrace:
    """Tests for extract_backend_from_trace function."""

    def test_extract_nvidia_backend(self):
        """Test extraction of NVIDIA backend metadata."""
        trace_data = {
            "compilation_metadata": {
                "registry_key": "nvidia",
                "target_backend": "cuda",
                "device_prefix": "cuda",
                "target": {
                    "backend": "cuda",
                    "arch": 80,
                    "warp_size": 32,
                },
            }
        }

        info = extract_backend_from_trace(trace_data)

        assert isinstance(info, BackendInfo)
        assert info.target_backend == "cuda"
        assert info.device_prefix == "cuda"
        assert info.registry_key == "nvidia"

    def test_extract_ascend_backend(self):
        """Test extraction of Ascend backend metadata."""
        trace_data = {
            "compilation_metadata": {
                "registry_key": "ascend",
                "target_backend": "npu",
                "device_prefix": "npu",
                "target": {
                    "backend": "npu",
                    "arch": 910,
                    "warp_size": 16,
                },
            }
        }

        info = extract_backend_from_trace(trace_data)

        assert info.target_backend == "npu"
        assert info.device_prefix == "npu"
        assert info.registry_key == "ascend"

    def test_extract_amd_backend(self):
        """Test extraction of AMD backend metadata."""
        trace_data = {
            "compilation_metadata": {
                "registry_key": "amd",
                "target_backend": "hip",
                "device_prefix": "cuda",  # AMD uses cuda namespace in PyTorch
                "target": {
                    "backend": "hip",
                    "arch": 908,
                    "warp_size": 64,
                },
            }
        }

        info = extract_backend_from_trace(trace_data)

        assert info.target_backend == "hip"
        assert info.device_prefix == "cuda"
        assert info.registry_key == "amd"

    def test_missing_required_fields_raises_error(self):
        """Test that missing required fields raises ValueError."""
        # Missing device_prefix
        trace_data = {
            "compilation_metadata": {
                "registry_key": "nvidia",
                "target_backend": "cuda",
                # device_prefix is missing
            }
        }

        with pytest.raises(ValueError) as exc_info:
            extract_backend_from_trace(trace_data)

        assert "missing required backend metadata" in str(exc_info.value).lower()

    def test_missing_compilation_metadata_raises_error(self):
        """Test that missing compilation_metadata raises ValueError."""
        trace_data = {}  # No compilation_metadata

        with pytest.raises(ValueError) as exc_info:
            extract_backend_from_trace(trace_data)

        assert "missing required backend metadata" in str(exc_info.value).lower()


class TestBackendInfoDataclass:
    """Tests for BackendInfo dataclass."""

    def test_create_backend_info(self):
        """Test creating a BackendInfo object."""
        info = BackendInfo(
            target_backend="cuda",
            device_prefix="cuda",
            registry_key="nvidia"
        )

        assert info.target_backend == "cuda"
        assert info.device_prefix == "cuda"
        assert info.registry_key == "nvidia"

    def test_backend_info_with_optional_registry_key(self):
        """Test creating BackendInfo with optional registry_key."""
        info = BackendInfo(
            target_backend="cuda",
            device_prefix="cuda"
        )

        assert info.registry_key is None
