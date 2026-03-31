#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Backend metadata extraction and device prefix management for TritonParse.

import logging
from dataclasses import dataclass
from typing import Dict, Optional

log = logging.getLogger(__name__)


@dataclass
class BackendInfo:
    """Backend metadata extracted from trace files (target_backend, device_prefix, registry_key)."""
    target_backend: str
    device_prefix: str
    registry_key: Optional[str] = None


def extract_backend_from_trace(trace_data: Dict) -> BackendInfo:
    """
    Extract backend information from trace event metadata.

    Args:
        trace_data: Trace event dictionary from .ndjson file

    Returns:
        BackendInfo object containing backend metadata

    Raises:
        ValueError: If trace data is missing required backend metadata fields
    """
    # Try multiple locations for compilation metadata (support both launch and compilation events)
    compilation_metadata = trace_data.get("compilation_metadata", {})
    if not compilation_metadata:
        payload = trace_data.get("payload", {})
        compilation_metadata = payload.get("compilation_metadata", {})

    device_prefix = compilation_metadata.get("device_prefix")
    target_backend = compilation_metadata.get("target_backend")
    registry_key = compilation_metadata.get("registry_key")

    if not target_backend or not device_prefix:
        raise ValueError(
            f"Trace file is missing required backend metadata.\n"
            f"Required fields: device_prefix, target_backend\n"
            f"Found in compilation_metadata: {list(compilation_metadata.keys())}\n"
            f"Please regenerate the trace file using a newer version of tritonparse "
            f"that supports backend metadata."
        )

    return BackendInfo(
        target_backend=target_backend,
        device_prefix=device_prefix,
        registry_key=registry_key
    )


def get_device_prefix_fallback(backend: str) -> str:
    """
    Fallback mapping from backend name to PyTorch device prefix (used when PyTorch unavailable).

    Args:
        backend: Triton backend name (e.g., "cuda", "hip", "npu")

    Returns:
        PyTorch device namespace prefix
    """
    mapping = {
        "cuda": "cuda",   # NVIDIA
        "hip": "cuda",    # AMD ROCm
        "npu": "npu",     # Ascend
    }

    return mapping.get(backend, backend)


def get_registry_key(target_backend: str) -> str:
    """
    Map target backend name to Triton backend registry key for backends lookup.

    Args:
        target_backend: Compilation target backend name

    Returns:
        Backend registry key for use with triton.backends.backends
    """
    mapping = {
        "cuda": "nvidia",
        "hip": "amd",
        "npu": "ascend",
    }

    return mapping.get(target_backend, target_backend)
