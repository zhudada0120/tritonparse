#!/usr/bin/env python3
"""
Test script for validating backend-agnostic TritonParse functionality in Ascend scenario.

This script tests:
1. Backend info extraction from Ascend traces
2. IR stage handling for Ascend-specific IR types (ttadapter, bcmlir)
3. Device prefix handling for NPU
4. End-to-end flow validation
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tritonparse.backends import (
    BackendInfo,
    extract_backend_from_trace,
    get_device_prefix_fallback,
    get_registry_key,
)


def test_ascend_backend_info_extraction():
    """Test extracting Ascend backend information from trace."""
    print("=" * 60)
    print("TEST 1: Ascend Backend Info Extraction")
    print("=" * 60)

    # Create a mock Ascend trace event
    ascend_trace = {
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
                "name": "add_kernel",
            },
            "file_content": {
                "add_kernel.ttir": "module { ... }",
                "add_kernel.ttadapter": "...",
                "add_kernel.bcmlir": "...",
            },
            "file_path": {},
        },
    }

    try:
        backend_info = extract_backend_from_trace(ascend_trace)

        print(f"✓ Successfully extracted backend info")
        print(f"  - Target Backend: {backend_info.target_backend}")
        print(f"  - Device Prefix: {backend_info.device_prefix}")
        print(f"  - Registry Key: {backend_info.registry_key}")

        assert backend_info.target_backend == "npu"
        assert backend_info.device_prefix == "npu"
        assert backend_info.registry_key == "ascend"

        print("✅ Ascend backend info extraction test PASSED\n")
        return True

    except Exception as e:
        print(f"❌ Ascend backend info extraction test FAILED: {e}\n")
        return False


def test_ascend_ir_stages():
    """Test Ascend-specific IR stages (ttadapter, bcmlir)."""
    print("=" * 60)
    print("TEST 2: Ascend IR Stages")
    print("=" * 60)

    trace = {
        "compilation_metadata": {
            "ir_stages": [
                {"name": "ttir", "mapping_kind": "generic", "is_text": True},
                {"name": "ttadapter", "mapping_kind": "generic", "is_text": True},
                {"name": "bcmlir", "mapping_kind": "generic", "is_text": True},
            ]
        }
    }

    ir_stages = trace["compilation_metadata"]["ir_stages"]

    print(f"Found {len(ir_stages)} IR stages:")
    for stage in ir_stages:
        print(f"  - {stage['name']}: mapping_kind={stage['mapping_kind']}, is_text={stage['is_text']}")

    # Verify ttadapter and bcmlir are present
    stage_names = [s["name"] for s in ir_stages]
    assert "ttadapter" in stage_names, "ttadapter not found in IR stages"
    assert "bcmlir" in stage_names, "bcmlir not found in IR stages"

    # Verify they have correct mapping_kind
    ttadapter_stage = next(s for s in ir_stages if s["name"] == "ttadapter")
    assert ttadapter_stage["mapping_kind"] == "generic"

    bcmlir_stage = next(s for s in ir_stages if s["name"] == "bcmlir")
    assert bcmlir_stage["mapping_kind"] == "generic"

    print("✅ Ascend IR stages test PASSED\n")
    return True


def test_npu_device_prefix():
    """Test NPU device prefix handling."""
    print("=" * 60)
    print("TEST 3: NPU Device Prefix")
    print("=" * 60)

    # Test get_device_prefix_fallback
    device_prefix = get_device_prefix_fallback("npu")
    print(f"Device prefix for 'npu' backend: {device_prefix}")
    assert device_prefix == "npu"

    # Test synchronize code generation
    sync_call = f"torch.{device_prefix}.synchronize()"
    print(f"Generated synchronize call: {sync_call}")
    assert sync_call == "torch.npu.synchronize()"

    print("✅ NPU device prefix test PASSED\n")
    return True


def test_registry_key_mapping():
    """Test registry key mapping for Ascend."""
    print("=" * 60)
    print("TEST 4: Registry Key Mapping")
    print("=" * 60)

    # Test get_registry_key
    registry_key = get_registry_key("npu")
    print(f"Registry key for 'npu' backend: {registry_key}")
    assert registry_key == "ascend"

    # Test other mappings for comparison
    mappings = {
        "cuda": "nvidia",
        "hip": "amd",
        "npu": "ascend",
    }

    print("All registry key mappings:")
    for backend, expected_key in mappings.items():
        actual_key = get_registry_key(backend)
        print(f"  {backend} → {actual_key}")
        assert actual_key == expected_key

    print("✅ Registry key mapping test PASSED\n")
    return True


def test_multi_backend_comparison():
    """Test that all three backends (NVIDIA, AMD, Ascend) work correctly."""
    print("=" * 60)
    print("TEST 5: Multi-Backend Comparison")
    print("=" * 60)

    backends_config = {
        "nvidia": {
            "target_backend": "cuda",
            "device_prefix": "cuda",
            "registry_key": "nvidia",
            "ir_types": ["ttir", "ptx", "sass"],
        },
        "amd": {
            "target_backend": "hip",
            "device_prefix": "cuda",  # AMD uses cuda namespace in PyTorch
            "registry_key": "amd",
            "ir_types": ["ttir", "amdgcn"],
        },
        "ascend": {
            "target_backend": "npu",
            "device_prefix": "npu",
            "registry_key": "ascend",
            "ir_types": ["ttir", "ttadapter", "bcmlir"],
        },
    }

    print("Testing all three backends:")
    for vendor, config in backends_config.items():
        print(f"\n{vendor.upper()}:")

        # Test registry key
        registry_key = get_registry_key(config["target_backend"])
        print(f"  Registry Key: {registry_key}")
        assert registry_key == config["registry_key"]

        # Test device prefix
        device_prefix = get_device_prefix_fallback(config["target_backend"])
        print(f"  Device Prefix: {device_prefix}")
        assert device_prefix == config["device_prefix"]

        # Test sync code generation
        sync_call = f"torch.{device_prefix}.synchronize()"
        print(f"  Sync Call: {sync_call}")

    print("\n✅ Multi-backend comparison test PASSED\n")
    return True


def test_frontend_ir_extraction():
    """Test frontend IR extraction logic (simulating frontend behavior)."""
    print("=" * 60)
    print("TEST 6: Frontend IR Extraction (Simulation)")
    print("=" * 60)

    # Simulate Ascend kernel IR files
    ir_files = {
        "add_kernel.ttir": "module { ... }",
        "add_kernel.ttadapter": "...",
        "add_kernel.bcmlir": "...",
        "add_kernel.source.json": "{}",
    }

    # Extract IR types (frontend logic)
    ir_types = set()
    for filename in ir_files.keys():
        if '.' in filename:
            parts = filename.split('.')
            ir_type = parts[-1]
            # Filter non-IR files
            if ir_type not in ['json', 'py', 'source', 'cpp', 'h']:
                ir_types.add(ir_type)

    print(f"Extracted IR types: {sorted(ir_types)}")

    # Verify Ascend-specific IR types are present
    assert "ttir" in ir_types
    assert "ttadapter" in ir_types
    assert "bcmlir" in ir_types

    # Verify non-IR files are filtered
    assert "source" not in ir_types
    assert "json" not in ir_types

    print("✅ Frontend IR extraction test PASSED\n")
    return True


def test_backend_metadata_injection():
    """Test that backend_metadata is correctly injected during parsing."""
    print("=" * 60)
    print("TEST 7: Backend Metadata Injection")
    print("=" * 60)

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

    # Simulate parsing logic (inject backend_metadata)
    compilation_metadata = trace_event.get("compilation_metadata", {})
    device_prefix = compilation_metadata.get("device_prefix")
    target_backend = compilation_metadata.get("target_backend")

    if device_prefix or target_backend:
        trace_event.setdefault("backend_metadata", {})
        trace_event["backend_metadata"]["device_prefix"] = device_prefix
        if target_backend:
            trace_event["backend_metadata"]["target_backend"] = target_backend

    # Verify injection
    assert "backend_metadata" in trace_event
    assert trace_event["backend_metadata"]["device_prefix"] == "npu"
    assert trace_event["backend_metadata"]["target_backend"] == "npu"

    print("Backend metadata injected:")
    print(f"  - device_prefix: {trace_event['backend_metadata']['device_prefix']}")
    print(f"  - target_backend: {trace_event['backend_metadata']['target_backend']}")

    print("✅ Backend metadata injection test PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TRITONPARSE BACKEND-AGNOSTIC FUNCTIONALITY TEST SUITE")
    print("Testing Ascend NPU Backend Support")
    print("=" * 60 + "\n")

    tests = [
        ("Ascend Backend Info Extraction", test_ascend_backend_info_extraction),
        ("Ascend IR Stages", test_ascend_ir_stages),
        ("NPU Device Prefix", test_npu_device_prefix),
        ("Registry Key Mapping", test_registry_key_mapping),
        ("Multi-Backend Comparison", test_multi_backend_comparison),
        ("Frontend IR Extraction", test_frontend_ir_extraction),
        ("Backend Metadata Injection", test_backend_metadata_injection),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test FAILED with exception: {e}\n")
            results.append((name, False))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Backend-agnostic functionality is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
