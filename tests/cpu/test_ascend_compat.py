import tempfile
from pathlib import Path

from tritonparse.ascend.compat import (
    collect_runtime_option_fields,
)
from tritonparse.structured_logging import extract_file_content


class DummyNPUOptions:
    num_warps = 4
    num_ctas = 1
    num_stages = 2
    enable_fp_fusion = True


def test_collect_runtime_option_fields_tolerates_missing_launch_flag():
    fields = collect_runtime_option_fields(DummyNPUOptions())

    assert fields["num_warps"] == 4
    assert fields["launch_cooperative_grid"] is False
    assert fields["extern_libs"] is None
def test_extract_file_content_includes_selected_ascend_ir_text_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        ttir_path = temp_path / "triton_add.ttir"
        ttadapter_path = temp_path / "triton_add.ttadapter"
        bcmlir_path = temp_path / "triton_add.bcmlir"
        json_path = temp_path / "triton_add.json"
        source_path = temp_path / "triton_add.source"
        mlirbc_path = temp_path / "triton_add.mlirbc"
        ttadapter_path.write_text("module { tt.func @kernel() }", encoding="utf-8")
        ttir_path.write_text("module { tt.func @kernel_ttir() }", encoding="utf-8")
        bcmlir_path.write_text("module { func.func @kernel_bcmlir() }", encoding="utf-8")
        json_path.write_text('{"name": "triton_add"}', encoding="utf-8")
        source_path.write_text("kernel source", encoding="utf-8")
        mlirbc_path.write_bytes(b"\xDE\xAD\xBE\xEF")

        npubin_path = temp_path / "triton_add.npubin"
        npubin_path.write_bytes(b"\x00\x01\x02\x03")

        trace_data = {"file_path": {}, "file_content": {}}
        metadata_group = {
            "triton_add.ttir": str(ttir_path),
            "triton_add.ttadapter": str(ttadapter_path),
            "triton_add.bcmlir": str(bcmlir_path),
            "triton_add.json": str(json_path),
            "triton_add.source": str(source_path),
            "triton_add.mlirbc": str(mlirbc_path),
            "triton_add.npubin": str(npubin_path),
        }

        extract_file_content(trace_data, metadata_group)

        assert trace_data["file_content"]["triton_add.ttir"] == "module { tt.func @kernel_ttir() }"
        assert trace_data["file_path"]["triton_add.ttadapter"] == str(ttadapter_path)
        assert trace_data["file_content"]["triton_add.ttadapter"] == "module { tt.func @kernel() }"
        assert trace_data["file_content"]["triton_add.bcmlir"] == "module { func.func @kernel_bcmlir() }"
        assert trace_data["file_content"]["triton_add.json"] == '{"name": "triton_add"}'
        assert "triton_add.source" not in trace_data["file_content"]
        assert "triton_add.mlirbc" not in trace_data["file_content"]
        assert "triton_add.npubin" not in trace_data["file_content"]