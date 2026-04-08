from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


GENERIC_TEXT_ARTIFACT_EXTENSIONS = {".json"}


class SourceMappingParser(Protocol):
    def parse(
        self, content: str, other_mappings: list[Any] | None = None
    ) -> dict[str, dict[str, Any]]: ...


@dataclass(frozen=True)
class IRStageDescriptor:
    name: str
    extension: str
    kind: str
    display_name: str
    display_order: int
    is_primary: bool
    is_text: bool
    supports_source_mapping: bool
    parser_id: str
    syntax_id: str


@dataclass(frozen=True)
class DerivedArtifactDescriptor:
    source_extension: str
    output_stage_name: str
    output_extension: str
    tool_name: str


@dataclass(frozen=True)
class AnalysisPassDescriptor:
    name: str
    required_stages: tuple[str, ...]
    optional: bool = True


class GenericLocParser:
    def __init__(self, ir_type: str):
        self.ir_type = ir_type

    def parse(
        self, content: str, other_mappings: list[Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        from tritonparse.parse.ir_parser import (
            extract_code_locations,
            extract_loc_definitions,
        )

        loc_defs = extract_loc_definitions(content)
        loc_refs = extract_code_locations(content)

        mappings: dict[str, dict[str, Any]] = {}
        for ln, loc_id in loc_refs.items():
            if loc_id.startswith("direct:"):
                _, file_path, line, col = loc_id.split(":", 3)
                mappings[str(ln)] = {
                    "file": file_path,
                    "line": int(line),
                    "column": int(col),
                    f"{self.ir_type}_line": ln,
                }
            elif loc_id in loc_defs:
                info = loc_defs[loc_id]
                entry = {
                    "file": info["file"],
                    "line": info["line"],
                    "column": info["column"],
                    f"{self.ir_type}_line": ln,
                }
                if info.get("is_callsite"):
                    entry["is_callsite"] = True
                    entry["callsite_callee"] = info["callsite_callee"]
                    entry["callsite_caller"] = info["callsite_caller"]
                if "alias_name" in info:
                    entry["alias_name"] = info["alias_name"]
                if "alias_of" in info:
                    entry["loc_id"] = loc_id
                mappings[str(ln)] = entry

        for loc_id, info in loc_defs.items():
            if "def_line" not in info:
                continue
            def_ln = info["def_line"]
            if str(def_ln) in mappings:
                continue
            entry = {
                "file": info["file"],
                "line": info["line"],
                "column": info["column"],
                f"{self.ir_type}_line": def_ln,
                "kind": "loc_def",
            }
            if "alias_name" in info:
                entry["alias_name"] = info["alias_name"]
            if "alias_of" in info:
                entry["loc_id"] = loc_id
            mappings[str(def_ln)] = entry

        return mappings


class PtxOrAmdParser:
    def __init__(self, ir_type: str):
        self.ir_type = ir_type

    def parse(
        self, content: str, other_mappings: list[Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        from tritonparse.parse.ir_parser import extract_ptx_amdgcn_mappings

        return extract_ptx_amdgcn_mappings(content, other_mappings, self.ir_type)


class SassParser:
    def parse(
        self, content: str, other_mappings: list[Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        from tritonparse.parse.ir_parser import extract_sass_mappings

        return extract_sass_mappings(content)


class NoOpParser:
    def parse(
        self, content: str, other_mappings: list[Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        return {}


def create_parser_from_id(parser_id: str, stage_name: str) -> SourceMappingParser:
    if parser_id == "generic_loc":
        return GenericLocParser(stage_name)
    if parser_id == "ptx_loc":
        return PtxOrAmdParser("ptx")
    if parser_id == "amdgcn_loc":
        return PtxOrAmdParser("amdgcn")
    if parser_id == "sass_loc":
        return SassParser()
    return NoOpParser()


class TritonParseBackendAdapter(ABC):
    @property
    @abstractmethod
    def adapter_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def runtime_backend(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def device_prefix(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def display_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_ir_stages(self) -> list[IRStageDescriptor]:
        raise NotImplementedError

    def get_stage(self, stage_name: str) -> IRStageDescriptor:
        for stage in self.get_ir_stages():
            if stage.name == stage_name:
                return stage
        raise KeyError(f"Unknown stage '{stage_name}' for adapter {self.adapter_name}")

    def classify_artifact(self, artifact_name: str) -> IRStageDescriptor | None:
        artifact_suffix = Path(artifact_name).suffix
        for stage in self.get_ir_stages():
            if stage.extension == artifact_suffix:
                return stage
        return None

    def get_derived_artifacts(self) -> list[DerivedArtifactDescriptor]:
        return []

    def collect_derived_artifact_contents(
        self,
        metadata_group: dict[str, str],
    ) -> dict[str, str]:
        return {}

    def create_source_mapping_parser(self, stage: IRStageDescriptor) -> SourceMappingParser:
        return create_parser_from_id(stage.parser_id, stage.name)

    def get_analysis_passes(self) -> list[AnalysisPassDescriptor]:
        return []

    def run_analysis_pass(self, pass_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    @abstractmethod
    def normalize_device_string(self, device: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_synchronize_snippet(self) -> str:
        raise NotImplementedError


def _stage(
    name: str,
    extension: str,
    kind: str,
    display_name: str,
    display_order: int,
    *,
    is_primary: bool = False,
    is_text: bool = True,
    supports_source_mapping: bool = True,
    parser_id: str = "generic_loc",
    syntax_id: str = "text",
) -> IRStageDescriptor:
    return IRStageDescriptor(
        name=name,
        extension=extension,
        kind=kind,
        display_name=display_name,
        display_order=display_order,
        is_primary=is_primary,
        is_text=is_text,
        supports_source_mapping=supports_source_mapping,
        parser_id=parser_id,
        syntax_id=syntax_id,
    )


class NvidiaBackendAdapter(TritonParseBackendAdapter):
    _stages = [
        _stage("ttir", ".ttir", "ir", "TTIR", 10, is_primary=True, syntax_id="mlir"),
        _stage("ttgir", ".ttgir", "ir", "TTGIR", 20, syntax_id="mlir"),
        _stage("llir", ".llir", "ir", "LLIR", 30, syntax_id="llvm"),
        _stage("ptx", ".ptx", "asm", "PTX", 40, parser_id="ptx_loc", syntax_id="ptx"),
        _stage(
            "cubin",
            ".cubin",
            "binary",
            "CUBIN",
            50,
            is_text=False,
            supports_source_mapping=False,
            parser_id="none",
            syntax_id="plaintext",
        ),
        _stage("sass", ".sass", "asm", "SASS", 60, parser_id="sass_loc", syntax_id="asm"),
    ]

    @property
    def adapter_name(self) -> str:
        return "nvidia"

    @property
    def runtime_backend(self) -> str:
        return "cuda"

    @property
    def device_prefix(self) -> str:
        return "cuda"

    @property
    def display_name(self) -> str:
        return "NVIDIA"

    def get_ir_stages(self) -> list[IRStageDescriptor]:
        return list(self._stages)

    def get_derived_artifacts(self) -> list[DerivedArtifactDescriptor]:
        return [
            DerivedArtifactDescriptor(
                source_extension=".cubin",
                output_stage_name="sass",
                output_extension=".sass",
                tool_name="nvdisasm",
            )
        ]

    def collect_derived_artifact_contents(
        self,
        metadata_group: dict[str, str],
    ) -> dict[str, str]:
        cubin_path = None
        cubin_name = None
        for artifact_name, artifact_path in metadata_group.items():
            if artifact_name.endswith(".cubin"):
                cubin_name = artifact_name
                cubin_path = artifact_path
                break
        if not cubin_path or not cubin_name:
            return {}

        from tritonparse.tools.disasm import extract

        sass_filename = f"{Path(cubin_name).stem}.sass"
        sass_content = extract(cubin_path)
        if not isinstance(sass_content, str):
            return {}
        return {sass_filename: sass_content}

    def normalize_device_string(self, device: str) -> str:
        if isinstance(device, str) and device.startswith("cuda"):
            return "cuda:0"
        return device

    def build_synchronize_snippet(self) -> str:
        return "torch.cuda.synchronize()"


class AmdBackendAdapter(TritonParseBackendAdapter):
    _stages = [
        _stage("ttir", ".ttir", "ir", "TTIR", 10, is_primary=True, syntax_id="mlir"),
        _stage("ttgir", ".ttgir", "ir", "TTGIR", 20, syntax_id="mlir"),
        _stage("llir", ".llir", "ir", "LLIR", 30, syntax_id="llvm"),
        _stage(
            "amdgcn",
            ".amdgcn",
            "asm",
            "AMDGCN",
            40,
            parser_id="amdgcn_loc",
            syntax_id="asm",
        ),
    ]

    @property
    def adapter_name(self) -> str:
        return "amd"

    @property
    def runtime_backend(self) -> str:
        return "hip"

    @property
    def device_prefix(self) -> str:
        return "cuda"

    @property
    def display_name(self) -> str:
        return "AMD"

    def get_ir_stages(self) -> list[IRStageDescriptor]:
        return list(self._stages)

    def normalize_device_string(self, device: str) -> str:
        if isinstance(device, str) and device.startswith("cuda"):
            return "cuda:0"
        return device

    def build_synchronize_snippet(self) -> str:
        return "torch.cuda.synchronize()"


class AscendBackendAdapter(TritonParseBackendAdapter):
    _stages = [
        _stage("ttir", ".ttir", "ir", "TTIR", 10, is_primary=True, syntax_id="mlir"),
        _stage("ttgir", ".ttgir", "ir", "TTGIR", 20, syntax_id="mlir"),
        _stage("llir", ".llir", "ir", "LLIR", 30, syntax_id="llvm"),
    ]

    @property
    def adapter_name(self) -> str:
        return "ascend"

    @property
    def runtime_backend(self) -> str:
        return "npu"

    @property
    def device_prefix(self) -> str:
        return "npu"

    @property
    def display_name(self) -> str:
        return "Ascend"

    def get_ir_stages(self) -> list[IRStageDescriptor]:
        return list(self._stages)

    def normalize_device_string(self, device: str) -> str:
        if isinstance(device, str) and device.startswith("npu") and ":" not in device:
            return "npu:0"
        return device

    def build_synchronize_snippet(self) -> str:
        return "torch.npu.synchronize()"


class BackendAdapterRegistry:
    def __init__(self) -> None:
        self._adapter_types: dict[str, type[TritonParseBackendAdapter]] = {}

    def register(self, adapter_cls: type[TritonParseBackendAdapter]) -> None:
        adapter = adapter_cls()
        self._adapter_types[adapter.adapter_name] = adapter_cls

    def create_all(self) -> list[TritonParseBackendAdapter]:
        return [adapter_cls() for adapter_cls in self._adapter_types.values()]

    def resolve(self, adapter_name: str | None = None) -> TritonParseBackendAdapter:
        resolved_name = (adapter_name or "nvidia").lower()
        adapter_cls = self._adapter_types.get(resolved_name)
        if adapter_cls is None:
            raise ValueError(
                "Unable to resolve backend adapter from adapter_name: "
                f"adapter_name={adapter_name!r}"
            )
        return adapter_cls()

    def resolve_from_runtime(
        self,
        *,
        adapter_name: str | None = None,
    ) -> TritonParseBackendAdapter:
        return self.resolve(adapter_name)

    def resolve_from_trace(
        self,
        *,
        compilation_metadata: dict[str, Any],
        launch_metadata: dict[str, Any] | None = None,
    ) -> TritonParseBackendAdapter:
        metadata = dict(compilation_metadata or {})
        if launch_metadata:
            metadata = {**launch_metadata, **metadata}

        return self.resolve(metadata.get("adapter_name"))

    @property
    def known_stage_extensions(self) -> set[str]:
        extensions: set[str] = set()
        for adapter in self.create_all():
            for stage in adapter.get_ir_stages():
                extensions.add(stage.extension)
        return extensions


def build_backend_trace_contract(adapter: TritonParseBackendAdapter) -> dict[str, Any]:
    return {"adapter_name": adapter.adapter_name}


def build_stage_descriptor_trace_contract(
    adapter: TritonParseBackendAdapter,
    artifact_names: list[str],
) -> list[dict[str, Any]]:
    stage_descriptors: list[dict[str, Any]] = []
    for artifact_name in sorted(artifact_names):
        stage = adapter.classify_artifact(artifact_name)
        if stage is None:
            continue
        serialized_stage = asdict(stage)
        serialized_stage["file_name"] = artifact_name
        stage_descriptors.append(serialized_stage)
    return stage_descriptors


def _deserialize_stage_descriptor(raw_stage: dict[str, Any]) -> IRStageDescriptor:
    return IRStageDescriptor(
        name=str(raw_stage["name"]),
        extension=str(raw_stage["extension"]),
        kind=str(raw_stage.get("kind", "ir")),
        display_name=str(
            raw_stage.get("display_name")
            or raw_stage.get("displayName")
            or raw_stage["name"]
        ),
        display_order=int(raw_stage.get("display_order") or raw_stage.get("displayOrder") or 0),
        is_primary=bool(raw_stage.get("is_primary") or raw_stage.get("isPrimary")),
        is_text=bool(raw_stage.get("is_text", raw_stage.get("isText", True))),
        supports_source_mapping=bool(
            raw_stage.get(
                "supports_source_mapping",
                raw_stage.get("supportsSourceMapping", True),
            )
        ),
        parser_id=str(raw_stage.get("parser_id") or raw_stage.get("parserId") or "none"),
        syntax_id=str(raw_stage.get("syntax_id") or raw_stage.get("syntaxId") or "plaintext"),
    )


def get_present_ir_stages(
    adapter: TritonParseBackendAdapter,
    artifact_names: list[str],
    *,
    source_mapping_only: bool = False,
) -> list[IRStageDescriptor]:
    present_stage_names = {
        stage.name
        for artifact_name in artifact_names
        if (stage := adapter.classify_artifact(artifact_name)) is not None
    }
    stages = [stage for stage in adapter.get_ir_stages() if stage.name in present_stage_names]
    if source_mapping_only:
        stages = [stage for stage in stages if stage.supports_source_mapping]
    return stages


def get_present_stage_descriptors_from_event(
    event: dict[str, Any], *, source_mapping_only: bool = False
) -> list[IRStageDescriptor]:
    payload = event.get("payload", {})
    metadata = payload.get("metadata", {})
    launch_metadata = event.get("compilation_metadata", {})
    serialized_stage_descriptors = metadata.get("stage_descriptors") or launch_metadata.get(
        "stage_descriptors"
    )
    if isinstance(serialized_stage_descriptors, list) and serialized_stage_descriptors:
        stages = [
            _deserialize_stage_descriptor(raw_stage)
            for raw_stage in serialized_stage_descriptors
            if isinstance(raw_stage, dict)
        ]
        if source_mapping_only:
            stages = [stage for stage in stages if stage.supports_source_mapping]
        return sorted(stages, key=lambda stage: stage.display_order)

    artifact_names = list((payload.get("file_content") or {}).keys()) + list(
        (payload.get("file_path") or {}).keys()
    )
    registry = get_backend_registry()
    adapter_name = metadata.get("adapter_name") or launch_metadata.get("adapter_name")
    if adapter_name is not None:
        adapter = registry.resolve(str(adapter_name))
        return get_present_ir_stages(
            adapter,
            artifact_names,
            source_mapping_only=source_mapping_only,
        )
    else:
        present_extensions = {Path(artifact_name).suffix for artifact_name in artifact_names}
        present_direct_fields = {
            field_name
            for field_name, field_value in payload.items()
            if isinstance(field_value, str) and field_value
        }
        descriptors: list[IRStageDescriptor] = []
        seen_stage_names: set[str] = set()
        for adapter in registry.create_all():
            for stage in adapter.get_ir_stages():
                if stage.name in seen_stage_names:
                    continue
                is_present = (
                    stage.extension in present_extensions or stage.name in present_direct_fields
                )
                if not is_present:
                    continue
                if source_mapping_only and not stage.supports_source_mapping:
                    continue
                descriptors.append(stage)
                seen_stage_names.add(stage.name)
        return sorted(descriptors, key=lambda stage: stage.display_order)


def get_stage_names_from_event(
    event: dict[str, Any], *, source_mapping_only: bool = False
) -> list[str]:
    stages = get_present_stage_descriptors_from_event(
        event,
        source_mapping_only=source_mapping_only,
    )
    return [stage.name for stage in stages if stage.is_text or stage.supports_source_mapping]


def get_default_ir_types(
    *events: dict[str, Any],
    source_mapping_only: bool = False,
) -> list[str]:
    discovered: list[str] = []
    for event in events:
        for stage_name in get_stage_names_from_event(
            event, source_mapping_only=source_mapping_only
        ):
            if stage_name not in discovered:
                discovered.append(stage_name)
    return discovered


_REGISTRY = BackendAdapterRegistry()
for _adapter_cls in (
    NvidiaBackendAdapter,
    AmdBackendAdapter,
    AscendBackendAdapter,
):
    _REGISTRY.register(_adapter_cls)


def get_backend_registry() -> BackendAdapterRegistry:
    return _REGISTRY