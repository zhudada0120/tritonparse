from __future__ import annotations

from typing import Any, Dict


_PATCH_FLAG = "__tritonparse_ascend_hook_patch__"


def collect_runtime_option_fields(options: Any) -> Dict[str, Any]:
    """Collect runtime option fields without assuming backend-specific attributes."""
    return {
        "num_warps": getattr(options, "num_warps", None),
        "num_ctas": getattr(options, "num_ctas", None),
        "num_stages": getattr(options, "num_stages", None),
        "enable_fp_fusion": getattr(options, "enable_fp_fusion", None),
        "launch_cooperative_grid": getattr(
            options, "launch_cooperative_grid", False
        ),
        "extern_libs": getattr(options, "extern_libs", None),
    }


def patch_triton_jit_hook_compatibility() -> bool:
    """Patch Triton JIT hook formatting to tolerate backend-specific option objects."""
    try:
        from triton.runtime import jit as jit_module
    except Exception:
        return False

    original_call_hook = jit_module.JITFunction._call_hook
    if getattr(original_call_hook, _PATCH_FLAG, False):
        return False

    def _patched_call_hook(
        self,
        hook,
        key,
        signature,
        device,
        constants,
        options,
        configs,
        is_warmup,
    ):
        if not hook:
            return None

        option_fields = collect_runtime_option_fields(options)
        name = self.fn.__qualname__
        module = self.fn.__module__
        arg_reprs = ", ".join(
            [f"{param.name}: {ty}" for param, ty in zip(self.params, key[1])]
        )
        repr = (
            f"{name}[num_warps={option_fields['num_warps']}, "
            f"num_ctas={option_fields['num_ctas']}, "
            f"num_stages={option_fields['num_stages']}, "
            f"enable_fp_fusion={option_fields['enable_fp_fusion']}, "
            f"launch_cooperative_grid={option_fields['launch_cooperative_grid']}]"
            f"({arg_reprs})"
        )

        specialization_data = None
        if configs:
            try:
                full_name = jit_module.get_full_name(self.fn)
                specialization_data = jit_module.serialize_specialization_data(
                    full_name,
                    signature,
                    constants,
                    configs[0],
                    options,
                    key,
                )
            except Exception:
                specialization_data = None

        compile_payload = {
            "key": key,
            "signature": signature,
            "device": device,
            "constants": constants,
            "num_warps": option_fields["num_warps"],
            "num_ctas": option_fields["num_ctas"],
            "num_stages": option_fields["num_stages"],
            "enable_fp_fusion": option_fields["enable_fp_fusion"],
            "launch_cooperative_grid": option_fields[
                "launch_cooperative_grid"
            ],
            "extern_libs": option_fields["extern_libs"],
            "configs": configs,
            "specialization_data": specialization_data,
            "is_warmup": is_warmup,
        }

        return hook(
            key=key,
            repr=repr,
            fn=jit_module.JitFunctionInfo(module, name, self),
            compile=compile_payload,
            is_manual_warmup=is_warmup,
            already_compiled=False,
        )

    setattr(_patched_call_hook, _PATCH_FLAG, True)
    jit_module.JITFunction._call_hook = _patched_call_hook
    return True