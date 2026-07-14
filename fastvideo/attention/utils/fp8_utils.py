"""FP8 (e4m3) quantization helpers for the FA4 attention path.

Torch-only module (no flash_attn import) so it stays importable — and unit
testable — on CPU-only hosts. The GPU kernel entry point lives in
``flash_attn_cute.flash_attn_fp8_func``; the FLASH_ATTN backend combines
the two.
"""

from __future__ import annotations

import torch

# torch.finfo(torch.float8_e4m3fn).max — hardcoded so the module stays
# importable on builds without fp8 dtypes.
FP8_E4M3_MAX = 448.0


def fp8_quantize_for_fa4(tensor_4d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(batch, seqlen, nheads, headdim)`` activations to fp8
    e4m3 with per-(batch, head) scaling.

    Returns ``(fp8 tensor, float32 descale)`` where the descale has shape
    ``(batch, nheads)`` — FA4's q_descale/k_descale/v_descale contract
    (the kernel multiplies scores by the descales to undo the scaling).
    """
    if tensor_4d.ndim != 4:
        raise ValueError(f"expected (batch, seqlen, nheads, headdim), got {tuple(tensor_4d.shape)}")
    amax = tensor_4d.abs().amax(dim=(1, 3)).to(torch.float32).clamp(min=1e-6)
    scale = (FP8_E4M3_MAX / amax).to(tensor_4d.dtype)
    scaled = (tensor_4d * scale[:, None, :, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return scaled.to(torch.float8_e4m3fn), amax / FP8_E4M3_MAX


def fa4_fp8_stage_active(stage1_enabled: bool, stage2_enabled: bool, stage_profile: str) -> bool:
    """Which per-stage FP8 flag governs the current forward.

    LTX-2's denoising stages tag the forward context with
    ``ltx2_fp4_stage_profile`` = ``base`` (stage 1) / ``refine`` (stage 2).
    Pipelines that don't tag default to ``refine`` upstream, so for them the
    stage-2 flag governs.
    """
    if stage_profile == "refine":
        return stage2_enabled
    return stage1_enabled
