"""CPU tests for the FA4 FP8 attention helpers (quantization math + the
per-stage gating rule). The GPU kernel itself (flash_attn_fp8_func) needs
sm100 and is exercised by the B200 A/B runs, not here."""

import pytest
import torch

from fastvideo.attention.utils.fp8_utils import (
    FP8_E4M3_MAX,
    fa4_fp8_stage_active,
    fp8_quantize_for_fa4,
)


def test_fp8_quantize_shapes_and_dtypes():
    x = torch.randn(2, 37, 4, 16, dtype=torch.bfloat16)
    x_fp8, descale = fp8_quantize_for_fa4(x)
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert x_fp8.shape == x.shape
    # FA4 contract: float32 (batch, nheads) descales.
    assert descale.dtype == torch.float32
    assert descale.shape == (2, 4)


def test_fp8_quantize_roundtrip_error_bounded():
    torch.manual_seed(0)
    # Mix of scales across heads to exercise the per-head amax.
    x = torch.randn(1, 128, 4, 32, dtype=torch.bfloat16)
    x[:, :, 1] *= 50.0
    x[:, :, 2] *= 0.02
    x_fp8, descale = fp8_quantize_for_fa4(x)
    x_rec = x_fp8.to(torch.float32) * descale[:, None, :, None]
    # e4m3 has 3 mantissa bits: relative step 2^-3, half-step rounding error
    # ~6.25%, plus the bf16 scale multiply. Bound the elementwise error
    # relative to each head's max magnitude.
    err = (x_rec - x.to(torch.float32)).abs()
    head_amax = x.to(torch.float32).abs().amax(dim=(1, 3), keepdim=True)
    assert (err / head_amax).max().item() < 0.07


def test_fp8_quantize_uses_full_range():
    x = torch.randn(1, 64, 2, 16, dtype=torch.bfloat16)
    x_fp8, _ = fp8_quantize_for_fa4(x)
    # The per-head max element should land at (or next to) the e4m3 max.
    assert x_fp8.to(torch.float32).abs().max().item() == pytest.approx(FP8_E4M3_MAX, rel=0.07)


def test_fp8_quantize_rejects_bad_rank():
    with pytest.raises(ValueError, match="batch, seqlen, nheads"):
        fp8_quantize_for_fa4(torch.randn(3, 4, 5))


def test_fp8_quantize_survives_zero_head():
    x = torch.randn(1, 8, 2, 4, dtype=torch.bfloat16)
    x[:, :, 0] = 0.0  # amax clamp path
    x_fp8, descale = fp8_quantize_for_fa4(x)
    assert torch.isfinite(descale).all()
    assert (x_fp8.to(torch.float32)[:, :, 0] == 0).all()


@pytest.mark.parametrize(
    "stage1,stage2,profile,expected",
    [
        (True, False, "base", True),
        (True, False, "refine", False),
        (False, True, "base", False),
        (False, True, "refine", True),
        (True, True, "base", True),
        (True, True, "refine", True),
        (False, False, "base", False),
        (False, False, "refine", False),
    ],
)
def test_fa4_fp8_stage_gating(stage1, stage2, profile, expected):
    assert fa4_fp8_stage_active(stage1, stage2, profile) is expected
