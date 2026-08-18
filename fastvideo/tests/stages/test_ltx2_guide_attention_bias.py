# SPDX-License-Identifier: Apache-2.0
"""ComfyUI guide self-attention bias (LTXVModel._build_self_attention_mask).

The reference biases content<->guide attention by log(strength) in both
directions and leaves guide<->guide / content<->content alone. Comfy appends
its guides at the END of the sequence; our reference tokens are a PREFIX, so
the two biased blocks are transposed — that transposition is exactly what a
silent bug would get wrong, hence the explicit block assertions here.
"""
from __future__ import annotations

import math

import pytest
import torch

from fastvideo.models.dits.ltx2 import build_guide_attention_bias

N_REF = 3
TOTAL = 8
STRENGTH = 0.8


def _bias(strength: float = STRENGTH, n_ref: int = N_REF, total: int = TOTAL):
    return build_guide_attention_bias(
        total_tokens=total,
        n_ref_tokens=n_ref,
        strength=strength,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_bias_lands_on_exactly_the_two_cross_blocks() -> None:
    bias = _bias()
    assert bias is not None and bias.shape == (1, 1, TOTAL, TOTAL)
    log_w = math.log(STRENGTH)
    m = bias[0, 0]
    # our layout: guide = prefix [0, N_REF), content = [N_REF, TOTAL)
    assert torch.allclose(m[N_REF:, :N_REF], torch.full((TOTAL - N_REF, N_REF), log_w))
    assert torch.allclose(m[:N_REF, N_REF:], torch.full((N_REF, TOTAL - N_REF), log_w))
    # untouched blocks
    assert torch.all(m[:N_REF, :N_REF] == 0.0), "guide<->guide must not be biased"
    assert torch.all(m[N_REF:, N_REF:] == 0.0), "content<->content must not be biased"


def test_strength_one_disables_the_bias_entirely() -> None:
    # Comfy returns None once every guide strength reaches 1.0; matching that
    # keeps strength 1.0 bit-identical to the unbiased fast path.
    assert _bias(strength=1.0) is None
    assert _bias(strength=1.5) is None


def test_no_reference_tokens_means_no_bias() -> None:
    assert _bias(n_ref=0) is None


def test_degenerate_strength_is_rejected() -> None:
    with pytest.raises(ValueError, match="strength"):
        _bias(strength=0.0)


def test_bias_scales_cross_attention_weights_by_strength() -> None:
    """After softmax the biased pairs carry `strength`x their unbiased weight
    (before renormalization) — the property the log-space bias encodes."""
    torch.manual_seed(0)
    scores = torch.randn(1, 1, TOTAL, TOTAL)
    bias = _bias()
    assert bias is not None
    plain = torch.softmax(scores, dim=-1)
    biased = torch.softmax(scores + bias, dim=-1)

    # For a content query row: guide columns are attenuated relative to
    # content columns by exactly `strength` in the pre-normalization ratio.
    row = N_REF  # first content row
    guide_ratio = biased[0, 0, row, :N_REF] / plain[0, 0, row, :N_REF]
    content_ratio = biased[0, 0, row, N_REF:] / plain[0, 0, row, N_REF:]
    # every content column shares one renormalization factor ...
    assert torch.allclose(content_ratio, content_ratio[0].expand_as(content_ratio), atol=1e-6)
    # ... and each guide column is `strength` times that factor
    assert torch.allclose(guide_ratio / content_ratio[0],
                          torch.full_like(guide_ratio, STRENGTH),
                          atol=1e-6)
    # Guide tokens end up with strictly less attention than without the bias.
    assert biased[0, 0, row, :N_REF].sum() < plain[0, 0, row, :N_REF].sum()


def test_matches_a_literal_transcription_of_the_comfy_builder() -> None:
    """Reference construction, transposed for our prefix layout."""
    log_w = math.log(STRENGTH)
    expected = torch.zeros((1, 1, TOTAL, TOTAL))
    guide_end = N_REF
    expected[:, :, guide_end:, :guide_end] = log_w
    expected[:, :, :guide_end, guide_end:] = log_w
    assert torch.equal(_bias(), expected)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as err:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(err).__name__}: {err}")
    raise SystemExit(1 if failures else 0)
