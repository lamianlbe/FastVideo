# SPDX-License-Identifier: Apache-2.0
"""Loading ComfyUI scaled-fp8 (pre-quantized) checkpoints without dequantizing.

The checkpoint layout under test is what ComfyUI's ModelSave writes for a
``comfy_quant`` model: fp8 e4m3 ``X.weight`` payloads, 0-d float32
``X.weight_scale`` siblings, ``X.comfy_quant`` descriptor blobs, bf16 biases,
and bf16 weights for the layers the mixed profile left unquantized. The
loader must route the payload+scale pairs VERBATIM into the FP8 runtime
buffers (never through a bf16 cast, which would silently drop the scale) and
leave every other tensor on the normal path.

CPU-only: the fp8 ``_scaled_mm`` path needs sm89+, so the forward checks
exercise the dequant fallback, which mirrors the same unit-scale activation
rounding.
"""
from __future__ import annotations

import json

import pytest
import torch

from fastvideo.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from fastvideo.layers.quantization.fp8_config import (
    FP8_DTYPE,
    FP8Config,
    FP8QuantizeMethod,
    attach_prequantized_fp8_layers,
)
from fastvideo.models.loader.fsdp_load import (
    _maybe_quantize_model,
    load_model_from_full_model_state_dict,
)
from fastvideo.models.loader.utils import get_param_names_mapping

IN_DIM, OUT_DIM = 8, 16


class _TinyMixed(torch.nn.Module):
    """One fp8-quantized linear + one bf16 linear, comfy's mixed profile."""

    def __init__(self, quant_config=None):
        super().__init__()
        self.to_q = ReplicatedLinear(IN_DIM, OUT_DIM, bias=True, params_dtype=torch.bfloat16,
                                     quant_config=quant_config, prefix="blocks.0.to_q")
        self.plain = ReplicatedLinear(OUT_DIM, IN_DIM, bias=True, params_dtype=torch.bfloat16)


def _comfy_quant_blob() -> torch.Tensor:
    return torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8)


def _make_checkpoint(seed: int = 0):
    """(state_dict, dequantized reference weight) with a per-tensor scale."""
    torch.manual_seed(seed)
    w_full = torch.randn(OUT_DIM, IN_DIM) * 3.0
    scale = (w_full.abs().amax() / torch.finfo(FP8_DTYPE).max).float()
    payload = (w_full / scale).to(FP8_DTYPE)
    sd = {
        "to_q.weight": payload,
        "to_q.weight_scale": scale.reshape(()),  # 0-d, like ModelSave
        "to_q.comfy_quant": _comfy_quant_blob(),
        "to_q.bias": torch.randn(OUT_DIM).to(torch.bfloat16),
        "plain.weight": torch.randn(IN_DIM, OUT_DIM).to(torch.bfloat16),
        "plain.bias": torch.randn(IN_DIM).to(torch.bfloat16),
    }
    return sd, payload, scale


def _load(model, sd, strict=True):
    return load_model_from_full_model_state_dict(
        model,
        iter(sd.items()),
        device=torch.device("cpu"),
        param_dtype=torch.bfloat16,
        strict=strict,
        param_names_mapping=get_param_names_mapping({}),
    )


def _meta_model(**kwargs):
    with torch.device("meta"):
        return _TinyMixed(**kwargs)


def test_prequantized_pair_becomes_fp8_buffers_verbatim():
    sd, payload, scale = _make_checkpoint()
    model = _meta_model()
    _load(model, sd)

    # The payload landed bit-for-bit — no dequant/requant round trip.
    assert model.to_q._fp8_weight.dtype == FP8_DTYPE
    assert torch.equal(model.to_q._fp8_weight.view(torch.uint8), payload.view(torch.uint8))
    assert model.to_q._fp8_weight_scale.dtype == torch.float32
    assert model.to_q._fp8_weight_scale.shape == (1, )
    assert model.to_q._fp8_weight_scale.item() == pytest.approx(scale.item())
    # The bf16 weight parameter is gone; bias stayed a normal parameter.
    assert "weight" not in dict(model.to_q.named_parameters())
    assert model.to_q.bias.dtype == torch.bfloat16
    # ComfyUI-parity runtime: per-tensor granularity, unit activation scale.
    qm = model.to_q.quant_method
    assert isinstance(qm, FP8QuantizeMethod)
    assert qm.granularity == "tensor" and qm.act_scale_mode == "unit"
    # The unquantized layer is untouched by the diversion.
    assert isinstance(model.plain.quant_method, UnquantizedLinearMethod)
    assert model.plain.weight.dtype == torch.bfloat16
    # Nothing is left on meta (the loader's post-load invariant).
    for name, param in model.named_parameters():
        assert not param.is_meta, name


def test_forward_matches_comfy_unit_scale_reference():
    """CPU fallback: activation fp8-rounded at scale 1, weight dequantized."""
    sd, payload, scale = _make_checkpoint(seed=1)
    model = _meta_model()
    _load(model, sd)

    torch.manual_seed(2)
    x = (torch.randn(3, IN_DIM) * 2.0).to(torch.bfloat16)
    out, _ = model.to_q(x)

    x_ref = x.clamp(-448.0, 448.0).to(FP8_DTYPE).to(torch.bfloat16)
    w_ref = payload.to(torch.bfloat16) * scale.to(torch.bfloat16)
    expected = torch.nn.functional.linear(x_ref, w_ref, model.to_q.bias)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_quantize_input_reuse_matches_direct_apply():
    """The q/k/v prequant-reuse path must produce the same fp8 activation."""
    sd, *_ = _make_checkpoint(seed=3)
    model = _meta_model()
    _load(model, sd)
    qm = model.to_q.quant_method
    x = torch.randn(4, IN_DIM).to(torch.bfloat16)
    x_fp8, x_scale, _ = qm.quantize_input(x)
    assert x_fp8.dtype == FP8_DTYPE
    assert torch.all(x_scale == 1.0)
    assert torch.equal(x_fp8.view(torch.uint8),
                       x.clamp(-448.0, 448.0).to(FP8_DTYPE).view(torch.uint8))


def test_per_channel_scale_attaches_channel_granularity():
    sd, payload, scale = _make_checkpoint(seed=4)
    sd["to_q.weight_scale"] = torch.full((OUT_DIM, ), float(scale))
    model = _meta_model()
    _load(model, sd)
    qm = model.to_q.quant_method
    assert qm.granularity == "channel"
    assert model.to_q._fp8_weight_scale.shape == (OUT_DIM, )


def test_fp8_weight_without_scale_is_refused():
    sd, *_ = _make_checkpoint()
    del sd["to_q.weight_scale"]
    with pytest.raises(ValueError, match="weight_scale"):
        _load(_meta_model(), sd)


def test_orphan_scale_is_refused():
    sd, *_ = _make_checkpoint()
    sd["plain.weight_scale"] = torch.tensor(1.0)
    with pytest.raises(ValueError, match="without an fp8 weight"):
        _load(_meta_model(), sd)


def test_shape_mismatch_is_refused():
    sd, *_ = _make_checkpoint()
    sd["to_q.weight"] = sd["to_q.weight"][:, :IN_DIM // 2]
    with pytest.raises(ValueError, match="shape"):
        _load(_meta_model(), sd)


def test_construction_time_quant_conflict_is_refused():
    """The checkpoint dictates the format: a model built with its own quant
    method (transformer_quant knob) must not silently double-quantize."""
    sd, payload, scale = _make_checkpoint()
    model = _meta_model(quant_config=FP8Config())
    assert isinstance(model.to_q.quant_method, FP8QuantizeMethod)  # suffix-matched
    with pytest.raises(ValueError, match="quant"):
        attach_prequantized_fp8_layers(
            model, {"to_q.weight": payload}, {"to_q.weight": scale}, device="cpu")


def test_maybe_quantize_model_leaves_prequantized_buffers_alone():
    sd, payload, _ = _make_checkpoint(seed=5)
    model = _meta_model()
    _load(model, sd)
    before = model.to_q._fp8_weight.view(torch.uint8).clone()
    _maybe_quantize_model(model)
    assert torch.equal(model.to_q._fp8_weight.view(torch.uint8), before)
    assert "weight" not in dict(model.to_q.named_parameters())
