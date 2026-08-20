# SPDX-License-Identifier: Apache-2.0
"""Generic FP8 quantization backed by ``torch._scaled_mm``.

Matches linear layers by suffix (``to_q/k/v/to_out``, ``ffn.fc_in/fc_out``).
Supports per-tensor (default, fast) and per-channel (higher accuracy) granularity.
Falls back to bf16 dequant on GPUs older than sm89.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fastvideo.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from fastvideo.models.utils import set_weight_attrs

logger = logging.getLogger(__name__)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448.0
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)

# Wan's "to_q"/"to_k"/"to_v" also substring-match Kandinsky5's "to_query"/
# "to_key"/"to_value", so only Kandinsky5's out-projection and FFN names need
# to be listed explicitly below.
_FP8_SUFFIXES = (
    "ffn.fc_in",
    "ffn.fc_out",
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    # Kandinsky5
    "self_attention.out_layer",
    "cross_attention.out_layer",
    "feed_forward.mlp.fc_in",
    "feed_forward.mlp.fc_out",
)


def _supports_fp8_compute() -> bool:
    """Whether the active device supports FP8 ``_scaled_mm`` (sm89+)."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] > 8 or (cap[0] == 8 and cap[1] >= 9)


def _quantize_tensorwise(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(x_fp8 [M, K], x_scale [1] float32)``."""
    x_absmax = x_2d.abs().amax().float()
    x_scale = (x_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    x_fp8 = (x_2d / x_scale.to(x_2d.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, x_scale.view(1)


def _quantize_unit(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(x_fp8 [M, K], ones [1] float32)`` — ComfyUI's input recipe
    for scaled-fp8 checkpoints without an input_scale: no rescaling, just a
    saturating cast to e4m3."""
    x_fp8 = x_2d.clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, torch.ones(1, device=x_2d.device, dtype=torch.float32)


def _quantize_rowwise(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(x_fp8 [M, K], x_scale [M, 1] float32)``."""
    x_absmax = x_2d.abs().amax(dim=-1, keepdim=True).float()
    x_scale = (x_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    x_fp8 = (x_2d / x_scale.to(x_2d.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, x_scale


class FP8QuantizeMethod(QuantizeMethodBase):
    """FP8 linear method.

    ``granularity='tensor'`` (default): per-tensor weight + per-tensor
    dynamic activation scales — the fast tensorwise ``_scaled_mm`` path.
    ``granularity='channel'``: per-output-channel weight + per-token
    activation scales (rowwise) — higher accuracy but slower ``_scaled_mm``.

    ``act_scale_mode`` selects how activations are quantized:
      * ``'dynamic'`` (default) — per-call absmax scaling, used by the
        online-quantization path (``transformer_quant: FP8``).
      * ``'unit'`` — scale 1.0 with a saturating cast (clamp to ±448),
        bias fused into ``_scaled_mm``. This is bit-faithful to ComfyUI's
        runtime for ``comfy_quant`` scaled-fp8 checkpoints that carry no
        ``input_scale`` (its ``TensorCoreFP8E4M3Layout.quantize`` defaults
        the input scale to ones and ``scaled_mm_v2`` fuses the bias), so
        pre-quantized checkpoints run through the same numerics that
        produced them.
    """

    def __init__(self, granularity: str = "tensor", act_scale_mode: str = "dynamic"):
        super().__init__()
        if act_scale_mode not in ("dynamic", "unit"):
            raise ValueError(f"act_scale_mode must be 'dynamic' or 'unit', got {act_scale_mode!r}")
        self.granularity = granularity
        self.act_scale_mode = act_scale_mode

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def quantize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Pre-quantize an activation for reuse across q/k/v projections."""
        assert x.dtype in (torch.bfloat16, torch.float16), (f"only allow bf16/fp16 inputs to fp8 linear, got {x.dtype}")
        x_2d = x.view(-1, x.shape[-1])
        if self.act_scale_mode == "unit":
            x_fp8, x_scale = _quantize_unit(x_2d)
        elif self.granularity == "channel":
            x_fp8, x_scale = _quantize_rowwise(x_2d)
        else:
            x_fp8, x_scale = _quantize_tensorwise(x_2d)
        return x_fp8, x_scale, None

    def wants_prequantized_input(self) -> bool:
        return True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        pre_quantized: tuple[torch.Tensor, torch.Tensor, Any] | None = None,
    ) -> torch.Tensor:
        out_dim = layer._fp8_weight.shape[0]
        original_shape = x.shape

        if not _supports_fp8_compute():
            return self._apply_dequant(layer, x, bias)

        if pre_quantized is not None:
            x_fp8, x_scale, _ = pre_quantized
            if x_fp8.dim() > 2:
                x_fp8 = x_fp8.reshape(-1, x_fp8.shape[-1])
            if x_scale.dim() > 2:
                x_scale = x_scale.reshape(-1, x_scale.shape[-1])
        elif self.act_scale_mode == "unit":
            x_fp8, x_scale = _quantize_unit(x.reshape(-1, x.shape[-1]))
        elif self.granularity == "channel":
            x_fp8, x_scale = _quantize_rowwise(x.reshape(-1, x.shape[-1]))
        else:
            x_fp8, x_scale = _quantize_tensorwise(x.reshape(-1, x.shape[-1]))

        w_fp8 = layer._fp8_weight
        w_scale = layer._fp8_weight_scale
        if self.granularity == "channel":
            scale_b = w_scale.view(1, -1)
            if x_scale.numel() == 1:
                # rowwise x rowwise is the only _scaled_mm recipe that takes a
                # per-channel scale_b; broadcast the per-tensor input scale.
                x_scale = x_scale.reshape(1, 1).expand(x_fp8.shape[0], 1).contiguous()
        else:
            scale_b = w_scale

        # 'unit' mode fuses the bias into _scaled_mm — the bias is added on
        # the fp32 accumulator before the single rounding to bf16, exactly
        # like ComfyUI's scaled_mm_v2 call. 'dynamic' keeps the historical
        # post-add so existing online-quant outputs stay bit-identical.
        fused_bias = bias if self.act_scale_mode == "unit" else None
        out = torch._scaled_mm(
            x_fp8,
            w_fp8.t(),
            scale_a=x_scale,
            scale_b=scale_b,
            bias=fused_bias,
            out_dtype=torch.bfloat16,
        )
        if isinstance(out, tuple):
            out = out[0]
        if bias is not None and fused_bias is None:
            out = out + bias
        return out.view(*original_shape[:-1], out_dim)

    def _apply_dequant(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """bf16 fallback for hosts without fp8 ``_scaled_mm`` (pre-sm89, CPU)."""
        out_dim = layer._fp8_weight.shape[0]
        original_shape = x.shape
        w_fp8 = layer._fp8_weight
        w_scale = layer._fp8_weight_scale.to(x.dtype)
        weight = w_fp8.to(x.dtype) * (w_scale.unsqueeze(1) if w_scale.numel() > 1 else w_scale)
        if self.act_scale_mode == "unit":
            # Keep the fallback faithful to the fp8 path: the activation is
            # fp8-rounded (saturating cast at scale 1.0) before the matmul.
            x = x.clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).to(weight.dtype)
        out = F.linear(x, weight, bias)
        return out.view(*original_shape[:-1], out_dim)


class FP8Config(QuantizationConfig):
    """FP8 (e4m3) quantization via suffix matching on standard linear layer names."""

    def __init__(self, granularity: str = "tensor"):
        super().__init__()
        if granularity not in ("tensor", "channel"):
            raise ValueError(f"granularity must be 'tensor' or 'channel', got {granularity!r}")
        self.granularity = granularity

    def get_name(self) -> str:
        return "FP8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FP8Config:
        return cls(granularity=config.get("granularity", "tensor"))

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from fastvideo.layers.linear import LinearBase

        if isinstance(layer, LinearBase) and any(s in prefix for s in _FP8_SUFFIXES):
            return FP8QuantizeMethod(granularity=self.granularity)
        return None


def attach_prequantized_fp8_layers(
    model: torch.nn.Module,
    payloads: dict[str, torch.Tensor],
    scales: dict[str, torch.Tensor],
    device: torch.device | str,
) -> int:
    """Wire checkpoint-provided fp8 weights straight into their linear layers.

    ``payloads`` maps state-dict weight names (e.g.
    ``model.transformer_blocks.0.attn1.to_q.weight``) to fp8 (e4m3/e5m2)
    tensors, ``scales`` maps the same names to their ``.weight_scale``
    dequantization scales — the layout ComfyUI's scaled-fp8 checkpoints ship.
    The payload and scale become the layer's ``_fp8_weight`` /
    ``_fp8_weight_scale`` buffers VERBATIM (no dequant-requant round trip:
    the runtime sees exactly the values the reference implementation runs),
    an :class:`FP8QuantizeMethod` is attached, and the unquantized ``weight``
    parameter is dropped. Layers not named in ``payloads`` are untouched, so
    a mixed checkpoint (fp8 linears + bf16 modulation/embeddings) keeps its
    exact precision profile.

    A scalar scale attaches the per-tensor ("tensor") method; a per-output-
    channel scale attaches the rowwise ("channel") method. Returns the
    number of layers converted.
    """
    from fastvideo.layers.linear import LinearBase, UnquantizedLinearMethod

    orphan_scales = sorted(set(scales) - set(payloads))
    if orphan_scales:
        raise ValueError(f"weight_scale entries without an fp8 weight payload: {orphan_scales[:5]}"
                         f"{' ...' if len(orphan_scales) > 5 else ''}")
    missing_scales = sorted(set(payloads) - set(scales))
    if missing_scales:
        raise ValueError("fp8 weight payload(s) without a weight_scale sibling — a bare fp8 tensor is "
                         f"ambiguous, refusing to guess a scale: {missing_scales[:5]}"
                         f"{' ...' if len(missing_scales) > 5 else ''}")

    with torch.no_grad():
        for name in sorted(payloads):
            payload = payloads[name]
            if payload.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                raise ValueError(f"{name}: expected an fp8 payload, got {payload.dtype}")
            module_path = name[:-len(".weight")]
            module = model.get_submodule(module_path)
            if not isinstance(module, LinearBase):
                raise TypeError(f"{module_path}: fp8 payload targets a {type(module).__name__}, "
                                "only LinearBase layers support pre-quantized fp8")
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None and not isinstance(quant_method, UnquantizedLinearMethod):
                raise ValueError(f"{module_path}: layer was built with a {type(quant_method).__name__} "
                                 "quant method; pre-quantized fp8 checkpoints require the model to be "
                                 "constructed WITHOUT a quant_config (the checkpoint dictates the format)")
            weight = module._parameters.get("weight")
            if weight is not None and tuple(weight.shape) != tuple(payload.shape):
                raise ValueError(f"{module_path}: fp8 payload shape {tuple(payload.shape)} does not match "
                                 f"the layer weight shape {tuple(weight.shape)}")
            out_features = payload.shape[0]
            scale = scales[name].detach().float()
            if scale.numel() == 1:
                granularity = "tensor"
                scale = scale.reshape(1)
            elif scale.numel() == out_features:
                granularity = "channel"
                scale = scale.reshape(out_features)
            else:
                raise ValueError(f"{module_path}: weight_scale has {scale.numel()} elements; expected 1 "
                                 f"(per-tensor) or {out_features} (per-output-channel)")
            module.register_buffer("_fp8_weight", payload.detach().to(device=device).contiguous(), persistent=False)
            module.register_buffer("_fp8_weight_scale", scale.to(device=device), persistent=False)
            # 'unit' activation mode = ComfyUI's runtime for these checkpoints
            # (input scale 1.0 + saturating cast, bias fused into the GEMM).
            module.quant_method = FP8QuantizeMethod(granularity=granularity, act_scale_mode="unit")
            removed_weight = module._parameters.pop("weight", None)
            if removed_weight is not None:
                removed_weight.grad = None
    return len(payloads)


def convert_model_to_fp8(model: torch.nn.Module) -> None:
    """Quantize all FP8-tagged linear layers in-place after weights are loaded."""
    import gc
    from torch.distributed.tensor import DTensor  # type: ignore

    with torch.no_grad():
        for mod in model.modules():
            qm = getattr(mod, "quant_method", None)
            if not isinstance(qm, FP8QuantizeMethod):
                continue
            weight = getattr(mod, "weight", None)
            if weight is None:
                continue
            weight_local = weight.to_local() if isinstance(weight, DTensor) else weight  # type: ignore[arg-type]
            if getattr(qm, "granularity", "tensor") == "channel":
                w_absmax = weight_local.detach().abs().amax(dim=1).nan_to_num().float()
                w_scale = (w_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
                w_fp8 = (weight_local / w_scale.to(weight_local.dtype).unsqueeze(1)).clamp(-FP8_MAX,
                                                                                           FP8_MAX).to(FP8_DTYPE)
            else:
                w_absmax = weight_local.detach().abs().amax().nan_to_num().to(torch.float32)
                w_scale = (w_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE).view(1)
                w_fp8 = (weight_local / w_scale.to(weight_local.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
            mod.register_buffer("_fp8_weight", w_fp8.contiguous(), persistent=False)
            mod.register_buffer("_fp8_weight_scale", w_scale.to(torch.float32), persistent=False)
            removed_weight = mod._parameters.pop("weight", None)
            if removed_weight is not None:
                removed_weight.grad = None
            del removed_weight, weight, weight_local, w_absmax, w_scale, w_fp8

    gc.collect()
    torch.cuda.empty_cache()


__all__ = [
    "FP8Config",
    "FP8QuantizeMethod",
    "attach_prequantized_fp8_layers",
    "convert_model_to_fp8",
]
