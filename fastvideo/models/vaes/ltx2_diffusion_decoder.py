# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.5 diffusion (NATTEN) video VAE decoder — the "HQ" alternative to the convolutional decoder.

Port of the official ``NADiffusionDecoder`` / ``DiffusionVideoDecoder`` from
``LTX-2/packages/ltx-core/src/ltx_core/model/video_vae/diffusion_video_decoder.py`` (and its
``transformer/`` building blocks), cross-checked against the diffusers ``LTX2VideoDiffusionDecoderModel``
port. Stages 1-4 deterministically upsample the latent into a context volume with neighborhood-attention
blocks; stage 5 denoises patchified noised pixels ``x_t`` conditioned on that context via AdaLN-Zero
scale/shift (ungated residuals; the checkpoint's static gates are folded into Linear weights by the
conversion script). Critically, stages 1-4 run once per decode — only stage 5 re-runs per diffusion step.

The neighborhood-attention backend is chosen by exactly one helper (:func:`neighborhood_attention_3d`):
``natten`` when importable on CUDA (NATTEN auto-picks its fastest kernel, e.g. the CUTLASS Blackwell
sm100 FNA backend on B200/B300), then a Triton port, then an eager tiled-SDPA fallback that also runs
on CPU. The Triton and eager fallbacks are ports of the official ``transformer/fallback_na`` package
(vendored there from comfy-kitchen, Apache-2.0).

This module is deliberately self-contained at import time (torch + ``fastvideo.logger`` only) so the
decoder can be unit-tested without the full package; the conv-VAE encoder used by the
``LTX2CausalDiffusionVAE`` wrapper is imported lazily from ``ltx2vae`` at construction time.

Known gaps vs. the official implementation (documented, not silent):
- No torch.compile integration (no ``mark_dynamic`` / custom-op packaging / CUDA graphs).
- Only the "combined" diffusion pathway is ported; the chunked/deferred-stage-4 modes and the
  Blackwell CuTe-DSL fused block are not.
- Tiling uses the diffusers-style fixed-size overlapping grid with linear seam blending instead of
  the official trapezoid-mask schedule and memory-budget-driven ``recommended_decode_tiling_config``.
"""

from __future__ import annotations

import math
import os
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.logger import init_logger

logger = init_logger(__name__)

try:
    import natten

    _NATTEN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    natten = None
    _NATTEN_AVAILABLE = False

# Production CausalVideoAutoencoderL decoder layout (class defaults mirror the official
# ``DiffusionVideoDecoder``); real checkpoints override these through their metadata config
# (LTX-2.5 ships stage_channels (2048, 1024, 512, 512, 256), stage5_kernel (11, 11, 11),
# model_output_type "x0", default_num_inference_steps 1, timestep_scale_multiplier 1000.0).
_L_STAGE_CHANNELS: Tuple[int, ...] = (1024, 512, 256, 256, 128)
_L_STAGE_DEPTHS: Tuple[int, ...] = (4, 6, 4, 2, 2)
# (stride, out_channels_reduction_factor) per upsample, in stage order.
_L_UPSAMPLES: Tuple[Tuple[Tuple[int, int, int], int], ...] = (
    ((1, 2, 2), 2),  # compress_space x2
    ((2, 1, 1), 2),  # compress_time x2
    ((2, 2, 2), 1),  # compress_all x1 (channel-preserving)
    ((2, 2, 2), 2),  # compress_all x2
)
# Per-stage 3D neighborhood (K_t, K_h, K_w).
_L_STAGE_KERNELS: Tuple[Tuple[int, int, int], ...] = (
    (3, 7, 7),
    (3, 7, 7),
    (3, 5, 5),
    (3, 5, 5),
    (3, 3, 3),
)

# Stage-5 (diffusion stage) defaults: wider kernel + more blocks than the deterministic stages.
_DIFF_STAGE5_KERNEL_DEFAULT: Tuple[int, int, int] = (3, 7, 7)
_DIFF_STAGE5_DEPTH_DEFAULT: int = 8
_DIFF_STAGE_DEPTHS_DEFAULT: Tuple[int, ...] = (*_L_STAGE_DEPTHS[:-1], _DIFF_STAGE5_DEPTH_DEFAULT)


# =============================================================================
# Neighborhood-attention backends (natten -> Triton -> eager tiled SDPA)
# =============================================================================

# Element budget for one tile's [Nq, Nk] attention mask in the eager fallback.
NA_SCORE_BUDGET = 2**25
# Element budget for the stacked K/V copies of one batched SDPA call on CUDA.
NA_KV_STACK_BUDGET = 2**28


def _window_bounds(length: int, kernel: int) -> tuple[list[int], list[int]]:
    """Per-index (start, end) of the attended window along one axis (NATTEN inward-shift semantics)."""
    kernel = min(kernel, length)
    lo = length - kernel
    half = kernel // 2
    starts: list[int] = []
    ends: list[int] = []
    for i in range(length):
        start = min(max(i - half, 0), lo)
        starts.append(start)
        ends.append(start + kernel)
    return starts, ends


def _pick_tiles(dims: tuple[int, int, int], kernels: list[int]) -> list[int]:
    """Per-axis query-tile lengths keeping one tile's [Nq, Nk] under budget."""
    tiles = list(dims)

    def cost(ts: list[int]) -> int:
        nq = math.prod(ts)
        nk = math.prod(min(d, t + k - 1) for t, k, d in zip(ts, kernels, dims, strict=True))
        return nq * nk

    while cost(tiles) > NA_SCORE_BUDGET and max(tiles) > 1:
        i = max(range(3), key=lambda a: tiles[a] / kernels[a])
        if tiles[i] <= 1:
            break
        tiles[i] = max(1, (tiles[i] + 1) // 2)
    return tiles


def _group_mask(
    rel_bounds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive ``[1, 1, Nq, Nk]`` mask for one tile-geometry group."""
    bools = []
    for starts, ends in rel_bounds:
        st = torch.tensor(starts, device=device)
        en = torch.tensor(ends, device=device)
        kj = torch.arange(int(en.max()), device=device)
        bools.append((kj[None, :] >= st[:, None]) & (kj[None, :] < en[:, None]))
    visible = (bools[0][:, None, None, :, None, None]
               & bools[1][None, :, None, None, :, None]
               & bools[2][None, None, :, None, None, :])
    nq = visible.shape[0] * visible.shape[1] * visible.shape[2]
    nk = visible.shape[3] * visible.shape[4] * visible.shape[5]
    mask = torch.zeros((nq, nk), dtype=dtype, device=device)
    mask.masked_fill_(~visible.reshape(nq, nk), torch.finfo(dtype).min)
    return mask.reshape(1, 1, nq, nk)


def _eager_na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
) -> torch.Tensor:
    """Limited-workspace 3D neighborhood attention (NATTEN ``na3d`` semantics) in pure torch.

    Port of the official ``transformer/fallback_na/eager.py`` (vendored from comfy-kitchen,
    Apache-2.0). Queries are tiled; tiles that share window geometry stack into batched
    ``scaled_dot_product_attention`` calls with one additive mask per group. Q is assumed
    pre-scaled (``scale=1.0`` semantics).
    """
    batch, t, h, w, nh, hd = q.shape
    dims = (t, h, w)
    kernels = [min(k_, d) for k_, d in zip(kernel_size, dims, strict=True)]
    device = q.device

    bounds = [_window_bounds(d, k_) for d, k_ in zip(dims, kernels, strict=True)]
    tile_t, tile_h, tile_w = _pick_tiles(dims, kernels)

    groups: dict[tuple, list[tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]]] = {}
    for t0 in range(0, t, tile_t):
        t1 = min(t0 + tile_t, t)
        rt0, rt1 = bounds[0][0][t0], bounds[0][1][t1 - 1]
        rel_t = (tuple(s - rt0 for s in bounds[0][0][t0:t1]), tuple(e - rt0 for e in bounds[0][1][t0:t1]))
        for h0 in range(0, h, tile_h):
            h1 = min(h0 + tile_h, h)
            rh0, rh1 = bounds[1][0][h0], bounds[1][1][h1 - 1]
            rel_h = (tuple(s - rh0 for s in bounds[1][0][h0:h1]), tuple(e - rh0 for e in bounds[1][1][h0:h1]))
            for w0 in range(0, w, tile_w):
                w1 = min(w0 + tile_w, w)
                rw0, rw1 = bounds[2][0][w0], bounds[2][1][w1 - 1]
                rel_w = (tuple(s - rw0 for s in bounds[2][0][w0:w1]), tuple(e - rw0 for e in bounds[2][1][w0:w1]))
                groups.setdefault((rel_t, rel_h, rel_w), []).append((
                    (slice(t0, t1), slice(h0, h1), slice(w0, w1)),
                    (slice(rt0, rt1), slice(rh0, rh1), slice(rw0, rw1)),
                ))

    out = torch.empty((batch, t, h, w, nh, hd), device=device, dtype=v.dtype)
    for rel, tiles in groups.items():
        mask = _group_mask(rel, q.dtype, device)
        nq, nk = mask.shape[2], mask.shape[3]
        g_max = max(1, NA_KV_STACK_BUDGET // max(1, batch * nh * nk * hd * 2)) if device.type == "cuda" else 1
        qs0, _ = tiles[0]
        tq = qs0[0].stop - qs0[0].start
        th = qs0[1].stop - qs0[1].start
        tw = qs0[2].stop - qs0[2].start
        for c0 in range(0, len(tiles), g_max):
            chunk = tiles[c0:c0 + g_max]
            g = len(chunk)
            q_s = torch.stack([q[:, qs[0], qs[1], qs[2]] for qs, _ in chunk])
            k_s = torch.stack([k[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            v_s = torch.stack([v[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            q_s = q_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nq, hd)
            k_s = k_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            v_s = v_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            o = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=mask, scale=1.0)
            o = o.view(g, batch, nh, tq, th, tw, hd).permute(0, 1, 3, 4, 5, 2, 6)
            for i, (qs, _) in enumerate(chunk):
                out[:, qs[0], qs[1], qs[2]] = o[i]

    return out


def _triton_na_available() -> bool:
    """True when CUDA is available and the ``triton`` package imports cleanly."""
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def _triton_na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
) -> torch.Tensor:
    """Triton 3D neighborhood attention (NATTEN ``na3d`` semantics), CUDA only.

    Port of the official ``transformer/fallback_na/triton_na.py`` (vendored from comfy-kitchen,
    Apache-2.0). One program handles a run of ``BLOCK_Q`` queries along W at a fixed (t, h);
    online softmax, fp32 accumulation, no materialized scores. Q is assumed pre-scaled.
    """
    import triton
    import triton.language as tl

    global _TRITON_NA3D_KERNEL
    if _TRITON_NA3D_KERNEL is None:

        neg_inf = tl.constexpr(-3.0e38)

        @triton.jit
        def _na3d_kernel(
            q_ptr, k_ptr, v_ptr, out_ptr,
            t_size, h_size, w_size, num_heads,
            s_b, s_t, s_h, s_w, s_n,
            scale,
            kt: tl.constexpr, kh: tl.constexpr, kw: tl.constexpr,
            hd: tl.constexpr, hd_pad: tl.constexpr,
            block_q: tl.constexpr, block_k: tl.constexpr,
            is_fp32: tl.constexpr,
        ):
            pid_w = tl.program_id(0)
            pid_th = tl.program_id(1)
            pid_bn = tl.program_id(2)

            t_q = pid_th // h_size
            h_q = pid_th % h_size
            base = (pid_bn // num_heads) * s_b + (pid_bn % num_heads) * s_n

            w_off = pid_w * block_q + tl.arange(0, block_q)
            w_valid = w_off < w_size
            d_off = tl.arange(0, hd_pad)
            d_mask = d_off < hd

            q_ptrs = q_ptr + base + t_q * s_t + h_q * s_h + w_off[:, None] * s_w + d_off[None, :]
            q_blk = tl.load(q_ptrs, mask=w_valid[:, None] & d_mask[None, :], other=0.0)

            t_lo = tl.minimum(tl.maximum(t_q - kt // 2, 0), t_size - kt)
            t_hi = t_lo + kt
            h_lo = tl.minimum(tl.maximum(h_q - kh // 2, 0), h_size - kh)
            h_hi = h_lo + kh

            w_q = tl.where(w_valid, w_off, w_size - 1)
            w_start = tl.minimum(tl.maximum(w_q - kw // 2, 0), w_size - kw)
            w_end = w_start + kw
            blk_first = tl.minimum(pid_w * block_q, w_size - 1)
            blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
            w_lo = tl.minimum(tl.maximum(blk_first - kw // 2, 0), w_size - kw)
            w_hi = tl.minimum(tl.maximum(blk_last - kw // 2, 0), w_size - kw) + kw

            m_i = tl.full((block_q,), neg_inf, dtype=tl.float32)
            l_i = tl.zeros((block_q,), dtype=tl.float32)
            acc = tl.zeros((block_q, hd_pad), dtype=tl.float32)

            for tk in range(t_lo, t_hi):
                for hk in range(h_lo, h_hi):
                    plane = base + tk * s_t + hk * s_h
                    for wk0 in range(w_lo, w_hi, block_k):
                        wk = wk0 + tl.arange(0, block_k)
                        kmask = wk < w_hi
                        kv_ptrs = plane + wk[:, None] * s_w + d_off[None, :]
                        kv_mask = kmask[:, None] & d_mask[None, :]
                        k_blk = tl.load(k_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                        if is_fp32:
                            s = tl.dot(q_blk, tl.trans(k_blk), input_precision="ieee") * scale
                        else:
                            s = tl.dot(q_blk, tl.trans(k_blk)) * scale
                        vis = (wk[None, :] >= w_start[:, None]) & (wk[None, :] < w_end[:, None]) & kmask[None, :]
                        s = tl.where(vis, s, neg_inf)
                        m_new = tl.maximum(m_i, tl.max(s, 1))
                        alpha = tl.exp(m_i - m_new)
                        p = tl.exp(s - m_new[:, None])
                        l_i = l_i * alpha + tl.sum(p, 1)
                        v_blk = tl.load(v_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                        if is_fp32:
                            acc = acc * alpha[:, None] + tl.dot(p, v_blk, input_precision="ieee")
                        else:
                            acc = acc * alpha[:, None] + tl.dot(p.to(v_blk.dtype), v_blk)
                        m_i = m_new

            out = acc / tl.maximum(l_i, 1e-30)[:, None]
            out_ptrs = out_ptr + base + t_q * s_t + h_q * s_h + w_off[:, None] * s_w + d_off[None, :]
            tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=w_valid[:, None] & d_mask[None, :])

        _TRITON_NA3D_KERNEL = _na3d_kernel

    batch, t, h, w, nh, hd = q.shape
    kt, kh, kw = (min(k_, d) for k_, d in zip(kernel_size, (t, h, w), strict=True))

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)

    hd_p = max(16, triton.next_power_of_2(hd))
    block_q = 16
    block_k = max(16, min(32, triton.next_power_of_2(min(w, block_q + kw))))

    grid = (triton.cdiv(w, block_q), t * h, batch * nh)
    _TRITON_NA3D_KERNEL[grid](
        q, k, v, out,
        t, h, w, nh,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4),
        1.0,
        kt=kt, kh=kh, kw=kw,
        hd=hd, hd_pad=hd_p,
        block_q=block_q, block_k=block_k,
        is_fp32=q.dtype == torch.float32,
        num_warps=4,
    )
    return out


_TRITON_NA3D_KERNEL: Any = None
_NA_BACKEND_WARNED: set[str] = set()


def _resolve_na3d_backend(device_type: str) -> str:
    """Resolve the NA backend name for a device: natten -> triton -> eager, with env override.

    ``FASTVIDEO_LTX2_NA_BACKEND`` in {"natten", "triton", "eager"} forces a backend (validated).
    natten/Triton require CUDA tensors; every other device gets the eager tiled-SDPA fallback.
    """
    override = os.getenv("FASTVIDEO_LTX2_NA_BACKEND", "").lower()
    if override:
        if override not in ("natten", "triton", "eager"):
            raise ValueError(f"FASTVIDEO_LTX2_NA_BACKEND must be natten/triton/eager, got {override!r}")
        if override == "natten" and not _NATTEN_AVAILABLE:
            raise ImportError("FASTVIDEO_LTX2_NA_BACKEND=natten but natten is not installed")
        if override == "triton" and not _triton_na_available():
            raise ImportError("FASTVIDEO_LTX2_NA_BACKEND=triton but CUDA/triton are unavailable")
        return override
    if device_type == "cuda":
        if _NATTEN_AVAILABLE:
            return "natten"
        backend = "triton" if _triton_na_available() else "eager"
        if backend not in _NA_BACKEND_WARNED:
            _NA_BACKEND_WARNED.add(backend)
            logger.warning(
                "LTX-2 diffusion decoder: natten is NOT installed; falling back to the slower %s "
                "neighborhood-attention backend. Install natten for production HQ decode "
                "(it ships a CUTLASS Blackwell sm100 FNA backend for B200/B300).", backend)
        return backend
    return "eager"


def neighborhood_attention_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
) -> torch.Tensor:
    """The single NA dispatch point. Q/K/V are ``(B, T, H, W, NH, HD)``, normed/scaled/RoPE'd.

    Returns ``(B, T, H, W, NH, HD)``-shaped output (natten/Triton/eager all follow ``na3d``
    semantics). ``scale=1.0`` everywhere: callers already applied the attention scale to Q.
    """
    # RMSNorm in fp32 autocast can leave Q/K in float32 while V stays bf16; the kernels
    # require a uniform dtype (same cast pattern as the official NattenAttention).
    if q.dtype != v.dtype or k.dtype != v.dtype:
        q = q.to(dtype=v.dtype)
        k = k.to(dtype=v.dtype)
    backend = _resolve_na3d_backend(q.device.type)
    if backend == "natten":
        pinned = os.getenv("FASTVIDEO_LTX2_NATTEN_BACKEND") or None
        return natten.na3d(q, k, v, kernel_size=tuple(kernel_size), scale=1.0, backend=pinned)
    if backend == "triton":
        return _triton_na3d(q, k, v, tuple(kernel_size))
    return _eager_na3d(q, k, v, tuple(kernel_size))


# =============================================================================
# Patchify / statistics / timestep embedding (self-contained; mirror ltx2vae semantics)
# =============================================================================


def _patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Space-to-depth on H/W only: ``(B, C, F, H, W) -> (B, C*p*p, F, H//p, W//p)``.

    Channel packing order is ``(c, width_offset, height_offset)``, identical to the official
    einops ``b c (f p) (h q) (w r) -> b (c p r q) f h w`` with ``p=1`` (pure-torch to keep this
    module importable without einops).
    """
    if patch_size == 1:
        return x
    b, c, f, hgt, wid = x.shape
    x = x.reshape(b, c, f, hgt // patch_size, patch_size, wid // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5)
    return x.reshape(b, c * patch_size * patch_size, f, hgt // patch_size, wid // patch_size)


def _unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Depth-to-space on H/W only, the exact inverse of :func:`_patchify`."""
    if patch_size == 1:
        return x
    b, c, f, hgt, wid = x.shape
    c = c // (patch_size * patch_size)
    x = x.reshape(b, c, patch_size, patch_size, f, hgt, wid)
    x = x.permute(0, 1, 4, 5, 3, 6, 2)
    return x.reshape(b, c, f, hgt * patch_size, wid * patch_size)


class PerChannelStatistics(nn.Module):
    """Per-channel latent statistics (official ``ops.PerChannelStatistics``: two buffers only)."""

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.register_buffer("std-of-means", torch.ones(latent_channels))
        self.register_buffer("mean-of-means", torch.zeros(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)
        mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)
        return x * std + mean

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)
        mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)
        return (x - mean) / std


class Timesteps(nn.Module):
    """Sinusoidal timestep embeddings (matches ltx2vae.Timesteps)."""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """MLP for timestep embeddings (matches ltx2vae.TimestepEmbedding)."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """Timestep embeddings for decoder conditioning (state-dict compatible with ltx2vae's)."""

    def __init__(self, embedding_dim: int, size_emb_dim: int = 0):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_dtype: torch.dtype) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))


# =============================================================================
# Absolute RoPE (official transformer/rope_math.py semantics)
# =============================================================================

_DEFAULT_ABS_ROPE_NUM_TILES = 4


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    """Default split of head_dim across (T, H, W) RoPE chunks (official rope_math)."""
    if head_dim % 8 != 0:
        raise ValueError(f"head_dim={head_dim} must be a multiple of 8 for the default RoPE split")
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    if d_t <= 0 or d_hw <= 0:
        raise ValueError(f"head_dim={head_dim} has no valid default RoPE split")
    return (d_t, d_hw, d_hw)


def rope_inv_freqs(dim: int, base: float = 10000.0) -> torch.Tensor:
    """Inverse RoPE frequencies ``1 / base**(i/dim)`` for ``i`` in ``[0, dim, 2)`` (fp64 -> fp32)."""
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")
    exponents = torch.arange(0, dim, 2, dtype=torch.float64) / dim
    return (1.0 / torch.pow(torch.tensor(float(base), dtype=torch.float64), exponents)).to(torch.float32)


def _rot_abs_axis(
    xc: torch.Tensor,
    pos: torch.Tensor,
    inv: torch.Tensor,
    axis: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Absolute RoPE on one axis chunk ``xc[..., D]`` (D even) -> new tensor."""
    out_dtype = xc.dtype
    pairs = xc.reshape(*xc.shape[:-1], xc.shape[-1] // 2, 2)
    xe = pairs[..., 0].to(compute_dtype)
    xo = pairs[..., 1].to(compute_dtype)
    shape = [1, 1, 1, 1, 1, inv.shape[0]]
    shape[axis] = pos.shape[0]
    ang = (pos[:, None] * inv[None, :]).reshape(shape)
    c = ang.cos().to(compute_dtype)
    s = ang.sin().to(compute_dtype)
    re = xe * c - xo * s
    ro = xe * s + xo * c
    out = torch.stack([re, ro], dim=-1).reshape(xc.shape)
    return out.to(out_dtype) if out.dtype != out_dtype else out


def _apply_abs_rope(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    num_tiles: int = _DEFAULT_ABS_ROPE_NUM_TILES,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """W-tiled absolute RoPE over ``(B, T, H, W, NH, HD)`` (positions are local 0-based).

    Local 0-based positions are equivalent to absolute ones under tiled decode: NA windows never
    cross tiles, and a global phase shift cancels inside the softmax (see the official
    ``det_attn_rope`` docstring).
    """
    d_t, d_h, _ = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    t = x.shape[1]
    h = x.shape[2]
    t_pos = torch.arange(t, dtype=torch.float32, device=x.device)
    h_pos = torch.arange(h, dtype=torch.float32, device=x.device)
    slabs = torch.chunk(x, num_tiles, dim=3)
    w_off = 0
    parts: list[torch.Tensor] = []
    for slab in slabs:
        w_slab = slab.shape[3]
        w_pos = torch.arange(w_slab, dtype=torch.float32, device=x.device) + w_off
        xt = _rot_abs_axis(slab[..., :d_t], t_pos, inv_t, axis=1, compute_dtype=compute_dtype)
        xh = _rot_abs_axis(slab[..., d_t:d_t + d_h], h_pos, inv_h, axis=2, compute_dtype=compute_dtype)
        xw = _rot_abs_axis(slab[..., d_t + d_h:], w_pos, inv_w, axis=3, compute_dtype=compute_dtype)
        parts.append(torch.cat([xt, xh, xw], dim=-1))
        w_off += w_slab
    return torch.cat(parts, dim=3)


# =============================================================================
# Transformer building blocks (official transformer/{qkv,layers,swiglu,attention,blocks}.py)
# =============================================================================


class QKVProjections(nn.Module):
    """Three separate Q/K/V linears. Checkpoints ship a fused ``qkv.{weight,bias}``; the
    conversion script splits it into ``qkv.to_q`` / ``qkv.to_k`` / ``qkv.to_v``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.to_q(x), self.to_k(x), self.to_v(x)


# Tokens per tile for the SwiGLU MLP (official DEFAULT_SWIGLU_TILE_SIZE). Tiling is exact
# (the MLP is pointwise across tokens); it only bounds the live hidden-width workspace.
_SWIGLU_TILE_SIZE = 16_384


class SwiGLU(nn.Module):
    """Gated MLP weights ``w_down(silu(w_gate(x)) * w_up(x))``, evaluated in token tiles."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        # RMSNorm under autocast can leave activations fp32 while weights stay bf16.
        if x.dtype != self.w_gate.weight.dtype:
            x = x.to(dtype=self.w_gate.weight.dtype)
        leading = x.shape[:-1]
        dim = x.shape[-1]
        flat = x.reshape(-1, dim)
        n_tok = flat.shape[0]
        if n_tok <= _SWIGLU_TILE_SIZE:
            return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        out = torch.empty_like(flat)
        for start in range(0, n_tok, _SWIGLU_TILE_SIZE):
            tile = flat[start:start + _SWIGLU_TILE_SIZE]
            out[start:start + _SWIGLU_TILE_SIZE] = self.w_down(F.silu(self.w_gate(tile)) * self.w_up(tile))
        return out.reshape(*leading, dim)


def _swiglu_hidden_dim(dim: int, mlp_ratio: float = 4.0) -> int:
    return (int(dim * mlp_ratio) + 15) // 16 * 16


class AdaLNZero(nn.Module):
    """Shared AdaLN-Zero modulation: ``t_emb`` -> 7 (scale/shift/gate) chunks.

    Only the four scale/shift chunks are consumed; residuals are ungated (the checkpoint's
    static gates are folded into Linear weights at conversion time).
    """

    NUM_CHUNKS: int = 7  # scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp, gate_ctx

    def __init__(self, dim: int, t_emb_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(t_emb_dim, self.NUM_CHUNKS * dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.proj(F.silu(t_emb))
        chunks = h.chunk(self.NUM_CHUNKS, dim=-1)
        return tuple(c[:, None, None, None, :] for c in chunks)


class LinearPixelShuffleUpsample(nn.Module):
    """Decoder-side resampler: Linear channel-expand, then channels-last pixel shuffle.

    When ``stride[0] == 2`` the shuffle produces a duplicate leading frame that must be dropped
    to preserve the causal 1:2 (composed 1:8) frame mapping. ``drop_leading_frame`` must be True
    only for the tile containing the tensor's true temporal origin (t=0).
    """

    def __init__(self, in_channels: int, stride: tuple[int, int, int], out_channels_reduction_factor: int = 1) -> None:
        super().__init__()
        self.stride = tuple(stride)
        self.proj_out_channels = math.prod(self.stride) * in_channels // out_channels_reduction_factor
        self.out_channels = self.proj_out_channels // math.prod(self.stride)
        self.proj = nn.Linear(in_channels, self.proj_out_channels, bias=True)

    def forward(self, x: torch.Tensor, drop_leading_frame: bool = True) -> torch.Tensor:
        b, t, h, w, _ = x.shape
        p1, p2, p3 = self.stride
        x = self.proj(x)
        # einops: "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c"
        x = x.reshape(b, t, h, w, self.out_channels, p1, p2, p3)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.reshape(b, t * p1, h * p2, w * p3, self.out_channels)
        if p1 == 2 and drop_leading_frame:
            x = x[:, 1:, :, :, :]
        return x


class NeighborhoodAttention3D(nn.Module):
    """3D Neighborhood Attention with absolute RoPE and the pluggable NA backend."""

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        rope_dim_split: tuple[int, int, int] | None = None,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % head_dim != 0:
            raise ValueError(f"dim={dim} not divisible by head_dim={head_dim}")
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5

        if rope_dim_split is None:
            rope_dim_split = default_rope_dim_split(head_dim)
        if sum(rope_dim_split) != head_dim:
            raise ValueError(f"rope_dim_split={rope_dim_split} must sum to head_dim={head_dim}")
        self.rope_dim_split = tuple(rope_dim_split)
        self.rope_num_tiles = _DEFAULT_ABS_ROPE_NUM_TILES
        self.rope_compute_dtype = torch.float32

        self.register_buffer("rope_inv_t", rope_inv_freqs(rope_dim_split[0], rope_base), persistent=False)
        self.register_buffer("rope_inv_h", rope_inv_freqs(rope_dim_split[1], rope_base), persistent=False)
        self.register_buffer("rope_inv_w", rope_inv_freqs(rope_dim_split[2], rope_base), persistent=False)

        self.qkv = QKVProjections(dim)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Channels-last in/out: ``(B, T, H, W, C)``. Norm, scale, RoPE, then windowed attention."""
        batch, t, h, w, _ = x.shape
        kt, kh, kw = self.kernel_size
        if t < kt or h < kh or w < kw:
            raise ValueError(f"3D neighborhood attention requires spatial dims >= kernel_size; "
                             f"got (T,H,W)=({t},{h},{w}) vs kernel={self.kernel_size}")

        q, k, v = self.qkv(x)
        shape = (batch, t, h, w, self.num_heads, self.head_dim)
        q = q.view(shape)
        k = k.view(shape)
        v = v.view(shape)
        q = self.q_norm(q) * self.scale
        k = self.k_norm(k)
        inv_freqs = (
            self.rope_inv_t.to(device=x.device),
            self.rope_inv_h.to(device=x.device),
            self.rope_inv_w.to(device=x.device),
        )
        q = _apply_abs_rope(q, self.rope_dim_split, inv_freqs, self.rope_num_tiles, self.rope_compute_dtype)
        k = _apply_abs_rope(k, self.rope_dim_split, inv_freqs, self.rope_num_tiles, self.rope_compute_dtype)
        # natten's CUTLASS kernels silently produce wrong output for non-contiguous inputs.
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = neighborhood_attention_3d(q, k, v, self.kernel_size)
        out = out.reshape(batch, t, h, w, self.dim)
        return self.proj(out)


class NABlock(nn.Module):
    """Pre-norm transformer block: NA -> SwiGLU MLP with residual adds (deterministic stages)."""

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        rope_dim_split: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim, rope_dim_split=rope_dim_split)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, _swiglu_hidden_dim(dim, mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DiffusionNABlock(nn.Module):
    """Diffusion NA + SwiGLU with shared AdaLN-Zero scale/shift and per-block table residual.

    Combined-pathway semantics: inject projected latent context, then modulated attention and
    MLP residuals. Residuals are ungated (checkpoint gates folded into Linears at load time).
    """

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        context_channels: int,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        rope_dim_split: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.context_channels = context_channels
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(AdaLNZero.NUM_CHUNKS, dim))

        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim, rope_dim_split=rope_dim_split)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, _swiglu_hidden_dim(dim, mlp_ratio))

    def forward(
        self,
        x: torch.Tensor,
        latent_context: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(AdaLNZero.NUM_CHUNKS)
        ]
        x = x + self.context_proj(latent_context)
        x = x + self.attn(self.norm1(x) * (1.0 + scale_msa) + shift_msa)
        x = x + self.mlp(self.norm2(x) * (1.0 + scale_mlp) + shift_mlp)
        return x


# =============================================================================
# Size-floor / pad / crop helpers (official diffusion_tiling.py subset)
# =============================================================================


class _AxisPad:
    """How many elements were added on each side of one axis."""

    __slots__ = ("before", "after")

    def __init__(self, before: int, after: int) -> None:
        self.before = before
        self.after = after


def _resize_axis(x: torch.Tensor, dim: int, size: int, mode: str) -> tuple[torch.Tensor, _AxisPad]:
    """Pad or crop axis ``dim`` so its length becomes ``size`` (official ``resize_axis``).

    ``repeat_last`` appends/drops trailing copies; ``symmetric`` edge-replicates/crops both ends
    (``before = need // 2``, ``after = need - before``).
    """
    if size < 1:
        raise ValueError(f"resize_axis target size must be >= 1, got {size}")
    length = x.shape[dim]
    if length == size:
        return x, _AxisPad(0, 0)

    if length < size:
        need = size - length
        if mode == "repeat_last":
            last = x.narrow(dim, length - 1, 1)
            expand_shape = list(x.shape)
            expand_shape[dim] = need
            return torch.cat([x, last.expand(expand_shape)], dim=dim), _AxisPad(0, need)
        before = need // 2
        after = need - before
        parts: list[torch.Tensor] = []
        if before:
            expand_shape = list(x.shape)
            expand_shape[dim] = before
            parts.append(x.narrow(dim, 0, 1).expand(expand_shape))
        parts.append(x)
        if after:
            expand_shape = list(x.shape)
            expand_shape[dim] = after
            parts.append(x.narrow(dim, length - 1, 1).expand(expand_shape))
        return torch.cat(parts, dim=dim), _AxisPad(before, after)

    need = length - size
    if mode == "repeat_last":
        return x.narrow(dim, 0, size).contiguous(), _AxisPad(0, need)
    before = need // 2
    after = need - before
    return x.narrow(dim, before, size).contiguous(), _AxisPad(before, after)


def _ensure_min_latent_shape(
    latent: torch.Tensor,
    min_sizes: tuple[int, int, int],
) -> tuple[torch.Tensor, tuple[_AxisPad, _AxisPad, _AxisPad]]:
    """Pad latent ``(B, C, T, H, W)`` up to ``min_sizes`` if needed (edge policy per official)."""
    min_t, min_h, min_w = min_sizes
    t_pad = _AxisPad(0, 0)
    h_pad = _AxisPad(0, 0)
    w_pad = _AxisPad(0, 0)
    x = latent
    if x.shape[2] < min_t:
        x, t_pad = _resize_axis(x, 2, min_t, mode="repeat_last")
    if x.shape[3] < min_h:
        x, h_pad = _resize_axis(x, 3, min_h, mode="symmetric")
    if x.shape[4] < min_w:
        x, w_pad = _resize_axis(x, 4, min_w, mode="symmetric")
    return x, (t_pad, h_pad, w_pad)


def _all_stages_min_tile_size(
    stage_kernels: Tuple[Tuple[int, int, int], ...],
    upsamples: Tuple[Tuple[Tuple[int, int, int], int], ...],
    stage5_kernel: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Per-axis latent-grid floor so every stage's NA sees dims >= kernel_size."""
    cumulative = [(1, 1, 1)]
    t, h, w = 1, 1, 1
    for stride, _ in upsamples:
        t, h, w = t * stride[0], h * stride[1], w * stride[2]
        cumulative.append((t, h, w))
    mins = [1, 1, 1]
    for stage_i in range(len(upsamples)):
        strides = cumulative[stage_i]
        for axis in range(3):
            mins[axis] = max(mins[axis], -(-stage_kernels[stage_i][axis] // strides[axis]))
    strides5 = cumulative[len(upsamples)]
    for axis in range(3):
        mins[axis] = max(mins[axis], -(-stage5_kernel[axis] // strides5[axis]))
    return (mins[0], mins[1], mins[2])


def _tile_intervals(length: int, tile_size: int, stride: int, min_size: int) -> list[tuple[int, int]]:
    """Overlapping ``[start, end)`` tiles covering ``[0, length)`` with starts spaced ``stride``.

    A trailing remnant shorter than ``min_size`` merges into the previous tile: neighborhood
    attention rejects grids smaller than its kernel, so a remnant tile cannot stand alone.
    """
    if length <= tile_size:
        return [(0, length)]
    starts = list(range(0, length, stride))
    while len(starts) > 1 and length - starts[-1] < min_size:
        starts.pop()
    return [(start, min(start + tile_size, length)) for start in starts[:-1]] + [(starts[-1], length)]


# =============================================================================
# The diffusion video decoder
# =============================================================================


class LTX2DiffusionVideoDecoder(nn.Module):
    """LTX-2.5 diffusion-based video VAE decoder (Neighborhood-Attention backbone).

    Port of the official ``DiffusionVideoDecoder`` (minimal port of the reference
    ``NADiffusionDecoder``). Stages 1-4 deterministically upsample the latent into a context
    volume; stage 5 runs ``DiffusionNABlock``s that denoise the patchified noised pixels,
    guided by that context via AdaLN-Zero scale/shift. Only stage 5 re-runs per diffusion step.

    Last-frame NATTEN window-shift is mitigated by temporarily replicating the last latent frame
    ``(stage1_K_t // 2) * 2`` times through stages 1-4, then cropping that appendix from context
    before stage 5 — but only down to ``max(original_context_T, stage5_kernel[0])`` so undersized
    clips (e.g. a single latent frame) still satisfy NATTEN's kernel floor. Latents below
    ``stage_min_tile_sizes`` are edge-padded first; leftover pad is cropped from the final pixels.
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        rope_dim_split: Tuple[int, int, int] | None = None,
        stage_channels: Tuple[int, ...] = _L_STAGE_CHANNELS,
        stage_depths: Tuple[int, ...] = _DIFF_STAGE_DEPTHS_DEFAULT,
        stage_kernels: Tuple[Tuple[int, int, int], ...] = _L_STAGE_KERNELS,
        upsamples: Tuple[Tuple[Tuple[int, int, int], int], ...] = _L_UPSAMPLES,
        stage5_kernel: Tuple[int, int, int] = _DIFF_STAGE5_KERNEL_DEFAULT,
        stage5_channels: int | None = None,
        t_emb_dim: int = 384,
        default_num_inference_steps: int = 2,
        timestep_scale_multiplier: float = 1.0,
        model_output_type: str = "v",
    ) -> None:
        super().__init__()
        if not (len(stage_channels) == len(stage_depths) == len(stage_kernels)):
            raise ValueError("stage_channels, stage_depths, and stage_kernels must have equal lengths")
        if len(upsamples) != len(stage_channels) - 1:
            raise ValueError("len(upsamples) must be len(stage_channels) - 1")
        for c in stage_channels:
            if c % head_dim != 0:
                raise ValueError(f"stage_channels {stage_channels} must each be a multiple of head_dim={head_dim}")
        for stage_i, (_, reduction) in enumerate(upsamples):
            expected = stage_channels[stage_i] // reduction
            if stage_channels[stage_i + 1] != expected:
                raise ValueError(f"stage_channels[{stage_i + 1}] must be stage_channels[{stage_i}] // "
                                 f"reduction ({expected}), got {stage_channels[stage_i + 1]}")
        if model_output_type not in ("v", "x0"):
            raise ValueError(f"model_output_type must be 'v' or 'x0', got {model_output_type!r}")

        self.patch_size = patch_size
        self.out_channels = out_channels
        self.stage_channels = tuple(stage_channels)
        self.stage_depths = tuple(stage_depths)
        self.stage5_kernel: Tuple[int, int, int] = tuple(stage5_kernel)
        self.default_num_inference_steps = default_num_inference_steps
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.model_output_type = model_output_type

        # Composed latent-cell -> pixel scale factors (8 / 32 / 32 for the production config).
        self.time_scale = math.prod(stride[0] for stride, _ in upsamples)
        self.spatial_scale_h = math.prod(stride[1] for stride, _ in upsamples) * patch_size
        self.spatial_scale_w = math.prod(stride[2] for stride, _ in upsamples) * patch_size

        # NATTEN last-frame border workaround (see class docstring).
        self._natten_trailing_pad_latent_frames = (stage_kernels[0][0] // 2) * 2
        self.stage_min_tile_sizes = _all_stages_min_tile_size(stage_kernels, upsamples, self.stage5_kernel)

        # Encoder output is per-channel normalized; undo before conv_in (same as the conv decoder).
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        self.conv_in = nn.Linear(in_channels, stage_channels[0], bias=True)

        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage_i in range(len(stage_channels) - 1):
            c = stage_channels[stage_i]
            self.det_stages.append(
                nn.ModuleList([
                    NABlock(dim=c, kernel_size=stage_kernels[stage_i], head_dim=head_dim,
                            rope_dim_split=rope_dim_split) for _ in range(stage_depths[stage_i])
                ]))
            stride, reduction = upsamples[stage_i]
            self.upsamples.append(
                LinearPixelShuffleUpsample(in_channels=c, stride=stride, out_channels_reduction_factor=reduction))

        self.t_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim=t_emb_dim, size_emb_dim=0)

        c_ctx = stage_channels[-1]
        self.context_channels = c_ctx
        c5 = stage5_channels if stage5_channels is not None else c_ctx
        if c5 % head_dim != 0:
            raise ValueError(f"stage5_channels {c5} must be a multiple of head_dim={head_dim}")
        noised_pixel_channels = out_channels * (patch_size**2)

        self.conv_in_x_t = nn.Linear(noised_pixel_channels, c5, bias=True)
        self.shared_adaln = AdaLNZero(dim=c5, t_emb_dim=t_emb_dim)
        self.diff_blocks = nn.ModuleList([
            DiffusionNABlock(dim=c5, kernel_size=self.stage5_kernel, context_channels=c_ctx, head_dim=head_dim,
                             rope_dim_split=rope_dim_split) for _ in range(stage_depths[-1])
        ])
        self.norm_out = nn.RMSNorm(c5, eps=1e-6)
        self.conv_out = nn.Linear(c5, noised_pixel_channels, bias=True)

        # Tiled-decode geometry in output pixels/frames (reference defaults). Stages 1-3 always
        # run on the full latent; stage 4 + stage 5 run per tile with linear seam blending.
        self.use_tiling = False
        self.tile_sample_min_height = 768
        self.tile_sample_min_width = 768
        self.tile_sample_min_num_frames = 80
        self.tile_sample_stride_height = 704
        self.tile_sample_stride_width = 704
        self.tile_sample_stride_num_frames = 56

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def forward_stages_1_to_3(self, latent: torch.Tensor) -> torch.Tensor:
        """Stages 1-3 on the full latent -> stage-4 input feature, channels-last ``(B,T,H,W,C)``.

        Applies the NATTEN trailing ghost pad (replicated last latent frame) before the stages;
        :meth:`forward_stage_4` crops the appendix off the context. Always runs on the full
        volume — tiled decode splits only stages 4-5.
        """
        num_pad = self._natten_trailing_pad_latent_frames
        if num_pad > 0:
            latent, _ = _resize_axis(latent, 2, latent.shape[2] + num_pad, mode="repeat_last")
        latent = self.per_channel_statistics.un_normalize(latent)
        x = latent.permute(0, 2, 3, 4, 1)
        x = self.conv_in(x)
        for blocks, upsample in zip(list(self.det_stages)[:-1], list(self.upsamples)[:-1]):
            for block in blocks:
                x = block(x)
            x = upsample(x, drop_leading_frame=True)
        return x

    def forward_stage_4(
        self,
        x: torch.Tensor,
        drop_leading_frame: bool = True,
        pad_trailing: bool = True,
    ) -> torch.Tensor:
        """Stage 4 on a stage-4-input feature (tile) -> stage-5 context.

        ``drop_leading_frame`` is True only for the tile containing t=0; ``pad_trailing`` is True
        only for the tile carrying the trailing NATTEN ghost frames, which are soft-cropped here
        down to at least ``stage5_kernel[0]`` frames.
        """
        for block in self.det_stages[-1]:
            x = block(x)
        x = self.upsamples[-1](x, drop_leading_frame=drop_leading_frame)
        num_pad = self._natten_trailing_pad_latent_frames
        if pad_trailing and num_pad > 0:
            ghost = num_pad * self.time_scale
            content_t = max(x.shape[1] - ghost, 1)
            keep = min(x.shape[1], max(content_t, self.stage5_kernel[0]))
            x, _ = _resize_axis(x, 1, keep, mode="repeat_last")
        return x

    def forward_diff_step(
        self,
        latent_context: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """One stage-5 diffusion step. Returns the model prediction in pixel space ``(B,C,F,H,W)``."""
        x = _patchify(x_t, self.patch_size).permute(0, 2, 3, 4, 1)
        x = self.conv_in_x_t(x)
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, hidden_dtype=x.dtype)
        modulation = self.shared_adaln(t_emb)
        for block in self.diff_blocks:
            x = block(x, latent_context, modulation)
        x = self.norm_out(x)
        x = self.conv_out(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return _unpatchify(x, self.patch_size)

    def _euler_step(
        self,
        x_t: torch.Tensor,
        model_out: torch.Tensor,
        t_now: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        """One reverse-diffusion Euler update in fp32, cast back to the compute dtype."""
        compute_dtype = x_t.dtype
        dt = (t_now - t_next).view(-1, *([1] * (x_t.ndim - 1))).to(torch.float32)
        x_t_fp32 = x_t.to(torch.float32)
        if self.model_output_type == "v":
            v_pred = model_out
        else:
            # to_velocity: (sample - denoised) / sigma (broadcast instead of .item() for batches).
            sigma = t_now.view(-1, *([1] * (x_t.ndim - 1))).to(torch.float32)
            v_pred = (x_t_fp32 - model_out.to(torch.float32)) / sigma
        return (x_t_fp32 - dt * v_pred).to(compute_dtype)

    def denoise(self, latent_context: torch.Tensor, x_t: torch.Tensor, num_inference_steps: int) -> torch.Tensor:
        """Stage-5 diffusion loop with the official sigma schedule ``linspace(1, 1/N, N)``.

        Intermediate steps are Euler updates; the final step returns the ``x0`` prediction
        directly for ``model_output_type == "x0"``, and an Euler step to sigma 0 for ``"v"``.
        """
        batch_size = x_t.shape[0]
        timesteps = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps,
                                   device=x_t.device, dtype=torch.float32)
        for i in range(num_inference_steps - 1):
            t_now = timesteps[i].expand(batch_size)
            t_next = timesteps[i + 1].expand(batch_size)
            model_out = self.forward_diff_step(latent_context, x_t, t_now).to(torch.float32)
            x_t = self._euler_step(x_t, model_out, t_now, t_next)

        t_now = timesteps[-1].expand(batch_size)
        model_out = self.forward_diff_step(latent_context, x_t, t_now)
        if self.model_output_type == "x0":
            return model_out
        return self._euler_step(x_t, model_out.to(torch.float32), t_now, torch.zeros_like(t_now))

    # ------------------------------------------------------------------
    # Decode entry points
    # ------------------------------------------------------------------

    def _randn(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Official RNG semantics: draw on the generator's device, then move to the compute device."""
        randn_device = generator.device if generator is not None else device
        return torch.randn(shape, dtype=dtype, generator=generator, device=randn_device).to(device)

    def _crop_to_content(
        self,
        pixels: torch.Tensor,
        content_frames: int,
        content_h: int,
        content_w: int,
        h_pad: _AxisPad,
        w_pad: _AxisPad,
    ) -> torch.Tensor:
        """Crop decode output back to the content shape (temporal pad trails; spatial honors pad)."""
        x, _ = _resize_axis(pixels, 2, content_frames, mode="repeat_last")
        before_h = h_pad.before * self.spatial_scale_h
        if before_h + content_h > x.shape[3]:
            raise ValueError(f"H crop out of range: before={before_h}, height={content_h}, got {x.shape[3]}")
        x = x.narrow(3, before_h, content_h)
        before_w = w_pad.before * self.spatial_scale_w
        if before_w + content_w > x.shape[4]:
            raise ValueError(f"W crop out of range: before={before_w}, width={content_w}, got {x.shape[4]}")
        x = x.narrow(4, before_w, content_w)
        return x.contiguous()

    def decode(
        self,
        latent: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Decode normalized latents ``(B, C, T, H, W)`` to pixels ``(B, C, F, H, W)`` in [-1, 1]."""
        num_inference_steps = num_inference_steps or self.default_num_inference_steps
        if self.use_tiling:
            tile_latent_t = self.tile_sample_min_num_frames // self.time_scale
            tile_latent_h = self.tile_sample_min_height // self.spatial_scale_h
            tile_latent_w = self.tile_sample_min_width // self.spatial_scale_w
            if (latent.shape[2] > tile_latent_t or latent.shape[3] > tile_latent_h
                    or latent.shape[4] > tile_latent_w):
                return self.tiled_decode(latent, generator=generator, num_inference_steps=num_inference_steps)
        return self._decode_untiled(latent, generator=generator, num_inference_steps=num_inference_steps)

    def _decode_untiled(
        self,
        latent: torch.Tensor,
        generator: torch.Generator | None,
        num_inference_steps: int,
    ) -> torch.Tensor:
        content_frames = (latent.shape[2] - 1) * self.time_scale + 1
        content_h = latent.shape[3] * self.spatial_scale_h
        content_w = latent.shape[4] * self.spatial_scale_w
        latent, (_t_pad, h_pad, w_pad) = _ensure_min_latent_shape(latent, self.stage_min_tile_sizes)

        features = self.forward_stages_1_to_3(latent)
        context = self.forward_stage_4(features, drop_leading_frame=True, pad_trailing=True)

        pixel_shape = (
            latent.shape[0],
            self.out_channels,
            context.shape[1],
            context.shape[2] * self.patch_size,
            context.shape[3] * self.patch_size,
        )
        x_t = self._randn(pixel_shape, latent.dtype, latent.device, generator)
        pixels = self.denoise(context, x_t, num_inference_steps)
        return self._crop_to_content(pixels, content_frames, content_h, content_w, h_pad, w_pad)

    # ------------------------------------------------------------------
    # Tiled decode (stages 1-3 once on the full volume; stages 4-5 per tile)
    # ------------------------------------------------------------------

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_num_frames: int | None = None,
        tile_sample_stride_height: int | None = None,
        tile_sample_stride_width: int | None = None,
        tile_sample_stride_num_frames: int | None = None,
    ) -> None:
        """Enable tiled decoding (stage 4 + diffusion stage per tile, linear seam blending)."""
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_num_frames = tile_sample_min_num_frames or self.tile_sample_min_num_frames
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self.tile_sample_stride_num_frames = tile_sample_stride_num_frames or self.tile_sample_stride_num_frames

    def disable_tiling(self) -> None:
        self.use_tiling = False

    @staticmethod
    def _blend(a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        """Linearly blend the trailing ``blend_extent`` of ``a`` into the leading part of ``b``."""
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        if blend_extent <= 0:
            return b
        ramp_shape = [1] * b.ndim
        ramp_shape[dim] = blend_extent
        ramp = torch.linspace(0.0, 1.0, blend_extent + 1, device=b.device, dtype=b.dtype)[:-1].view(ramp_shape)
        a_tail = a.narrow(dim, a.shape[dim] - blend_extent, blend_extent)
        b_head = b.narrow(dim, 0, blend_extent)
        b_head.copy_(a_tail * (1.0 - ramp) + b_head * ramp)
        return b

    def tiled_decode(
        self,
        latent: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Tiled decode on the stage-4 input grid (one cell = ``upsamples[3]`` stride x patch pixels).

        Temporal tiles follow the causal frame mapping: only the tile containing t=0 drops the
        temporal upsample's duplicate leading frame, and only the tile containing the video end
        carries the NATTEN ghost frames (cropped in :meth:`forward_stage_4`). A single-step x0
        decode draws fresh noise per tile; a multi-step decode integrates noise across steps, so
        overlapping tiles slice one shared full-canvas ``x_t`` (official RNG semantics).
        """
        num_inference_steps = num_inference_steps or self.default_num_inference_steps
        content_frames = (latent.shape[2] - 1) * self.time_scale + 1
        content_h = latent.shape[3] * self.spatial_scale_h
        content_w = latent.shape[4] * self.spatial_scale_w
        latent, (_t_pad, h_pad, w_pad) = _ensure_min_latent_shape(latent, self.stage_min_tile_sizes)

        batch_size = latent.shape[0]
        up3_stride = self.upsamples[-1].stride
        scale_t = up3_stride[0]
        scale_h = up3_stride[1] * self.patch_size
        scale_w = up3_stride[2] * self.patch_size
        tile_t = max(1, self.tile_sample_min_num_frames // scale_t)
        stride_t = max(1, self.tile_sample_stride_num_frames // scale_t)
        tile_h = max(1, self.tile_sample_min_height // scale_h)
        stride_h = max(1, self.tile_sample_stride_height // scale_h)
        tile_w = max(1, self.tile_sample_min_width // scale_w)
        stride_w = max(1, self.tile_sample_stride_width // scale_w)
        # Every tile must satisfy the stage-4 kernel as-is and the stage-5 kernel after upsample.
        stage4_blocks = self.det_stages[-1]
        stage4_kernel = stage4_blocks[0].attn.kernel_size if len(stage4_blocks) else (1, 1, 1)
        min_sizes = [
            max(stage4_kernel[axis], -(-self.stage5_kernel[axis] // up3_stride[axis])) for axis in range(3)
        ]

        features = self.forward_stages_1_to_3(latent)
        # Ghost latent frames replicate through the first three upsamples' composed temporal stride.
        ghost_frames = self._natten_trailing_pad_latent_frames * math.prod(
            up.stride[0] for up in list(self.upsamples)[:-1])
        num_frames = features.shape[1] - ghost_frames
        height, width = features.shape[2], features.shape[3]

        temporal_tiles = _tile_intervals(num_frames, tile_t, stride_t, min_sizes[0])
        height_tiles = _tile_intervals(height, tile_h, stride_h, min_sizes[1])
        width_tiles = _tile_intervals(width, tile_w, stride_w, min_sizes[2])
        blend_frames = (tile_t - stride_t) * scale_t
        blend_height = (tile_h - stride_h) * scale_h
        blend_width = (tile_w - stride_w) * scale_w

        single_step_x0 = num_inference_steps == 1 and self.model_output_type == "x0"
        x_t_full = None
        if not single_step_x0:
            pixel_frames = num_frames * scale_t - (1 if scale_t == 2 else 0)
            x_t_full = self._randn(
                (batch_size, self.out_channels, pixel_frames, height * scale_h, width * scale_w),
                latent.dtype, latent.device, generator)

        frame_groups: list[torch.Tensor] = []
        for t0, t1 in temporal_tiles:
            is_origin = t0 == 0
            is_trailing = t1 == num_frames
            feature_t1 = features.shape[1] if is_trailing else t1
            rows: list[list[torch.Tensor]] = []
            for h0, h1 in height_tiles:
                row: list[torch.Tensor] = []
                for w0, w1 in width_tiles:
                    context = self.forward_stage_4(
                        features[:, t0:feature_t1, h0:h1, w0:w1],
                        drop_leading_frame=is_origin,
                        pad_trailing=is_trailing,
                    )
                    tile_pixel_shape = (
                        batch_size,
                        self.out_channels,
                        context.shape[1],
                        context.shape[2] * self.patch_size,
                        context.shape[3] * self.patch_size,
                    )
                    if single_step_x0:
                        x_t = self._randn(tile_pixel_shape, latent.dtype, latent.device, generator)
                    else:
                        # Non-origin tiles keep the duplicate leading frame, placing their first
                        # cell one pixel frame before ``t0 * scale_t`` (causal frame mapping).
                        pixel_t0 = t0 * scale_t - (1 if not is_origin and scale_t == 2 else 0)
                        x_t = x_t_full[
                            :, :,
                            pixel_t0:pixel_t0 + tile_pixel_shape[2],
                            h0 * scale_h:h0 * scale_h + tile_pixel_shape[3],
                            w0 * scale_w:w0 * scale_w + tile_pixel_shape[4],
                        ]
                        # Edge-extend when the stage-5 kernel floor stretches context past the
                        # canvas (same edge policy as the latent size floor / ghost pad).
                        x_t, _ = _resize_axis(x_t, 2, tile_pixel_shape[2], mode="repeat_last")
                        x_t, _ = _resize_axis(x_t, 3, tile_pixel_shape[3], mode="symmetric")
                        x_t, _ = _resize_axis(x_t, 4, tile_pixel_shape[4], mode="symmetric")
                    row.append(self.denoise(context, x_t, num_inference_steps))
                rows.append(row)

            result_rows: list[torch.Tensor] = []
            for i, row in enumerate(rows):
                result_row: list[torch.Tensor] = []
                for j, tile in enumerate(row):
                    if i > 0:
                        tile = self._blend(rows[i - 1][j], tile, blend_height, dim=3)
                    if j > 0:
                        tile = self._blend(row[j - 1], tile, blend_width, dim=4)
                    keep_height = stride_h * scale_h if i < len(rows) - 1 else tile.shape[3]
                    keep_width = stride_w * scale_w if j < len(row) - 1 else tile.shape[4]
                    result_row.append(tile[:, :, :, :keep_height, :keep_width])
                result_rows.append(torch.cat(result_row, dim=4))
            frame_groups.append(torch.cat(result_rows, dim=3))

        result: list[torch.Tensor] = []
        for k, group in enumerate(frame_groups):
            if k > 0:
                group = self._blend(frame_groups[k - 1], group, blend_frames, dim=2)
            if k < len(frame_groups) - 1:
                # The origin group is one frame short of ``stride * scale`` (causal mapping).
                keep_frames = stride_t * scale_t - (1 if k == 0 and scale_t == 2 else 0)
                group = group[:, :, :keep_frames]
            result.append(group)
        pixels = torch.cat(result, dim=2)
        return self._crop_to_content(pixels, content_frames, content_h, content_w, h_pad, w_pad)

    def forward(
        self,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        return self.decode(sample, generator=generator, num_inference_steps=num_inference_steps)


# =============================================================================
# Configurators (checkpoint config.json / safetensors metadata driven)
# =============================================================================


def _decoder_kwargs_from_config(config: dict) -> dict:
    """Extract ``LTX2DiffusionVideoDecoder`` kwargs from a ``vae`` configuration dictionary.

    Mirrors the official ``model_configurator._build_diffusion_video_decoder``: real checkpoints
    (``_class_name: "CausalDiffusionVAE"``) nest decoder hyperparameters under ``vae.decoder``;
    the flat ``vae`` dict is the fallback. Architecture fields absent from config keep the class
    defaults by being omitted.
    """
    decoder_config = config.get("decoder", config)
    kwargs: dict[str, Any] = {}
    if "stage_channels" in decoder_config:
        kwargs["stage_channels"] = tuple(decoder_config["stage_channels"])
    if "stage_depths" in decoder_config:
        kwargs["stage_depths"] = tuple(decoder_config["stage_depths"])
    if "stage_kernels" in decoder_config:
        kwargs["stage_kernels"] = tuple(tuple(kernel) for kernel in decoder_config["stage_kernels"])
    if "upsamples" in decoder_config:
        kwargs["upsamples"] = tuple((tuple(stride), reduction) for stride, reduction in decoder_config["upsamples"])
    if "stage5_kernel" in decoder_config:
        kwargs["stage5_kernel"] = tuple(decoder_config["stage5_kernel"])
    if "stage5_channels" in decoder_config:
        kwargs["stage5_channels"] = decoder_config["stage5_channels"]
    if "rope_dim_split" in decoder_config:
        kwargs["rope_dim_split"] = tuple(decoder_config["rope_dim_split"])

    # Some configs list only the deterministic stages' kernels (one per upsample hop, as the
    # diffusers port does); pad with stage5_kernel — the trailing entry is never consumed.
    stage_channels = kwargs.get("stage_channels", _L_STAGE_CHANNELS)
    stage_kernels = kwargs.get("stage_kernels")
    if stage_kernels is not None and len(stage_kernels) == len(stage_channels) - 1:
        stage5_kernel = tuple(kwargs.get("stage5_kernel", _DIFF_STAGE5_KERNEL_DEFAULT))
        kwargs["stage_kernels"] = tuple(stage_kernels) + (stage5_kernel, )

    kwargs.update({
        "in_channels": decoder_config.get("in_channels", config.get("latent_channels", 128)),
        "out_channels": decoder_config.get("out_channels", 3),
        "patch_size": decoder_config.get("patch_size", 4),
        "head_dim": decoder_config.get("head_dim", decoder_config.get("na_head_dim", 64)),
        "t_emb_dim": decoder_config.get("t_emb_dim", 384),
        "default_num_inference_steps": decoder_config.get("default_num_inference_steps", 2),
        "timestep_scale_multiplier": decoder_config.get("timestep_scale_multiplier", 1.0),
        # Sibling of "decoder" at the top vae level, not nested under decoder_config.
        "model_output_type": config.get("model_output_type", "v"),
    })
    return kwargs


class LTX2DiffusionVideoDecoderConfigurator:
    """Configurator for creating the diffusion video decoder from a configuration dictionary."""

    @classmethod
    def from_config(cls, config: dict) -> LTX2DiffusionVideoDecoder:
        config = config.get("vae", config)
        return LTX2DiffusionVideoDecoder(**_decoder_kwargs_from_config(config))


def _build_video_encoder(vae_config: dict) -> nn.Module:
    """Build the conv VideoEncoder from a diffusion-VAE config (official nested-layout aware).

    Mirrors ``model_configurator._prepare_video_encoder_kwargs``: nested ``CausalDiffusionVAE``
    checkpoints put fields under ``vae.encoder`` with ``blocks``/``out_channels`` naming
    (``out_channels`` there is the latent width); flat configs use the LTX-2 conv field names.
    Imported lazily so decoder-only use does not pull the conv-VAE module chain.
    """
    from fastvideo.models.vaes.ltx2vae import (LogVarianceType, NormLayerType, PaddingModeType, VideoEncoder)

    if "encoder" in vae_config:
        encoder_config = vae_config["encoder"]
        out_channels = encoder_config.get("out_channels", vae_config.get("latent_channels", 128))
        encoder_blocks = encoder_config.get("blocks", encoder_config.get("encoder_blocks", []))
    else:
        encoder_config = vae_config
        out_channels = vae_config.get("latent_channels", 128)
        encoder_blocks = vae_config.get("encoder_blocks", [])

    return VideoEncoder(
        convolution_dimensions=encoder_config.get("dims", vae_config.get("dims", 3)),
        in_channels=encoder_config.get("in_channels", 3),
        out_channels=out_channels,
        encoder_blocks=encoder_blocks,
        patch_size=encoder_config.get("patch_size", 4),
        norm_layer=NormLayerType(encoder_config.get("norm_layer", "pixel_norm")),
        latent_log_var=LogVarianceType(encoder_config.get("latent_log_var", "uniform")),
        encoder_spatial_padding_mode=PaddingModeType(
            encoder_config.get("spatial_padding_mode", vae_config.get("encoder_spatial_padding_mode", "zeros"))),
    )


# =============================================================================
# Public API (wrapper exposing FastVideo's VAE encode/decode interface)
# =============================================================================


class LTX2CausalDiffusionVAE(nn.Module):
    """LTX-2.5 VAE with the conv encoder and the diffusion (HQ) decoder.

    Selected when the checkpoint config declares ``_class_name: "CausalDiffusionVAE"`` (metadata
    driven, like the official ``is_diffusion_video_vae``). Latents are interchangeable with the
    conv ``LTX2CausalVideoAutoencoder`` — the encoder half is byte-identical — so pointing the
    pipeline's ``vae`` path at a diffusion checkpoint switches only the decode.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        vae_config = config.get("vae", config)
        self.encoder = _build_video_encoder(vae_config)
        self.decoder = LTX2DiffusionVideoDecoderConfigurator.from_config(config)
        # Per-decode RNG installed by the LTX-2 decoding stage (batch.seed / batch.generator).
        self._decode_generator: torch.Generator | None = None

    @property
    def TIME_SCALE(self) -> int:  # noqa: N802 - parity with LTX2CausalVideoAutoencoder
        return self.decoder.time_scale

    @property
    def SPATIAL_SCALE(self) -> int:  # noqa: N802 - parity with LTX2CausalVideoAutoencoder
        return self.decoder.spatial_scale_h

    def set_decode_generator(self, generator: torch.Generator | None) -> None:
        """Install (or clear) the generator used by subsequent :meth:`decode` calls."""
        self._decode_generator = generator

    def encode(self, x: torch.Tensor):
        from fastvideo.models.vaes.common import DiagonalGaussianDistribution

        means = self.encoder(x)
        zeros = torch.zeros_like(means)
        return DiagonalGaussianDistribution(torch.cat([means, zeros], dim=1), deterministic=True)

    def decode(
        self,
        z: torch.Tensor,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Decode latents to video. ``timestep`` is accepted for conv-decoder interface parity."""
        del timestep
        if generator is None:
            generator = self._decode_generator
        return self.decoder.decode(z, generator=generator, num_inference_steps=num_inference_steps)

    def enable_tiling(self) -> None:
        self.decoder.use_tiling = True

    def disable_tiling(self) -> None:
        self.decoder.use_tiling = False


# Entry point for model registry
EntryClass = LTX2CausalDiffusionVAE
