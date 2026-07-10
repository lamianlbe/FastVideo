# SPDX-License-Identifier: Apache-2.0
"""LTX-2 latent anchor (port of the ComfyUI 10s-nodes LTXLatentAnchorAware).

Pure activation-space identity stabilizer: inside selected transformer
blocks, every video token of the attn1 output is pulled additively toward
its most-similar token of the anchor frame (centered-cosine matching),
gated by a similarity sigmoid, an optional reference-image energy mask, and
a per-frame temporal-distance falloff. At a chosen sampling step the anchor
frame's token matrix is snapshotted per block and reused as a frozen pull
target for the remaining steps.

Eager-only: the per-block snapshot cache mutates a Python dict inside the
block forward, which is incompatible with torch.compile(fullgraph=True).
The denoising stage rejects the combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

# Sigmoid sharpness constants from the ComfyUI node.
_TRACK_SHARPNESS = 8.0
_ENERGY_SHARPNESS = 16.0


@dataclass
class LatentAnchorContext:
    """Carried through the transformer forward for the selected blocks.

    ``cache`` maps block idx -> (anchor_flat [1,K,D], anchor_mean [1,1,D]);
    populated when ``capture`` is True, used whenever an entry exists.
    ``token_offset`` skips a reference-token prefix so the anchor grid maps
    exactly onto the target's (F, H, W) token grid.
    """
    strength: float
    blocks: list[int]
    frames: int
    height_tokens: int
    width_tokens: int
    similarity_threshold: float = 0.5
    decay_with_distance: float = 0.15
    energy_threshold: float = 0.3
    anchor_frame: int = 0
    energy_grid: torch.Tensor | None = None  # [H_tok, W_tok] in [0, 1]
    token_offset: int = 0
    capture: bool = False
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


def extract_energy_map(ref_latent: torch.Tensor, anchor_frame: int = 0) -> torch.Tensor:
    """[B,C,F,H,W] reference latent -> [H,W] normalized spatial energy.

    Centered per-channel spatial mean, L2 across channels, min-max
    normalized (verbatim ComfyUI `_extract_energy_map`).
    """
    f_idx = min(int(anchor_frame), ref_latent.shape[2] - 1)
    frame = ref_latent[:1, :, f_idx, :, :].to(torch.float32)  # [1, C, H, W]
    spatial_mean = frame.mean(dim=(2, 3), keepdim=True)
    centered = frame - spatial_mean
    energy = centered.norm(dim=1)[0]  # [H, W]
    e_min, e_max = energy.min(), energy.max()
    return (energy - e_min) / (e_max - e_min + 1e-6)


def resample_energy_map(energy: torch.Tensor, height_tokens: int, width_tokens: int) -> torch.Tensor:
    if tuple(energy.shape) == (height_tokens, width_tokens):
        return energy
    return F.interpolate(
        energy[None, None],
        size=(height_tokens, width_tokens),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def apply_latent_anchor(
    x: torch.Tensor,
    ctx: LatentAnchorContext,
    block_idx: int,
) -> torch.Tensor:
    """Apply the anchor pull to an attn1 output ``x`` of shape [B, seq, D].

    Returns ``x`` unchanged when the token grid does not match (defensive,
    mirrors the ComfyUI hook's silent abort).
    """
    off = ctx.token_offset
    fdim, hdim, wdim = ctx.frames, ctx.height_tokens, ctx.width_tokens
    n_target = fdim * hdim * wdim
    if x.shape[1] - off != n_target:
        return x

    prefix = x[:, :off] if off else None
    grid = x[:, off:].reshape(x.shape[0], fdim, hdim, wdim, x.shape[-1])
    bsz, _, _, _, dim = grid.shape
    anchor_idx = min(max(int(ctx.anchor_frame), 0), fdim - 1)

    compute_dtype = torch.float32
    grid_f = grid.to(compute_dtype)

    cached = ctx.cache.get(block_idx)
    if cached is not None:
        anchor_flat = cached[0].to(device=x.device, dtype=compute_dtype).expand(bsz, -1, -1)
        anchor_mean = cached[1].to(device=x.device, dtype=compute_dtype).expand(bsz, -1, -1)
    else:
        anchor_flat = grid_f[:, anchor_idx].reshape(bsz, hdim * wdim, dim)
        anchor_mean = anchor_flat.mean(dim=1, keepdim=True)
        if ctx.capture:
            ctx.cache[block_idx] = (
                anchor_flat[:1].detach().clone(),
                anchor_mean[:1].detach().clone(),
            )

    # Centered cosine matching of every token against the anchor frame.
    all_flat = grid_f.reshape(bsz, n_target, dim)
    frame_mean = grid_f.mean(dim=(2, 3), keepdim=True)  # [B, F, 1, 1, D]
    centered_all = (grid_f - frame_mean).reshape(bsz, n_target, dim)
    centered_anchor = anchor_flat - anchor_mean
    all_norm = F.normalize(centered_all, dim=-1, eps=1e-6)
    anchor_norm = F.normalize(centered_anchor, dim=-1, eps=1e-6)
    sim = torch.bmm(all_norm, anchor_norm.transpose(1, 2))  # [B, N, K]
    best_sim, best_idx = sim.max(dim=-1)
    gathered = torch.gather(
        anchor_flat,
        1,
        best_idx.unsqueeze(-1).expand(-1, -1, dim),
    )

    mask = torch.sigmoid((best_sim - ctx.similarity_threshold) * _TRACK_SHARPNESS)
    mask = mask.reshape(bsz, fdim, hdim, wdim, 1)

    if ctx.energy_grid is not None and ctx.energy_threshold > 0.0:
        e_grid = ctx.energy_grid.to(device=x.device, dtype=compute_dtype)
        e_factor = torch.sigmoid((e_grid - ctx.energy_threshold) * _ENERGY_SHARPNESS)
        mask = mask * e_factor.reshape(1, 1, hdim, wdim, 1)

    dist = (torch.arange(fdim, device=x.device, dtype=compute_dtype) - anchor_idx).abs() / max(1, fdim - 1)
    fs = ctx.strength * (1.0 - ctx.decay_with_distance * dist).clamp(min=0.0)
    fs[anchor_idx] = 0.0

    diff = gathered.reshape(bsz, fdim, hdim, wdim, dim) - grid_f
    out = grid_f + fs.view(1, fdim, 1, 1, 1) * mask * diff
    out = out.reshape(bsz, n_target, dim).to(x.dtype)
    if prefix is not None:
        out = torch.cat([prefix, out], dim=1)
    return out


__all__ = [
    "LatentAnchorContext",
    "apply_latent_anchor",
    "extract_energy_map",
    "resample_energy_map",
]
