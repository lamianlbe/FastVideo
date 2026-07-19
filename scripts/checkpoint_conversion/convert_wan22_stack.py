#!/usr/bin/env python3
"""Dequantize + LoRA-merge the Wan2.2 I2V A14B ComfyUI stack into
diffusers-format transformers that FastVideo loads directly.

Inputs (verified against the real stack via inspect_wan22_stack.py):
- Base high/low-noise checkpoints: ComfyUI-native Wan keys
  (``blocks.N.self_attn.q.weight`` ...), 400 linears in fp8_e4m3 with 0-dim
  bf16 ``.scale_weight`` companions; biases/norms/etc in f32/bf16.
- LoRAs in three dialects, all merged here:
  1. ``diffusion_model.*`` + ``lora_down/lora_up`` (+ optional ``diff_b``
     bias deltas, ``.diff`` norm deltas, ``diff_m`` modulation deltas).
     No alpha keys -> scale = strength (alpha defaults to rank).
     Targets missing from the base (e.g. Wan2.1 ``k_img/v_img`` heads on a
     2.2 base) are SKIPPED, matching ComfyUI behavior.
  2. kohya ``lora_unet__blocks_0_cross_attn_k`` names with per-layer alpha
     and dynamic ranks -> scale = strength * alpha / rank. Underscore names
     are resolved against the base key set (no guessy substitution).

Output: one directory per call in diffusers WanTransformer3DModel naming
(bf16, sharded safetensors + index + config.json copied from an official
Wan2.2 diffusers repo). Run four times (sfw/nsfw x high/low):

  python convert_wan22_stack.py \
      --base wan2.2_i2v_A14b_high_noise_..._comfyui_1030.safetensors \
      --lora 'Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors:2.0' \
      --lora 'Wan2.2-Fun-A14B-InP-HIGH-MPS_resized_dynamic_avg_rank_21_bf16.safetensors:0.5' \
      --base-config Wan2.2-I2V-A14B-Diffusers/transformer \
      --out out/sfw/transformer

CPU-only, streams key-by-key (peak RAM ~= one layer + one shard).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# ComfyUI-native -> diffusers key mapping (ordered: longest/most-specific
# first; applied as prefix-aware substring rewrites like diffusers'
# convert_wan_to_diffusers.py).
# ---------------------------------------------------------------------------
RENAME_RULES: list[tuple[str, str]] = [
    ("time_embedding.0.", "condition_embedder.time_embedder.linear_1."),
    ("time_embedding.2.", "condition_embedder.time_embedder.linear_2."),
    ("text_embedding.0.", "condition_embedder.text_embedder.linear_1."),
    ("text_embedding.2.", "condition_embedder.text_embedder.linear_2."),
    ("time_projection.1.", "condition_embedder.time_proj."),
    ("head.modulation", "scale_shift_table"),
    ("head.head.", "proj_out."),
    # Wan2.1 I2V image branch (absent on 2.2 A14B; kept for completeness).
    ("img_emb.proj.0.", "condition_embedder.image_embedder.norm1."),
    ("img_emb.proj.1.", "condition_embedder.image_embedder.ff.net.0.proj."),
    ("img_emb.proj.3.", "condition_embedder.image_embedder.ff.net.2."),
    ("img_emb.proj.4.", "condition_embedder.image_embedder.norm2."),
    (".cross_attn.norm_k_img.", ".attn2.norm_added_k."),
    (".cross_attn.k_img.", ".attn2.add_k_proj."),
    (".cross_attn.v_img.", ".attn2.add_v_proj."),
    (".self_attn.norm_q.", ".attn1.norm_q."),
    (".self_attn.norm_k.", ".attn1.norm_k."),
    (".self_attn.q.", ".attn1.to_q."),
    (".self_attn.k.", ".attn1.to_k."),
    (".self_attn.v.", ".attn1.to_v."),
    (".self_attn.o.", ".attn1.to_out.0."),
    (".cross_attn.norm_q.", ".attn2.norm_q."),
    (".cross_attn.norm_k.", ".attn2.norm_k."),
    (".cross_attn.q.", ".attn2.to_q."),
    (".cross_attn.k.", ".attn2.to_k."),
    (".cross_attn.v.", ".attn2.to_v."),
    (".cross_attn.o.", ".attn2.to_out.0."),
    (".ffn.0.", ".ffn.net.0.proj."),
    (".ffn.2.", ".ffn.net.2."),
    (".norm3.", ".norm2."),          # comfy norm3 == diffusers norm2 (pre-cross-attn LN)
    (".modulation", ".scale_shift_table"),
]


def to_diffusers_key(key: str) -> str:
    for old, new in RENAME_RULES:
        if old in key:
            key = key.replace(old, new)
    return key


# ---------------------------------------------------------------------------
# LoRA parsing
# ---------------------------------------------------------------------------
LORA_PREFIXES = ("diffusion_model.", "model.diffusion_model.")


class LoraFile:
    """One LoRA safetensors, normalized to comfy-native target names.

    Exposes per-target entries:
      target -> {"lora": (down_key, up_key, alpha_or_None),
                 "diff_b": key, "diff": key, "diff_m": key}
    """

    def __init__(self, path: Path, strength: float, base_keys: set[str]):
        self.path = path
        self.strength = float(strength)
        self.handle = safe_open(str(path), framework="pt", device="cpu")
        keys = list(self.handle.keys())
        self._keyset = set(keys)
        self.entries: dict[str, dict] = {}
        self.skipped: set[str] = set()

        kohya = [k for k in keys if k.startswith("lora_unet_")]
        if kohya:
            self._parse_kohya(keys, base_keys)
        else:
            self._parse_native(keys, base_keys)

    # -- native dialect: diffusion_model.blocks.N....{lora_down,lora_up,diff_b,diff,diff_m}
    def _parse_native(self, keys: list[str], base_keys: set[str]) -> None:
        for k in keys:
            name = k
            for p in LORA_PREFIXES:
                if name.startswith(p):
                    name = name[len(p):]
                    break
            if name.endswith(".lora_down.weight") or name.endswith(".lora_A.weight"):
                target = re.sub(r"\.(lora_down|lora_A)\.weight$", "", name)
                up = k.replace("lora_down", "lora_up").replace("lora_A", "lora_B")
                # This stack's native-dialect files carry no alpha keys
                # (alpha defaults to rank -> factor 1); honor one if present.
                alpha = f"{k.rsplit('.lora_', 1)[0]}.alpha"
                alpha_key = alpha if alpha in self._keyset else None
                self._add(target, base_keys, "lora", (k, up, alpha_key), suffix=".weight")
            elif name.endswith(".diff_b"):
                self._add(name[:-len(".diff_b")], base_keys, "diff_b", k, suffix=".bias")
            elif name.endswith(".diff_m"):
                # ComfyUI convention: blocks.N.diff_m patches the per-block
                # modulation table (blocks.N.modulation).
                self._add(f"{name[:-len('.diff_m')]}.modulation", base_keys, "diff_m", k, suffix="")
            elif name.endswith(".diff"):
                self._add(name[:-len(".diff")], base_keys, "diff", k, suffix=".weight")

    # -- kohya dialect: lora_unet__blocks_0_cross_attn_k.{lora_down,lora_up,alpha}
    def _parse_kohya(self, keys: list[str], base_keys: set[str]) -> None:
        # Resolve underscore names by lookup against the base's weight keys
        # (layer names themselves contain underscores, so naive replacement
        # is ambiguous — a table is exact).
        table = {}
        for bk in base_keys:
            if bk.endswith(".weight"):
                dotted = bk[:-len(".weight")]
                table[dotted.replace(".", "_")] = dotted
        for k in keys:
            if not k.endswith(".lora_down.weight"):
                continue
            stem = k[:-len(".lora_down.weight")]
            uname = stem[len("lora_unet_"):].lstrip("_")
            target = table.get(uname)
            if target is None:
                self.skipped.add(uname)
                continue
            up = f"{stem}.lora_up.weight"
            alpha = f"{stem}.alpha"
            alpha_key = alpha if alpha in set(keys) else None
            self._add(target, base_keys, "lora", (k, up, alpha_key), suffix=".weight")

    def _add(self, target: str, base_keys: set[str], kind: str, payload, suffix: str) -> None:
        base_key = f"{target}{suffix}" if suffix else target
        if base_key not in base_keys:
            self.skipped.add(base_key)
            return
        self.entries.setdefault(base_key, {})[kind] = payload

    # -- application ------------------------------------------------------
    def delta_for(self, base_key: str) -> torch.Tensor | None:
        entry = self.entries.get(base_key)
        if entry is None:
            return None
        total: torch.Tensor | None = None
        if "lora" in entry:
            down_k, up_k, alpha_k = entry["lora"]
            down = self.handle.get_tensor(down_k).to(torch.float32)
            up = self.handle.get_tensor(up_k).to(torch.float32)
            rank = down.shape[0]
            scale = self.strength
            if alpha_k is not None:
                scale *= float(self.handle.get_tensor(alpha_k)) / rank
            total = scale * (up @ down)
        for kind in ("diff_b", "diff", "diff_m"):
            if kind in entry:
                d = self.handle.get_tensor(entry[kind]).to(torch.float32) * self.strength
                total = d if total is None else total + d
        return total


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert(base_path: Path, loras: list[LoraFile], out_dir: Path,
            base_config: Path | None, shard_gb: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"dequant": 0, "merged": 0, "plain": 0}

    with safe_open(str(base_path), framework="pt", device="cpu") as base:
        base_keys = [k for k in base.keys() if not k.endswith(".scale_weight")]
        scale_keys = {k for k in base.keys() if k.endswith(".scale_weight")}

        shards: list[dict[str, torch.Tensor]] = [{}]
        shard_bytes = [0]
        limit = int(shard_gb * 1e9)

        for key in sorted(base_keys):
            t = base.get_tensor(key)
            if t.dtype == torch.float8_e4m3fn:
                sk = f"{key[:-len('.weight')]}.scale_weight" if key.endswith(".weight") else None
                if sk not in scale_keys:
                    raise RuntimeError(f"fp8 tensor without scale: {key}")
                scale = base.get_tensor(sk).to(torch.float32)
                t = t.to(torch.float32) * scale
                stats["dequant"] += 1
            else:
                t = t.to(torch.float32)
                stats["plain"] += 1

            for lora in loras:
                delta = lora.delta_for(key)
                if delta is not None:
                    if delta.shape != t.shape:
                        raise RuntimeError(f"delta shape {tuple(delta.shape)} != base "
                                           f"{tuple(t.shape)} for {key} ({lora.path.name})")
                    t = t + delta
                    stats["merged"] += 1

            dk = to_diffusers_key(key)
            t = t.to(torch.bfloat16).contiguous()
            nbytes = t.numel() * t.element_size()
            if shard_bytes[-1] + nbytes > limit and shards[-1]:
                shards.append({})
                shard_bytes.append(0)
            shards[-1][dk] = t
            shard_bytes[-1] += nbytes

    # Write shards + index
    n = len(shards)
    weight_map: dict[str, str] = {}
    total = 0
    for i, shard in enumerate(shards, 1):
        fname = (f"diffusion_pytorch_model-{i:05d}-of-{n:05d}.safetensors"
                 if n > 1 else "diffusion_pytorch_model.safetensors")
        save_file(shard, str(out_dir / fname))
        for k, v in shard.items():
            weight_map[k] = fname
            total += v.numel() * v.element_size()
        print(f"  wrote {fname} ({sum(v.numel() * v.element_size() for v in shard.values()) / 1e9:.2f} GB)")
    if n > 1:
        index = {"metadata": {"total_size": total}, "weight_map": weight_map}
        (out_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    if base_config is not None:
        cfg = base_config / "config.json" if base_config.is_dir() else base_config
        shutil.copyfile(cfg, out_dir / "config.json")
        print(f"  copied config from {cfg}")
    else:
        print("  WARNING: no --base-config given; copy the official transformer config.json manually")

    print(f"  stats: {stats}")
    for lora in loras:
        applied = len(lora.entries)
        print(f"  {lora.path.name}: strength={lora.strength} targets applied={applied} "
              f"skipped={len(lora.skipped)}")
        if lora.skipped:
            sample = sorted(lora.skipped)[:5]
            print(f"    skipped sample (expected for 2.1-only branches like k_img): {sample}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="ComfyUI fp8_scaled base checkpoint (high or low noise)")
    ap.add_argument("--lora", action="append", default=[],
                    help="'path:strength' — repeat in chain order (e.g. lightx2v:2.0 then fun:0.5)")
    ap.add_argument("--out", required=True, help="Output transformer directory (diffusers layout)")
    ap.add_argument("--base-config", default=None,
                    help="Official Wan2.2 diffusers transformer dir (its config.json is copied)")
    ap.add_argument("--shard-gb", type=float, default=9.5, help="Max shard size in GB")
    args = ap.parse_args()

    base_path = Path(args.base)
    with safe_open(str(base_path), framework="pt", device="cpu") as f:
        base_keys = {k for k in f.keys() if not k.endswith(".scale_weight")}

    loras = []
    for spec in args.lora:
        path_str, _, strength = spec.rpartition(":")
        loras.append(LoraFile(Path(path_str), float(strength), base_keys))

    print(f"base: {base_path.name} ({len(base_keys)} tensors) + {len(loras)} lora(s)")
    convert(base_path, loras, Path(args.out),
            Path(args.base_config) if args.base_config else None, args.shard_gb)


if __name__ == "__main__":
    main()
