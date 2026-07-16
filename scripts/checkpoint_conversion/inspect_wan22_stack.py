#!/usr/bin/env python3
"""Inspect the Wan2.2 SFW/NSFW checkpoint + LoRA stack (shapes/dtypes/keys).

Run this on the box that holds the weights and paste the output back — it is
the ground truth for writing the dequant + LoRA-merge conversion scripts.

    python inspect_wan22_stack.py FILE_OR_DIR [...]
    # e.g.
    python inspect_wan22_stack.py \
        /root/autodl-tmp/models/unet/wan2.2_i2v_A14b_high_noise_*.safetensors \
        /root/autodl-tmp/models/loras/*.safetensors \
        /root/autodl-tmp/models/clip/*umt5*.safetensors

Only needs `pip install safetensors`. Reads headers only (no tensor data), so
it is fast and RAM-light even for 14B files.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from safetensors import safe_open


def classify_key(key: str) -> str:
    k = key.lower()
    if re.search(r"lora_(a|b|down|up)|\.alpha$|^alpha", k):
        return "lora"
    if "scale_weight" in k or k.endswith(".scale") or "scaled_fp8" in k:
        return "fp8_scale"
    if "diff_b" in k or k.endswith(".diff"):
        return "diff"
    return "plain"


def lora_rank_of(f, key: str):
    try:
        sl = f.get_slice(key)
        shape = sl.get_shape()
        return min(shape) if len(shape) == 2 else None
    except Exception:  # noqa: BLE001
        return None


def inspect(path: Path) -> None:
    print(f"\n{'=' * 78}\nFILE {path}  ({path.stat().st_size / 1e9:.2f} GB)")
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        keys = list(f.keys())
        dtypes: Counter = Counter()
        kinds: Counter = Counter()
        prefixes: Counter = Counter()
        lora_ranks: Counter = Counter()
        alpha_samples = []
        scale_samples = []
        for k in keys:
            sl = f.get_slice(k)
            dt = str(sl.get_dtype())
            dtypes[dt] += 1
            kind = classify_key(k)
            kinds[kind] += 1
            # prefix histogram: first 3 dot-separated components w/ block idx
            parts = k.split(".")
            prefixes[".".join(parts[:3])] += 1
            if kind == "lora" and re.search(r"lora_(a|b|down|up)", k.lower()):
                r = lora_rank_of(f, k)
                if r is not None:
                    lora_ranks[r] += 1
            if kind == "lora" and k.lower().endswith("alpha") and len(alpha_samples) < 4:
                alpha_samples.append((k, f.get_tensor(k).item()))
            if kind == "fp8_scale" and len(scale_samples) < 4:
                t = f.get_tensor(k)
                scale_samples.append((k, tuple(t.shape), str(t.dtype),
                                      float(t.flatten()[0]) if t.numel() else None))

        print(f"keys={len(keys)}  dtypes={dict(dtypes)}  kinds={dict(kinds)}")
        if meta:
            small_meta = {k: v[:80] for k, v in list(meta.items())[:6]}
            print(f"metadata: {small_meta}")
        print("top prefixes:")
        for p, c in prefixes.most_common(12):
            print(f"  {c:5d}  {p}")
        if lora_ranks:
            print(f"lora ranks (min-dim histogram): {dict(lora_ranks.most_common(4))}")
        if alpha_samples:
            print(f"alpha samples: {alpha_samples}")
        if scale_samples:
            print(f"fp8 scale samples: {scale_samples}")

        # Representative full keys: first/last few + one of each kind
        shown: dict[str, str] = {}
        for k in keys:
            kind = classify_key(k)
            if kind not in shown:
                shown[kind] = k
        print("sample keys with shapes:")
        for k in list(keys[:4]) + list(shown.values()) + list(keys[-2:]):
            sl = f.get_slice(k)
            print(f"  {str(sl.get_dtype()):10} {str(tuple(sl.get_shape())):24} {k}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    targets: list[Path] = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            targets += sorted(p.rglob("*.safetensors"))
        elif p.is_file():
            targets.append(p)
        else:
            print(f"WARN: not found: {arg}")
    for t in targets:
        try:
            inspect(t)
        except Exception as err:  # noqa: BLE001
            print(f"ERROR inspecting {t}: {err}")
    # Also dump any sibling config.json (diffusers layouts)
    seen_cfg = set()
    for t in targets:
        cfg = t.parent / "config.json"
        if cfg.is_file() and cfg not in seen_cfg:
            seen_cfg.add(cfg)
            try:
                data = json.loads(cfg.read_text())
                print(f"\nCONFIG {cfg}: {json.dumps(data, ensure_ascii=False)[:600]}")
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
