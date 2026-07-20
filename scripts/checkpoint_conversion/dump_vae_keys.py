#!/usr/bin/env python3
"""Dump VAE tensor keys+shapes+dtypes for cross-checking comfy vs diffusers.

    python dump_vae_keys.py <comfy_vae.safetensors> <diffusers_vae_dir_or_file>

Prints a stable sorted listing per file plus a quick summary so we can tell
whether the ComfyUI Wan2_1 VAE is key-identical to the official diffusers
`vae/` (in which case just use the official one) or needs a rename map.
"""

from __future__ import annotations

import sys
from pathlib import Path

from safetensors import safe_open


def load_keys(path_arg: str) -> dict[str, tuple]:
    p = Path(path_arg)
    files: list[Path] = []
    if p.is_dir():
        files = sorted(p.glob("*.safetensors"))
    elif p.is_file():
        files = [p]
    else:
        raise SystemExit(f"not found: {path_arg}")
    out: dict[str, tuple] = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as h:
            for k in h.keys():
                sl = h.get_slice(k)
                out[k] = (tuple(sl.get_shape()), str(sl.get_dtype()))
    return out


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a = load_keys(sys.argv[1])
    b = load_keys(sys.argv[2])
    print(f"# A (comfy)     {sys.argv[1]}: {len(a)} tensors")
    print(f"# B (diffusers) {sys.argv[2]}: {len(b)} tensors\n")

    ak, bk = set(a), set(b)
    common = ak & bk
    shape_mismatch = [k for k in sorted(common) if a[k][0] != b[k][0]]

    print(f"## summary: common={len(common)}  only_in_A={len(ak - bk)}  "
          f"only_in_B={len(bk - ak)}  shape_mismatch={len(shape_mismatch)}")
    if not (ak - bk) and not (bk - ak) and not shape_mismatch:
        print("## VERDICT: key/shape identical — the official diffusers vae/ is a drop-in.")
    print()

    def block(title, keys, table):
        print(f"===== {title} ({len(keys)}) =====")
        for k in sorted(keys):
            shp, dt = table[k]
            print(f"  {dt:8} {str(shp):26} {k}")
        print()

    block("ONLY IN A (comfy)", ak - bk, a)
    block("ONLY IN B (diffusers)", bk - ak, b)
    if shape_mismatch:
        print(f"===== SHAPE MISMATCH ({len(shape_mismatch)}) =====")
        for k in shape_mismatch:
            print(f"  A={a[k]}  B={b[k]}  {k}")
        print()

    # Full A listing (so we have comfy's exact key style even if B is absent)
    block("ALL A KEYS (comfy)", ak, a)


if __name__ == "__main__":
    main()
