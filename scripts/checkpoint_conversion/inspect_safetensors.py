#!/usr/bin/env python3
"""Print a compact structural summary of a safetensors file (header only,
no tensor data is loaded — instant even for 40 GB files).

Usage:
    python inspect_safetensors.py file1.safetensors [file2.safetensors ...]

Paste the output back for format analysis. It reports: metadata, tensor
count, dtype histogram, key patterns (numbers collapsed to N), LoRA
rank/alpha samples, and a few concrete example keys with shapes.
"""

from __future__ import annotations

import collections
import json
import re
import struct
import sys


def inspect(path: str) -> None:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    meta = header.pop("__metadata__", {})

    print(f"\n{'=' * 70}\nFILE: {path}")
    print(f"header_bytes={hlen} tensors={len(header)}")
    if meta:
        print("metadata:")
        for k, v in sorted(meta.items()):
            print(f"  {k} = {str(v)[:160]}")

    dtypes = collections.Counter(v["dtype"] for v in header.values())
    print(f"dtypes: {dict(dtypes)}")

    pats: dict[str, list[str]] = collections.defaultdict(list)
    for k in header:
        pats[re.sub(r"\d+", "N", k)].append(k)
    print(f"key patterns ({len(pats)} unique):")
    for p, keys in sorted(pats.items()):
        info = header[keys[0]]
        print(f"  {len(keys):5d}x  {p}  {info['dtype']}{info['shape']}")

    # LoRA specifics: rank distribution + a few alpha values (alpha is tiny,
    # safe to read from the data section).
    ranks = collections.Counter()
    for k, v in header.items():
        if k.endswith("lora_A.weight") or k.endswith("lora_down.weight"):
            ranks[v["shape"][0]] += 1
    if ranks:
        print(f"lora ranks (from lora_A/lora_down dim0): {dict(ranks)}")
    alpha_keys = [k for k in header if k.endswith(".alpha") or k.endswith("lora_alpha")][:4]
    if alpha_keys:
        with open(path, "rb") as f:
            data_start = 8 + hlen
            for k in alpha_keys:
                info = header[k]
                s, e = info["data_offsets"]
                f.seek(data_start + s)
                raw = f.read(e - s)
                if info["dtype"] == "F32":
                    val = struct.unpack("<f", raw)[0]
                elif info["dtype"] == "F64":
                    val = struct.unpack("<d", raw)[0]
                elif info["dtype"] == "BF16":
                    val = struct.unpack("<f", b"\x00\x00" + raw)[0]
                elif info["dtype"] == "F16":
                    import numpy as np
                    val = float(np.frombuffer(raw, dtype=np.float16)[0])
                else:
                    val = f"<{info['dtype']}>"
                print(f"alpha sample: {k} = {val}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        inspect(p)
