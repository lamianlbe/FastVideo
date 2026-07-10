#!/usr/bin/env python3
"""Merge a stack of LoRAs (plain and component-gated) into a single-file
LTX-2 checkpoint, replicating the ComfyUI Power-Lora-Loader +
LTX2LoraLoaderAdvanced semantics of the all-in-one workflow.

Semantics (verified against comfy/weight_adapter/lora.py and
kjnodes LTX2LoraLoaderAdvanced):

    W += strength * (alpha/rank if alpha stored else 1.0) * (B @ A)

Component gating replicates kjnodes' substring rules, applied to the
checkpoint key (order matters — first match wins):
    "video_to_audio_attn"            -> group "cross"
    "audio_to_video_attn"            -> group "cross"
    "audio_attn" | "audio_ff.net"    -> group "audio"
    "attn" | "ff.net"                -> group "video"
    otherwise                        -> group "other"

The workflow's two chained gated loads of the same file (0.88 on
video+other, 0.9 on audio+cross) collapse into one gated application with
per-group strengths. NOTE: the workflow only ever uses 0/1 component
gates; kjnodes' fractional-gate path has an alpha/rank quirk we do NOT
reproduce — this script applies fractional group strengths linearly.

Usage (the all-in-one workflow, stages unified on stage-1 strengths):
    python merge_ltx2_lora_stack.py \
        --base 10Eros_v1_bf16.safetensors \
        --output 10Eros_v1_allinone_merged_bf16.safetensors \
        --lora LTX2.3_reasoning_I2V_V3.safetensors 0.5 \
        --lora Penile_Praxis_V4.safetensors 0.35 \
        --lora LTX2-i2v-OralSuite.safetensors 0.06 \
        --gated-lora ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors \
        --gated video=0.88 other=0.88 audio=0.9 cross=0.9

Streaming: peak RAM ~= largest single tensor; the ~44 GB base is never
resident. LoRA deltas are accumulated in fp32 per tensor.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys

import torch
from safetensors import safe_open

DTYPE_SIZES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}
LORA_PREFIX = "diffusion_model."
CKPT_PREFIX = "model.diffusion_model."
GROUPS = ("video", "audio", "cross", "other")


def classify_component(module_path: str) -> str:
    """kjnodes LTX2LoraLoaderAdvanced substring rules (order matters)."""
    if "video_to_audio_attn" in module_path or "audio_to_video_attn" in module_path:
        return "cross"
    if "audio_attn" in module_path or "audio_ff.net" in module_path:
        return "audio"
    if "attn" in module_path or "ff.net" in module_path:
        return "video"
    return "other"


def read_header(path: str) -> tuple[dict, dict]:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    meta = header.pop("__metadata__", {})
    return header, meta


def tensor_nbytes(info: dict) -> int:
    n = 1
    for d in info["shape"]:
        n *= d
    return n * DTYPE_SIZES[info["dtype"]]


def collect_lora_modules(lora_path: str) -> dict[str, dict]:
    header, _ = read_header(lora_path)
    modules: dict[str, dict] = {}
    for key in header:
        if key.endswith(".lora_A.weight"):
            modules.setdefault(key[len(LORA_PREFIX):-len(".lora_A.weight")], {})["A"] = key
        elif key.endswith(".lora_B.weight"):
            modules.setdefault(key[len(LORA_PREFIX):-len(".lora_B.weight")], {})["B"] = key
        elif key.endswith(".alpha"):
            modules.setdefault(key[len(LORA_PREFIX):-len(".alpha")], {})["alpha"] = key
    bad = [m for m, v in modules.items() if "A" not in v or "B" not in v]
    if bad:
        raise ValueError(f"{lora_path}: modules missing A or B: {bad[:5]} ({len(bad)} total)")
    if not modules:
        raise ValueError(f"{lora_path}: no diffusion_model.*.lora_A/lora_B keys found — "
                         f"unsupported LoRA layout (run inspect_safetensors.py on it)")
    return modules


def to_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


class LoraEntry:
    def __init__(self, path: str, group_strengths: dict[str, float]):
        self.path = path
        self.group_strengths = group_strengths  # group -> strength
        self.modules = collect_lora_modules(path)
        self.handle = None
        self.used: set[str] = set()

    def strength_for(self, module_path: str) -> float:
        return self.group_strengths[classify_component(module_path)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lora", nargs=2, action="append", default=[],
                    metavar=("PATH", "STRENGTH"),
                    help="plain LoRA: uniform strength on all its layers (repeatable)")
    ap.add_argument("--gated-lora", default=None,
                    help="component-gated LoRA path (kjnodes-style grouping)")
    ap.add_argument("--gated", nargs="+", default=[],
                    metavar="GROUP=STRENGTH",
                    help="per-group strengths for --gated-lora, e.g. "
                         "video=0.88 other=0.88 audio=0.9 cross=0.9")
    args = ap.parse_args()

    entries: list[LoraEntry] = []
    for path, strength in args.lora:
        s = float(strength)
        entries.append(LoraEntry(path, {g: s for g in GROUPS}))
    if args.gated_lora:
        gated = {g: 1.0 for g in GROUPS}
        for spec in args.gated:
            group, _, val = spec.partition("=")
            if group not in GROUPS or not val:
                sys.exit(f"bad --gated spec {spec!r}; groups: {GROUPS}")
            gated[group] = float(val)
        entries.append(LoraEntry(args.gated_lora, gated))
    if not entries:
        sys.exit("nothing to merge: pass --lora and/or --gated-lora")

    base_header, base_meta = read_header(args.base)

    # module -> [(entry, effective check info)] resolution + orphan report
    plan: dict[str, list[LoraEntry]] = {}
    for entry in entries:
        orphans = []
        for mod in entry.modules:
            ckpt_key = f"{CKPT_PREFIX}{mod}.weight"
            if ckpt_key in base_header:
                plan.setdefault(ckpt_key, []).append(entry)
            else:
                orphans.append(mod)
        print(f"{entry.path}: {len(entry.modules) - len(orphans)} modules matched, "
              f"{len(orphans)} orphaned"
              + (f" e.g. {orphans[:3]}" if orphans else ""))

    keys = list(base_header.keys())
    out_header: dict[str, dict] = {}
    offset = 0
    for k in keys:
        nb = tensor_nbytes(base_header[k])
        out_header[k] = {"dtype": base_header[k]["dtype"], "shape": base_header[k]["shape"],
                         "data_offsets": [offset, offset + nb]}
        offset += nb

    out_meta = dict(base_meta)
    out_meta["merged_lora_stack"] = json.dumps(
        [{"file": e.path.split("/")[-1], "strengths": e.group_strengths} for e in entries])
    header_bytes = json.dumps({"__metadata__": out_meta, **out_header},
                              separators=(",", ":")).encode()
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)

    handles = [safe_open(e.path, framework="pt", device="cpu") for e in entries]
    merged_layers = 0
    with safe_open(args.base, framework="pt", device="cpu") as base_f, \
         open(args.output, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for i, k in enumerate(keys):
            t = base_f.get_tensor(k)
            todo = plan.get(k, [])
            if todo:
                mod = k[len(CKPT_PREFIX):-len(".weight")]
                orig_dtype = t.dtype
                acc = t.to(torch.float32)
                changed = False
                for entry, handle in zip(entries, handles, strict=True):
                    if entry not in todo:
                        continue
                    strength = entry.strength_for(mod)
                    if strength == 0.0:
                        entry.used.add(mod)
                        continue
                    info = entry.modules[mod]
                    A = handle.get_tensor(info["A"]).to(torch.float32)
                    B = handle.get_tensor(info["B"]).to(torch.float32)
                    rank = A.shape[0]
                    # comfy semantics: alpha stored -> scale alpha/rank; else 1.0
                    scale = 1.0
                    if "alpha" in info:
                        scale = float(handle.get_tensor(info["alpha"]).to(torch.float32)) / rank
                    if B.shape[0] != t.shape[0] or A.shape[1] != t.shape[1]:
                        print(f"SHAPE MISMATCH {entry.path} :: {mod}: "
                              f"W{tuple(t.shape)} vs B{tuple(B.shape)}@A{tuple(A.shape)} — skipped")
                        continue
                    acc = acc + (B @ A) * (strength * scale)
                    entry.used.add(mod)
                    changed = True
                if changed:
                    t = acc.to(orig_dtype)
                    merged_layers += 1
            out.write(to_bytes(t))
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(keys)} tensors written ({merged_layers} modified)")

    print(f"\ndone: {merged_layers} base tensors received LoRA deltas")
    for entry in entries:
        unused = set(entry.modules) - entry.used
        if unused:
            print(f"WARNING {entry.path}: {len(unused)} matched modules never applied "
                  f"e.g. {sorted(unused)[:3]}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
