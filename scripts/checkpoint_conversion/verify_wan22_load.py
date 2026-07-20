#!/usr/bin/env python3
"""Fast load-only sanity check for a converted Wan2.2 diffusers directory.

Instantiates each component's architecture from its config and load-checks
the converted weights against it — reporting missing / unexpected / shape-
mismatched keys per component — WITHOUT running a full generation (no VAE
decode, no denoising). Catches key-mapping bugs from the conversion in
seconds instead of after a multi-minute generate.

    python verify_wan22_load.py Wan22-Custom/sfw

Checks transformer, transformer_2 (WanTransformer3DModel) and vae
(AutoencoderKLWan) via diffusers' own from_pretrained/from_config, plus a
raw key diff so a rename bug surfaces even if diffusers is lenient. The
text_encoder (UMT5) is checked as a raw key/shape reconciliation against
the config's expected parameter names.

Needs a CUDA-free load: everything is instantiated on meta/cpu and only
key sets + shapes are compared, so it runs anywhere (even this Mac).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def _load_shards(comp_dir: Path, patterns=("*.safetensors",)) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    files = sorted(f for p in patterns for f in comp_dir.glob(p))
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as h:
            for k in h.keys():
                sl = h.get_slice(k)
                out[k] = tuple(sl.get_shape())
    return out


def _diff(title: str, expected: dict[str, tuple], got: dict[str, tuple]) -> bool:
    ek, gk = set(expected), set(got)
    missing = sorted(ek - gk)
    unexpected = sorted(gk - ek)
    mism = sorted(k for k in ek & gk if expected[k] != got[k])
    ok = not missing and not unexpected and not mism
    flag = "OK" if ok else "FAIL"
    print(f"[{flag}] {title}: expected={len(ek)} got={len(gk)} "
          f"missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(mism)}")
    for label, items in (("missing", missing), ("unexpected", unexpected)):
        for k in items[:8]:
            print(f"    {label}: {k}")
        if len(items) > 8:
            print(f"    ... +{len(items) - 8} more {label}")
    for k in mism[:8]:
        print(f"    shape: {k} expected {expected[k]} got {got[k]}")
    return ok


def _expected_from_model(build_fn, cfg: dict) -> dict[str, tuple]:
    """Instantiate the module on the meta device and read its state_dict
    shapes (no real allocation)."""
    with torch.device("meta"):
        model = build_fn(cfg)
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def check_transformer(comp_dir: Path) -> bool:
    from diffusers import WanTransformer3DModel
    cfg = json.loads((comp_dir / "config.json").read_text())
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    expected = _expected_from_model(lambda c: WanTransformer3DModel.from_config(c), cfg)
    got = _load_shards(comp_dir)
    return _diff(f"{comp_dir.name} (WanTransformer3DModel)", expected, got)


def check_vae(comp_dir: Path) -> bool:
    from diffusers import AutoencoderKLWan
    cfg = json.loads((comp_dir / "config.json").read_text())
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    expected = _expected_from_model(lambda c: AutoencoderKLWan.from_config(c), cfg)
    got = _load_shards(comp_dir)
    return _diff("vae (AutoencoderKLWan)", expected, got)


def check_text_encoder(comp_dir: Path) -> bool:
    # UMT5EncoderModel — instantiate config-only on meta and compare.
    from transformers import UMT5EncoderModel
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(comp_dir))
    with torch.device("meta"):
        model = UMT5EncoderModel(cfg)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    got = _load_shards(comp_dir)
    # UMT5 ties the input embedding: `encoder.embed_tokens.weight` IS
    # `shared.weight`, saved once as `shared.weight` and re-bound by
    # from_pretrained. Its absence from the file is correct, not a missing
    # weight — accept it when the shared source is present with a matching
    # shape.
    tied = "encoder.embed_tokens.weight"
    if tied in expected and tied not in got and expected.get(tied) == got.get("shared.weight"):
        print("    (note: encoder.embed_tokens is tied to shared.weight; re-bound on load, ignoring)")
        expected.pop(tied, None)
    return _diff("text_encoder (UMT5EncoderModel)", expected, got)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1])
    results: dict[str, bool] = {}
    checks = [
        ("transformer", check_transformer),
        ("transformer_2", check_transformer),
        ("vae", check_vae),
        ("text_encoder", check_text_encoder),
    ]
    for name, fn in checks:
        comp = root / name
        if not (comp / "config.json").is_file():
            print(f"[SKIP] {name}: no {comp}/config.json")
            continue
        try:
            results[name] = fn(comp)
        except Exception as err:  # noqa: BLE001
            print(f"[ERROR] {name}: {type(err).__name__}: {err}")
            results[name] = False

    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAIL'}")
    if all(results.values()) and results:
        print("All components load-check clean — key mapping verified. Safe to generate.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
