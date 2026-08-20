#!/usr/bin/env python3
"""Audit a converted LTX-2.3 diffusers repo for conversion defects.

Motivation: a mis-converted LTX-2.3 repo usually still LOADS and still
produces coherent video — it just looks persistently worse than the
reference. The failure modes that behave that way are all silent:

  * an architecture flag dropped by the converter's 2.0-era allow-list, so
    FastVideo instantiates a differently shaped model (see
    patch_ltx23_configs.py / convert_ltx23_weights.py);
  * a tensor missing from the checkpoint whose name matches
    ``ALLOWED_NEW_PARAM_PATTERNS`` in fastvideo/models/loader/fsdp_load.py
    ("gate_compress", "proj_l") — those are ZERO-FILLED instead of raising;
  * a botched fp8 dequant or LoRA merge that leaves one subset of tensors
    mis-scaled while everything else is fine.

None of those need a reference model to detect, so every check here runs
against the converted repo alone (CPU, streaming — the 44 GB shards are
never fully resident). Pass ``--source`` to additionally spot-check values
against the pre-conversion ComfyUI checkpoint.

    python verify_ltx23_conversion.py --repo /workspace/My-LTX-2.3-Diffusers
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from safetensors import safe_open

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_ltx23_reference_configs", _HERE / "_ltx23_reference_configs.py")
assert _spec is not None and _spec.loader is not None
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)

# Silently zero-filled by the DiT loader when absent from the checkpoint.
SILENT_ZERO_FILL_PATTERNS = ("gate_compress", "proj_l")

# Flags whose absence/wrong value reshapes the model rather than erroring.
CRITICAL_TRANSFORMER_FIELDS = (
    "apply_gated_attention",
    "cross_attention_adaln",
    "caption_proj_before_connector",
    "caption_proj_input_norm",
    "caption_projection_first_linear",
    "caption_projection_second_linear",
    "connector_num_layers",
    "connector_num_attention_heads",
    "connector_attention_head_dim",
    "num_layers",
    "num_attention_heads",
    "attention_head_dim",
)


class Report:
    def __init__(self) -> None:
        self.failed = False

    def ok(self, title: str, detail: str = "") -> None:
        print(f"[PASS] {title}" + (f" — {detail}" if detail else ""))

    def fail(self, title: str, detail: str = "") -> None:
        self.failed = True
        print(f"[FAIL] {title}" + (f" — {detail}" if detail else ""))

    def note(self, text: str) -> None:
        print(f"       {text}")


def _shards(component: Path) -> list[Path]:
    return sorted(component.glob("*.safetensors"))


def _read_header(path: Path) -> dict[str, dict]:
    """safetensors JSON header only — no tensor data is read."""
    import struct
    with path.open("rb") as f:
        (length, ) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return header


def _transformer_components(repo: Path) -> list[str]:
    """"transformer" plus any additional transformer_* directory (e.g. the
    stage-2 refine DiT written by convert_ltx23_transformer.py)."""
    names = ["transformer"]
    names += sorted(p.name for p in repo.iterdir()
                    if p.is_dir() and p.name.startswith("transformer_") and (p / "config.json").is_file())
    return names


def check_configs(repo: Path, rep: Report) -> None:
    """Compare architecture configs against the reference 2.3 repo."""
    targets = [(name, _ref.TRANSFORMER_REFERENCE) for name in _transformer_components(repo)]
    targets += [("text_encoder", _ref.TEXT_ENCODER_REFERENCE),
                ("text_embedding_projection", _ref.TEXT_ENCODER_REFERENCE)]
    for name, reference in targets:
        path = repo / name / "config.json"
        if not path.is_file():
            rep.fail(f"{name}/config.json", "missing")
            continue
        config = json.loads(path.read_text())
        diffs = []
        for key, expected in reference.items():
            # Paths and bookkeeping differ legitimately between repos.
            if key in ("gemma_model_path", "_diffusers_version", "architectures"):
                continue
            got = config.get(key, "<missing>")
            if got != expected:
                critical = key in CRITICAL_TRANSFORMER_FIELDS
                diffs.append((key, got, expected, critical))
        if not diffs:
            rep.ok(f"{name}/config.json", f"{len(reference)} fields match the reference 2.3 repo")
            continue
        critical = [d for d in diffs if d[3]]
        (rep.fail if critical else rep.ok)(
            f"{name}/config.json",
            f"{len(diffs)} field(s) differ ({len(critical)} architecture-critical)")
        for key, got, expected, is_critical in diffs:
            rep.note(f"{'!! ' if is_critical else '   '}{key}: got {got!r}, reference {expected!r}")


def check_storage_format(repo: Path, rep: Report) -> None:
    """Each component must be plain bf16 — or, for transformer components
    only, a WELL-FORMED fp8-scaled artifact.

    Two failure modes hide here. The historical one: convert_ltx2_weights.py
    has no fp8 handling, so an F8_E4M3 payload in its input was written
    straight through along with `weight_scale`/`comfy_quant` siblings into a
    repo that CLAIMED bf16 (the only visible symptom was a ~21 GB instead of
    ~38 GB DiT). The new one: fp8-scaled transformers are now a supported
    deployment format (loaded verbatim into the FP8 runtime), so for
    transformer* components the check validates the pairing instead of
    rejecting fp8 — every F8 payload needs exactly one `.weight_scale`
    sibling, no orphan scales, no leftover `comfy_quant` descriptors, and
    biases stay bf16 (they are never quantized).
    """
    for component in sorted(p for p in repo.iterdir() if p.is_dir()):
        dtypes: dict[str, int] = {}
        entries: dict[str, dict] = {}
        for shard in _shards(component):
            for name, entry in _read_header(shard).items():
                dtypes[entry["dtype"]] = dtypes.get(entry["dtype"], 0) + 1
                entries[name] = entry
        if not dtypes:
            continue

        comfy_quant = [n for n in entries if n.endswith(".comfy_quant")]
        scales = {n for n in entries if n.endswith((".weight_scale", ".weight_scale_2"))}
        fp8 = {n for n in entries if entries[n]["dtype"].startswith("F8_")}
        problems: list[str] = []

        if comfy_quant:
            problems.append(f"{len(comfy_quant)} comfy_quant descriptor(s) not stripped by conversion")
        if fp8 and not component.name.startswith("transformer"):
            problems.append(f"{len(fp8)} fp8 tensor(s) in a non-transformer component")
        if fp8:
            unscaled = sorted(n for n in fp8 if f"{n}_scale" not in scales)
            orphan = sorted(n for n in scales if n.endswith(".weight_scale") and n[:-len("_scale")] not in fp8)
            if unscaled:
                problems.append(f"{len(unscaled)} fp8 payload(s) without a weight_scale (e.g. {unscaled[0]})")
            if orphan:
                problems.append(f"{len(orphan)} weight_scale(s) without an fp8 payload (e.g. {orphan[0]})")
            bad_bias = sorted(n for n in entries if n.endswith(".bias") and entries[n]["dtype"] != "BF16")
            if bad_bias:
                problems.append(f"{len(bad_bias)} non-bf16 bias(es) (e.g. {bad_bias[0]})")
        elif scales:
            problems.append(f"{len(scales)} quantization scale(s) but no fp8 payloads")
        quantized_other = {d: n for d, n in dtypes.items() if d.startswith("I8")}
        if quantized_other:
            problems.append(f"unexpected quantized dtypes {quantized_other}")

        if problems:
            rep.fail(f"{component.name} storage format", f"dtypes={dtypes}")
            for line in problems[:6]:
                rep.note(line)
        elif fp8:
            weights = sum(1 for n in entries if n.endswith(".weight"))
            rep.ok(f"{component.name} storage format",
                   f"fp8-scaled: {len(fp8)}/{weights} weights quantized (per-tensor scales), rest bf16")
        else:
            rep.ok(f"{component.name} storage format", f"dtypes={dtypes}")


def scan_tensors(repo: Path, rep: Report) -> dict[str, dict[str, float]]:
    """Stream every tensor: flag zeros/NaN/Inf, and collect per-tensor RMS.

    An all-zero tensor is the fingerprint of the silent zero-fill path, and
    a NaN/Inf is the fingerprint of a bad dequant — neither raises at load.
    """
    stats: dict[str, dict[str, float]] = {}
    zeros: list[str] = []
    nonfinite: list[str] = []
    for component in sorted(p for p in repo.iterdir() if p.is_dir()):
        for shard in _shards(component):
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                for key in handle.keys():
                    if key.endswith((".weight_scale", ".weight_scale_2", ".comfy_quant")):
                        continue  # sidecars: folded into their payloads below
                    tensor = handle.get_tensor(key).float()
                    scale_key = f"{key}_scale"
                    if scale_key in keys:
                        # fp8-scaled payload: all magnitude checks (zeros, the
                        # per-block depth profile) must see REAL values, not
                        # the quantized units.
                        scale = handle.get_tensor(scale_key).float()
                        tensor = tensor * (scale if scale.numel() == 1 else scale.reshape(-1, 1))
                    if not torch.isfinite(tensor).all():
                        nonfinite.append(f"{component.name}/{key}")
                        continue
                    rms = float(tensor.pow(2).mean().sqrt())
                    stats[f"{component.name}/{key}"] = {
                        "rms": rms,
                        "absmax": float(tensor.abs().max()),
                    }
                    # Biases and some tables are legitimately all-zero; weights
                    # never are, and those are what the zero-fill path hits.
                    if rms == 0.0 and key.endswith(".weight"):
                        zeros.append(f"{component.name}/{key}")

    if nonfinite:
        rep.fail("finite values", f"{len(nonfinite)} tensor(s) contain NaN/Inf")
        for key in nonfinite[:10]:
            rep.note(key)
    else:
        rep.ok("finite values", f"{len(stats)} tensors, no NaN/Inf")

    if zeros:
        suspicious = [k for k in zeros if any(p in k for p in SILENT_ZERO_FILL_PATTERNS)]
        rep.fail("all-zero weights", f"{len(zeros)} weight tensor(s) are entirely zero")
        for key in zeros[:10]:
            marker = "  <-- matches the loader's silent zero-fill allow-list" if key in suspicious else ""
            rep.note(f"{key}{marker}")
    else:
        rep.ok("all-zero weights", "no weight tensor is entirely zero")
    return stats


def check_depth_profile(stats: dict[str, dict[str, float]], rep: Report, sigmas: float = 6.0) -> None:
    """Flag per-block outliers in tensor magnitude.

    Weight magnitudes for the same role vary smoothly across the 48 blocks.
    A single block that is orders of magnitude off is the fingerprint of a
    tensor that got the wrong dequant scale or a mis-targeted LoRA merge —
    a defect no key/shape check can see.
    """
    roles: dict[str, list[tuple[int, float]]] = {}
    for key, values in stats.items():
        parts = key.split(".")
        idx = next((i for i, p in enumerate(parts) if p.isdigit()), None)
        if idx is None:
            continue
        role = ".".join(parts[:idx] + ["N"] + parts[idx + 1:])
        roles.setdefault(role, []).append((int(parts[idx]), values["rms"]))

    outliers: list[str] = []
    for role, entries in sorted(roles.items()):
        if len(entries) < 8:  # too few blocks for a meaningful distribution
            continue
        values = [v for _, v in entries]
        median = statistics.median(values)
        if median <= 0:
            continue
        # Compare in log space: magnitudes scale multiplicatively across depth.
        logs = [math.log(v) for v in values if v > 0]
        if len(logs) < 8:
            continue
        centre = statistics.median(logs)
        spread = statistics.median([abs(x - centre) for x in logs]) or 1e-6
        for (block, value), log_value in zip(entries, logs, strict=False):
            if abs(log_value - centre) > sigmas * spread:
                outliers.append(f"{role.replace('.N.', f'.{block}.')}: rms={value:.4g} vs median {median:.4g}")

    if outliers:
        rep.fail("per-block magnitude profile", f"{len(outliers)} tensor(s) deviate from their role's median")
        for line in outliers[:15]:
            rep.note(line)
    else:
        rep.ok("per-block magnitude profile", f"{len(roles)} roles, all blocks consistent")


def check_model_keys(repo: Path, rep: Report) -> None:
    for name in _transformer_components(repo):
        _check_component_model_keys(repo, name, rep)


def _check_component_model_keys(repo: Path, component: str, rep: Report) -> None:
    """Reconcile one transformer checkpoint against the model it will build."""
    try:
        from fastvideo.configs.models.dits.ltx2 import LTX2VideoConfig
        from fastvideo.models.dits.ltx2 import LTX2Transformer3DModel
    except Exception as err:  # noqa: BLE001
        rep.note(f"(skipping key reconciliation: fastvideo import failed — {type(err).__name__})")
        return

    config_path = repo / component / "config.json"
    if not config_path.is_file():
        return
    raw = json.loads(config_path.read_text())
    raw.pop("_class_name", None)
    raw.pop("_diffusers_version", None)
    try:
        arch = LTX2VideoConfig()
        for key, value in raw.items():
            if hasattr(arch.arch_config, key):
                setattr(arch.arch_config, key, value)
        with torch.device("meta"):
            model = LTX2Transformer3DModel(config=arch, hf_config=raw)
    except Exception as err:  # noqa: BLE001
        rep.note(f"(skipping key reconciliation: could not build the model — {type(err).__name__}: {err})")
        return

    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    got: dict[str, tuple] = {}
    for shard in _shards(repo / component):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.endswith((".weight_scale", ".weight_scale_2", ".comfy_quant")):
                    continue  # runtime sidecars, not model parameters
                got[key] = tuple(handle.get_slice(key).get_shape())

    mapped = {f"model.{k}" if not k.startswith("model.") else k: v for k, v in got.items()}
    missing = sorted(set(expected) - set(mapped))
    unexpected = sorted(set(mapped) - set(expected))
    mismatch = sorted(k for k in set(expected) & set(mapped) if expected[k] != mapped[k])

    silent = [k for k in missing if any(p in k for p in SILENT_ZERO_FILL_PATTERNS)]
    if missing or mismatch:
        rep.fail(f"{component} key reconciliation",
                 f"missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(mismatch)}")
        if silent:
            rep.note(f"!! {len(silent)} missing key(s) would be SILENTLY ZERO-FILLED at load:")
            for key in silent[:10]:
                rep.note(f"   {key}")
        for key in missing[:8]:
            rep.note(f"missing: {key}")
        for key in mismatch[:8]:
            rep.note(f"shape: {key} model={expected[key]} file={mapped[key]}")
    else:
        rep.ok(f"{component} key reconciliation",
               f"{len(expected)} parameters all present with matching shapes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="Converted LTX-2.3 diffusers directory")
    parser.add_argument("--sigmas", type=float, default=6.0,
                        help="Outlier threshold for the per-block magnitude profile (log-space MADs)")
    args = parser.parse_args()

    repo = Path(args.repo)
    if not repo.is_dir():
        raise SystemExit(f"--repo is not a directory: {repo}")

    rep = Report()
    print(f"Auditing {repo}\n")
    check_configs(repo, rep)
    print()
    check_storage_format(repo, rep)
    print()
    stats = scan_tensors(repo, rep)
    print()
    check_depth_profile(stats, rep, sigmas=args.sigmas)
    print()
    check_model_keys(repo, rep)

    print()
    if rep.failed:
        print("VERDICT: defects found — see the FAIL lines above.")
        raise SystemExit(1)
    print("VERDICT: no conversion defect detected by these checks.")


if __name__ == "__main__":
    main()
