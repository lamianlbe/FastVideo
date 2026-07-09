# SPDX-License-Identifier: Apache-2.0
"""Convert an official LTX-2 latent upsampler checkpoint to FastVideo layout.

The official single-file checkpoints (e.g.
``Lightricks/LTX-2.3/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors`` or
``ltx-2.3-spatial-upscaler-x2-1.1.safetensors``) carry the model config in
their safetensors metadata and use the same bare state-dict key layout
FastVideo loads, so conversion is just: extract metadata config ->
``config.json`` (with the FastVideo wrapper class name), copy tensors ->
``model.safetensors``.

Example:
    python scripts/checkpoint_conversion/convert_ltx2_upsampler.py \\
        --source ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors \\
        --output <MODEL_REPO>/spatial_upscaler_x1_5

Then point FastVideo at it via ``ltx2_refine_upsampler_path`` (or the
example script's ``LTX23_UPSAMPLER_PATH``). The refine pipeline reads
``spatial_scale`` from the config and adjusts the stage-1 resolution
automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LTX-2 latent upsampler to FastVideo layout")
    parser.add_argument("--source", required=True, help="official upsampler .safetensors file")
    parser.add_argument("--output", required=True, help="output directory")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    with safe_open(str(source), framework="pt") as f:
        metadata = f.metadata() or {}
    if "config" not in metadata:
        raise ValueError(f"{source} has no `config` safetensors metadata; expected an official "
                         "LTX-2 latent upsampler checkpoint.")
    config = json.loads(metadata["config"])
    if "upsampler" in config and isinstance(config["upsampler"], dict):
        config = config["upsampler"]
    config["_class_name"] = "LTX2LatentUpsampler"

    output.mkdir(parents=True, exist_ok=True)
    with (output / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)
        fp.write("\n")

    tensors = load_file(str(source))
    save_file(tensors, str(output / "model.safetensors"))

    print(f"Converted {source} -> {output}")
    print(f"  tensors: {len(tensors)}")
    print(f"  spatial_scale: {config.get('spatial_scale', 2.0)}, "
          f"rational_resampler: {config.get('rational_resampler', False)}, "
          f"temporal_upsample: {config.get('temporal_upsample', False)}")


if __name__ == "__main__":
    main()
