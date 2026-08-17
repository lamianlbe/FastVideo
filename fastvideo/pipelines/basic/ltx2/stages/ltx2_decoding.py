# SPDX-License-Identifier: Apache-2.0
"""LTX-2 decoding stage: generic VAE decode plus RNG plumbing for the diffusion (HQ) decoder.

The LTX-2.5 diffusion video decoder denoises pixels, so its decode is stochastic. The official
pipelines pass the run's seeded generator into decode; FastVideo's generic ``DecodingStage`` calls
``vae.decode(latents)`` with no RNG. This subclass derives a generator from ``batch.seed`` (or
falls back to ``batch.generator``) and installs it on the VAE around the decode call via the
diffusion wrapper's ``set_decode_generator`` hook. The conv-decoder VAE has no such hook, so this
stage degrades to the stock behavior for conv checkpoints.
"""

from __future__ import annotations

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.decoding import DecodingStage

logger = init_logger(__name__)


class LTX2DecodingStage(DecodingStage):
    """Decoding stage that threads the batch seed/generator into the LTX-2.5 diffusion decoder."""

    _pending_seed: int | None = None
    _pending_generator: torch.Generator | None = None

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        self._pending_seed = batch.seed
        generator = batch.generator
        if isinstance(generator, list):
            generator = generator[0] if generator else None
        self._pending_generator = generator
        try:
            return super().forward(batch, fastvideo_args)
        finally:
            self._pending_seed = None
            self._pending_generator = None

    def _resolve_decode_generator(self) -> torch.Generator | None:
        """Fresh per-decode generator from the batch seed; batch generator as fallback.

        A fresh ``manual_seed(batch.seed)`` generator makes decode noise deterministic for a
        given seed. Note this is not bit-identical to the official pipelines, which reuse the
        run-wide generator whose state was already advanced by earlier sampling.
        """
        if self._pending_seed is not None:
            device = get_local_torch_device()
            try:
                return torch.Generator(device=device).manual_seed(int(self._pending_seed))
            except RuntimeError:
                # Devices without generator support (draws happen on the generator's device
                # and are moved to the compute device afterwards, so CPU is always valid).
                return torch.Generator(device="cpu").manual_seed(int(self._pending_seed))
        return self._pending_generator

    def decode(self, latents: torch.Tensor, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        setter = getattr(self.vae, "set_decode_generator", None)
        if setter is None:
            return super().decode(latents, fastvideo_args)
        setter(self._resolve_decode_generator())
        try:
            return super().decode(latents, fastvideo_args)
        finally:
            setter(None)
