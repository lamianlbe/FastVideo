"""Shared engine for the LTX-2.5 production HTTP server.

Wraps the validated LTX-2.5 two-stage distilled recipe (see
examples/inference/basic/basic_ltx2_5_i2av_two_stage.py and the
``ltx2_5_distilled_two_stage_i2v`` preset) behind a config file + a small
API used by both server.py (online serving) and build_compile_cache.py
(offline compilation).

This directory is a deliberate, self-contained FORK of
examples/inference/ltx23_server/: that server runs live production traffic
and must stay untouched, so the operational machinery (auth, readiness
gate, request logging, GPU lock, H.264/AAC encoding, LQ variant, S3
upload, warmup) is duplicated here verbatim in behaviour rather than
imported. Only the RECIPE differs — see the "recipe" constants below and
``generate_for_mode``.

Recipe vs the 2.3 server (all deviations are 2.5's validated recipe):
  - BOTH stages use the ComfyUI CFG++ ancestral sampler at cfg=1
    (2.3: euler_ancestral in stage 1, cfg_pp only in stage 2).
  - Stage-1 sigmas are COMPUTED per mode from the LTXVScheduler formula
    shifted by the stage-1 latent token count, so every mode gets its own
    schedule (2.3 uses one hardcoded list for all modes).
  - Conditioning is inplace-pinned at 0.8 (stage 1) / 1.0 (stage 2) with a
    CRF-38 re-encode reused by both stages; there are no identity
    reference tokens in 2.5's recipe.
  - The mode's (width, height) is the FINAL size; stage 1 renders at HALF
    of it and the x2 latent upsampler brings it back up.

Import order matters: call ``setup_environment(cfg)`` BEFORE anything that
imports torch/fastvideo so TORCHINDUCTOR_CACHE_DIR and the attention
backend are picked up. All fastvideo/torch imports in this module are
deliberately function-local.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Validated LTX-2.5 two-stage distilled recipe. Every constant here mirrors
# examples/inference/basic/basic_ltx2_5_i2av_two_stage.py, which is the
# source of truth; tests/local_tests/ltx2_5/test_ltx2_5_server.py asserts
# the two never drift apart.
# ---------------------------------------------------------------------------
# Stage-1 schedule: ComfyUI LTXVScheduler(steps=8, max_shift=4.0,
# base_shift=1.5, stretch=True, terminal=0.1). Unlike 2.3's fixed list this
# is a FORMULA whose shift depends on the stage-1 latent token count, so the
# concrete sigmas are computed per mode (see stage1_sigmas_for_mode).
STAGE1_SCHEDULER_KWARGS: dict[str, Any] = dict(max_shift=4.0, base_shift=1.5, stretch=True, terminal=0.1)
DEFAULT_STAGE1_STEPS = 8
# Stage-2 schedule: the workflow's manual sigma list (3 refine steps).
DEFAULT_STAGE2_SIGMAS = [0.85, 0.7250, 0.4219, 0.0]
# Both stages run ComfyUI's CFG++ ancestral sampler at cfg=1. It still runs
# the uncond pass every step (except the degenerate sigma=1.0 first step,
# whose uncond output ComfyUI itself discards), so an EMPTY negative prompt
# is fine and is what the reference workflow uses.
DEFAULT_SAMPLER = "euler_ancestral_cfg_pp"
DEFAULT_REFINE_SAMPLER = "euler_ancestral_cfg_pp"
DEFAULT_NEGATIVE_PROMPT = ""
# LTXVPreprocess img_compression=38 (H.264 CRF re-encode of the conditioning
# image). Unlike the 2.3 server — which re-anchors stage 2 with a clean CRF-0
# encode — the 2.5 recipe reuses this value in BOTH stages.
DEFAULT_IMAGE_CRF = 38.0
# Inplace conditioning strengths: stage 1 pins the first frame softly so the
# model still moves, stage 2 re-pins it hard at full resolution.
DEFAULT_FIRST_FRAME_STRENGTH_STAGE1 = 0.8
DEFAULT_FIRST_FRAME_STRENGTH_STAGE2 = 1.0
DEFAULT_LAST_FRAME_STRENGTH = 0.8
# Per-stage distilled-LoRA strengths, used ONLY in runtime-LoRA mode (a
# pre-merged transformer runs both stages with no runtime adapter).
DEFAULT_STAGE1_LORA_STRENGTH = 0.7
DEFAULT_REFINE_LORA_STRENGTH = 0.5
# One static graph per shape, no graph breaks — the same recipe the 2.3
# server has run in production (and what the example's --warmup copies).
COMPILE_KWARGS: dict[str, Any] = {
    "backend": "inductor",
    "fullgraph": True,
    "mode": "default",
    "dynamic": False,
}
WARMUP_PROMPT = ("A person slowly turns their head toward the camera and smiles, "
                 "soft warm light, gentle camera drift.")


@dataclass(frozen=True)
class Ltx25Mode:
    """One supported (resolution, frames, fps) combination.

    ``width``/``height`` are the FINAL (stage-2) dimensions. Stage 1 renders
    at half of them and the x2 latent upsampler restores the full size.
    """
    width: int
    height: int
    num_frames: int
    fps: int

    def validate(self) -> None:
        # Both stages patchify on a /32 latent grid and stage 1 runs at HALF
        # the configured size, so the FINAL dims must be multiples of 64.
        if self.width % 64 or self.height % 64:
            raise ValueError(f"mode {self.width}x{self.height}: both FINAL dims must be divisible by 64 "
                             "(stage 1 renders at half resolution and needs /32 dims)")
        if (self.num_frames - 1) % 8:
            raise ValueError(f"mode num_frames={self.num_frames}: must be 8*k+1 "
                             "(temporal VAE compression)")
        if self.fps <= 0:
            raise ValueError(f"mode fps={self.fps}: must be positive")

    def stage1_size(self) -> tuple[int, int]:
        """(width, height) stage 1 actually denoises — half the final size."""
        return (self.width // 2, self.height // 2)

    def stage1_tokens(self) -> int:
        """Stage-1 latent token count T*H*W (8x temporal, 32x spatial VAE
        compression of the HALF-resolution grid). LTXVScheduler's sigma shift
        is derived from it, exactly as the reference workflow does by wiring
        the stage-1 latent into the scheduler node."""
        stage1_width, stage1_height = self.stage1_size()
        return ((self.num_frames - 1) // 8 + 1) * (stage1_height // 32) * (stage1_width // 32)

    def shape_key(self) -> tuple[int, int, int, int]:
        """Compile-shape identity. fps is part of it: the audio latent
        length is derived from the clip duration (num_frames / fps), so a
        different fps at the same frame count changes the DiT sequence
        length and requires its own compiled graph."""
        return (self.width, self.height, self.num_frames, self.fps)


@dataclass
class Ltx25S3Config:
    """S3 destination for the /v1/generate_s3 endpoint."""
    region: str
    bucket: str
    access_key: str
    secret_key: str
    endpoint_url: str = ""  # "" = AWS; set for R2/MinIO-compatible stores
    # Directory (key prefix) for uploaded files. "" = bucket root. Each
    # output is <prefix>/<uuid4>.mp4 — e.g. prefix "sg" ->
    # s3://<bucket>/sg/xxxxxxxx-xxxx-...-xxxx.mp4.
    prefix: str = ""

    def validate(self) -> None:
        for field_name in ("region", "bucket", "access_key", "secret_key"):
            if not getattr(self, field_name):
                raise ValueError(f"s3.{field_name} is required")


@dataclass
class Ltx25ServerConfig:
    model_path: str
    modes: list[Ltx25Mode]
    # "" = auto-detect <model>/spatial_upsampler (what convert_ltx2_weights.py
    # writes for 2.5) or spatial_upscaler. MUST be LTX-2.5's OWN x2 upsampler:
    # the 2.3 upscaler does not match the 2.5 latent distribution.
    upsampler_path: str = ""
    # Runtime distilled-LoRA override. "" = whatever the converted
    # model_index.json wires (fastvideo_refine_lora_path), which is nothing
    # for a pre-merged transformer.
    distilled_lora_path: str = ""
    # Force pre-merged mode: run BOTH stages on the transformer as-is with no
    # runtime LoRA. Auto-detected from model_index.json; set this only to
    # override a directory that still bundles a runtime distilled LoRA.
    pre_merged: bool = False
    # bf16 by default: LTX-2.5 quantized deployment is explicitly follow-up
    # work (docs/inference/ltx2_5.md "Current scope"), unlike 2.3 where nvfp4
    # is the validated serving tier. nvfp4 | fp8 | fp8_channel | none.
    quant: str = "none"
    num_gpus: int = 1
    # Which physical GPU(s) this instance runs on, e.g. "1" or "0,1".
    # "" = inherit the shell / all visible. On a multi-GPU box, run one
    # server per GPU (num_gpus=1, distinct cuda_visible_devices + port +
    # log_dir). Set BEFORE torch initializes, so it actually pins the
    # process. The count should match num_gpus.
    cuda_visible_devices: str = ""
    attention_backend: str = "FLASH_ATTN"
    # FA4 (flash_attn.cute) under the FLASH_ATTN backend — the validated
    # serving stack. Requires the pinned flash-attn cute install (see
    # ../ltx23_server/deploy/install.sh, which is shared infrastructure).
    fa4: bool = True
    # FP8 (e4m3) attention per stage: q/k/v quantized with per-head descales,
    # both attention GEMMs at the fp8 tensor-core rate. OFF = the bf16 FA4
    # path. Toggling changes the compiled graph — re-run build_compile_cache.py.
    fa4_fp8_stage1: bool = False
    fa4_fp8_stage2: bool = False
    # Neighborhood-attention backend for the HQ (diffusion) video decoder.
    # "" = auto (natten on CUDA, which is REQUIRED for the HQ decoder — see
    # the README). "triton"/"eager" are correct-but-slow debug escapes.
    na_backend: str = ""
    # VAE tiling for the decode. Recommended at 720p and above when the model
    # directory carries the HQ diffusion decoder (see docs/inference/ltx2_5.md).
    vae_tiling: bool = False
    inductor_cache_dir: str = ""  # "" = torch default (NOT persistent)
    compile: bool = True
    warmup_on_start: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    output_dir: str = ""  # "" = system temp; holds per-request scratch dirs
    # "" = log to stdout only. Otherwise a directory receiving
    # requests.jsonl (rotating JSON-lines request log) and failed/<id>/
    # (preserved inputs + params of failed generations, for repro).
    log_dir: str = ""
    # Exit the process after this many CONSECUTIVE generation failures so
    # the supervisor (docker --restart / run_server.sh) replaces a wedged
    # GPU worker with a fresh process. 0 disables. Validation errors (4xx)
    # don't count; any success resets the counter.
    max_consecutive_failures: int = 3
    # Allowed API keys. Non-empty: /v1/* requests must present one via
    # "X-API-Key: <key>" or "Authorization: Bearer <key>" or get 401
    # (/healthz stays open for the docker HEALTHCHECK). Empty: no auth.
    api_keys: list[str] = field(default_factory=list)
    # CPU H.264 encoding (B200 has no NVENC). Average bitrate for the
    # libx264 main-profile VBR encode; per-request video_bitrate_kbps
    # overrides it.
    video_bitrate_kbps: int = 3000
    x264_preset: str = "medium"
    # H.264 by default; set e.g. libx265 (needs an ffmpeg with that encoder;
    # the main/baseline profile is skipped for non-H.264 codecs). Applies to
    # both HQ and LQ.
    video_codec: str = "libx264"
    # Extra ffmpeg video-encoder args (shlex-split), appended after the
    # built-in opts, e.g. "-x265-params asm=avx512 -tag:v hvc1".
    extra_video_args: str = ""
    # x264 encoder threads (0 = auto ~1.5x logical cores). If encoding is
    # slow, check the container's ACTUAL cpu allocation first (nproc) and
    # consider x264_preset: faster / veryfast — at a fixed VBR bitrate the
    # preset mostly trades quality-per-bit, not target quality.
    encode_threads: int = 0
    # Encodes run OUTSIDE the GPU lock (generation of the next request
    # overlaps encoding of the previous). This caps simultaneous CPU
    # encodes so a burst can't starve the host. /v1/generate_s3's HQ+LQ
    # pair counts as ONE unit (they run in parallel inside it).
    max_concurrent_encodes: int = 2
    # /v1/generate_s3 low-quality variant: half width/height, GPU gaussian
    # blur (sigma in pixels at the LQ resolution; 0 disables), H.264
    # constrained baseline @ lq_bitrate_kbps with preset fast, AAC-LC
    # mono 64 kbps.
    lq_blur_radius: float = 2.0
    lq_bitrate_kbps: int = 1000
    lq_x264_preset: str = "ultrafast"
    # S3 destination (required only for /v1/generate_s3).
    s3: Ltx25S3Config | None = None
    # --- recipe knobs (defaults = the validated two-stage recipe) ---
    stage1_steps: int = DEFAULT_STAGE1_STEPS
    stage1_max_shift: float = float(STAGE1_SCHEDULER_KWARGS["max_shift"])
    stage1_base_shift: float = float(STAGE1_SCHEDULER_KWARGS["base_shift"])
    stage1_stretch: bool = bool(STAGE1_SCHEDULER_KWARGS["stretch"])
    stage1_terminal: float = float(STAGE1_SCHEDULER_KWARGS["terminal"])
    # true = shift the stage-1 schedule by LTXVScheduler's detached
    # tokens=4096 anchor instead of the mode's actual stage-1 token count.
    # The reference workflow attaches the latent, so false matches it.
    sigmas_token_anchor: bool = False
    stage2_sigmas: list[float] = field(default_factory=lambda: list(DEFAULT_STAGE2_SIGMAS))
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    image_crf: float = DEFAULT_IMAGE_CRF
    first_frame_strength_stage1: float = DEFAULT_FIRST_FRAME_STRENGTH_STAGE1
    first_frame_strength_stage2: float = DEFAULT_FIRST_FRAME_STRENGTH_STAGE2
    last_frame_strength: float = DEFAULT_LAST_FRAME_STRENGTH
    # Runtime-LoRA mode only (ignored when the transformer is pre-merged).
    stage1_lora_strength: float = DEFAULT_STAGE1_LORA_STRENGTH
    refine_lora_strength: float = DEFAULT_REFINE_LORA_STRENGTH


@dataclass
class GenerationRequest:
    """One resolved generation job (paths point at files on local disk)."""
    prompt: str
    first_frame_path: str
    last_frame_path: str | None = None
    negative_prompt: str | None = None
    seed: int = 0
    last_frame_strength: float = DEFAULT_LAST_FRAME_STRENGTH
    # Whether the last-frame anchor also enters the stage-2 refine pass.
    # Defaults OFF here (the 2.3 server defaults it ON): the validated 2.5
    # example keeps the tail anchor out of the refine pass because it
    # misbehaved in ComfyUI testing.
    last_in_upscale: bool = False
    image_crf: float = DEFAULT_IMAGE_CRF


def load_config(path: str | Path) -> Ltx25ServerConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    modes_raw = raw.pop("modes", None)
    if not modes_raw:
        raise ValueError(f"{path}: 'modes' must list at least one "
                         "{width, height, num_frames, fps} combination")
    modes = [Ltx25Mode(**m) for m in modes_raw]
    for mode in modes:
        mode.validate()
    s3_raw = raw.pop("s3", None)
    s3_cfg = None
    if s3_raw is not None:
        if not isinstance(s3_raw, dict):
            raise ValueError(f"{path}: 's3' must be a mapping")
        s3_cfg = Ltx25S3Config(**s3_raw)
        s3_cfg.validate()
    known = {f for f in Ltx25ServerConfig.__dataclass_fields__ if f not in ("modes", "s3")}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    cfg = Ltx25ServerConfig(modes=modes, s3=s3_cfg, **raw)
    if not cfg.model_path:
        raise ValueError(f"{path}: 'model_path' is required")
    if cfg.quant not in ("nvfp4", "fp8", "fp8_channel", "none"):
        raise ValueError(f"{path}: quant must be nvfp4 | fp8 | fp8_channel | none, got {cfg.quant}")
    if cfg.na_backend and cfg.na_backend not in ("natten", "triton", "eager"):
        raise ValueError(f"{path}: na_backend must be natten | triton | eager (or '' for auto), "
                         f"got {cfg.na_backend!r}")
    if cfg.stage1_steps < 1:
        raise ValueError(f"{path}: stage1_steps must be >= 1, got {cfg.stage1_steps}")
    _validate_sigmas(path, "stage2_sigmas", cfg.stage2_sigmas)
    if cfg.pre_merged and cfg.distilled_lora_path:
        # Same contradiction the example rejects: a pre-merged transformer
        # must not get a runtime LoRA stacked on top (it would double-apply).
        raise ValueError(f"{path}: pre_merged and distilled_lora_path contradict each other")
    for name in ("first_frame_strength_stage1", "first_frame_strength_stage2", "last_frame_strength"):
        value = getattr(cfg, name)
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{path}: {name} must be in [0, 1], got {value}")
    if any(not isinstance(k, str) or not k.strip() for k in cfg.api_keys):
        raise ValueError(f"{path}: api_keys entries must be non-empty strings")
    cvd = cfg.cuda_visible_devices
    if cvd and not all(p.strip().isdigit() for p in cvd.split(",")):
        raise ValueError(f"{path}: cuda_visible_devices must be comma-separated GPU indices, got {cvd!r}")
    return cfg


def _validate_sigmas(path: str | Path, name: str, sigmas: list[float]) -> None:
    """Strictly decreasing and ending at 0.0 — the same contract
    FastVideoArgs enforces on ltx2_stage*_sigmas, checked here so a bad
    config fails at load instead of minutes later inside the pipeline."""
    if len(sigmas) < 2:
        raise ValueError(f"{path}: {name} needs at least 2 entries, got {sigmas}")
    if float(sigmas[-1]) != 0.0:
        raise ValueError(f"{path}: {name} must end at 0.0, got {sigmas}")
    if any(float(b) >= float(a) for a, b in zip(sigmas, sigmas[1:], strict=False)):
        raise ValueError(f"{path}: {name} must be strictly decreasing, got {sigmas}")


def setup_environment(cfg: Ltx25ServerConfig) -> None:
    """Set process env consumed by torch/fastvideo. Call before importing
    either; already-exported env vars win (operator override)."""
    # GPU pinning is authoritative when set in config/CLI (unlike the other
    # knobs below): a specific GPU is an explicit deployment choice, so it
    # overrides any inherited CUDA_VISIBLE_DEVICES rather than deferring.
    if cfg.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    if cfg.inductor_cache_dir:
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cfg.inductor_cache_dir)
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", cfg.attention_backend)
    os.environ.setdefault("FASTVIDEO_FA4", "1" if cfg.fa4 else "0")
    os.environ.setdefault("FASTVIDEO_FA4_FP8_STAGE1", "1" if cfg.fa4_fp8_stage1 else "0")
    os.environ.setdefault("FASTVIDEO_FA4_FP8_STAGE2", "1" if cfg.fa4_fp8_stage2 else "0")
    # HQ (diffusion) video decoder only; ignored by the conv decoder.
    if cfg.na_backend:
        os.environ.setdefault("FASTVIDEO_LTX2_NA_BACKEND", cfg.na_backend)
    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")


def match_mode(
    modes: list[Ltx25Mode],
    width: int,
    height: int,
    num_frames: int,
    fps: int,
) -> tuple[Ltx25Mode, bool]:
    """Exact (width, height, num_frames, fps) match, else the mode with the
    closest resolution (aspect-aware log distance); frames/fps only break
    ties. Returns (mode, exact)."""
    for mode in modes:
        if (mode.width, mode.height, mode.num_frames, mode.fps) == (width, height, num_frames, fps):
            return mode, True

    def distance(mode: Ltx25Mode) -> tuple[float, int, int]:
        res = (math.log(width / mode.width)**2 + math.log(height / mode.height)**2)
        return (res, abs(num_frames - mode.num_frames), abs(fps - mode.fps))

    return min(modes, key=distance), False


def resolve_upsampler(model_root: str, override: str = "") -> str:
    """LTX-2.5's OWN x2 spatial latent upsampler. convert_ltx2_weights.py
    writes it as spatial_upsampler/ for 2.5; spatial_upscaler/ is accepted
    for hand-assembled directories. Resolving it here (instead of relying on
    the model_index auto-wire the example uses) turns a missing upsampler
    into a startup error rather than a mid-request failure."""
    candidates = ([override] if override else []) + [
        str(Path(model_root) / "spatial_upsampler"),
        str(Path(model_root) / "spatial_upscaler"),
    ]
    for cand in candidates:
        if cand and (Path(cand) / "config.json").is_file():
            return cand
    raise FileNotFoundError("No x2 spatial upsampler found; set 'upsampler_path' in the config or convert "
                            "with --spatial-upscaler-source so spatial_upsampler/ exists. The LTX-2.3 "
                            "upscaler is NOT a substitute (different latent distribution).")


def read_model_index(model_root: str | Path) -> dict:
    path = Path(model_root) / "model_index.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def detect_lora_mode(model_index: dict, cfg: Ltx25ServerConfig) -> tuple[bool, str]:
    """(use_runtime_lora, human-readable explanation).

    Mirrors basic_ltx2_5_i2av_two_stage.py exactly: a directory converted
    with --transformer-lora records ``_fastvideo_transformer_merged_loras``
    and omits ``fastvideo_refine_lora_path``, so both stages run the merged
    transformer with NO runtime adapter and nothing is double-applied."""
    # The unprefixed spelling is what conversions before the metadata-key
    # rename wrote; those directories are otherwise identical, so keep
    # reading it.
    merged_loras = (model_index.get("_fastvideo_transformer_merged_loras")
                    or model_index.get("fastvideo_transformer_merged_loras"))
    runtime_lora_available = bool(cfg.distilled_lora_path or model_index.get("fastvideo_refine_lora_path"))
    use_runtime_lora = runtime_lora_available and not cfg.pre_merged
    if use_runtime_lora:
        source = cfg.distilled_lora_path or model_index.get("fastvideo_refine_lora_path")
        return True, (f"runtime distilled LoRA {source} at per-stage strengths "
                      f"{cfg.stage1_lora_strength} / {cfg.refine_lora_strength}")
    detail = f"offline-merged LoRAs: {merged_loras}" if merged_loras else "no distilled LoRA is wired"
    return False, f"both stages run the transformer as-is ({detail}); per-stage runtime strengths disabled"


def describe_video_decoder(model_root: str | Path) -> str:
    """Which video decoder the converted vae/ directory carries. There is no
    config field for this — whichever VAE was converted in decides — but the
    HQ path costs ~2-3x the conv decode and needs natten, so it must be
    obvious in the startup log which one is loaded."""
    config_path = Path(model_root) / "vae" / "config.json"
    if not config_path.is_file():
        return "unknown (no vae/config.json)"
    try:
        class_name = json.loads(config_path.read_text()).get("_class_name", "")
    except (OSError, ValueError):
        return "unknown (unreadable vae/config.json)"
    if class_name == "CausalDiffusionVAE":
        return "HQ diffusion decoder (CausalDiffusionVAE; needs natten, ~2-3x conv decode cost)"
    if class_name == "CausalVideoAutoencoder":
        return "convolutional decoder (CausalVideoAutoencoder)"
    return f"unrecognized vae class {class_name!r}"


_STAGE1_SIGMA_CACHE: dict[tuple, list[float]] = {}


def stage1_sigmas_for_mode(cfg: Ltx25ServerConfig, mode: Ltx25Mode) -> list[float]:
    """LTXVScheduler sigmas for this mode's stage-1 latent.

    The shift depends on the stage-1 token count, so — unlike the 2.3
    server's single hardcoded list — every mode has its own schedule. A
    mode's token count is fixed, so this is computed once per mode and
    cached."""
    scheduler_kwargs = dict(
        max_shift=cfg.stage1_max_shift,
        base_shift=cfg.stage1_base_shift,
        stretch=cfg.stage1_stretch,
        terminal=cfg.stage1_terminal,
    )
    # tokens=None reproduces the ComfyUI node's detached default (4096).
    tokens = None if cfg.sigmas_token_anchor else mode.stage1_tokens()
    key = (mode.shape_key(), cfg.stage1_steps, tokens, tuple(sorted(scheduler_kwargs.items())))
    cached = _STAGE1_SIGMA_CACHE.get(key)
    if cached is not None:
        return list(cached)

    from fastvideo.pipelines.basic.ltx2.stages import compute_ltxv_scheduler_sigmas

    sigmas = compute_ltxv_scheduler_sigmas(cfg.stage1_steps, tokens=tokens, **scheduler_kwargs).tolist()
    _STAGE1_SIGMA_CACHE[key] = list(sigmas)
    return sigmas


def create_generator(cfg: Ltx25ServerConfig) -> Any:
    """Build the resident VideoGenerator with the validated two-stage recipe
    wired in. Everything except the stage-1 sigma schedule is fixed at init;
    the schedule is per-mode and re-applied by generate_for_mode."""
    if cfg.compile:
        import torch._inductor.config as _inductor

        _inductor.shape_padding = False  # mandatory on Blackwell (pad_mm crash)
        _inductor.conv_1x1_as_mm = True
        _inductor.coordinate_descent_tuning = True
        _inductor.coordinate_descent_check_all_directions = True
        _inductor.epilogue_fusion = False

    from fastvideo import VideoGenerator
    from fastvideo.configs.pipelines.base import PipelineConfig
    from fastvideo.utils import maybe_download_model

    model_root = maybe_download_model(cfg.model_path)
    upsampler_path = resolve_upsampler(model_root, cfg.upsampler_path)
    model_index = read_model_index(model_root)
    use_runtime_lora, lora_note = detect_lora_mode(model_index, cfg)
    print(f"[engine] video decoder: {describe_video_decoder(model_root)}")
    print(f"[engine] x2 spatial upsampler: {upsampler_path}")
    print(f"[engine] distilled LoRA: {lora_note}")

    pipeline_config = PipelineConfig.from_pretrained(model_root)
    # Linear quantization ladder (loss high -> none): nvfp4 (e2m1, fastest),
    # fp8 (e4m3 per-tensor), fp8_channel (per-channel weights + per-token
    # activations, most conservative quantized tier), none (bf16). LTX-2.5
    # defaults to none — quantized 2.5 deployment is not validated yet.
    if cfg.quant == "nvfp4":
        from fastvideo.layers.quantization.nvfp4_config import NVFP4Config
        pipeline_config.dit_config.quant_config = NVFP4Config()
    elif cfg.quant in ("fp8", "fp8_channel"):
        from fastvideo.layers.quantization.fp8_config import FP8Config
        pipeline_config.dit_config.quant_config = FP8Config(
            granularity="channel" if cfg.quant == "fp8_channel" else "tensor")
    else:
        pipeline_config.dit_config.quant_config = None

    compile_kwargs: dict[str, Any] = {}
    if cfg.compile:
        compile_kwargs = dict(
            enable_torch_compile=True,
            enable_torch_compile_text_encoder=True,
            enable_torch_compile_vae=True,
            torch_compile_kwargs=dict(COMPILE_KWARGS),
            torch_compile_kwargs_vae=dict(COMPILE_KWARGS),
        )

    lora_kwargs: dict[str, Any] = {}
    if cfg.distilled_lora_path:
        lora_kwargs["ltx2_refine_lora_path"] = cfg.distilled_lora_path
    elif not use_runtime_lora:
        # Explicit empty path blocks the model_index fastvideo_refine_lora_path
        # auto-resolve, so a pre-merged transformer can never get an adapter
        # stacked on top of already-merged weights.
        lora_kwargs["ltx2_refine_lora_path"] = ""
    if use_runtime_lora:
        # Per-stage distilled-LoRA strengths (0.7 base denoise, 0.5 refine).
        # Left unset in pre-merged mode: with no refine-LoRA path resolved the
        # pipeline builds no LoRA stages at all. NOTE each strength switch is
        # an exact unmerge + re-merge weight sweep per stage per RUN — fine for
        # experiments, a real per-request cost in serving. Prefer a pre-merged
        # transformer in production.
        lora_kwargs["ltx2_stage1_lora_strength"] = float(cfg.stage1_lora_strength)
        lora_kwargs["ltx2_refine_lora_strength"] = float(cfg.refine_lora_strength)

    return VideoGenerator.from_pretrained(
        model_root,
        num_gpus=cfg.num_gpus,
        pipeline_config=pipeline_config,
        **compile_kwargs,
        **lora_kwargs,
        ltx2_refine_enabled=True,
        ltx2_refine_upsampler_path=upsampler_path,
        ltx2_refine_guidance_scale=1.0,
        ltx2_refine_add_noise=True,
        # ComfyUI-style CFG++ ancestral sampler in BOTH stages (2.3 runs
        # plain euler_ancestral in stage 1).
        ltx2_sampler=DEFAULT_SAMPLER,
        ltx2_refine_sampler=DEFAULT_REFINE_SAMPLER,
        # Seeded with the first mode's schedule so a single-mode deployment
        # never depends on the per-request re-apply below.
        ltx2_stage1_sigmas=stage1_sigmas_for_mode(cfg, cfg.modes[0]),
        ltx2_stage2_sigmas=list(cfg.stage2_sigmas),
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        ltx2_vae_tiling=cfg.vae_tiling,
    )


def apply_stage1_sigmas(generator: Any, sigmas: list[float]) -> None:
    """Point the resident engine at this mode's stage-1 schedule.

    ``ltx2_stage1_sigmas`` is an engine-level FastVideoArgs field with no
    per-request override surface, but the executor ships fastvideo_args to
    the worker with EVERY forward, so re-assigning it between generations
    takes effect on the next one. Callers must hold the GPU lock (the
    server does), which keeps this safe under serialized generation."""
    args = getattr(generator, "fastvideo_args", None)
    if args is None:
        # Fail loudly rather than silently serving another mode's schedule.
        raise RuntimeError("generator has no fastvideo_args; cannot set the per-mode stage-1 schedule")
    args.ltx2_stage1_sigmas = list(sigmas)


def generate_for_mode(
    generator: Any,
    cfg: Ltx25ServerConfig,
    mode: Ltx25Mode,
    request: GenerationRequest,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run one generation at the given mode's shape and return RAW frames
    (+ audio) instead of writing an mp4 — encoding happens on the CPU via
    encode_video_h264, outside the caller's GPU lock, so the next request's
    generation overlaps the previous request's encode.

    ``mode.width``/``mode.height`` are the FINAL size: LTX2RefineInitStage
    halves them for stage 1 and the x2 latent upsampler restores them.

    Conditioning images are cover-fit (aspect-preserving resize + center
    crop, no letterboxing) to the stage resolution inside the pipeline, so
    callers can pass uploads as-is. ``output_path`` is only pipeline path
    bookkeeping; nothing is written to it."""
    stage1_sigmas = stage1_sigmas_for_mode(cfg, mode)
    apply_stage1_sigmas(generator, stage1_sigmas)

    # ltx2_images frame indices are LATENT-frame indices (8x temporal
    # compression); the last anchor pins the final latent frame == the final
    # ~8 pixel frames.
    last_latent_idx = (mode.num_frames - 1) // 8
    images: list[tuple[str, int, float]] = [(request.first_frame_path, 0, cfg.first_frame_strength_stage1)]
    if request.last_frame_path:
        images.append((request.last_frame_path, last_latent_idx, request.last_frame_strength))
    # Stage 2 is ALWAYS given its own list (unlike the 2.3 server, where None
    # means "same keyframes in both stages"): the 2.5 recipe re-pins the first
    # frame at a DIFFERENT strength (1.0 vs stage 1's 0.8).
    images_stage2: list[tuple[str, int, float]] = [(request.first_frame_path, 0, cfg.first_frame_strength_stage2)]
    if request.last_frame_path and request.last_in_upscale:
        images_stage2.append((request.last_frame_path, last_latent_idx, request.last_frame_strength))

    result = generator.generate_video(
        prompt=request.prompt,
        # `is not None` (not a falsy check as in the 2.3 server): the 2.5
        # default IS the empty string, so an explicitly empty request value
        # must stay empty rather than fall back to the config.
        negative_prompt=(request.negative_prompt if request.negative_prompt is not None else cfg.negative_prompt),
        output_path=str(output_path),
        seed=request.seed,
        guidance_scale=1.0,
        height=mode.height,
        width=mode.width,
        num_frames=mode.num_frames,
        fps=mode.fps,
        num_inference_steps=len(stage1_sigmas) - 1,
        ltx2_images=images,
        ltx2_images_stage2=images_stage2,
        # One CRF for both stages — the 2.5 recipe reuses the stage-1
        # preprocessed image in stage 2 (no ltx2_image_crf_stage2 override,
        # unlike the 2.3 server's clean CRF-0 re-anchor).
        ltx2_image_crf=request.image_crf,
        # Plain CFG++ guider: no STG / modality isolation / rescale, and the
        # official-2.5 ancestral path stays off (the cfg_pp sampler drives).
        ltx2_use_ancestral_sampler=False,
        ltx2_cfg_scale_video=1.0,
        ltx2_cfg_scale_audio=1.0,
        ltx2_modality_scale_video=1.0,
        ltx2_modality_scale_audio=1.0,
        ltx2_rescale_scale=0.0,
        ltx2_stg_scale_video=0.0,
        ltx2_stg_scale_audio=0.0,
        save_video=False,
        return_frames=True,
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"unexpected generate_video result type {type(result)}")
    frames = result.get("frames")
    if not frames:
        raise RuntimeError("generation returned no frames")
    return {
        "frames": frames,
        "audio": result.get("audio"),
        "audio_sample_rate": result.get("audio_sample_rate"),
        "gen_seconds": float(result.get("e2e_latency") or 0.0),
    }


def _audio_to_int16(audio: Any) -> tuple[Any, int]:
    """[samples] / [samples, ch] / [ch, samples] float in ~[-1, 1] ->
    (int16 [samples, ch], num_channels). Mirrors the normalization the
    stock save path applies."""
    import numpy as np

    if hasattr(audio, "detach"):  # torch tensor without importing torch here
        audio = audio.detach().cpu().float().numpy()
    audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]
    elif audio_np.ndim == 2:
        if audio_np.shape[0] <= 8 and audio_np.shape[1] > audio_np.shape[0]:
            audio_np = audio_np.T
    else:
        raise ValueError(f"Unexpected audio shape {audio_np.shape}.")
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)
    return audio_int16, audio_int16.shape[1]


def _resolve_ffmpeg() -> str:
    """Path to an ffmpeg binary: system first, else imageio-ffmpeg's bundle
    (a pip dep, so always available even without apt)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as err:  # noqa: BLE001
        raise RuntimeError("ffmpeg not found: install a system ffmpeg or `pip install imageio-ffmpeg`") from err


def _rgb_frames_to_yuv420p_bytes(frames: list[Any], device: str) -> bytes:
    """RGB uint8 frames -> planar yuv420p byte stream (BT.709 limited
    range), computed on ``device`` (cuda). Doing the color conversion +
    chroma subsample here offloads libav's single-threaded swscale — the
    serial bottleneck in CPU H.264 encoding — to the GPU."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.stack(frames)).to(device).float()  # N,H,W,3
    n = x.shape[0]
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = (16 + (0.2126 * r + 0.7152 * g + 0.0722 * b) * (219 / 255)).clamp(0, 255)
    u = 128 + (-0.1146 * r - 0.3854 * g + 0.5 * b) * (224 / 255)
    v = 128 + (0.5 * r - 0.4542 * g - 0.0458 * b) * (224 / 255)
    u = F.avg_pool2d(u.unsqueeze(1), 2).squeeze(1).clamp(0, 255)  # N,H/2,W/2
    v = F.avg_pool2d(v.unsqueeze(1), 2).squeeze(1).clamp(0, 255)
    buf = torch.cat([
        y.to(torch.uint8).reshape(n, -1),
        u.to(torch.uint8).reshape(n, -1),
        v.to(torch.uint8).reshape(n, -1),
    ], dim=1).reshape(-1)
    return buf.cpu().numpy().tobytes()


def encode_video_h264(
    frames: list[Any],
    fps: int,
    output_path: str | Path,
    *,
    bitrate_kbps: int = 3000,
    preset: str = "medium",
    profile: str = "main",
    audio: Any = None,
    audio_sample_rate: int | None = None,
    audio_bitrate_kbps: int | None = None,
    audio_mono: bool = False,
    threads: int = 0,
    gpu_yuv: bool = True,
    codec: str = "libx264",
    extra_video_args: str = "",
) -> float:
    """Encode RGB frames (+ optional audio) to MP4 via an ffmpeg subprocess:
    ``codec`` at the given profile/preset, VBR at the average bitrate with a
    2x/4x VBV envelope, AAC audio (optionally mono at a fixed bitrate),
    +faststart for web delivery. Returns wall time in seconds.

    ``extra_video_args`` (shlex-split) is appended to the video-encoder
    options for codec tuning without code edits, e.g.
    ``-x265-params asm=avx512 -tag:v hvc1``. ``-profile:v`` is emitted only
    for H.264 encoders (the main/baseline profile names are H.264-specific);
    for other codecs set the profile via extra_video_args.

    The RGB->YUV420 color conversion is done on the GPU (``gpu_yuv``, when
    CUDA is present) and yuv420p is piped straight to ffmpeg, bypassing
    libav's single-threaded swscale. Without CUDA it falls back to piping
    rgb24 and letting ffmpeg convert (used by CPU-only test hosts)."""
    import numpy as np

    if not frames:
        raise ValueError("no frames to encode")
    t0 = time.perf_counter()
    h, w = int(frames[0].shape[0]), int(frames[0].shape[1])
    ffmpeg = _resolve_ffmpeg()

    use_gpu = False
    if gpu_yuv:
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            use_gpu = False

    if use_gpu:
        video_bytes = _rgb_frames_to_yuv420p_bytes(frames, "cuda")
        in_pix = "yuv420p"
        # We produced BT.709 limited-range YUV; tag the stream so players
        # interpret it correctly (no conversion, just metadata).
        color_args = ["-colorspace", "bt709", "-color_primaries", "bt709",
                      "-color_trc", "bt709", "-color_range", "tv"]
    else:
        video_bytes = np.ascontiguousarray(np.stack(frames)).tobytes()
        in_pix = "rgb24"  # ffmpeg swscale converts (CPU fallback path)
        color_args = []

    audio_path = None
    if audio is not None and audio_sample_rate:
        import wave
        audio_int16, num_channels = _audio_to_int16(audio)
        audio_path = str(Path(output_path).with_suffix(".wav"))
        with wave.open(audio_path, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(2)
            wf.setframerate(int(audio_sample_rate))
            wf.writeframes(audio_int16.tobytes())

    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", in_pix, "-s", f"{w}x{h}", "-r", str(int(fps)),
           "-i", "pipe:0"]
    if audio_path:
        cmd += ["-i", audio_path]
    cmd += ["-c:v", codec, "-preset", preset]
    if "264" in codec:  # main/baseline profile names are H.264-specific
        cmd += ["-profile:v", profile]
    cmd += ["-pix_fmt", "yuv420p",
            "-b:v", f"{int(bitrate_kbps)}k", "-maxrate", f"{int(bitrate_kbps) * 2}k",
            "-bufsize", f"{int(bitrate_kbps) * 4}k", "-threads", str(max(0, int(threads)))]
    cmd += color_args
    if extra_video_args.strip():
        import shlex
        cmd += shlex.split(extra_video_args)
    if audio_path:
        cmd += ["-c:a", "aac"]
        if audio_bitrate_kbps:
            cmd += ["-b:a", f"{int(audio_bitrate_kbps)}k"]
        if audio_mono:
            cmd += ["-ac", "1"]
        cmd += ["-shortest"]
    cmd += ["-movflags", "+faststart", str(output_path)]

    try:
        proc = subprocess.run(cmd, input=video_bytes, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed (rc={proc.returncode}): "
                               f"{proc.stderr.decode(errors='replace')[-800:]}")
    finally:
        if audio_path:
            Path(audio_path).unlink(missing_ok=True)
    return time.perf_counter() - t0


def make_lq_frames(
    frames: list[Any],
    blur_radius: float,
    device: str | None = None,
    chunk_size: int = 16,
) -> list[Any]:
    """Low-quality variant of RGB uint8 frames: half width/height (area
    downscale) + separable gaussian blur, computed on the GPU in chunks.
    ``blur_radius`` is the gaussian sigma in pixels AT THE LQ RESOLUTION
    (0 disables the blur). Runs outside the GPU lock — the tensor work is
    tiny next to a DiT step, so contention with the next request's
    generation is negligible."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    if not frames:
        raise ValueError("no frames")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sigma = float(blur_radius)
    kernel_x = kernel_y = None
    pad = 0
    if sigma > 0:
        ksize = 2 * int(math.ceil(3.0 * sigma)) + 1
        pad = ksize // 2
        coords = torch.arange(ksize, dtype=torch.float32, device=dev) - pad
        gauss = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
        gauss = gauss / gauss.sum()
        kernel_x = gauss.view(1, 1, 1, ksize).repeat(3, 1, 1, 1)
        kernel_y = gauss.view(1, 1, ksize, 1).repeat(3, 1, 1, 1)

    out: list[Any] = []
    for start in range(0, len(frames), chunk_size):
        batch = torch.from_numpy(np.stack(frames[start:start + chunk_size])).to(dev)
        batch = batch.permute(0, 3, 1, 2).float().div_(255.0)  # N,C,H,W
        batch = F.interpolate(batch, scale_factor=0.5, mode="area")
        if kernel_x is not None:
            batch = F.conv2d(F.pad(batch, (pad, pad, 0, 0), mode="reflect"), kernel_x, groups=3)
            batch = F.conv2d(F.pad(batch, (0, 0, pad, pad), mode="reflect"), kernel_y, groups=3)
        batch = batch.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        batch = batch.permute(0, 2, 3, 1).cpu().numpy()
        out.extend(list(batch))
    return out


def build_s3_key(s3_cfg: Ltx25S3Config, filename: str) -> str:
    """<prefix>/<filename>, or just <filename> when prefix is empty (root).
    Leading/trailing slashes on the prefix are ignored."""
    prefix = s3_cfg.prefix.strip("/")
    return f"{prefix}/{filename}" if prefix else filename


def create_s3_client(s3_cfg: Ltx25S3Config) -> Any:
    import boto3

    return boto3.client(
        "s3",
        region_name=s3_cfg.region,
        aws_access_key_id=s3_cfg.access_key,
        aws_secret_access_key=s3_cfg.secret_key,
        **({"endpoint_url": s3_cfg.endpoint_url} if s3_cfg.endpoint_url else {}),
    )


def upload_file_to_s3(client: Any, s3_cfg: Ltx25S3Config, local_path: str | Path, key: str) -> str:
    """Upload an mp4 and return its s3://bucket/key URI (the caller's
    downstream signs/serves it)."""
    client.upload_file(str(local_path), s3_cfg.bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    return f"s3://{s3_cfg.bucket}/{key}"


def make_warmup_image(path: str | Path, width: int, height: int) -> None:
    """Synthetic but structured conditioning image (diagonal gradient)."""
    import numpy as np
    from PIL import Image

    x = np.linspace(0.0, 255.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 255.0, height, dtype=np.float32)[:, None]
    arr = np.stack(
        [
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            np.broadcast_to((x + y) / 2.0, (height, width)),
        ],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(arr).save(path)


def run_warmup(
    generator: Any,
    cfg: Ltx25ServerConfig,
    runs_per_shape: int = 1,
    log=print,
    encode_check: bool = True,
) -> None:
    """One generation per distinct compile shape so every dynamo trace /
    inductor compile happens before real traffic. Duplicate mode entries
    are traced once. With encode_check the first generation is also run
    through the CPU H.264 encoder to validate that path before serving.

    Each warmup request carries a last-frame anchor: flf2v changes only
    conditioning VALUES (inplace latent pinning, no extra tokens), so it
    shares the compiled graph with i2v — tracing it here costs nothing and
    exercises the anchor path once."""
    seen: set[tuple[int, int, int, int]] = set()
    encode_checked = not encode_check
    # Dynamo re-traces every shape in every process (expected, minutes per
    # mode); kernel compilation itself should be served from this cache dir.
    effective_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
    log(f"[warmup] inductor cache: {effective_cache or '(default, not persistent!)'}")
    workdir = Path(tempfile.mkdtemp(prefix="ltx25_warmup_"))
    try:
        for mode in cfg.modes:
            key = mode.shape_key()
            if key in seen:
                log(f"[warmup] {mode} duplicates an earlier mode; skipping")
                continue
            seen.add(key)
            stage1_width, stage1_height = mode.stage1_size()
            first = workdir / f"first_{mode.width}x{mode.height}.png"
            last = workdir / f"last_{mode.width}x{mode.height}.png"
            make_warmup_image(first, mode.width, mode.height)
            make_warmup_image(last, mode.width, mode.height)
            request = GenerationRequest(
                prompt=WARMUP_PROMPT,
                first_frame_path=str(first),
                last_frame_path=str(last),
                seed=42,
                image_crf=cfg.image_crf,
                last_frame_strength=cfg.last_frame_strength,
            )
            for run in range(runs_per_shape):
                t0 = time.perf_counter()
                log(f"[warmup] {mode.width}x{mode.height} f{mode.num_frames} "
                    f"(stage 1 at {stage1_width}x{stage1_height}) run {run + 1}/{runs_per_shape}…")
                result = generate_for_mode(generator, cfg, mode, request, workdir / "warmup.mp4")
                log(f"[warmup] {mode.width}x{mode.height} f{mode.num_frames} "
                    f"run {run + 1}/{runs_per_shape} wall={time.perf_counter() - t0:.1f}s")
                if not encode_checked:
                    encode_checked = True
                    encode_seconds = encode_video_h264(
                        result["frames"],
                        mode.fps,
                        workdir / "warmup.mp4",
                        bitrate_kbps=cfg.video_bitrate_kbps,
                        preset=cfg.x264_preset,
                        audio=result.get("audio"),
                        audio_sample_rate=result.get("audio_sample_rate"),
                        threads=cfg.encode_threads,
                        codec=cfg.video_codec,
                        extra_video_args=cfg.extra_video_args,
                    )
                    log(f"[warmup] CPU H.264 encode check passed ({encode_seconds:.1f}s)")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def cache_dir_size(path: str) -> str:
    """Human-readable size of the inductor cache dir (best effort)."""
    if not path or not Path(path).is_dir():
        return "n/a"
    try:
        out = subprocess.run(["du", "-sh", path], capture_output=True, text=True, check=True)
        return out.stdout.split()[0]
    except Exception:  # noqa: BLE001
        return "unknown"
