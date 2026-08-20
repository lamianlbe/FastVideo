"""Shared engine for the LTX-2.3 production HTTP server.

Wraps the validated DMD-stack recipe (the ComfyUI two-pass I2V workflow
port, see examples/inference/basic/eager_ltx2_3_ancestral_reference_i2v.py
and .../eager_ltx2_3_flf_i2v.py) behind a config file + a small API used by
both server.py (online serving) and build_compile_cache.py (offline
compilation).

Import order matters: call ``setup_environment(cfg)`` BEFORE anything that
imports torch/fastvideo so TORCHINDUCTOR_CACHE_DIR and the attention
backend are picked up. All fastvideo/torch imports in this module are
deliberately function-local.
"""

from __future__ import annotations

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
# Validated DMD-stack sampling recipe (ComfyUI workflow port).
# ---------------------------------------------------------------------------
DEFAULT_STAGE1_SIGMAS = [1.000, 0.955, 0.893, 0.812, 0.715, 0.603, 0.482, 0.241, 0.121, 0.0]
DEFAULT_STAGE2_SIGMAS = [0.92, 0.725, 0.421875, 0.0]
DEFAULT_NEGATIVE_PROMPT = ("3D, phasing, captions, VR, still image, bad quality, subtitles, text, "
                           "watermark, overlay effects, pc game, yelling, console game, video game, "
                           "cartoon, childish, ugly, text, blur, logo, wordmark, static, low quality, "
                           "noise, white noise, bleep, censoring, censor, bleeping, beep, beeping, "
                           "newscast, interview, podcast, non-english, foreign language, russian, "
                           "chinese, japanese, mutant, horror, 70's, film grain, cinematic, comedy, "
                           "stand-up ")
# Stage-1 conditioning CRF = LTXVPreprocess "motion strength".
DEFAULT_IMAGE_CRF = 35.0
# Server default: stage 2 re-anchors with a clean (CRF 0) encode so the
# final first frame stays sharp while stage 1 keeps the motion CRF.
DEFAULT_IMAGE_CRF_STAGE2 = 0.0
DEFAULT_LAST_FRAME_STRENGTH = 0.8
# ComfyUI STGGuiderAdvanced picks cfg by SIGMA LOOKUP, not by step index:
# for the current sigma it takes the smallest entry of its own sigma list
# that is still >= that sigma and reads the cfg at that entry's index. The
# node's list is the workflow's RAW ManualSigmas, while the sampler runs the
# eased schedule, so the effective per-step list has to be derived — see
# derive_stage1_cfg_values(). These are the optimized workflow's raw values
# (node 926:944; the cfg list is longer than the sigma list there, which is
# fine — only indices reachable from the sigma list are ever read). Shipped
# for reference; the config knobs default to empty (off).
WORKFLOW_CFG_SIGMA_LIST = [
    1.0, 0.9550, 0.8925, 0.8120, 0.7150, 0.6030, 0.4824, 0.3618, 0.2412, 0.1206, 0.0
]
WORKFLOW_CFG_VALUES = [2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# Guide-image longer edge of the workflow's ResizeImageMaskNode (node
# 926:956, "scale longer dimension" + lanczos).
DEFAULT_GUIDE_LONGER_SIZE = 1536
WARMUP_PROMPT = ("A person slowly turns their head toward the camera and smiles, "
                 "soft warm light, gentle camera drift.")


@dataclass(frozen=True)
class Ltx23Mode:
    """One supported (resolution, frames, fps) combination."""
    width: int
    height: int
    num_frames: int
    fps: int

    def validate(self) -> None:
        if self.width % 64 or self.height % 64:
            raise ValueError(f"mode {self.width}x{self.height}: both dims must be divisible by 64 "
                             "(x2 refine; the x1.5 upsampler additionally needs divisibility by 96)")
        if (self.num_frames - 1) % 8:
            raise ValueError(f"mode num_frames={self.num_frames}: must be 8*k+1 "
                             "(temporal VAE compression)")
        if self.fps <= 0:
            raise ValueError(f"mode fps={self.fps}: must be positive")

    def shape_key(self) -> tuple[int, int, int, int]:
        """Compile-shape identity. fps is part of it: the audio latent
        length is derived from the clip duration (num_frames / fps), so a
        different fps at the same frame count changes the DiT sequence
        length and requires its own compiled graph."""
        return (self.width, self.height, self.num_frames, self.fps)


@dataclass
class Ltx23S3Config:
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
class Ltx23ServerConfig:
    model_path: str
    modes: list[Ltx23Mode]
    upsampler_path: str = ""  # "" = auto-detect <model>/spatial_upscaler|spatial_upsampler
    # Stage-2 (refine) transformer directory. The two ComfyUI passes run
    # DIFFERENT models (the distilled LoRA is merged at different strengths:
    # stage 1 at 0.88 video/other + 0.9 audio/cross, stage 2 at 0.58 + 1.0),
    # so sharing stage 1's weights in the refine pass visibly degrades the
    # result. "" = auto-detect <model>/transformer_stage2 (or the repo's
    # model_index.json "fastvideo_refine_transformer_path"), falling back to
    # the shared stage-1 transformer. Relative paths resolve inside the model
    # root. Both transformers stay resident on the GPU.
    stage2_transformer_path: str = ""
    quant: str = "nvfp4"  # nvfp4 | none
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
    # deploy/install.sh).
    fa4: bool = True
    # FP8 (e4m3) attention per LTX-2 stage: q/k/v quantized with per-head
    # descales, both attention GEMMs at the fp8 tensor-core rate. OFF = the
    # current bf16 FA4 path. Toggling changes the compiled graph — re-run
    # build_compile_cache.py after changing these.
    fa4_fp8_stage1: bool = False
    fa4_fp8_stage2: bool = False
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
    s3: Ltx23S3Config | None = None
    stage1_sigmas: list[float] = field(default_factory=lambda: list(DEFAULT_STAGE1_SIGMAS))
    stage2_sigmas: list[float] = field(default_factory=lambda: list(DEFAULT_STAGE2_SIGMAS))
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    image_crf: float = DEFAULT_IMAGE_CRF
    image_crf_stage2: float = DEFAULT_IMAGE_CRF_STAGE2
    last_frame_strength: float = DEFAULT_LAST_FRAME_STRENGTH

    # --- ComfyUI-parity knobs (all default to today's server behaviour) ---
    # Stage-1 first-frame conditioning:
    #   inplace_and_reference — hard-pin latent frame 0 to the encoded image
    #     (strength 1.0) AND prepend a clean reference-token prefix.
    #   guide_only — the ComfyUI workflow's actual conditioning: no in-place
    #     write at all, only the appended guide/reference tokens at
    #     stage1_guide_strength (LTXPlusBatchAddGuide -> LTXVAddGuide
    #     .append_keyframe, noise_mask = 1 - strength). Stage 2 keeps the
    #     in-place keyframe at 1.0 either way (= LTXVImgToVideoInplace).
    stage1_conditioning: str = "inplace_and_reference"
    stage1_guide_strength: float = 0.8  # node 926:939 strength
    # Comfy also biases content<->guide self-attention by log(strength)
    # (LTXVModel._build_self_attention_mask), scaling those attention weights
    # by ~0.8x. Opt-in: it forces attn1 onto the masked SDPA path, giving up
    # FA4 for stage 1. Only meaningful with stage1_conditioning: guide_only.
    guide_attention_bias: bool = False
    # Per-step CFG. Give the RAW sigma list + cfg values exactly as they
    # appear in the STGGuiderAdvanced node; the engine derives the per-step
    # list for the ACTUAL stage1_sigmas with comfy's sigma lookup and logs
    # it at startup. Both empty (default) = flat cfg, i.e. today's behaviour.
    stage1_cfg_sigma_list: list[float] = field(default_factory=list)
    stage1_cfg_values_by_sigma: list[float] = field(default_factory=list)
    # Guide-image geometry:
    #   cover_crop — aspect-preserving resize + center crop to the mode
    #     resolution (today's behaviour, applied inside the pipeline): an
    #     upload whose aspect differs from the mode loses its edges.
    #   comfy_lanczos_stretch — the workflow's path: lanczos resize so the
    #     LONGER edge is guide_longer_size (aspect preserved, no crop), then
    #     LTXVAddGuide.encode's plain bilinear resize to the mode resolution
    #     (comfy.utils.common_upscale with crop="disabled"), i.e. the image
    #     is squashed rather than cropped when the aspect differs.
    guide_resize: str = "cover_crop"
    guide_longer_size: int = DEFAULT_GUIDE_LONGER_SIZE


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
    last_in_upscale: bool = True
    image_crf: float = DEFAULT_IMAGE_CRF
    image_crf_stage2: float | None = DEFAULT_IMAGE_CRF_STAGE2


def derive_stage1_cfg_values(
    stage1_sigmas: list[float],
    cfg_sigma_list: list[float],
    cfg_values: list[float],
) -> list[float]:
    """Per-step CFG for ``stage1_sigmas`` under ComfyUI's sigma lookup.

    Port of ``STGGuiderAdvanced.sigma_to_params_mapping``: for the sigma the
    sampler is currently at, take the SMALLEST entry of ``cfg_sigma_list``
    that is still ``>= sigma`` and read ``cfg_values`` at that entry's (first)
    index; when no entry is >= sigma, comfy falls through to the last cfg
    value. The lists come from the guider node and are independent of the
    sampler's schedule, which is why this cannot be a plain zip.

    Returns one value per denoising step (``len(stage1_sigmas) - 1``, the
    trailing 0.0 is the endpoint, not a step) — the form
    ``ltx2_stage1_cfg_values`` expects, which FastVideo indexes by step.

    ``cfg_values`` is one shorter than the sigma list in the shipped workflow
    (13 vs 14); an index that lands past its end is clamped to the last
    entry (comfy would raise IndexError there — it can only happen at
    sigma 0, which is never sampled).
    """
    if not cfg_sigma_list or not cfg_values:
        raise ValueError("derive_stage1_cfg_values needs a non-empty sigma list and cfg list")
    derived: list[float] = []
    for sigma in stage1_sigmas[:-1]:
        higher = [s for s in cfg_sigma_list if s >= sigma]
        idx = cfg_sigma_list.index(min(higher)) if higher else len(cfg_values) - 1
        derived.append(float(cfg_values[min(idx, len(cfg_values) - 1)]))
    return derived


def resolve_stage1_cfg_values(cfg: Ltx23ServerConfig) -> list[float] | None:
    """``cfg``'s raw sigma/cfg lists mapped onto its stage-1 schedule, or
    None when the per-step schedule is off (both lists empty)."""
    if not cfg.stage1_cfg_sigma_list or not cfg.stage1_cfg_values_by_sigma:
        return None
    return derive_stage1_cfg_values(cfg.stage1_sigmas, cfg.stage1_cfg_sigma_list, cfg.stage1_cfg_values_by_sigma)


def load_config(path: str | Path) -> Ltx23ServerConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    modes_raw = raw.pop("modes", None)
    if not modes_raw:
        raise ValueError(f"{path}: 'modes' must list at least one "
                         "{width, height, num_frames, fps} combination")
    modes = [Ltx23Mode(**m) for m in modes_raw]
    for mode in modes:
        mode.validate()
    s3_raw = raw.pop("s3", None)
    s3_cfg = None
    if s3_raw is not None:
        if not isinstance(s3_raw, dict):
            raise ValueError(f"{path}: 's3' must be a mapping")
        s3_cfg = Ltx23S3Config(**s3_raw)
        s3_cfg.validate()
    known = {f for f in Ltx23ServerConfig.__dataclass_fields__ if f not in ("modes", "s3")}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    cfg = Ltx23ServerConfig(modes=modes, s3=s3_cfg, **raw)
    if not cfg.model_path:
        raise ValueError(f"{path}: 'model_path' is required")
    if cfg.quant not in ("nvfp4", "fp8", "fp8_channel", "none"):
        raise ValueError(f"{path}: quant must be nvfp4 | fp8 | fp8_channel | none, got {cfg.quant}")
    if any(not isinstance(k, str) or not k.strip() for k in cfg.api_keys):
        raise ValueError(f"{path}: api_keys entries must be non-empty strings")
    cvd = cfg.cuda_visible_devices
    if cvd and not all(p.strip().isdigit() for p in cvd.split(",")):
        raise ValueError(f"{path}: cuda_visible_devices must be comma-separated GPU indices, got {cvd!r}")
    validate_parity_config(cfg, source=str(path))
    return cfg


def validate_parity_config(cfg: Ltx23ServerConfig, source: str = "config") -> None:
    """Check the ComfyUI-parity knobs (split out so tests can call it on a
    constructed config without a YAML file)."""
    if cfg.stage1_conditioning not in ("inplace_and_reference", "guide_only"):
        raise ValueError(f"{source}: stage1_conditioning must be inplace_and_reference | guide_only, "
                         f"got {cfg.stage1_conditioning!r}")
    if not 0.0 < cfg.stage1_guide_strength <= 1.0:
        raise ValueError(f"{source}: stage1_guide_strength must be in (0, 1], got {cfg.stage1_guide_strength}")
    if bool(cfg.stage1_cfg_sigma_list) != bool(cfg.stage1_cfg_values_by_sigma):
        raise ValueError(f"{source}: stage1_cfg_sigma_list and stage1_cfg_values_by_sigma must be set together "
                         "(both empty disables the per-step CFG schedule)")
    if cfg.stage1_cfg_sigma_list:
        sig = cfg.stage1_cfg_sigma_list
        if any(b > a for a, b in zip(sig, sig[1:], strict=False)):
            raise ValueError(f"{source}: stage1_cfg_sigma_list must be non-increasing, got {sig}")
        if any(v < 1.0 for v in cfg.stage1_cfg_values_by_sigma):
            raise ValueError(f"{source}: stage1_cfg_values_by_sigma entries must be >= 1.0, got "
                             f"{cfg.stage1_cfg_values_by_sigma}")
        if len(cfg.stage1_cfg_values_by_sigma) < len(sig) - 1:
            # The workflow's node ships MORE cfg values than sigmas (extra
            # tail entries are unreachable) — that is fine. Fewer than
            # len(sigmas) - 1 would leave reachable indices unmapped.
            raise ValueError(f"{source}: stage1_cfg_values_by_sigma needs at least len(stage1_cfg_sigma_list) - 1 "
                             f"entries ({len(sig) - 1}), got {len(cfg.stage1_cfg_values_by_sigma)}")
    if cfg.guide_resize not in ("cover_crop", "comfy_lanczos_stretch"):
        raise ValueError(f"{source}: guide_resize must be cover_crop | comfy_lanczos_stretch, "
                         f"got {cfg.guide_resize!r}")
    if cfg.guide_longer_size < 64:
        raise ValueError(f"{source}: guide_longer_size must be >= 64, got {cfg.guide_longer_size}")


def setup_environment(cfg: Ltx23ServerConfig) -> None:
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
    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")


def scale_longer_dimension(size: tuple[int, int], longer_size: int) -> tuple[int, int]:
    """(w, h) scaled so the longer edge is ``longer_size``, aspect preserved.
    Verbatim rounding of comfy_extras.nodes_post_processing
    .scale_longer_dimension (used by the workflow's ResizeImageMaskNode)."""
    width, height = size
    if height > width:
        return max(1, round((width / height) * longer_size)), longer_size
    if width > height:
        return longer_size, max(1, round((height / width) * longer_size))
    return longer_size, longer_size


def preprocess_guide_image(src: str | Path, dst: str | Path, width: int, height: int, longer_size: int) -> str:
    """Write the workflow's guide-image pipeline result to ``dst``.

    ComfyUI feeds the guide through ResizeImageMaskNode ("scale longer
    dimension" ``longer_size``, lanczos — aspect preserved, NO crop) and then
    LTXVAddGuide.encode resizes to the latent grid with
    ``common_upscale(..., "bilinear", crop="disabled")``, i.e. a plain
    non-antialiased stretch. Producing an image that is already exactly
    ``width x height`` makes the pipeline's own resize + center crop a no-op,
    so this reproduces both steps without touching fastvideo.

    Deviation from comfy: the intermediate lands in an 8-bit PNG instead of
    staying a float tensor (comfy's lanczos step also round-trips through
    uint8, so only the final bilinear result is additionally quantized).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    image = Image.open(src).convert("RGB")
    lanczos_size = scale_longer_dimension(image.size, longer_size)
    # comfy.utils.lanczos == PIL LANCZOS on uint8.
    image = image.resize(lanczos_size, resample=Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    # torch bilinear without antialias — exactly what common_upscale does.
    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear")
    array = (tensor[0].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
    Image.fromarray(array).save(dst)
    return str(dst)


def match_mode(
    modes: list[Ltx23Mode],
    width: int,
    height: int,
    num_frames: int,
    fps: int,
) -> tuple[Ltx23Mode, bool]:
    """Exact (width, height, num_frames, fps) match, else the mode with the
    closest resolution (aspect-aware log distance); frames/fps only break
    ties. Returns (mode, exact)."""
    for mode in modes:
        if (mode.width, mode.height, mode.num_frames, mode.fps) == (width, height, num_frames, fps):
            return mode, True

    def distance(mode: Ltx23Mode) -> tuple[float, int, int]:
        res = (math.log(width / mode.width)**2 + math.log(height / mode.height)**2)
        return (res, abs(num_frames - mode.num_frames), abs(fps - mode.fps))

    return min(modes, key=distance), False


def resolve_stage2_transformer(model_root: str, override: str = "") -> str | None:
    """Stage-2 transformer directory, or None to share stage 1's.

    An explicit override must exist (fail at startup, not mid-request);
    otherwise probe the conventional <model>/transformer_stage2 directory.
    The repo-level model_index.json "fastvideo_refine_transformer_path" key
    is handled by the pipeline itself and needs no engine plumbing."""
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = Path(model_root) / override
        if not (candidate / "config.json").is_file():
            raise FileNotFoundError(f"stage2_transformer_path {override!r}: no config.json under {candidate}")
        return str(candidate)
    candidate = Path(model_root) / "transformer_stage2"
    if (candidate / "config.json").is_file():
        return str(candidate)
    return None


def resolve_upsampler(model_root: str, override: str = "") -> str:
    candidates = ([override] if override else []) + [
        str(Path(model_root) / "spatial_upscaler"),
        str(Path(model_root) / "spatial_upsampler"),
    ]
    for cand in candidates:
        if cand and (Path(cand) / "config.json").is_file():
            return cand
    raise FileNotFoundError("No spatial upsampler found; set 'upsampler_path' in the config or add "
                            "spatial_upscaler/ to the model repo.")


def create_generator(cfg: Ltx23ServerConfig) -> Any:
    """Build the resident VideoGenerator with the validated recipe wired in.
    The reference-token image is per-request (ltx2_reference_image_path in
    generation kwargs), everything else is fixed at init."""
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
    stage2_transformer_path = resolve_stage2_transformer(model_root, cfg.stage2_transformer_path)
    if stage2_transformer_path is not None:
        print(f"[engine] stage-2 refine transformer: {stage2_transformer_path} "
              "(stage 1 and stage 2 run different merged DiTs)")
    else:
        print("[engine] stage-2 refine transformer: shared with stage 1")

    pipeline_config = PipelineConfig.from_pretrained(model_root)
    # Linear quantization ladder (loss high -> none): nvfp4 (e2m1, fastest),
    # fp8 (e4m3 per-tensor), fp8_channel (per-channel weights + per-token
    # activations, most conservative quantized tier), none (bf16).
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
        torch_compile_kwargs = {
            "backend": "inductor",
            "fullgraph": True,
            "mode": "default",
            "dynamic": False,
        }
        compile_kwargs = dict(
            enable_torch_compile=True,
            enable_torch_compile_text_encoder=True,
            enable_torch_compile_vae=True,
            torch_compile_kwargs=torch_compile_kwargs,
            torch_compile_kwargs_vae=torch_compile_kwargs,
        )

    # Per-step CFG derived from the raw guider lists against the ACTUAL
    # stage-1 schedule (see derive_stage1_cfg_values); logged because the
    # mapping is a sigma lookup and not obvious from the config.
    stage1_cfg_values = resolve_stage1_cfg_values(cfg)
    if stage1_cfg_values is not None:
        print(f"[engine] stage-1 per-step CFG {stage1_cfg_values} "
              f"(derived from {len(cfg.stage1_cfg_sigma_list)} guider sigmas over "
              f"{len(cfg.stage1_sigmas) - 1} steps)")
    guide_strength = (cfg.stage1_guide_strength if cfg.stage1_conditioning == "guide_only" else None)
    if guide_strength is not None:
        print(f"[engine] stage-1 conditioning: guide_only (append-style guide tokens at "
              f"strength {guide_strength}, no in-place frame-0 pin)")
    return VideoGenerator.from_pretrained(
        model_root,
        num_gpus=cfg.num_gpus,
        pipeline_config=pipeline_config,
        **compile_kwargs,
        ltx2_refine_enabled=True,
        ltx2_refine_upsampler_path=upsampler_path,
        ltx2_refine_transformer_path=stage2_transformer_path,
        ltx2_refine_lora_path="",
        ltx2_refine_guidance_scale=1.0,
        ltx2_refine_add_noise=True,
        ltx2_sampler="euler_ancestral",
        ltx2_refine_sampler="euler_ancestral_cfg_pp",
        ltx2_stage1_sigmas=cfg.stage1_sigmas,
        ltx2_stage2_sigmas=cfg.stage2_sigmas,
        ltx2_stage1_cfg_values=stage1_cfg_values,
        ltx2_reference_strength=1.0,
        ltx2_reference_position_mode="reference",
        ltx2_guide_attention_bias=cfg.guide_attention_bias,
        ltx2_reference_zero_timesteps=False,
        # None keeps the clean strength-scaled prefix; a float switches
        # stage 1 to comfy's append_keyframe guide semantics.
        ltx2_reference_guide_strength=guide_strength,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        ltx2_vae_tiling=False,
    )


def generate_for_mode(
    generator: Any,
    cfg: Ltx23ServerConfig,
    mode: Ltx23Mode,
    request: GenerationRequest,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run one generation at the given mode's shape and return RAW frames
    (+ audio) instead of writing an mp4 — encoding happens on the CPU via
    encode_video_h264, outside the caller's GPU lock, so the next request's
    generation overlaps the previous request's encode.

    Conditioning images are cover-fit (aspect-preserving resize + center
    crop, no letterboxing) to the mode resolution inside the pipeline, so
    callers can pass uploads as-is; ``guide_resize: comfy_lanczos_stretch``
    instead pre-resizes them here the way the ComfyUI workflow does.
    ``output_path`` is only pipeline path bookkeeping; nothing is written to
    it — the scratch dir around it does receive the pre-resized inputs."""
    last_latent_idx = (mode.num_frames - 1) // 8
    workdir = Path(output_path).parent
    first_path = request.first_frame_path
    last_path = request.last_frame_path
    if cfg.guide_resize == "comfy_lanczos_stretch":
        first_path = preprocess_guide_image(request.first_frame_path, workdir / "guide_first.png", mode.width,
                                            mode.height, cfg.guide_longer_size)
        if last_path:
            last_path = preprocess_guide_image(last_path, workdir / "guide_last.png", mode.width, mode.height,
                                               cfg.guide_longer_size)

    first_keyframe = (first_path, 0, 1.0)
    tail_keyframe = ((last_path, last_latent_idx, request.last_frame_strength) if last_path else None)
    if cfg.stage1_conditioning == "guide_only":
        # Comfy never writes the first frame into the latent: it appends the
        # guide as extra tokens (handled by the reference prefix below), so
        # stage 1 only carries the optional FLF tail anchor. Stage 2 still
        # gets the in-place keyframe = LTXVImgToVideoInplace(strength 1).
        images: list[tuple[str, int, float]] = [tail_keyframe] if tail_keyframe else []
        images_stage2: list[tuple[str, int, float]] | None = [first_keyframe]
        if tail_keyframe and request.last_in_upscale:
            images_stage2.append(tail_keyframe)
    else:
        images = [first_keyframe]
        if tail_keyframe:
            images.append(tail_keyframe)
        # None = same keyframes in both stages; a reduced list keeps the tail
        # anchor out of the stage-2 refine pass.
        images_stage2 = None
        if tail_keyframe and not request.last_in_upscale:
            images_stage2 = [first_keyframe]

    result = generator.generate_video(
        prompt=request.prompt,
        negative_prompt=(request.negative_prompt if request.negative_prompt else cfg.negative_prompt),
        output_path=str(output_path),
        seed=request.seed,
        guidance_scale=1.0,
        height=mode.height,
        width=mode.width,
        num_frames=mode.num_frames,
        fps=mode.fps,
        num_inference_steps=len(cfg.stage1_sigmas) - 1,
        ltx2_images=images,
        ltx2_images_stage2=images_stage2,
        ltx2_image_crf=request.image_crf,
        ltx2_image_crf_stage2=request.image_crf_stage2,
        # Identity reference tokens always come from the first frame — a
        # constant-on setting so the DiT sequence length (and thus the
        # compiled graph) never changes between i2v and FLF requests. Under
        # stage1_conditioning=guide_only this prefix IS the first-frame
        # conditioning (comfy's appended guide keyframe).
        ltx2_reference_image_path=first_path,
        ltx2_stg_scale_video=0.0,
        ltx2_stg_scale_audio=0.0,
        ltx2_cfg_scale_video=1.0,
        ltx2_cfg_scale_audio=1.0,
        ltx2_modality_scale_video=1.0,
        ltx2_modality_scale_audio=1.0,
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
    rgb24 and letting ffmpeg convert (used by CPU-only test hosts). The
    frame-by-frame PyAV path is gone — one native ffmpeg call encodes the
    whole clip with proper x264 threading."""
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


def build_s3_key(s3_cfg: Ltx23S3Config, filename: str) -> str:
    """<prefix>/<filename>, or just <filename> when prefix is empty (root).
    Leading/trailing slashes on the prefix are ignored."""
    prefix = s3_cfg.prefix.strip("/")
    return f"{prefix}/{filename}" if prefix else filename


def create_s3_client(s3_cfg: Ltx23S3Config) -> Any:
    import boto3

    return boto3.client(
        "s3",
        region_name=s3_cfg.region,
        aws_access_key_id=s3_cfg.access_key,
        aws_secret_access_key=s3_cfg.secret_key,
        **({"endpoint_url": s3_cfg.endpoint_url} if s3_cfg.endpoint_url else {}),
    )


def upload_file_to_s3(client: Any, s3_cfg: Ltx23S3Config, local_path: str | Path, key: str) -> str:
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
    cfg: Ltx23ServerConfig,
    runs_per_shape: int = 1,
    log=print,
    encode_check: bool = True,
) -> None:
    """One generation per distinct compile shape so every dynamo trace /
    inductor compile happens before real traffic. Duplicate mode entries
    are traced once. With encode_check the first generation is also run
    through the CPU H.264 encoder to validate that path before serving."""
    seen: set[tuple[int, int, int, int]] = set()
    encode_checked = not encode_check
    # Dynamo re-traces every shape in every process (expected, minutes per
    # mode); kernel compilation itself should be served from this cache dir.
    effective_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
    log(f"[warmup] inductor cache: {effective_cache or '(default, not persistent!)'}")
    workdir = Path(tempfile.mkdtemp(prefix="ltx23_warmup_"))
    try:
        for mode in cfg.modes:
            key = mode.shape_key()
            if key in seen:
                log(f"[warmup] {mode} duplicates an earlier mode; skipping")
                continue
            seen.add(key)
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
                image_crf_stage2=cfg.image_crf_stage2,
                last_frame_strength=cfg.last_frame_strength,
            )
            for run in range(runs_per_shape):
                t0 = time.perf_counter()
                log(f"[warmup] {mode.width}x{mode.height} f{mode.num_frames} "
                    f"run {run + 1}/{runs_per_shape}…")
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
