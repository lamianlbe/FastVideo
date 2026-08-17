# SPDX-License-Identifier: Apache-2.0
"""LTX2 model family pipeline presets."""
from fastvideo.api.presets import InferencePreset, PresetStageSpec
from fastvideo.pipelines.basic.ltx2.stage_overrides import (
    refine_stage_override_fields, )

_LTX2_NEGATIVE_PROMPT = ("blurry, out of focus, overexposed, underexposed, low contrast, "
                         "washed out colors, excessive noise, grainy texture, poor lighting, "
                         "flickering, motion blur, distorted proportions, unnatural skin "
                         "tones, deformed facial features, asymmetrical face, missing facial "
                         "features, extra limbs, disfigured hands, wrong hand count, "
                         "artifacts around text, inconsistent perspective, camera shake, "
                         "incorrect depth of field, background too sharp, background clutter, "
                         "distracting reflections, harsh shadows, inconsistent lighting "
                         "direction, color banding, cartoonish rendering, 3D CGI look, "
                         "unrealistic materials, uncanny valley effect, incorrect ethnicity, "
                         "wrong gender, exaggerated expressions, wrong gaze direction, "
                         "mismatched lip sync, silent or muted audio, distorted voice, "
                         "robotic voice, echo, background noise, off-sync audio, incorrect "
                         "dialogue, added dialogue, repetitive speech, jittery movement, "
                         "awkward pauses, incorrect timing, unnatural transitions, "
                         "inconsistent framing, tilted camera, flat lighting, inconsistent "
                         "tone, cinematic oversaturation, stylized filters, or AI artifacts.")

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Main denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

_REFINE_STAGE = PresetStageSpec(
    name="refine",
    kind="refinement",
    description="Latent-upsample + second-pass refine",
    allowed_overrides=refine_stage_override_fields(),
)

LTX2_BASE = InferencePreset(
    name="ltx2_base",
    version=1,
    model_family="ltx2",
    description="LTX-2 base at 512x768",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 10,
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 3.0,
        "num_inference_steps": 40,
        "negative_prompt": _LTX2_NEGATIVE_PROMPT,
        "ltx2_cfg_scale_video": 3.0,
        "ltx2_cfg_scale_audio": 7.0,
        "ltx2_modality_scale_video": 3.0,
        "ltx2_modality_scale_audio": 3.0,
        "ltx2_rescale_scale": 0.7,
        "ltx2_stg_scale_video": 1.0,
        "ltx2_stg_scale_audio": 1.0,
        "ltx2_stg_blocks_video": [29],
        "ltx2_stg_blocks_audio": [29],
    },
)

LTX2_3_BASE = InferencePreset(
    name="ltx2_3_base",
    version=1,
    model_family="ltx2",
    description="LTX-2.3 base at 512x768",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 10,
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "fps": 24,
        # LTX-2.3 base: 30 steps (vs 40 for LTX-2.0 base) + CFG 3.0 + negative
        # prompt; STG perturbs block 28 (vs 29 for LTX-2.0).
        "guidance_scale": 3.0,
        "num_inference_steps": 30,
        "negative_prompt": _LTX2_NEGATIVE_PROMPT,
        "ltx2_cfg_scale_video": 3.0,
        "ltx2_cfg_scale_audio": 7.0,
        "ltx2_modality_scale_video": 3.0,
        "ltx2_modality_scale_audio": 3.0,
        "ltx2_rescale_scale": 0.7,
        "ltx2_stg_scale_video": 1.0,
        "ltx2_stg_scale_audio": 1.0,
        "ltx2_stg_blocks_video": [28],
        "ltx2_stg_blocks_audio": [28],
    },
)

LTX2_DISTILLED = InferencePreset(
    name="ltx2_distilled",
    version=1,
    model_family="ltx2",
    description="LTX-2 distilled at 1024x1536",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 10,
        "height": 1024,
        "width": 1536,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 8,
        "negative_prompt": "",
    },
)

LTX2_TWO_STAGE = InferencePreset(
    name="ltx2_two_stage",
    version=1,
    model_family="ltx2",
    description="LTX-2 distilled with 2x spatial refine (stage 1 half-res + stage 2 upsample+denoise)",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, _REFINE_STAGE),
    defaults={
        "seed": 10,
        "height": 1024,
        "width": 1536,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 8,
        "negative_prompt": "",
    },
    stage_defaults={
        "refine": {
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
        },
    },
)

LTX2_5_DEV = InferencePreset(
    name="ltx2_5_dev",
    version=1,
    model_family="ltx2",
    description="LTX-2.5 dev joint video/audio generation at 512x768",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 10,
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 3.0,
        "num_inference_steps": 30,
        "negative_prompt": _LTX2_NEGATIVE_PROMPT,
        "ltx2_cfg_scale_video": 3.0,
        "ltx2_cfg_scale_audio": 7.0,
        "ltx2_modality_scale_video": 3.0,
        "ltx2_modality_scale_audio": 3.0,
        "ltx2_rescale_scale": 0.7,
        "ltx2_stg_scale_video": 1.0,
        "ltx2_stg_scale_audio": 1.0,
        "ltx2_stg_blocks_video": [28],
        "ltx2_stg_blocks_audio": [28],
        "ltx2_image_crf": 18.0,
        "ltx2_use_ancestral_sampler": False,
    },
)

LTX2_5_DISTILLED = InferencePreset(
    name="ltx2_5_distilled",
    version=1,
    model_family="ltx2",
    description="LTX-2.5 distilled joint video/audio generation at 1024x1536",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 10,
        "height": 1024,
        "width": 1536,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 8,
        "negative_prompt": "",
        "ltx2_image_crf": 18.0,
        "ltx2_use_ancestral_sampler": True,
    },
)

LTX2_5_DISTILLED_TWO_STAGE = InferencePreset(
    name="ltx2_5_distilled_two_stage",
    version=1,
    model_family="ltx2",
    description="LTX-2.5 distilled joint video/audio generation with 2x spatial refinement",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, _REFINE_STAGE),
    defaults=dict(LTX2_5_DISTILLED.defaults),
    stage_defaults={
        "refine": {
            "num_inference_steps": 3,
            "guidance_scale": 1.0,
            "image_crf": 18,
        },
    },
)

# The production ComfyUI two-stage distilled i2v/flf2v recipe on the LTX-2.5
# dev transformer + distilled LoRA (per-stage strengths ~0.7 / ~0.5). These
# defaults cover the per-request sampling surface; the engine-level half of
# the recipe (euler_ancestral_cfg_pp both stages, LTXVScheduler stage-1 sigmas
# with max_shift=4.0 / base_shift=1.5 / stretch / terminal=0.1, manual stage-2
# sigmas [0.85, 0.7250, 0.4219, 0.0], refine + per-stage LoRA strengths) is
# wired by examples/inference/basic/basic_ltx2_5_i2av_two_stage.py.
LTX2_5_DISTILLED_TWO_STAGE_I2V = InferencePreset(
    name="ltx2_5_distilled_two_stage_i2v",
    version=1,
    model_family="ltx2",
    description=("LTX-2.5 two-stage distilled i2v/flf2v recipe (dev transformer + distilled LoRA): "
                 "stage 1 at half resolution with CFG++ ancestral sampling, x2 latent upsample, "
                 "stage-2 re-pin + refine, joint audio"),
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, _REFINE_STAGE),
    defaults={
        "seed": 10,
        # Final (stage-2) resolution; LTX2RefineInitStage halves it for stage 1,
        # putting the stage-1 long side at ~1024 like the ComfyUI recipe.
        "height": 1152,
        "width": 2048,
        "num_frames": 121,
        "fps": 24,
        # CFG++ at cfg=1: the sampler still runs the uncond pass (except at the
        # degenerate sigma=1.0 first step); an empty negative prompt suffices.
        "guidance_scale": 1.0,
        "num_inference_steps": 8,
        "negative_prompt": "",
        # LTXVPreprocess img_compression=38 (H.264 CRF re-encode of the
        # conditioning image); stage 2 reuses the same value.
        "ltx2_image_crf": 38.0,
        # Plain CFG++ guider: no STG / modality-isolation / rescale terms.
        "ltx2_cfg_scale_video": 1.0,
        "ltx2_cfg_scale_audio": 1.0,
        "ltx2_modality_scale_video": 1.0,
        "ltx2_modality_scale_audio": 1.0,
        "ltx2_rescale_scale": 0.0,
        "ltx2_stg_scale_video": 0.0,
        "ltx2_stg_scale_audio": 0.0,
        # The recipe uses the ComfyUI-style cfg_pp sampler (ltx2_sampler /
        # ltx2_refine_sampler engine args), not the official 2.5 ancestral path.
        "ltx2_use_ancestral_sampler": False,
    },
    stage_defaults={
        "refine": {
            # Matches the manual stage-2 sigma list [0.85, 0.7250, 0.4219, 0.0].
            "num_inference_steps": 3,
            "guidance_scale": 1.0,
        },
    },
)

ALL_PRESETS = (
    LTX2_BASE,
    LTX2_3_BASE,
    LTX2_DISTILLED,
    LTX2_TWO_STAGE,
    LTX2_5_DEV,
    LTX2_5_DISTILLED,
    LTX2_5_DISTILLED_TWO_STAGE,
    LTX2_5_DISTILLED_TWO_STAGE_I2V,
)

__all__ = [
    "ALL_PRESETS",
    "LTX2_BASE",
    "LTX2_3_BASE",
    "LTX2_DISTILLED",
    "LTX2_TWO_STAGE",
    "LTX2_5_DEV",
    "LTX2_5_DISTILLED",
    "LTX2_5_DISTILLED_TWO_STAGE",
    "LTX2_5_DISTILLED_TWO_STAGE_I2V",
]
