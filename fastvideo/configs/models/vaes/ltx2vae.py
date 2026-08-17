# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 VAE configuration.
"""

from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class LTX2VAEArchConfig(VAEArchConfig):
    # Mirrors LTX-2 safetensors metadata config under "vae"
    _class_name: str = "CausalVideoAutoencoder"
    dims: int = 3
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 128
    z_dim: int = 128  # follow num_channels_latents
    encoder_blocks: list = field(default_factory=list)
    decoder_blocks: list = field(default_factory=list)
    patch_size: int = 4
    norm_layer: str = "pixel_norm"
    latent_log_var: str = "uniform"
    encoder_spatial_padding_mode: str = "zeros"
    decoder_spatial_padding_mode: str = "reflect"
    causal_decoder: bool = False
    timestep_conditioning: bool = True
    decoder_base_channels: int = 128
    use_quant_conv: bool = False
    scaling_factor: float = 1.0
    normalize_latent_channels: bool = False

    # Match FastVideo naming for compression ratios (LTX-2 default)
    temporal_compression_ratio: int = 8
    spatial_compression_ratio: int = 32


@dataclass
class LTX2VAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=LTX2VAEArchConfig)

    # LTX-2 tiling defaults (match ltx_core.video_vae.TilingConfig.default()).
    ltx2_spatial_tile_size_in_pixels: int = 512
    ltx2_spatial_tile_overlap_in_pixels: int = 64
    ltx2_temporal_tile_size_in_frames: int = 64
    ltx2_temporal_tile_overlap_in_frames: int = 24


@dataclass
class LTX2DiffusionVAEArchConfig(VAEArchConfig):
    """LTX-2.5 diffusion (NATTEN) decoder VAE arch config.

    Field names mirror the official checkpoint metadata (``config.vae`` with nested
    ``encoder``/``decoder`` sections and ``model_output_type`` as a sibling of ``decoder``),
    so a converted checkpoint's config.json drives instantiation directly. Defaults below are
    the official class defaults; the shipped LTX-2.5 checkpoint overrides most of them
    (stage_channels (2048, 1024, 512, 512, 256), stage5_kernel (11, 11, 11), x0 output, ...).
    """
    _class_name: str = "CausalDiffusionVAE"
    dims: int = 3
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 128
    z_dim: int = 128  # follow num_channels_latents

    # Conv encoder half (byte-identical to the conv VAE's encoder).
    encoder_blocks: list = field(default_factory=list)
    patch_size: int = 4
    norm_layer: str = "pixel_norm"
    latent_log_var: str = "uniform"
    encoder_spatial_padding_mode: str = "zeros"

    # Diffusion decoder half (official DiffusionVideoDecoder defaults).
    decoder_head_dim: int = 64
    decoder_stage_channels: tuple = (1024, 512, 256, 256, 128)
    decoder_stage_depths: tuple = (4, 6, 4, 2, 8)
    decoder_stage_kernels: tuple = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5), (3, 3, 3))
    decoder_upsamples: tuple = (((1, 2, 2), 2), ((2, 1, 1), 2), ((2, 2, 2), 1), ((2, 2, 2), 2))
    decoder_stage5_kernel: tuple = (3, 7, 7)
    decoder_stage5_channels: int | None = None
    decoder_t_emb_dim: int = 384
    default_num_inference_steps: int = 2
    timestep_scale_multiplier: float = 1.0
    model_output_type: str = "v"

    scaling_factor: float = 1.0
    temporal_compression_ratio: int = 8
    spatial_compression_ratio: int = 32


@dataclass
class LTX2DiffusionVAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=LTX2DiffusionVAEArchConfig)

    # Diffusion-decoder tiling defaults in output pixels/frames (reference implementation
    # values; the difference between size and stride is the blended overlap).
    ltx2_diffusion_tile_sample_min_height: int = 768
    ltx2_diffusion_tile_sample_min_width: int = 768
    ltx2_diffusion_tile_sample_min_num_frames: int = 80
    ltx2_diffusion_tile_sample_stride_height: int = 704
    ltx2_diffusion_tile_sample_stride_width: int = 704
    ltx2_diffusion_tile_sample_stride_num_frames: int = 56
