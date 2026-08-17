# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the LTX-2.5 diffusion (HQ) video decoder and its conversion key mapping.

Runs without a GPU and without natten, exercising the eager tiled-SDPA neighborhood-attention
fallback. When the full FastVideo dependency set is unavailable (e.g. no einops/cloudpickle),
the ``fastvideo`` package is stubbed so the decoder module loads standalone; the wrapper test
that needs the conv encoder from ``ltx2vae`` is skipped in that case.

Run directly (``python3 tests/local_tests/ltx2_5/test_ltx2_5_diffusion_decoder.py``) when pytest
is not installed; the __main__ guard executes every test function.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def _ensure_fastvideo_importable() -> None:
    """Import the real package, or stub it so submodules load without heavy dependencies."""
    try:
        import fastvideo  # noqa: F401
    except Exception:
        package = types.ModuleType("fastvideo")
        package.__path__ = [str(REPO_ROOT / "fastvideo")]
        sys.modules["fastvideo"] = package


_ensure_fastvideo_importable()
dd = importlib.import_module("fastvideo.models.vaes.ltx2_diffusion_decoder")

CONVERTER_PATH = REPO_ROOT / "scripts" / "checkpoint_conversion" / "convert_ltx2_weights.py"
SPEC = importlib.util.spec_from_file_location("convert_ltx2_weights_diffusion_test", CONVERTER_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


try:
    import pytest as _pytest

    SkipTest = _pytest.skip.Exception

    def _skip(message: str) -> None:
        _pytest.skip(message)
except ImportError:

    class SkipTest(Exception):  # type: ignore[no-redef]
        pass

    def _skip(message: str) -> None:
        raise SkipTest(message)


def _tiny_decoder(**overrides) -> "dd.LTX2DiffusionVideoDecoder":
    kwargs = dict(
        in_channels=8,
        out_channels=3,
        patch_size=2,
        head_dim=16,
        stage_channels=(64, 32, 16, 16, 16),
        stage_depths=(1, 1, 1, 1, 2),
        stage_kernels=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
        upsamples=(((1, 2, 2), 2), ((2, 1, 1), 2), ((2, 2, 2), 1), ((2, 2, 2), 1)),
        stage5_kernel=(3, 3, 3),
        t_emb_dim=8,
        default_num_inference_steps=2,
        timestep_scale_multiplier=1000.0,
        model_output_type="v",
    )
    kwargs.update(overrides)
    return dd.LTX2DiffusionVideoDecoder(**kwargs)


TINY_VAE_CONFIG = {
    "_class_name": "CausalDiffusionVAE",
    "vae": {
        "_class_name": "CausalDiffusionVAE",
        "latent_channels": 8,
        "model_output_type": "v",
        "encoder": {
            "dims": 3,
            "in_channels": 3,
            "out_channels": 8,
            "blocks": [["res_x", {"num_layers": 1}]],
            "patch_size": 2,
            "norm_layer": "pixel_norm",
            "latent_log_var": "uniform",
            "spatial_padding_mode": "zeros",
        },
        "decoder": {
            "in_channels": 8,
            "out_channels": 3,
            "patch_size": 2,
            "head_dim": 16,
            "stage_channels": [64, 32, 16, 16, 16],
            "stage_depths": [1, 1, 1, 1, 2],
            "stage_kernels": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            "upsamples": [[[1, 2, 2], 2], [[2, 1, 1], 2], [[2, 2, 2], 1], [[2, 2, 2], 1]],
            "stage5_kernel": [3, 3, 3],
            "t_emb_dim": 8,
            "default_num_inference_steps": 2,
            "timestep_scale_multiplier": 1000.0,
        },
    },
}


def test_eager_na3d_matches_naive_reference() -> None:
    """The eager fallback matches a brute-force NATTEN-style windowed attention."""
    torch.manual_seed(0)
    b, t, h, w, nh, hd = 1, 4, 5, 6, 2, 8
    kernel = (3, 3, 3)
    q = torch.randn(b, t, h, w, nh, hd) * hd**-0.5  # pre-scaled, scale=1.0 semantics
    k = torch.randn(b, t, h, w, nh, hd)
    v = torch.randn(b, t, h, w, nh, hd)

    def window(idx: int, length: int, kern: int) -> tuple[int, int]:
        kern = min(kern, length)
        start = min(max(idx - kern // 2, 0), length - kern)
        return start, start + kern

    expected = torch.empty_like(v)
    for ti in range(t):
        t0, t1 = window(ti, t, kernel[0])
        for hi in range(h):
            h0, h1 = window(hi, h, kernel[1])
            for wi in range(w):
                w0, w1 = window(wi, w, kernel[2])
                keys = k[:, t0:t1, h0:h1, w0:w1].reshape(b, -1, nh, hd)
                vals = v[:, t0:t1, h0:h1, w0:w1].reshape(b, -1, nh, hd)
                query = q[:, ti, hi, wi]  # (b, nh, hd)
                scores = torch.einsum("bnd,bknd->bnk", query, keys)
                attn = torch.softmax(scores, dim=-1)
                expected[:, ti, hi, wi] = torch.einsum("bnk,bknd->bnd", attn, vals)

    actual = dd.neighborhood_attention_3d(q, k, v, kernel)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5), "eager na3d deviates from the naive reference"


def test_tiny_decoder_forward_shape_v_two_steps() -> None:
    """Instantiate the tiny decoder and decode a small latent on CPU (2-step v-prediction)."""
    torch.manual_seed(0)
    decoder = _tiny_decoder().eval()
    assert decoder.time_scale == 8
    assert decoder.spatial_scale_h == 16 and decoder.spatial_scale_w == 16
    latent = torch.randn(1, 8, 3, 4, 4)
    with torch.no_grad():
        pixels = decoder.decode(latent, generator=torch.Generator().manual_seed(11))
    assert pixels.shape == (1, 3, 17, 64, 64), pixels.shape
    assert torch.isfinite(pixels).all()


def test_tiny_decoder_forward_shape_x0_single_step() -> None:
    """Single-step x0 decode (how LTX-2.5 ships) returns the model prediction directly."""
    torch.manual_seed(0)
    decoder = _tiny_decoder(model_output_type="x0", default_num_inference_steps=1).eval()
    latent = torch.randn(1, 8, 3, 4, 4)
    with torch.no_grad():
        pixels = decoder.decode(latent, generator=torch.Generator().manual_seed(11))
    assert pixels.shape == (1, 3, 17, 64, 64), pixels.shape
    assert torch.isfinite(pixels).all()


def test_decode_is_deterministic_per_seed() -> None:
    torch.manual_seed(0)
    decoder = _tiny_decoder().eval()
    latent = torch.randn(1, 8, 3, 4, 4)
    with torch.no_grad():
        first = decoder.decode(latent, generator=torch.Generator().manual_seed(7))
        second = decoder.decode(latent, generator=torch.Generator().manual_seed(7))
        other = decoder.decode(latent, generator=torch.Generator().manual_seed(8))
    assert torch.equal(first, second), "same seed must reproduce the decode exactly"
    assert not torch.equal(first, other), "different seeds must change the decode noise"


def test_tiled_decode_shape_matches_untiled() -> None:
    """Tiled decode covers the causal frame mapping and blends to the untiled output shape."""
    torch.manual_seed(0)
    decoder = _tiny_decoder(model_output_type="x0", default_num_inference_steps=1).eval()
    latent = torch.randn(1, 8, 5, 4, 6)
    decoder.enable_tiling(
        tile_sample_min_height=2048,
        tile_sample_stride_height=1024,
        tile_sample_min_width=48,
        tile_sample_stride_width=32,
        tile_sample_min_num_frames=16,
        tile_sample_stride_num_frames=8,
    )
    with torch.no_grad():
        pixels = decoder.decode(latent, generator=torch.Generator().manual_seed(3))
    expected_shape = (1, 3, (5 - 1) * 8 + 1, 4 * 16, 6 * 16)
    assert pixels.shape == expected_shape, pixels.shape
    assert torch.isfinite(pixels).all()


def test_min_latent_floor_pads_and_crops() -> None:
    """A single-latent-frame clip is edge-padded to the stage floor and cropped back."""
    torch.manual_seed(0)
    decoder = _tiny_decoder().eval()
    latent = torch.randn(1, 8, 1, 3, 3)  # below the (3, 3, 3) latent floor on every axis
    with torch.no_grad():
        pixels = decoder.decode(latent, generator=torch.Generator().manual_seed(5))
    assert pixels.shape == (1, 3, 1, 48, 48), pixels.shape
    assert torch.isfinite(pixels).all()


def test_configurator_reads_nested_checkpoint_config() -> None:
    """The metadata-shaped config (nested encoder/decoder) drives instantiation."""
    decoder = dd.LTX2DiffusionVideoDecoderConfigurator.from_config(TINY_VAE_CONFIG)
    assert isinstance(decoder, dd.LTX2DiffusionVideoDecoder)
    assert decoder.patch_size == 2
    assert decoder.context_channels == 16
    assert decoder.default_num_inference_steps == 2
    assert decoder.timestep_scale_multiplier == 1000.0
    assert decoder.model_output_type == "v"
    assert len(decoder.diff_blocks) == 2
    assert decoder.conv_in.in_features == 8 and decoder.conv_in.out_features == 64
    assert decoder.conv_in_x_t.in_features == 3 * 2 * 2

    # Configs that list one kernel per deterministic stage only (diffusers-style, 4 entries)
    # are padded with stage5_kernel; the trailing entry is never consumed.
    import copy

    four_kernel_config = copy.deepcopy(TINY_VAE_CONFIG)
    four_kernel_config["vae"]["decoder"]["stage_kernels"] = [[3, 3, 3]] * 4
    decoder = dd.LTX2DiffusionVideoDecoderConfigurator.from_config(four_kernel_config)
    assert len(decoder.det_stages) == 4 and len(decoder.diff_blocks) == 2


def _official_state_dict_from(decoder: "dd.LTX2DiffusionVideoDecoder") -> tuple[dict, dict]:
    """Reverse-map the native decoder state dict into a synthetic official checkpoint dict.

    Returns ``(official_sd, gates)``. Official keys carry the ``decoder.`` prefix, fused
    ``qkv.{weight,bias}``, ``t_embedder.mlp.{0,2}`` naming, gates on every diff block, a bundled
    ``coarse_*`` preview head, and top-level shared ``per_channel_statistics``.
    """
    native = decoder.state_dict()
    official: dict[str, torch.Tensor] = {}
    gates: dict[str, torch.Tensor] = {}

    fused: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in native.items():
        if key.startswith("per_channel_statistics."):
            official[key] = torch.randn_like(tensor)
            continue
        official_key = key.replace("t_embedder.timestep_embedder.linear_1.", "t_embedder.mlp.0.")
        official_key = official_key.replace("t_embedder.timestep_embedder.linear_2.", "t_embedder.mlp.2.")
        for role in ("to_q", "to_k", "to_v"):
            marker = f".qkv.{role}."
            if marker in official_key:
                leaf = official_key.rsplit(".", 1)[1]
                fused_key = "decoder." + official_key.split(marker)[0] + f".qkv.{leaf}"
                fused.setdefault(fused_key, {})[role] = torch.randn_like(tensor)
                break
        else:
            official["decoder." + official_key] = torch.randn_like(tensor)

    for fused_key, parts in fused.items():
        assert set(parts) == {"to_q", "to_k", "to_v"}, fused_key
        official[fused_key] = torch.cat([parts["to_q"], parts["to_k"], parts["to_v"]], dim=0)

    dim = decoder.diff_blocks[0].context_proj.out_features
    for block_index in range(len(decoder.diff_blocks)):
        for suffix in ("gate_msa", "gate_mlp", "gate_ctx"):
            gate_key = f"decoder.diff_blocks.{block_index}.{suffix}"
            gate = torch.rand(dim) + 0.5
            official[gate_key] = gate
            gates[gate_key] = gate

    official["decoder.coarse_head.weight"] = torch.randn(4, 4)
    official["decoder.diff_blocks.0.coarse_proj.weight"] = torch.randn(4, 4)
    # Training-only type embedding shipped by the real 2.5 HQ checkpoint; no released
    # decoder consumes it and the converter drops it explicitly.
    official["decoder.type_emb"] = torch.randn(8)
    return official, gates


def test_conversion_key_names_and_shapes_round_trip() -> None:
    """convert_diffusion_vae_weights maps official keys 1:1 onto the native state dict."""
    torch.manual_seed(0)
    decoder = _tiny_decoder()
    official, gates = _official_state_dict_from(decoder)

    converted = converter.convert_diffusion_vae_weights(official)

    # Only decoder.* (and the mirrored statistics) may come out for a decoder-only input.
    decoder_sd = {
        key[len("decoder."):]: value for key, value in converted.items() if key.startswith("decoder.")
    }
    encoder_stats = [key for key in converted if key.startswith("encoder.")]
    assert encoder_stats == [
        "encoder.per_channel_statistics.std-of-means",
        "encoder.per_channel_statistics.mean-of-means",
    ], encoder_stats

    native = decoder.state_dict()
    assert set(decoder_sd) == set(native), (
        f"missing={sorted(set(native) - set(decoder_sd))} unexpected={sorted(set(decoder_sd) - set(native))}")
    for key, tensor in decoder_sd.items():
        assert tensor.shape == native[key].shape, f"{key}: {tensor.shape} vs {native[key].shape}"

    # Strict load must succeed (shapes and names both line up).
    decoder.load_state_dict(decoder_sd, strict=True)

    # Gate folding: context_proj / attn.proj / mlp.w_down carry the folded gate.
    for name, gate_suffix in (
        ("diff_blocks.0.context_proj.weight", "gate_ctx"),
        ("diff_blocks.0.attn.proj.weight", "gate_msa"),
        ("diff_blocks.0.mlp.w_down.weight", "gate_mlp"),
        ("diff_blocks.0.context_proj.bias", "gate_ctx"),
    ):
        gate = gates[f"decoder.diff_blocks.0.{gate_suffix}"].to(torch.float32)
        raw = official[f"decoder.{name}"].to(torch.float32)
        folded = gate.unsqueeze(1) * raw if raw.ndim == 2 else gate * raw
        assert torch.allclose(decoder_sd[name], folded), f"gate fold mismatch for {name}"

    # Ungated (deterministic-stage) projections pass through untouched.
    assert torch.equal(decoder_sd["det_stages.0.0.attn.proj.weight"],
                       official["decoder.det_stages.0.0.attn.proj.weight"])

    # QKV split: thirds of the fused tensor land on to_q / to_k / to_v.
    fused = official["decoder.diff_blocks.0.attn.qkv.weight"]
    chunk = fused.shape[0] // 3
    assert torch.equal(decoder_sd["diff_blocks.0.attn.qkv.to_q.weight"], fused[:chunk])
    assert torch.equal(decoder_sd["diff_blocks.0.attn.qkv.to_k.weight"], fused[chunk:2 * chunk])
    assert torch.equal(decoder_sd["diff_blocks.0.attn.qkv.to_v.weight"], fused[2 * chunk:])

    # Bundled preview heads, gates, and documented skipped keys are dropped.
    assert not any("coarse" in key for key in converted)
    assert not any(key.endswith(("gate_msa", "gate_mlp", "gate_ctx")) for key in converted)
    assert not any(key.endswith("type_emb") for key in converted)


def test_cuda_backend_requires_natten() -> None:
    """On CUDA without natten the resolver raises instead of silently degrading;
    explicit env overrides and non-CUDA devices keep their fallbacks."""
    saved_available = dd._NATTEN_AVAILABLE
    saved_env = os.environ.pop("FASTVIDEO_LTX2_NA_BACKEND", None)
    try:
        dd._NATTEN_AVAILABLE = False
        try:
            dd._resolve_na3d_backend("cuda")
        except ImportError as exc:
            assert "natten" in str(exc)
            assert "FASTVIDEO_LTX2_NA_BACKEND" in str(exc)
        else:
            raise AssertionError("CUDA without natten must raise ImportError")
        # CPU keeps the eager fallback (this is what lets these tests run).
        assert dd._resolve_na3d_backend("cpu") == "eager"
        # An explicit eager override is honored even on CUDA.
        os.environ["FASTVIDEO_LTX2_NA_BACKEND"] = "eager"
        assert dd._resolve_na3d_backend("cuda") == "eager"
        # natten present again: CUDA resolves to natten.
        dd._NATTEN_AVAILABLE = True
        os.environ.pop("FASTVIDEO_LTX2_NA_BACKEND", None)
        assert dd._resolve_na3d_backend("cuda") == "natten"
    finally:
        dd._NATTEN_AVAILABLE = saved_available
        if saved_env is None:
            os.environ.pop("FASTVIDEO_LTX2_NA_BACKEND", None)
        else:
            os.environ["FASTVIDEO_LTX2_NA_BACKEND"] = saved_env


def test_wrapper_encode_decode_with_conv_encoder() -> None:
    """Full CausalDiffusionVAE wrapper (needs einops etc. for the conv encoder)."""
    try:
        importlib.import_module("fastvideo.models.vaes.ltx2vae")
    except Exception as exc:
        _skip(f"full FastVideo deps unavailable ({exc})")
        return

    torch.manual_seed(0)
    vae = dd.LTX2CausalDiffusionVAE(TINY_VAE_CONFIG).eval()
    latent = torch.randn(1, 8, 3, 4, 4)
    vae.set_decode_generator(torch.Generator().manual_seed(2))
    with torch.no_grad():
        first = vae.decode(latent)
    vae.set_decode_generator(torch.Generator().manual_seed(2))
    with torch.no_grad():
        second = vae.decode(latent)
    assert first.shape == (1, 3, 17, 64, 64), first.shape
    assert torch.equal(first, second)

    # Encoder half: 9 frames (1 + 8x), tiny spatial size; latent means come back normalized.
    video = torch.randn(1, 3, 9, 8, 8)
    with torch.no_grad():
        posterior = vae.encode(video)
    assert posterior.sample().shape[1] == 8


ALL_TESTS = [
    test_eager_na3d_matches_naive_reference,
    test_tiny_decoder_forward_shape_v_two_steps,
    test_tiny_decoder_forward_shape_x0_single_step,
    test_decode_is_deterministic_per_seed,
    test_tiled_decode_shape_matches_untiled,
    test_min_latent_floor_pads_and_crops,
    test_configurator_reads_nested_checkpoint_config,
    test_conversion_key_names_and_shapes_round_trip,
    test_cuda_backend_requires_natten,
    test_wrapper_encode_decode_with_conv_encoder,
]

if __name__ == "__main__":
    failures = 0
    for test in ALL_TESTS:
        name = test.__name__
        try:
            test()
        except SkipTest as exc:
            print(f"SKIP {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            import traceback

            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    if failures:
        sys.exit(1)
    print("all tests passed")
