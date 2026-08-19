# SPDX-License-Identifier: Apache-2.0
"""Architecture configs of the reference FastVideo/LTX-2.3-Distilled-Diffusers repo.

Embedded verbatim (fetched from that repo) so verify_ltx23_conversion.py can
check a converted repo offline. These are ARCHITECTURE facts, not paths: a
converted LTX-2.3 repo whose configs differ here instantiates a differently
shaped model than the checkpoint was trained for, which degrades output
without necessarily failing to load.
"""

TRANSFORMER_REFERENCE = {
    'apply_gated_attention': True,
    'attention_head_dim': 128,
    'attention_type': 'default',
    'audio_attention_head_dim': 64,
    'audio_connector_attention_head_dim': 64,
    'audio_connector_num_attention_heads': 32,
    'audio_cross_attention_dim': 2048,
    'audio_num_attention_heads': 32,
    'audio_out_channels': 128,
    'audio_positional_embedding_max_pos': [20],
    'av_ca_timestep_scale_multiplier': 1000.0,
    'caption_channels': 3840,
    'caption_proj_before_connector': True,
    'caption_proj_input_norm': False,
    'caption_projection_first_linear': False,
    'caption_projection_second_linear': False,
    'connector_attention_head_dim': 128,
    'connector_num_attention_heads': 32,
    'connector_num_layers': 8,
    'cross_attention_adaln': True,
    'cross_attention_dim': 4096,
    'double_precision_rope': True,
    'in_channels': 128,
    'norm_eps': 1e-06,
    'num_attention_heads': 32,
    'num_layers': 48,
    'out_channels': 128,
    'positional_embedding_max_pos': [20, 2048, 2048],
    'positional_embedding_theta': 10000.0,
    'rope_type': 'split',
    'timestep_scale_multiplier': 1000,
    'use_middle_indices_grid': True,
}

TEXT_ENCODER_REFERENCE = {
    'audio_connector_attention_head_dim': 64,
    'audio_connector_num_attention_heads': 32,
    'audio_connector_num_layers': 8,
    'audio_feature_extractor_out_features': 2048,
    'caption_proj_before_connector': True,
    'caption_proj_input_norm': False,
    'caption_projection_first_linear': False,
    'caption_projection_second_linear': False,
    'connector_apply_gated_attention': True,
    'connector_attention_head_dim': 128,
    'connector_double_precision_rope': True,
    'connector_num_attention_heads': 32,
    'connector_num_layers': 8,
    'connector_num_learnable_registers': 128,
    'connector_positional_embedding_max_pos': [4096],
    'connector_positional_embedding_theta': 10000.0,
    'connector_rope_type': 'split',
    'eos_token_id': 2,
    'feature_extractor_in_features': 188160,
    'feature_extractor_out_features': 3840,
    'gemma_dtype': 'bfloat16',
    'hidden_size': 3840,
    'num_attention_heads': 30,
    'num_hidden_layers': 48,
    'pad_token_id': 0,
    'padding_side': 'left',
    'text_len': 1024,
    'video_feature_extractor_out_features': 4096,
}
