from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
    detect_expert_quant,
)


def _layer_types(hf_config: Any) -> list[str]:
    layer_types = list(getattr(hf_config, "layer_types", ()) or ())
    if not layer_types:
        layer_types = ["full_attention"] * hf_config.num_hidden_layers
    return layer_types


def parse_config(hf_config: Any) -> ModelConfig:
    num_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
    head_dim = getattr(hf_config, "head_dim", None) or (
        hf_config.hidden_size // hf_config.num_attention_heads
    )
    rope_theta = getattr(hf_config, "rope_theta", 10000.0)
    rope_scaling = getattr(hf_config, "rope_scaling", None)

    rotary_config = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=head_dim,
        max_position=hf_config.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
    )

    layer_types = _layer_types(hf_config)
    swa_ids = tuple(i for i, t in enumerate(layer_types) if t == "sliding_attention")
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    if len(swa_ids) + len(full_ids) != hf_config.num_hidden_layers:
        raise ValueError(
            "kolibri1 layer_types must be sliding_attention/full_attention only: "
            f"{sorted(set(layer_types))}"
        )
    sliding_window = int(getattr(hf_config, "sliding_window", 0) or 0)
    if swa_ids and sliding_window <= 0:
        raise ValueError("kolibri1 sliding-attention layers need a positive sliding_window")

    # Full-attention layers are RNoPE (no positional encoding); their group's
    # rotary_config is unused by Kolibri1Attention. The SWA group carries the rope.
    attention_groups: tuple = ()
    if swa_ids:
        attention_groups += (
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=swa_ids,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=rotary_config,
                sliding_window=sliding_window,
            ),
        )
    if full_ids:
        attention_groups += (
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_ids,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=rotary_config,
            ),
        )

    return ModelConfig(
        num_layers=hf_config.num_hidden_layers,
        num_qo_heads=hf_config.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        intermediate_size=getattr(hf_config, "intermediate_size", 0) or 0,
        rms_norm_eps=hf_config.rms_norm_eps,
        rotary_config=rotary_config,
        hidden_act=hf_config.hidden_act,
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        num_experts=getattr(
            hf_config, "num_local_experts", getattr(hf_config, "num_experts", 0)
        ),
        num_experts_per_tok=getattr(hf_config, "num_experts_per_tok", 0),
        moe_intermediate_size=getattr(hf_config, "moe_intermediate_size", 0),
        norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", False)),
        model_type=getattr(hf_config, "model_type", "kolibri1"),
        architectures=getattr(hf_config, "architectures", ["Kolibri1ForCausalLM"]),
        moe_enabled=True,
        shared_expert_intermediate_size=getattr(
            hf_config, "shared_expert_intermediate_size", 0
        ),
        use_qk_norm=True,
        n_shared_experts=1,
        attention_groups=attention_groups,
        expert_quant=detect_expert_quant(hf_config),
    )


__all__ = ["parse_config"]
