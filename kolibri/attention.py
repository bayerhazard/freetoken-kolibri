from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.attention import AttentionSpec
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, LinearOProj, LinearQKVMerged, RMSNorm
from freetoken.layers.rotary import get_rope
from freetoken.models.config import SWAAttentionGroupConfig
from freetoken.utils import div_even, nvtx_annotate

if TYPE_CHECKING:
    import torch

    from freetoken.models.config import ModelConfig


class Kolibri1Attention(BaseOP):
    """GQA with per-head q/k RMSNorm.

    Sliding-attention layers apply RoPE; full-attention layers are RNoPE
    (no positional encoding at all).
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.layer_id = layer_id
        group = config.attention_group_for_layer(layer_id)
        self.is_swa = isinstance(group, SWAAttentionGroupConfig)

        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.head_dim = group.head_dim
        self.qo_attn_dim = self.num_qo_heads * self.head_dim
        self.kv_attn_dim = self.num_kv_heads * self.head_dim
        self.sliding_window = group.sliding_window if self.is_swa else None
        self.sm_scale = config.attn_sm_scale

        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=self.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=config.has_attn_bias,
            quant_config=config.quant,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = LinearOProj(
            input_size=config.num_qo_heads * self.head_dim,
            output_size=config.hidden_size,
            has_bias=config.has_attn_bias,
            quant_config=config.quant,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if self.is_swa:
            rotary_config = group.rotary_config
            self.rotary = get_rope(
                head_dim=self.head_dim,
                rotary_dim=rotary_config.rotary_dim,
                max_position=rotary_config.max_position,
                base=rotary_config.base,
                rope_scaling=(
                    tuple(rotary_config.scaling.items())
                    if rotary_config.scaling
                    else None
                ),
            )
        else:
            self.rotary = None

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        qkv = self.qkv_proj.forward(x)
        q, k, v = qkv.split(
            [self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1
        )
        del x, qkv

        self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        if self.rotary is not None:
            q, k = self.rotary.forward(ctx.batch.get_attn_positions(), q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            self.layer_id,
            ctx.batch,
            attn_spec=AttentionSpec(
                sliding_window=self.sliding_window,
                sm_scale=self.sm_scale,
            ),
        )
        return self.o_proj.forward(o.view(-1, self.qo_attn_dim))


__all__ = ["Kolibri1Attention"]
