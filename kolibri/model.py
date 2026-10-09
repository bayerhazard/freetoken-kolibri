from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Kolibri1Attention
from .moe import Kolibri1SparseBlock

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Kolibri1DecoderLayer(BaseOP):
    """Every layer is MoE; attention and MoE are wrapped in sandwich norms
    (a post-norm on each sublayer output before the residual add)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.self_attn = Kolibri1Attention(
            config, layer_id, prefix=f"{prefix}.self_attn"
        )
        self.mlp = Kolibri1SparseBlock(config, layer_id, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_ffn_norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x = self.post_attn_norm.forward(x)[0]
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        x = self.post_ffn_norm.forward(x)[0]
        return x, residual


class Kolibri1Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Kolibri1DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Kolibri1ForCausalLM(BaseLLMModel):
    model_cls = Kolibri1Model

    def __init__(self, config: ModelConfig):
        self.model = self.model_cls(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Kolibri1ForCausalLM"]
