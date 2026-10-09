from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer
from freetoken.models.glm4_moe.mlp import GlmGatedMLP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

TopK = Tuple[torch.Tensor, torch.Tensor]


class Kolibri1SparseBlock(BaseOP):
    """Kolibri 1 sparse MoE block: top-k routed experts + one always-on shared expert.

    Routing matches HF ``Kolibri1``: fp32 router logits, select top-k on
    ``logits + e_score_correction_bias``, weight by the unbiased ``sigmoid(logits)``,
    no renormalization (``norm_topk_prob`` is False). The shared expert is ungated and
    added to the routed output.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = LinearReplicated(
            config.hidden_size, config.num_experts, has_bias=False
        )
        # Keep the selection bias in fp32: rounding can change the selected experts.
        self.e_score_correction_bias = torch.empty(
            config.num_experts, dtype=torch.float32
        )

        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        self.shared_experts = GlmGatedMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            quant_config=config.quant,
            prefix=f"{prefix}.shared_experts",
        )

    def _route(self, hidden_states: torch.Tensor) -> TopK:
        logits = F.linear(hidden_states.float(), self.gate.weight.float())
        # Selection on raw logits + correction bias; weights from unbiased sigmoid(logits).
        scores_for_choice = logits + self.e_score_correction_bias.float()
        _, topk_ids = torch.topk(scores_for_choice, self.top_k, dim=-1)
        topk_weights = torch.sigmoid(logits.gather(-1, topk_ids))
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_weights.to(torch.float32).contiguous(), topk_ids.to(torch.int32).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_ids = self._route(hidden_states)
        # The expert kernel may overwrite hidden_states; the shared expert needs the input.
        shared = self.shared_experts.forward(hidden_states)
        out = self.experts.routed_forward(hidden_states, topk_weights, topk_ids)
        return (out + shared).view(num_tokens, hidden_dim)


__all__ = ["Kolibri1SparseBlock"]
