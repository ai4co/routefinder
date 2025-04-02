import math

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.attention import PointerAttention

# from rl4co.models.nn.env_embeddings import env_context_embedding, env_dynamic_embedding
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.models.zoo.am.decoder import AttentionModelDecoder, PrecomputedCache
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict
from torch.nn.functional import scaled_dot_product_attention

from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbedding

from .moe import MoE

log = get_pylogger(__name__)


class PointerAttentionMoE(PointerAttention):
    """
    MVMoE replaces the project_out to obtain the glimpse
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mask_inner: bool = True,
        out_bias: bool = False,
        check_nan: bool = True,
        sdpa_fn=None,
        num_experts=4,
        routing_method="input_choice",
        routing_level="node",
        topk=2,
        hierarchical_gating=False,
        temperature=1.0,
    ):
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.mask_inner = mask_inner
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention
        self.check_nan = check_nan

        self.hierarchical_gating = hierarchical_gating
        self.temperature = temperature

        if num_experts > 0:
            print("Using MoE with {} experts in decoder".format(num_experts))

            self.project_out = MoE(
                input_size=embed_dim,
                output_size=embed_dim,
                num_experts=num_experts,
                k=topk,
                T=1.0,
                noisy_gating=True,
                routing_level=routing_level,
                routing_method=routing_method,
                moe_model="Linear",
            )
            if self.hierarchical_gating:
                print("Hierarchical gating in PointerAttentionMoE initializing")
                self.dense_or_moe = nn.Linear(embed_dim, 2, bias=False)
                self.project_out_dense = nn.Linear(embed_dim, embed_dim, bias=out_bias)

        else:
            self.project_out = nn.Linear(embed_dim, embed_dim, bias=out_bias)
        self.moe_loss = 0  # init to 0

    def forward(self, query, key, value, logit_key, attn_mask=None):
        # Compute inner multi-head attention with no projections.
        heads = self._inner_mha(query, key, value, attn_mask)

        # MoE loss
        moe_loss = 0

        # Hierarchical gating MoE
        if self.hierarchical_gating:
            # we do this only at the "second" step, which is depot -> pomo -> first select
            # in our case, it means that we visited at most 1 due to pomo only. Depot is not considered for us
            # get largest number of elements in attention mask
            # step is 2 if the the action mask is >= action space - 1
            num_nodes = key.shape[-2]
            num_available_nodes = attn_mask.sum(-1)
            if (num_available_nodes >= num_nodes - 1).any():
                head_reduction = (
                    heads.mean(0).mean(0) if heads.dim() == 3 else heads.mean(0)
                )
                self.probs = F.softmax(
                    self.dense_or_moe(head_reduction) / self.temperature,
                    dim=-1,
                )  # [1, 2]
            selected = self.probs.multinomial(1).squeeze(0)
            if selected.item() == 1:
                mh_atten_out, moe_loss = self.project_out(heads)
            else:
                mh_atten_out = self.project_out_dense(heads)
            mh_atten_out = mh_atten_out * self.probs.squeeze(0)[selected]

        else:
            # Normal MoE
            if isinstance(self.project_out, MoE):
                mh_atten_out, moe_loss = self.project_out(heads)
            else:
                mh_atten_out = self.project_out(heads)

        glimpse = mh_atten_out

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = (torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1))).squeeze(
            -2
        ) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        # MoE loss is saved in the module
        # Note that this is re-initialized by the `pre_decoder_hook` of the decoder
        self.moe_loss += moe_loss
        return logits


class MVMoEDecoder(AttentionModelDecoder):
    """
    TODO
    Note that the real change is the pointer attention
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "tsp",
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        sdpa_fn: callable = None,
        num_experts=4,
        routing_method="input_choice",
        routing_level="node",
        topk=2,
        moe_loc=["enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec"],
        hierarchical_gating=False,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0

        if context_embedding is None:
            log.info("Using default MTVRPContextEmbedding")
            context_embedding = MTVRPContextEmbedding(embed_dim)
        self.context_embedding = context_embedding

        if dynamic_embedding is None:
            log.info("Using default StaticEmbedding")
            self.dynamic_embedding = StaticEmbedding()
        self.is_dynamic_embedding = (
            False if isinstance(self.dynamic_embedding, StaticEmbedding) else True
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embed_dim
        self.project_node_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=linear_bias
        )
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context

        if "dec" in moe_loc:
            num_experts_dec = num_experts
        else:
            num_experts_dec = 0

        self.pointer = PointerAttentionMoE(
            embed_dim,
            num_heads,
            mask_inner=mask_inner,
            out_bias=out_bias_pointer_attn,
            check_nan=check_nan,
            sdpa_fn=sdpa_fn,
            num_experts=num_experts_dec,
            routing_method=routing_method,
            routing_level=routing_level,
            topk=topk,
            hierarchical_gating=hierarchical_gating,
        )

    def pre_decoder_hook(
        self, td, env, embeddings, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, PrecomputedCache]:
        """Precompute the embeddings cache before the decoder is called"""
        # Re-intialize the moe loss in the pointer
        self.pointer.moe_loss = 0
        return td, env, self._precompute_cache(embeddings, num_starts=num_starts)
