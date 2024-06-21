import torch
import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.attention import MultiHeadAttention
from rl4co.models.nn.graph.attnnet import GraphAttentionNetwork
from rl4co.models.nn.ops import Normalization
from rl4co.models.zoo.am.encoder import AttentionModelEncoder
from rl4co.utils.pylogger import get_pylogger
from torch import Tensor

from routefinder.models.env_embeddings.mtvrp.init import MTVRPInitEmbedding

from .moe import MoE

log = get_pylogger(__name__)


class MVMoEInitEmbedding(MTVRPInitEmbedding):
    def __init__(
        self,
        embed_dim=128,
        num_experts=4,
        routing_method="input_choice",
        routing_level="node",
        topk=2,
        bias=False,
        **kw,
    ):  # node: linear bias should be false in order not to influence the embedding if
        super(MVMoEInitEmbedding, self).__init__(embed_dim, bias, **kw)

        # If MoE is provided, we re-initialize the projections with MoE
        if num_experts > 0:
            print("MoE in init embedding initializing")
            self.project_global_feats = MoE(
                input_size=2,
                output_size=embed_dim,
                num_experts=num_experts,
                k=topk,
                T=1.0,
                noisy_gating=True,
                routing_level=routing_level,
                routing_method=routing_method,
                moe_model="Linear",
            )
            self.project_customers_feats = MoE(
                input_size=7,
                output_size=embed_dim,
                num_experts=num_experts,
                k=topk,
                T=1.0,
                noisy_gating=True,
                routing_level=routing_level,
                routing_method=routing_method,
                moe_model="Linear",
            )

    def forward(self, td):
        # Global (batch, 1, 2) -> (batch, 1, embed_dim)
        global_feats = td["locs"][:, :1, :]

        # Customers (batch, N, 5) -> (batch, N, embed_dim)
        # note that these feats include the depot (but unused) so we exclude the first node
        cust_feats = torch.cat(
            (
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
                td["locs"][:, 1:, :],
            ),
            -1,
        )

        # If some features are infinity (e.g. distance limit is inf because of no limit), replace with 0 so that it does not affect the embedding
        global_feats = torch.nan_to_num(global_feats, nan=0.0, posinf=0.0, neginf=0.0)
        cust_feats = torch.nan_to_num(cust_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # MoE loss is 0 if layer is not MoE
        moe_loss_global, moe_loss_cust = 0, 0
        if isinstance(self.project_global_feats, MoE):
            global_embeds, moe_loss_global = self.project_global_feats(global_feats)
        else:
            global_embeds = self.project_global_feats(global_feats)
        if isinstance(self.project_customers_feats, MoE):
            cust_embeds, moe_loss_cust = self.project_customers_feats(cust_feats)
        else:
            cust_embeds = self.project_customers_feats(cust_feats)
        self.moe_loss = moe_loss_global + moe_loss_cust
        return torch.cat((global_embeds, cust_embeds), -2)


class MultiHeadAttentionLayerMoE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        feedforward_hidden: int = 512,
        normalization="instance",
        sdpa_fn=None,
        num_experts=4,
        routing_method="input_choice",
        routing_level="node",
        topk=2,
    ):
        super(MultiHeadAttentionLayerMoE, self).__init__()

        if num_experts > 0:
            print("MoE in MultiHeadAttentionLayer initializing")
            dense_net = MoE(
                input_size=embed_dim,
                output_size=embed_dim,
                num_experts=num_experts,
                hidden_size=feedforward_hidden,
                k=topk,
                T=1.0,
                noisy_gating=True,
                routing_level=routing_level,
                routing_method=routing_method,
                moe_model="MLP",
            )
        else:
            dense_net = nn.Sequential(
                nn.Linear(embed_dim, feedforward_hidden),
                nn.ReLU(),
                nn.Linear(feedforward_hidden, embed_dim),
            )

        self.mha = MultiHeadAttention(embed_dim, num_heads, sdpa_fn=sdpa_fn)
        self.norm1 = Normalization(embed_dim, normalization)
        self.dense = dense_net
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, x: Tensor) -> Tensor:
        out_mha = self.mha(x)
        h = out_mha + x  # skip connection
        h = self.norm1(h)
        moe_loss = 0
        if isinstance(self.dense, MoE):
            out_dense, moe_loss = self.dense(h)
        else:
            out_dense = self.dense(h)
        # save moe loss
        self.moe_loss = moe_loss
        h = out_dense + h  # skip connection
        h = self.norm2(h)
        return h


class GraphAttentionNetworkMVMoE(GraphAttentionNetwork):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        num_layers: int,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        sdpa_fn=None,
        moe_loc=["enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec"],
        num_experts=4,
        routing_method="input_choice",
        routing_level="node",
        topk=2,
    ):
        nn.Module.__init__(self)

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayerMoE(
                    embed_dim,
                    num_heads,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    sdpa_fn=sdpa_fn,
                    num_experts=0 if f"enc{i}" not in moe_loc else num_experts,
                    routing_method=routing_method,
                    routing_level=routing_level,
                    topk=topk,
                )
                for i in range(num_layers)
            )
        )

    def forward(self, x: Tensor, mask=None) -> Tensor:
        """Forward pass of the encoder

        Args:
            x: [batch_size, graph_size, embed_dim] initial embeddings to process
            mask: [batch_size, graph_size, graph_size] mask for the input embeddings. Unused for now.
        """
        assert mask is None, "Mask not yet supported!"
        h = self.layers(x)
        return h


class MVMoEEncoder(AttentionModelEncoder):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        env_name="mtvrp",
        sdpa_fn=None,
        init_embedding=None,
        num_experts=4,
        routing_method="input_choice",
        routing_level="node",
        topk=2,
        moe_loc=["enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec"],
        **unused,
    ):
        # super(MVMoEEncoder, self).__init__()
        nn.Module.__init__(self)

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        assert self.env_name == "mtvrp", "Only mtvrp is supported for MVMoE"

        # assert init_embedding is None, "init embedding is manually set in MVMoE"

        # Initialize raw features only if provided
        if "raw" in moe_loc:
            num_experts_init = num_experts
        else:
            num_experts_init = 0

        if not init_embedding:
            init_embedding = MVMoEInitEmbedding(
                embed_dim,
                num_experts=num_experts_init,
                routing_method=routing_method,
                routing_level=routing_level,
                topk=topk,
            )
        else:
            if num_experts_init > 0:
                log.warning("MoE requested for init embedding but already provided")
        self.init_embedding = init_embedding

        self.net = GraphAttentionNetworkMVMoE(
            num_heads,
            embed_dim,
            num_layers,
            normalization,
            feedforward_hidden,
            sdpa_fn=sdpa_fn,
            moe_loc=moe_loc,
            num_experts=num_experts,
            routing_method=routing_method,
            routing_level=routing_level,
            topk=topk,
        )
