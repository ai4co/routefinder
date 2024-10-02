from typing import Tuple, Union

import torch.nn as nn

from rl4co.utils.pylogger import get_pylogger
from torch import Tensor

from routefinder.models.env_embeddings.mtvrp import MTVRPInitEmbeddingRouteFinder
from routefinder.models.nn.transformer import Normalization, TransformerBlock

log = get_pylogger(__name__)


class RouteFinderEncoder(nn.Module):
    """
    Encoder for RouteFinder model based on the Transformer Architecture.
    Here we include additional embedding from raw to embedding space, as
    well as more modern architecture options compared to the usual Attention Models
    based on POMO (including multi-task VRP ones).
    """

    def __init__(
        self,
        init_embedding: nn.Module = None,
        num_heads: int = 8,
        embed_dim: int = 128,
        num_layers: int = 6,
        feedforward_hidden: int = 512,
        normalization: str = "instance",
        use_prenorm: bool = False,
        use_post_layers_norm: bool = False,
        parallel_gated_kwargs: dict = None,
        **transformer_kwargs,
    ):
        super(RouteFinderEncoder, self).__init__()

        if init_embedding is None:
            init_embedding = MTVRPInitEmbeddingRouteFinder(embed_dim=embed_dim)
        else:
            log.warning("Using custom init_embedding")
        self.init_embedding = init_embedding

        self.layers = nn.Sequential(
            *(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    use_prenorm=use_prenorm,
                    feedforward_hidden=feedforward_hidden,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )

        self.post_layers_norm = (
            Normalization(embed_dim, normalization) if use_post_layers_norm else None
        )

    def forward(
        self, td: Tensor, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:

        # Transfer to embedding space
        init_h = self.init_embedding(td)  # [B, N, H]

        # Process embedding
        h = init_h
        for layer in self.layers:
            h = layer(h, mask)

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.post_layers_norm is not None:
            h = self.post_layers_norm(h)

        # Return latent representation
        return h, init_h  # [B, N, H]
