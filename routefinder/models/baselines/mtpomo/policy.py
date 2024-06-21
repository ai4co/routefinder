from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.utils.pylogger import get_pylogger
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding

from routefinder.models.env_embeddings.mtvrp import MTVRPInitEmbedding, MTVRPContextEmbedding

log = get_pylogger(__name__)


class MTPOMOPolicy(AttentionModelPolicy):
    """
    Main MTPOMO policy based on Attention Model with POMO settings.
    The main difference are the embeddings used for the new environment.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        env_name: str = "mtvrp",
        use_graph_context: bool = False,
        init_embedding: MTVRPInitEmbedding = None,
        context_embedding: MTVRPContextEmbedding = None,
        **kwargs,
    ):
        
        if init_embedding is None:
            init_embedding = MTVRPInitEmbedding(embed_dim=embed_dim)
        else:
            log.warning("Using custom init_embedding")
        
        if context_embedding is None:
            context_embedding = MTVRPContextEmbedding(embed_dim=embed_dim)
        
        # mtvrp does not use dynamic embedding (i.e. only modifies the query, not key or value)
        dynamic_embedding = StaticEmbedding()
        
        super(MTPOMOPolicy, self).__init__(
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            init_embedding=init_embedding,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            **kwargs,
        )