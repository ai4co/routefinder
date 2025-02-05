import torch.nn as nn

from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbeddingFull
from routefinder.models.env_embeddings.mtvrp.init import MTVRPInitEmbeddingFull

from .utils import freeze_backbone


def model_from_scratch(
    model,
    init_embedding_cls=MTVRPInitEmbeddingFull,
    context_embedding_cls=MTVRPContextEmbeddingFull,
):
    """Reinitializes from scratch with new model and new embeddings"""

    print("Reinitializing full model from scratch")
    embed_dim = model.policy.encoder.init_embedding.embed_dim

    def reset_weights(m):
        if isinstance(m, nn.Module) and hasattr(m, "reset_parameters"):
            m.reset_parameters()

    model.policy.apply(reset_weights)
    model.policy.encoder.init_embedding = init_embedding_cls(embed_dim=embed_dim)
    model.policy.decoder.context_embedding = context_embedding_cls(embed_dim=embed_dim)

    # Add `_multistart` to decode type for train, val and test in policy
    for phase in ["train", "val", "test"]:
        model.set_decode_type_multistart(phase)

    return model


def adapter_layers(
    model,
    init_embedding_cls=MTVRPInitEmbeddingFull,
    context_embedding_cls=MTVRPContextEmbeddingFull,
    adapter_only=False,
):
    """Adapter Layers (AL) from Lin et al., 2024.
    Only initializes new adapter layers (embeddings), but keeps the model parameters the same.
    """
    print("Using Adapter Layers (AL)")

    embed_dim = model.policy.encoder.init_embedding.embed_dim
    policy = model.policy

    new_init_embedding = init_embedding_cls(embed_dim=embed_dim)
    new_context_embedding = context_embedding_cls(embed_dim=embed_dim)

    policy.encoder.init_embedding = new_init_embedding
    policy.decoder.context_embedding = new_context_embedding

    # If not full, then we freeze the backbone
    if adapter_only:
        policy = freeze_backbone(policy)

    model.policy = policy
    return model
