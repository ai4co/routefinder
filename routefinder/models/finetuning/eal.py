from copy import deepcopy

import torch

from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbeddingFull
from routefinder.models.env_embeddings.mtvrp.init import MTVRPInitEmbeddingFull

from .utils import freeze_backbone


def efficient_adapter_layers(
    model,
    init_embedding_cls=MTVRPInitEmbeddingFull,
    context_embedding_cls=MTVRPContextEmbeddingFull,
    init_embedding_num_new_feats=1,
    context_embedding_num_new_feats=3,
    adapter_only=False,
):
    """Efficient Active Layers.
    Keep the model the same, replace the embeddings with new zero-padded embeddings for unseen features.

    Args:
        model: the model to be adapted
        init_embedding_cls: the new init embedding class
        context_embedding_cls: the new context embedding class
        init_embedding_num_new_feats: the number of new features to be added to the init embedding (initiate with zeros)
        context_embedding_num_new_feats: the number of new features to be added to the context embedding (initiate with zeros)
        adapter_only: if True, only the new embeddings are trained, otherwise the whole model is trained.
    """

    print("Using Efficient Adapter Layers (EAL)")

    policy = model.policy
    embed_dim = policy.decoder.context_embedding.embed_dim

    policy_new = deepcopy(policy)

    init_embedding_new_feat = init_embedding_cls(embed_dim=embed_dim)
    context_embedding_new_feat = context_embedding_cls(embed_dim=embed_dim)

    policy_new.encoder.init_embedding = init_embedding_new_feat
    policy_new.decoder.context_embedding = context_embedding_new_feat

    policy_new = policy_new.to(next(policy.parameters()).device)

    # Init Embedding with EAL (encoder)
    init_embedding_old = deepcopy(policy.encoder.init_embedding)
    # use previous k weights and pad l new number of weights with zeros
    proj_glob_params_old = init_embedding_old.project_global_feats.weight.data
    proj_glob_params_new = torch.cat(
        [
            proj_glob_params_old,
            torch.zeros(
                (proj_glob_params_old.shape[0], init_embedding_num_new_feats),
                device=proj_glob_params_old.device,
            ),
        ],
        dim=-1,
    )
    init_embed_new = init_embedding_cls(embed_dim=embed_dim)
    init_embed_new.project_global_feats.weight.data = proj_glob_params_new
    init_embed_new.project_customers_feats.weight.data = (
        init_embedding_old.project_customers_feats.weight.data
    )

    # Context Embedding with EAL (decoder)
    context_embedding_old = deepcopy(policy.decoder.context_embedding)
    # use previous k weights and pad l new number of weights with zeros
    proj_context_old = context_embedding_old.project_context.weight.data
    proj_context_new = torch.cat(
        [
            proj_context_old,
            torch.zeros(
                (proj_context_old.shape[0], context_embedding_num_new_feats),
                device=proj_context_old.device,
            ),
        ],
        dim=-1,
    )
    context_embed_new = context_embedding_cls(embed_dim=embed_dim)
    context_embed_new.project_context.weight.data = proj_context_new

    # Replace above into the policy
    policy_new.encoder.init_embedding = init_embed_new
    policy_new.decoder.context_embedding = context_embed_new

    # If not full, then we freeze the backbone
    if adapter_only:
        policy_new = freeze_backbone(policy_new)

    model.policy = policy_new
    return model
