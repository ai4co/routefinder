def freeze_backbone(policy):
    # Freeze all the parameters in the model
    for param in policy.parameters():
        param.requires_grad = False
    # Unfreeze embeddings
    for param in policy.encoder.init_embedding.parameters():
        param.requires_grad = True
    for param in policy.decoder.context_embedding.parameters():
        param.requires_grad = True
    return policy
