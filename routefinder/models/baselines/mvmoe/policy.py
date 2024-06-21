from typing import Callable

import torch.nn as nn

from rl4co.models.zoo.am import AttentionModelPolicy

from .decoder import MVMoEDecoder
from .encoder import MVMoEEncoder


class MVMoEPolicy(AttentionModelPolicy):
    """
    https://github.com/RoyalSkye/Routing-MVMo
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        env_name: str = "mtvrp",
        encoder_network: nn.Module = None,
        init_embedding: nn.Module = None,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        use_graph_context: bool = False,
        linear_bias_decoder: bool = False,
        sdpa_fn: Callable = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        check_nan: bool = True,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        # MoE specific
        num_experts=4,
        routing_method="input_choice",
        routing_level="node",
        topk=2,
        moe_loc=["enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec"],
        hierarchical_gating=False,  # if True, corresponds to MVMoE-L
        **unused_kwargs,
    ):
        if encoder is None:
            encoder = MVMoEEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                env_name=env_name,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                net=encoder_network,
                init_embedding=init_embedding,
                sdpa_fn=sdpa_fn,
                num_experts=num_experts,
                routing_method=routing_method,
                routing_level=routing_level,
                topk=topk,
                moe_loc=moe_loc,
            )

        if decoder is None:
            decoder = MVMoEDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                sdpa_fn=sdpa_fn,
                mask_inner=mask_inner,
                out_bias_pointer_attn=out_bias_pointer_attn,
                linear_bias=linear_bias_decoder,
                use_graph_context=use_graph_context,
                check_nan=check_nan,
                num_experts=num_experts,
                routing_method=routing_method,
                routing_level=routing_level,
                topk=topk,
                moe_loc=moe_loc,
                hierarchical_gating=hierarchical_gating,
            )

        super(AttentionModelPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **unused_kwargs,
        )


class MVMoELightPolicy(MVMoEPolicy):
    def __init__(self, *args, **kwargs):
        # assert hierarchical_gating is set to true
        if "hierarchical_gating" in kwargs:
            assert kwargs[
                "hierarchical_gating"
            ], "hierarchical_gating must be set to True for MVMoELPolicy"

        kwargs["hierarchical_gating"] = True

        super(MVMoELightPolicy, self).__init__(*args, **kwargs)
