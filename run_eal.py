from copy import deepcopy

import torch
import torch.nn as nn

from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary
from lightning.pytorch.loggers import WandbLogger
from rl4co.utils.callbacks.speed_monitor import SpeedMonitor

# new model
from rl4co.utils.trainer import RL4COTrainer

import wandb

from routefinder.envs import MTVRPEnv, MTVRPGenerator
from routefinder.models.baselines.mvmoe.model import MVMoE

## Normal training (note that we will actually just load a checkpoint)
## Zero shot (after training)
from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbeddingM
from routefinder.models.env_embeddings.mtvrp.init import MTVRPInitEmbeddingM
from routefinder.models.model import (
    RouteFinderBase,
    RouteFinderMoE,
    RouteFinderSingleVariantSampling,
)

# Load data into env
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def model_from_scratch(model):
    """Reinitializes from scratch with new model and new embeddings"""

    print("Reinitializing full model from scratch")
    embed_dim = model.policy.encoder.init_embedding.embed_dim

    def reset_weights(m):
        if isinstance(m, nn.Module) and hasattr(m, "reset_parameters"):
            m.reset_parameters()

    model.policy.apply(reset_weights)
    model.policy.encoder.init_embedding = MTVRPInitEmbeddingM(embed_dim=embed_dim)
    model.policy.decoder.context_embedding = MTVRPContextEmbeddingM(embed_dim=embed_dim)

    # Add `_multistart` to decode type for train, val and test in policy
    for phase in ["train", "val", "test"]:
        model.set_decode_type_multistart(phase)

    return model


def adapter_layers(model, adapter_only=False):
    """Adapter Layers (AL) from Lin et al., 2024.
    Only initializes new adapter layers (embeddings), but keeps the model parameters the same.
    """
    print("Using Adapter Layers (AL)")

    embed_dim = model.policy.encoder.init_embedding.embed_dim
    policy = model.policy

    new_init_embedding = MTVRPInitEmbeddingM(embed_dim=embed_dim)
    new_context_embedding = MTVRPContextEmbeddingM(embed_dim=embed_dim)

    policy.encoder.init_embedding = new_init_embedding
    policy.decoder.context_embedding = new_context_embedding

    # If not full, then we freeze the backbone
    if adapter_only:
        policy = freeze_backbone(policy)

    model.policy = policy
    return model


def efficient_adapter_layers(model, adapter_only=False):
    """Efficient Active Layers (ours).
    Keep the model the same, replace the embeddings with
    new zero-padded embeddings for unseen features
    """

    print("Using Efficient Adapter Layers (EAL)")

    policy = model.policy
    embed_dim = policy.decoder.context_embedding.embed_dim

    policy_new = deepcopy(policy)

    init_embedding_new_feat = MTVRPInitEmbeddingM(embed_dim=embed_dim)
    context_embedding_new_feat = MTVRPContextEmbeddingM(embed_dim=embed_dim)

    policy_new.encoder.init_embedding = init_embedding_new_feat
    policy_new.decoder.context_embedding = context_embedding_new_feat

    policy_new = policy_new.to(device)

    # Now, let's initialize the parameters: Encoder
    init_embedding_old = deepcopy(policy.encoder.init_embedding)

    # The new init embedding only has a new column (last one). So we can pad that with 0
    proj_glob_params_old = init_embedding_old.project_global_feats.weight.data
    proj_glob_params_new = torch.cat(
        [proj_glob_params_old, torch.zeros_like(proj_glob_params_old[:, :1])], dim=-1
    )

    init_embed_new = MTVRPInitEmbeddingM(embed_dim=embed_dim)
    init_embed_new.project_global_feats.weight.data = proj_glob_params_new
    init_embed_new.project_customers_feats.weight.data = (
        init_embedding_old.project_customers_feats.weight.data
    )

    # Now, let's initialize the parameters: Decoder
    context_embedding_old = deepcopy(policy.decoder.context_embedding)

    # The new context embedding only has a new column (last one). So we can pad that with 0
    proj_context_old = context_embedding_old.project_context.weight.data
    proj_context_new = torch.cat(
        [proj_context_old, torch.zeros_like(proj_context_old[:, :1])], dim=-1
    )

    context_embed_new = MTVRPContextEmbeddingM(embed_dim=embed_dim)
    context_embed_new.project_context.weight.data = proj_context_new

    # Replace above into the policy
    policy_new.encoder.init_embedding = init_embed_new
    policy_new.decoder.context_embedding = context_embed_new

    # If not full, then we freeze the backbone
    if adapter_only:
        policy_new = freeze_backbone(policy_new)

    model.policy = policy_new
    return model


# Load checkpoint
def main(path, model_type="rf", train_type="eal-full", lr=3e-4):
    if "rf" in model_type:
        if "moe" in model_type:
            model = RouteFinderMoE.load_from_checkpoint(path, map_location="cpu")
        else:
            model = RouteFinderBase.load_from_checkpoint(path, map_location="cpu")
    elif model_type == "mvmoe":
        model = MVMoE.load_from_checkpoint(path, map_location="cpu")
    elif model_type == "mtpomo":
        model = RouteFinderSingleVariantSampling.load_from_checkpoint(
            path, map_location="cpu"
        )
    else:
        raise ValueError("Model type not recognized: {}".format(model_type))

    model = model.to(device)

    if "eal" in train_type:
        model = efficient_adapter_layers(model, adapter_only="adapter" in train_type)
    # elif train_type == "al":
    elif "al" in train_type:
        model = adapter_layers(model, adapter_only="adapter" in train_type)
    elif train_type == "scratch":
        model = model_from_scratch(model)
    else:
        raise ValueError(
            "Training type not recognized: {}. Choose from ['eal', 'al', 'scratch']".format(
                train_type
            )
        )

    # Set correct paths
    dataloader_names = [
        "cvrp100",
        "ovrp100",
        "ovrpb100",
        "ovrpbl100",
        "ovrpbltw100",
        "ovrpbtw100",
        "ovrpl100",
        "ovrpltw100",
        "ovrptw100",
        "vrpb100",
        "vrpl100",
        "vrpbltw100",
        "vrpbtw100",
        "vrpbl100",
        "vrpltw100",
        "vrptw100",
    ]

    test_data = [
        "cvrp/test/100.npz",
        "ovrp/test/100.npz",
        "ovrpb/test/100.npz",
        "ovrpbl/test/100.npz",
        "ovrpbltw/test/100.npz",
        "ovrpbtw/test/100.npz",
        "ovrpl/test/100.npz",
        "ovrpltw/test/100.npz",
        "ovrptw/test/100.npz",
        "vrpb/test/100.npz",
        "vrpl/test/100.npz",
        "vrpbltw/test/100.npz",
        "vrpbtw/test/100.npz",
        "vrpbl/test/100.npz",
        "vrpltw/test/100.npz",
        "vrptw/test/100.npz",
    ]

    # Add the mixed backhaul variants
    b_variants = [d for d in dataloader_names if "b" in d]
    test_dataloader_names = dataloader_names + [d.replace("b", "mb") for d in b_variants]
    test_data = test_data + [name.replace("b", "mb") for name in test_data if "b" in name]

    val_data = [name.replace("test", "val") for name in test_data]
    val_dataloader_names = test_dataloader_names

    # Create env: the new setting is with backhaul sampling (so we have the new MB variants)
    # and also we have slightly more backhauls
    generator = MTVRPGenerator(
        num_loc=100, variant_preset="all", sample_backhaul_class=True, backhaul_ratio=0.3
    )

    env = MTVRPEnv(
        generator,
        check_solution=False,
        data_dir="./data/",
        val_file=val_data,
        test_file=test_data,
        val_dataloader_names=val_dataloader_names,
        test_dataloader_names=test_dataloader_names,
    )

    # Reset learning rate
    model.optimizer_kwargs["lr"] = lr

    # Test model
    model.env = env
    model.setup()
    model.data_cfg["batch_size"] = 128
    model.data_cfg["val_batch_size"] = 1024
    model.data_cfg["test_batch_size"] = 1024
    model.data_cfg["train_data_size"] = 10_000  # instead of 100k

    # Test model
    logger = WandbLogger(
        project="routefinder-eal",
        name=f"{model_type}-{train_type}-{lr}",
        reinit=True,
    )
    rich_model_summary = RichModelSummary(max_depth=3)
    speed_monitor = SpeedMonitor(
        intra_step_time=True, inter_step_time=True, epoch_time=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    max_epochs = 10
    trainer = RL4COTrainer(
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[rich_model_summary, speed_monitor, lr_monitor],
    )

    # Test zero-shot generalization reporting
    trainer.validate(model)

    # Main training loop
    trainer.fit(model)

    print("Finished training")
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="rf", help="Model type: rf, mvmoe, mtpomo"
    )
    parser.add_argument("--experiment", type=str, default="all")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/100/rf-transformer.ckpt"
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_runs", type=int, default=3)

    args = parser.parse_args()

    if args.experiment == "all":
        exps = [
            "eal-full",
            "eal-adapter",
            "al-full",
            "al-adapter",
            "scratch",
        ]
    else:
        exps = [args.experiment]

    for exp in exps:
        print(f"Training for {exp}")
        for i in range(args.num_runs):
            print(f"Run {i+1}/{args.num_runs}")
            main(args.checkpoint, args.model_type, exp, args.lr)
