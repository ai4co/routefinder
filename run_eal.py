import os

import torch
import wandb

from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary
from lightning.pytorch.loggers import WandbLogger
from rl4co.utils.callbacks.speed_monitor import SpeedMonitor

# new model
from rl4co.utils.trainer import RL4COTrainer

from routefinder.envs.mtdvrp import MTVRPEnv, MTVRPGenerator
from routefinder.models.baselines.mvmoe.model import MVMoE
from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbeddingFull
from routefinder.models.env_embeddings.mtvrp.init import MTVRPInitEmbeddingFull
from routefinder.models.finetuning.baselines import adapter_layers, model_from_scratch
from routefinder.models.finetuning.eal import efficient_adapter_layers

## Normal training (note that we will actually just load a checkpoint)
## Zero shot (after training)
from routefinder.models.model import (
    RouteFinderBase,
    RouteFinderMoE,
    RouteFinderSingleVariantSampling,
)

# Load data into env
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We have 3 different EAL settings: MB (mixed backhaul), MD (multi-depot), and both
VARIANT_GEN_SETTINGS = {
    "mb": {"backhaul_ratio": 0.3, "sample_backhaul_class": True, "num_depots": 1},
    "md": {"backhaul_ratio": 0.2, "sample_backhaul_class": False, "num_depots": 3},
    "both": {"backhaul_ratio": 0.3, "sample_backhaul_class": True, "num_depots": 3},
}


def finetune_variant_names(variant_name):
    # 3 cases: finetune_var = "mb", "md", "both"
    variant_names = []
    for root, dirs, files in os.walk("data"):
        # get dir whose name contains "finetune_variant"
        for dir in dirs:
            if "md" not in dir and "mb" not in dir:
                continue
            if variant_name == "mb":
                if "md" in dir:
                    continue
            elif variant_name == "md":
                if "mb" in dir:
                    continue
            else:
                if not ("mb" in dir and "md" in dir):
                    continue
            variant_names.append(dir)
    return variant_names


# Load checkpoint
def main(
    path,
    model_type="rf",
    train_type="eal-full",
    finetune_variant="mb",
    lr=3e-4,
    test_zero_shot=False,
):
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
        model = efficient_adapter_layers(
            model,
            init_embedding_cls=MTVRPInitEmbeddingFull,
            context_embedding_cls=MTVRPContextEmbeddingFull,
            init_embedding_num_new_feats=1,
            context_embedding_num_new_feats=3,
            adapter_only="adapter" in train_type,
        )
    # elif train_type == "al":
    elif "al" in train_type:
        model = adapter_layers(
            model,
            init_embedding_cls=MTVRPInitEmbeddingFull,
            context_embedding_cls=MTVRPContextEmbeddingFull,
            adapter_only="adapter" in train_type,
        )
    elif train_type == "scratch":
        model = model_from_scratch(
            model,
            init_embedding_cls=MTVRPInitEmbeddingFull,
            context_embedding_cls=MTVRPContextEmbeddingFull,
        )
    else:
        raise ValueError(
            "Training type not recognized: {}. Choose from ['eal', 'al', 'scratch']".format(
                train_type
            )
        )

    # Now, list all possible variant names in data/ which contain "finetune_variant" in their name
    variant_names = finetune_variant_names(finetune_variant)

    # Prepare dataloader names and data
    test_dataloader_names = [v + "100" for v in variant_names]
    test_data = [f"{v}/test/100.npz" for v in variant_names]
    val_data = [name.replace("test", "val") for name in test_data]
    val_dataloader_names = test_dataloader_names

    # Create env: the new setting is with backhaul sampling (so we have the new MB variants)
    # and also we have slightly more backhauls
    generator = MTVRPGenerator(
        num_loc=100, variant_preset="all", **VARIANT_GEN_SETTINGS[finetune_variant]
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
    model.data_cfg["val_batch_size"] = 256
    model.data_cfg["test_batch_size"] = 256
    model.data_cfg["train_data_size"] = 10_000  # instead of 100k

    # Test model
    logger = WandbLogger(
        project=f"routefinder-eal-{finetune_variant}",
        name=f"{model_type}-{train_type}",
        reinit=True,
        tags=[model_type, train_type, finetune_variant, "final"],
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

    # Validate and test zero-shot generalization reporting
    trainer.validate(model)
    if test_zero_shot:
        # TODO: ensure this does not overwrite the test results
        print(
            "Testing zero-shot generalization. Note that this will overwrite test results!"
        )
        trainer.test(model)

    # Main training loop
    trainer.fit(model)

    # Test model
    trainer.test(model)

    print("Finished training")
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="rf", help="Model type: rf, mvmoe, mtpomo"
    )
    parser.add_argument("--experiment", type=str, default="all", help="Experiment type")
    parser.add_argument(
        "--variants_finetune", type=str, default="all", help="Variants to finetune on"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/100/rf-transformer.ckpt"
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_runs", type=int, default=3)

    args = parser.parse_args()

    # available_experiments = ["eal-full", "eal-adapter", "al-full", "al-adapter", "scratch"]
    available_experiments = ["eal-full", "al-full", "scratch"]
    if args.experiment == "all":
        exps = available_experiments
    else:
        exps = [args.experiment]
        assert all(
            [e in available_experiments for e in exps]
        ), f"Invalid experiment: {exps}. Choose from {available_experiments}"

    available_variants = ["mb", "md", "both"]  # mixed backhaul, multi-depot, both
    if args.variants_finetune == "all":
        variants = available_variants
    else:
        variants = [args.variants_finetune]
        assert all(
            [v in available_variants for v in variants]
        ), f"Invalid variant: {variants}. Choose from {available_variants}"

    for finetune_variant in variants:
        print(f"Finetuning on {finetune_variant} variants")
        for exp in exps:
            print(f"Training for {exp}")
            for i in range(args.num_runs):
                print(f"Run {i+1}/{args.num_runs}")
                main(args.checkpoint, args.model_type, exp, finetune_variant, args.lr)
