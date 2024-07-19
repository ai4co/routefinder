import os

from rl4co.utils.trainer import RL4COTrainer

from routefinder.envs.mtvrp import MTVRPEnv
from routefinder.models import RouteFinderBase, RouteFinderPolicy

# Get env variable MAC_OS_GITHUB_RUNNER and force CPU in that case
if "MAC_OS_GITHUB_RUNNER" in os.environ:
    accelerator = "cpu"
else:
    accelerator = "auto"


def test_training():
    env = MTVRPEnv(generator_params={"num_loc": 10, "variant_preset": "all"})
    policy = RouteFinderPolicy()
    model = RouteFinderBase(
        env,
        policy,
        batch_size=3,
        train_data_size=3,
        val_data_size=3,
        test_data_size=3,
        optimizer_kwargs={"lr": 3e-4, "weight_decay": 1e-6},
    )
    trainer = RL4COTrainer(
        max_epochs=1,
        gradient_clip_val=None,
        devices=1,
        accelerator=accelerator,
    )
    trainer.fit(model)
    trainer.test(model)
