import os

from lightning.pytorch import seed_everything
from rl4co.data.utils import save_tensordict_to_npz

from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator

# Reproducibility, hardcoded
seed_everything(42, workers=True)
folder = "data/"


def generate(num_loc, num_data, variant, phase="val"):
    filename = f"{variant}/{phase}/{num_loc}.npz"
    path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    generator = MTVRPGenerator(num_loc=num_loc, variant_preset=variant)
    env = MTVRPEnv(generator, check_solution=False)
    td_data = env.generator(num_data)

    print(f"Saving {path}")
    save_tensordict_to_npz(td_data, path)


def main():
    for variant in MTVRPGenerator.available_variants():
        # Validation (less data for faster training)
        generate(50, 128, variant, phase="val")
        generate(100, 128, variant, phase="val")
        generate(200, 128, variant, phase="val")

        # Test
        generate(50, 1000, variant, phase="test")
        generate(100, 1000, variant, phase="test")
        generate(200, 1000, variant, phase="test")
        generate(500, 128, variant, phase="test")
        generate(1000, 128, variant, phase="test")


if __name__ == "__main__":
    input(
        "WARNING: you should not generate the dataset but download it from Github"
        " since generation results are not reproducible across devices. Press Enter to continue anyways."
    )

    main()
