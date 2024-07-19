import os

from lightning.pytorch import seed_everything
from rl4co.data.utils import save_tensordict_to_npz

from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator

# Reproducibility, hardcoded
seed_everything(42, workers=True)
folder = "data/"


def generate(num_loc, num_data, variant, phase="val", mixed=False):
    if mixed:
        # variant mb: find "b", insert "m" before i
        new_variant = variant[: variant.find("b")] + "m" + variant[variant.find("b") :]
        filename = f"{new_variant}/{phase}/{num_loc}.npz"
        backhaul_class = 2
    else:
        filename = f"{variant}/{phase}/{num_loc}.npz"
        backhaul_class = 1

    path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    generator = MTVRPGenerator(
        num_loc=num_loc, variant_preset=variant, backhaul_class=backhaul_class
    )
    env = MTVRPEnv(generator, check_solution=False)
    td_data = env.generator(num_data)

    print(f"Saving {path}")
    save_tensordict_to_npz(td_data, path)


def main():
    # validation has less data for faster training
    for variant in MTVRPGenerator.available_variants():
        generate(50, 128, variant, phase="val")
        generate(100, 128, variant, phase="val")
        generate(200, 128, variant, phase="val")
        generate(50, 1000, variant, phase="test")
        generate(100, 1000, variant, phase="test")
        generate(200, 1000, variant, phase="test")

        # mixed variants: if not contains "b", skip
        if "b" not in variant:
            continue
        else:
            generate(50, 128, variant, phase="val", mixed=True)
            generate(100, 128, variant, phase="val", mixed=True)
            generate(200, 128, variant, phase="val", mixed=True)
            generate(50, 1000, variant, phase="test", mixed=True)
            generate(100, 1000, variant, phase="test", mixed=True)
            generate(200, 1000, variant, phase="test", mixed=True)


if __name__ == "__main__":
    input(
        "Warning: generated data may differ slightly across devices due to PyTorch's random number generator. "
        "The distribution will however be the same, so results should be comparable. "
        "To ensure full reproducibility and make sure the data is exactly the same, you may use the uploaded files under the data/ folder."
        "Note that this will overwrite any existing datasets. Press Enter to confirm."
    )

    main()
