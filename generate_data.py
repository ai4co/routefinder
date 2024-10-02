import os

from routefinder.data.generate import generate_dataset, VARIANT_FEATURES
import logging

logging.basicConfig(level=logging.INFO)
    

if __name__ == "__main__":
    data_dir = "data"
    seeds = {
        "val": 4321,
        "test": 1234,
    }
    sizes = [50, 100]

    for problem in VARIANT_FEATURES:
        problem = problem.lower()
        for phase, seed in seeds.items():
            for size in sizes:
                generate_dataset(
                    problem=problem,
                    data_dir=data_dir,
                    filename=data_dir + f"/{problem}/{phase}/{size}.npz",
                    dataset_size=1000,
                    graph_sizes=size,
                    seed=seed,
                )
