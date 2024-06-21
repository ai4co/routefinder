# RouteFinder

_Towards Foundation Models for Vehicle Routing Problems_

---

<div align="center">
    <img src="assets/overview.png" alt="RouteFinder Overview" style="width: 100%; height: auto;">
</div>



## 🚀 Installation

Install the package in editable mode:

```bash
pip install -e .
```

if you would like to install baseline solvers as well, please install using `pip install -e '.[solvers]'`


## 🏁 Quickstart

We recommend exploring [this quickstart notebook](examples/1.quickstart.ipynb) to get started with the `RouteFinder` codebase!

### Running

The main runner (example here of main baseline) can be called via:

```bash
python run.py experiment=main/rf/rf-100
```
You may change the experiment by using the `experiment=YOUR_EXP`, with the path under [`configs/experiment`](configs/experiment) directory.


## 🚚 Available Environments

<div align="center">
    <img src="assets/vrp.png" alt="VRP Problems" style="width: 100%; height: auto;">
</div>


We consider 24 variants, which include the base Capacity (C). The $k=4$ features O, B, L, and TW can be combined into any subset, including the empty set and itself (i.e., a *power set*) with $2^k = 16$ possible combinations. Finally, we study the additional Mixed (M) global feature that creates new Backhaul (B) variants in generalization studies, adding 8 more variants.

We have the following environments available:

| | Capacity<br>(C) | Open Route<br>(O) | Backhaul<br>(B) | Mixed<br>(M) | Duration Limit<br>(L) | Time Windows<br>(TW) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| CVRP | ✔ | | | | | |
| OVRP | ✔ | ✔ | | | | |
| VRPB | ✔ | | ✔ | | | |
| VRPL | ✔ | | | | ✔ | |
| VRPTW | ✔ | | | | | ✔ |
| OVRPTW | ✔ | ✔ | | | | ✔ |
| OVRPB | ✔ | ✔ | ✔ | | | |
| OVRPL | ✔ | ✔ | | | ✔ | |
| VRPBL | ✔ | | ✔ | | ✔ | |
| VRPBTW | ✔ | | ✔ | | | ✔ |
| VRPLTW | ✔ | | | | ✔ | ✔ |
| OVRPBL | ✔ | ✔ | ✔ | | ✔ | |
| OVRPBTW | ✔ | ✔ | ✔ | | | ✔ |
| OVRPLTW | ✔ | ✔ | | | ✔ | ✔ |
| VRPBLTW | ✔ | | ✔ | | ✔ | ✔ |
| OVRPBLTW | ✔ | ✔ | ✔ | | ✔ | ✔ |
| VRPMB | ✔ | | ✔ | ✔ | | |
| OVRPMB | ✔ | ✔ | ✔ | ✔ | | |
| VRPMBL | ✔ | | ✔ | ✔ | ✔ | |
| VRPMBTW | ✔ | | ✔ | ✔ | | ✔ |
| OVRPMBL | ✔ | ✔ | ✔ | ✔ | ✔ | |
| OVRPMBTW | ✔ | ✔ | ✔ | ✔ | | ✔ |
| VRPMBLTW | ✔ | | ✔ | ✔ | ✔ | ✔ |
| OVRPMBLTW | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |

We additionally provide as baseline solvers for all baselines 1) [OR-Tools](https://github.com/google/or-tools) and 2) the SotA [PyVRP](https://github.com/PyVRP/PyVRP).

## 🔁 Reproducing Experiments

### Main Experiments
The `main` experiments on 100 nodes are (rf=RouteFinder) RF-POMO: [`rf/rf-100`](configs/experiment/main/rf/rf-100.yaml), RF-MoE-L: [`rf/rf-moe-L-100`](configs/experiment/main/rf/rf-moe-L-100.yaml), MTPOMO [`mtpomo-100`](configs/experiment/main/mtpomo/mtpomo-100.yaml) and MVMoE [`mvmoe-100`](configs/experiment/main/mvmoe/mvmoe-100.yaml). You may substitute `50` instead for 50 nodes. Note that we separate 50 and 100 because we created an automatic validation dataset reporting for all variants at different sizes (i.e. [here](configs/experiment/rfbase-100.yaml)).

Note that for MVMoE-based models, one can additionally pass the `model.hierarchical_gating=True` to enable hierarchical gating (L, Light version) - and similarly, `model.hierarchical_gating=False` for the full version.


Note that additional Hydra options as described [here](https://rl4co.readthedocs.io/en/latest/_content/start/hydra.html). For instance, you can add `+trainer.devices="[0]"` to run on a specific GPU (i.e., GPU 0).

### Ablations and more

Other configs are available under [configs/experiment](configs/experiment) directory.

### EAL (Efficient Adapter Layers)

To run EAL, you may use the following command:

```bash
python eal.py --path [MODEL_PATH]
```

with additional parameters that can be found in the [eal.py](eal.py) file.


### Known Bugs
- For some reason, there seem to be bugs when training on M series processors from Apple (but not during inference somehow?). We recommend training with a discrete GPU. We'll keep you posted with updates!


### 🤗 Acknowledgements

- https://github.com/FeiLiu36/MTNCO/tree/main
- https://github.com/RoyalSkye/Routing-MVMoE
- https://github.com/yd-kwon/POMO
- https://github.com/ai4co/rl4co


### 🤩 Citation
If you find `RouteFinder` valuable for your research or applied projects:

```
@article{berto2024routefinder,
title={{RouteFinder}: Towards Foundation Models for Vehicle Routing Problems},
author={Berto, Federico and Hua, Chuanbo and Zepeda, Nayeli Gast and Hottung, Andr{\'e} and Wouda, Niels and Lan, Leon and Tierney, Kevin and Park, Jinkyoo},
year={2024},
journal={Arxiv},
url={https://github.com/ai4co/routefinder},
}
```