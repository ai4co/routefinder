import torch

from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict

log = get_pylogger(__name__)


def render(
    td: TensorDict,
    actions=None,
    ax=None,
    scale_xy: bool = True,
    vehicle_capacity=None,
    fontsize: int = 10,
    return_ax: bool = False,
):
    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    route_starts = (actions[:-1] == 0) & (actions[1:] != 0)
    num_routes = route_starts.sum().item()
    cmap = cm.get_cmap("turbo")
    color_list = [cmap(x) for x in np.linspace(0.05, 0.95, num_routes)]
    out = ListedColormap(color_list)

    if ax is None:
        _, ax = plt.subplots(dpi=100, figsize=(6, 6))

    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)

    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]

    locs = td["locs"]
    scale_demand = td["capacity_original"]
    demands_linehaul = td["demand_linehaul"] * scale_demand
    demands_backhaul = td["demand_backhaul"] * scale_demand

    # scale to closest integer
    if demands_linehaul.max() <= 1:  # fallback for no scaling
        # scale min value except 0 to 1 and max value to 9
        demands_linehaul = (
            (demands_linehaul - demands_linehaul.min())
            / (demands_linehaul.max() - demands_linehaul.min())
            * 9
        )
        demands_backhaul = (
            (demands_backhaul - demands_backhaul.min())
            / (demands_backhaul.max() - demands_backhaul.min())
            * 9
        )

        demands_linehaul = demands_linehaul.round().int()
        demands_backhaul = demands_backhaul.round().int()

    if actions is None:
        log.warning("No action in TensorDict, rendering unsorted locs")
    else:
        actions = torch.cat([torch.tensor([0]), actions, torch.tensor([0])])

    # Depot
    ax.scatter(
        locs[0, 0],
        locs[0, 1],
        edgecolors=cm.Set2(2),
        facecolors="none",
        s=100,
        linewidths=2,
        marker="s",
        alpha=1,
    )

    for node_idx, loc in enumerate(locs):
        if node_idx == 0:
            continue
        delivery, pickup = demands_linehaul[node_idx], demands_backhaul[node_idx]
        if delivery > 0:
            ax.text(
                loc[0],
                loc[1] + 0.02,
                f"{delivery.item()}",
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=fontsize,
                color=cm.Set2(0),
            )
            # scatter delivery as downward triangle
            ax.scatter(
                loc[0],
                loc[1],
                edgecolors=cm.Set2(0),
                facecolors="none",
                s=30,
                linewidths=2,
                marker="v",
                alpha=1,
            )
        elif pickup > 0:
            ax.text(
                loc[0],
                loc[1] - 0.02,
                f"{pickup.item()}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=fontsize,
                color=cm.Set2(1),
            )
            ax.scatter(
                loc[0],
                loc[1],
                edgecolors=cm.Set2(1),
                facecolors="none",
                s=30,
                linewidths=2,
                marker="^",
                alpha=1,
            )
        else:
            print("Error: no demand")

    color_idx = 0
    next_actions = torch.roll(actions, -1, 0)
    for ai, aj in zip(actions, next_actions):
        # if open and next is depot (=0), skip
        if td["open_route"].item() and aj == 0:
            continue
        if ai == 0:
            color_idx += 1
        from_loc = locs[ai]
        to_loc = locs[aj]
        # if any of from_loc or to_loc is depot, change color and linewidth
        if ai == 0 or aj == 0:
            color, lw, alpha, style = "lightgrey", 1, 0.5, "--"
        else:
            color, lw, alpha, style = out(color_idx), 1, 1, ""
        ax.plot(
            [from_loc[0], to_loc[0]],
            [from_loc[1], to_loc[1]],
            color=color,
            lw=lw,
            alpha=alpha,
            linestyle=style,
        )
        ax.annotate(
            "",
            xy=(to_loc[0], to_loc[1]),
            xytext=(from_loc[0], from_loc[1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, alpha=alpha),
            size=15,
            annotation_clip=False,
        )

    if scale_xy:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    # Remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    if return_ax:
        return ax
    else:
        plt.show()
