import torch

from rl4co.envs import RL4COEnvBase


def get_select_start_nodes_fn(select_start_nodes, **kwargs):
    if select_start_nodes == "random":
        return RandomStartNodes(**kwargs)
    elif select_start_nodes == "all":  # default from POMO
        return AllSelectStartNodes(**kwargs)
    elif select_start_nodes == "smart":
        return SmartSelectStartNodes(**kwargs)
    else:
        raise ValueError(f"Unknown select_start_nodes: {select_start_nodes}")


class SelectStartNodes:
    def __init__(self, num_starts=None):
        self.num_starts = num_starts

    def __call__(self, td, num_starts, backup_n_starts, **kwargs):
        if self.num_starts is not None:
            num_starts = self.num_starts
        if isinstance(
            num_starts, RL4COEnvBase
        ):  # to avoid current RL4CO problematic API, to fix
            num_starts = backup_n_starts
        return self._select(td, num_starts)

    def _select(self, td, num_starts):
        raise NotImplementedError("Implement this method in a subclass")

    def get_num_starts(self, td):
        return (
            self.num_starts if self.num_starts is not None else td["locs"].shape[-2] - 1
        )  # exclude depot


class RandomStartNodes(SelectStartNodes):
    def _select(self, td, num_starts):
        return torch.randint(0, td["locs"].size(1), (td["locs"].size(0), num_starts))


class AllSelectStartNodes(SelectStartNodes):
    def _select(self, td, num_starts):
        num_loc = td["locs"].shape[-2] - 1
        selected = (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_loc
            + 1
        )
        return selected


class SmartSelectStartNodes(SelectStartNodes):
    def __init__(
        self,
        num_starts=None,
        num_nearest=20,
        exclude_backhauls=True,
    ):
        self.num_nearest = num_nearest
        self.exclude_backhauls = exclude_backhauls
        super().__init__(num_starts)

    def _select(self, td, num_starts):
        """let's select as starting nodes all nodes in a `top_k` closest neighborhood
        of the depot of **linehaul customers only**
        """
        # compute distance from depot to all customers
        d_0i = torch.cdist(td["locs"][:, 0:1, :], td["locs"][:, 1:, :], p=2)

        # set to a high value the distance to backhauls
        if self.exclude_backhauls:
            is_backhaul = td["demand_backhaul"][:, 1:] > 0
            d_0i[is_backhaul[:, None]] += 1e6

        # get top k closest customers
        # flatten for [B, num_starts] -> [B*num_starts]
        selected = (
            torch.topk(d_0i, self.num_nearest, largest=False).indices.flatten(0) + 1
        )  # depot idx

        # select starting nodes
        return selected

    def get_num_starts(self, td):
        return self.num_nearest
