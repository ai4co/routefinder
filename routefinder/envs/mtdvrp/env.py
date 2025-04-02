import os

from typing import List, Optional, Union

import numpy as np
import torch

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, UnboundedContinuous, UnboundedDiscrete

from routefinder.utils import rollout_actions

from .generator import MTVRPGenerator
from .selectstartnodes import get_select_start_nodes_fn
from .utils import get_starting_points

log = get_pylogger(__name__)


class MTVRPEnv(RL4COEnvBase):
    r"""MTVRPEnv is a Multi-Task VRP environment which can take any combination of the following constraints:

    Features:

    - *Capacity (C)*
        - Each vehicle has a maximum capacity :math:`Q`, restricting the total load that can be in the vehicle at any point of the route.
        - The route must be planned such that the sum of demands and pickups for all customers visited does not exceed this capacity.
    - *Time Windows (TW)*
        - Every node :math:`i` has an associated time window :math:`[e_i, l_i]` during which service must commence.
        - Additionally, each node has a service time :math:`s_i`. Vehicles must reach node :math:`i` within its time window; early arrivals must wait at the node location until time :math:`e_i`.
    - *Open Routes (O)*
        - Vehicles are not required to return to the depot after serving all customers.
        - Note that this does not need to be counted as a constraint since it can be modelled by setting zero costs on arcs returning to the depot :math:`c_{i0} = 0` from any customer :math:`i \in C`, and not counting the return arc as part of the route.
    - *Backhauls (B)*
        - Backhauls generalize demand to also account for return shipments. Customers are either linehaul or backhaul customers.
        - Linehaul customers require delivery of a demand :math:`q_i > 0` that needs to be transported from the depot to the customer, whereas backhaul customers need a pickup of an amount :math:`p_i > 0` that is transported from the client back to the depot.
        - It is possible for vehicles to serve a combination of linehaul and backhaul customers in a single route, but then any linehaul customers must precede the backhaul customers in the route.
    - *Duration Limits (L)*
        - Imposes a limit on the total travel duration (or length) of each route, ensuring a balanced workload across vehicles.
    - *Mixed (M) Backhaul (M)*
        - This is a variant of the backhaul constraint where the vehicle can pick up and deliver linehaul customers in any order.
        - However, we need to ensure that the vehicle has enough capacity to deliver the linehaul customers and that the vehicle can pick up backhaul customers only if it has enough capacity to deliver the linehaul customers.

    The environment covers the following 16 variants depending on the data generation:

    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRP Variant  || Capacity (C) | Open Route (O) | Backhaul (B) | Duration Limit (L) | Time Window (TW) |
    +==============++==============+================+==============+====================+==================+
    | CVRP         || ✔            |                |              |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRP         || ✔            | ✔              |              |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPB         || ✔            |                | ✔            |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPL         || ✔            |                |              | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPTW        || ✔            |                |              |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPTW       || ✔            | ✔              |              |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPB        || ✔            | ✔              | ✔            |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPL        || ✔            | ✔              |              | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPBL        || ✔            |                | ✔            | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPBTW       || ✔            |                | ✔            |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPLTW       || ✔            |                |              | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPBL       || ✔            | ✔              | ✔            | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPBTW      || ✔            | ✔              | ✔            |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPLTW      || ✔            | ✔              |              | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPBLTW      || ✔            |                | ✔            | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPBLTW     || ✔            | ✔              | ✔            | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+

    Additionally, with the mixed backhaul (M) variant, we obtain 24 variants.

    You may also check out `"Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization" (Liu et al., 2024) <https://arxiv.org/abs/2402.16891>`_
    and `"MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts" (Zhou et al, 2024) <https://arxiv.org/abs/2405.01029>`_.


    Note:
        Have a look at https://pyvrp.org/ for more information about VRP and its variants and their solutions. Kudos to their help and great job!

    Args:
        generator: Generator for the environment, see :class:`MTVRPGenerator`.
        generator_params: Parameters for the generator.
    """

    name = "mtdvrp"

    def __init__(
        self,
        generator: MTVRPGenerator = None,
        generator_params: dict = {},
        select_start_nodes_fn: Union[str, callable] = "all",
        check_solution: bool = False,
        load_solutions: bool = True,
        solution_fname: str = "_sol_pyvrp.npz",
        **kwargs,
    ):
        super().__init__(check_solution=check_solution, **kwargs)
        if generator is None:
            generator = MTVRPGenerator(**generator_params)

        if check_solution:
            log.warning(
                "Solution checking is enabled. This may slow down the environment."
                " We recommend disabling this for training by passing `check_solution=False`."
            )

        self.generator = generator
        if isinstance(select_start_nodes_fn, str):
            self.select_start_nodes_fn = get_select_start_nodes_fn(select_start_nodes_fn)
        else:
            self.select_start_nodes_fn = select_start_nodes_fn

        self.solution_fname = solution_fname
        self.load_solutions = load_solutions
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        # depot nodes
        num_depots = int(td["num_depots"][0].item())
        depot_idx = torch.arange(num_depots, device=td.device)

        # Get locations and distance
        prev_node, curr_node = td["current_node"], td["action"]
        prev_loc = gather_by_index(td["locs"], prev_node)
        curr_loc = gather_by_index(td["locs"], curr_node)
        depot_loc = gather_by_index(td["locs"], td["current_depot"])
        distance = get_distance(prev_loc, curr_loc)[..., None]
        dist2depot = get_distance(prev_loc, depot_loc)[..., None]

        # for indexing
        in_depot = torch.isin(curr_node, depot_idx)
        not_in_depot = ~in_depot[..., None]  # note the dimensions
        depot2depot = torch.isin(prev_node, depot_idx) & in_depot

        distance[in_depot] = dist2depot[in_depot]  # always return to *current* depot
        distance[in_depot & td["open_route"].squeeze(-1)] = 0.0  # discard for open route
        distance[depot2depot] = 0.0
        td["current_depot"][in_depot] = curr_node[in_depot]  # update current depot

        # Update current time
        service_time = gather_by_index(
            src=td["service_time"], idx=curr_node, dim=1, squeeze=False
        )
        start_times = gather_by_index(
            src=td["time_windows"], idx=curr_node, dim=1, squeeze=False
        )[..., 0]
        # we cannot start before we arrive and we should start at least at start times
        curr_time = not_in_depot * (
            torch.max(td["current_time"] + distance / td["speed"], start_times)
            + service_time
        )

        # Update current route length (reset at depot)
        curr_route_length = not_in_depot * (td["current_route_length"] + distance)
        total_distance = td["total_distance"] + distance

        # Linehaul (delivery) demands
        selected_demand_linehaul = gather_by_index(
            td["demand_linehaul"], curr_node, dim=1, squeeze=False
        )
        selected_demand_backhaul = gather_by_index(
            td["demand_backhaul"], curr_node, dim=1, squeeze=False
        )

        # Backhaul (pickup) demands
        # this holds for backhaul_classes 0, 1, and 2:
        used_capacity_linehaul = not_in_depot * (
            td["used_capacity_linehaul"] + selected_demand_linehaul
        )
        used_capacity_backhaul = not_in_depot * (
            td["used_capacity_backhaul"] + selected_demand_backhaul
        )

        # Done when all customers are visited
        visited = td["visited"].scatter(-1, curr_node[..., None], True)
        done = visited[..., num_depots:].all(-1)
        reward = torch.zeros_like(
            done
        ).float()  # we use the `get_reward` method to compute the reward

        td.update(
            {
                "current_node": curr_node,
                "current_route_length": curr_route_length,
                "current_time": curr_time,
                "done": done,
                "reward": reward,
                "total_distance": total_distance,
                "used_capacity_linehaul": used_capacity_linehaul,
                "used_capacity_backhaul": used_capacity_backhaul,
                "visited": visited,
            }
        )
        td = self.get_action_mask(td)
        return td

    def _reset(
        self,
        td: Optional[TensorDict],
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        device = td.device
        num_depots = int(td["num_depots"][0].item())

        # Demands: linehaul (C) and backhaul (B). Backhaul defaults to 0
        demand_linehaul = torch.cat(
            [
                torch.zeros_like(td["demand_linehaul"][..., :num_depots]),
                td["demand_linehaul"],
            ],
            dim=1,
        )
        demand_backhaul = td.get(
            "demand_backhaul",
            torch.zeros_like(td["demand_linehaul"]),
        )
        demand_backhaul = torch.cat(
            [torch.zeros_like(td["demand_linehaul"][..., :num_depots]), demand_backhaul],
            dim=1,
        )
        # Backhaul class (MB). 1 is the default backhaul class
        backhaul_class = td.get(
            "backhaul_class",
            torch.full((*batch_size, 1), 1, dtype=torch.int32),
        )

        # Time windows (TW). Defaults to [0, inf] and service time to 0
        time_windows = td.get("time_windows", None)
        if time_windows is None:
            time_windows = torch.zeros_like(td["locs"])
            time_windows[..., 1] = float("inf")
        service_time = td.get("service_time", torch.zeros_like(demand_linehaul))

        # Open (O) route. Defaults to 0
        open_route = td.get(
            "open_route", torch.zeros_like(demand_linehaul[..., :1], dtype=torch.bool)
        )

        # Distance limit (L). Defaults to inf
        distance_limit = td.get(
            "distance_limit", torch.full_like(demand_linehaul[..., :1], float("inf"))
        )

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "num_depots": td["num_depots"],
                "locs": td["locs"],
                "demand_backhaul": demand_backhaul,
                "demand_linehaul": demand_linehaul,
                "backhaul_class": backhaul_class,
                "distance_limit": distance_limit,
                "service_time": service_time,
                "open_route": open_route,
                "time_windows": time_windows,
                "speed": td.get("speed", torch.ones_like(demand_linehaul[..., :1])),
                "vehicle_capacity": td.get(
                    "vehicle_capacity", torch.ones_like(demand_linehaul[..., :1])
                ),
                "capacity_original": td.get(
                    "capacity_original", torch.ones_like(demand_linehaul[..., :1])
                ),
                "current_depot": torch.zeros(
                    (*batch_size,), dtype=torch.long, device=device
                ),
                "current_node": torch.zeros(
                    (*batch_size,), dtype=torch.long, device=device
                ),
                "current_route_length": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),  # for distance limits
                "current_time": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),  # for time windows
                "total_distance": torch.full(
                    (*batch_size, 1), -1, dtype=torch.float32, device=device
                ),  # for reward calculation
                "used_capacity_backhaul": torch.zeros(
                    (*batch_size, 1), device=device
                ),  # for capacity constraints in backhaul
                "used_capacity_linehaul": torch.zeros(
                    (*batch_size, 1), device=device
                ),  # for capacity constraints in linehaul
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2]),
                    dtype=torch.bool,
                    device=device,
                ),
                "depot_available": torch.ones(
                    (*batch_size, num_depots),
                    dtype=torch.bool,
                    device=device,
                ),
            },
            batch_size=batch_size,
            device=device,
        )
        td_reset = self.get_action_mask(td_reset)
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> TensorDict:
        """Get action mask for the environment at the current step.
        Note that other variables might need to be changed.
        """
        num_depots = int(td["num_depots"][0].item())
        if (td["total_distance"] == -1).all():
            # in the first step sample a depot
            initial_mask = torch.zeros_like(td["demand_linehaul"], dtype=torch.bool)
            initial_mask[..., :num_depots] = True
            td["total_distance"][...] = 0
            td.set("action_mask", initial_mask)
            return td

        curr_node = td["current_node"]  # note that this was just updated!
        locs = td["locs"]
        d_ij = get_distance(
            gather_by_index(locs, curr_node)[..., None, :], locs
        )  # i (current) -> j (next)

        # distance to *current* depot
        curr_depot_loc = locs[
            torch.arange(locs.shape[0], device=locs.device), None, td["current_depot"]
        ]
        d_j0 = get_distance(locs, curr_depot_loc)  # j (next) -> 0 (depot)

        # Time constraint (TW):
        early_tw, late_tw = (
            td["time_windows"][..., 0],
            td["time_windows"][..., 1],
        )
        arrival_time = td["current_time"] + (d_ij / td["speed"])
        # can reach in time -> only need to *start* in time
        can_reach_customer = arrival_time < late_tw
        # we must ensure that we can return to depot in time *if* route is closed
        # i.e. start time + service time + time back to depot < late_tw
        can_reach_depot = (
            torch.max(arrival_time, early_tw) + td["service_time"] + (d_j0 / td["speed"])
        ) * ~td["open_route"] < late_tw[
            ..., 0:1
        ]  # note tws are the same for all depots

        # Distance limit (L): do not add distance to depot if open route (O)
        exceeds_dist_limit = (
            td["current_route_length"] + d_ij + (d_j0 * ~td["open_route"])
            > td["distance_limit"]
        )

        # Capacity constraints linehaul (C) and backhaul (B)
        exceeds_cap_linehaul = (
            td["demand_linehaul"] + td["used_capacity_linehaul"] > td["vehicle_capacity"]
        )
        exceeds_cap_backhaul = (
            td["demand_backhaul"] + td["used_capacity_backhaul"] > td["vehicle_capacity"]
        )

        # Backhaul class 1 (classical backhaul) (B)
        # every customer is either backhaul or linehaul, all linehauls are visited before backhauls
        linehauls_missing = ((td["demand_linehaul"] * ~td["visited"]).sum(-1) > 0)[
            ..., None
        ]
        is_carrying_backhaul = (
            gather_by_index(
                src=td["demand_backhaul"],
                idx=curr_node,
                dim=1,
                squeeze=False,
            )
            > 0
        )
        meets_demand_constraint_backhaul_1 = (
            linehauls_missing
            & ~exceeds_cap_linehaul
            & ~is_carrying_backhaul
            & (td["demand_linehaul"] > 0)
        ) | (~exceeds_cap_backhaul & (td["demand_backhaul"] > 0))

        # Backhaul class 2 (mixed pickup and delivery / mixed backhaul) (MB)
        # to serve linehaul customers we additionally need to check the remaining capacity in the vehicle
        # capacity is vehicle_capacity-used_capacity_backhauls, as all used_capacity_linehaul at this point have already been *delivered*
        cannot_serve_linehaul = (
            td["demand_linehaul"] > td["vehicle_capacity"] - td["used_capacity_backhaul"]
        )
        meets_demand_constraint_backhaul_2 = (
            ~exceeds_cap_linehaul & ~exceeds_cap_backhaul & ~cannot_serve_linehaul
        )

        # Now we merge the constraints of backhaul class 1 and 2 depending on the backhaul class
        meets_demand_constraint = (
            (td["backhaul_class"] == 1) & meets_demand_constraint_backhaul_1
        ) | ((td["backhaul_class"] == 2) & meets_demand_constraint_backhaul_2)

        # Condense constraints
        can_visit = (
            can_reach_customer
            & can_reach_depot
            & meets_demand_constraint
            & ~exceeds_dist_limit
            & ~td["visited"]
        )

        # Mask depot: don't visit depot if coming from there and there are still customer nodes I can visit
        can_visit[:, :num_depots] = ~(
            (torch.isin(curr_node, torch.arange(num_depots, device=curr_node.device)))
            & (can_visit[:, num_depots:].sum(-1) > 0)
            # TODO depot available?
        ).reshape(-1, 1)

        # If we are in a depot, not all customers have been visited, but we cannot visit any customer, we have a deadlock
        depot_deadlock = (
            (td["current_node"] < num_depots)
            & (~td["visited"][..., num_depots:].all(-1))
            & (~can_visit[:, num_depots:].any(-1))
        )

        # # if we are in a deadlock and only the current depot is available, set all depots as available
        depot_available = torch.where(
            depot_deadlock[..., None] & can_visit[:, :num_depots].sum(-1, keepdim=True)
            == 0,
            torch.ones_like(td["depot_available"]),
            td["depot_available"],
        )  # [b, num_depots]

        # set current depot as unavailable if there is a deadlock since it got us stuck
        depot_available.scatter_(
            -1, td["current_depot"][..., None], ~depot_deadlock[..., None]
        )
        # if there is a deadlock, set visitable depots as depot_available
        can_visit[:, :num_depots] = torch.where(
            depot_deadlock[:, None], depot_available, can_visit[:, :num_depots]
        )  # [b, num_depots]

        td.set("depot_available", depot_available)
        td.set("action_mask", can_visit)
        return td

    @staticmethod
    def _get_reward(td: TensorDict, actions: TensorDict) -> TensorDict:

        go_from = actions  # note: we don't append any slack action here
        go_to = torch.roll(go_from, -1, dims=1)  # [b, seq_len]
        loc_from = gather_by_index(td["locs"], go_from)
        loc_to = gather_by_index(td["locs"], go_to)

        starting_points = get_starting_points(actions, td["num_depots"])
        actual_depot = torch.roll(
            starting_points, 1, dims=1
        )  # "overwrite" the destination depot with the actual depot
        loc_actual_depot = gather_by_index(td["locs"], actual_depot)

        # Get tour length. If route is open and goes to depot, don't count the distance
        distances = get_distance(loc_from, loc_to)  # [b, seq_len]
        distances_to_depot = get_distance(loc_from, loc_actual_depot)  # [b, seq_len]

        # where the route goes back to depot, the distance is to depot
        is_depot = go_to < td["num_depots"]
        distances = torch.where(
            is_depot, distances_to_depot * ~td["open_route"], distances
        )

        # If depot to depot, distance is 0
        is_depot_to_depot = (go_from < td["num_depots"]) & (go_to < td["num_depots"])
        distances = torch.where(is_depot_to_depot, torch.zeros_like(distances), distances)

        # Sum up and return
        tour_length = distances.sum(-1)  # [b]
        return -tour_length  # reward is negative cost

    def _get_reward_full_rollout_old(
        self, td: TensorDict, actions: TensorDict
    ) -> TensorDict:
        if not (td["total_distance"] > 0).all():
            td = rollout_actions(env=self, td=td, actions=actions)
        # consider way back to depot
        td.set("action", torch.zeros_like(td["action"]))
        td = self.step(td)["next"]
        return -td["total_distance"].squeeze(-1)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        # TODO adjust for multi-depot
        batch_size, n_loc = td["demand_linehaul"].size()
        num_depots = int(
            td["num_depots"][0].item()
        )  # note: it should be max for flexibility
        n_loc -= num_depots  # exclude depot
        sorted_pi = actions.data.sort(1)[0]

        # all customer nodes visited exactly once
        assert (
            torch.arange(num_depots, n_loc + num_depots, out=sorted_pi.data.new())
            .view(1, -1)
            .expand(batch_size, n_loc)
            == sorted_pi[:, -n_loc:]
        ).all(), "Invalid tour: not all customer nodes visited exactly once"
        assert (sorted_pi[:, :-n_loc] < num_depots).all(), "Invalid tour"  # why?

        # Time windows (TW) and distance limits (L)
        assert torch.all(td["time_windows"] >= 0.0), "Time windows must be non-negative."
        assert torch.all(td["service_time"] >= 0.0), "Service time must be non-negative."
        assert torch.all(
            td["time_windows"][..., 0] < td["time_windows"][..., 1]
        ), "There are unfeasible time windows with start time > end time."
        assert (td["distance_limit"] >= 0).all(), "Distance limits must be non-negative."
        curr_time = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        # todo: current node need not be 0 actually but hey
        curr_node = torch.zeros(batch_size, dtype=torch.int64, device=td.device)
        curr_length = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        for ii in range(actions.size(1)):
            next_node = actions[:, ii]
            curr_loc = gather_by_index(td["locs"], curr_node)
            next_loc = gather_by_index(td["locs"], next_node)
            dist = get_distance(curr_loc, next_loc)

            # distance limit (L)
            curr_length = curr_length + dist * ~(
                td["open_route"].squeeze(-1) & (next_node == 0)
            )  # do not count back to depot for open route
            assert torch.all(
                curr_length <= td["distance_limit"].squeeze(-1)
            ), "Route exceeds distance limit"
            curr_length[next_node < num_depots] = 0.0  # reset length for depot

            curr_time = torch.max(
                curr_time + dist, gather_by_index(td["time_windows"], next_node)[..., 0]
            )
            assert torch.all(
                curr_time <= gather_by_index(td["time_windows"], next_node)[..., 1]
            ), "vehicle cannot start service before deadline"
            curr_time = curr_time + gather_by_index(td["service_time"], next_node)
            assert torch.all(
                curr_time <= gather_by_index(td["time_windows"], next_node)[..., 1]
            ), "vehicle cannot perform service by deadline"
            curr_node = next_node
            curr_time[curr_node < num_depots] = 0.0  # reset time for depot

        # Demand constraints (C) and (B) and (MB)
        # we keep track of the current picked up linehaul and backhaul
        # and the used capacity of both
        demand_l = td["demand_linehaul"].gather(dim=1, index=actions)
        demand_b = td["demand_backhaul"].gather(dim=1, index=actions)
        used_cap_l = torch.zeros_like(td["demand_linehaul"][:, 0])
        used_cap_b = torch.zeros_like(td["demand_backhaul"][:, 0])
        for ii in range(actions.size(1)):
            # reset at depot
            used_cap_l = used_cap_l * (actions[:, ii] != 0)
            used_cap_b = used_cap_b * (actions[:, ii] != 0)
            # increase counters
            used_cap_l += demand_l[:, ii]
            used_cap_b += demand_b[:, ii]

            # For backhaul_class 1 (B), we must ensure that if we are carrying backhaul, we are not picking up linehaul
            assert (
                (td["backhaul_class"] == 2)
                | (used_cap_b == 0)
                | ((td["backhaul_class"] == 1) & ~(demand_l[:, ii] > 0))
            ).all(), "Cannot pick up linehaul while carrying backhaul due to precedence constraints"

            # For backhaul_class 2 (MB), we cannot pick up linehaul if the used capacity of backhaul is already at the vehicle capacity
            # also, cannot pick up other backhauls if we are full
            assert (
                (td["backhaul_class"] == 1)
                | (used_cap_b == 0)
                | (
                    (td["backhaul_class"] == 2)
                    & (used_cap_b + demand_l[:, ii] <= td["vehicle_capacity"])
                )
            ).all(), "Cannot deliver linehaul, not enough load"

            # Assertions: total used linehaul and backhaul capacity should not exceed vehicle capacity
            assert (
                used_cap_l <= td["vehicle_capacity"]
            ).all(), "Used more linehaul than capacity: {} / {}".format(
                used_cap_l, td["vehicle_capacity"]
            )
            assert (
                used_cap_b <= td["vehicle_capacity"]
            ).all(), "Used more backhaul than capacity: {} / {}".format(
                used_cap_b, td["vehicle_capacity"]
            )

    def get_num_starts(self, td):
        return self.select_start_nodes_fn.get_num_starts(td)

    def select_start_nodes(self, td, num_starts):
        return self.select_start_nodes_fn(td, num_starts, self.get_num_starts(td))

    @staticmethod
    def render(*args, **kwargs):
        """Simple wrapper for render function"""
        from .render import render

        return render(*args, **kwargs)

    def _make_spec(self, td_params: TensorDict):
        # TODO: include extra vars (but we don't really need them for now)
        """Make the observation and action specs from the parameters."""
        self.observation_spec = Composite(
            locs=Bounded(
                low=self.generator.min_loc,
                high=self.generator.max_loc,
                shape=(self.generator.num_loc + 1, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            current_node=UnboundedDiscrete(
                shape=(1),
                dtype=torch.int64,
                device=self.device,
            ),
            demand_linehaul=Bounded(
                low=-self.generator.capacity,
                high=self.generator.max_demand,
                shape=(self.generator.num_loc, 1),  # demand is only for customers
                dtype=torch.float32,
                device=self.device,
            ),
            demand_backhaul=Bounded(
                low=-self.generator.capacity,
                high=self.generator.max_demand,
                shape=(self.generator.num_loc, 1),  # demand is only for customers
                dtype=torch.float32,
                device=self.device,
            ),
            action_mask=UnboundedDiscrete(
                shape=(self.generator.num_loc + 1, 1),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            low=0,
            high=self.generator.num_loc + 1,
            shape=(1,),
            dtype=torch.int64,
            device=self.device,
        )
        self.reward_spec = UnboundedContinuous(
            shape=(1,), dtype=torch.float32, device=self.device
        )
        self.done_spec = UnboundedDiscrete(
            shape=(1,), dtype=torch.bool, device=self.device
        )

    @staticmethod
    def check_variants(td):
        """Check if the problem has the variants"""
        has_open = td["open_route"].squeeze(-1)
        has_tw = (td["time_windows"][..., 1] != float("inf")).any(-1)
        has_limit = (td["distance_limit"] != float("inf")).squeeze(-1)
        has_backhaul = (td["demand_backhaul"] != 0).any(-1)
        backhaul_class = td.get("backhaul_class", torch.full_like(has_open, 1))
        multi_depot = td["num_depots"] > 1
        return has_open, has_tw, has_limit, has_backhaul, backhaul_class, multi_depot

    @staticmethod
    def get_variant_names(td: TensorDict) -> Union[str, List[str]]:
        (
            has_open,
            has_time_window,
            has_duration_limit,
            has_backhaul,
            backhaul_class,
            multi_depot,
        ) = MTVRPEnv.check_variants(td)

        def _name(o, b, bc, l_, tw, md):
            if not o and not b and not l_ and not tw:
                instance_name = "CVRP"
            else:
                instance_name = "VRP"
                if o:
                    instance_name = "O" + instance_name
                if b:
                    if bc == 2:  # mixed backhaul
                        instance_name += "M"
                    instance_name += "B"
                if l_:
                    instance_name += "L"
                if tw:
                    instance_name += "TW"
            # prepend multi-depot if necessary
            if md:
                instance_name = "MD" + instance_name
            return instance_name

        if len(has_open.shape) == 0:
            return _name(
                has_open,
                has_backhaul,
                backhaul_class,
                has_duration_limit,
                has_time_window,
                multi_depot,
            )
        else:
            return [
                _name(o, b, bc, l_, tw, md)
                for o, b, bc, l_, tw, md in zip(
                    has_open,
                    has_backhaul,
                    backhaul_class,
                    has_duration_limit,
                    has_time_window,
                    multi_depot,
                )
            ]

    def print_presets(self):
        self.generator.print_presets()

    def available_variants(self):
        return self.generator.available_variants()

    def load_data(self, fpath, batch_size=[]):
        """Dataset loading from file"""
        td = load_npz_to_tensordict(fpath)
        if self.load_solutions:
            # Load solutions if they exist depending on the file name
            solution_fpath = fpath.replace(".npz", self.solution_fname)
            if os.path.exists(solution_fpath):
                sol = np.load(solution_fpath)
                sol_dict = {}
                for key, value in sol.items():
                    if isinstance(value, np.ndarray) and len(value.shape) > 0:
                        if value.shape[0] == td.batch_size[0]:
                            key = "costs_bks" if key == "costs" else key
                            key = "actions_bks" if key == "actions" else key
                            sol_dict[key] = torch.tensor(value)
                td.update(sol_dict)
            else:
                log.warning(f"No solution file found at {solution_fpath}")
        return td


if __name__ == "__main__":
    from routefinder.envs.mtvrp import MTVRPEnv as SingleDepotEnv
    from routefinder.utils import random_policy, rollout

    from .generator import MTVRPGenerator

    def test_mtdvrp(num_depots: int):
        print(f"\n### ----- Running for {num_depots} depots ----- ###\n")
        gen = MTVRPGenerator(num_depot=num_depots)
        env = MTVRPEnv(generator=gen)
        td = env.reset(None, batch_size=[3])

        # manual override
        if num_depots > 1:
            td["open_route"][0] = False
            td["open_route"][1] = True

        actions = rollout(env=env, td=td, policy=random_policy, max_steps=100)
        print("actions", actions)
        print("done", td["done"])

        reward_n1 = env._get_reward(td, actions)
        print("reward_n1", reward_n1)

        # reset total distance
        td["total_distance"] = torch.full_like(td["total_distance"], -1)

        # verify reward calculation
        reward_n2 = env._get_reward(td, actions)
        print("reward_n2", reward_n2)

        print("reward_n1 == reward_n2", reward_n1 == reward_n2)
        print("actions", actions)
        return {"td": td, "actions": actions, "reward": reward_n1}

    out1 = test_mtdvrp(num_depots=1)
    out3 = test_mtdvrp(num_depots=3)

    sd = SingleDepotEnv()
    rew_sd = sd.get_reward(td=out1["td"], actions=out1["actions"])
    rew_md = out1["reward"]

    print("### ----- Compare to Single Depot Environment ----- ###")
    print("reward single depot", rew_sd)
    print("reward multi depot", rew_md)
    print("rew_sd == rew_md", rew_sd == rew_md)
    print("reward identical", torch.allclose(rew_sd, rew_md, atol=1e-32))
