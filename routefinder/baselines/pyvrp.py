import numpy as np
import pyvrp as pyvrp

from pyvrp import Client, Depot, ProblemData, VehicleType, solve as _solve
from pyvrp.stop import MaxRuntime
from tensordict.tensordict import TensorDict
from torch import Tensor

from .utils import scale

PYVRP_SCALING_FACTOR = 1_000

# NOTE: no idea why, it still got stuck somehow, so lowered it to 1 << 32
PYVRP_MAX_VALUE = 1 << 32  # noqa: F811


def solve(instance: TensorDict, max_runtime: float, **kwargs) -> tuple[Tensor, Tensor]:
    """
    Solves the AnyVRP instance with PyVRP.

    Parameters
    ----------
    instance
        The AnyVRP instance to solve.
    max_runtime
        Maximum runtime for the solver.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple consisting of the action and the cost, respectively.
    """
    data = instance2data(instance, PYVRP_SCALING_FACTOR)
    stop = MaxRuntime(max_runtime)
    result = _solve(data, stop)

    solution = result.best
    action = solution2action(solution)
    cost = -result.cost() / PYVRP_SCALING_FACTOR

    return action, cost


def instance2data(instance: TensorDict, scaling_factor: int) -> ProblemData:
    """
    Converts an AnyVRP instance to a ProblemData instance.

    Parameters
    ----------
    instance
        The AnyVRP instance to convert.

    Returns
    -------
    ProblemData
        The ProblemData instance.
    """
    num_locs = instance["locs"].size()[0]
    num_depots = instance["num_depots"]
    num_clients = num_locs - num_depots

    time_windows = scale(instance["time_windows"], scaling_factor)  # num_locs
    pickup = scale(instance["demand_backhaul"], scaling_factor)  # num_clients
    delivery = scale(instance["demand_linehaul"], scaling_factor)  # num_clients
    service = scale(instance["durations"], scaling_factor)  # num_locs
    capacity = scale(instance["vehicle_capacity"], scaling_factor)  # 1
    max_distance = scale(instance["distance_limit"], scaling_factor)  # 1
    matrix = scale(instance["cost_matrix"], scaling_factor)  # num_locs

    # Some checks that the depot values are zero.
    assert np.all(delivery[:num_depots] == 0)
    assert np.all(pickup[:num_depots] == 0)

    # If locs is not provided, simply use zeros
    # They are not needed since the cost matrix is used instead
    if "locs" in instance:
        coords = scale(instance["locs"], scaling_factor)
    else:
        coords = np.zeros((num_locs, 2))

    depots = [Depot(x=coords[idx][0], y=coords[idx][1]) for idx in range(num_depots)]
    clients = [
        Client(
            x=coords[idx][0],
            y=coords[idx][1],
            tw_early=time_windows[idx][0],
            tw_late=time_windows[idx][1],
            delivery=delivery[idx],  # client idx
            pickup=pickup[idx],  # client idx
            service_duration=service[idx],
        )
        for idx in range(num_depots, num_locs)
    ]

    vehicle_types = [
        VehicleType(
            num_available=num_clients,  # one vehicle per client
            capacity=capacity,
            max_distance=max_distance,
            tw_early=time_windows[depot_idx][0],
            tw_late=time_windows[depot_idx][1],
            start_depot=depot_idx,
            end_depot=depot_idx,
        )
        for depot_idx in range(num_depots)
    ]

    if instance["open_route"]:
        # Vehicles do not need to return to the depots, so we set all arcs
        # to the depots to zero.
        matrix[:, :num_depots] = 0

    if instance["backhaul_class"] == 1:  # VRP with backhauls
        # In VRPB, linehauls must be served before backhauls. This can be
        # enforced by setting a high value for the distance/duration from depot
        # to backhaul (forcing linehaul to be served first) and a large value
        # from backhaul to linehaul (avoiding linehaul after backhaul clients).
        linehaul = np.flatnonzero(delivery > 0)
        backhaul = np.flatnonzero(pickup > 0)
        matrix[np.ix_(backhaul, linehaul)] = PYVRP_MAX_VALUE

        # Note: we remove the constraint that we cannot visit backhauls *only* in a
        # a single route as per Slack discussion
        # matrix[0, backhaul] = MAX_VALUE

    return ProblemData(clients, depots, vehicle_types, [matrix], [matrix])


def solution2action(solution: pyvrp.Solution) -> list[int]:
    """
    Converts a PyVRP solution to the action representation, i.e., a giant tour.
    Each route is represented by the location indices visited, including the
    start depot but excluding the end depot.
    """
    action = []
    for route in solution.routes():
        action.append(route.start_depot())
        action.extend(route.visits())

    return action
