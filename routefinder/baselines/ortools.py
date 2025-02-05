from dataclasses import dataclass
from typing import Optional

import numpy as np

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from tensordict import TensorDict
from torch import Tensor

import routefinder.baselines.pyvrp as pyvrp

from .constants import ORTOOLS_SCALING_FACTOR


def solve(instance: TensorDict, max_runtime: float, **kwargs) -> tuple[Tensor, Tensor]:
    """
    Solves an AnyVRP instance with OR-Tools.

    Parameters
    ----------
    instance
        The AnyVRP instance to solve.
    max_runtime
        The maximum runtime for the solver.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple consisting of the action and the cost, respectively.

    Notes
    -----
    This function depends on PyVRP's data converter to convert the AnyVRP
    instance to an OR-Tools compatible format. Future versions should
    implement a direct conversion.
    """
    data = instance2data(instance)
    action, cost = _solve(data, max_runtime)
    cost /= ORTOOLS_SCALING_FACTOR
    cost *= -1

    return action, cost


@dataclass
class ORToolsData:
    """
    Convenient dataclass for instance data when using OR-Tools as solver.

    Parameters
    ----------
    depots
        The depot indices.
    distance_matrix
        The distance matrix between locations.
    duration_matrix
        The duration matrix between locations. This includes service times.
    vehicle_capacities
        The capacity of each vehicle.
    vehicle_start_depots
        The start depot of each vehicle.
    vehicle_end_depots
        The end depot of each vehicle.
    vehicle_tw_early
        The early time window of each vehicle.
    vehicle_tw_late
        The late time window of each vehicle.
    max_distance
        The maximum distance a vehicle can travel.
    demands
        The demands of each location.
    time_windows
        The time windows for each location. Optional.
    backhauls
        The pickup quantity for backhaul at each location.
    """

    depots: list[int]
    distance_matrix: list[list[int]]
    duration_matrix: list[list[int]]
    vehicle_capacities: list[int]
    vehicle_start_depots: list[int]
    vehicle_end_depots: list[int]
    vehicle_tw_early: list[int]
    vehicle_tw_late: list[int]
    max_distance: int
    demands: list[int]
    time_windows: Optional[list[list[int]]]
    backhauls: Optional[list[int]]

    @property
    def num_locations(self) -> int:
        return len(self.distance_matrix)

    @property
    def num_depots(self) -> int:
        return len(self.depots)

    @property
    def num_clients(self) -> int:
        return self.num_locations - self.num_depots

    @property
    def num_vehicles(self) -> int:
        return len(self.vehicle_capacities)


def instance2data(instance: TensorDict) -> ORToolsData:
    """
    Converts an AnyVRP instance to an ORToolsData instance.
    """
    # TODO: Do not use PyVRP's data converter.
    data = pyvrp.instance2data(instance, ORTOOLS_SCALING_FACTOR)

    capacities = [
        veh_type.capacity
        for veh_type in data.vehicle_types()
        for _ in range(veh_type.num_available)
    ]
    start_depots = [
        veh_type.start_depot
        for veh_type in data.vehicle_types()
        for _ in range(veh_type.num_available)
    ]
    end_depots = [
        veh_type.end_depot
        for veh_type in data.vehicle_types()
        for _ in range(veh_type.num_available)
    ]
    veh_tw_early = [
        veh_type.tw_early
        for veh_type in data.vehicle_types()
        for _ in range(veh_type.num_available)
    ]
    veh_tw_late = [
        veh_type.tw_late
        for veh_type in data.vehicle_types()
        for _ in range(veh_type.num_available)
    ]
    max_distance = data.vehicle_type(0).max_distance

    padding = [0] * data.num_depots
    demands = padding + [client.delivery for client in data.clients()]
    backhauls = padding + [client.pickup for client in data.clients()]
    service = padding + [client.service_duration for client in data.clients()]

    tws = [[0, np.iinfo(np.int64).max]] * data.num_depots  # padding
    tws += [[client.tw_early, client.tw_late] for client in data.clients()]

    # Set data to None if instance does not contain explicit values.
    default_tw = [0, np.iinfo(np.int64).max]
    if all(tw == default_tw for tw in tws):
        tws = None  # type: ignore

    if all(val == 0 for val in backhauls):
        backhauls = None  # type: ignore

    distances = data.distance_matrices()[0].copy()
    durations = np.array(distances) + np.array(service)[:, np.newaxis]

    if backhauls is not None:
        # Serve linehauls before backhauls.
        linehaul = np.flatnonzero(np.array(demands) > 0)
        backhaul = np.flatnonzero(np.array(backhauls) > 0)
        distances[np.ix_(backhaul, linehaul)] = max_distance

    return ORToolsData(
        depots=list(range(data.num_depots)),
        distance_matrix=distances.tolist(),
        duration_matrix=durations.tolist(),
        vehicle_capacities=capacities,
        vehicle_start_depots=start_depots,
        vehicle_end_depots=end_depots,
        vehicle_tw_early=veh_tw_early,
        vehicle_tw_late=veh_tw_late,
        demands=demands,
        time_windows=tws,
        max_distance=max_distance,
        backhauls=backhauls,
    )


def _solve(data: ORToolsData, max_runtime: float, log: bool = False):
    """
    Solves an instance with OR-Tools.

    Parameters
    ----------
    data
        The instance data.
    max_runtime
        The maximum runtime in seconds.
    log
        Whether to log the search.

    Returns
    -------
    tuple[list[list[int]], int]
        A tuple containing the routes and the objective value.
    """
    # Manager for converting between nodes (location indices) and index
    # (internal CP variable indices).
    manager = pywrapcp.RoutingIndexManager(
        data.num_locations,
        data.num_vehicles,
        data.vehicle_start_depots,
        data.vehicle_end_depots,
    )
    routing = pywrapcp.RoutingModel(manager)

    # Set arc costs equal to distances.
    distance_transit_idx = routing.RegisterTransitMatrix(data.distance_matrix)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_transit_idx)

    # Max distance constraint.
    routing.AddDimension(
        distance_transit_idx,
        0,  # null distance slack
        data.max_distance,  # maximum distance per vehicle
        True,  # start cumul at zero
        "Distance",
    )

    # Vehicle capacity constraint.
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitVector(data.demands),
        0,  # null capacity slack
        data.vehicle_capacities,  # vehicle maximum capacities
        True,  # start cumul to zero
        "Demand",
    )

    # Backhauls: this assumes that VRPB is implemented by forbidding arcs
    # that go from backhauls to linehauls.
    if data.backhauls is not None:
        routing.AddDimensionWithVehicleCapacity(
            routing.RegisterUnaryTransitVector(data.backhauls),
            0,  # null capacity slack
            data.vehicle_capacities,  # vehicle maximum capacities
            True,  # start cumul to zero
            "Backhaul",
        )

    # Time window constraints.
    if data.time_windows is not None:
        time_ub = np.array(data.time_windows).max()
        time_ub = max(time_ub, np.array(data.vehicle_tw_late).max())
        time_ub = int(time_ub)

        # The depot's late time window is a valid upper bound for the waiting
        # time and maximum duration per vehicle.
        routing.AddDimension(
            routing.RegisterTransitMatrix(data.duration_matrix),
            time_ub,  # waiting time upper bound
            time_ub,  # maximum duration per vehicle
            False,  # don't force start cumul to zero
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        # Add time window constraints for each client.
        for node, (tw_early, tw_late) in enumerate(data.time_windows):
            if node in data.depots:
                continue

            index = manager.NodeToIndex(node)
            time_dim.CumulVar(index).SetRange(tw_early, tw_late)

        # Add time window constraints for each vehicle start node.
        for node in range(data.num_vehicles):
            start = routing.Start(node)
            time_dim.CumulVar(start).SetRange(
                data.vehicle_tw_early[node],
                data.vehicle_tw_late[node],
            )

        for node in range(data.num_vehicles):
            cumul_start = time_dim.CumulVar(routing.Start(node))
            routing.AddVariableMinimizedByFinalizer(cumul_start)

            cumul_end = time_dim.CumulVar(routing.End(node))
            routing.AddVariableMinimizedByFinalizer(cumul_end)

    # Setup search parameters.
    params = pywrapcp.DefaultRoutingSearchParameters()

    gls = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.local_search_metaheuristic = gls

    params.time_limit.FromSeconds(int(max_runtime))  # only accepts int
    params.log_search = log

    solution = routing.SolveWithParameters(params)
    action = solution2action(data, manager, routing, solution)
    objective = solution.ObjectiveValue()

    return action, objective


def solution2action(data, manager, routing, solution) -> list[list[int]]:
    """
    Converts an OR-Tools solution to the action representation, i.e., a giant tour.
    Each route is represented by the location indices visited, including the
    start depot but excluding the end depot.
    """
    routes = []
    distance = 0  # for debugging

    for vehicle_idx in range(data.num_vehicles):
        index = routing.Start(vehicle_idx)
        route = []  # includes start depot
        route_cost = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)

            prev_index = index
            index = solution.Value(routing.NextVar(index))
            route_cost += routing.GetArcCostForVehicle(prev_index, index, vehicle_idx)

        if len(route) > 1:  # at least one client
            routes.append(route)
            distance += route_cost

    return [visit for route in routes for visit in route]
