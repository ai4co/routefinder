from functools import partial
from multiprocessing import Pool

from tensordict.tensordict import TensorDict
from torch import Tensor


class NoSolver:
    def solve(self, *args, **kwargs):
        pass

try:
    import routefinder.baselines.pyvrp as pyvrp
except ImportError:
    pyvrp = NoSolver()
try:
    import routefinder.baselines.lkh as lkh
except ImportError:
    lkh = NoSolver()
try:
    import routefinder.baselines.ortools as ortools
except ImportError:
    ortools = NoSolver()

from .utils import mtvrp2anyvrp


def solve(
    instances: TensorDict,
    max_runtime: float,
    num_procs: int = 1,
    data_type: str = "mtvrp",
    solver: str = "pyvrp",
    **kwargs,
) -> tuple[Tensor, Tensor]:
    """
    Solves the AnyVRP instances with PyVRP.

    Parameters
    ----------
    instances
        TensorDict containing the AnyVRP instances to solve.
    max_runtime
        Maximum runtime for the solver.
    num_procs
        Number of processers to use to solve instances in parallel.
    data_type
        Environment mode. If "mtvrp", the instance data will be converted first.
    solver
        The solver to use. One of ["pyvrp", "ortools", "lkh"].

    Returns
    -------
    tuple[Tensor, Tensor]
        A Tensor containing the actions for each instance and a Tensor
        containing the corresponding costs.
    """
    if data_type == "mtvrp":
        instances = mtvrp2anyvrp(instances)
        
    if solver == "pyvrp" and isinstance(pyvrp, NoSolver):
        raise ImportError("PyVRP is not installed. Please install it using `pip install -e .[solvers]`.")
    if solver == "lkh" and isinstance(lkh, NoSolver):
        raise ImportError("LKH is not installed. Please install it using `pip install -e .[solvers]`")
    if solver == "ortools" and isinstance(ortools, NoSolver):
        raise ImportError("OR-Tools is not installed. Please install it using `pip install -e .[solvers]`.")

    solvers = {"pyvrp": pyvrp.solve, "ortools": ortools.solve, "lkh": lkh.solve}
    if solver not in solvers:
        raise ValueError(f"Unknown baseline solver: {solver}")

    _solve = solvers[solver]
    func = partial(_solve, max_runtime=max_runtime, **kwargs)

    if num_procs > 1:
        with Pool(processes=num_procs) as pool:
            results = pool.map(func, instances)
    else:
        results = [func(instance) for instance in instances]

    actions, costs = zip(*results)

    # Pad to ensure all actions have the same length.
    max_len = max(len(action) for action in actions)
    actions = [action + [0] * (max_len - len(action)) for action in actions]

    return Tensor(actions).long(), Tensor(costs)
