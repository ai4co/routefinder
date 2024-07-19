import pytest

from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator
from routefinder.models import RouteFinderPolicy


@pytest.mark.parametrize(
    "variant_preset",
    [
        "all",
        "single_feat",
        "single_feat_otw",
        "cvrp",
        "ovrp",
        "vrpb",
        "vrpl",
        "vrptw",
        "ovrptw",
        "ovrpb",
        "ovrpl",
        "vrpbl",
        "vrpbtw",
        "vrpltw",
        "ovrpbl",
        "ovrpbtw",
        "ovrpltw",
        "vrpbltw",
        "ovrpbltw",
    ],
)
def test_policy(variant_preset):
    # Sample all variants in the same batch (Mixed-Batch Training)
    generator = MTVRPGenerator(num_loc=10, variant_preset=variant_preset)
    env = MTVRPEnv(generator, check_solution=True)
    td_data = env.generator(3)
    td_test = env.reset(td_data)
    policy = RouteFinderPolicy()
    out = policy(
        td_test.clone(), env, phase="test", decode_type="greedy", return_actions=True
    )
    assert out["reward"].shape == (3,)
