import pytest

from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator
from routefinder.models import RouteFinderPolicy
from routefinder.utils import greedy_policy, rollout


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
def test_env(variant_preset):
    # Sample all variants in the same batch (Mixed-Batch Training)
    generator = MTVRPGenerator(num_loc=10, variant_preset=variant_preset)
    env = MTVRPEnv(generator, check_solution=True)
    td_data = env.generator(3)
    td_test = env.reset(td_data)
    actions = rollout(env, td_test.clone(), greedy_policy)
    rewards_nearest_neighbor = env.get_reward(td_test, actions)
    assert rewards_nearest_neighbor.shape == (3,)

    policy = RouteFinderPolicy()
    out = policy(
        td_test.clone(), env, phase="test", decode_type="greedy", return_actions=True
    )
    assert out["reward"].shape == (3,)
