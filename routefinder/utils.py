import torch

from rl4co.utils.ops import gather_by_index, unbatchify


# random policy for (pytest) testing
def random_policy(td):
    """Select closest available action"""
    available_actions = td["action_mask"]
    action = torch.multinomial(available_actions.float(), 1).squeeze()
    td.set("action", action)
    return td


# Simple heuristics (nearest neighbor + capacity check)
def greedy_policy(td):
    """Select closest available action"""
    available_actions = td["action_mask"]
    # distances
    curr_node = td["current_node"]
    if "cost_matrix" in td.keys():
        cost_matrix = td["cost_matrix"]
        idx_batch = torch.arange(cost_matrix.size(0))
        distances_next = cost_matrix[idx_batch, curr_node]
    else:
        loc_cur = gather_by_index(td["locs"], curr_node)
        distances_next = torch.cdist(loc_cur[:, None, :], td["locs"], p=2.0).squeeze(1)

    distances_next[~available_actions.bool()] = float("inf")
    # do not select depot if some capacity is left
    distances_next[:, 0] = float("inf") * (
        td["used_capacity_linehaul"] < td["vehicle_capacity"]
    ).float().squeeze(-1)

    # # if sum of available actions is 0, select depot
    # distances_next[available_actions.sum(-1) == 0, 0] = 0
    action = torch.argmin(distances_next, dim=-1)
    td.set("action", action)
    return td


def rollout(env, td, policy=greedy_policy, max_steps: int = None):
    """Helper function to rollout a policy. Currently, TorchRL does not allow to step
    over envs when done with `env.rollout()`. We need this because for environments that complete at different steps.
    """

    max_steps = float("inf") if max_steps is None else max_steps
    actions = []
    steps = 0

    while not td["done"].all():
        td = policy(td)
        actions.append(td["action"])
        td = env.step(td)["next"]
        steps += 1
        if steps > max_steps:
            print("Max steps reached")
            break
    return torch.stack(actions, dim=1)


def rollout_actions(env, td, actions, max_steps: int = None):
    """actions: [batch_size, num_steps]"""
    max_steps = float("inf") if max_steps is None else max_steps
    steps = 0

    # while not td["done"].all():
    num_steps = actions.size(1)

    for i in range(num_steps):
        td.set("action", actions[:, i])
        td = env.step(td)["next"]
        steps += 1
        if steps > max_steps:
            print("Max steps reached")
            break
    return td


def evaluate(
    model,
    td,
    num_augment=8,
    num_starts=None,
):
    with torch.inference_mode():
        n_start = model.env.get_num_starts(td) if num_starts is None else num_starts

        if num_augment > 1:
            td = model.augment(td)

        # Evaluate policy
        out = model.policy(
            td, model.env, phase="test", num_starts=n_start, return_actions=True
        )

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (num_augment, n_start))

        if n_start > 1:
            # max multi-start reward
            max_reward, max_idxs = reward.max(dim=-1)
            out.update({"max_reward": max_reward})

            if out.get("actions", None) is not None:
                # Reshape batch to [batch_size, num_augment, num_starts, ...]
                actions = unbatchify(out["actions"], (num_augment, n_start))
                out.update(
                    {
                        "best_multistart_actions": gather_by_index(
                            actions, max_idxs, dim=max_idxs.dim()
                        )
                    }
                )
                out["actions"] = actions

        # Get augmentation score only during inference
        if num_augment > 1:
            # If multistart is enabled, we use the best multistart rewards
            reward_ = max_reward if n_start > 1 else reward
            max_aug_reward, max_idxs = reward_.max(dim=1)
            out.update({"max_aug_reward": max_aug_reward})

            if out.get("actions", None) is not None:
                actions_ = (
                    out["best_multistart_actions"] if n_start > 1 else out["actions"]
                )
                out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

        return out
