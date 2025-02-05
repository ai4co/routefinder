import torch


def get_starting_points(actions, num_depots):
    """
    Get which depot (starting point) each action in the sequence starts from
    Example:
    >>> actions = torch.tensor([[1, 10, 2, 0, 3, 30, 21, 2], [2, 15, 20, 1, 25, 30, 0, 1]])
    >>> get_starting_points(actions, 3) -> torch.tensor([[1, 1, 2, 0, 0, 0, 0, 2], [2, 2, 2, 1, 1, 1, 0, 1]])
    """
    # Create mask for numbers < num_depots
    mask = actions < num_depots  # shape: (batch_size, seq_len)
    batch_size, seq_len = actions.shape

    # Compute the cumulative sum of the mask to get segment IDs
    segment_ids = torch.cumsum(mask.long(), dim=1)  # shape: (batch_size, seq_len)

    # Adjust segment IDs for indexing (shift by -1)
    segment_indices = segment_ids - 1

    # Create a mask for valid segment positions
    valid_positions = segment_ids > 0

    # Compute the number of masked elements per batch
    num_values_per_batch = mask.sum(dim=1)  # shape: (batch_size,)
    max_num_values = num_values_per_batch.max().item()

    # Generate batch indices
    batch_indices = (
        torch.arange(batch_size, device=actions.device)
        .unsqueeze(1)
        .expand(batch_size, seq_len)
    )

    # Get indices where mask is True
    masked_indices = torch.where(
        mask,
        torch.cumsum(mask.long(), dim=1) - 1,
        torch.tensor(-1, device=actions.device),
    )
    valid_masked_positions = masked_indices >= 0

    # Gather valid batch and masked indices
    valid_batch_indices = batch_indices[valid_masked_positions]
    valid_masked_indices = masked_indices[valid_masked_positions]
    valid_actions = actions[valid_masked_positions]

    # Initialize padded values tensor
    values_padded = torch.zeros(
        batch_size, max_num_values, dtype=actions.dtype, device=actions.device
    )

    # Fill in the padded values tensor
    values_padded[valid_batch_indices, valid_masked_indices] = valid_actions

    # Initialize the starting_points tensor
    starting_points = torch.zeros_like(actions)

    # Fill in the starting_points tensor using advanced indexing
    starting_points[valid_positions] = values_padded[
        batch_indices[valid_positions], segment_indices[valid_positions]
    ]

    return starting_points


if __name__ == "__main__":
    # Test
    batch = torch.tensor([[1, 10, 2, 0, 3, 30, 21, 2], [2, 15, 20, 2, 25, 30, 0, 1]])
    result = get_starting_points(batch, 3)
    print(result)
