from torch.utils.data import DataLoader
from rl4co.data.dataset import TensorDictDataset


def get_dataloader(td, batch_size=4):
    """Get a dataloader from a TensorDictDataset"""
    # Set up the dataloader
    dataloader = DataLoader(
        TensorDictDataset(td.clone()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=TensorDictDataset.collate_fn,
    )
    return dataloader