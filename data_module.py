import torch
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------------
# Dataset definition
# -------------------------------------------------------------------------

class PreprocessedDataset(Dataset):
    """
    Dataset wrapper for preprocessed event-pair data.

    Each sample corresponds to a pair of event indices (i, j)
    and a target scalar cost value (e.g., distance or similarity).
    The full event embeddings/features are loaded once and indexed
    on demand.

    Expected file formats:
        - pairs_file: torch-saved tensor of shape [N, 3] → (i, j, cost)
        - events_file: torch-saved tensor of shape [num_events, feat_dim]
    """
    def __init__(self, pairs_file: str, events_file: str):
        # Load precomputed (i, j, cost) triples
        pairs = torch.load(pairs_file)  # Tensor: [N, 3]
        self.i = pairs[:, 0].long()     # indices of first event
        self.j = pairs[:, 1].long()     # indices of second event
        self.cost = pairs[:, 2].float() # target distance/similarity

        # Load event features (each row = event embedding)
        self.events = torch.load(events_file)  # Tensor: [num_events, feat_dim]

    def __len__(self):
        """Number of available event pairs."""
        return len(self.cost)

    def __getitem__(self, idx):
        """
        Retrieve one pair of events and its associated cost.

        Returns:
            ((x1, x2), cost)
            where x1, x2 are feature tensors for events i and j.
        """
        i, j = self.i[idx].item(), self.j[idx].item()
        x1, x2 = self.events[i], self.events[j]
        return (x1, x2), self.cost[idx]


# -------------------------------------------------------------------------
# DataLoader utility
# -------------------------------------------------------------------------

def get_dataloaders(train_pairs, val_pairs, test_pairs,
                    train_events, val_events, test_events,
                    batch_size=2000):
    """
    Construct PyTorch DataLoaders for train/val/test splits.

    Args:
        train_pairs, val_pairs, test_pairs: paths to torch-saved pair tensors
        train_events, val_events, test_events: paths to torch-saved event tensors
        batch_size: training batch size; validation/test use fixed 512

    Returns:
        (train_loader, val_loader, test_loader)
    """

    train_data = PreprocessedDataset(train_pairs, train_events)
    val_data   = PreprocessedDataset(val_pairs, val_events)
    test_data  = PreprocessedDataset(test_pairs, test_events)

    # Training loader: large batches, shuffle, drop_last for stability
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True,
                              persistent_workers=True, drop_last=True)

    # Validation/test loaders: deterministic order, moderate batch size
    val_loader = DataLoader(val_data, batch_size=512, shuffle=False,
                            num_workers=4, pin_memory=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False,
                             num_workers=4, pin_memory=True,
                             persistent_workers=True)

    return train_loader, val_loader, test_loader
