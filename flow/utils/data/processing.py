import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader


def get_loaders(dataset, seed=None, val_split=0.25, n_data=None, shuffle=True, batch_size=32, num_workers=12):
    np.random.seed(seed)
    if n_data is not None:
        dataset = dataset[:n_data]
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
