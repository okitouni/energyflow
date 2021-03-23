import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import dense_to_sparse

from flow.utils.data.dataset import EICdata


def get_loaders(root, seed=None, val_split=0.25, n_data=None, shuffle=True, batch_size=32, num_workers=12):
    np.random.seed(seed)
    dataset = EICdata(root=root)
    if n_data is not None:
        dataset = dataset[:n_data]
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def _trim_df(df, index):
    return df[(df['STATUS'] > 0) & (df['EVENT'] == index)].iloc[1:]


def _process(df, target, scalars):
    vec4 = df[df['STATUS'] > 0][["E", "px", "py", "pz"]].iloc[1:].values
    x = vec4[:, 1:] / vec4[:, 0].reshape(-1, 1)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(target).float().view(1, 4)
    edge_index = dense_to_sparse(torch.eye(x.shape[0]))[0]  # self only
    data = Data(x=x, y=y, edge_index=edge_index)
    data.p = torch.from_numpy(vec4).float()
    data.scalars = torch.Tensor(scalars).float().view(1, 4)
    return data