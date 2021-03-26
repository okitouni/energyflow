import torch
from torch_geometric.data import InMemoryDataset, Data
from pandas import read_hdf
from os.path import join
import numpy as np

from torch_geometric.utils import dense_to_sparse


class EICdata(InMemoryDataset):
    def __init__(self, root='./data/EIC/', transform=None, pre_transform=None):
        super(EICdata, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['breit0.h5', 'targets_breit0.h5', 'scalars0.h5']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass  # Download to `self.raw`.

    def process(self):
        # Read data into huge `Data` list.
        events_breit = read_hdf(join(self.raw_dir, 'breit.h5'), key=f'data')
        targets = read_hdf(join(self.raw_dir, 'targets_breit.h5'), key='data').values
        scalars = read_hdf(join(self.raw_dir, 'scalars.h5'), key='data').values
        data_list = [_process(events_breit[events_breit['EVENT'] == i], y, z) for i, y, z in
                     zip(range(len(targets)), targets, scalars)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    __all__ = ['EICdata']


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


def _trim_df(df, index=None):
    if index is None:
        # need to remove outgoing hard scattering electron
        # mask_first = df.groupby(['EVENT'])['EVENT'].transform(_mask_first).astype(bool)
        # return df[(df['STATUS'] > 0)].groupby(['EVENT']).apply(lambda x: x[1:])
        return df[(df['STATUS'] > 0) & (df['STATUS'] != 44) & (df['N'] != 6)]
    return df[(df['STATUS'] > 0) & (df['EVENT'] == index)].iloc[1:]

# def _mask_first(x):
#     result = np.ones_like(x)
#     result[0] = 0
#     return result
