import torch
from torch_geometric.data import InMemoryDataset
from flow.utils.data.processing import _process
from pandas import read_hdf
from os.path import join


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
        events_breit = read_hdf(join(self.raw_dir, '/breit0.h5'), key=f'data')
        targets = read_hdf(join(self.raw_dir, '/targets_breit0.h5'), key='data').values
        scalars = read_hdf(join(self.raw_dir, '/scalars0.h5'), key='data').values
        data_list = [_process(events_breit[events_breit['EVENT'] == i], y, z) for i, y, z in
                     zip(range(len(targets)), targets, scalars)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    __all__ = ['EICdata']
