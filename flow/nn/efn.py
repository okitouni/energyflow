# noinspection PyProtectedMember
from torch_geometric.nn import global_add_pool
from torch.nn import Module
from torch import cat, zeros
from .ptconv import PTConv
from torch_geometric.data.batch import Batch
from torch import Tensor
from typing import Union


class EFN(Module):
    def __init__(self, nn):
        super().__init__()
        self.ptconv = PTConv(nn=nn, aggr='add').jittable()


class EFNtoLocal(EFN):
    def forward(self, data: Union[Batch, Tensor], extra_scalars=None):
        x = data.x
        scalars = data.scalars
        edge_index = data.edge_index
        if extra_scalars is not None:
            scalars = cat([scalars, extra_scalars], dim=-1)
        if len(scalars) > 1:
            if type(data) != Batch:
                raise ValueError(f'scalars has shape {scalars.shape} but no batch index was given.')
            scalars = scalars[data.batch]
        else:
            scalars = scalars.view(1, -1).expand(x.shape[0], -1)
        x = cat([x, scalars], dim=1)
        x = self.ptconv(x, edge_index)
        return x


class EFNtoGlobal(EFN):
    def forward(self, data):
        x = data.x
        e = data.p[:, 0]
        edge_index = data.edge_index
        if type(data) == Batch:
            batch = data.batch
        else:
            batch = zeros(len(x)).to(x.device)

        x = self.ptconv(x, edge_index)
        x = e * x
        x = global_add_pool(x, batch)
        return x


class EFNHybrid(Module):
    def __init__(self, local_nn, global_nn):
        super(EFNHybrid, self).__init__()
        self.local_nn = EFNtoLocal(nn=local_nn)
        self.global_nn = EFNtoGlobal(nn=global_nn)

    def forward(self, data):
        scalars = self.global_nn(data)
        w_i = self.local_nn(data, extra_scalars=scalars)
        return w_i
