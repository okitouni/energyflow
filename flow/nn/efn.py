# noinspection PyProtectedMember
from torch_geometric.nn import global_add_pool
from torch.nn import Module
from torch import cat, zeros, Tensor, sigmoid
from .ptconv import PTConv
from torch_geometric.data.batch import Batch
from typing import Union


class EFN(Module):
    def __init__(self, nn):
        super().__init__()
        self.ptconv = PTConv(nn=nn, aggr='add').jittable()


# TODO
# Test 1,2,3,5 output global EFN
# Compare against no scalars
# Compare with scalars only.
# Implement no scalars
# LocalQ2xextra2: Local netwrok uses Q2
# globalwLIextra2: Global NN uses all LI and Local uses 2 extra only
class EFNLocal(EFN):
    def __init__(self, nn, use_scalars=False):
        super().__init__(nn)
        self.use_scalars = use_scalars

    def forward(self, data: Union[Batch, Tensor], extra_scalars=None):
        if self.use_scalars:
            scalars = data.scalars
            #scalars = scalars[:, [0, 1]]
        else:
            scalars = Tensor([]).to(data.x.device)
        if extra_scalars is not None:
            scalars = cat([scalars, extra_scalars], dim=-1)
        scalars = _reshape_scalars(scalars, data)
        x = _combine_feature_scalars(data.x, scalars)
        x = self.ptconv(x, data.edge_index)
        return x


class EFNGlobal(EFN):
    def __init__(self, nn, use_scalars=False, phi=None):
        super().__init__(nn)
        self.use_scalars = use_scalars
        self.phi = phi

    def forward(self, data):
        x = data.x
        if self.use_scalars:
            scalars = _reshape_scalars(data.scalars, data)
            x = _combine_feature_scalars(x, scalars)

        e = data.p[:, 0].view(-1, 1)
        edge_index = data.edge_index
        if type(data) == Batch:
            batch = data.batch
        else:
            batch = zeros(len(x)).long().to(x.device)

        x = self.ptconv(x, edge_index)
        x = e * x
        x = global_add_pool(x, batch)
        if self.phi is not None:
            x = self.phi(x)
        return x


class EFNHybrid(Module):
    def __init__(self, local_nn, global_nn, local_use_scalars=False, global_use_scalars=False, phi=None):
        super().__init__()
        self.local_nn = EFNLocal(nn=local_nn, use_scalars=local_use_scalars)
        self.global_nn = EFNGlobal(nn=global_nn, use_scalars=global_use_scalars, phi=phi)

    def forward(self, data):
        scalars = self.global_nn(data)
        w_i = self.local_nn(data, extra_scalars=scalars)
        return w_i


def _reshape_scalars(scalars, data):
    if len(scalars) > 1:
        if type(data) != Batch:
            raise ValueError(f'scalars has shape {scalars.shape} but no batch index was given.')
        scalars = scalars[data.batch]
    else:
        scalars = scalars.view(1, -1).expand(data.x.shape[0], -1)
    return scalars


def _combine_feature_scalars(x, scalars):
    return cat([x, scalars], dim=-1)