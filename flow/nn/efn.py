from torch_geometric.nn import global_add_pool
from torch.nn import Module
from torch import cat, zeros
from .ptconv import PTConv
from torch_geometric.data.batch import Batch


class EFN(Module):
    def __init__(self, nn):
        super().__init__()
        self.ptconv = PTConv(nn=nn, aggr='add').jittable()

    def forward(self, data):
        x = data.x
        scalars = data.scalars
        edge_index = data.edge_index
        if len(scalars)>1:
            if type(data)!=Batch:
                raise ValueError(f'scalars has shape {scalars.shape} but no batch index was given.')
            scalars = scalars[data.batch]
        else:
            scalars = scalars.view(1, -1).expand(x.shape[0], -1)
      #  p = data.p
      #  if type(data) == Data:
      #      batch = zeros(len(x)).to(x.device)
      #  else:
      #      batch = data.batch
        x = cat([x, scalars], dim=1)
        x = self.ptconv(x, edge_index)
      #  x = p*x
      #  x = global_add_pool(x,batch)
        return x
