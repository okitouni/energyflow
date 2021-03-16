from torch_geometric.nn import global_add_pool
from torch.nn import Module
from torch import cat
from .ptconv import PtConv

class EFN(Module):
    def __init__(self, nn):
        super().__init__()
        self.ptconv = PtConv(nn=nn, aggr='add')
    def forward(self, x, scalars, p, edge_index=None,batch=None):
        scalars = scalars.view(1,-1).expand(x.shape[0],-1)
        x = cat([x,scalars],dim=1)
        x = self.ptconv(x,edge_index)
#         x = p*x
#         x = global_add_pool(x,batch)
        #x = F.dropout(x, p=0.5, training=self.training)
        return x
