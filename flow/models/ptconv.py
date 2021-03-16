from torch_geometric.nn import MessagePassing
from typing import Optional, Callable, Union
from torch import Tensor
from torch_geometric.typing import PairTensor, Adj

class PtConv(MessagePassing):
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super(PtConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None,test=None)
        # test here is place holder for anything that could used in the
        # message passing step

    def message(self, x_i: Tensor, x_j: Tensor, test) -> Tensor:
        return self.nn(x_i)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
