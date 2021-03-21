from torch_geometric.nn import MessagePassing
from typing import Optional, Callable, Union
from torch import Tensor
from torch_geometric.typing import PairTensor, Adj

class PTConv(MessagePassing):
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)
        # size here is place holder for anything that could used in the
        # message passing step
        # should be passed with the same name to message

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(x_i)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
