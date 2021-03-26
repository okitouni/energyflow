from typing import Callable, Union
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool


class PTLoss():
    def __init__(self, criterion: Union[str, Callable] = 'MSE'):
        if criterion == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'PM':
            self.criterion = pm_loss()
        else:
            if callable(criterion):
                self.criterion = criterion
            else:
                raise ValueError(f'loss {criterion} is not known.')


    def __call__(self, weights, data):
        pred = (weights * data.p)
        if type(data) == Batch:
            pred = global_add_pool(pred, data.batch)
        else:
            pred = pred.sum(axis=0, keepdim=True)
        loss = self.criterion(pred, data.y)
        loss = loss.mean()  # mean over batch
        return loss


def pm_loss(pred, target):
    m2_pred = pred[:, [0]] ** 2 - pred[:, [3]] ** 2
    #m2_target = target[:, [0]] ** 2 - target[:, [3]] ** 2
    return abs(m2_pred) + ((pred[:, 1:] - target[:, 1:])**2).sum(axis=-1, keepdim=True)
