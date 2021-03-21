import torch
from torch_geometric.nn import global_add_pool


class PTLoss():
    def __init__(self,criterion=None):
        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()
    def __call__(self,weights, data):
        pred = (weights * data.p)
        try:
            pred = global_add_pool(pred,data.batch)
        except:
            pred = pred.sum(axis=-2,keepdim=True)
        loss = self.criterion(pred, data.y)
        loss = loss.mean() #mean over batch
        return loss
