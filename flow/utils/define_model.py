import torch

from flow import EFNHybrid, EFNLocal
from flow.nn import MLP


def define_model(global_scalars, hidden_channels, hybrid, mlp_kwargs, use_scalars):
    if hybrid:
        nn = MLP(3 + global_scalars, **mlp_kwargs)
        nn2 = torch.nn.Sequential(torch.nn.Linear(3, hidden_channels),
                                  torch.nn.ReLU(),
                                  torch.nn.BatchNorm1d(hidden_channels),
                                  torch.nn.Linear(hidden_channels, hidden_channels),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(hidden_channels, hidden_channels),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(hidden_channels, 10),
                                  torch.nn.ReLU())

        phi = torch.nn.Sequential(torch.nn.Linear(10, hidden_channels),
                                  torch.nn.ReLU(),
                                  torch.nn.BatchNorm1d(hidden_channels),
                                  torch.nn.Linear(hidden_channels, hidden_channels),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(hidden_channels, hidden_channels),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(hidden_channels, hidden_channels),
                                  torch.nn.ReLU(),
                                  torch.nn.Dropout(0.5),
                                  torch.nn.Linear(hidden_channels, global_scalars))

        efn = EFNHybrid(local_nn=nn, global_nn=nn2, local_use_scalars=use_scalars, phi=phi)
    else:
        nn = MLP(3, **mlp_kwargs)
        efn = EFNLocal(nn=nn, use_scalars=use_scalars)
    return efn