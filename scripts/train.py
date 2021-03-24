import os
from flow.utils import readme, mkdir, Logger, ProgressBar
from flow.utils.data.processing import get_loaders
from flow import LightningModel, PTLoss, EFNHybrid, EFNLocal
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


def train(root, seed=None, hidden_channels=128, note=None, logdir="./lightning_logs/segmentation", name='experiment',
          shuffle=True, val_split=0.25, n_data=None, num_workers=12, batch_size=32):
    from flow.utils.data.dataset import EICdata
    dataset = EICdata(root=root)

    if seed is not None:
        torch.manual_seed(seed)
        pl.seed_everything(seed)

    train_loader, val_loader = get_loaders(dataset, seed=seed, val_split=val_split, n_data=n_data,
                                           shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

    nn = torch.nn.Sequential(torch.nn.Linear(3, hidden_channels),
                             torch.nn.ReLU(),
                             torch.nn.BatchNorm1d(hidden_channels),
                             torch.nn.Linear(hidden_channels, hidden_channels),
                             torch.nn.ReLU(),
                             torch.nn.Linear(hidden_channels, hidden_channels),
                             torch.nn.ReLU(),
                             torch.nn.Linear(hidden_channels, hidden_channels),
                             torch.nn.ReLU(),
                             torch.nn.Dropout(0.5),
                             torch.nn.Linear(hidden_channels, 1),
                             torch.nn.Sigmoid())

    # nn2 = torch.nn.Sequential(torch.nn.Linear(3, hidden_channels),
    #                           torch.nn.ReLU(),
    #                           torch.nn.BatchNorm1d(hidden_channels),
    #                           torch.nn.Linear(hidden_channels, hidden_channels),
    #                           torch.nn.ReLU(),
    #                           torch.nn.Linear(hidden_channels, hidden_channels),
    #                           torch.nn.ReLU(),
    #                           torch.nn.Linear(hidden_channels, hidden_channels),
    #                           torch.nn.ReLU(),
    #                           torch.nn.Dropout(0.5),
    #                           torch.nn.Linear(hidden_channels, 1),
    #                           torch.nn.Sigmoid())

    # efn = EFNHybrid(local_nn=nn, global_nn=nn2)

    efn = EFNLocal(nn=nn, scalars=False)

    optim = torch.optim.Adam(efn.parameters(), weight_decay=0, lr=1e-5)
    module = LightningModel(efn, optim=optim,
                            criterion=PTLoss(torch.nn.MSELoss()))

    # Run
    mkdir(logdir)
    log = Logger(logdir, name=name, default_hp_metric=False)
    checkpoint_path = os.path.join(logdir, f"{name}/version_{log.version}")
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, filename="weights.cpkt", save_top_k=1,
                                          monitor='val_loss', mode='min')
    bar = ProgressBar()
    trainer = pl.Trainer(logger=log, gpus=1, max_epochs=200, progress_bar_refresh_rate=1,
                         callbacks=[bar, checkpoint_callback])
    trainer.fit(module, train_dataloader=train_loader, val_dataloaders=val_loader)

    if note is not None:
        readme(checkpoint_path, note)
