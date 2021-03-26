import os

from flow.nn.define_model import define_model
from flow.utils import readme, mkdir, Logger, ProgressBar
from flow.utils.data.processing import get_loaders
from flow import LightningModel
from flow.nn.losses import PTLoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


def train(root, seed=None, hidden_channels=128, note=None, loss='MSE', hybrid=False,
          logging=True, logdir="./lightning_logs/segmentation",
          name='experiment', epochs=20, lr=1e-4, mlp_kwargs=None, use_scalars=False,
          global_scalars=2,
          shuffle=True, val_split=0.25, n_data=None, num_workers=12, batch_size=32):
    from flow.utils.data.dataset import EICdata
    dataset = EICdata(root=root)

    if seed is not None:
        torch.manual_seed(seed)
        pl.seed_everything(seed)

    train_loader, val_loader = get_loaders(dataset, seed=seed, val_split=val_split, n_data=n_data,
                                           shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

    efn = define_model(global_scalars, hidden_channels, hybrid, mlp_kwargs, use_scalars)

    optim = torch.optim.Adam(efn.parameters(), weight_decay=0, lr=lr)
    module = LightningModel(efn, optim=optim,
                            criterion=PTLoss(loss))

    # Run
    bar = ProgressBar()
    out = None
    if logging:
        mkdir(logdir)
        log = Logger(logdir, name=name, default_hp_metric=False)
        checkpoint_path = os.path.join(logdir, f"{name}/version_{log.version}")
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, filename="weights.cpkt", save_top_k=1,
                                              monitor='val_loss', mode='min')
        callbacks = [bar, checkpoint_callback]
        if note is not None:
            readme(checkpoint_path, note)
        out = checkpoint_path
    else:
        log = True
        callbacks = [bar]
    trainer = pl.Trainer(logger=log, gpus=-1 if torch.cuda.is_available() else None,
                         max_epochs=epochs, progress_bar_refresh_rate=1,
                         callbacks=callbacks)
    trainer.fit(module, train_dataloader=train_loader, val_dataloaders=val_loader)
    return out


