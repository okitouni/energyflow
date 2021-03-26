import argparse
from flow.utils.data.dataset import EICdata
from flow.utils.data.processing import get_loaders
import torch
import optuna
import os
from optuna.integration import PyTorchLightningPruningCallback
from flow import LightningModel, PTLoss
import pytorch_lightning as pl
import logging

from flow.nn.define_model import define_model

PERCENT_VALID_EXAMPLES = 0.25
BATCHSIZE = 128
NDATA = 5000
EPOCHS = 10
SEED = 0
DIR = os.getcwd()
HYBRID = False
GLOBAL_SCALARS = 2
USE_SCALARS = False
SIGMOID = True
LOSS = torch.nn.MSELoss()

torch.manual_seed(SEED)
pl.seed_everything(SEED)


def objective(trial: optuna.trial.Trial) -> float:
    # Load data
    dataset = EICdata(root='~/projects/Geometric-HEP/pythia-gen/data/EIC')

    train_loader, val_loader = get_loaders(dataset, seed=SEED, val_split=PERCENT_VALID_EXAMPLES, n_data=NDATA,
                                           shuffle=True, batch_size=BATCHSIZE, num_workers=12)

    # We optimize the number of layers, hidden units in each layer and dropouts.
    n_layers = trial.suggest_int("n_layers", 1, 5)
    # batch_norm_layer = trial.suggest_int("batch_norm_layer", 0, n_layers - 1)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    output_dims = [
        trial.suggest_int("n_units_l{}".format(i), 64, 128, log=True) for i in range(n_layers)
    ]

    hidden_channels = trial.suggest_int("hidden_channels", 64, 128, log=True) if HYBRID else 1

    mlp_kwargs = {'dropout': dropout, 'output_dims': output_dims, 'sigmoid': SIGMOID}
    efn = define_model(GLOBAL_SCALARS, hidden_channels, HYBRID, mlp_kwargs, USE_SCALARS)

    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    # weight_decay = trial.suggest_float("weight_decay", 0, 1e-2, log=True)
    optimizer = torch.optim.Adam(efn.parameters(), weight_decay=0, lr=lr)

    module = LightningModel(efn, optim=optimizer,
                            criterion=PTLoss(LOSS))

    # Run
    trainer = pl.Trainer(logger=True, gpus=-1 if torch.cuda.is_available() else None,
                         max_epochs=EPOCHS,
                         progress_bar_refresh_rate=0,
                         weights_summary=None,
                         checkpoint_callback=False,
                         callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")]
                         )

    hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(module, train_dataloader=train_loader, val_dataloaders=val_loader)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    logging.getLogger('lightning').setLevel(0)
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
             "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
