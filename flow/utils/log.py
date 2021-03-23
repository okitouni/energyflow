from typing import Union

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.tensorboard.summary import hparams


class Logger(pl.loggers.TensorBoardLogger):
    def __init__(self, save_dir: str,
                 name: Union[str, None] = 'default',
                 version: Union[int, str, None] = None,
                 log_graph: bool = False,
                 default_hp_metric: bool = True,
                 **kwargs):
        super().__init__(save_dir, name, version, log_graph, default_hp_metric, **kwargs)

    @rank_zero_only
    def log_hyperparams(self, params, metrics=None):
        # store params to output
        self.hparams.update(params)

        # format params into the suitable for tensorboard
        params = self._flatten_dict(params)
        params = self._sanitize_params(params)

        if metrics is None:
            if self._default_hp_metric:
                metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        if metrics:
            exp, ssi, sei = hparams(params, metrics)
            writer = self.experiment._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)