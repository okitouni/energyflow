import torch
import pytorch_lightning as pl

class LightningModel(pl.LightningModule):
    def __init__(self, model, criterion=None, lr=1e-3, optim=None,data_cov=None,zeroparams=None):
        super().__init__()
        self.model = model
        self.Loss = criterion if criterion is not None else torch.nn.MSELoss() 
        self.optim = optim
        self.zeroparams = zeroparams
        if optim is None:
            self.lr = lr
        else:
            self.lr = self.optim.defaults["lr"]
        self.data_cov = data_cov

        self.hparams["params"] = sum([x.size().numel()
                                      for x in self.model.parameters()])
        self.hparams["loss"] = self.Loss
        self.hparams["optim"] = optim.__repr__().replace("\n","")
        self.hparams["model"] = self.model.__repr__().replace("\n", "")
        self.hparams["lr"] = self.lr

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        yhat = self(batch)
        loss = self.Loss(yhat, batch)
        self.log('train_loss', loss)
        if self.zeroparams is not None:
            self.log("zero_params", sum([torch.sum(abs(g)<self.zeroparams)  for g in self.model.parameters()]))
        return loss

    def validation_step(self, batch, batch_idx):
        yhat = self(batch)
        loss = self.Loss(yhat, batch)
        #acc = accuracy(preds, y)
        # Calling self.log will surface up scalars for you in TensorBoard
        metrics = {'val_loss': loss} #, 'val_acc': acc}
        self.log_dict(metrics, prog_bar=True, logger=True,
                      on_epoch=True, on_step=False)
        try:
            self.logger.log_hyperparams(self.hparams, metrics=metrics)
        except:
            self.logger.log_hyperparams(self.hparams)
        return metrics

    def validation_epoch_end(self, outputs):
        val_loss_mean = 0
        #val_acc_mean = 0
        for output in outputs:
            val_loss_mean += output['val_loss']
            #val_acc_mean += output['val_acc']
        val_loss_mean /= len(outputs)
        #val_acc_mean /= len(outputs)
        metrics = {'val_loss': val_loss_mean}#, 'val_acc': val_acc_mean}

        if self.zeroparams is not None:
            metrics["zero_params"] = sum([torch.sum(abs(g)<self.zeroparams)  for g in self.model.parameters()])
        self.log_dict(metrics, prog_bar=True,logger=True,on_epoch=True,on_step=False)
        self.logger.log_hyperparams(self.hparams,metrics=metrics)
        return

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self,learning_rate=1e-3):
        if self.optim is not None:
            optimizer = self.optim
            for g in optimizer.param_groups:
                g["lr"] = learning_rate
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

