from typing import Any, List, Optional
import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from src.model.modules.vgg_base import VGG16NetClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

class ClassifierModel(pl.LightningModule):
    def __init__(
        self, 
        pretrained : bool,
        freeze_features : bool,
        num_classes : int,
        optimizer : str,
        learning_rate : Optional[float],
        beta_1 : Optional[float],
        beta_2 : Optional[float],
        momentum : Optional[float],
        weight_decay : Optional[float],
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.model = VGG16NetClassifier(
            pretrained = pretrained,
            freeze_features = freeze_features,
            num_classes = num_classes
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)

    def step(self, batch: Any):
        data, target = batch
        logits = self.forward(data)
        loss = self.criterion(logits, target)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, target

    def training_step(self, batch: Any, batch_idx: int):        
        loss, preds, target = self.step(batch)
        acc = accuracy_score(preds.cpu().numpy(), target.cpu().numpy())
        prec = precision_score(preds.cpu().numpy(), target.cpu().numpy(), average = 'micro')
        rec = recall_score(preds.cpu().numpy(), target.cpu().numpy(), average = 'micro')
        f1 = f1_score(preds.cpu().numpy(), target.cpu().numpy(), average = 'micro')

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_prec", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_rec", rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}
    
    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, target = self.step(batch)
        acc = accuracy_score(preds.cpu().numpy(), target.cpu().numpy())
        prec = precision_score(preds.cpu().numpy(), target.cpu().numpy(), average = 'micro')
        rec = recall_score(preds.cpu().numpy(), target.cpu().numpy(), average = 'micro')
        f1 = f1_score(preds.cpu().numpy(), target.cpu().numpy(), average = 'micro')

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_prec", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rec", rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass
        
    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, target = self.step(batch)
        acc = accuracy_score(preds.cpu().numpy(), target.cpu().numpy())
        prec = precision_score(preds.cpu().numpy(), target.cpu().numpy(), average = 'micro')
        rec = recall_score(preds.cpu().numpy(), target.cpu().numpy(), average = 'micro')
        f1 = f1_score(preds.cpu().numpy(), target.cpu().numpy(), average = 'micro')

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_prec", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_rec", rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}
    
    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        if self.hparams['optimizer'] == 'adam':
            return torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.hparams['learning_rate'],
                betas=(self.hparams['beta_1'],self.hparams['beta_2']),
                weight_decay=self.hparams['weight_decay'],
            )

        elif self.hparams['optimizer'] == 'sgd':
            return torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.hparams['learning_rate'],
                momentum=self.hparams['momentum'],
                weight_decay=self.hparams['weight_decay'],
            )

#
            