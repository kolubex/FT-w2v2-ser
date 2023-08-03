import argparse
import os
import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
import pytorch_lightning as pl

from .dataloader import CustomEmoDataset
from utils.metrics import ConfusionMetrics

from pretrain.trainer import PretrainedRNNHead
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score

class DownstreamGeneral(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hp = hparams
        self.dataset = CustomEmoDataset(self.hp.datadir, self.hp.labelpath, maxseqlen=self.hp.maxseqlen)

        if self.hp.pretrained_path is not None:
            self.model = PretrainedRNNHead.load_from_checkpoint(self.hp.pretrained_path, strict=False,
                                                                n_classes=self.dataset.nemos,
                                                                backend=self.hp.model_type)
        else:
            self.model = PretrainedRNNHead(n_classes=self.dataset.nemos,
                                           backend=self.hp.model_type)
        counter = self.dataset.train_dataset.emos
        weights = torch.tensor(
            [counter[c] for c in self.dataset.emoset]
        ).float()
        weights = weights.sum() / weights
        weights = weights / weights.sum()
        print(
            f"Weigh losses by prior distribution of each class: {weights}."
        )
        self.criterion = nn.BCEWithLogitsLoss()

        # Define metrics
        # if hasattr(self.dataset, 'val_dataset'):
        #     self.valid_met = ConfusionMetrics(self.dataset.nemos)
        # if hasattr(self.dataset, 'test_dataset'):
        #     self.test_met = ConfusionMetrics(self.dataset.nemos)

    def forward(self, x, length):
        return self.model(x, length)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.trainable_params(), lr=self.hp.lr)
        return optimizer

    def train_dataloader(self):
        loader = DataLoader(dataset=self.dataset.train_dataset,
                            collate_fn=self.dataset.seqCollate,
                            batch_size=self.hp.batch_size,
                            shuffle=True,
                            num_workers=self.hp.nworkers,
                            drop_last=True)
        return loader

    def val_dataloader(self):
        if not hasattr(self.dataset, 'val_dataset'):
            return
        loader = DataLoader(dataset=self.dataset.val_dataset,
                            collate_fn=self.dataset.seqCollate,
                            batch_size=self.hp.batch_size,
                            shuffle=False,
                            num_workers=self.hp.nworkers,
                            drop_last=False)
        return loader

    def test_dataloader(self):
        if not hasattr(self.dataset, 'test_dataset'):
            return
        loader = DataLoader(dataset=self.dataset.test_dataset,
                            batch_size=1,
                            num_workers=self.hp.nworkers,
                            drop_last=False)
        return loader

    def training_step(self, batch, batch_idx):
        feats, length, label = batch
        pout = self(feats, length)
        pout = torch.sigmoid(pout)
        loss = self.criterion(pout, label)
        tqdm_dict = {'loss': loss}
        self.log_dict(tqdm_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feats, length, label = batch
        pout = self(feats, length)
        pout = torch.sigmoid(pout)
        loss = self.criterion(pout, label)
        label = label.squeeze().cpu().numpy()
        pout = pout.squeeze().cpu().numpy()
        # print(pout.shape, label.shape)
        # for index, prob in enumerate(pout):
        #     self.valid_met.fit(int(label[index]), int(1 if prob > 0.5 else 0))
        # for l, p in zip(label, pout):
        #     self.valid_met.fit(int(l), int(p.argmax()))
        self.log('valid_loss', loss, on_epoch=True, logger=True)

    # def on_validation_epoch_end(self):
    #     print (self.valid_met.uar)
    #     self.log('valid_UAR', self.valid_met.uar)
    #     self.log('valid_WAR', self.valid_met.war)
    #     self.log('valid_macroF1', self.valid_met.macroF1)
    #     self.log('valid_microF1', self.valid_met.microF1)
    #     self.valid_met.clear()

    def on_validation_epoch_end(self):
        predictions, targets,loss = self._get_predictions_and_targets(self.val_dataloader())
        mean_average_precision = self.calculate_mean_average_precision(predictions, targets)
        self.log('valid_MAP', mean_average_precision, on_epoch=True, logger=True)

        print(f"Validation MAP: {mean_average_precision:.4f}")
        # self.valid_met.clear()

    def test_step(self, batch, batch_idx):
        # feats, length, label = batch
        # pout = self(feats, length)
        # pout = torch.sigmoid(pout)
        # label = label.squeeze().cpu().numpy()
        # pout = pout.squeeze().cpu().numpy()
        # for index, prob in enumerate(pout):
        #     self.test_met.fit(int(label[index]), int(1 if prob > 0.5 else 0))
        # self.test_met.fit(int(label), int(pout.argmax()))
        pass

    # def on_test_epoch_end(self):
    #     """Report metrics."""
    #     self.log('test_UAR', self.test_met.uar, logger=True)
    #     self.log('test_WAR', self.test_met.war, logger=True)
    #     self.log('test_macroF1', self.test_met.macroF1, logger=True)
    #     self.log('test_microF1', self.test_met.microF1, logger=True)

    #     print(f"""++++ Classification Metrics ++++
    #               UAR: {self.test_met.uar:.4f}
    #               WAR: {self.test_met.war:.4f}
    #               macroF1: {self.test_met.macroF1:.4f}
    #               microF1: {self.test_met.microF1:.4f}""")
    def on_test_epoch_end(self):
        predictions, targets, loss = self._get_predictions_and_targets(self.test_dataloader())
        mean_average_precision = self.calculate_mean_average_precision(predictions, targets)
        self.log('test_MAP', mean_average_precision, on_epoch=True, logger=True)

        print(f"Test MAP: {mean_average_precision:.4f}")


    def _get_predictions_and_targets(self, dataloader):
        all_predictions = []
        all_targets = []
        loss=0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for batch in dataloader:
            print("BATCH: ", batch)
            feats, batch_idx,label = batch
            print("LABEL: ", label)
            label=label.to(device)
            length = torch.LongTensor([feats.size(1)]).to(device)
            feats = feats.to(device)
            pout = self(feats, length)
            pout = torch.sigmoid(pout)
            loss += self.criterion(pout, label)
            # print("SHAPE OF LABEL", label.shape)
            all_predictions.append(pout.detach().cpu().numpy())
            all_targets.append(label.detach().cpu().numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        return predictions, targets, loss
    
    def calculate_mean_average_precision(self, predictions, targets):
        average_precisions = []

        for label_idx in range(self.dataset.nemos):
            # print("SHAPES: ", predictions.shape, targets.shape)
            label_predictions = predictions[:, label_idx]
            label_targets = targets[:, label_idx]
            label_targets = label_targets.reshape(-1,1)

            ap = average_precision_score(label_targets, label_predictions)
            average_precisions.append(ap)

        mean_average_precision = np.mean(average_precisions)
        return mean_average_precision
