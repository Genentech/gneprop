from descriptastorus.descriptors import rdNormalizedDescriptors  # needs to be first import

import joblib
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import pytorch_lightning as pl
import random
import ray
import sys
import torch
import torch.nn.functional as F
import warnings
from argparse import ArgumentParser
from ax.service.ax_client import AxClient
from ax.service.managed_loop import optimize
from enum import Enum
from functools import partial
from omegaconf import OmegaConf
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data import auto_move_data
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from re import match
from sklearn.utils import compute_class_weight
from torch import nn
from torch_geometric.loader import DataLoader
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import mean_absolute_error, mean_squared_error, r2_score, explained_variance
from torchmetrics import F1Score, Precision, Recall, AUROC, Accuracy, AveragePrecision
# from torchmetrics.functional import accuracy
# from torchmetrics.functional import mean_absolute_error, f1, mean_squared_error, r2_score, explained_variance
from torchmetrics.functional.classification import auroc
# from torchmetrics.functional.classification.average_precision import average_precision
from tqdm import tqdm
from typing import Union
import learn2learn as l2l
from collections import defaultdict
import clr
import gneprop.utils
from data import MolDatasetOD, MoleculeSubset, convert_smiles_to_mols, load_dataset_multi_format, AugmentedDataset, \
    split_data, MolDatasetODBaseline, ConcatMolDataset, preprocess_output_data
from data import add_mol_features
from gneprop.augmentations import AugmentationFactory
from gneprop.custom_objects import ModelCheckpointSkipFirst, GradualWarmupScheduler
from gneprop.options import GNEPropTask
from gneprop.plot_utils import plot_confusion_matrix_as_torch_tensor
from gneprop.scaffold import scaffold_to_smiles
from gneprop.utils import aggregate_metrics, cast_scalar, describe_data, print_save_aggregated_metrics, timeit, \
    get_time_string, compute_metrics, get_accelerator, read_csv, clean_local_log, sync_s3, prepare_pretrain_file
from models import GNEpropGIN, SupervisedContrastiveLoss
from data import ACDataset


def get_loaders(dataset, split_type='random', sizes=(0.8, 0.1, 0.1), seed=0, train_batch_size=50, test_batch_size=500,
                print_description=False, num_workers=1, additional_training_data=None, balanced_batch_training=False,
                ig_baseline_ratio=0., augmentation=None, augmentation_label='same',
                task=GNEPropTask.BINARY_CLASSIFICATION, drop_last=True, static_val_augmentation=None, val_multiplier=1,
                output_preprocess=None, meta=False, meta_test='val', add_ac=False, ac_skip=-1.):
    train, val, test = split_data(dataset, split_type=split_type, sizes=sizes, seed=seed,
                                  additional_training_data=additional_training_data)

    output_pp_statistics = dict()
    if output_preprocess is not None:
        if task is GNEPropTask.REGRESSION:
            train = train.to_dataset()

            train, output_pp_statistics = preprocess_output_data(train, preprocess_type=output_preprocess)

    if ig_baseline_ratio > 0:  # IG
        if task is GNEPropTask.BINARY_CLASSIFICATION:
            ig_baseline_dataset = MolDatasetODBaseline.sample_baseline_dataset(train, frac=ig_baseline_ratio)
        elif task is GNEPropTask.MULTI_CLASSIFICATION:
            ig_baseline_dataset = MolDatasetODBaseline.sample_baseline_dataset_multiclass(train, frac=ig_baseline_ratio)
        else:
            raise ValueError

        train = ConcatMolDataset((train, ig_baseline_dataset))

    if augmentation is not None:
        train = AugmentedDataset(dataset=train, aug=augmentation, aug_behavior=augmentation_label)
        print('Using augmentation: ')
        print(augmentation)

    prefetch_factor = 20
    pin_memory = False

    if balanced_batch_training:
        from torchsampler import ImbalancedDatasetSampler

        train_loader = DataLoader(train, batch_size=train_batch_size,
                                  sampler=ImbalancedDatasetSampler(train, callback_get_label=lambda d: d.y),
                                  num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                                  drop_last=drop_last)
    else:
        train_loader = DataLoader(train, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory, drop_last=drop_last)

    if val_multiplier > 1:
        val = ConcatMolDataset((val,) * val_multiplier)

    if static_val_augmentation is not None:
        val = AugmentedDataset(dataset=val, aug=static_val_augmentation, aug_behavior='same', static=True)

    if print_description:
        describe_data(dataset, train, val, test, task=task)

    val_loader = DataLoader(val, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)

    if meta:
        if meta_test == 'val':
            meta_test = DataLoader(val, batch_size=train_batch_size, num_workers=num_workers, pin_memory=pin_memory)

            train_loader = {'train': train_loader, 'meta_test': meta_test}
            val_loader = DataLoader(test, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
        elif meta_test == 'half_val':
            val = val.to_dataset()
            meta_test, val, _ = split_data(val, split_type='scaffold', sizes=(0.5, 0.5, 0), seed=0,)

            meta_test_loader = DataLoader(meta_test, batch_size=train_batch_size, num_workers=num_workers, pin_memory=pin_memory)
            train_loader = {'train': train_loader, 'meta_test': meta_test_loader}

            val_loader = DataLoader(val, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
        else:
            raise NotImplementedError()
    else:
        train_loader = {'train': train_loader}

    if add_ac:
        pass

    return train_loader, val_loader, test_loader, output_pp_statistics


class GNEprop(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self._add_default_args(kwargs)  # backcompatibility, add defaults
        self.save_hyperparameters()

        gmt_args = self._get_gmt_args()

        self.mpn_layers = GNEpropGIN(in_channels=self.hparams.node_feat_size,
                                     edge_dim=self.hparams.edge_feat_size,
                                     hidden_channels=self.hparams.hidden_size,
                                     ffn_hidden_channels=self.hparams.ffn_hidden_size,
                                     num_layers=self.hparams.depth,
                                     out_channels=self.hparams.out_channels, dropout=self.hparams.dropout,
                                     num_readout_layers=self.hparams.num_readout_layers,
                                     mol_features_size=self.hparams.mol_features_size,
                                     aggr=self.hparams.aggr,
                                     jk=self.hparams.jk,
                                     gmt_args=gmt_args,
                                     use_proj_head=self.hparams.use_proj_head,
                                     proj_dims=(256, 64),
                                     skip_last_relu=self.hparams.skip_last_relu,)

        self.task = GNEPropTask(self.hparams.task)

        self.loss_function = self.get_loss_func()

        self.reset_best_val_metric()

        self.name_best_val_metric = "best_" + self.hparams.metric

        self._set_automatic_optimization()

        self.compute_representations = False

        self.supcon_loss = SupervisedContrastiveLoss()

    def _get_gmt_args(self):
        gmt_args = dict()
        gmt_args['hidden_channels'] = self.hparams.gmt_hidden_channels
        gmt_args['gmt_pooling_ratio'] = self.hparams.gmt_pooling_ratio
        gmt_args['gmt_num_heads'] = self.hparams.gmt_num_heads
        gmt_args['gmt_layer_norm'] = self.hparams.gmt_layer_norm
        gmt_args['gmt_sequence'] = self.hparams.gmt_sequence
        return gmt_args

    def _set_automatic_optimization(self):
        automatic_optimization = self.hparams.adv == 'none' and not self.hparams.meta

        self.automatic_optimization = automatic_optimization

    def _add_default_args(self, kwargs):
        model_parser = self.add_model_specific_args(ArgumentParser())
        default_args = model_parser.parse_known_args()[0]._get_kwargs()
        for k, v in default_args:
            if k not in kwargs:
                kwargs[k] = v

    def get_loss_func(self):
        if self.task is GNEPropTask.BINARY_CLASSIFICATION:
            if self.hparams.pos_weight != 1:
                loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.hparams.pos_weight))
            else:
                loss_function = nn.BCEWithLogitsLoss()

        elif self.task is GNEPropTask.REGRESSION:
            loss_function = nn.MSELoss()
        elif self.task is GNEPropTask.MULTI_CLASSIFICATION:
            if self.hparams.class_weight is None:
                loss_function = nn.CrossEntropyLoss()
            else:
                loss_function = nn.CrossEntropyLoss(weight=torch.tensor(self.hparams.class_weight))

        return loss_function

    def reset_best_val_metric(self):
        if self.task in [GNEPropTask.REGRESSION]:
            self.best_val_metric = np.inf
        elif self.task in [GNEPropTask.BINARY_CLASSIFICATION, GNEPropTask.MULTI_CLASSIFICATION]:
            self.best_val_metric = 0

    @auto_move_data
    def forward(self, data):
        if self.compute_representations:
            return self.get_representations(data)
        else:
            o = self._compute_with_activation(data, return_logits=False, inverse_preprocess_output=True)
            return o

    def _compute(self, data, perturb=None):
        mol_features = None if self.hparams.mol_features_size == 0 else data.mol_features
        o = self.mpn_layers(data.x, data.edge_index, data.edge_attr, mol_features, data.batch, perturb=perturb)

        if self.hparams.use_proj_head:
            o, o_proj = o
            return o, o_proj
        else:
            return o

    def _inverse_preprocess_output(self, o):
        if self.hparams.output_preprocess is not None:
            output_pp_statistics = self.hparams.output_pp_statistics

            if self.hparams.output_preprocess == 'log+std':
                mean_tensor = torch.tensor(output_pp_statistics['mean']).to(device=self.device)
                scale_tensor = torch.tensor(output_pp_statistics['scale']).to(device=self.device)

                o_inverse_processed = torch.exp(o * scale_tensor + mean_tensor)

            elif self.hparams.output_preprocess == 'standard':
                mean_tensor = torch.tensor(output_pp_statistics['mean']).to(device=self.device)
                scale_tensor = torch.tensor(output_pp_statistics['scale']).to(device=self.device)
                o_inverse_processed = o * scale_tensor + mean_tensor

            elif self.hparams.output_preprocess == 'minmax':
                min_tensor = torch.tensor(output_pp_statistics['min']).to(device=self.device)
                max_tensor = torch.tensor(output_pp_statistics['max']).to(device=self.device)
                o_inverse_processed = o * (max_tensor - min_tensor) + min_tensor

            elif self.hparams.output_preprocess == 'log':
                o_inverse_processed = torch.exp(o)

            elif self.hparams.output_preprocess == 'log+1':
                o_inverse_processed = torch.exp(o) - 1

            elif self.hparams.output_preprocess == 'sqrt':
                o_inverse_processed = o ** 2

            elif self.hparams.output_preprocess == 'cbrt':
                o_inverse_processed = o ** 3

            elif self.hparams.output_preprocess == 'boxcox':

                def invboxcox(y_transform, lamda):
                    if lamda == torch.tensor(0):
                        return (torch.exp(y_transform))
                    else:
                        return (torch.exp(torch.log(lamda * y_transform + 1) / lamda))

                lambda_tensor = torch.tensor(output_pp_statistics['lambda']).to(device=self.device)
                o_inverse_processed = invboxcox(o, lambda_tensor)

            return o_inverse_processed

    def _preprocess_output(self, o):
        if self.hparams.output_preprocess is not None:
            output_pp_statistics = self.hparams.output_pp_statistics

            if self.hparams.output_preprocess == 'log+std':
                mean_tensor = torch.tensor(output_pp_statistics['mean']).to(device=self.device)
                scale_tensor = torch.tensor(output_pp_statistics['scale']).to(device=self.device)
                o_inverse_processed = (torch.log(o) - mean_tensor) / scale_tensor

            if self.hparams.output_preprocess == 'standard':
                mean_tensor = torch.tensor(output_pp_statistics['mean']).to(device=self.device)
                scale_tensor = torch.tensor(output_pp_statistics['scale']).to(device=self.device)
                o_inverse_processed = (o - mean_tensor) / scale_tensor

            elif self.hparams.output_preprocess == 'minmax':
                min_tensor = torch.tensor(output_pp_statistics['min']).to(device=self.device)
                max_tensor = torch.tensor(output_pp_statistics['max']).to(device=self.device)
                o_inverse_processed = (o - min_tensor) / (max_tensor - min_tensor)

            elif self.hparams.output_preprocess == 'log':
                o_inverse_processed = torch.log(o)

            elif self.hparams.output_preprocess == 'log+1':
                o_inverse_processed = torch.log(o + 1)

            elif self.hparams.output_preprocess == 'sqrt':
                o_inverse_processed = torch.sqrt(o)

            elif self.hparams.output_preprocess == 'cbrt':
                # o_inverse_processed = torch.pow(o, 1/3)
                o_inverse_processed = torch.sign(o) * torch.pow(torch.abs(o), 1 / 3)

            elif self.hparams.output_preprocess == 'boxcox':
                def boxcox(y, lamda):
                    if lamda == torch.tensor(0):
                        return torch.log(y)
                    else:
                        return (y ** lamda - 1) / lamda

                lambda_tensor = torch.tensor(output_pp_statistics['lambda']).to(device=self.device)
                o_inverse_processed = boxcox(o, lambda_tensor)

            return o_inverse_processed

    def _compute_with_activation(self, data, return_logits=False, inverse_preprocess_output=False, perturb=None):
        out = self._compute(data, perturb=perturb)
        if self.hparams.use_proj_head:
            logits, _ = out
        else:
            logits = out

        if self.task is GNEPropTask.BINARY_CLASSIFICATION:
            o = torch.sigmoid(logits)
        elif self.task is GNEPropTask.REGRESSION:
            if self.hparams.final_relu:
                o = F.relu(logits)
            else:
                o = logits
        elif self.task is GNEPropTask.MULTI_CLASSIFICATION:
            o = torch.softmax(logits, dim=1)
        else:
            raise ValueError

        if inverse_preprocess_output and self.hparams.output_preprocess is not None:
            o = self._inverse_preprocess_output(o)

        if return_logits:
            return o, out  # out can be logits or (logits, repr)
        else:
            return o

    def _compute_loss(self, logits, targets, include_supcon=False, only_supcon=False, supcon_weight=1.):
        if self.hparams.use_proj_head:
            logits, repr = logits
            if only_supcon:
                loss = self.supcon_loss(repr, targets.flatten())
                return loss

        loss = self.loss_function(logits, targets)

        if include_supcon:
            if targets.sum() >= 2:
                loss_supcon = self.supcon_loss(repr, targets.flatten())
                loss = loss + (supcon_weight*loss_supcon)

        return loss

    def _prepare_labels(self, targets):
        if self.task in (GNEPropTask.BINARY_CLASSIFICATION, GNEPropTask.REGRESSION):
            targets = torch.unsqueeze(targets, 1)

        if self.task is GNEPropTask.MULTI_CLASSIFICATION:
            targets = targets.long()

        return targets

    def _prepare_labels_return(self, targets):
        if self.task in (GNEPropTask.BINARY_CLASSIFICATION, GNEPropTask.MULTI_CLASSIFICATION):
            targets = targets.long()
        return targets

    def training_step_adv(self, batch, batch_idx):
        batch = batch['train']
        opt = self.optimizers()
        opt.zero_grad()

        y = batch.y
        y = self._prepare_labels(y)

        sch = self.lr_schedulers()

        if self.hparams.adv == 'flag':
            a_perturb_shape = (batch.x.shape[0], self.hparams.hidden_size)
            step_size = self.hparams.adv_step_size
            m = self.hparams.adv_m
            perturb = torch.FloatTensor(*a_perturb_shape).uniform_(-step_size, step_size).to(self.device)
            perturb.requires_grad_()

            probs, logits = self._compute_with_activation(batch, return_logits=True, perturb={'perturb_a': perturb})
            loss = self._compute_loss(logits, y)
            loss /= m

            for _ in range(m - 1):
                self.manual_backward(loss)
                perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
                perturb.data = perturb_data.data
                perturb.grad[:] = 0

                probs, logits = self._compute_with_activation(batch, return_logits=True, perturb={'perturb_a': perturb})
                loss = self._compute_loss(logits, y)
                loss /= m

            self.manual_backward(loss)

            opt.step()

            sch.step()

            y = self._prepare_labels_return(y)

            return {'loss': loss, 'probs': probs.detach(), 'labels': y}

    def training_step_meta(self, batch, batch_idx):
        meta_train_batch = batch['train']
        meta_test_batch = batch['meta_test']
        if 'ac' in batch:
            ac_batch = batch['ac']
        else:
            ac_batch = None

        opt = self.optimizers()
        sch = self.lr_schedulers()
        if self.hparams.lr_strategy == 'constant':
            current_lr = self.hparams.lr
        else:
            current_lr = sch.get_last_lr()[0]
        maml = l2l.algorithms.MAML(self, lr=current_lr, first_order=False, allow_unused=True, allow_nograd=True)

        maml = maml.clone()
        maml_original = maml.clone()

        y_metatrain = self._prepare_labels(meta_train_batch.y)
        y_metatest = self._prepare_labels(meta_test_batch.y)

        # METATRAIN
        probs_metatrain, logits_metatrain = maml._compute_with_activation(meta_train_batch, return_logits=True)
        loss_metatrain = maml._compute_loss(logits_metatrain, y_metatrain, include_supcon=self.hparams.use_proj_head, supcon_weight=self.hparams.supcon_weight, only_supcon=self.hparams.only_supcon)

        maml.adapt(loss_metatrain)

        # METATEST
        opt.zero_grad()

        probs_metatest, logits_metatest = maml._compute_with_activation(meta_test_batch, return_logits=True)
        loss_metatest = maml._compute_loss(logits_metatest, y_metatest, include_supcon=self.hparams.use_proj_head, supcon_weight=self.hparams.supcon_weight, only_supcon=self.hparams.only_supcon)

        probs_metatrain2, logits_metatrain2 = maml_original._compute_with_activation(meta_train_batch, return_logits=True)
        loss_original = maml_original._compute_loss(logits_metatrain2, y_metatrain, include_supcon=self.hparams.use_proj_head, supcon_weight=self.hparams.supcon_weight, only_supcon=self.hparams.only_supcon)

        final_loss = (self.hparams.meta_weight * loss_metatest) + loss_original

        # AC
        if ac_batch is not None and len(ac_batch) > 0:
            y_ac = self._prepare_labels(ac_batch.y)
            probs_ac, logits_ac = maml_original._compute_with_activation(ac_batch, return_logits=True)
            loss_ac = maml_original._compute_loss(logits_ac, y_ac, include_supcon=self.hparams.use_proj_head, supcon_weight=self.hparams.supcon_weight, only_supcon=self.hparams.only_supcon)
            final_loss += loss_ac

        maml.manual_backward(final_loss)

        opt.step()

        if sch is not None:
            sch.step()

        y_metatrain = self._prepare_labels_return(y_metatrain)

        return {'loss': loss_metatrain, 'probs': probs_metatrain.detach(), 'labels': y_metatrain}

    def training_step(self, batch, batch_idx):
        if self.hparams.adv != 'none':
            return self.training_step_adv(batch, batch_idx)

        if self.hparams.meta:
            return self.training_step_meta(batch, batch_idx)

        batch = batch['train']
        if 'ac' in batch:
            ac_batch = batch['ac']
        else:
            ac_batch = None

        y = batch.y
        y = self._prepare_labels(y)

        probs, logits = self._compute_with_activation(batch, return_logits=True)
        loss = self._compute_loss(logits, y, include_supcon=self.hparams.supcon_train, supcon_weight=self.hparams.supcon_weight, only_supcon=self.hparams.only_supcon)

        # AC
        if ac_batch is not None and len(ac_batch) > 0:
            y_ac = self._prepare_labels(ac_batch.y)
            probs_ac, logits_ac = self._compute_with_activation(ac_batch, return_logits=True)
            loss_ac = self._compute_loss(logits_ac, y_ac, include_supcon=self.hparams.use_proj_head, supcon_weight=self.hparams.supcon_weight, only_supcon=self.hparams.only_supcon)
            loss += loss_ac

        y = self._prepare_labels_return(y)
        return {'loss': loss, 'probs': probs.detach(), 'labels': y}

    def on_train_start(self) -> None:
        if self.task is GNEPropTask.REGRESSION:
            self.logger.log_hyperparams({
                "test_mse": np.inf, "test_mae": np.inf, "test_ev": 0.0, 'test_r2': 0.0,
                self.name_best_val_metric: np.inf})
        else:
            self.logger.log_hyperparams({
                "test_auc": 0, "test_ap": 0, self.name_best_val_metric: 0})

    def validation_step(self, batch, batch_idx):
        y = batch.y
        y = self._prepare_labels(y)

        probs, logits = self._compute_with_activation(batch, return_logits=True,
                                                      inverse_preprocess_output=True)  # "logits" in the current implementation was "o" in the previous version
        loss = self._compute_loss(logits, y)

        y = self._prepare_labels_return(y)
        return {'loss': loss, 'probs': probs.detach(), 'labels': y}

    def test_step(self, batch, batch_idx):
        y = batch.y
        y = self._prepare_labels(y)

        probs, logits = self._compute_with_activation(batch, return_logits=True,
                                                      inverse_preprocess_output=True)  # "logits" in the current implementation was "o" in the previous version
        loss = self._compute_loss(logits, y)

        y = self._prepare_labels_return(y)
        return {'loss': loss, 'probs': probs.detach(), 'labels': y}

    def _group_end_results(self, step_outputs):
        all_probs = torch.cat([i['probs'] for i in step_outputs])
        all_labels = torch.cat([i['labels'] for i in step_outputs])
        return all_probs, all_labels

    def training_epoch_end(self, training_step_outputs):
        all_training_probs, all_training_labels = self._group_end_results(training_step_outputs)

        if self.task is GNEPropTask.REGRESSION:
            mse = mean_squared_error(all_training_probs, all_training_labels)
            mae = mean_absolute_error(all_training_probs, all_training_labels)
            ev = explained_variance(all_training_probs, all_training_labels)
            r2 = r2_score(all_training_probs, all_training_labels)
            self.log('train_mae', mae, prog_bar=True, sync_dist=True)
            self.log('train_mse', mse, prog_bar=True, sync_dist=True)
            self.log('train_ev', ev, prog_bar=True, sync_dist=True)
            self.log('train_r2', r2, prog_bar=True, sync_dist=True)

        elif self.task in (GNEPropTask.BINARY_CLASSIFICATION, GNEPropTask.MULTI_CLASSIFICATION):
            num_classes = None if self.task is GNEPropTask.BINARY_CLASSIFICATION else self.hparams.out_channels
            task_name = 'binary' if self.task is GNEPropTask.BINARY_CLASSIFICATION else 'multiclass'
            self._log_metric(prefix='train', average='macro', probs=all_training_probs, labels=all_training_labels, num_classes=num_classes, task_name=task_name)

    def _log_log(self, preds, labels, prefix):

        def _find_first_pos(pred_tensor):
            pred_tensor = pred_tensor.sort(descending=False)
            for x in pred_tensor[0]:
                if x > 0:
                    return x

        neg_cnt = torch.numel(preds[preds < 0])

        min_pos = _find_first_pos(preds)
        if not min_pos:
            print('no positive preds')
            return
        else:
            preds = torch.where(preds < 0, min_pos, preds)

            labels = torch.log(labels)
            preds = torch.log(preds)
            mse = mean_squared_error(preds, labels)
            mae = mean_absolute_error(preds, labels)
            ev = explained_variance(preds, labels)
            r2 = r2_score(preds, labels)

            self.log('negative pred cnt', neg_cnt, prog_bar=True, sync_dist=True)
            self.log('{}_mae_log'.format(prefix), mae, prog_bar=True, sync_dist=True)
            self.log('{}_mse_log'.format(prefix), mse, prog_bar=True, sync_dist=True)
            self.log('{}_ev_log'.format(prefix), ev, prog_bar=True, sync_dist=True)
            self.log('{}_r2_log'.format(prefix), r2, prog_bar=True, sync_dist=True)

    def _log_transform(self, preds, labels, prefix):
        labels = self._preprocess_output(labels)
        preds = self._preprocess_output(preds)
        mse = mean_squared_error(preds, labels)
        mae = mean_absolute_error(preds, labels)
        ev = explained_variance(preds, labels)
        r2 = r2_score(preds, labels)
        self.log('{}_mae_transform'.format(prefix), mae, prog_bar=True, sync_dist=True)
        self.log('{}_mse_transform'.format(prefix), mse, prog_bar=True, sync_dist=True)
        self.log('{}_ev_transform'.format(prefix), ev, prog_bar=True, sync_dist=True)
        self.log('{}_r2_transform'.format(prefix), r2, prog_bar=True, sync_dist=True)


    def _log_metric(self, prefix, average, probs, labels, num_classes, task_name):

        average_precision = AveragePrecision(task=task_name, num_classes=num_classes, average=average).to(self.device)
        average_precision_score = average_precision(probs, labels)
        self.log('{}_ap'.format(prefix), average_precision_score, prog_bar=True, sync_dist=True)

        if self.task is GNEPropTask.BINARY_CLASSIFICATION:
            try:
                auroc = AUROC(task=task_name, average=average, num_classes=num_classes).to(self.device)
                auc = auroc(probs, labels)
                self.log('{}_auc'.format(prefix), auc, prog_bar=True, sync_dist=True)
            except ValueError:  # accounts for cases where not all clases are in the validation set
                pass

        if prefix == 'test' and self.task is GNEPropTask.MULTI_CLASSIFICATION:  # accounts for class imbalance
            weighted_auc = auroc(probs, labels, num_classes=num_classes, average='weighted').to(self.device)
            self.log('{}_weighted_auc'.format(prefix), weighted_auc, prog_bar=True, sync_dist=True)

        if prefix in ['val', 'test']:
            f1 = F1Score(num_classes=num_classes, task=task_name, average=average).to(self.device)
            f1_score = f1(probs, labels)
            self.log('{}_f1'.format(prefix), f1_score, prog_bar=True, sync_dist=True)

            acc = Accuracy(task=task_name, num_classes=num_classes, average=average).to(self.device)
            accuracy = acc(probs, labels)
            self.log('{}_acc'.format(prefix), accuracy, prog_bar=True, sync_dist=True)

            rec = Recall(task=task_name, num_classes=num_classes, average=average).to(self.device)
            recall = rec(probs, labels)
            self.log('{}_precision'.format(prefix), recall, prog_bar=True, sync_dist=True)

            prec = Precision(task=task_name, num_classes=num_classes, average=average).to(self.device)
            precision = prec(probs, labels)
            self.log('{}_recall'.format(prefix), precision, prog_bar=True, sync_dist=True)

            if prefix == 'val':
                return auc, average_precision_score, accuracy

    def validation_epoch_end(self, validation_step_outputs):
        all_validation_probs, all_validation_labels = self._group_end_results(validation_step_outputs)

        if self.task is GNEPropTask.REGRESSION:
            mse = mean_squared_error(all_validation_probs, all_validation_labels)
            mae = mean_absolute_error(all_validation_probs, all_validation_labels)
            ev = explained_variance(all_validation_probs, all_validation_labels)
            r2 = r2_score(all_validation_probs, all_validation_labels)
            self.log('val_mae', mae, prog_bar=True, sync_dist=True)
            self.log('val_mse', mse, prog_bar=True, sync_dist=True)
            self.log('val_ev', ev, prog_bar=True, sync_dist=True)
            self.log('val_r2', r2, prog_bar=True, sync_dist=True)

            if self.hparams.output_preprocess:
                self._log_log(all_validation_probs, all_validation_labels, prefix='val')

            target_metric = None
            if self.hparams.metric == 'val_mse':
                target_metric = mse
            if self.hparams.metric == 'val_mae':
                target_metric = mae
            if target_metric < self.best_val_metric:
                self.best_val_metric = target_metric

        elif self.task in (GNEPropTask.BINARY_CLASSIFICATION, GNEPropTask.MULTI_CLASSIFICATION):
            num_classes = None if self.task is GNEPropTask.BINARY_CLASSIFICATION else self.hparams.out_channels
            task_name = 'binary' if self.task is GNEPropTask.BINARY_CLASSIFICATION else 'multiclass'
            auc, average_precision_score, acc = self._log_metric(prefix='val', average='macro', probs=all_validation_probs, labels=all_validation_labels, num_classes=num_classes, task_name=task_name)

            # log plots
            if self.task is GNEPropTask.MULTI_CLASSIFICATION:
                cm = ConfusionMatrix(num_classes=num_classes)(all_validation_probs.detach().cpu(),
                                                              all_validation_labels.detach().cpu()).numpy().astype(
                    np.int32)
                cm_im = plot_confusion_matrix_as_torch_tensor(cm)
                tb = self.logger.experiment
                tb.add_image("val_confusion_matrix", cm_im, global_step=self.current_epoch)

            target_metric = None
            if self.hparams.metric == 'val_auc':
                target_metric = auc
            if self.hparams.metric == 'val_ap':
                target_metric = average_precision_score
            if self.hparams.metric == 'val_acc':
                target_metric = acc

            if target_metric > self.best_val_metric:
                self.best_val_metric = target_metric

        self.log(self.name_best_val_metric, self.best_val_metric, prog_bar=True, sync_dist=True)

    def test_epoch_end(self, test_step_outputs):
        all_test_probs, all_test_labels = self._group_end_results(test_step_outputs)

        if self.task is GNEPropTask.REGRESSION:
            mse = mean_squared_error(all_test_probs, all_test_labels)
            mae = mean_absolute_error(all_test_probs, all_test_labels)
            ev = explained_variance(all_test_probs, all_test_labels)
            r2 = r2_score(all_test_probs, all_test_labels)
            self.log('test_mae', mae, prog_bar=True, sync_dist=True)
            self.log('test_mse', mse, prog_bar=True, sync_dist=True)
            self.log('test_ev', ev, prog_bar=True, sync_dist=True)
            self.log('test_r2', r2, prog_bar=True, sync_dist=True)

            if self.hparams.output_preprocess:
                self._log_log(all_test_probs, all_test_labels, prefix='test')

        elif self.task in (GNEPropTask.BINARY_CLASSIFICATION, GNEPropTask.MULTI_CLASSIFICATION):
            num_classes = None if self.task is GNEPropTask.BINARY_CLASSIFICATION else self.hparams.out_channels
            task_name = 'binary' if self.task is GNEPropTask.BINARY_CLASSIFICATION else 'multiclass'

            self._log_metric(prefix='test', average=None, probs=all_test_probs, labels=all_test_labels, num_classes=num_classes, task_name=task_name)

            if self.task is GNEPropTask.MULTI_CLASSIFICATION:  # accounts for class imbalance
                weighted_auc = auroc(all_test_probs, all_test_labels, num_classes=num_classes, average='weighted')
                self.log('test_weighted_auc', weighted_auc, prog_bar=True, sync_dist=True)

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{
            'params': params,
            'weight_decay': weight_decay
        }, {
            'params': excluded_params,
            'weight_decay': 0.,
        }]

    def configure_optimizers(self):
        if self.hparams.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.hparams.weight_decay)
        else:
            params = self.parameters()

        optimizer = torch.optim.Adam(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        if self.hparams.lr_strategy == 'constant':
            return optimizer
        if self.hparams.lr_strategy == 'warmup':
            scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5)
            return [optimizer], [scheduler_warmup]
        if self.hparams.lr_strategy == 'warmup_cosine':
            scheduler_warmup_cosine = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=6,
                                                                    max_epochs=self.hparams.max_epochs,
                                                                    eta_min=self.hparams.lr / 10)
            return [optimizer], [scheduler_warmup_cosine]
        if self.hparams.lr_strategy == 'warmup_cosine_step':
            scheduler_warmup_cosine = LinearWarmupCosineAnnealingLR(optimizer,
                                                                    warmup_epochs=5 * self.hparams.num_train_batches,
                                                                    max_epochs=self.hparams.max_epochs * self.hparams.num_train_batches,
                                                                    eta_min=self.hparams.lr / 10)
            return [optimizer], [{'scheduler': scheduler_warmup_cosine, 'interval': 'step'}]
        if self.hparams.lr_strategy == 'finetune':
            all_parameters = [i[1] for i in self.named_parameters() if not i[0].startswith('mpn_layers.classifier')]
            classifier_parameters = self.mpn_layers.classifier.parameters()

            optimizer = torch.optim.Adam([{'params': all_parameters, 'lr': self.hparams.lr / 10},
                                          {'params': classifier_parameters, 'lr': self.hparams.lr}])

            scheduler_classifier_warmup_cosine = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=6,
                                                                               max_epochs=self.hparams.max_epochs,
                                                                               eta_min=self.hparams.lr / 10)

            return [optimizer], [scheduler_classifier_warmup_cosine]

    def predict_data(self, dataset, gpus=1, batch_size=100, num_workers=16):
        accelerator = get_accelerator(gpus)
        trainer = pl.Trainer(gpus=gpus,
                             logger=False,
                             auto_select_gpus=True, accelerator=accelerator)

        dataloader = convert_to_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)

        res = trainer.predict(self, dataloaders=dataloader)
        res = [i.cpu().numpy() for i in res]
        res = np.concatenate(res)

        return res

    def test_data(self, data, gpus=1, batch_size=100, num_workers=16):
        accelerator = get_accelerator(gpus)
        trainer = pl.Trainer(gpus=gpus,
                             logger=False,
                             auto_select_gpus=True, accelerator=accelerator)
        test_loader = convert_to_dataloader(data, batch_size=batch_size, num_workers=num_workers)
        test_output = trainer.test(model=self, dataloaders=test_loader)
        return test_output

    @auto_move_data
    def get_representations(self, data, use_ffn_layers=0):
        if use_ffn_layers == 0:
            o = self.mpn_layers.compute_representations(data.x, data.edge_index, data.edge_attr, data.batch)
        else:
            if self.hparams.mol_features_size == 0:
                o = self.mpn_layers(data.x, data.edge_index, data.edge_attr, None, data.batch,
                                    restrict_output_layers=use_ffn_layers)
            else:
                o = self.mpn_layers(data.x, data.edge_index, data.edge_attr, data.mol_features, data.batch,
                                    restrict_output_layers=use_ffn_layers)
        return o

    def get_representations_dataset(self, dataset, use_ffn_layers=0, disable_progress_bar=False):
        dataloader = convert_to_dataloader(dataset)
        outs = []
        with torch.no_grad():
            for i in tqdm(dataloader, position=0, leave=True, disable=disable_progress_bar):
                out = self.get_representations(i, use_ffn_layers=use_ffn_layers).cpu().numpy()
                outs.append(out)
        return np.vstack(outs)

    def freeze_first_layers(self, freeze_ab_embeddings=False, mp_to_freeze=1):
        # freeze node and edge encoder
        if freeze_ab_embeddings:
            for p in self.mpn_layers.node_encoder.parameters():
                p.requires_grad = False
            for p in self.mpn_layers.edge_encoder.parameters():
                p.requires_grad = False
        if mp_to_freeze <= 0:
            return
        for conv in self.mpn_layers.convs[:mp_to_freeze]:
            for p in conv.parameters():
                p.requires_grad = False

    def freeze_params(self, freeze_list=('bias', 'bn'), only_convs=False):
        target = self.mpn_layers.convs if only_convs else self
        for name, param in target.named_parameters():
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in freeze_list):
                param.requires_grad = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GNEModel")
        parser.add_argument('--train_batch_size', type=int, default=50)
        parser.add_argument('--node_feat_size', type=int, default=133)
        parser.add_argument('--edge_feat_size', type=int, default=12)
        parser.add_argument('--hidden_size', type=int, default=500)
        parser.add_argument('--ffn_hidden_size', type=int, default=None)
        parser.add_argument('--depth', type=int, default=5)
        parser.add_argument('--num_readout_layers', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1.e-05)
        parser.add_argument('--dropout', type=float, default=0.)
        parser.add_argument('--jk', default='cat', const='cat', nargs='?', choices=['cat', 'none'])
        parser.add_argument('--mol_features_size', type=int, default=0.)
        parser.add_argument('--lr_strategy', default='constant', const='constant',
                            nargs='?',
                            choices=['constant', 'warmup', 'warmup_cosine', 'warmup_cosine_step', 'finetune'])
        parser.add_argument('--pos_weight', type=int, default=1, help='Loss weight for binary classification.')
        parser.add_argument('--class_weight', type=float, nargs='+', default=None,
                            help='Loss weight for multi classification.')
        parser.add_argument('--auto_loss_weight', action="store_true", default=False,
                            help='Automatically compute loss weight (binary or multi-class).')
        parser.add_argument("--balanced_batch_training", action="store_true")
        parser.add_argument("--keep_all_checkpoints", action="store_true")
        parser.add_argument("--keep_last_checkpoint", action="store_true")
        parser.add_argument("--metric", type=str, default='val_auc')
        parser.add_argument('--use_early_stopping', action="store_true", default=False, help='Enable early stopping')
        parser.add_argument('--aggr', default='mean', const='mean', nargs='?',
                            choices=['mean', 'sum', 'global_attention', 'gmt'])
        parser.add_argument("--ensemble_incr_seed", type=int, default=0)
        parser.add_argument("--weight_decay", type=float, default=0.)
        parser.add_argument('--exclude_bn_bias', action='store_true', help="Exclude bn/bias from weight decay")
        parser.add_argument('--use_mol_features', action='store_true', default=False,
                            help="Use precomputed molecular features")
        parser.add_argument('--mol_features', default='rdkit_normalized', const='rdkit_normalized', nargs='?',
                            choices=['rdkit_normalized', 'pretrain'])
        parser.add_argument('--extra_mol_features', nargs='+', default=None, help='additional molecular feature names')
        parser.add_argument('--mp_to_freeze', type=int, default=0)
        parser.add_argument('--freeze_ab_embeddings', action="store_true", help='Freeze atom/bond embeddings')
        parser.add_argument('--freeze_batchnorm', action="store_true", help='Freeze batch norm')
        parser.add_argument('--freeze_bias', action="store_true", help='Freeze bias')
        parser.add_argument('--reuse_pretrain_readout', type=int, default=0)
        parser.add_argument('--log_directory', type=str, default='gneprop/logs', help='Where logs are saved')
        parser.add_argument('--final_relu', action="store_true", help='Toggle final relu activation for regression')
        parser.add_argument('--task', default='binary_classification', const='binary_classification', nargs='?',
                            choices=[i.value for i in GNEPropTask], help='GNEprop task')
        parser.add_argument('--out_channels', type=int, default=1,
                            help='Number of out channels (classes) for multiclass classification')
        parser.add_argument('--output_preprocess', default=None, nargs='?',
                            choices=['log+std', 'minmax', 'standard', 'log', 'log+1', 'sqrt', 'cbrt', 'boxcox'])

        parser.add_argument('--skip_last_relu', action="store_true")

        # ADV PARAMS
        parser.add_argument('--adv', default='none', const='none', nargs='?', choices=['none', 'flag', 'gne', 'gne_2'])
        parser.add_argument('--adv_step_size', type=float, default=1e-3)
        parser.add_argument('--adv_m', type=int, default=3)

        # GMT PARAMS
        parser.add_argument('--gmt_hidden_channels', type=int, default=500)
        parser.add_argument('--gmt_pooling_ratio', type=float, default=0.25)
        parser.add_argument('--gmt_num_heads', type=int, default=4)
        parser.add_argument('--gmt_layer_norm', action="store_true", default=False)
        parser.add_argument('--gmt_sequence', type=int, default=3)

        # META PARAMS
        parser.add_argument('--meta', action="store_true", default=False)
        parser.add_argument('--meta_test', default='val', const='val', nargs='?', choices=['val', 'half_val'])
        parser.add_argument('--meta_weight', type=float, default=1.)

        # PROJ PARAMS
        parser.add_argument('--use_proj_head', action="store_true", default=False)
        parser.add_argument('--add_ac', action="store_true", default=False)
        parser.add_argument('--supcon_train', action="store_true", default=False)
        parser.add_argument('--supcon_weight', type=float, default=1.)
        parser.add_argument('--only_supcon', action="store_true", default=False)
        parser.add_argument('--ac_skip', type=float, default=-1.)

        return parent_parser


def convert_to_dataloader(data: Union[str, torch.utils.data.Dataset, torch.utils.data.DataLoader], batch_size=100,
                          num_workers=16):
    if isinstance(data, str):  # convert from csv
        data = MolDatasetOD.load_csv_dataset(data)
    if isinstance(data, torch.utils.data.Dataset):
        dataloader = DataLoader(data, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    elif isinstance(data, torch.utils.data.DataLoader):
        dataloader = data
    else:
        raise NotImplementedError("Data type not supported.")
    return dataloader


def predict_from_checkpoints_return_data(data: torch.utils.data.Dataset, checkpoint_path=None, checkpoints_paths=None,
                                         checkpoint_dir=None, gpus=1, return_data=False, no_labels=False):
    list_checkpoints = gneprop.utils.get_checkpoint_paths(checkpoint_path=checkpoint_path,
                                                          checkpoint_paths=checkpoints_paths,
                                                          checkpoint_dir=checkpoint_dir)
    # works only for single input
    all_preds = []
    for checkpoint in list_checkpoints:
        print(f'Predicting checkpoint <{checkpoint}>...')
        model = GNEprop.load_from_checkpoint(checkpoint)
        preds = model.predict_data(data, gpus=gpus).flatten()
        all_preds.append(preds)

    all_res = np.stack(all_preds)  # n_ensemble x n_mols
    preds, epi_uncs = all_res.mean(axis=0), all_res.var(axis=0)

    indices = list(range(len(data)))
    data = MoleculeSubset(data, indices)

    if return_data & (not no_labels):
        return preds, epi_uncs, data.y[np.newaxis, :].flatten(), data
    elif not no_labels:
        return preds, epi_uncs, data.y[np.newaxis, :].flatten()
    elif return_data:
        return preds, epi_uncs, data
    else:
        return preds, epi_uncs


def predict_ensemble(list_checkpoints, data: Union[torch.utils.data.Dataset, torch.utils.data.DataLoader], aggr='mean',
                     gpus=1):
    # works only for single input
    all_preds = []
    for checkpoint in list_checkpoints:
        print(f'Predicting checkpoint <{checkpoint}>...')
        model = GNEprop.load_from_checkpoint(checkpoint)
        preds = model.predict_data(data, gpus=gpus).flatten()
        all_preds.append(preds)

    all_res = np.stack(all_preds)  # n_ensemble x n_mols
    if aggr == 'mean':
        preds, epi_uncs = all_res.mean(axis=0), all_res.var(axis=0)
    elif aggr == 'max':
        preds, epi_uncs = all_res.max(axis=0), all_res.var(axis=0)
    else:
        raise ValueError

    return preds, epi_uncs


def predict_from_checkpoints(data: Union[str, torch.utils.data.Dataset, torch.utils.data.DataLoader],
                             checkpoint_path=None, checkpoints_paths=None, checkpoint_dir=None, aggr='mean', gpus=1):
    list_checkpoints = gneprop.utils.get_checkpoint_paths(checkpoint_path=checkpoint_path,
                                                          checkpoint_paths=checkpoints_paths,
                                                          checkpoint_dir=checkpoint_dir)
    return predict_ensemble(list_checkpoints, data, aggr=aggr, gpus=gpus)


def resplit(all_data, hparams, **kwargs):
    # kwargs arguments used as backup
    if 'split_sizes' in hparams and hparams['split_sizes'] is not None:
        sizes = hparams['split_sizes']
    else:
        sizes = kwargs['split_sizes']
    if 'seed' in hparams and hparams['seed'] is not None:
        seed = hparams['seed']
    else:
        seed = kwargs['seed']
    if 'split_type' in hparams and hparams['split_type'] is not None:
        split_type = hparams['split_type']
    else:
        split_type = kwargs['split_type']
    if 'index_predetermined_file' in hparams and hparams['index_predetermined_file'] is not None:
        with open(hparams['index_predetermined_file'], 'rb') as rf:
            all_data.index_predetermined = pickle.load(rf)
    else:
        all_data.index_predetermined = None
    return split_data(all_data, split_type=split_type, sizes=sizes, seed=seed)


def test_resplit(all_data, checkpoint, num_workers=16, **kwargs):
    print(f'Testing checkpoint: {checkpoint}')
    model = GNEprop.load_from_checkpoint(checkpoint, **kwargs)
    _, _, test_data = resplit(all_data, hparams=model.hparams, **kwargs)
    return model.test_data(test_data, num_workers=num_workers)[0]


def test_resplit_folds(all_data, checkpoint_dir, num_workers=16, **kwargs):
    list_checkpoints = gneprop.utils.get_checkpoint_paths(checkpoint_dir=checkpoint_dir)
    test_dicts = [test_resplit(all_data, checkpoint, num_workers=num_workers, **kwargs) for checkpoint in list_checkpoints]
    agg_dict = gneprop.utils.aggregate_metrics(test_dicts)
    return agg_dict


def predict_resplit(all_data, checkpoint, return_data=False, return_model=False, **kwargs):
    print(f'Predict with checkpoint: {checkpoint}')

    model = GNEprop.load_from_checkpoint(checkpoint)
    train_data, val_data, test_data = resplit(all_data, hparams=model.hparams, **kwargs)

    preds = model.predict_data(test_data)
    if model.hparams.out_channels == 1:
        preds = preds.flatten()

    res = []
    res.append(preds)
    if return_data:
        res.append(test_data.y[np.newaxis, :].flatten())
        res.append((train_data, val_data, test_data))
    if return_model:
        res.append(model)
    return res


def predict_resplit_folds(all_data, checkpoint_dir, return_data=False, return_model=False, **kwargs):
    list_checkpoints = gneprop.utils.get_checkpoint_paths(checkpoint_dir=checkpoint_dir)
    all_preds = [predict_resplit(all_data, checkpoint, return_data=return_data, return_model=return_model, **kwargs) for
                 checkpoint in
                 list_checkpoints]
    return all_preds


def get_lcc(max_epochs):
    # temporary function to fix lighting max_epochs bug
    def lcc(*l_args):
        l_args[0].max_epochs = max_epochs

    return lcc


def load_pretrained_model_freeze_params(args, model):
    if os.path.isfile(args.pretrain_path):
        model_pretrained = clr.SimCLR.load_from_checkpoint(args.pretrain_path)
    elif os.path.isdir(args.pretrain_path):
        pretrain_full_path = gneprop.utils.get_checkpoint_paths(checkpoint_dir=args.pretrain_path)[0]
        model_pretrained = clr.SimCLR.load_from_checkpoint(pretrain_full_path)
    else:
        raise ValueError("Invalid argument: pretrain path")

    # Better option with dictionaries
    original_classifier_dict = model.mpn_layers.classifier.state_dict()
    # Strict False is ok because pretrained model has no global attention parameters
    model_pretrained.encoder.classifier = model.mpn_layers.classifier  # the classifier of the pretrained model is not trained, but this avoids error in loading the state
    model.mpn_layers.load_state_dict(model_pretrained.encoder.state_dict(), strict=False)
    model.mpn_layers.classifier.load_state_dict(original_classifier_dict)

    if args.reuse_pretrain_readout > 0:
        for i in range(args.reuse_pretrain_readout):
            model.mpn_layers.classifier[i][0].load_state_dict(
                model_pretrained.projection.model[3 * i + 0].state_dict())  # copy linear layer
            model.mpn_layers.classifier[i][1].load_state_dict(
                model_pretrained.projection.model[3 * i + 1].state_dict())  # copy batchnorm layer

    model.freeze_first_layers(freeze_ab_embeddings=args.freeze_ab_embeddings, mp_to_freeze=args.mp_to_freeze)
    if args.freeze_batchnorm:
        model.freeze_params(('bn',), only_convs=True)
    if args.freeze_bias:
        model.freeze_params(('bias',), only_convs=True)


def load_supervised_pretrained_model_freeze_params(args, model):
    assert os.path.isfile(args.supervised_pretrain_path)

    model_pretrained = GNEprop.load_from_checkpoint(args.supervised_pretrain_path)
    model.load_state_dict(model_pretrained.state_dict())
    model.freeze_first_layers(freeze_ab_embeddings=args.freeze_ab_embeddings, mp_to_freeze=args.mp_to_freeze)
    if args.freeze_batchnorm:
        model.freeze_params(('bn',), only_convs=True)
    if args.freeze_bias:
        model.freeze_params(('bias',), only_convs=True)


def load_supervised_pretrained_model_freeze_params_folds(args, model):
    assert args.supervised_pretrain_paths is not None

    model_pretrained = GNEprop.load_from_checkpoint(args.supervised_pretrain_paths[args.seed])
    model.load_state_dict(model_pretrained.state_dict())
    model.freeze_first_layers(freeze_ab_embeddings=args.freeze_ab_embeddings, mp_to_freeze=args.mp_to_freeze)
    if args.freeze_batchnorm:
        model.freeze_params(('bn',), only_convs=True)
    if args.freeze_bias:
        model.freeze_params(('bias',), only_convs=True)


def gen_run_details(args, experiment_name):
    if args.wb_enable:
        job_type, group, fold_id = None, None, None
        if args.parallel_folds > 1 or args.parallel_ensemble > 1 or args.num_folds > 1:
            job_type = "eval"
            group = experiment_name.split('/')[0]
            fold_id = group + '-' + experiment_name.split('/')[-1]
        dir_base = os.path.abspath(os.getcwd()) if args.log_directory.startswith('s3:') else args.log_directory
        log_dir = os.path.join(dir_base, group if group is not None else '')

        os.makedirs(log_dir, exist_ok=True)

        return log_dir, job_type, group, fold_id


@timeit()
def run_training(dataset, args, name=None, additional_training_data=None, ensemble_size=1):
    assert ensemble_size >= 1
    # Set size splits (temporary)
    if args.split_sizes is None:
        sizes = (0.8, 0.1, 0.1) if len(dataset) < 10000 else (0.9, 0.05, 0.05)
        args.split_sizes = sizes
    else:
        sizes = args.split_sizes

    # Set num workers
    if args.num_workers < 0:
        num_workers = 1 if len(dataset) < 10000 else 4
    else:
        num_workers = args.num_workers

    augmentation = None if args.augmentation is None else AugmentationFactory.create_augmentation_from_file(
        args.augmentation)
    static_val_augmentation = None if args.static_val_augmentation is None else AugmentationFactory.create_augmentation_from_file(
        args.static_val_augmentation)

    task = GNEPropTask(args.task)
    train_loader, val_loader, test_loader, output_pp_statistics = get_loaders(dataset, split_type=args.split_type,
                                                                              sizes=sizes, seed=args.seed,
                                                                              train_batch_size=args.train_batch_size,
                                                                              print_description=args.print_description,
                                                                              num_workers=num_workers,
                                                                              additional_training_data=additional_training_data,
                                                                              balanced_batch_training=args.balanced_batch_training,
                                                                              ig_baseline_ratio=args.ig_baseline_ratio,
                                                                              augmentation=augmentation,
                                                                              augmentation_label=args.augmentation_label,
                                                                              task=task,
                                                                              drop_last=(not args.keep_last_batch),
                                                                              static_val_augmentation=static_val_augmentation,
                                                                              val_multiplier=args.val_multiplier,
                                                                              output_preprocess=args.output_preprocess,
                                                                              meta=args.meta,
                                                                              meta_test=args.meta_test,
                                                                              add_ac=args.add_ac,
                                                                              ac_skip=args.ac_skip)

    args.output_pp_statistics = output_pp_statistics

    if args.invert_val_test:  # debug option
        val_loader, test_loader, = test_loader, val_loader

    if args.alternative_test_data is not None:
        test_loader = DataLoader(
            args.alternative_test_data, batch_size=500, num_workers=1, pin_memory=True)

    if name is None:
        name = get_time_string()

    print('Experiment name: ', name)

    args.num_train_batches = len(train_loader)

    ensemble_checkpoints = []
    for ensemble_ix in range(ensemble_size):
        if ensemble_size == 1:
            experiment_name = name
        else:
            experiment_name = name + "/ensemble_" + str(ensemble_ix)
        seed_everything(args.all_seed + ensemble_ix + args.ensemble_incr_seed)

        # Define callbacks
        model_callbacks = []
        if args.keep_all_checkpoints:
            if args.keep_last_checkpoint:
                checkpoint_callback = ModelCheckpoint()
            else:
                checkpoint_callback = ModelCheckpoint(monitor=args.metric, mode='max' if task in (
                GNEPropTask.BINARY_CLASSIFICATION, GNEPropTask.MULTI_CLASSIFICATION) else 'min')
        else:
            checkpoint_callback = ModelCheckpointSkipFirst(monitor=args.metric, mode='max' if task in (
            GNEPropTask.BINARY_CLASSIFICATION, GNEPropTask.MULTI_CLASSIFICATION) else 'min')

        model_callbacks.append(checkpoint_callback)

        if args.lr_strategy != 'constant':  # avoids warning that a callback is being used without a lr scheduler
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            model_callbacks.append(lr_monitor)

        if args.use_early_stopping:
            early_stop_callback = EarlyStopping(monitor=args.metric, min_delta=0.00, patience=15, verbose=False,
                                                mode='max' if task in (GNEPropTask.BINARY_CLASSIFICATION,
                                                                       GNEPropTask.MULTI_CLASSIFICATION) else 'min')
            model_callbacks.append(early_stop_callback)
        if args.wb_enable:
            local_log_dir, job_type, group, fold_id = gen_run_details(args, name)

        if args.wb_enable:
            from pytorch_lightning.loggers import WandbLogger

            logger = WandbLogger(project=args.wb_project, entity=args.wb_entity, name=experiment_name, log_model="all",
                                 mode="online", id=fold_id, save_dir=local_log_dir, group=group, job_type=job_type, )
        else:
            logger = TensorBoardLogger(args.log_directory, name=experiment_name, default_hp_metric=False)

        accelerator = 'horovod' if args.use_horovod else get_accelerator(args.gpus)
        if accelerator == 'horovod':
            # using horovod requires defining only 1 gpu, the effective number is defined by the horovod command
            args.gpus = 1
        trainer = Trainer.from_argparse_args(args,
                                             logger=logger,
                                             callbacks=model_callbacks,
                                             accelerator=accelerator, num_sanity_val_steps=0,
                                             max_epochs=args.max_epochs)

        if args.wb_enable:
            import wandb

        dict_args = vars(args)
        model = GNEprop(**dict_args)
        # pretrained model
        if args.pretrain_path is not None:
            load_pretrained_model_freeze_params(args, model)
        elif args.supervised_pretrain_path is not None:
            load_supervised_pretrained_model_freeze_params(args, model)
        elif args.supervised_pretrain_paths is not None:
            load_supervised_pretrained_model_freeze_params_folds(args, model)

        trainer.fit(model, train_loader, val_loader)
        best_model_path = checkpoint_callback.best_model_path

        print('Best model path: ', best_model_path)
        print('Best validation score: ', cast_scalar(model.best_val_metric))

        ensemble_checkpoints.append(best_model_path)

        if args.log_directory.startswith('s3:'):
            sync_s3(args, local_log_dir, fold_id, group)

        # if trainer.num_gpus > 1:
        if trainer.num_devices > 1:
            return None  # multi-gpu does not allow train and test in the same program
        else:
            if len(test_loader) > 0:
                test_output = trainer.test(dataloaders=test_loader, ckpt_path=best_model_path)

        if args.wb_enable:
            wandb.finish()  # fix logging for num_folds training


    if len(test_loader) == 0:
        return None
    else:
        if len(ensemble_checkpoints) == 1:
            return test_output
        else:
            preds_ensemble, _ = predict_ensemble(ensemble_checkpoints, test_loader)
            test_output = [compute_metrics(preds_ensemble, test_loader.dataset.y.astype(np.int32))]
            return test_output
    


@ray.remote(num_cpus=8, num_gpus=1)
class RunTrainingObject(object):
    def __init__(self, data_id, args, name, ensemble_size):
        self.data_id = data_id
        self.args = args
        self.name = name
        self.ensemble_size = ensemble_size

    def run_training(self):
        print('Using seed ', self.args.seed)
        return run_training(self.data_id, self.args, name=self.name, ensemble_size=self.ensemble_size)


def parallel_folds(dataset, args, data_id=None, n_folds=8, ensemble_size=1, return_individual_metric=None):
    ray.init(dashboard_host='0.0.0.0', ignore_reinit_error=True, )  # port=8265
    if data_id is not None:
        data_id = data_id
    else:
        data_id = ray.put(dataset)

    experiment_name = get_time_string()

    all_refs = []
    for ix_run, seed_fold in enumerate(range(args.all_seed, args.all_seed + n_folds)):
        args.seed = seed_fold
        remote_run_training = RunTrainingObject.options(num_cpus=args.num_workers,
                                                        num_gpus=args.num_gpus_per_fold).remote(
            data_id, args, name=experiment_name + '/fold_' + str(ix_run), ensemble_size=ensemble_size)
        ref = remote_run_training.run_training.remote()
        all_refs.append(ref)

    out = ray.get(all_refs)  # waits until all executions terminate

    if out[0] is None:  # no test set
        return

    all_metrics = [i[0] for i in out]
    print(all_metrics)
    metrics_aggregated = aggregate_metrics(all_metrics)

    print_save_aggregated_metrics(args, metrics_aggregated, experiment_name=experiment_name)

    return metrics_aggregated


def parallel_ensemble(data, args, data_id=None, ensemble_size=1):
    ray.init(dashboard_host='0.0.0.0', ignore_reinit_error=True, )  # port=8265
    if data_id is not None:
        data_id = data_id
    else:
        data_id = ray.put(data)

    experiment_name = get_time_string()

    all_refs = []
    args.seed = args.all_seed
    for ix_ensemble in range(ensemble_size):
        args.ensemble_incr_seed = ix_ensemble
        # always fold0 and ensemble_size=1 because the ensembling is parallel
        remote_run_training = RunTrainingObject.options(num_cpus=args.num_workers,
                                                        num_gpus=args.num_gpus_per_fold).remote(data_id, args,
                                                                                                name=experiment_name + '/fold_0/ensemble_' + str(
                                                                                                    ix_ensemble),
                                                                                                ensemble_size=1)
        ref = remote_run_training.run_training.remote()
        all_refs.append(ref)

    out = ray.get(all_refs)  # waits until all executions terminate

    if out[0] is None:  # no test set
        return

    all_metrics = [i[0] for i in out]
    metrics_aggregated = gneprop.utils.aggregate_metrics(all_metrics)
    print_save_aggregated_metrics(args, metrics_aggregated, experiment_name=experiment_name)

    return metrics_aggregated


def train_evaluate(args, additional_args=None):
    if additional_args is not None:
        for k, v in additional_args.items():  # update args
            args.__dict__[k] = v
    print('\n Training with arguments: ')
    print(additional_args)
    print('\n')

    test_output = run_training(data, args)[0]
    test_output_sd = {}
    for v, k in test_output.items():
        test_output_sd[v] = (k, 0)
    return test_output_sd


def train_evaluate_multiple_folds(args, data_id, n_folds, return_error: bool, additional_args=None):
    if additional_args is not None:
        for k, v in additional_args.items():  # update args
            args.__dict__[k] = v

    print('\n Training with arguments: ')
    print(additional_args)
    print('\n')

    metrics_aggregated = parallel_folds(None, args, data_id=data_id, n_folds=n_folds)

    test_output_sd = {}
    for v, k in metrics_aggregated.items():
        if return_error:
            test_output_sd[v] = (k['mean'], k['std'])
        else:
            test_output_sd[v] = (k['mean'], 0)
    return test_output_sd


def check_arguments(args):
    assert args.dataset_path is not None
    if args.parallel_folds > 1:
        assert args.parallel_ensemble == args.num_folds == 1
    if args.parallel_ensemble > 1:
        assert args.parallel_folds == args.num_ensemble == args.num_folds == 1
    if args.num_folds > 1:
        assert args.parallel_folds == 1
    if args.num_ensemble > 1:
        assert args.parallel_ensemble == 1
    if args.max_epochs is None:
        warnings.warn(
            "The argument: max_epochs was not set, using default value: 80",
            UserWarning,
        )
        args.max_epochs = 80

    if args.hparams_search_conf_path is not None:
        tune_hyperparameters_folds = OmegaConf.load(args.hparams_search_conf_path).experiment.tune_hyperparameters_folds
        if tune_hyperparameters_folds > 0:
            assert args.num_folds == args.num_ensemble == 1

    task = GNEPropTask(args.task)
    allowed_val_metrics = GNEPropTask.validation_names(task.get_metrics())
    if args.metric not in allowed_val_metrics:
        default_val_metric = GNEPropTask.validation_names(task.get_default_metric())
        warnings.warn(
            f"The argument: metric was set incorrectly, \
            for {args.task} either one of {allowed_val_metrics} \
            Using {default_val_metric} as default",
            UserWarning,
        )
        args.metric = default_val_metric

    # just one pretrain source
    pc = 0
    if args.pretrain_path is not None:
        pc += 1
    if args.supervised_pretrain_path is not None:
        pc += 1
    if args.supervised_pretrain_path_folds is not None:
        pc += 1
    assert pc <= 1

    if args.supervised_pretrain_path_folds is not None:
        num_folds = args.parallel_folds if args.parallel_folds > 1 else args.num_folds
        assert os.path.isdir(args.supervised_pretrain_path_folds)

        pretrained_checkpoints = gneprop.utils.get_checkpoint_paths(checkpoint_dir=args.supervised_pretrain_path_folds)
        assert len(pretrained_checkpoints) == num_folds

        pretrained_checkpoints_dict = dict()
        for i in range(len(pretrained_checkpoints)):
            fold_checkpoint = list(filter(lambda v: match(f'.+fold_{i}.+', v), pretrained_checkpoints))[0]
            pretrained_checkpoints_dict[i] = fold_checkpoint

        args.supervised_pretrain_paths = pretrained_checkpoints_dict
    else:
        args.supervised_pretrain_paths = None


def check_argument_post_data_loading(args, data):
    task = GNEPropTask(args.task)
    if task is GNEPropTask.MULTI_CLASSIFICATION:
        # check number of channels
        if args.out_channels == 1:
            num_classes = len(np.unique(np.array(data.y)))
            warnings.warn(
                f"The argument 'out_channels' should be > 1 for multi_classification task. Setting 'out_channels' equal to the number of distinct classes in the dataset ( = {num_classes})",
                UserWarning,
            )
            args.out_channels = num_classes

        # check class weights
        if args.class_weight is not None:
            assert args.out_channels == len(args.class_weight)

    if args.auto_loss_weight:
        if task is GNEPropTask.BINARY_CLASSIFICATION:
            assert args.pos_weight == 1
            num_pos = np.count_nonzero(data.y)
            pos_weight = (len(data) - num_pos) / num_pos
            args.pos_weight = pos_weight
            warnings.warn(
                f"auto_loss_weight: setting pos_weight = {args.pos_weight}"
            )
        elif task is GNEPropTask.MULTI_CLASSIFICATION:
            assert args.class_weight is None

            class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(data.y), y=data.y)
            args.class_weight = class_weight.tolist()
            warnings.warn(
                f"auto_loss_weight: setting class_weight = {args.class_weight}"
            )
        elif task is GNEPropTask.REGRESSION:
            raise ValueError('Regression task not compatible with --auto_loss_weight.')

    if args.split_type == 'index_predetermined':
        assert args.index_predetermined_file is not None
        with open(args.index_predetermined_file, 'rb') as rf:
            data.index_predetermined = pickle.load(rf)
        assert (len(data.index_predetermined) == args.num_folds) or (
                    len(data.index_predetermined) == args.parallel_folds), 'Number of folds not compatible with index predetermined file.'
    else:
        data.index_predetermined = None


## for hparams search
class HparamSearch():
    def __init__(self, args):
        h_conf = OmegaConf.load(args.hparams_search_conf_path)
        hparams_names = h_conf.search_space.keys()
        if args.pretrain_path:
            hparams_names = [_ for _ in hparams_names if _ not in ['hidden_size', 'depth']]
        self.args = args
        self.total_trials = h_conf.experiment.total_trials
        self.n_folds = h_conf.experiment.tune_hyperparameters_folds
        self.objective_name = h_conf.experiment.objective_name
        self.minimize = h_conf.experiment.minimize
        if self.n_folds > 0:
            self.use_error = h_conf.experiment.use_error
        else:
            self.use_error = None
        self.log_dir = h_conf.experiment.log_dir
        self.search_space = [
            {'name': hparam_name,
             'type': h_conf.search_space[hparam_name]['type'],
             'value_type': h_conf.search_space[hparam_name]['value_type'],
             'bounds': list(h_conf.search_space[hparam_name]['bounds']),
             'values': list(h_conf.search_space[hparam_name]['values']),
             'log_scale': h_conf.search_space[hparam_name]['log_scale'],
             }
            for hparam_name in hparams_names
        ]

        log_keys = []
        log_values = []
        for h in hparams_names:
            log_keys.append(h + '_space')
            f = 'bounds' if h_conf.search_space[h]['type'] == 'range' else 'values'
            v1 = ', '.join(str(_) for _ in h_conf.search_space[h][f])
            v2 = str(h_conf.search_space[h]['log_scale']) if h_conf.search_space[h]['type'] == 'range' else ''
            v = v1 + ', ' + v2
            log_values.append(v)

        self.search_space_log = dict(zip(log_keys, log_values))

        self._validate_conf()

    def search(self):
        if h_search.n_folds == 0:
            best_parameters = self._hyper_optimization()
        else:
            best_parameters = self._hyper_optimization_multiple_folds()

        print('Best parameters: ', best_parameters)

    def _validate_conf(self):
        if self.objective_name == 'test_mse':
            assert (args.task == 'regression') & (self.minimize is True)
        if self.objective_name == 'test_ap':
            assert self.minimize is False
        if not self.objective_name:
            self.objective_name = 'test_mse' if args.task == 'regression' else 'test_auc'

    def _log_best(self, best_parameters, means):
        row = dict()
        row.update({'total_trials': [self.total_trials]})
        row.update({'objective_name': [self.objective_name]})
        row.update({'minimize': [self.minimize]})
        row.update({'tune_hyperparameters_folds': [self.n_folds]})
        row.update(self.search_space_log)
        row.update(best_parameters)
        row.update(means)

        df = pd.DataFrame(row)
        df.to_csv(self.log_dir, mode='a', index=False, header=True)

    def _hyper_optimization(self):
        self.args.seed = self.args.all_seed
        train_evaluate_additional_args = partial(train_evaluate, self.args)

        best_parameters, values, experiment, model = optimize(
            parameters=self.search_space,
            evaluation_function=train_evaluate_additional_args,
            objective_name=self.objective_name,
            minimize=self.minimize,
            random_seed=self.args.all_seed,
            total_trials=self.total_trials
        )
        means, covariances = values
        self._log_best(best_parameters, means)
        print('Hyper, best values ', means, covariances)

        return best_parameters

    def _hyper_optimization_multiple_folds(self):
        ray.init(dashboard_host='0.0.0.0', ignore_reinit_error=True)  # 8265
        data_id = ray.put(data)

        train_evaluate_additional_args = partial(train_evaluate_multiple_folds, self.args, data_id, self.n_folds,
                                                 self.use_error)

        best_parameters, values, experiment, model = optimize(
            parameters=self.search_space,
            evaluation_function=train_evaluate_additional_args,
            objective_name=self.objective_name,
            minimize=self.minimize,
            random_seed=self.args.all_seed,
            total_trials=self.total_trials
        )
        means, covariances = values
        self._log_best(best_parameters, means)
        print('Hyper, best values ', means, covariances)
        return best_parameters


if __name__ == '__main__':
    parser = ArgumentParser()

    ###
    # add PROGRAM level args
    parser.add_argument('--split_type', default='random', const='random',
                        nargs='?',
                        choices=['random', 'scaffold', 'cluster', 'stratified', 'scaffold_dgl', 'scaffold_stratified',
                                 'fixed', 'fixed_test_scaffold_rest', 'fp', 'fp_scaffold', 'index_predetermined'])
    parser.add_argument('--index_predetermined_file', type=str, default=None)
    parser.add_argument('--num_workers', nargs='?', const=-1, default=1, type=int)
    parser.add_argument('--split_sizes', nargs=3, metavar=('train_size', 'validation_size', 'test_size'), default=None,
                        type=float)
    parser.add_argument('--pretrain_path', type=str, default=None, help='Path of unsupervised pretrained model')
    parser.add_argument('--supervised_pretrain_path', type=str, default=None,
                        help='Path of supervised pretrained model')
    parser.add_argument('--supervised_pretrain_path_folds', type=str, default=None,
                        help='Path of supervised pretrained model (multiple folds)')
    parser.add_argument("--invert_val_test", action="store_true", help='debug option')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--cluster_name', type=str, default=None)
    parser.add_argument('--parallel_folds', type=int, default=1)
    parser.add_argument('--num_gpus_per_fold', type=float, default=1.0)
    parser.add_argument('--parallel_ensemble', type=int, default=1)
    parser.add_argument('--num_folds', type=int, default=1)
    parser.add_argument('--num_ensemble', type=int, default=1)
    parser.add_argument('--all_seed', type=int, default=0)
    parser.add_argument("--use_horovod", action="store_true", help='Use horovod instead of ddp for multi-gpu')
    parser.add_argument("--ig_baseline_ratio", type=float, default=0.)
    parser.add_argument("--hparams_search_conf_path", type=str, default=None,
                        help='configuration file for setup up hparam search')
    parser.add_argument('--print_description', action="store_true", help='Toggle print of dataset statistics')
    parser.add_argument('--target_name', type=str, default='Activity', help='Store the name of the variable to predict')
    parser.add_argument('--additional_training_data', type=str, default=None)
    parser.add_argument('--augmentation', type=str, default=None, help='Path to the file describing the augmentation')
    parser.add_argument('--static_val_augmentation', type=str, default=None,
                        help='Path to the file describing the static validation augmentation')
    parser.add_argument("--val_multiplier", type=int, default=1, help='Multipler factor to increase the validation set')
    parser.add_argument('--augmentation_label', default='same', const='same',
                        nargs='?', choices=['same', 'zero_augmented'])
    parser.add_argument("--keep_last_batch", action="store_true")
    parser.add_argument("--wb_enable", action="store_true", help='Enable log with wandb')
    parser.add_argument("--wb_entity", type=str, default='antibiotics',
                        help='Speficy wandb entity (username/team name) who send the runs')
    parser.add_argument("--wb_project", type=str, default='test',
                        help='Speficy project name that current run belongs to for wandb logger')
    parser.add_argument("--eval_log", action="store_true",
                        help="Whether produce evaluation plots and dataframes in log scale")
    ###
    # add mpn_layers specific args
    parser = GNEprop.add_model_specific_args(parser)
    ###

    ###
    # add training specific args
    parser = Trainer.add_argparse_args(parser)
    ###

    args = parser.parse_args()

    # args.all_seed = 0
    seed_everything(args.all_seed)

    args.deterministic = False

    if args.pretrain_path:
        args.pretrain_path = prepare_pretrain_file(args.pretrain_path)

    check_arguments(args)
    print('All_args: ', args)

    fixed_split = args.split_type in {'fixed', 'fixed_test_scaffold_rest'}

    data = load_dataset_multi_format(args.dataset_path, cluster_name=args.cluster_name, fixed_split=fixed_split,
                                     target_name=args.target_name)

    check_argument_post_data_loading(args, data)

    # precomputes scaffolds efficiently
    if args.split_type in ['scaffold', 'scaffold_dgl', 'scaffold_stratified'] and not hasattr(data, 'mols'):
        print('Computing RDKit molecules...')
        data.mols = convert_smiles_to_mols(data.smiles, parallelize=True)
        if args.split_type in ['scaffold', 'scaffold_stratified']:
            data.scaffolds = scaffold_to_smiles(data.mols, use_indices=True, parallelize=True)

    args.mol_features_size = 0
    if args.use_mol_features:
        print("Adding mol features...")
        if args.mol_features in ['rdkit_normalized']:
            add_mol_features(data, mol_features_mode='rdkit_normalized', parallelize=True)
        elif args.mol_features == 'pretrain':
            if os.path.isfile(args.pretrain_path):
                model_pretrained = clr.SimCLR.load_from_checkpoint(args.pretrain_path)
            elif os.path.isdir(args.pretrain_path):
                pretrain_full_path = gneprop.utils.get_checkpoint_paths(checkpoint_dir=args.pretrain_path)[0]
                model_pretrained = clr.SimCLR.load_from_checkpoint(pretrain_full_path)
            else:
                raise ValueError("Invalid argument: pretrain path")
            model_pretrained.eval()
            model_pretrained.to(device='cuda:0')
            mol_features = model_pretrained.get_representations_dataset(data, use_projection_layers=1,
                                                                        use_batch_norm=True)
            data.mol_features = torch.tensor(mol_features, dtype=torch.float32)

        args.mol_features_size = data[0].mol_features.size(1)

    if args.extra_mol_features is not None:
        df = read_csv(args.dataset_path)
        file_mol_features_matrix = df[args.extra_mol_features].to_numpy()

        file_mol_features_tensor = torch.tensor(file_mol_features_matrix, dtype=torch.float32)
        if args.mol_features_size == 0:
            data.mol_features = file_mol_features_tensor
        else:
            data.mol_features = torch.hstack((data.mol_features, file_mol_features_tensor))

        args.mol_features_size = data[0].mol_features.size(1)

    additional_training_data = args.additional_training_data
    if additional_training_data is not None:
        additional_training_data = load_dataset_multi_format(additional_training_data)

    args.alternative_test_data = None

    # FOR HPARAM SEARCH
    if args.hparams_search_conf_path:
        h_search = HparamSearch(args)
        h_search.search()
        sys.exit(0)

    if args.parallel_folds > 1:
        parallel_folds(data, args, n_folds=args.parallel_folds, ensemble_size=args.num_ensemble)
        sys.exit(0)

    if args.parallel_ensemble > 1:
        parallel_ensemble(data, args, ensemble_size=args.parallel_ensemble)
        sys.exit(0)

    # not parallel but serial
    experiment_name = get_time_string()

    metrics = []
    for ix_run, seed_fold in enumerate(range(args.all_seed, args.all_seed + args.num_folds)):
        print('Using seed ', seed_fold)
        args.seed = seed_fold
        fold_output = run_training(data, args=args, name=experiment_name + '/fold_' + str(ix_run),
                                   additional_training_data=additional_training_data, ensemble_size=1)
        if fold_output is not None:
            metrics.append(fold_output[0])

    if len(metrics) == 0:
        sys.exit(0)
    else:
        metrics_aggregated = gneprop.utils.aggregate_metrics(metrics)
        print_save_aggregated_metrics(args, metrics_aggregated, experiment_name=experiment_name)
