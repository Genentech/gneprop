import data
import json
import math
from argparse import ArgumentParser
from typing import Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler
from sklearn import metrics
from torch import Tensor, device
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional import accuracy

import gneprop_pyg
from data import AugmentedDatasetPair, load_dataset_multi_format, MoleculeSubset
from models import GNEpropGIN
from gneprop import scaffold
from gneprop.augmentations import RemoveSubgraph, MoCL, MixedAugmentation
from gneprop.utils import get_time_string, get_accelerator
from tqdm import trange
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import numpy as np
from tqdm import tqdm

import GNEpropCPP

import time

from torch.cuda import nvtx

def train_test_k_folds_model(model, dataset, n_folds=10, use_projection_layers=0, split_type='scaffold'):
    nvtx.range_push('train_test_k_folds_model')
    all_test_ap = []
    all_test_auc = []
    for i in trange(n_folds):
        train_set, _, test_set = gneprop_pyg.split_data(dataset, split_type=split_type, sizes=(0.8, 0., 0.2), seed=i)

        train_loader = gneprop_pyg.convert_to_dataloader(train_set, num_workers=1)
        test_loader = gneprop_pyg.convert_to_dataloader(test_set, num_workers=1)

        reprs_train = model.get_representations_dataset(train_loader, use_projection_layers=use_projection_layers,
                                                        disable_progress_bar=True)
        reprs_test = model.get_representations_dataset(test_loader, use_projection_layers=use_projection_layers,
                                                       disable_progress_bar=True)
        clf = linear_model.LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000).fit(reprs_train,
                                                                                                  train_set.y)

        clf_preds = clf.predict(reprs_test)

        test_ap = metrics.average_precision_score(test_set.y, clf_preds)
        test_auc = metrics.roc_auc_score(test_set.y, clf_preds)
        all_test_ap.append(test_ap)
        all_test_auc.append(test_auc)
    nvtx.range_pop()
    return {'auc': np.mean(all_test_auc), 'ap': np.mean(all_test_ap)}


class LinearTransferOnlineEvaluator(Callback):
    def __init__(self, dataset, dict_args, num_folds=10, seed=0):
        super().__init__()

        self.seed = seed
        self.num_folds = num_folds
        self.transfer_dataset = dataset

        self.transfer_dataset.mols = gneprop_pyg.convert_smiles_to_mols(self.transfer_dataset.smiles)
        self.transfer_dataset.scaffolds = scaffold.scaffold_to_smiles(self.transfer_dataset.mols, use_indices=True)

        self.dict_args = dict_args
        self.transfer_model = SimCLR(**dict_args)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.transfer_model.load_state_dict(pl_module.state_dict())
        self.transfer_model.eval()
        self.transfer_model.to(device=pl_module.device)

        with torch.no_grad():
            transfer_results = train_test_k_folds_model(self.transfer_model, self.transfer_dataset, n_folds=self.num_folds,
                                                        use_projection_layers=0, split_type='scaffold')

        pl_module.log('transfer_auc', transfer_results['auc'], prog_bar=True, sync_dist=True)
        pl_module.log('transfer_ap', transfer_results['ap'], prog_bar=True, sync_dist=True)


class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone().contiguous()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class Projection(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class ProjectionLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.model = nn.Linear(self.input_dim, self.output_dim, bias=False)

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(pl.LightningModule):
    def __init__(self, **kwargs):

        super().__init__()

        # backcompatibility, add defaults
        self._add_default_args(kwargs)

        self.save_hyperparameters()

        self.encoder = self.init_model(self.hparams.model_hidden_size, self.hparams.model_depth, jk=self.hparams.jk, num_readout_layers=self.hparams.num_readout_layers)

        if self.hparams.jk == 'none':
            output_model_dim = self.hparams.model_hidden_size
        elif self.hparams.jk == 'cat':
            output_model_dim = self.hparams.model_hidden_size * (self.hparams.model_depth + 1)
        else:
            raise NotImplementedError

        if self.hparams.linear_projection:
            self.projection = ProjectionLinear(input_dim=output_model_dim,
                                               output_dim=self.hparams.project_output_dim)
        else:
            self.projection = Projection(input_dim=output_model_dim,
                                         hidden_dim=self.hparams.project_hidden_dim,
                                         output_dim=self.hparams.project_output_dim)

        # compute iters per epoch
        global_batch_size = self.hparams.num_nodes * self.hparams.gpus * self.hparams.batch_size if self.hparams.gpus > 0 else self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

        self.predict_with_projection = False

    def on_train_start(self):
        self.print(f'Training model: {self.hparams.experiment_name}')

    def on_train_epoch_start(self):
        if self.hparams.timing_log_path is not None:
            file = open(self.hparams.timing_log_path, "a")
            file.write("on_train_epoch_start "+str(time.time())+" "+time.asctime()+"\n")
            file.close()

    def on_train_epoch_end(self):
        if self.hparams.timing_log_path is not None:
            file = open(self.hparams.timing_log_path, "a")
            file.write("on_train_epoch_end "+str(time.time())+" "+time.asctime()+"\n")
            file.close()

    def on_validation_epoch_start(self):
        if self.hparams.timing_log_path is not None:
            file = open(self.hparams.timing_log_path, "a")
            file.write("on_validation_epoch_start "+str(time.time())+" "+time.asctime()+"\n")
            file.close()

    def on_validation_epoch_end(self):
        if self.hparams.timing_log_path is not None:
            file = open(self.hparams.timing_log_path, "a")
            file.write("on_validation_epoch_end "+str(time.time())+" "+time.asctime()+"\n")
            file.close()

    def init_model(self, hidden_size, depth, num_readout_layers=1, jk='cat'):
        if self.hparams.pretrained_model is not None:
            return self.load_pretrain_classifier(self.hparams.pretrained_model, self.hparams.freeze_classifier)
        return GNEpropGIN(in_channels=self.hparams.node_feat_size,
                          edge_dim=self.hparams.edge_feat_size,
                          hidden_channels=hidden_size,
                          ffn_hidden_channels=None,
                          num_layers=depth,
                          out_channels=1, dropout=0.13,
                          num_readout_layers=num_readout_layers,
                          mol_features_size=0,
                          aggr='mean',
                          jk=jk)

    def _add_default_args(self, kwargs):
        model_parser = self.add_model_specific_args(ArgumentParser())
        default_args = model_parser.parse_known_args()[0]._get_kwargs()
        for k, v in default_args:
            if k not in kwargs:
                kwargs[k] = v

    def forward(self, data):
        o = self.encoder.compute_representations(data.x, data.edge_index, data.edge_attr, data.batch)
        if self.predict_with_projection:
            o = self.projection(o)
        return o

    def shared_step(self, batch):
        data1, data2 = batch

        # get h representations
        h1 = self.encoder.compute_representations(data1.x, data1.edge_index, data1.edge_attr, data1.batch)
        h2 = self.encoder.compute_representations(data2.x, data2.edge_index, data2.edge_attr, data2.batch)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('train_loss', loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        return loss

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

        if self.hparams.optim == 'lars':
            optimizer = LARS(
                params,
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.hparams.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.hparams.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.hparams.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def nt_xent_loss(self, out_1, out_2, temperature=0.1, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

    def get_representations(self, data, use_projection_layers=0, use_batch_norm=False):
        assert use_projection_layers >= -1

        o = self.encoder.compute_representations(data.x, data.edge_index, data.edge_attr, data.batch)
        if use_projection_layers == 0:
            return o
        elif use_projection_layers == -1:
            o = self.projection(o)
            return o
        else:
            part_ix = 3 * (use_projection_layers - 1) + 1  # account for intermediate non-linear layers
            if use_batch_norm:
                part_ix += 1
            part_projection = self.projection.model[:part_ix]
            o = part_projection(o)
            return o

    def get_representations_dataset(self, dataset, use_projection_layers=0, use_batch_norm=False,
                                    disable_progress_bar=False, batch_size=100, num_workers=16):
        dataloader = gneprop_pyg.convert_to_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)
        outs = []
        with torch.no_grad():
            for i in tqdm(dataloader, position=0, leave=True, disable=disable_progress_bar):
                out = self.get_representations(i, use_projection_layers=use_projection_layers,
                                               use_batch_norm=use_batch_norm).cpu().numpy()
                outs.append(out)
        return np.vstack(outs)

    def get_representations_smiles(self, smiles, use_projection_layers=0, disable_progress_bar=False):
        if isinstance(smiles, str):
            smiles = [smiles]
        dataset = data.MolDatasetOD(smiles_list=smiles, code_version=self.hparams.code_version)
        dataloader = gneprop_pyg.convert_to_dataloader(dataset)
        outs = []
        with torch.no_grad():
            for i in tqdm(dataloader, position=0, leave=True, disable=disable_progress_bar):
                out = self.get_representations(i, use_projection_layers=use_projection_layers).cpu().numpy()
                outs.append(out)
        return np.vstack(outs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--exclude_bn_bias', action='store_true', help="exclude bn/bias from weight decay")
        parser.add_argument("--optim", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs")
        parser.add_argument('--lr', type=float, default=1.e-05)
        parser.add_argument("--weight_decay", default=0., type=float, help="weight decay")
        parser.add_argument("--model_hidden_size", default=300, type=int)
        parser.add_argument("--project_hidden_dim", default=512, type=int)
        parser.add_argument("--project_output_dim", default=128, type=int)
        parser.add_argument("--num_readout_layers", default=1, type=int)
        parser.add_argument("--model_depth", default=5, type=int)
        parser.add_argument('--jk', default='cat', const='cat',
                            nargs='?', choices=['cat', 'none'])
        parser.add_argument("--linear_projection", action="store_true")
        parser.add_argument("--train_resample_number", default=0, type=int)
        parser.add_argument('--log_directory', type=str, default='gneprop_clr/logs', help='Where logs are saved')
        parser.add_argument("--linear_evaluation_dataset", type=str, default=None)
        parser.add_argument("--linear_evaluation_folds", type=int, default=10)
        parser.add_argument("--shared_memory", action="store_true")
        parser.add_argument('--pretrained_model', type=str, default=None,)
        parser.add_argument("--node_feat_size", type=int, default=55)
        parser.add_argument('--edge_feat_size', type=int, default=12)
        parser.add_argument('--fprecision', type=int, default=32)
        parser.add_argument('--compatible_features', type=bool, default=True)
        parser.add_argument('--code_version', type=int, default=2)
        parser.add_argument('--timing_log_path', type=str, default=None)
        return parser


def run_training(args):
    nvtx.range_push('run_training.part0')
    experiment_name = get_time_string()
    args.experiment_name = experiment_name

    aug1 = RemoveSubgraph(fraction_nodes_to_remove=0.2, code_version=args.code_version)
    aug2 = RemoveSubgraph(fraction_nodes_to_remove=0.2, code_version=args.code_version)

    print('Using augmentations')
    print(f'Augmentation1: {aug1}')
    print(f'Augmentation2: {aug2}')

    args.arg1 = str(aug1)
    args.arg2 = str(aug2)

    nvtx.range_pop()
    nvtx.range_push('run_training.part1')

    if args.timing_log_path is not None:
        file = open(args.timing_log_path, "a")
        file.write("Beginning training with "+str(args.gpus)+" GPUs "+str(args.num_workers)+" workers, batch size "+str(args.batch_size)+"\n")
        file.write("Start time "+str(time.time())+" "+time.asctime()+"\n")
        file.close()

    dataset = load_dataset_multi_format(args.dataset_path, code_version=args.code_version)

    train_set, val_set, _ = data.split_data(dataset, split_type='random', sizes=(0.90, 0.05, 0.05), seed=0)

    train_set = data.MolDatasetOD(train_set.smiles, code_version=args.code_version)
    val_set = data.MolDatasetOD(val_set.smiles, code_version=args.code_version)

    augmented_train_set = AugmentedDatasetPair(train_set, args.code_version, aug1, aug2)
    augmented_val_set = AugmentedDatasetPair(val_set, args.code_version, aug1, aug2)

    ### set args
    args.num_samples = len(train_set) if args.train_resample_number == 0 else args.train_resample_number

    nvtx.range_pop()
    nvtx.range_push('run_training.SimCLR')

    dict_args = vars(args)
    model = SimCLR(**dict_args)

    nvtx.range_pop()
    nvtx.range_push('run_training.MolDatasetResample')

    datamodule = data.MolDatasetResample(train_set=augmented_train_set, val_set=augmented_val_set, test_set=None,
                                         batch_size=args.batch_size, num_workers=args.num_workers,
                                         resample_number=args.train_resample_number, pin_memory=False)

    nvtx.range_pop()
    nvtx.range_push('run_training.TensorBoardLogger')

    logger = TensorBoardLogger(args.log_directory, name=experiment_name, default_hp_metric=False)

    nvtx.range_pop()
    nvtx.range_push('run_training.LearningRateMonitor')

    lr_monitor = LearningRateMonitor(logging_interval="step")

    nvtx.range_pop()
    nvtx.range_push('run_training.ModelCheckpoint')

    model_checkpoint_val_loss = ModelCheckpoint(
        monitor='val_loss',
        save_last=True,
        filename="{epoch}-{step}-{val_loss:.2f}"
    )

    callbacks = [model_checkpoint_val_loss, lr_monitor]

    # linear evaluator callback
    args.linear_evaluation = args.linear_evaluation_dataset is not None
    nvtx.range_pop()
    if args.linear_evaluation:
        nvtx.range_push('run_training.linear_evaluation')
        linear_evaluation_dataset = data.load_dataset_multi_format(args.linear_evaluation_dataset)
        linear_evaluation_callback = LinearTransferOnlineEvaluator(linear_evaluation_dataset, dict_args, num_folds=args.linear_evaluation_folds)
        callbacks.append(linear_evaluation_callback)

        model_checkpoint_transfer = ModelCheckpoint(
            monitor='transfer_ap',
            mode='max',
            filename="{epoch}-{step}-{transfer_ap:.2f}"
        )
        callbacks.append(model_checkpoint_transfer)
        nvtx.range_pop()

    nvtx.range_push('run_training.part8')
    log_frequency = 10  # default: 1
    log_every_n_steps = 50 * log_frequency
    flush_logs_every_n_steps = 100 * log_frequency

    accelerator = 'horovod' if args.use_horovod else get_accelerator(args.gpus)

    GNEpropCPP.configCPPOptions(args.compatible_features, [], 0, args.fprecision==16)

    nvtx.range_pop()
    nvtx.range_push('run_training.Trainer')

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps, #None if args.max_steps == -1 else args.max_steps,
        #gpus=1 if accelerator == 'horovod' else args.gpus,
        num_nodes=1 if accelerator == 'horovod' else args.num_nodes,
        accelerator='gpu',
        devices=1 if accelerator == 'horovod' else args.gpus,
        strategy=None if (accelerator == 'horovod' or args.gpus == 1) else 'ddp',
        sync_batchnorm=True if args.gpus > 1 else False,
        callbacks=callbacks,
        logger=logger,
        reload_dataloaders_every_n_epochs=(1 if bool(args.train_resample_number) else 0),
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        #flush_logs_every_n_steps=flush_logs_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        precision=args.fprecision
    )
    nvtx.range_pop()

    trainer.fit(model, datamodule=datamodule)

    if args.timing_log_path is not None:
        file = open(args.timing_log_path, "a")
        file.write("Finish time "+str(time.time())+" "+time.asctime()+"\n")
        file.write("Finished training with "+str(args.gpus)+" GPUs "+str(args.num_workers)+" workers, batch size "+str(args.batch_size)+"\n\n")
        file.close()


# python clr.py --gpu 1
if __name__ == '__main__':
    nvtx.range_push('clr.py.main')
    parser = ArgumentParser()

    ###
    # add PROGRAM level args
    parser.add_argument('--num_workers', nargs='?', const=-1, default=1, type=int)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument("--use_horovod", action="store_true", help='Use horovod instead of ddp for multi-gpu')
    ###

    # add mpn_layers specific args
    parser = SimCLR.add_model_specific_args(parser)

    # add training specific args
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    if args.compatible_features or args.code_version < 2:
        args.node_feat_size = 133

    nvtx.range_pop()

    run_training(args)

