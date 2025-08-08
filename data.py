from descriptastorus.descriptors import rdNormalizedDescriptors  # needs to be first import

import itertools
import math
from math import isclose
import os.path as osp
import pickle
from pathlib import Path
from random import Random
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable, Union, Iterable

import dgllife
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import warnings
from dgllife.utils import SingleTaskStratifiedSplitter
from dgllife.utils import smiles_to_bigraph, mol_to_bigraph
from dgllife.utils.mol_to_graph import construct_bigraph_from_mol
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import preprocessing
from torch.utils.data import Subset, ConcatDataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange
from tqdm.contrib.concurrent import process_map

import chemprop_featurizer
from gneprop.augmentations import Augmentation, AugmentationType
from gneprop.featurization import m_to_bigraph, sm_to_bigraph, smiles_to_data, mol_to_data
from gneprop.scaffold import scaffold_to_smiles
from gneprop.utils import s3_setup, canonical_order
import pytorch_lightning as pl
from multiprocessing import shared_memory
from dgllife.utils import ScaffoldSplitter

from functools import lru_cache
import logging

from torch_geometric.transforms import BaseTransform
from torch import Tensor
from torch_geometric.data.batch import Batch

from functools import wraps
from typing import Callable

def auto_move_data(fn: Callable) -> Callable:
    """
    Decorator for :class:`~pytorch_lightning.core.lightning.LightningModule` methods for which
    input arguments should be moved automatically to the correct device.
    It as no effect if applied to a method of an object that is not an instance of
    :class:`~pytorch_lightning.core.lightning.LightningModule` and is typically applied to ``__call__``
    or ``forward``.

    Args:
        fn: A LightningModule method for which the arguments should be moved to the device
            the parameters are on.

    Example::

        # directly in the source code
        class LitModel(LightningModule):

            @auto_move_data
            def forward(self, x):
                return x

        # or outside
        LitModel.forward = auto_move_data(LitModel.forward)

        model = LitModel()
        model = model.to('cuda')
        model(torch.zeros(1, 3))

        # input gets moved to device
        # tensor([[0., 0., 0.]], device='cuda:0')

    """

    @wraps(fn)
    def auto_transfer_args(self, *args, **kwargs):
        from pytorch_lightning.core.lightning import LightningModule
        if not isinstance(self, LightningModule):
            return fn(self, *args, **kwargs)

        args, kwargs = self.transfer_batch_to_device((args, kwargs))
        return fn(self, *args, **kwargs)

    return auto_transfer_args


class VirtualNode(BaseTransform):
    r"""Appends a virtual node to the given homogeneous graph that is connected
    to all other nodes, as described in the `"Neural Message Passing for
    Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper
    (functional name: :obj:`virtual_node`).
    The virtual node serves as a global scratch space that each node both reads
    from and writes to in every step of message passing.
    This allows information to travel long distances during the propagation
    phase.

    Node and edge features of the virtual node are added as zero-filled input
    features.
    Furthermore, special edge types will be added both for in-coming and
    out-going information to and from the virtual node.
    """
    def __call__(self, data: Data) -> Data:
        num_nodes, (row, col) = data.num_nodes, data.edge_index
        edge_type = data.get('edge_type', torch.zeros_like(row))

        arange = torch.arange(num_nodes, device=row.device)
        full = row.new_full((num_nodes, ), num_nodes)
        row = torch.cat([row, arange, full], dim=0)
        col = torch.cat([col, full, arange], dim=0)
        edge_index = torch.stack([row, col], dim=0)

        new_type = edge_type.new_full((num_nodes, ), int(edge_type.max()) + 1)
        edge_type = torch.cat([edge_type, new_type, new_type + 1], dim=0)

        for key, value in data.items():
            if key == 'edge_index' or key == 'edge_type' or key == 'y':
                continue

            if isinstance(value, Tensor):
                dim = data.__cat_dim__(key, value)
                size = list(value.size())

                fill_value = None
                if key == 'edge_weight':
                    size[dim] = 2 * num_nodes
                    fill_value = 1.
                elif data.is_edge_attr(key):
                    size[dim] = 2 * num_nodes
                    fill_value = 0.
                elif data.is_node_attr(key):
                    size[dim] = 1
                    fill_value = 0.

                if fill_value is not None:
                    new_value = value.new_full(size, fill_value)
                    data[key] = torch.cat([value, new_value], dim=dim)

        data.edge_index = edge_index
        data.edge_type = edge_type

        if 'num_nodes' in data:
            data.num_nodes = data.num_nodes + 1

        return data


def split_data(dataset, split_type='random', sizes=(0.8, 0.1, 0.1), seed=0, additional_training_data=None):
    """
    Split dataset in train/val/test sets, using a specific splitting strategy.
    Each random seed leads to a different split.

    :param dataset: Dataset to be split
    :param split_type: Split type
    :param sizes: 3-tuple with training/val/test fractions
    :param seed: Random seed
    :param additional_training_data: Additional dataset to be added to the training split
    :return: 3-tuple with training/val/test datasets
    """
    assert isclose(np.array(sizes).sum(), 1, abs_tol=1e-03)

    if split_type == 'random':
        train, val, test = random_split(dataset, sizes=sizes, seed=seed)
    elif split_type in ['scaffold', 'scaffold_stratified', 'scaffold_dgl']:
        if not hasattr(dataset, 'mols'):
            print('Computing RDKit molecules...')
            dataset.mols = convert_smiles_to_mols(dataset.smiles)
        if split_type == 'scaffold':
            train, val, test = scaffold_split(dataset, sizes=sizes, seed=seed)
        elif split_type == 'scaffold_stratified':
            train, val, test = scaffold_stratified_split(dataset, sizes=sizes, seed=seed)
        elif split_type == 'scaffold_dgl':
            train, val, test = scaffold_split_dgl(dataset, sizes=sizes)
    elif split_type == 'stratified':
        train, val, test = stratified_split(dataset, sizes=sizes, seed=seed)
    elif split_type == 'cluster':
        train, val, test = cluster_split(dataset, sizes=sizes, seed=seed)
    elif split_type in ['fixed_test_scaffold_rest', 'fixed']:
        train, val, test = fixed_test_scaffold_rest(dataset, seed=seed)
    elif split_type == 'index_predetermined':
        train, val, test = index_predetermined_split(dataset, seed=seed)
    else:
        raise ValueError('split_type not implemented.')

    if additional_training_data is not None:
        train = ConcatMolDataset((train, additional_training_data))

    return train, val, test


def index_predetermined_split(dataset, seed):
    split_indices = dataset.index_predetermined[seed]

    if len(split_indices) != 3:
        raise ValueError('Split indices must have three splits: train, validation, and test')

    train_ix, val_ix, test_ix = split_indices

    train = MoleculeSubset(dataset, train_ix)
    val = MoleculeSubset(dataset, val_ix)
    test = MoleculeSubset(dataset, test_ix)
    return train, val, test


class MolDataset(InMemoryDataset):
    def __init__(self, root, file_name, transform=None, pre_transform=None):
        super(MolDataset, self).__init__(root, transform, pre_transform)
        self.file_name = file_name
        self.data, self.slices, self.smiles, self.mols = torch.load(self.processed_paths[0])
        self.clusters = None

    @property
    def processed_file_names(self):
        return [self.file_name]

    @property
    def y(self):
        return self.data.y

    @property
    def smiles_subset(self):
        return self.smiles

    @staticmethod
    def load_dataset(root, path):
        ds = MolDataset(root, path)
        return ds

    @staticmethod
    def load_from_full_path(full_path):
        root, path = full_path.split('processed/')
        ds = MolDataset(root, path)
        return ds


class MolDatasetLightWeight(InMemoryDataset):
    def __init__(self, root, file_name, transform=None, pre_transform=None):
        super(MolDatasetLightWeight, self).__init__(root, transform, pre_transform)
        self.file_name = file_name
        self.data, self.slices, self.smiles = torch.load(self.processed_paths[0])
        self.clusters = None

    @property
    def processed_file_names(self):
        return [self.file_name]

    @property
    def y(self):
        return self.data.y

    @property
    def smiles_subset(self):
        return self.smiles

    @staticmethod
    def load_dataset(root, path):
        ds = MolDataset(root, path)
        return ds

    @staticmethod
    def load_from_full_path(full_path):
        root, path = full_path.split('processed/')
        ds = MolDatasetLightWeight(root, path)
        return ds


class MolDatasetOD(torch.utils.data.Dataset):
    def __init__(self, smiles_list, y_list=None, cluster_list=None, separate_label=False, include_smiles=False,
                 fixed_list=None, enable_vn=False, legacy=True):
        self.smiles = smiles_list
        self.ids = self.smiles  # for deepchem compatibility
        self.y = y_list
        self.clusters = cluster_list
        if fixed_list is not None:
            self.fixed_list = fixed_list

        if self.y is not None:
            assert len(self.smiles) == len(self.y)

        if self.clusters is not None:
            assert len(self.smiles) == len(self.clusters)

        self.precomputed = False
        self.data_list = None  # precomputed data

        self.separate_label = separate_label
        self.include_smiles = include_smiles

        self.enable_vn = enable_vn
        if self.enable_vn:
            self.vn = VirtualNode()
        self.legacy = legacy

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        if self.precomputed:
            return self.data_list[idx]

        n = self.smiles_to_data(self.smiles[idx],
                                torch.tensor(self.y[idx]).float() if self.y is not None else None,
                                self.mol_features[idx].unsqueeze(0) if hasattr(self, 'mol_features') else None, 
                                self.legacy)

        if self.enable_vn:
            n = self.vn(n)

        if self.include_smiles:
            n.smiles = self.smiles[idx]

        if self.separate_label:
            return n, (n.y).long()
        else:
            return n

    def precompute(self, parallelize=False):
        print('Precomputing data...')
        if parallelize:
            self.data_list = process_map(self.__getitem__, range(len(self.smiles)), max_workers=8, chunksize=256)
        else:
            self.data_list = [self.__getitem__(i) for i in tqdm(range(len(self.smiles)))]

        self.precomputed = True

    @staticmethod
    def smiles_to_data(s, label=None, mol_features=None, legacy=True):
        return smiles_to_data(s, label=label, mol_features=mol_features, legacy=legacy)

    @staticmethod
    def mol_to_data(m, label=None, mol_features=None):
        return mol_to_data(m, label=label, mol_features=mol_features)

    @staticmethod
    def load_df_dataset(df, cluster_name=None, smiles_name='SMILES', target_name='Activity', fixed_split=False, legacy=True):
        dataset = df
        smiles = dataset[smiles_name].values

        if target_name in dataset.columns:
            y = dataset[target_name].values
        else:
            y = None
        if cluster_name is not None:
            try:
                clusters = dataset[cluster_name].values
            except AttributeError:
                clusters = None
        else:
            clusters = None

        if fixed_split:
            fixed_list = dataset.Fixed.values if 'Fixed' in dataset.columns else dataset.Traingp.values  ## for backcomp
        else:
            fixed_list = None

        return MolDatasetOD(smiles_list=smiles, y_list=y, cluster_list=clusters, fixed_list=fixed_list, legacy=legacy)

    @staticmethod
    def load_csv_dataset(csv_file, cluster_name=None, smiles_name='SMILES', target_name='Activity', fixed_split=False, legacy=True):
        dataset = pd.read_csv(csv_file)
        return MolDatasetOD.load_df_dataset(dataset, cluster_name=cluster_name, smiles_name=smiles_name,
                                            target_name=target_name, fixed_split=fixed_split, legacy=legacy)

    @staticmethod
    def load_ftr_dataset(ftr_file, cluster_name=None, smiles_name='SMILES', target_name='Activity', fixed_split=False, legacy=True):
        dataset = pd.read_feather(ftr_file)
        return MolDatasetOD.load_df_dataset(dataset, cluster_name=cluster_name, smiles_name=smiles_name,
                                            target_name=target_name, fixed_split=fixed_split, legacy=legacy)


class ConcatMolDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Iterable[MolDatasetOD]):
        super().__init__(datasets=datasets)

        self.smiles = np.concatenate([i.smiles for i in datasets])
        self.y = np.concatenate([i.y for i in datasets])
        self.id = self.smiles
        self.include_smiles = any([i.include_smiles for i in datasets])
        self.separate_label = any([i.separate_label for i in datasets])


class MolDatasetODBaseline(MolDatasetOD):
    def __init__(self, smiles_list, choices=None):
        if choices is None:
            choices = [0, 1]
        super().__init__(smiles_list, y_list=np.random.choice(a=choices, size=(len(smiles_list))), cluster_list=None)

    def __getitem__(self, idx):
        n = super().__getitem__(idx)
        n.x = torch.zeros_like(n.x)
        n.edge_attr = torch.zeros_like(n.edge_attr)
        return n

    @staticmethod
    def sample_baseline_dataset(dataset, frac=0.2):
        pos_ix = np.nonzero(dataset.y)[0]  # only positive indices
        num_to_select = int(frac * len(pos_ix))
        rng = np.random.default_rng()
        replace = False if frac <= 1.0 else True
        selected = rng.choice(pos_ix, num_to_select, replace=replace)

        sel_smiles = dataset.smiles[selected]

        return MolDatasetODBaseline(sel_smiles, choices=[0, 1])

    @staticmethod
    def sample_baseline_dataset_multiclass(dataset, frac=0.2):
        pos_ix = np.nonzero(dataset.y)[0]  # only non-zero indices
        num_to_select = int(frac * len(pos_ix))
        rng = np.random.default_rng()
        selected = rng.choice(pos_ix, num_to_select, replace=False)

        sel_smiles = dataset.smiles[selected]

        return MolDatasetODBaseline(sel_smiles, choices=np.unique(np.array(dataset.y)))


class MolDatasetResample(pl.LightningDataModule):
    def __init__(self, train_set, val_set, test_set, batch_size=256, num_workers=8, pin_memory=False,
                 resample_number=0, meta_dataset=None, meta_batch_size=128):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resample_number = resample_number
        self.pin_memory = pin_memory

        self.meta_dataset = meta_dataset
        self.meta_batch_size = meta_batch_size

    def train_dataloader(self):
        if self.resample_number > 0:
            self.resample_ix = np.random.choice(range(len(self.train_set)), self.resample_number, replace=False)
            self.resample_smiles = self.train_set.smiles[self.resample_ix]
            self.train_set_resample = MoleculeSubset(self.train_set, self.resample_ix)
            return DataLoader(self.train_set_resample, batch_size=self.batch_size, shuffle=True,
                              num_workers=self.num_workers, pin_memory=self.pin_memory)
        else:
            train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                              pin_memory=self.pin_memory)
            if self.meta_dataset is not None:
                transfer_loader = DataLoader(self.meta_dataset, batch_size=self.meta_batch_size, shuffle=True, num_workers=1, pin_memory=True)
                return train_loader, transfer_loader
            else:
                return train_loader

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, aug: Union[Augmentation, None], aug_behavior: str = 'same', static=False):
        """
        Encapsulates a Dataset applying augmentations on-the-fly.
        :param dataset: MolDatasetOD.
        :param aug: Augmentation.
        :param aug_behavior: How to handle labels for augmented samples.
            'same' (default): keeps labels untouched.
            'zero_augmented': sets to 0 labels of augmented samples, keeps labels of skipped samples untouched.
        :param static: Whether to compute the augmentation statically. Defaults to False (dynamic augmentation).
        """
        self.dataset = dataset
        self.aug = aug
        self.aug_behavior = aug_behavior
        self.static = static
        if self.aug:
            self.aug.return_applied_flag = True

        if self.static:
            self._static_compute()

    def __len__(self):
        return len(self.dataset)

    def _change_label(self, n, applied_flag):
        if self.aug_behavior == 'zero_augmented' and n.y is not None and applied_flag:
            n.y = torch.tensor(0.).float()
        return n

    def _static_compute(self):
        print('Computing static val augmentations...')
        self._static_dataset = [self._augment_idx(i) for i in trange(len(self), position=0, leave=True)]

    def _augment_idx(self, idx):
        if self.aug is None:
            return self.dataset[idx]
        elif self.aug.augmentation_type is AugmentationType.PYG_GRAPH:
            n = self.dataset[idx]
            n, applied_flag = self.aug(n)
            n = self._change_label(n, applied_flag)
            return n
        elif self.aug.augmentation_type is AugmentationType.RDKIT_MOL:
            smi = self.dataset.smiles[idx]
            mol = Chem.MolFromSmiles(smi)
            mol_aug, applied_flag = self.aug(mol)
            try:
                n = MolDatasetOD.mol_to_data(mol_aug,
                                             torch.tensor(self.y[idx]).float() if self.y is not None else None,
                                             self.mol_features[idx].unsqueeze(0) if hasattr(self,
                                                                                            'mol_features') else None)
                n = self._change_label(n, applied_flag)
            except KeyError:
                warnings.warn(
                    f'Error: augmenting molecule {smi}',
                    UserWarning,
                )
                n = MolDatasetOD.mol_to_data(mol,
                                             torch.tensor(self.y[idx]).float() if self.y is not None else None,
                                             self.mol_features[idx].unsqueeze(0) if hasattr(self,
                                                                                            'mol_features') else None)

            if self.include_smiles:
                n.smiles = self.smiles[idx]

            if self.separate_label:
                return n, (n.y).long()
            else:
                return n
        elif self.aug.augmentation_type is AugmentationType.MIXED:
            smi = self.dataset.smiles[idx]
            mol = Chem.MolFromSmiles(smi)
            mol = canonical_order(mol)
            n, applied_flag = self.aug(mol)
            if not applied_flag:
                n = MolDatasetOD.mol_to_data(mol,
                                             torch.tensor(self.y[idx]).float() if self.y is not None else None,
                                             self.mol_features[idx].unsqueeze(0) if hasattr(self,
                                                                                            'mol_features') else None)
                if self.include_smiles:
                    n.smiles = self.smiles[idx]

                if self.separate_label:
                    return n, (n.y).long()
            else:
                n.y = torch.tensor(self.y[idx]).float() if self.y is not None else None
                n.mol_features = self.mol_features[idx].unsqueeze(0) if hasattr(self, 'mol_features') else None

            n = self._change_label(n, applied_flag)

            return n

    def __getitem__(self, idx):
        if self.static:
            return self._static_dataset[idx]
        else:
            return self._augment_idx(idx)

    def __getattr__(self, item):
        return getattr(self.dataset, item)


class AugmentedDatasetPair(torch.utils.data.Dataset):
    def __init__(self, dataset, aug1: Union[Augmentation, None], aug2: Union[Augmentation, None]):
        self.dataset = dataset
        self.smiles = dataset.smiles
        self.y = dataset.y
        self.clusters = dataset.clusters
        self.aug1 = aug1
        self.aug2 = aug2

        if (self.aug1 is AugmentationType.MIXED and self.aug2 is not AugmentationType.MIXED) or (
                self.aug2 is AugmentationType.MIXED and self.aug1 is not AugmentationType.MIXED):
            raise NotImplementedError(
                'Current implementation for mixed augmentations only support both mixed or none mixed.')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # high efficient to minimize transformations

        # special case for both mixed augmentations
        if self.aug1.augmentation_type is AugmentationType.MIXED and self.aug2.augmentation_type is AugmentationType.MIXED:
            smi = self.dataset.smiles[idx]
            mol = Chem.MolFromSmiles(smi)

            n1 = self.aug1(mol)
            n2 = self.aug2(mol)
            return n1, n2

        # all other cases
        graph = None
        if self.aug1.augmentation_type is AugmentationType.PYG_GRAPH or self.aug2.augmentation_type is AugmentationType.PYG_GRAPH:
            graph = self.dataset[idx]

        mol = None
        if self.aug1.augmentation_type is AugmentationType.RDKIT_MOL or self.aug2.augmentation_type is AugmentationType.RDKIT_MOL:
            smi = self.dataset.smiles[idx]
            mol = Chem.MolFromSmiles(smi)

        if self.aug1 is None or self.aug2 is None:
            if graph is None:
                if mol is not None:
                    graph = MolDatasetOD.mol_to_data(mol)
                else:
                    graph = self.dataset[idx]

        if self.aug1 is None:
            n1 = graph
        elif self.aug1.augmentation_type is AugmentationType.PYG_GRAPH:
            n1 = self.aug1(graph)
        elif self.aug1.augmentation_type is AugmentationType.RDKIT_MOL:
            try:
                mol_aug1 = self.aug1(mol)
                n1 = MolDatasetOD.mol_to_data(mol_aug1)
            except KeyError:
                print(f'Error: loading molecule {smi}')
                n1 = MolDatasetOD.mol_to_data(mol)

        if self.aug2 is None:
            n2 = graph
        elif self.aug2.augmentation_type is AugmentationType.PYG_GRAPH:
            n2 = self.aug2(graph)
        elif self.aug2.augmentation_type is AugmentationType.RDKIT_MOL:
            try:
                mol_aug2 = self.aug2(mol)
                n2 = MolDatasetOD.mol_to_data(mol_aug2)
            except KeyError:
                print(f'Error: loading molecule {smi}')
                n2 = MolDatasetOD.mol_to_data(mol)

        return n1, n2


class MoleculeSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.smiles = self.dataset.smiles[self.indices] if self.dataset.smiles is not None else None
        self.y = self.dataset.y[self.indices] if self.dataset.y is not None else None
        self.clusters = self.dataset.clusters[self.indices] if self.dataset.clusters is not None else None

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        return item

    def __len__(self):
        return len(self.smiles)

    def __getattr__(self, item):
        return getattr(self.dataset, item)

    def recompute_y(self):
        self.y = self.dataset.y[self.indices] if self.dataset.y is not None else None

    def to_dataset(self) -> MolDatasetOD:
        d = MolDatasetOD(smiles_list=self.smiles, cluster_list=self.clusters, y_list=self.y, separate_label=self.separate_label, include_smiles=self.include_smiles)
        if hasattr(self.dataset, 'mol_features'):
            d.mol_features = self.dataset.mol_features[self.indices, :]
        return d


RDKit2D_generator = rdNormalizedDescriptors.RDKit2DNormalized()
RDKit2DNormalized_generator = rdNormalizedDescriptors.RDKit2DNormalized()


def rdkit_2d_features_generator(mol) -> np.ndarray:
    """
    Generates RDKit 2D features for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :return: A 1D numpy array containing the RDKit 2D features.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    features = RDKit2D_generator.process(smiles)[1:]

    return np.array(features)


def rdkit_2d_normalized_features_generator(mol) -> np.ndarray:
    """
    Generates RDKit 2D normalized features for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :return: A 1D numpy array containing the RDKit 2D normalized features.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    features = RDKit2DNormalized_generator.process(smiles)[1:]

    return np.array(features)


def add_mol_features(data, mol_features_mode='rdkit_normalized', extra_features=None, parallelize=False):
    print('Computing mol features')
    mol_features = None
    if mol_features_mode == 'rdkit_normalized':
        if parallelize:
            mol_features = process_map(rdkit_2d_normalized_features_generator, data.smiles, max_workers=16, chunksize=1000)
        else:
            mol_features = [rdkit_2d_normalized_features_generator(i) for i in tqdm(data.smiles)]
        replace_token = 0
        mol_features = np.where(np.isnan(mol_features), replace_token, mol_features)
    elif mol_features_mode == 'rdkit':
        if parallelize:
            mol_features = process_map(rdkit_2d_features_generator, data.smiles, max_workers=16, chunksize=1000)
        else:
            mol_features = [rdkit_2d_features_generator(i) for i in tqdm(data.smiles)]
        replace_token = 0
        mol_features = np.where(np.isnan(mol_features), replace_token, mol_features)

    elif mol_features_mode is None:
        mol_features = np.array([])
    
    data.mol_features = torch.tensor(mol_features, dtype=torch.float32)
    if extra_features is not None:
        print('Adding extra features')
        extra_features_tensor = torch.tensor(extra_features, dtype=torch.float32)
        data.mol_features = torch.hstack((data.mol_features, extra_features_tensor))
        print('Shape of mol_features: {}'.format(data.mol_features.size))


def random_split(data, seed, sizes=(0.8, 0.1, 0.1)):
    random = Random(seed)

    indices = list(range(len(data)))
    random.shuffle(indices)

    train_size = int(sizes[0] * len(data))
    train_val_size = int((sizes[0] + sizes[1]) * len(data))

    train = MoleculeSubset(data, indices[:train_size])
    val = MoleculeSubset(data, indices[train_size:train_val_size])
    test = MoleculeSubset(data, indices[train_val_size:])
    return train, val, test


def stratified_split(data, seed, sizes=(0.8, 0.1, 0.1), bucket_size=10):
    train, val, test = SingleTaskStratifiedSplitter.train_val_test_split(data, torch.tensor(data.y[:, np.newaxis]),
                                                                         frac_train=sizes[0], frac_val=sizes[1],
                                                                         frac_test=sizes[2],
                                                                         task_id=0, random_state=seed,
                                                                         bucket_size=bucket_size)

    train = MoleculeSubset(train.dataset, train.indices)
    val = MoleculeSubset(val.dataset, val.indices)
    test = MoleculeSubset(test.dataset, test.indices)

    return train, val, test


def scaffold_split_balanced(data, seed, sizes=(0.8, 0.1, 0.1)):  # never used
    train, val, test = scaffold_split(data, sizes=sizes, seed=seed)
    return train, val, test


def scaffold_split_dgl(data, sizes=(0.8, 0.1, 0.1)):
    train, val, test = ScaffoldSplitter.train_val_test_split(data, mols=data.mols, frac_train=sizes[0],
                                                             frac_val=sizes[1], frac_test=sizes[2], log_every_n=10000)
    train = MoleculeSubset(train.dataset, train.indices)
    val = MoleculeSubset(val.dataset, val.indices)
    test = MoleculeSubset(test.dataset, test.indices)
    return train, val, test


def fixed_test_scaffold_rest(data, seed, only_test=False):
    test_idx = np.argwhere(data.fixed_list).flatten()
    train_val_idx = np.argwhere(~data.fixed_list).flatten()

    test = MoleculeSubset(data, test_idx)
    if only_test:
        return test
    train_val = MoleculeSubset(data, train_val_idx)
    train_val = MolDatasetOD(smiles_list=train_val.smiles, y_list=train_val.y)
    train_val.mols = convert_smiles_to_mols(train_val.smiles)
    train, val, _ = scaffold_split(train_val, sizes=(0.9, 0.1, 0.), seed=seed)
    return train, val, test


def split_dataset_by_tasks(dataset, train_tasks, val_tasks, test_tasks):
    include_smiles = True

    train_filter = np.isin(dataset.y, train_tasks)
    val_filter = np.isin(dataset.y, val_tasks)
    test_filter = np.isin(dataset.y, test_tasks)

    train_set = MolDatasetOD(dataset.smiles[train_filter], y_list=dataset.y[train_filter], separate_label=True,
                             include_smiles=include_smiles)
    val_set = MolDatasetOD(dataset.smiles[val_filter], y_list=dataset.y[val_filter], separate_label=True,
                           include_smiles=include_smiles)
    test_set = MolDatasetOD(dataset.smiles[test_filter], y_list=dataset.y[test_filter], separate_label=True,
                            include_smiles=include_smiles)

    return train_set, val_set, test_set


def sm_to_mol(s):
    return Chem.MolFromSmiles(s)


def convert_smiles_to_mols(smiles, parallelize=False):
    if parallelize:
        mols = process_map(sm_to_mol, smiles, max_workers=16, chunksize=1000)
    else:
        mols = [sm_to_mol(s) for s in tqdm(smiles)]
    return mols


def scaffold_fingerprint_split(dataset, sizes=(0.8, 0.1, 0.1), seed=0):
    if not hasattr(dataset, 'mols'):
        dataset.mols = convert_smiles_to_mols(dataset.smiles, parallelize=True)
    if not hasattr(dataset, 'scaffolds'):
        dataset.scaffolds = scaffold_to_smiles(dataset.mols, use_indices=True, parallelize=True)

    dataset_scaffolds_list = np.array(list(dataset.scaffolds.keys()))
    # introduce randomness
    random = Random(seed)
    indices = list(range(len(dataset_scaffolds_list)))
    random.shuffle(indices)
    dataset_scaffolds_list = dataset_scaffolds_list[indices]

    dataset_scaffolds = MolDatasetOD(smiles_list=dataset_scaffolds_list)

    fp_splitter = FingerprintSplitter()
    train_scaffold_inds, valid_scaffold_inds, test_scaffold_inds = fp_splitter.split(dataset_scaffolds,
                                                                                     frac_train=sizes[0],
                                                                                     frac_valid=sizes[1],
                                                                                     frac_test=sizes[2], seed=seed)

    train_scaffolds, val_scaffolds, test_scaffolds = dataset_scaffolds_list[train_scaffold_inds], \
                                                     dataset_scaffolds_list[valid_scaffold_inds], \
                                                     dataset_scaffolds_list[test_scaffold_inds]

    train_inds = np.concatenate([np.array(list(dataset.scaffolds[i])) for i in train_scaffolds])
    valid_inds = np.concatenate([np.array(list(dataset.scaffolds[i])) for i in val_scaffolds])
    test_inds = np.concatenate([np.array(list(dataset.scaffolds[i])) for i in test_scaffolds])

    train = MoleculeSubset(dataset, train_inds)
    val = MoleculeSubset(dataset, valid_inds)
    test = MoleculeSubset(dataset, test_inds)
    return train, val, test


try:
    from deepchem.splits.splitters import Splitter
    # adapted from deepchem
    class RandomGroupSplitter(Splitter):
        """Random split based on groupings.
        A splitter class that splits on groupings. An example use case is when
        there are multiple conformations of the same molecule that share the same
        topology.  This splitter subsequently guarantees that resulting splits
        preserve groupings.
        Note that it doesn't do any dynamic programming or something fancy to try
        to maximize the choice such that frac_train, frac_valid, or frac_test is
        maximized.  It simply permutes the groups themselves. As such, use with
        caution if the number of elements per group varies significantly.
        """

        def __init__(self, groups: Sequence):
            """Initialize this object.
            Parameters
            ----------
            groups: Sequence
              An array indicating the group of each item.
              The length is equals to `len(dataset.X)`
            Note
            ----
            The examples of groups is the following.
            | groups    : 3 2 2 0 1 1 2 4 3
            | dataset.X : 0 1 2 3 4 5 6 7 8
            | groups    : a b b e q x a a r
            | dataset.X : 0 1 2 3 4 5 6 7 8
            """
            self.groups = groups

        def split(self,
                  dataset: torch.utils.data.Dataset,
                  frac_train: float = 0.8,
                  frac_valid: float = 0.1,
                  frac_test: float = 0.1,
                  seed: Optional[int] = None,
                  log_every_n: Optional[int] = None
                  ) -> Tuple[List[int], List[int], List[int]]:
            """Return indices for specified split
            Parameters
            ----------
            dataset: Dataset
              Dataset to be split.
            frac_train: float, optional (default 0.8)
              The fraction of data to be used for the training split.
            frac_valid: float, optional (default 0.1)
              The fraction of data to be used for the validation split.
            frac_test: float, optional (default 0.1)
              The fraction of data to be used for the test split.
            seed: int, optional (default None)
              Random seed to use.
            log_every_n: int, optional (default None)
              Log every n examples (not currently used).
            Returns
            -------
            Tuple[List[int], List[int], List[int]]
              A tuple `(train_inds, valid_inds, test_inds` of the indices (integers) for
              the various splits.
            """

            np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)

            if seed is not None:
                np.random.seed(seed)

            # dict is needed in case groups aren't strictly flattened or
            # hashed by something non-integer like
            group_dict: Dict[Any, List[int]] = {}
            for idx, g in enumerate(self.groups):
                if g not in group_dict:
                    group_dict[g] = []
                group_dict[g].append(idx)

            group_idxs = np.array([g for g in group_dict.values()])

            num_groups = len(group_idxs)
            train_cutoff = int(frac_train * num_groups)
            valid_cutoff = int((frac_train + frac_valid) * num_groups)
            shuffled_group_idxs = np.random.permutation(range(num_groups))

            train_groups = shuffled_group_idxs[:train_cutoff]
            valid_groups = shuffled_group_idxs[train_cutoff:valid_cutoff]
            test_groups = shuffled_group_idxs[valid_cutoff:]

            train_idxs = list(itertools.chain(*group_idxs[train_groups]))
            valid_idxs = list(itertools.chain(*group_idxs[valid_groups]))
            test_idxs = list(itertools.chain(*group_idxs[test_groups]))

            return train_idxs, valid_idxs, test_idxs


    def cluster_split(dataset, seed, sizes=(0.8, 0.1, 0.1)):
        fs = RandomGroupSplitter(dataset.clusters)
        train_ix, val_ix, test_ix = fs.split(dataset=dataset, frac_train=sizes[0], frac_valid=sizes[1],
                                             frac_test=sizes[2],
                                             seed=seed)
        train = MoleculeSubset(dataset, train_ix)
        val = MoleculeSubset(dataset, val_ix)
        test = MoleculeSubset(dataset, test_ix)
        return train, val, test
except ImportError:
    # dummy functions
    def cluster_split(dataset, seed, sizes=(0.8, 0.1, 0.1)):
        raise NotImplementedError("Currently this functions require deepchem. Install deepchem to use this function")


def load_dataset_multi_format(full_path: str, cluster_name=None, fixed_split=False, target_name='Activity', smiles_name='SMILES', legacy=True):
    File = full_path
    if 's3:' in full_path:
        fs = s3_setup()
        S3File = fs.open(full_path)
        File = S3File

    if full_path.endswith('.pt'):
        return MolDataset.load_from_full_path(File)
    elif full_path.endswith('.csv'):
        return MolDatasetOD.load_csv_dataset(File, cluster_name=cluster_name, fixed_split=fixed_split,
                                             target_name=target_name, smiles_name=smiles_name, legacy=legacy)
    elif full_path.endswith('.ftr'):
        return MolDatasetOD.load_ftr_dataset(File, cluster_name=cluster_name, fixed_split=fixed_split, 
                                             target_name=target_name, smiles_name=smiles_name, legacy=legacy)


def preprocess_output_data(train, preprocess_type):
    """
    Preprocess data and returns a dictionary with statistics to do inverse transformation.
    """

    output_pp_statistics = dict()
    if preprocess_type == 'log+std':
        y = np.log(train.y)
        y = y.reshape(-1, 1)
        scaler = preprocessing.StandardScaler().fit(y)
        train.y = scaler.transform(y)

        output_pp_statistics['mean'] = scaler.mean_
        output_pp_statistics['scale'] = scaler.scale_

    if preprocess_type == 'standard':
        y = train.y.reshape(-1, 1)
        scaler = preprocessing.StandardScaler().fit(y)
        train.y = scaler.transform(y)
        output_pp_statistics['mean'] = scaler.mean_
        output_pp_statistics['scale'] = scaler.scale_

    elif preprocess_type == 'minmax':
        y = train.y.reshape(-1, 1)
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(y)
        train.y = scaler.transform(y)
        output_pp_statistics['min'] = scaler.data_min_
        output_pp_statistics['max'] = scaler.data_max_

    elif preprocess_type == 'log':
        train.y = np.log(train.y)

    elif preprocess_type == 'log+1':
        train.y = np.log(train.y+1)

    elif preprocess_type == 'sqrt':
        train.y = np.sqrt(train.y)

    elif preprocess_type == 'cbrt':
        train.y = np.cbrt(train.y)

    elif preprocess_type == 'boxcox':
        from scipy import stats, special
        y_transform, lamda = stats.boxcox(train.y)
        train.y = y_transform
        output_pp_statistics['lambda'] = lamda

    return train, output_pp_statistics

# adapted from https://github.com/chemprop/chemprop
def scaffold_split(data,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   balanced: bool = True,
                   seed: int = 0,
                   logger: logging.Logger = None):
    r"""
    Splits a :class:`~chemprop.data.MoleculeDataset` by scaffold so that no molecules sharing a scaffold are in different splits.

    :param data: A :class:`MoleculeDataset`.
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param balanced: Whether to balance the sizes of scaffolds in each set rather than putting the smallest in test set.
    :param seed: Random seed for shuffling when doing balanced splitting.
    :param logger: A logger for recording output.
    :return: A tuple of :class:`~chemprop.data.MoleculeDataset`\ s containing the train,
             validation, and test splits of the data.
    """
    assert sum(sizes) == 1

    # Split
    train_size, val_size, test_size = int(sizes[0] * len(data)), int(sizes[1] * len(data)), int(sizes[2] * len(data))
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    try:
        # precomputed
        scaffold_to_indices = data.scaffolds
    except (KeyError, AttributeError):
        # Map from scaffold to index in the data
        scaffold_to_indices = scaffold_to_smiles(data.mols, use_indices=True)

    # Seed randomness
    random = Random(seed)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    if logger is not None:
        logger.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                     f'train scaffolds = {train_scaffold_count:,} | '
                     f'val scaffolds = {val_scaffold_count:,} | '
                     f'test scaffolds = {test_scaffold_count:,}')

    return MoleculeSubset(data, train), MoleculeSubset(data, val), MoleculeSubset(data, test)

def scaffold_stratified_split(data,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   seed: int = 0,
                   logger: logging.Logger = None):
    r"""
    Splits a :class:`~chemprop.data.MoleculeDataset` by scaffold so that no molecules sharing a scaffold are in different splits.
    It also conserve the ratio of positive samples of the whole dataset in each of the three splits.

    :param data: A :class:`MoleculeDataset`.
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param seed: Random seed for shuffling when doing balanced splitting.
    :param logger: A logger for recording output.
    :return: A tuple of :class:`~chemprop.data.MoleculeDataset`\ s containing the train,
             validation, and test splits of the data.
    """
    assert math.isclose(np.array(sizes).sum(), 1, abs_tol=1e-03)

    # Split
    train_size, val_size, test_size = int(sizes[0] * len(data)), int(sizes[1] * len(data)), int(sizes[2] * len(data))
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    if train_size == val_size == 0:
        return None, None, MoleculeSubset(data, range(len(data)))

    try:
        # precomputed
        scaffold_to_indices = data.scaffolds
    except (KeyError, AttributeError):
        # Map from scaffold to index in the data
        scaffold_to_indices = scaffold_to_smiles(data.mols, use_indices=True)

    # Seed randomness
    random = Random(seed)

    index_sets = list(scaffold_to_indices.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 and len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    dict_grp = [{'set':big_index_sets.copy(),
                'size':train_size,
                'scaffold_count':len(big_index_sets)},
                {'set':test,
                'size':test_size,
                'scaffold_count':0},
                {'set':val,
                'size':val_size,
                'scaffold_count':0}]

    #sort idx sets according to nb positives, okay to do because it was randomized before
    small_index_lists = [list(el) for el in small_index_sets] # list of sets to list of lists
    sorted_idx = np.argsort([np.sum(data.y[el] == 1) for el in small_index_lists]) # get indexes of sorting on numbers of positives
    sorted_idx_lists = [small_index_lists[idx] for idx in sorted_idx] # get the scaffolds groups, sorted
    sorted_idx_lists.reverse() # get the groups with the biggest numbers of positives first

    for index_set in sorted_idx_lists:
        ratio_train = np.sum(data.y[dict_grp[0]['set']] == 1) / (dict_grp[0]['size'] + 1e-03)
        ratio_test = np.sum(data.y[dict_grp[1]['set']] == 1) / (dict_grp[1]['size'] + 1e-03)
        ratio_val = np.sum(data.y[dict_grp[2]['set']] == 1) / (dict_grp[2]['size'] + 1e-03)

        ratios = [ratio_train, ratio_test, ratio_val]

        sort_index = np.arange(len(ratios))
        if(math.isclose(ratio_train, ratio_test, rel_tol=1e-3) & math.isclose(ratio_test, ratio_val, rel_tol=1e-3)):
            random.shuffle(sort_index)
        else:
            sort_index = np.argsort(ratios)

        priority_group1, priority_group2, priority_group3 = dict_grp[sort_index[0]], dict_grp[sort_index[1]], dict_grp[sort_index[2]] # assign priorities of which group is going to get assigned indexes first
        if len(priority_group1['set']) + len(index_set) <= priority_group1['size']:
            priority_group1['set'] += index_set
            priority_group1['scaffold_count'] += 1
        elif len(priority_group2['set']) + len(index_set) <= priority_group2['size']:
            priority_group2['set'] += index_set
            priority_group2['scaffold_count'] += 1
        else:
            priority_group3['set'] += index_set
            priority_group3['scaffold_count'] += 1

    if logger is not None:
        logger.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                     f'train scaffolds = {train_scaffold_count:,} | '
                     f'val scaffolds = {val_scaffold_count:,} | '
                     f'test scaffolds = {test_scaffold_count:,}')

    if (val_size == 0):
        MoleculeSubset(data, dict_grp[0]['set']), None, MoleculeSubset(data, dict_grp[1]['set'])
    elif (test_size == 0):

        MoleculeSubset(data, dict_grp[0]['set']), MoleculeSubset(data, dict_grp[2]['set']), None

    return MoleculeSubset(data, dict_grp[0]['set']), MoleculeSubset(data, dict_grp[2]['set']), MoleculeSubset(data, dict_grp[1]['set'])


class ACDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, ac_list, bound_size=None, skip_ratio=-1.):
        self.dataset = dataset
        self.ac_list = ac_list

        if bound_size is not None and bound_size > len(ac_list):
            self.bound_size = len(ac_list)
        else:
            self.bound_size = bound_size

        self.skip_ratio = skip_ratio
        if self.skip_ratio < 0:
            self.skip_ratio = None

    def __len__(self):
        if self.bound_size is not None:
            return self.bound_size
        else:
            return len(self.ac_list)

    def __getitem__(self, idx):
        if self.skip_ratio is not None:
            rand_p = random.random()
            if rand_p <= self.skip_ratio:
                return []
        ac_batch = []
        for b_idx in self.ac_list[idx]:
            ac_batch.append(self.dataset[b_idx])

        return Batch.from_data_list(ac_batch)

    def __getattr__(self, item):
        return getattr(self.dataset, item)


