import logging
import sys

sys.path.append("..") # Adds higher directory to python modules path.

import warnings
from collections import defaultdict
from random import Random
from typing import Dict, List, Set, Tuple, Union
import math
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from torch.utils.data import Subset
from tqdm.contrib.concurrent import process_map

import torch
from gneprop.featurization import m_to_bigraph, sm_to_bigraph, smiles_to_data, mol_to_data


def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False) -> str:
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False, parallelize=False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).

    :param mols: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    if not parallelize:
        scaffolds = defaultdict(set)
        for i, mol in tqdm(enumerate(mols), total=len(mols)):
            scaffold = generate_scaffold(mol)
            if use_indices:
                scaffolds[scaffold].add(i)
            else:
                scaffolds[scaffold].add(mol)

        return scaffolds
    else:
        scaffolds = defaultdict(set)
        scaffold_list = process_map(generate_scaffold, mols, max_workers=16, chunksize=1000)
        for i, (mol, scaffold) in tqdm(enumerate(zip(mols, scaffold_list)), total=len(mols)):
            if use_indices:
                scaffolds[scaffold].add(i)
            else:
                scaffolds[scaffold].add(mol)

        return scaffolds

