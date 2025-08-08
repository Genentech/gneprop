import dgllife
import torch
from dgllife.utils import smiles_to_bigraph, mol_to_bigraph, one_hot_encoding
from torch_geometric.data import Data
from typing import List, Tuple, Union

import chemprop_featurizer

from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops


def sm_to_bigraph(s):
    feat = chemprop_featurizer.atoms_features
    g = smiles_to_bigraph(s, node_featurizer=feat,
                          edge_featurizer=dgllife.utils.CanonicalBondFeaturizer(bond_data_field='edge_attr'))
    return g


def m_to_bigraph(s):
    feat = chemprop_featurizer.atoms_features
    return mol_to_bigraph(s, node_featurizer=feat,
                          edge_featurizer=dgllife.utils.CanonicalBondFeaturizer(bond_data_field='edge_attr'))


MAX_ATOMIC_NUM = 100
DEGREES_OFFSET = MAX_ATOMIC_NUM + 1
NUM_DEGREES = 6
CHARGE_OFFSET = DEGREES_OFFSET + NUM_DEGREES + 1
NUM_CHARGE_OPTIONS = 5
CHIRAL_OFFSET = CHARGE_OFFSET + NUM_CHARGE_OPTIONS + 1
NUM_CHIRAL_TAGS = 4
HS_OFFSET = CHIRAL_OFFSET + NUM_CHIRAL_TAGS + 1
NUM_HS_LIMIT = 5
HYBRIDIZATION_OFFSET = HS_OFFSET + NUM_HS_LIMIT + 1
NUM_HYBRIDIZATION = len(chemprop_featurizer.ATOM_FEATURES['hybridization'])
OTHER_OFFSET = HYBRIDIZATION_OFFSET + NUM_HYBRIDIZATION + 1
NUM_OTHER = 2
NUM_TOTAL = OTHER_OFFSET + NUM_OTHER


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    encoding = [0] * NUM_TOTAL

    value = atom.GetAtomicNum() - 1
    value = value if (value >= 0 and value < MAX_ATOMIC_NUM) else MAX_ATOMIC_NUM
    encoding[value] = 1

    value = atom.GetTotalDegree()
    value = value if (value >= 0 and value < NUM_DEGREES) else NUM_DEGREES
    encoding[value + DEGREES_OFFSET] = 1

    # NOTE: The original had the charges in a different order, but this hopefully shouldn't matter.
    value = atom.GetFormalCharge() + 2
    value = value if (value >= 0 and value < NUM_CHARGE_OPTIONS) else NUM_CHARGE_OPTIONS
    encoding[value + CHARGE_OFFSET] = 1

    value = int(atom.GetChiralTag())
    value = value if (value >= 0 and value < NUM_CHIRAL_TAGS) else NUM_CHIRAL_TAGS
    encoding[value + CHIRAL_OFFSET] = 1

    value = int(atom.GetTotalNumHs())
    value = value if (value >= 0 and value < NUM_HS_LIMIT) else NUM_HS_LIMIT
    encoding[value + HS_OFFSET] = 1

    value = int(atom.GetHybridization())
    choices = chemprop_featurizer.ATOM_FEATURES['hybridization']
    value = choices.index(value) if value in choices else NUM_HYBRIDIZATION
    encoding[value + HYBRIDIZATION_OFFSET] = 1

    encoding[OTHER_OFFSET] = 1 if atom.GetIsAromatic() else 0
    encoding[OTHER_OFFSET + 1] = atom.GetMass() * 0.01  # scaled to about the same range as other features

    if functional_groups is not None:
        encoding += functional_groups
    return encoding


def smiles_to_data(s, label=None, mol_features=None, legacy=False):
    mol = Chem.MolFromSmiles(s)

    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)

    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])

    node_feature = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()])

    edge_feature = []
    type_set = [Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC]
    stereo_set = [Chem.rdchem.BondStereo.STEREONONE,
                  Chem.rdchem.BondStereo.STEREOANY,
                  Chem.rdchem.BondStereo.STEREOZ,
                  Chem.rdchem.BondStereo.STEREOE,
                  Chem.rdchem.BondStereo.STEREOCIS,
                  Chem.rdchem.BondStereo.STEREOTRANS]
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        feat = one_hot_encoding(bond.GetBondType(), type_set) + [bond.GetIsConjugated(),
                                                                 bond.IsInRing()] + one_hot_encoding(bond.GetStereo(),
                                                                                                     stereo_set)
        edge_feature.extend([feat, feat.copy()])

    edge_feature = torch.tensor(edge_feature, dtype=torch.float32)

    if legacy:
        node_feature_swp = node_feature
        node_feature_swp[:, [110, 112]] = node_feature[:, [112, 110]]
        n = Data(x=node_feature_swp, edge_attr=edge_feature, edge_index=torch.tensor([src_list, dst_list], dtype=torch.int64))
    else:
        n = Data(x=node_feature, edge_attr=edge_feature, edge_index=torch.tensor([src_list, dst_list], dtype=torch.int64))

    if label is not None:
        n.y = label
    if mol_features is not None:
        n.mol_features = mol_features
    return n


def mol_to_data(mol, label=None, mol_features=None):
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)

    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])

    node_feature = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()])

    edge_feature = []
    type_set = [Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC]
    stereo_set = [Chem.rdchem.BondStereo.STEREONONE,
                  Chem.rdchem.BondStereo.STEREOANY,
                  Chem.rdchem.BondStereo.STEREOZ,
                  Chem.rdchem.BondStereo.STEREOE,
                  Chem.rdchem.BondStereo.STEREOCIS,
                  Chem.rdchem.BondStereo.STEREOTRANS]
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        feat = one_hot_encoding(bond.GetBondType(), type_set) + [bond.GetIsConjugated(),
                                                                 bond.IsInRing()] + one_hot_encoding(bond.GetStereo(),
                                                                                                     stereo_set)
        edge_feature.extend([feat, feat.copy()])

    edge_feature = torch.tensor(edge_feature, dtype=torch.float32)

    n = Data(x=node_feature, edge_attr=edge_feature, edge_index=torch.tensor([src_list, dst_list], dtype=torch.int64))

    if label is not None:
        n.y = label
    if mol_features is not None:
        n.mol_features = mol_features
    return n
