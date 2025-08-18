from typing import Callable, List, Union
import random

import json
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, remove_isolated_nodes, dropout_adj
from enum import Enum
from gneprop import featurization
import os

AugmentationType = Enum('AugmentationType', 'RDKIT_MOL PYG_GRAPH MIXED')

def transform_remove_isolated_nodes(graph):
    new_edge_index, new_edge_attr, nodes_mask = remove_isolated_nodes(graph.edge_index, edge_attr=graph.edge_attr)
    return Data(x=graph.x[nodes_mask], edge_index=new_edge_index, edge_attr=new_edge_attr, y=graph.y)


def get_neighbors(g, node_ix):
    return g.edge_index[1][g.edge_index[0] == node_ix]


class Augmentation(object):
    def __init__(self, p_skip=0., return_applied_flag=False):
        self.p_skip = p_skip
        self.augmentation_type = AugmentationType.PYG_GRAPH
        self.return_applied_flag = return_applied_flag

    def transform(self, mol):
        return mol

    def __call__(self, mol):
        rand_p = random.random()
        if rand_p < self.p_skip:
            if self.return_applied_flag:
                return mol, False
            else:
                return mol
        else:
            if self.return_applied_flag:
                return self.transform(mol), True
            else:
                return self.transform(mol)

    @classmethod
    def create_from_config(cls, config):
        pass

AUGMENTATION_REGISTRY = {}

def register_augmentation(augmentation_name: str):
    """
    Creates a decorator which registers an augmentation in a global dictionary to enable access by name.
    """
    def decorator(augmentation):
        AUGMENTATION_REGISTRY[augmentation_name] = augmentation
        return augmentation

    return decorator


def get_augmentation(augmentation_name: str) -> Augmentation:
    """
    Gets an augmentation generator by name.
    """
    if augmentation_name not in AUGMENTATION_REGISTRY:
        raise ValueError(f'Augmentation "{augmentation_name}" could not be found. ')

    return AUGMENTATION_REGISTRY[augmentation_name]


def get_available_augmentations() -> List[str]:
    """Returns a list of names of available features generators."""
    return list(AUGMENTATION_REGISTRY.keys())


@register_augmentation('RemoveNodes')
class RemoveNodes(Augmentation):
    def __init__(self, fraction_nodes_to_remove=0.20, p_skip=0.):
        super().__init__(p_skip=p_skip)

        self.fraction_nodes_to_remove = fraction_nodes_to_remove
        self.augmentation_type = AugmentationType.PYG_GRAPH

    def transform(self, graph):
        num_nodes = graph.num_nodes
        num_nodes_to_keep = num_nodes - int(num_nodes * self.fraction_nodes_to_remove)
        try:
            nodes_to_keep = random.sample(range(num_nodes), k=num_nodes_to_keep)
            new_edge_index, new_edge_attr = subgraph(nodes_to_keep, edge_index=graph.edge_index,
                                                     edge_attr=graph.edge_attr,
                                                     relabel_nodes=True)
        except:
            print(f'Error: RemoveNodes')
            return graph

        new_graph = graph.clone()
        new_graph.x, new_graph.edge_index, new_graph.edge_attr = graph.x[nodes_to_keep], new_edge_index, new_edge_attr

        return new_graph

    def __repr__(self):
        return f'RemoveNodes(fraction_nodes_to_remove={self.fraction_nodes_to_remove}, p_skip={self.p_skip})'

    @classmethod
    def create_from_config(cls, config):
        fraction_nodes_to_remove = config['fraction_nodes_to_remove']
        p_skip = config['p_skip']
        return cls(fraction_nodes_to_remove=fraction_nodes_to_remove, p_skip=p_skip)


@register_augmentation('RemoveSubgraph')
class RemoveSubgraph(Augmentation):
    def __init__(self, fraction_nodes_to_remove=0.20, p_skip=0.):
        super().__init__(p_skip=p_skip)

        self.fraction_nodes_to_remove = fraction_nodes_to_remove
        self.augmentation_type = AugmentationType.PYG_GRAPH

    def transform(self, graph):
        num_nodes_to_remove = int(graph.num_nodes * self.fraction_nodes_to_remove)
        start_ix = random.randrange(graph.num_nodes)
        visited = [0] * graph.num_nodes
        s = start_ix
        visited[s] = 1
        v_samp = [s]
        current_neigh = get_neighbors(graph, s)
        v_neigh = []
        for n in current_neigh:
            visited[n] = 1
            v_neigh.append(n)

        if num_nodes_to_remove > 1:
            while len(v_neigh) != 0:  # probably disconnected graph, stop here
                si = random.randrange(len(v_neigh))
                s = v_neigh[si]
                v_samp.append(s)
                if len(v_samp) > num_nodes_to_remove:
                    break
                v_neigh[si] = v_neigh[-1]
                v_neigh.pop()

                current_neigh = get_neighbors(graph, s)
                for n in current_neigh:
                    if visited[n] == 0:
                        visited[n] = 1
                        v_neigh.append(n)

        node_map = [-1] * graph.num_nodes
        new_graph_nodes_id = [0] * (graph.num_nodes - len(v_samp))
        nodei = 0
        for i in range(graph.num_nodes):
            if not (i in v_samp):
                new_graph_nodes_id[nodei] = i
                node_map[i] = nodei
                nodei = nodei + 1
        edges = []
        for i in range(graph.edge_index.shape[0]):
            a = node_map[graph.edge_index[0][i]]
            b = node_map[graph.edge_index[1][i]]
            if a >= 0 and b >= 0:
                edges.append(i)
        new_edge_index = torch.zeros([2, len(edges)], dtype=graph.edge_index.dtype)
        for i in range(len(edges)):
            new_edge_index[0][i] = node_map[graph.edge_index[0][i]]
            new_edge_index[1][i] = node_map[graph.edge_index[1][i]]
        new_edge_attr = graph.edge_attr[edges]

        new_graph = Data(x=graph.x[new_graph_nodes_id], edge_attr=new_edge_attr, edge_index=new_edge_index)
        if hasattr(graph, "y"):
            new_graph.y = graph.y
        if hasattr(graph, "mol_features"):
            new_graph.mol_features = graph.mol_features

        return new_graph

    def __repr__(self):
        return f'RemoveSubgraph(fraction_nodes_to_remove={self.fraction_nodes_to_remove}, p_skip={self.p_skip})'

    @classmethod
    def create_from_config(cls, config):
        fraction_nodes_to_remove = config['fraction_nodes_to_remove']
        p_skip = config['p_skip']
        return cls(fraction_nodes_to_remove=fraction_nodes_to_remove, p_skip=p_skip)


@register_augmentation('DropEdges')
class DropEdges(Augmentation):
    def __init__(self, drop_p=0.20, force_undirected=True, p_skip=0.):
        super().__init__(p_skip=p_skip)

        self.drop_p = drop_p
        self.force_undirected = force_undirected
        self.augmentation_type = AugmentationType.PYG_GRAPH

    def transform(self, graph):
        new_edge_index, new_edge_attr = dropout_adj(graph.edge_index, graph.edge_attr, p=self.drop_p,
                                                    force_undirected=self.force_undirected)
        new_graph = graph.clone()
        new_graph.edge_index, new_graph.edge_attr = new_edge_index, new_edge_attr

        return new_graph

    def __repr__(self):
        return f'DropEdges(drop_p={self.drop_p}, force_undirected={self.force_undirected}, p_skip={self.p_skip})'

    @classmethod
    def create_from_config(cls, config):
        drop_p = config['drop_p']
        force_undirected = config['force_undirected']
        p_skip = config['p_skip']
        return cls(drop_p=drop_p, force_undirected=force_undirected, p_skip=p_skip)


@register_augmentation('MoCL')
class MoCL(Augmentation):
    def __init__(self, rules, aug_times=1, p_skip=0.):
        super().__init__(p_skip=p_skip)

        assert aug_times >= 1
        self.rules = rules
        self.rules_smarts = [r['smarts'] for r in self.rules]
        self.reactions = [AllChem.ReactionFromSmarts(r) for r in self.rules_smarts]
        self.reactants = [r.GetReactants()[0] for r in self.reactions]
        self.reactants_smiles = [Chem.MolToSmiles(i) for i in self.reactants]
        self.reactants_groups = {}
        for index, rs in enumerate(self.reactants_smiles):
            self.reactants_groups[index] = set([_i for _i, _rs in enumerate(self.reactants_smiles) if _rs == rs])

        self.num_reactions = len(self.reactions)
        self.aug_times = aug_times

        self.augmentation_type = AugmentationType.RDKIT_MOL

    def transform(self, mol):
        mol_prev = mol
        for time in range(self.aug_times):
            # random order of augmentations to try
            aug_indices = list(range(self.num_reactions))
            random.shuffle(aug_indices)

            already_checked_reactants = set()

            for aug_ix in aug_indices:
                if aug_ix in already_checked_reactants:
                    continue
                if not mol_prev.HasSubstructMatch(self.reactants[aug_ix]):
                    same_reactants_group = self.reactants_groups[aug_ix]
                    already_checked_reactants.update(same_reactants_group)
                    continue
                rxn = self.reactions[aug_ix]
                products = rxn.RunReactants((mol_prev,))
                n_products = len(products)
                if n_products > 0:  # successful reaction
                    prod_ix = random.choice(range(n_products))
                    product = products[prod_ix][0]
                    try:
                        Chem.SanitizeMol(product)
                        mol_prev = product
                    except:  # TODO: add detailed exception
                        continue
                    break
            else:  # no valid reaction found for this molecule
                return mol_prev

        return mol_prev

    def __repr__(self):
        return f'MoCL(aug_times={self.aug_times}, p_skip={self.p_skip})'

    @classmethod
    def create_from_config(cls, config):
        mode = config['mode']
        if mode == 'inline':
            rules = config['rules']
        elif mode == 'path':
            dir_path = os.path.dirname(os.path.realpath(__file__))
            rules_path = os.path.join(dir_path, config['rules'])
            rules = json.load(open(rules_path))
        else:
            raise ValueError('Invalid mode value.')

        aug_times = config['aug_times']
        p_skip = config['p_skip']
        return cls(rules=rules, aug_times=aug_times, p_skip=p_skip)



@register_augmentation('RepeatedAugmentation')
class RepeatedAugmentation(Augmentation):
    def __init__(self, augmentation, min_number=1, max_number=1, p_skip=0.):
        super().__init__(p_skip=p_skip)

        assert min_number >= 1
        assert max_number >= min_number

        self.augmentation = augmentation
        self.min_number = min_number
        self.max_number = max_number
        self.augmentation_type = augmentation.augmentation_type

    def transform(self, obj):
        sample_num = random.randint(self.min_number, self.max_number)
        for i in range(sample_num):
            obj = self.augmentation(obj)
        return obj

    def __repr__(self):
        repr = f'RepeatedAugmentation(augmentation={self.augmentation}, min_number={self.min_number}, max_number={self.max_number}, p_skip={self.p_skip})'
        return repr

    @classmethod
    def create_from_config(cls, config):
        min_number = config['min_number']
        max_number = config['max_number']
        p_skip = config['p_skip']
        augmentation = AugmentationFactory.create_augmentation_from_config(config['augmentation'])
        return RepeatedAugmentation(augmentation=augmentation, min_number=min_number, max_number=max_number,
                                    p_skip=p_skip)


@register_augmentation('MixedAugmentation')
class MixedAugmentation(Augmentation):
    def __init__(self, mol_augmentations, graph_augmentations, p_skip=0.):
        super().__init__(p_skip=p_skip)

        self.mol_augmentations = mol_augmentations
        self.graph_augmentations = graph_augmentations

        for i in self.mol_augmentations:
            assert i.augmentation_type is AugmentationType.RDKIT_MOL
        for i in self.graph_augmentations:
            assert i.augmentation_type is AugmentationType.PYG_GRAPH

        self.augmentation_type = AugmentationType.MIXED

    def transform(self, mol):
        try:
            mol_updated = mol
            for mol_aug in self.mol_augmentations:
                mol_updated = mol_aug(mol_updated)

            n = featurization.mol_to_data(mol_updated)
        except KeyError:
            print(f'Error: loading molecule')
            n = featurization.mol_to_data(mol)

        n_updated = n
        for graph_aug in self.graph_augmentations:
            n_updated = graph_aug(n_updated)
        return n_updated

    def __repr__(self):
        repr = f'MixedAugmentation(p_skip={self.p_skip})'
        for i in self.mol_augmentations:
            repr += f'\n\t {i}'
        for i in self.graph_augmentations:
            repr += f'\n\t {i}'

        return repr

    @classmethod
    def create_from_config(cls, config):
        mol_augmentations_configs = config['mol_augmentations']
        mol_augmentations = [AugmentationFactory.create_augmentation_from_config(i) for i in mol_augmentations_configs]

        graph_augmentations_config = config['graph_augmentations']
        graph_augmentations = [AugmentationFactory.create_augmentation_from_config(i) for i in
                               graph_augmentations_config]

        p_skip = config['p_skip']
        return MixedAugmentation(mol_augmentations=mol_augmentations, graph_augmentations=graph_augmentations,
                                 p_skip=p_skip)


@register_augmentation('RemoveNodes')
class RemoveNodes(Augmentation):
    def __init__(self, fraction_nodes_to_remove=0.20, p_skip=0.):
        super().__init__(p_skip=p_skip)

        self.fraction_nodes_to_remove = fraction_nodes_to_remove
        self.augmentation_type = AugmentationType.PYG_GRAPH

    def transform(self, graph):
        num_nodes = graph.num_nodes
        num_nodes_to_keep = num_nodes - int(num_nodes * self.fraction_nodes_to_remove)
        try:
            nodes_to_keep = random.choices(range(num_nodes), k=num_nodes_to_keep)
            new_edge_index, new_edge_attr = subgraph(nodes_to_keep, edge_index=graph.edge_index,
                                                     edge_attr=graph.edge_attr,
                                                     relabel_nodes=True)
        except:
            print(f'Error: RemoveNodes')
            return graph

        new_graph = graph.clone()
        new_graph.x, new_graph.edge_index, new_graph.edge_attr = graph.x[nodes_to_keep], new_edge_index, new_edge_attr

        return new_graph

    def __repr__(self):
        return f'RemoveNodes(fraction_nodes_to_remove={self.fraction_nodes_to_remove}, p_skip={self.p_skip})'

    @classmethod
    def create_from_config(cls, config):
        fraction_nodes_to_remove = config['fraction_nodes_to_remove']
        p_skip = config['p_skip']
        return cls(fraction_nodes_to_remove=fraction_nodes_to_remove, p_skip=p_skip)


@register_augmentation('RemoveNodesPreserving')
class RemoveNodesPreserving(Augmentation):
    def __init__(self, fraction_nodes_to_remove=0.20, patterns=[], p_skip=0.):
        super().__init__(p_skip=p_skip)

        self.fraction_nodes_to_remove = fraction_nodes_to_remove
        self.patterns = patterns
        self.mol_patterns = [Chem.MolFromSmarts(s) for s in self.patterns]

        self.augmentation_type = AugmentationType.MIXED

    def transform(self, mol):
        atoms_matching = self.get_atoms_matching(mol, self.mol_patterns)

        graph = featurization.mol_to_data(mol)

        num_nodes = graph.num_nodes

        available_nodes = list(set(list(range(num_nodes))) - set(atoms_matching))

        num_nodes_to_keep = num_nodes - int(num_nodes * self.fraction_nodes_to_remove)
        num_nodes_to_sample = num_nodes_to_keep - len(atoms_matching)

        try:
            nodes_to_keep = random.sample(available_nodes, k=num_nodes_to_sample)

            nodes_to_keep = nodes_to_keep + atoms_matching

            new_edge_index, new_edge_attr = subgraph(nodes_to_keep, edge_index=graph.edge_index,
                                                     edge_attr=graph.edge_attr,
                                                     relabel_nodes=True)
        except:
            print(f'Error: RemoveNodesPreserving')
            return graph

        new_graph = graph.clone()
        new_graph.x, new_graph.edge_index, new_graph.edge_attr = graph.x[nodes_to_keep], new_edge_index, new_edge_attr

        return new_graph

    def __repr__(self):
        return f'RemoveNodesPreserving(fraction_nodes_to_remove={self.fraction_nodes_to_remove}, p_skip={self.p_skip}, patterns={self.patterns})'

    @staticmethod
    def get_atoms_matching(mol, patterns):
        o = []
        for p in patterns:
            o_p = mol.GetSubstructMatches(p)
            for pp in o_p:
                o.extend(pp)
        return list(set(o))

    @classmethod
    def create_from_config(cls, config):
        mode = config['mode']
        if mode == 'inline':
            patterns = config['patterns']
        elif mode == 'path':
            dir_path = os.path.dirname(os.path.realpath(__file__))
            patterns_path = os.path.join(dir_path, config['patterns'])
            patterns = json.load(open(patterns_path))['patterns']
        else:
            raise ValueError('Invalid mode value.')

        fraction_nodes_to_remove = config['fraction_nodes_to_remove']
        p_skip = config['p_skip']
        return cls(fraction_nodes_to_remove=fraction_nodes_to_remove, patterns=patterns, p_skip=p_skip)


@register_augmentation('UnionAugmentation')
class UnionAugmentation(Augmentation):
    def __init__(self, augmentations, probs, p_skip=0.):
        super().__init__(p_skip=p_skip)

        assert len(augmentations) == len(probs)
        np.testing.assert_almost_equal(np.sum(probs), 1.)

        for i in augmentations:
            assert i.augmentation_type is AugmentationType.RDKIT_MOL or i.augmentation_type is AugmentationType.PYG_GRAPH

        self.augmentations = augmentations
        self.probs = probs

        self.augmentation_type = AugmentationType.MIXED

    def transform(self, mol):
        selected_augmentation = random.choices(self.augmentations, weights=self.probs)[0]

        if selected_augmentation.augmentation_type is AugmentationType.RDKIT_MOL:
            try:
                mol_updated = selected_augmentation(mol)
                n = featurization.mol_to_data(mol_updated)
            except:
                n = featurization.mol_to_data(mol)

        elif selected_augmentation.augmentation_type is AugmentationType.PYG_GRAPH:
            n = featurization.mol_to_data(mol)
            n = selected_augmentation(n)
        else:
            raise NotImplementedError
        return n

    def __repr__(self):
        repr = f'UnionAugmentation(p_skip={self.p_skip})'
        for a, p in zip(self.augmentations, self.probs):
            repr += f'\n\t p={p} -> {a}'
        return repr

    @classmethod
    def create_from_config(cls, config):
        augmentations_config = config['augmentations']
        augmentations = [AugmentationFactory.create_augmentation_from_config(i) for i in augmentations_config]
        probs = config['probs']
        p_skip = config['p_skip']
        return UnionAugmentation(augmentations=augmentations, probs=probs, p_skip=p_skip)


class AugmentationFactory:
    @staticmethod
    def create_augmentation_from_file(file_path):
        with open(file_path, 'r') as f:
            config = json.load(f)
        return AugmentationFactory.create_augmentation_from_config(config)

    @staticmethod
    def create_augmentation_from_config(config) -> Augmentation:
        augmentation_name = config['name']
        augmentation = get_augmentation(augmentation_name)

        return augmentation.create_from_config(config)


