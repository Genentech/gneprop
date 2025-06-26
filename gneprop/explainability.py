import numpy as np
import torch
import tqdm
from IPython.core.display import SVG
from captum.attr import IntegratedGradients
from collections import defaultdict
from matplotlib import pyplot as plt
from rdkit.Chem import rdDepictor, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Batch
from torch_geometric.nn import GNNExplainer

from gneprop import image_utils
from gneprop.plot_utils import dict_to_color, list_to_color_pos_neg, dict_to_color_pos_neg
from gneprop.utils import aggregate_edge_directions, canonical_order, scale_dictionary, bonddict_to_ixdict, \
    scale_data_signed, scale_dictionary_signed


class IgExplainer:
    def __init__(self, model):
        self.model = model

        def model_forward(x, edge_attr, data):
            data.x = x
            data.edge_attr = edge_attr
            batch = Batch.from_data_list([data])
            out = model(batch)
            return out

        self.model_forward = model_forward

        self.ig = IntegratedGradients(model_forward)

    def explain_graph(self, data, target=0, n_steps=50, ignore_negatives=False, aggregate_in_nodes=False):
        mask = self.ig.attribute((data.x, data.edge_attr), target=target,
                            additional_forward_args=(data,),
                            internal_batch_size=data.x.shape[0], n_steps=n_steps)

        _mask = mask[0].cpu().detach().numpy(), mask[1].cpu().detach().numpy()

        at, bt = _mask[0].sum(axis=1), _mask[1].sum(axis=1)

        bt_agg = aggregate_edge_directions(bt, data)

        if aggregate_in_nodes:
            for k, v in bt_agg.items():
                i, j = k
                at[i] += v/2
                at[j] += v/2

        if ignore_negatives:
            at[at <= 0] = 0
            bt_agg = {k: (v if v >= 0 else 0) for k, v in bt_agg.items()}

        return at, bt_agg

    def explain_graph_and_draw(self, target_graph, target_mol, target=0, cmap_pos='Greens', cmap_neg='Reds', n_steps=50, ignore_negatives=False, show_atoms=True, show_bonds=True, already_canonical_mol=False, aggregate_in_nodes=False, scaffold_mol=None):
        if not already_canonical_mol:
            target_mol = canonical_order(target_mol)

        at, bt_agg = self.explain_graph(target_graph, target=target, n_steps=n_steps, ignore_negatives=ignore_negatives, aggregate_in_nodes=aggregate_in_nodes)
        node_ix_to_color, edge_ix_to_color = self.explanation_to_colors(at, bt_agg, target_mol, cmap_pos=cmap_pos, cmap_neg=cmap_neg, already_canonical_mol=True)
        svg = self.draw_explanation(target_mol, node_ix_to_color, edge_ix_to_color, show_atoms=show_atoms, show_bonds=show_bonds, already_canonical_mol=True, scaffold_mol=scaffold_mol)
        return svg

    @staticmethod
    def explanation_to_colors(at, bt_agg, target_mol, cmap_pos='Greens', cmap_neg='Reds', already_canonical_mol=False):
        if not already_canonical_mol:
            target_mol = canonical_order(target_mol)

        at = scale_data_signed(at)
        bt_agg = scale_dictionary_signed(bt_agg)

        node_ix_to_color = list_to_color_pos_neg(at, cmap_pos=cmap_pos, cmap_neg=cmap_neg)
        bond_ix_to_w = bonddict_to_ixdict(bt_agg, target_mol)

        edge_ix_to_color = dict_to_color_pos_neg(bond_ix_to_w, cmap_pos=cmap_pos, cmap_neg=cmap_neg)

        return node_ix_to_color, edge_ix_to_color

    @staticmethod
    def draw_explanation(target_mol, node_ix_to_color, edge_ix_to_color, show_atoms=True, show_bonds=True, already_canonical_mol=False, scaffold_mol=None):
        if not already_canonical_mol:
            target_mol = canonical_order(target_mol)

        rdDepictor.Compute2DCoords(target_mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
        drawer.SetFontSize(1)
        op = drawer.drawOptions()

        op.useBWAtomPalette()

        dd = defaultdict(lambda: (1, 0, 0))  # <- use a default color of black
        op.elemDict = dd

        if scaffold_mol is not None:
            AllChem.GenerateDepictionMatching2DStructure(target_mol, scaffold_mol)

        target_mol = rdMolDraw2D.PrepareMolForDrawing(target_mol)
        drawer.DrawMolecule(target_mol, highlightAtoms=(node_ix_to_color.keys() if show_atoms else []),
                            highlightAtomColors=(node_ix_to_color if show_atoms else []),
                            highlightBonds=(edge_ix_to_color.keys() if show_bonds else []),
                            highlightBondColors=(edge_ix_to_color if show_bonds else []), )
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = svg.replace('svg:', '')
        svg = SVG(svg.replace('svg:', ''))
        return svg

    @staticmethod
    def aggregate_multiple_explanations(explanations):
        at_combined = np.zeros_like(explanations[0][0])
        bt_agg_combined = {k: 0 for k in explanations[0][1].keys()}

        for at, bt_agg in explanations:
            at_combined += at
            for k in bt_agg_combined.keys():
                bt_agg_combined[k] += bt_agg[k]

        return at_combined, bt_agg_combined

def get_importance_bar(cmap='viridis'):
    gradient = np.linspace(0, 1, 256)
    gradient = -np.vstack((gradient,)*10).T

    fig = plt.figure(figsize=(2,5))

    plt.imshow(gradient, cmap=cmap)
    plt.axis('off')
    plt.text(-5, 256.5, 'Low', va='center', ha='right', fontsize=9,)
    plt.text(-5, 0.5, 'High', va='center', ha='right', fontsize=9,)
    plt.text(35, -15, 'Importance', va='center', ha='right', fontsize=11,)

    fig_bar = image_utils.fig2img(fig)

    plt.close()
    return fig
