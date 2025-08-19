import os
import sys
sys.path.append('<path_of_GNEprop_repo>')
import gneprop_pyg
import gneprop
import data
from gneprop import chem_utils
import pandas as pd
import numpy as np
import random
import logging
from collections import defaultdict
from rdkit import Chem
from tqdm import tqdm
from pytorch_lightning import seed_everything
import clr
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib as mpl



SEED=42
seed_everything(SEED)

logging.getLogger().setLevel(logging.INFO)


def q_value(data, perc):
    """
    This function produces min and max values according to percentile for colormap in plot functions
    Args:
        data (numpy array): input
        perc (float): percentile that between 0 and 100 inclusive
    Returns:
        tuple of floats: will be later used to define the data range covers by the colormap
    """
    vmin = np.nanpercentile(data, perc)
    vmax = np.nanpercentile(data, 100 - perc)
    
    vmin=int(np.floor(vmin))     # colorbar min value
    vmax=int(np.ceil(vmax))      # colorbar max value

    return vmin, vmax



def show_cbar(ax, cmap_name, vmin, vmax, prop_name, shrink=None, location=None):

    md=(vmax-vmin)/2
    
    cax, _ = matplotlib.colorbar.make_axes(ax, location=location, shrink=shrink)
    
    cmap = mpl.cm.get_cmap(cmap_name)
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, orientation='vertical', 
                           norm=plt.Normalize(vmin,vmax))
    
    v1 = np.linspace(vmin, vmax, 3, endpoint=True)
    cb.set_ticks(v1)
    cb.ax.set_yticklabels(['<={}'.format(vmin), md, '>={}'.format(vmax)])


def get_prop_df(mols):  
    prop_dict = defaultdict(list)
    
    for mol in tqdm(mols, position=0, leave=True):
        # basic prop
        prop_dict['mw'].append(Chem.Descriptors.MolWt(mol))
        prop_dict['tpsa'].append(Chem.Descriptors.TPSA(mol))
        prop_dict['logp'].append(Chem.Descriptors.MolLogP(mol))
        prop_dict['hbd'].append(Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol))
        prop_dict['hba'].append(Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol))
        prop_dict['rotbonds'].append(Chem.rdMolDescriptors.CalcNumRotatableBonds(mol))
        
        # filters
        f = chem_utils.compute_filters(mol)
        prop_dict['lipinski'].append(f['lipinski'])
        prop_dict['lipinski_1'].append(f['lipinski_1'])
        prop_dict['ghose'].append(f['ghose'])
        prop_dict['veber'].append(f['veber'])
        prop_dict['rule3'].append(f['rule3'])
        prop_dict['reos'].append(f['reos'])
        prop_dict['drug-like'].append(f['drug-like'])
        prop_dict['antibiotic-like'].append(f['antibiotic-like'])
 
    prop_df = pd.DataFrame.from_dict(prop_dict)

    return prop_df

def gen_mols(smiles):
    logging.info('Generating molecules...')
    mols_subset = [Chem.MolFromSmiles(i) for i in tqdm(smiles, position=0, leave=True)]
    
    return mols_subset
    
def gen_dataset(dataset, smile_field='SMILES', sample_size=None, indicies=None):
    if isinstance(dataset, str) and os.path.exists(dataset):
        dataset = data.load_dataset_multi_format(dataset)

    elif isinstance(dataset, pd.DataFrame):
        dataset = data.MolDatasetOD(list(dataset[smile_field].values))

    if sample_size:
        indicies = random.sample(list(range(len(dataset))), sample_size)
        dataset_subset = data.MoleculeSubset(dataset, indicies)
    else:
        dataset_subset = dataset
            
    if indicies:
        dataset_subset = data.MoleculeSubset(dataset, indicies)
        
    return dataset_subset

def get_mrgs(smiles):
    mrgs = []

    for s in tqdm(smiles):
        mol = Chem.MolFromSmiles(s)
        mg = gneprop.chemprop.features.morgan_counts_features_generator(mol)
        mrgs.append(mg)
        
    return mrgs

def gen_repr(dataset, ckp_dir=None, smile_field='SMILES', use_projection_layers=0, model_to_use=None, mode='unsupervised', sample_size=None, indicies=None, return_mol=False, use_gpu=False, repr_type='gne'):
    
    if repr_type=='morgan':
        if isinstance(dataset, str) and os.path.exists(dataset):
            df = pd.read_csv(dataset)

        elif isinstance(dataset, pd.DataFrame):
            df = dataset

        smiles = df['SMILES'].values
        reprs = np.array(get_mrgs(smiles))
        
    elif repr_type=='gne':
        assert ckp_dir is not None
        
        dataset_subset = gen_dataset(dataset, smile_field='SMILES', sample_size=sample_size, indicies=indicies)

        if not model_to_use:
            model_to_use = gneprop_pyg.GNEprop(mol_features_size=0)

        ckp_path = gneprop.utils.get_checkpoint_paths(checkpoint_dir=ckp_dir)[0]

        if mode=='unsupervised':
            model_to_use = clr.SimCLR.load_from_checkpoint(ckp_path)

        elif mode=='supervised':
            model_to_use = model_to_use.load_from_checkpoint(ckp_path)

        if use_gpu:
            model_to_use.to(device='cuda:0')   
        model_to_use.eval()

        reprs = model_to_use.get_representations_dataset(dataset_subset, use_projection_layers)

        smiles = dataset_subset.smiles

    mols_subset = None
    if return_mol:
        gen_mols(smiles)

    return reprs, mols_subset, smiles


def gen_emb_df(reprs, prop, prop_name, method='UMAP', metric=None, reduce=True, clean=True, clean_freq=True, n_neighbors=15, min_dist=0.1, sample_frac=None):
    
    assert reprs.shape[1]==2
    print('Using existed 2d representations')
    df_repr = pd.DataFrame(reprs, columns=['repr_x', 'repr_y'])
    df_repr = df_repr.reset_index(drop=True)
    df_prop = prop.reset_index(drop=True)
    df_new = pd.concat([df_prop, df_repr], axis=1)

    return df_new



def plot_repr(emb_df, prop_name=None, mode='quali', title=None, ax=None, show_ticks=False, show_legend=False, show_labels=False, save_image=False, show_score=True, s=40, cl=None, fname='repr_vs_prop.png'):
        
    if not ax:
        fig, ax = plt.subplots(figsize=(7, 5))
            
    lbl_count = emb_df[prop_name].nunique()
    
    # plot
    if mode=='bg':
        if not cl:
            cl = 'lightgray'
        sns.scatterplot(ax=ax, x='repr_x', y='repr_y', data=emb_df, color=cl, s=s, alpha=0.5)

    elif mode=='quali':
        if not cl:
            cl = 'tab20'
            if len(list(emb_df[prop_name].unique())) < 10:
                cl = 'Set1'
        if isinstance (cl, list):
            cl = cl[:lbl_count]
        sns.scatterplot(ax=ax, x='repr_x', y='repr_y', data=emb_df, hue=prop_name, palette=cl, s=s, alpha=0.5)
    
    elif mode=='quanti':
        if not cl:
            cl = "Spectral"
        vmin, vmax = q_value(emb_df[prop_name].values, perc=1)
        sc = ax.scatter(emb_df['repr_x'].values, emb_df['repr_y'].values, c=emb_df[prop_name], cmap=cl, s=s, vmin=vmin, vmax=vmax, alpha=0.5, marker='o', edgecolors='none')
        show_cbar(ax, cl, vmin, vmax, prop_name, shrink=0.3, location='right')

    # title
    if not title or title=='empty':
        ax.set_title('')
    else:
        ax.set_title(title, fontsize=11)
    
    # legend
    if ax.get_legend() and not show_legend:
        ax.get_legend().remove()
    elif show_legend and mode=='quali':
        ax.legend(bbox_to_anchor=(1.02,1), borderaxespad=0, loc=2, frameon=False, fontsize=8)
      
    # ticks
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        
    if not show_labels:
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)
        
#     ax.set_aspect(1)
        
    if save_image:
        fig.savefig(fname, dpi=300)
        
        
def plot_repr_multi_prop(reprs, prop_df, props, mode='quali', nrows=2, ncols=3, show_legend=False, show_score=False, fname=None, s=40, cl=None, tight=False, metric='euclidean', clean=True, clean_freq=True, n_neighbors=15, min_dist=0.1, title=None, dpi=300):

    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*8, nrows*6))   
    if nrows==1 & ncols==1:
            ax_f = [ax]
    else:
        ax_f = ax.flatten()
           
    for idx, prop_name in enumerate(props):
        ax = ax_f[idx]

        emb_df = gen_emb_df(reprs, prop_df, prop_name=prop_name, method='UMAP', metric=metric, reduce=True, clean=clean, clean_freq=clean_freq, n_neighbors=n_neighbors, min_dist=min_dist)
        plot_repr(emb_df, prop_name, mode=mode, ax=ax, show_score=show_score, show_legend=show_legend, s=s, cl=cl)

        # set title
        # ax.set_title("{} ({})".format(prop_name, metric), fontsize=14)
        ax.set_aspect(1./ax.get_data_ratio())

    if tight:
        plt.tight_layout()
        
    if title:
        plt.suptitle(title, fontsize=16)
        
    if fname:
        if fname.endswith('pdf'):
            plt.savefig(fname, bbox_inches='tight')
        else:
            plt.savefig(fname, dpi=dpi, bbox_inches='tight')
            
    return emb_df