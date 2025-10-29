import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import scipy
import seaborn as sns
import sklearn
import time
import torchvision
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from umap import UMAP

import gneprop.utils
from gneprop import scaffold
from gneprop.chemprop import features

sns.set_context('notebook')


def confusion_matrix_plot(ground_truth, preds, ax=None, group_names=['TN', 'FP', 'FN', 'TP'],
                          categories=['Non active', 'active'],
                          counts=True,
                          percents=True,
                          figsize=None,
                          cmap='magma',
                          title=None):
    '''
    make a visualization of the confusion matrix and returns the axes
    ---------
    group_names:   List of strings, labels for each of the groups
    categories:    List of strings, categories of the classfification
    count:         If True, show the counts for each group
    percent:       If True, shows the precentages for each group
    cbar:          If True, show the color bar.
    figsize:       Tuple, determines figure size.
    cmap:          Matplotlib colormap, map for the plot                  
    title:         String, title of the plot.
    '''
    plt.rcParams.update(plt.rcParamsDefault)
    if not figsize == None:
        plt.rcParams["figure.figsize"] = figsize
    if ax is None:
        ax = plt.gca()

    cf = confusion_matrix(ground_truth, preds)
    n_cols = cf.shape[1]
    total_per_row = np.sum(cf, axis=1)

    empty = ['' for i in range(cf.size)]
    group_labels = ["{}\n".format(el) for el in group_names] if (group_names and len(group_names) == cf.size) else empty
    group_counts = ["{0:0.0f}\n".format(el) for el in cf.flatten()] if counts else empty
    group_percentages = ["{0:.2%}".format(el / total_per_row[i // n_cols]) for i, el in
                         enumerate(cf.flatten())] if percents else empty

    all_labels = [f"{label1}{label2}{label3}".strip() for label1, label2, label3 in
                  zip(group_labels, group_counts, group_percentages)]
    all_labels = np.asarray(all_labels).reshape(cf.shape[0], cf.shape[1])

    sns.heatmap(cf, annot=all_labels, fmt="", cmap=cmap, xticklabels=categories, yticklabels=categories)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('{}'.format(title))
    return ax


def scatter_plot_regression(ground_truth, preds, errors=None, title=None,
                            ax=None, target_name='y',
                            figsize=None, classification=False, markersize=3,
                            show_thresh=False, r2=True, thresh_pred=0.5, thresh_gt=10):
    '''
    make a scatter plot visualization of the regression predictions and returns the axes
    ---------
    ground_truth:  List of floats, values against which we regress
    preds:         List of floats, predictions of the regressor
    errors:        List of floats, errors of the predictions
    title:         String, title of the plot
    thresh_gt:     Float, threshold for binary classification on ground truth
    thresh_preds:     Float, threshold for binary classification on predictions
    ax:            Matplotlib axis, to plot the figure
    target_name:   String, target name to be displayed              
    title:         String, title of the plot.
    figsize:       Tuple of int, sets the size of the figure
    classification If True, performs binarizes the predictions
    markersize     Int, sets the marker size for the scatter plot
    show_thresh    If True, shows the thresholds for classification
    '''
    plt.rcParams.update(plt.rcParamsDefault)
    if not figsize == None:
        plt.rcParams["figure.figsize"] = figsize
    if ax is None:
        ax = plt.gca()
    if errors is None:
        errors = np.zeros(ground_truth.shape)

    test_df = pd.DataFrame(data={'preds': preds, 'y': ground_truth, 'errors': errors})
    x, y, err = test_df['preds'], test_df['y'], test_df['errors']
    x_lin = np.linspace(min(x - errors), max(x + errors))
    y_lin = np.linspace(min(y), max(y))

    if classification & (thresh_pred != None):
        test_df['binary_preds'] = test_df['preds'] > thresh_pred
        test_df['Activity'] = test_df['y'] > thresh_gt

        fp_idx = test_df['binary_preds'] > test_df['Activity']
        fn_idx = test_df['binary_preds'] < test_df['Activity']
        tp_idx = ((test_df['binary_preds'] == 1) & (test_df['Activity'] == 1))
        tn_idx = ((test_df['binary_preds'] == 0) & (test_df['Activity'] == 0))

        category_names = ['fp', 'fn', 'tp', 'tn']
        categrory_indexes = [fp_idx, fn_idx, tp_idx, tn_idx]
        category_colors = ['blue', 'red', 'orange', 'green']

        for cat_idx, cat_name, color in zip(categrory_indexes, category_names, category_colors):
            markers, caps, bars = ax.errorbar(x.loc[cat_idx], y.loc[cat_idx], fmt='o',
                                              xerr=err.loc[cat_idx], c=color, label=cat_name, ms=markersize)
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

    else:
        ax.errorbar(x, y, fmt='o', xerr=err, ms=markersize)

    if show_thresh:
        ax.plot(x_lin, thresh_gt * np.ones(len(x_lin)), 'k--', alpha=0.5, linewidth=1)
        ax.plot(thresh_pred * np.ones(len(y_lin)), y_lin, 'k--', alpha=0.5, linewidth=1)

    if r2:
        slope, intercept, r_value, _, _ = scipy.stats.linregress(x, y)
        label_r2 = 'r2={0:.2}'.format(r_value)
        ax.plot(x_lin, intercept + slope * x_lin, 'k', linewidth=1, label=label_r2)
        ax.legend()

    if title == None:
        ax.set_title('Regression: Model prediction against {}'.format(target_name))
    else:
        ax.set_title('{}'.format(title))
    ax.set_ylabel(target_name)
    ax.set_xlabel('Model prediction')

    return ax


def all_stats(ground_truth, preds, ):
    # AUROC
    tick_fr = 0.1
    fpr, tpr, _ = metrics.roc_curve(ground_truth, preds)
    auroc = metrics.roc_auc_score(ground_truth, preds)
    plt.gca().set(xlim=(0, 1.01), ylim=(0, 1.01), xlabel='FPR', ylabel='TPR')

    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.plot(fpr, tpr, label="AUROC=" + str(round(auroc, 2)))
    plt.title('ROC')
    plt.legend(loc=4)
    plt.xticks(np.arange(0, 1 + tick_fr, tick_fr))
    plt.yticks(np.arange(0, 1 + tick_fr, tick_fr))
    plt.show()

    # AUPCR
    precision, recall, _ = metrics.precision_recall_curve(ground_truth, preds)
    baseline = len(ground_truth[ground_truth == 1]) / len(ground_truth)
    plt.axhline(y=baseline, color='black', linestyle='--')
    average_precision = sklearn.metrics.average_precision_score(ground_truth, preds)
    auc = metrics.auc(recall, precision)
    plt.plot(recall, precision, label="AP=" + str(round(average_precision, 2)))
    plt.gca().set(xlim=(0, 1.01), ylim=(0, 1.01), xlabel='Recall', ylabel='Precision')

    plt.title('PR Curve')
    plt.legend(loc=1)
    plt.xticks(np.arange(0, 1 + tick_fr, tick_fr))
    plt.yticks(np.arange(0, 1 + tick_fr, tick_fr))
    plt.show()

    # CONFUSION MATRIX
    discretized_preds = gneprop.utils.discretize(preds)
    cf_m = sklearn.metrics.confusion_matrix(ground_truth, discretized_preds)
    # cf_m_norm = sklearn.metrics.confusion_matrix(ground_truth, discretized_preds, normalize='true')
    # cf_m = np.core.defchararray.add(cf_m, cf_m_norm)
    # print(cf_m)
    disp = sklearn.metrics.ConfusionMatrixDisplay(cf_m)
    disp.plot()
    plt.show()
    plt.close()
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(ground_truth, discretized_preds,
                                                                               average='binary')
    print('Precision: ', round(precision, 2))
    print('Recall: ', round(recall, 2))
    print('F1: ', round(f1, 2))
    print('AUROC: ', round(auroc, 2))
    print('AUPRC (AP): ', round(average_precision, 2))


def sns_boxplots(value_lists, x_axis, y_axis=None):
    if y_axis is None:
        y_axis = 'value'
    dfs = [pd.DataFrame({'label': [x_ax] * len(vl), y_axis: vl}) for x_ax, vl in zip(x_axis, value_lists)]
    df = pd.concat(dfs, axis=0)
    bp = sns.boxplot(data=df, x='label', y=y_axis)
    bp.set(xlabel=None)
    return bp
    # return df


def format_results(agg_dict, error_metric='sem', format_type='default'):
    if format_type == 'default':
        for k, v in agg_dict.items():
            mean = v['mean']
            e = v[error_metric]
            print(k, f'{mean:.3f} ± {e:.3f}')
    elif format_type == 'latex':
        for k, v in agg_dict.items():
            mean = v['mean']
            e = v[error_metric]
            print(k, f'{mean:.3f} \\textcolor{{gray}}{{$\pm$ {e:.3f}}}')


def auroc_k_fold_plot(outcome_pairs, ax=None, label=None, plot_baseline=True, legend_loc=4, legend_outside=False):
    mean_fpr = np.linspace(0, 1, 200)
    tick_fr = 0.1

    if ax is None:
        ax = plt.gca()

    fpr_all = []
    tpr_all = []
    fold_label_all = []
    all_aurocs = []
    for index, (preds, y_true) in enumerate(outcome_pairs):
        all_aurocs.append(metrics.roc_auc_score(y_true, preds))
        fpr, tpr, _ = metrics.roc_curve(y_true, preds)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0
        fpr_all.extend(mean_fpr)
        tpr_all.extend(interp_tpr)
        fold_label_all.extend([str(index)] * len(mean_fpr))

    auroc_mean = np.mean(all_aurocs)
    auroc_sem = scipy.stats.sem(all_aurocs)

    label = f"{label + ':' if label is not None else ''} AUROC={auroc_mean:.3f} ± {auroc_sem:.3f}"
    auroc_df = pd.DataFrame({'FPR': fpr_all, 'TPR': tpr_all, 'exp_id': fold_label_all})

    sns.lineplot(data=auroc_df, x='FPR', y='TPR', ax=ax, legend='full', label=label, ci=68)

    if plot_baseline:
        ax.plot([0, 1], [0, 1], color='black', linestyle='--')

    ax.set(xlim=(0, 1.01), ylim=(0, 1.01))

    ax.set_title('ROC')
    if legend_outside:
        ax.legend(loc=legend_loc, bbox_to_anchor=(1, 0.5))
    else:
        ax.legend(loc=legend_loc)
    ax.set_xticks(np.arange(0, 1 + tick_fr, tick_fr))
    ax.set_yticks(np.arange(0, 1 + tick_fr, tick_fr))
    return ax


def auprc_k_fold_plot(outcome_pairs, ax=None, label=None, plot_baseline=True, legend_loc=1, legend_outside=False,
                      color=None):
    mean_recall = np.linspace(0, 1, 100)
    tick_fr = 0.1

    if ax is None:
        ax = plt.gca()

    recall_all = []
    precision_all = []
    fold_label_all = []
    all_ap = []
    for index, (preds, y_true) in enumerate(outcome_pairs):
        all_ap.append(metrics.average_precision_score(y_true, preds))
        precision, recall, _ = metrics.precision_recall_curve(y_true, preds)

        reversed_recall = np.fliplr([recall])[0]
        reversed_precision = np.fliplr([precision])[0]

        interp_precision = np.interp(mean_recall, reversed_recall, reversed_precision)
        interp_precision[0] = 1
        recall_all.extend(mean_recall)
        precision_all.extend(interp_precision)
        fold_label_all.extend([str(index)] * len(mean_recall))

    ap_mean = np.mean(all_ap)
    ap_sem = scipy.stats.sem(all_ap)

    label = f"{label + ':' if label is not None else ''} AP={ap_mean:.3f} ± {ap_sem:.3f}"
    auprc_df = pd.DataFrame({'Recall': recall_all, 'Precision': precision_all, 'exp_id': fold_label_all})

    if color is not None:
        sns.lineplot(data=auprc_df, x='Recall', y='Precision', ax=ax, legend='full', label=label, ci=68, color=color)
    else:
        sns.lineplot(data=auprc_df, x='Recall', y='Precision', ax=ax, legend='full', label=label, ci=68)

    if plot_baseline:
        av_baseline = 0
        for _, y_true in outcome_pairs:
            baseline = len(y_true[y_true == 1]) / len(y_true)
            av_baseline += baseline
        av_baseline /= len(outcome_pairs)
        ax.axhline(y=av_baseline, color='black', linestyle='--')

    ax.set(xlim=(0, 1.01), ylim=(0, 1.01))

    ax.set_title('PR Curve')
    if legend_outside:
        ax.legend(loc=legend_loc, bbox_to_anchor=(1, 0.5))
    else:
        ax.legend(loc=legend_loc)
    ax.set_xticks(np.arange(0, 1 + tick_fr, tick_fr))
    ax.set_yticks(np.arange(0, 1 + tick_fr, tick_fr))
    return ax


def draw_mols_pred_gt(smiles, preds, gt, return_scaffold_mol=False):
    scaffold_smiles = scaffold.generate_scaffold(smiles[0])
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
    AllChem.Compute2DCoords(scaffold_mol)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    for m in mols:
        AllChem.GenerateDepictionMatching2DStructure(m, scaffold_mol)

    labels = [f'Pred:\t{p:.2f}\n\b\nLabel:\t{g}' for p, g in zip(preds, gt)]

    img = Draw.MolsToGridImage(mols, molsPerRow=len(mols), subImgSize=(200, 200), legends=labels, useSVG=True)

    if return_scaffold_mol:
        return img, scaffold_mol
    else:
        return img


def plot_confusion_matrix_as_torch_tensor(cm):
    cm_size = cm.shape[0]
    df_cm = pd.DataFrame(cm, index=np.arange(cm_size), columns=np.arange(cm_size))
    plt.figure()

    sns.set(font_scale=1.2)
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
    buf = io.BytesIO()

    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    return im


def value_to_color(v, cmap='Greens'):
    cmap_pos = matplotlib.cm.get_cmap(cmap)
    return cmap_pos(v)


def value_to_color_pos_neg(v, cmap_pos='Greens', cmap_neg='Reds'):
    cmap_pos = matplotlib.cm.get_cmap(cmap_pos)
    cmap_neg = matplotlib.cm.get_cmap(cmap_neg)
    return cmap_pos(v) if v >= 0 else cmap_neg(-v)


def list_to_color(l, cmap='Greens'):
    ix_to_color = dict()
    for i in range(len(l)):
        ix_to_color[i] = value_to_color(l[i], cmap=cmap)
    return ix_to_color


def list_to_color_pos_neg(l, cmap_pos='Greens', cmap_neg='Reds'):
    ix_to_color = dict()
    for i in range(len(l)):
        ix_to_color[i] = value_to_color_pos_neg(l[i], cmap_pos=cmap_pos, cmap_neg=cmap_neg)
    return ix_to_color


def dict_to_color(d, cmap='Greens', ):
    return {k: value_to_color(v, cmap) for k, v in d.items()}


def dict_to_color_pos_neg(d, cmap_pos='Greens', cmap_neg='Reds'):
    return {k: value_to_color_pos_neg(v, cmap_pos, cmap_neg) for k, v in d.items()}


def gen_palette(df):
    """Generate palette for barstrip plot which set the color of GNEprop model to blue if only 1 GNEprop model detected in the dataframe

    Parameters:
    df (pandas DataFrame): produced by function barstrip_df

    Returns:
    seaborn.color_palette or None

    """
    models = df['bar'].unique()
    bar_names = list(df['bar'].unique())

    bar_names_skip = [b for b in bar_names if not b.startswith('GNEprop')]

    GNE_model_name = None
    cnt = 0
    for n in bar_names:
        if n.startswith('GNEprop'):
            cnt += 1
            GNE_model_name = n

    if cnt > 1:
        return None

    palette = dict()
    palette[GNE_model_name] = sns.color_palette("Set2", 8)[2]

    for idx, b in enumerate(bar_names_skip):
        palette[b] = sns.color_palette('Set2', 8)[3 + idx]

    return palette


def barstrip_df(path, model='bar', metric_rename_dict=None, model_rename_dict=None, key=None):
    """Produces a dataframe for barstrip plot

    Parameters:
    path (str): directory where multiple model-metric-score .pkl files are stored
    model (str): values can be 'bar' or 'group'. If 'bar', bars are grouped by metrics, and each bar in a group represents a model, if 'group', bars are grouped by model. Default value is 'bar'. 
    metric_rename_dict (dict): keys are old metric names, values are new metric names. If not None, metrics will be filtered, reordered and renamed according to this dictionary. Default values is None.
    model_rename_dict (dict): keys are old model names, values are new model names. If not None, models will be filtered, reordered and renamed according to this dictionary. Default values is None.
    key (Union[str, int]): name of parent group (e.g. dataset) of the experiments of a model

    Returns:
    pandas DataFrame

    """

    fs = os.listdir(path)
    #     metrics = None
    df_list = []
    for i in range(len(fs)):
        f_name = fs[i]
        try:
            if key:
                for x in pickle.load(open(os.path.join(path, f_name), 'rb')):
                    for k in x.keys():
                        if k == key:
                            scores = x[key]  # this is actually a dct_list
            else:
                scores = pickle.load(open(os.path.join(path, f_name), 'rb'))

            if isinstance(scores, dict):
                metrics = list(scores.keys())
            elif isinstance(scores, list):
                metrics = list(scores[0].keys())
            df = pd.DataFrame(scores)
            df['model'] = f_name.split('.')[0]
            if model == 'bar':
                df['bar'] = df['model']
                var_name = 'group'
            elif model == 'group':
                df['group'] = df['model']
                var_name = 'bar'

            df_list.append(df)

        except EOFError as err:
            print('File {} has EOFError: {}'.format(i, err))

    meta_df = pd.concat(df_list)

    # filter metrics

    metrics_keep = list(metric_rename_dict.keys())

    meta_df = meta_df[metrics_keep + [model]]

    meta_df = pd.melt(meta_df,
                      id_vars=model,
                      value_vars=metrics_keep,
                      var_name=var_name,
                      value_name='score')

    if model == 'bar':
        metric_col = 'group'
        model_col = 'bar'
    else:
        metric_col = 'bar'
        model_col = 'group'

    # rename and order dataframe
    if metric_rename_dict is not None:
        meta_df[metric_col].replace(metric_rename_dict, inplace=True)
        meta_df[metric_col] = pd.Categorical(meta_df[metric_col], list(metric_rename_dict.values()))

    if model_rename_dict is not None:
        meta_df[model_col].replace(model_rename_dict, inplace=True)
        meta_df[model_col] = pd.Categorical(meta_df[model_col], list(model_rename_dict.values()))

    #     meta_df = meta_df.sort_values(metric_col)

    return meta_df


def barstrip_plot(df, fname=None, bound=False, bar_palette=None, size=2, yaxis_title=None):
    """Generate palette for barstrip plot which set the color of GNEprop model to blue if only 1 GNEprop model detected in the dataframe

    Parameters:
    df (pandas DataFrame): produced by function barstrip_df
    fname (str or None): if str, plot will be saved and named by this argument. if None, plot will not be saved. Default value is None.
    bar_palette (dict, str, or None): if str, should be color palette name; if dict, should be a dictionary with bar names as keys and color as values: if None, 'Set2' will be used with the barplot

    Returns:
    
    """

    if not bar_palette:
        bar_palette = gen_palette(df)

    n_group = df['group'].nunique()
    sns.set_theme(style="whitegrid")
    colors = ['white' for i in range(n_group)]
    palette = sns.xkcd_palette(colors)
    sns.stripplot(x="group", y="score", hue='bar', data=df, dodge=True, marker='o', palette=sns.xkcd_palette(colors),
                  linewidth=0.2, size=size, edgecolor="black")
    if not bar_palette:
        bar_palette = 'Set2'
    sns.barplot(x="group", y="score", hue='bar', data=df, palette=bar_palette)
    handles, labels = plt.gca().get_legend_handles_labels()
    cnt = len(handles) // 2
    plt.legend(handles[cnt:], labels[cnt:], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if bound:
        plt.ylim([0., 1.])
    plt.gca().set(xlabel=None)
    if yaxis_title:
        plt.ylabel(yaxis_title)
    if fname:
        if fname.endswith('pdf'):
            plt.savefig(fname, bbox_inches='tight')
        else:
            plt.savefig(fname, dpi=300, bbox_inches='tight')
