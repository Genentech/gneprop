import torch_geometric
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import sys

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.stats import sem

import random
from random import Random
from collections import Counter, defaultdict

import torch
from sklearn.metrics import confusion_matrix

from gneprop.chemprop import features
from scipy.spatial.distance import mahalanobis


def add_n_occ(df, target_label):
    """
    Adds total number of occurrences of target_label for each entry of df
    """
    count_dict = df[target_label].value_counts().to_dict()
    df['n_occ'] = df[target_label].apply(lambda i: count_dict[i])
    return df


def split_classes(classes, num_train, num_test, seed=0):
    """
    Split a list of classes randomly in two sets of num_train and num_test classes, respectively.
    """
    assert (num_train + num_test) == len(classes)

    classes = np.array(classes)
    random = Random(seed)
    classes_indices = list(range(len(classes)))
    random.shuffle(classes_indices)
    train_indices = classes_indices[:num_train]
    test_indices = classes_indices[num_train:]
    return classes[train_indices], classes[test_indices]


def split_indices_v2(df, num_train, num_test, target_label, seed=0, num_to_keep=1):
    """
    Split a dataframe randomly in two dataframes (train and test) of num_train and num_test samples, respectively.
    Moreover, it ensures that the training dataframe has at least num_to_keep samples for each target_label.
    If a certain target_label has less than num_to_keep samples in total, the maximum number of samples for target_label are put in the training set.
    """
    assert (num_train + num_test) == len(df)

    random = Random(seed)
    df_indices = list(df.index)
    random.shuffle(df_indices)

    df = df.loc[df_indices]

    index_saved_list = []
    for i in df[target_label].unique():
        index_saved = df[df[target_label] == i].index[:num_to_keep]
        index_saved_list.extend(index_saved)

    train_indices = index_saved_list
    other_indices = list(set(list(df.index)) - set(train_indices))

    random.shuffle(other_indices)

    train_indices_additional = other_indices[:num_train - len(train_indices)]
    test_indices = other_indices[num_train - len(train_indices):]

    train_indices = train_indices + train_indices_additional

    return df.loc[train_indices], df.loc[test_indices]


def compute_centroids(df, reprs, target_label):
    """
    Aggregates a representation matrix based on column target_label of df.

    :param df: A dataframe of shape N x M that includes a column named target_label.
    :param reprs: A matrix of N representations with dimension H (N x H matrix).
    :param target_label: Column in df used for the aggregation.
    :return: A dict where the keys are the unique values in df.target_label, and the values are dict with two entris: 'mean' and 'var', reporting the mean and the variance of the representations for that label.
    """
    unique_classes = df[target_label].unique()
    num_unique_classes = len(unique_classes)

    reprs_mean_var = defaultdict(dict)

    for i in unique_classes:
        target_index = df[df[target_label] == i].index.values
        reprs_mean_var[i]['mean'] = reprs[target_index].mean(axis=0)
        reprs_mean_var[i]['var'] = reprs[target_index].var(axis=0)

    return reprs_mean_var


EPS = 0.0000000001


def get_inv_diag_cov(x):
    """
    Compute the inverse of the matrix x, assuming x is diagonal.
    """
    x = x + EPS
    return np.diag(1 / x)


def neg_distance_to_clusters(x, mean_var_dict):
    """
    Compute the maximum of the negative distances between x and each cluster (Eq. 2 in K. Lee, NIPS. 2018)

    :param x: np.array
    :param mean_var_dict: dict as returned by the function compute_centroids
    """
    neg_dists = []
    for k, v in mean_var_dict.items():
        c_mean = v['mean']
        c_var = v['var']

        c_var_p = get_inv_diag_cov(c_var)
        n_dist = -mahalanobis(x, c_mean, c_var_p)
        neg_dists.append(n_dist)
    return max(neg_dists)


def tnr_at_tpr95_custom(gt, preds):
    ls = np.linspace(max(preds), min(preds), 1000)

    tpr = []
    tnr = []
    fpr = []
    for thr in ls:
        preds_bin = (preds > thr).astype('float')
        tn, fp, fn, tp = confusion_matrix(gt, preds_bin).ravel()

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
        tnr.append(tn / (tn + fp))

    tpr = np.array(tpr)
    tnr = np.array(tnr)
    fpr = np.array(fpr)

    if min(tpr) <= 0.95:
        tnr_before = tnr[tpr <= 0.95][-1]
    else:
        tnr_before = tnr[0]

    if max(tpr) >= 0.95:
        tnr_after = tnr[tpr >= 0.95][0]
    else:
        tnr_after = tnr[-1]

    return (tnr_before + tnr_after) / 2
