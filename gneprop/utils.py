import logging
import numpy as np
import os
import pandas as pd
import pickle
import random
import random
import scipy.stats
import time
from collections import defaultdict
from datetime import timedelta
from functools import reduce
from functools import wraps
from rdkit import Chem, DataStructs
from rdkit.Chem import rdmolfiles, rdmolops, AllChem
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from scipy.spatial.distance import cdist
from scipy.stats import sem
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing import Any, List, Optional, Union
from typing import Callable

from gneprop.chemprop import features
from gneprop.options import GNEPropTask


def s3_setup():
    import s3fs
    fs = s3fs.S3FileSystem(profile='gneprop',
                           client_kwargs={'endpoint_url': "FILL"},
                           )
    return fs


def get_time_string():
    return time.strftime("%Y%m%d-%H%M%S")


def timeit(logger_name: str = None) -> Callable[[Callable], Callable]:
    """
    Creates a decorator which wraps a function with a timer that prints the elapsed time.

    :param logger_name: The name of the logger used to record output. If None, uses :code:`print` instead.
    :return: A decorator which wraps a function with a timer that prints the elapsed time.
    """

    def timeit_decorator(func: Callable) -> Callable:
        """
        A decorator which wraps a function with a timer that prints the elapsed time.

        :param func: The function to wrap with the timer.
        :return: The function wrapped with the timer.
        """

        @wraps(func)
        def wrap(*args, **kwargs) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            delta = timedelta(seconds=round(time.time() - start_time))
            info = logging.getLogger(logger_name).info if logger_name is not None else print
            info(f'Elapsed time = {delta}')

            return result

        return wrap

    return timeit_decorator


def describe_data(*datasets, names=None, task: Union[str, GNEPropTask] = 'binary_classification'):
    if isinstance(task, str):
        task = GNEPropTask(task)

    if names is None and len(datasets) == 4:
        names = ['Total', 'Train', 'Validation', 'Test']
    elif names is None:
        names = [f'Dataset {index}' for index in range(len(datasets))]
    else:
        assert len(datasets) == len(names)
    for dataset, name in zip(datasets, names):
        if dataset:
            labels = dataset.y
            num_total = len(dataset.smiles)
            if labels is not None:
                if task is GNEPropTask.REGRESSION:
                    print(
                        f'Dataset {name}: total = {num_total}, target variable max:({np.max(labels):.3f}), min:({np.min(labels):.3f}), avg:({np.mean(labels):.3f}), std:({np.std(labels):.3f})')
                elif task is GNEPropTask.BINARY_CLASSIFICATION:
                    num_nonzero = np.count_nonzero(labels)
                    num_total = len(labels)
                    num_zero = num_total - num_nonzero
                    print(
                        f'Dataset {name}: total = {num_total}, pos = {num_nonzero} ({num_nonzero / num_total * 100:.3f}%), neg = {num_zero} ({num_zero / num_total * 100:.3f}%)')

                elif task is GNEPropTask.MULTI_CLASSIFICATION:
                    num_total = len(labels)
                    class_values = list(set(dataset.y))
                    out_string = ''
                    for c in class_values:
                        nb_in_class = np.sum(dataset.y == c)
                        out_string += f', class = {c} ({nb_in_class / num_total * 100:.3f}%)'
                    print(f'Dataset {name:}: total = {num_total} {out_string}')
            else:
                print(f'Dataset {name}: total = {num_total}, no labels provided')
        else:
            print(f'Dataset {name} is empty')


def cast_scalar(res):
    try:
        res = res.detach()
        res = res.cpu()
    except Exception:
        pass
    res = np.round(float(res), 5)
    return res


def get_canonical_SMILES(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s))


def canonical_order(mol):
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return mol


def discretize(v, th=0.5):
    return np.array([1 if i >= th else 0 for i in v])


def discretize_multiclass(v):
    return v.argmax(axis=1)


def one_hot_encode_classes(n):
    n_hot_encoded = np.zeros((n.size, n.max() + 1))
    n_hot_encoded[np.arange(n.size), n] = 1
    return n_hot_encoded


def compute_metrics(probs, gt, th=0.5, task: Union[str, GNEPropTask] = 'binary_classification'):
    if isinstance(task, str):
        task = GNEPropTask(task)

    metrics_dict = {}

    if task is GNEPropTask.BINARY_CLASSIFICATION:
        probs_discrete = discretize(probs, th=th)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(gt, probs_discrete, average='binary')

        metrics_dict['test_auc'] = metrics.roc_auc_score(gt, probs)
        metrics_dict['test_ap'] = metrics.average_precision_score(gt, probs)
        metrics_dict['test_precision'] = precision
        metrics_dict['test_recall'] = recall
        metrics_dict['test_f1'] = f1

        # bedroc
        _t = np.stack((probs, gt), axis=1)
        _t = _t[(-_t[:, 0]).argsort()]  # sort values by score, descending
        bedroc = CalcBEDROC(_t, 1, 20)
        metrics_dict['test_bedroc'] = bedroc

    elif task is GNEPropTask.MULTI_CLASSIFICATION:
        probs_discrete = discretize_multiclass(probs)

        metrics_dict['test_auc'] = metrics.roc_auc_score(gt, probs, multi_class='ovr')
        metrics_dict['test_acc'] = metrics.accuracy_score(gt, probs_discrete)

        precision, recall, f1, _ = metrics.precision_recall_fscore_support(gt, probs_discrete, average='weighted')
        metrics_dict['test_precision'] = precision
        metrics_dict['test_recall'] = recall
        metrics_dict['test_f1'] = f1

    elif task is GNEPropTask.REGRESSION:
        metrics_dict['test_mse'] = metrics.mean_squared_error(probs, gt)
        metrics_dict['test_mae'] = metrics.mean_absolute_error(probs, gt)
        metrics_dict['test_ev'] = metrics.explained_variance(probs, gt)
        metrics_dict['test_r2'] = metrics.r2_score(probs, gt)

    return metrics_dict


def compute_metrics_multiple_folds(preds_folds):
    all_m = []
    for p in preds_folds:
        m = compute_metrics(*p)
        all_m.append(m)
    return aggregate_metrics(all_m)


# from chemprop
def get_checkpoint_paths(checkpoint_path: Optional[str] = None,
                         checkpoint_paths: Optional[List[str]] = None,
                         checkpoint_dir: Optional[str] = None,
                         ext: str = '.ckpt') -> Optional[List[str]]:
    """
    Gets a list of checkpoint paths either from a single checkpoint path or from a directory of checkpoints.

    If :code:`checkpoint_path` is provided, only collects that one checkpoint.
    If :code:`checkpoint_paths` is provided, collects all of the provided checkpoints.
    If :code:`checkpoint_dir` is provided, walks the directory and collects all checkpoints.
    A checkpoint is any file ending in the extension ext.

    :param checkpoint_path: Path to a checkpoint.
    :param checkpoint_paths: List of paths to checkpoints.
    :param checkpoint_dir: Path to a directory containing checkpoints.
    :param ext: The extension which defines a checkpoint file.
    :return: A list of paths to checkpoints or None if no checkpoint path(s)/dir are provided.
    """
    if sum(var is not None for var in [checkpoint_dir, checkpoint_path, checkpoint_paths]) > 1:
        raise ValueError('Can only specify one of checkpoint_dir, checkpoint_path, and checkpoint_paths')

    if checkpoint_path is not None:
        return [checkpoint_path]

    if checkpoint_paths is not None:
        return checkpoint_paths

    if checkpoint_dir is not None:
        checkpoint_paths = []

        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith(ext):
                    checkpoint_paths.append(os.path.join(root, fname))

        if len(checkpoint_paths) == 0:
            raise ValueError(f'Failed to find any checkpoints with extension "{ext}" in directory "{checkpoint_dir}"')

        return checkpoint_paths

    return None


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.

    When it is fit on a dataset, the :class:`StandardScaler` learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[Optional[float]]]) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.

        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none


def get_subset(x, n=1000):
    if len(x) <= n:
        return x
    try:
        return np.random.choice(x, n, replace=False)
    except AttributeError:
        return random.sample(x, n)


def aggregate_metrics(dicts):
    all_keys = reduce(lambda x, y: x.union(y.keys()), dicts, set())
    r_dict = defaultdict(dict)
    for k in all_keys:
        vs = [d.get(k) for d in dicts if d.get(k) is not None]
        mean = np.mean(vs)
        r_dict[k]['mean'] = mean
        sem = scipy.stats.sem(vs)
        r_dict[k]['sem'] = sem
        std = np.std(vs)
        r_dict[k]['std'] = std

    return r_dict


def print_save_aggregated_metrics(args, dicts, experiment_name=None):
    print('-------------------------------')
    print('FINAL AGGREGATED METRICS')
    dicts = dict(dicts)
    for key in sorted(dicts.keys()):
        mean = np.round(dicts[key]['mean'], 4)
        sem = np.round(dicts[key]['sem'], 4)
        std = np.round(dicts[key]['std'], 4)
        # print('{}: {} ± {} ({})'.format(key, mean, sem, std))
        print('{}: {} ± {}'.format(key, mean, sem))

    if experiment_name != None:
        if args.log_directory.startswith('s3:'):
            out_path = os.path.join(args.log_directory, experiment_name, 'final_metrics.pkl')
            fs = s3_setup()
            pickle.dump(dicts, fs.open(out_path, 'wb'), 0)
        else:
            out_path = os.path.join(args.log_directory, experiment_name, 'final_metrics.pkl')
            with open(out_path, 'wb') as f:
                pickle.dump(dicts, f, 0)

        print('Metrics dictionary logged at: ', out_path)
        return out_path


def get_accelerator(gpus):
    if isinstance(gpus, int) and gpus > 1:
        return 'ddp'
    else:
        return None


def find_similar_mols_matrix(test_smiles: List[str],
                             train_smiles: List[str]) -> List:
    feature_generator = features.get_features_generator('morgan')

    test_vecs = np.array([feature_generator(smiles) for smiles in tqdm(test_smiles, total=len(test_smiles))])
    train_vecs = np.array([feature_generator(smiles) for smiles in tqdm(train_smiles, total=len(train_smiles))])
    metric = 'jaccard'

    print('Computing distances')
    distances = cdist(test_vecs, train_vecs, metric=metric)
    return 1 - distances


def scale_data(x):
    return (x - x.min()) / (x.max() - x.min())


def scale_dictionary(x):
    all_values = np.array(list(x.values()))
    return {k: (v - all_values.min()) / (all_values.max() - all_values.min()) for k, v in x.items()}


def scale_data_signed(x):
    x_sign = np.sign(x)
    x_abs = np.abs(x)
    scaled_abs = (x_abs - x_abs.min()) / (x_abs.max() - x_abs.min())
    return scaled_abs * x_sign


def scale_dictionary_signed(x):
    all_values = np.array(list(x.values()))
    all_values_abs = np.abs(all_values)
    abs_max = all_values_abs.max()
    abs_min = all_values_abs.min()

    return {k: ((np.abs(v) - abs_min) / (abs_max - abs_min)) * np.sign(v) for k, v in x.items()}


def scale_data_double(x, y):
    min_tot = min(x.min(), y.min())
    max_tot = max(x.max(), y.max())

    return ((x - min_tot) / (max_tot - min_tot), (y - min_tot) / (max_tot - min_tot))


def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict


def bonddict_to_ixdict(bt, mol):
    bond_ix_to_w = dict()
    for k, v in bt.items():
        ix = mol.GetBondBetweenAtoms(*k).GetIdx()
        bond_ix_to_w[ix] = v
    return bond_ix_to_w


def dt_acc(y_true, probs):
    """
    Detection Accuracy (DtAcc): is a measurement of the maximum classification accuracy that we can achieve between in-distribution and out-of-distribution examples, by choosing the optimal threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    accuracy_scores = []
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(y_true, [m > thresh for m in probs]))

    accuracies = np.array(accuracy_scores)
    max_accuracy = accuracies.max()
    return max_accuracy


def compute_bedrocs(all_preds, alpha=1):
    all_scores = []
    for i in all_preds:
        _t = np.stack(i, axis=1)
        _t = _t[(-_t[:, 0]).argsort()]  # sort values by score, descending
        score = CalcBEDROC(_t, 1, alpha)
        all_scores.append(score)
    # return all_scores
    return np.mean(all_scores), sem(all_scores)


def compute_bulk_fp(smiles_list, feature_type='morgan', parallelize=True):
    feature_generator = features.get_features_generator(feature_type)
    if parallelize:
        ft = process_map(feature_generator, smiles_list, max_workers=16, chunksize=1000)
    else:
        ft = [feature_generator(i) for i in tqdm(smiles_list)]
    return np.vstack(ft)


def compute_sim_smiles(target_smile, other_smiles, radius=2, nBits=2048):
    target_mol = Chem.MolFromSmiles(target_smile)
    other_mols = [Chem.MolFromSmiles(i) for i in other_smiles]
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, radius, nBits=nBits)
    other_fps = [AllChem.GetMorganFingerprintAsBitVect(i, radius, nBits=nBits) for i in other_mols]

    sims = np.array(DataStructs.BulkTanimotoSimilarity(target_fp, other_fps))

    ix_sort = sims.argsort()[::-1]
    return np.array(other_smiles[ix_sort]), sims[ix_sort]


def compute_sim_mols(target_mol, other_mols, parallelize=False):
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
    other_fps = [AllChem.GetMorganFingerprintAsBitVect(i, 2, nBits=2048) for i in other_mols]

    sims = np.array(DataStructs.BulkTanimotoSimilarity(target_fp, other_fps))
    return sims


def s3_upload(s3_path, local_path):
    # upload
    fs = s3_setup()
    print('Upload logs to {} from {}'.format(s3_path, local_path))
    fs.put(local_path, s3_path, recursive=True)


def clean_local_log():
    import shutil, glob
    dir_list = glob.iglob(os.path.join(os.getcwd(), "2022*"))
    for path in dir_list:
        if os.path.isdir(path):
            shutil.rmtree(path)


def s3_read(s3_path):
    fs = s3_setup()
    S3File = fs.open(s3_path)

    return S3File


def s3_write(s3_path, df):
    fs = s3_setup()
    S3File = fs.open(s3_path, 'w')

    with S3File as f:
        df.to_csv(f, index=False)


def s3_dwl(s3_path, local_path='~/scratch/s3_ckp'):
    fs = s3_setup()
    fs.get(
        rpath=s3_path,
        lpath=local_path,
        recursive=True)


def prepare_pretrain_file(pretrain_path, local_path='./s3_temp'):
    if pretrain_path.startswith('s3:'):
        s3_dwl(pretrain_path, local_path=local_path)
        pretrain_path = local_path
    return pretrain_path


def sync_s3(args, local_dir, fold_id, group):
    remote = os.path.join(args.log_directory, group, fold_id)
    local = os.path.join(local_dir, args.wb_project, fold_id)
    s3_upload(remote, local)


def read_csv(csv_path):
    if csv_path.startswith('s3:'):
        df = pd.read_csv(s3_read(csv_path))
    else:
        df = pd.read_csv(csv_path)

    return df


def optimize_thr(val_probs, val_y, thresholds=None, return_seq=False):
    if thresholds is None:
        # thresholds = np.arange(0.005, 0.55, 0.005)
        thresholds = np.arange(0.01, 0.55, 0.01)
    best_f1 = 0
    ret_th = None
    f1_list = []
    for t in thresholds:
        f1 = f1_score(val_y, discretize(val_probs, th=t))
        f1_list.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            ret_th = t
    if not return_seq:
        return ret_th
    else:
        return ret_th, thresholds, f1_list
