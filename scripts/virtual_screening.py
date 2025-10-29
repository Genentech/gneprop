import os
import sys
from argparse import ArgumentParser

import numpy as np
import ray
from pytorch_lightning import Trainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from gneprop_pyg import convert_to_dataloader, predict_from_checkpoints
from gneprop import utils

import data
from tqdm import tqdm
import glob
import pandas as pd
import re
from pathlib import Path

import GNEpropCPP


def get_paths_indices(dir_path, start_ix=None, end_ix=None, skip_holes=False):
    file_names = [os.path.basename(i) for i in glob.glob(f'{dir_path}/*_*.smi')]
    dict_num_to_file = {}
    regex = re.compile(r'\d+')
    for fn in file_names:
        try:
            num = int(regex.findall(fn)[0])
            dict_num_to_file[num] = fn
        except:
            continue

    if start_ix is None:
        start_ix = min(dict_num_to_file.keys())
    if end_ix is None:
        end_ix = max(dict_num_to_file.keys())

    if skip_holes:
        selected_paths = []
        for i in range(start_ix, end_ix):
            try:
                selected_paths.append(os.path.join(dir_path, dict_num_to_file[i]))
            except KeyError:
                pass
    else:
        selected_paths = [os.path.join(dir_path, dict_num_to_file[i]) for i in range(start_ix, end_ix)]

    return selected_paths


def read_z_csv(path):
    if path[-4:] == '.smi':
        names = ['SMILES', 'Z_ID']
    else:
        names = ['SMILES']
    return pd.read_csv(os.path.join(path), delim_whitespace=True, names=names)


@ray.remote(num_cpus=8, num_gpus=1)
class RunPredictionObject(object):
    def __init__(self, data_id, args, ckpt_path, data_name=None):
        self.data_id = data_id
        self.args = args
        self.ckpt_path = ckpt_path
        self.data_name = data_name

    def run_prediction(self):
        if self.data_name is not None:
            print(f'Predicting data: {self.data_name}')
        dataloader = convert_to_dataloader(self.data_id, batch_size=args.batch_size, num_workers=args.num_workers)
        preds, _ = predict_from_checkpoints(data=dataloader, checkpoint_path=self.ckpt_path, gpus=1)
        return preds


def main_cli(args):
    if args.data_dir_path != '':
        selected_paths = get_paths_indices(args.data_dir_path, args.indices[0], args.indices[1])
    elif args.data_file_path != '':
        selected_paths = [args.data_file_path]
    else:
        raise ValueError

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    GNEpropCPP.configCPPOptions(args.compatible_features, [], 0, args.fprecision==16)

    if args.parallelize_folds:
        ray.init(dashboard_host='0.0.0.0', ignore_reinit_error=True, )  # port=8265
        for path in tqdm(selected_paths):
            if args.skip_if_exists:
                output_path = os.path.join(args.output_dir, os.path.basename(path))
                if os.path.exists(output_path):
                    print('Path exists, skipping')
                    continue
            df = read_z_csv(path)
            values = df.SMILES.values if (args.max_molecules < 0 or args.max_molecules >= len(df.SMILES.values)) else df.SMILES.values[0:args.max_molecules]
            dataset = data.MolDatasetOD(values, code_version=args.code_version)
            data_id = ray.put(dataset)

            all_checkpoints = utils.get_checkpoint_paths(checkpoint_dir=args.model_path)

            all_refs = []
            for checkpoint in all_checkpoints:
                remote_run_prediction = RunPredictionObject.options(num_cpus=args.num_workers, num_gpus=1).remote(data_id, args, ckpt_path=checkpoint, data_name=path)
                ref = remote_run_prediction.run_prediction.remote()
                all_refs.append(ref)

            out = ray.get(all_refs)  # waits until all executions terminate
            all_preds = [i.flatten() for i in out]
            # all_res = np.stack(all_preds, axis=1)  # n_mols x n_models

            for model_ix in range(len(all_checkpoints)):
                df[model_ix] = all_preds[model_ix]

            df.to_csv(os.path.join(args.output_dir, os.path.basename(path)), index=False)

    else:
        for path in tqdm(selected_paths):
            if args.skip_if_exists:
                output_path = os.path.join(args.output_dir, os.path.basename(path))
                if os.path.exists(output_path):
                    print('Path exists, skipping')
                    continue
            df = read_z_csv(path)
            if df.SMILES.values[0] == 'SMILES':
                values = df.SMILES.values[1:] if (args.max_molecules < 0 or args.max_molecules+1 >= len(df.SMILES.values)) else df.SMILES.values[1:args.max_molecules+1]
                dataset = data.MolDatasetOD(values, code_version=args.code_version)
                has_extra = True
            else:
                values = df.SMILES.values if (args.max_molecules < 0 or args.max_molecules >= len(df.SMILES.values)) else df.SMILES.values[0:args.max_molecules]
                dataset = data.MolDatasetOD(values, code_version=args.code_version)
                has_extra = False
            dataloader = convert_to_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

            preds, _ = predict_from_checkpoints(data=dataloader, checkpoint_dir=args.model_path, gpus=args.gpus, precision=args.fprecision)

            if has_extra:
                # Add an extra 0 at the beginning for the "SMILES" line.
                df['pred'] = [0.0] + preds
            else:
                df['pred'] = preds
            df.to_csv(os.path.join(args.output_dir, os.path.basename(path)), index=False)


def check_arguments(args):
    assert args.data_dir_path != args.data_file_path  # one of the two needs to be not empty
    if args.data_dir_path != '':
        assert args.indices is not None


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir_path', type=str, default='')
    parser.add_argument('--data_file_path', type=str, default='')
    parser.add_argument('--indices', nargs=2, metavar=('start_ix', 'end_ix'), type=int, default=None)
    parser.add_argument('--num_workers', nargs='?', const=-1, default=1, type=int)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--skip_if_exists',  action="store_true")

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument("--parallelize_folds", action="store_true")
    parser.add_argument('--max_molecules', type=int, default=-1)

    parser.add_argument('--fprecision', type=int, default=32)
    parser.add_argument('--compatible_features', type=bool, default=True)
    parser.add_argument('--code_version', type=int, default=2)

    args = parser.parse_args()

    check_arguments(args)

    main_cli(args)
