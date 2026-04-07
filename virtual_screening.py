import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import data
from gneprop_pyg import convert_to_dataloader, predict_from_checkpoints


def infer_input_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".csv", ".feather", ".ftr"}:
        return suffix.lstrip(".")
    if suffix in {".smi", ".smiles", ".txt", ".tsv"}:
        return "smi"
    raise ValueError(f"Unsupported input file extension: {suffix}")


def read_smi_file(path: str, smiles_column: str = "SMILES", id_column: str = "ID") -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", header=None, comment="#")
    if df.shape[1] == 0:
        raise ValueError(f"Input file is empty: {path}")

    if df.shape[1] == 1:
        df.columns = [smiles_column]
    else:
        columns = [smiles_column, id_column]
        extra_columns = [f"extra_{i}" for i in range(df.shape[1] - len(columns))]
        df.columns = columns + extra_columns

    df = df.dropna(subset=[smiles_column]).reset_index(drop=True)
    return df


def load_input_dataframe(path: str, input_format: str, smiles_column: str, id_column: str) -> pd.DataFrame:
    if input_format == "csv":
        df = pd.read_csv(path)
    elif input_format in {"feather", "ftr"}:
        df = pd.read_feather(path)
    elif input_format == "smi":
        df = read_smi_file(path, smiles_column=smiles_column, id_column=id_column)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    if smiles_column not in df.columns:
        raise ValueError(f'Missing required smiles column "{smiles_column}" in {path}')

    df = df.dropna(subset=[smiles_column]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No valid SMILES rows found in {path}")

    return df


def build_dataset(df: pd.DataFrame, smiles_column: str, legacy: bool) -> data.MolDatasetOD:
    return data.MolDatasetOD(smiles_list=df[smiles_column].astype(str).values, legacy=legacy)


def resolve_output_path(input_path: str, output_path: str | None, output_dir: str | None) -> str:
    if output_path and output_dir:
        raise ValueError("Only one of --output_path or --output_dir can be provided.")

    if output_path:
        return output_path

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(output_dir) / Path(input_path).name)

    default_dir = Path("predictions")
    default_dir.mkdir(parents=True, exist_ok=True)
    return str(default_dir / Path(input_path).name)


def add_per_model_predictions(df: pd.DataFrame, per_model_preds) -> pd.DataFrame:
    for index, preds in enumerate(per_model_preds):
        df[f"pred_model_{index}"] = preds
    return df


def run_screening(args) -> str:
    input_format = args.input_format or infer_input_format(args.input_path)
    df = load_input_dataframe(args.input_path, input_format, args.smiles_column, args.id_column)

    dataset = build_dataset(df, smiles_column=args.smiles_column, legacy=not args.no_legacy)
    dataloader = convert_to_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    preds, epi_unc = predict_from_checkpoints(
        data=dataloader,
        checkpoint_path=args.checkpoint_path,
        checkpoint_dir=args.checkpoint_dir,
        aggr=args.aggr,
        gpus=args.gpus,
    )

    df["pred"] = preds
    df["epi_unc"] = epi_unc

    if args.save_per_model:
        from gneprop import utils
        from gneprop_pyg import predict_ensemble

        checkpoint_paths = utils.get_checkpoint_paths(
            checkpoint_path=args.checkpoint_path,
            checkpoint_dir=args.checkpoint_dir,
        )
        per_model_preds = []
        for checkpoint_path in tqdm(checkpoint_paths, desc="Saving per-checkpoint predictions"):
            checkpoint_preds, _ = predict_ensemble(
                [checkpoint_path],
                dataloader,
                aggr="mean",
                gpus=args.gpus,
            )
            per_model_preds.append(checkpoint_preds)
        df = add_per_model_predictions(df, per_model_preds)

    if args.sort_desc:
        df = df.sort_values("pred", ascending=False).reset_index(drop=True)

    output_path = resolve_output_path(args.input_path, args.output_path, args.output_dir)
    df.to_csv(output_path, index=False)
    return output_path


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run virtual screening with trained GNEprop checkpoints.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to an input CSV/SMI/TSV/feather file.")
    parser.add_argument(
        "--input_format",
        type=str,
        choices=["csv", "smi", "feather", "ftr"],
        default=None,
        help="Optional explicit input format. If omitted, inferred from the file extension.",
    )
    parser.add_argument("--smiles_column", type=str, default="SMILES", help="Name of the SMILES column.")
    parser.add_argument("--id_column", type=str, default="ID", help="Name used for the ID column in SMI-like files.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a single checkpoint file.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing one or more checkpoints.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the output CSV file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. Defaults to ./predictions.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--aggr",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Aggregation function for multi-checkpoint ensembles.",
    )
    parser.add_argument("--save_per_model", action="store_true", help="Save one prediction column per checkpoint.")
    parser.add_argument("--sort_desc", action="store_true", help="Sort output rows by descending prediction score.")
    parser.add_argument("--no_legacy", action="store_true", help="Disable legacy featurization in MolDatasetOD.")
    return parser


def validate_args(args) -> None:
    if (args.checkpoint_path is None) == (args.checkpoint_dir is None):
        raise ValueError("Exactly one of --checkpoint_path or --checkpoint_dir must be provided.")

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    if args.checkpoint_path is not None and not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")

    if args.checkpoint_dir is not None and not os.path.isdir(args.checkpoint_dir):
        raise NotADirectoryError(f"Checkpoint directory not found: {args.checkpoint_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    output_path = run_screening(args)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
