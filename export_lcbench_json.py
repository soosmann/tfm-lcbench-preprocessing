#!/usr/bin/env python3
"""Export LCBench configurations to one JSON file per run."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import tqdm

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lcbench_matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "lcbench_cache")
)

from api import Benchmark

DEFAULT_DATA_PATH = Path("data/bench_full.json")
DEFAULT_OUTPUT_DIR = Path("data/exported")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LCBench data through api.py as per-configuration JSON files."
    )
    parser.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_PATH), help="Path to bench_full.json."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for exported JSON files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset names to export. Defaults to all datasets.",
    )
    parser.add_argument(
        "--config-ids",
        nargs="+",
        help="Configuration ids to export. Defaults to all configs.",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        choices=[6, 12, 25, 50],
        help="Epoch budget keys to export, e.g. 6 12 25 50. Defaults to all available budgets.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        help="Run/seed ids to export. Defaults to all available runs.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing JSON files."
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="JSON indentation. Use 0 for compact JSON.",
    )
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    """Convert API values to plain JSON-friendly Python values."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    There are some values that are bool encoded (instead `1` written es `True`).
    This function ensures that they are represented using a number and not bool.

    Parameters:
        config (`dict[str, Any]`): The config that should be checked.

    Returns:
        dict[str, Any]: The normalized dict.
    """
    normalized = json_ready(config)

    # Some one-layer configs are encoded as JSON booleans in the source data.
    for key in ("batch_size", "cosine_annealing_T_max", "max_units", "num_layers"):
        if key in normalized and isinstance(normalized[key], (bool, int)):
            normalized[key] = int(normalized[key])

    return normalized


def get_available_budgets(
    bench: Benchmark, dataset_name: str, config_id: int | str
) -> list[str]:
    """
    Provides the possible amounts of epochs (budget) that can be retrieved.

    Parameters:
        dataset_name (`str`): Name of the dataset for building the json path.
        config_id (`int | str`): Config id for building the json path.

    Returns:
        list[str]: Provides all possible budget options as String.
    """
    return sorted(
        bench.data[dataset_name][str(config_id)].keys(),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )


def get_available_runs(
    bench: Benchmark, dataset_name: str, config_id: int | str, budget: int | str
) -> list[str]:
    """
    Provides the possible amounts of epochs (budget) that can be retrieved.

    Parameters:
        dataset_name (`str`): Name of the dataset for building the json path.
        config_id (`int | str`): Config id for building the json path.
        budget (`int | str`): The budget for building the json path.

    Returns:
        list[str]: Provides all possible budget options as String.
    """
    return sorted(
        bench.data[dataset_name][str(config_id)][str(budget)]["results"].keys(),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )


def get_flops_per_sample(
    num_layers: int, max_units: int, input_dim: int, num_classes: int
) -> int:
    """
    Compute FLOPs for a single forward pass through a shaped (funnel) MLP,
    as described in the Auto-PyTorch / LCBench paper.

    Each layer width follows:
        n_i = n_{i-1} - (n_max - n_out) / (n_layers - 1)
    with the first hidden layer at max_units and the output at num_classes.

    FLOPs are counted as 2 * in_features * out_features per linear layer
    (one multiply + one add per weight).

    Parameters:
        num_layers (`int`): Number of hidden layers (1-5 in LCBench Search Space 1)
        max_units (`int`): Maximum number of units in the first hidden layer (64-512)
        input_dim (`int`): Number of input features (after preprocessing)
        num_classes (`int`): Number of output classes

    Returns:
        int: Total FLOPs per sample
    """
    # Build hidden layer widths using the funnel formula (Eq. 1 in the paper)
    if num_layers == 1:
        hidden_widths = [max_units]
    else:
        step = (max_units - num_classes) / (num_layers - 1)
        hidden_widths = [round(max_units - i * step) for i in range(num_layers)]

    # Full sequence: input -> hidden layers -> output
    layer_dims = [input_dim] + hidden_widths + [num_classes]

    # FLOPs = 2 * in * out for each Linear layer (multiply-accumulate)
    flops = sum(
        2 * layer_dims[i] * layer_dims[i + 1] for i in range(len(layer_dims) - 1)
    )
    return flops


def metric_point(
    values: list[float], *, mode: str, final: bool = False
) -> dict[str, Any]:
    """
    Helper function for selecting the correct value for metric values like best loss or final accuracy.

    Parameters:
        values (`list[float]`): The list where a special value should be extracted.
        mode (`str`): Which metric mode is wanted? Can be either `min` or `max`.
        final (`bool`): Is the final value wanted?

    Returns:
        dict[str, Any]: Returns a dict with the value and the epoch where value was found.
    """

    if not values:
        return {"value": None, "epoch": None}

    index = len(values) - 1
    if not final:
        if mode == "min":
            index = min(range(len(values)), key=values.__getitem__)
        elif mode == "max":
            index = max(range(len(values)), key=values.__getitem__)
        else:
            raise ValueError(f"Unsupported metric mode: {mode}")

    return {"value": values[index], "epoch": index}


def build_summary(
    train_loss: list[float],
    train_acc: list[float],
    eval_loss: list[float],
    eval_acc: list[float],
) -> dict[str, Any]:
    """
    Function for building the training summaries for train and test with best and final losses.
    """
    return {
        "train": {
            "best_loss": metric_point(train_loss, mode="min"),
            "final_loss": metric_point(train_loss, mode="min", final=True),
            "best_accuracy": metric_point(train_acc, mode="max"),
            "final_accuracy": metric_point(train_acc, mode="max", final=True),
        },
        "eval": {
            "best_loss": metric_point(eval_loss, mode="min"),
            "final_loss": metric_point(eval_loss, mode="min", final=True),
            "best_accuracy": metric_point(eval_acc, mode="max"),
            "final_accuracy": metric_point(eval_acc, mode="max", final=True),
        },
    }


def query(
    bench: Benchmark,
    dataset: str,
    tag: str,
    config_id: int | str,
    budget: int | str,
    run: int | str,
) -> Any:
    """
    Function to perform the actual query in the LCBENCH set.

    Parameters:
        dataset (`str`): Name of the dataset (top-level key)
        tag (`str`): Metric / field to retrieve
        config_id (`int | str`): The config index (second-level key)
        budget (`int | str`): Budget / epoch key, either `5, `12`, `25` or `50`
        run_nr (`int | str`): Seed / repetition key, either `1`, `2` or `3`

    Returns:
        Any: The queried data.
    """
    return json_ready(bench.query(dataset, tag, config_id, epochs=budget, run_nr=run))


def build_record(
    bench: Benchmark,
    dataset: str,
    config_id: int | str,
    budget: int | str,
    run: int | str,
) -> dict[str, Any]:
    """
    The creation of the json record. Includes queries and additional data generation.

    Parameters:
        bench (`Benchmark`): The `Benchmark` object used for queries.
        dataset (`str`): Name of the dataset (top-level key)
        config_id (`int | str`): The config index (second-level key)
        budget (`int | str`): Budget / epoch key, either `5, `12`, `25` or `50`
        run_nr (`int | str`): Seed / repetition key, either `1`, `2` or `3`

    Returns:
        dict: The dict containing the final json structure for saving the sample.
    """
    # data retrieval
    config = normalize_config(query(bench, dataset, "config", config_id, budget, run))

    num_features = int(query(bench, dataset, "features", config_id, budget, run))
    num_classes = int(query(bench, dataset, "classes", config_id, budget, run))
    num_samples = int(query(bench, dataset, "instances", config_id, budget, run))
    openml_task_id = int(
        query(bench, dataset, "OpenML_task_id", config_id, budget, run)
    )
    param_count = int(query(bench, dataset, "model_parameters", config_id, budget, run))

    epoch = query(bench, dataset, "epoch", config_id, budget, run)
    train_loss = query(
        bench, dataset, "Train/train_cross_entropy", config_id, budget, run
    )
    train_acc = query(bench, dataset, "Train/train_accuracy", config_id, budget, run)
    eval_loss = query(bench, dataset, "Train/val_cross_entropy", config_id, budget, run)
    eval_acc = query(bench, dataset, "Train/val_accuracy", config_id, budget, run)
    time_cumulative = query(bench, dataset, "time", config_id, budget, run)

    # generation of additionally necessary data
    time_per_epoch = []
    previous = 0.0
    for current in time_cumulative:
        time_per_epoch.append(current - previous)
        previous = current

    flops = get_flops_per_sample(
        num_layers=int(config["num_layers"]),
        max_units=int(config["max_units"]),
        input_dim=num_features,
        num_classes=num_classes,
    )

    # final json structure creation
    return {
        "search_space": "lcbench",
        "dataset": dataset,
        "dataset_metadata": {
            "num_input_features": num_features,
            "num_classes": num_classes,
            "num_samples": num_samples,
            "openml_task_id": openml_task_id,
        },
        "eval_split": query(bench, dataset, "test_split", config_id, budget, run),
        "config_id": int(config_id) if str(config_id).isdigit() else config_id,
        "seed": int(query(bench, dataset, "seed", config_id, budget, run)),
        "hp": int(query(bench, dataset, "budget", config_id, budget, run)),
        "epochs": len(epoch),
        "architecture": {
            "config": config,
            "param_count": param_count,
            "flops": flops,
        },
        "summary": build_summary(train_loss, train_acc, eval_loss, eval_acc),
        "curves": {
            "epoch": epoch,
            "train": {
                "loss": train_loss,
                "accuracy": train_acc,
                "time_cumulative": time_cumulative,
                "time_per_epoch": time_per_epoch,
            },
            "eval": {
                "loss": eval_loss,
                "accuracy": eval_acc,
            },
        },
    }


def selected_or_all(selected: list[str] | None, available: Iterable[str]) -> list[str]:
    """
    Function to retrieve a list of all values possible to retrieve or the selected ones.
    Is used to generate a list at a certain level in the json hierarchie to create a loop to iterate through all configs.

    Parameters:
        selected (`list[str] | None`): An optional list of selected values for retrieval.
        available (`Iterable[str]`): All available values that can be retrieved.

    Returns:
        list[str]: List with all values that will deal as loop.
    """
    return (
        [str(item) for item in selected]
        if selected
        else [str(item) for item in available]
    )


def export_record(
    record: dict[str, Any], output_dir: Path, overwrite: bool, indent: int
) -> tuple[Path, bool]:
    """
    File creation for a learning curve sample:

    Parameters:
        record (`tuple[Path, bool]`): The dict structure to be saved as json.
        output_dir (`Path`): In which directory should the record be saved?
        overwrite (`bool`): Should already existing files for a sample be overwritten?
        indent (`int`): How many spaces should be used as indent when creating the json?

    Returns:
        tuple[Path, bool]: The full output path where the record was saved and a bool indicating if file was written or not.
    """
    # usage of "__" to separate dataset name from rest (some dataset names contain "_")
    output_path = output_dir / (
        f"{record["dataset"]}__config_{record['config_id']}_budget_{record['hp']}_seed_{record['seed']}.json"
    )
    if output_path.exists() and not overwrite:
        return output_path, False

    if not output_dir.exists():
        os.mkdir(output_dir)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=indent if indent > 0 else None)
        handle.write("\n")

    return output_path, True


def main() -> None:
    args = parse_args()
    bench = Benchmark(args.data_dir, cache=False)
    output_dir = Path(args.output_dir)
    exported = 0
    skipped = 0

    # first stage: dataset retrieval
    datasets = selected_or_all(args.datasets, bench.get_dataset_names())
    for dataset in tqdm.tqdm(datasets):
        # second stage: config id retrieval
        config_ids = selected_or_all(args.config_ids, bench.get_config_ids(dataset))
        for config_id in config_ids:
            # third stage: selection of budget (amount of epochs)
            available_budgets = get_available_budgets(bench, dataset, config_id)
            budgets = selected_or_all(args.budgets, available_budgets)
            for budget in budgets:
                if budget in available_budgets:
                    # fourth stage: select the runs that should be saved
                    runs = selected_or_all(
                        args.runs, get_available_runs(bench, dataset, config_id, budget)
                    )
                    # fifth stage: export each run needed
                    for run in runs:
                        record = build_record(bench, dataset, config_id, budget, run)
                        _, wrote = export_record(
                            record, output_dir, args.overwrite, args.indent
                        )
                        if wrote:
                            exported += 1
                        else:
                            skipped += 1

    print(
        f"Export complete: {exported} written, {skipped} existing files left unchanged."
    )
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
