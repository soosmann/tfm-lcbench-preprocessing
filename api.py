import gzip
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


class Benchmark:
    """API for TabularBench / LCBench."""

    def __init__(self, data_dir, cache=False, cache_dir="cached/"):
        if not os.path.isfile(data_dir) or not data_dir.endswith(".json"):
            raise ValueError("Please specify path to the bench json file.")

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.cache = cache

        print("==> Loading data...")
        self.data = self._read_data(data_dir)
        self.dataset_names = list(self.data.keys())
        print("==> Done.")

    ##### QUERY API #####

    def query(self, dataset_name, tag, config_id, epochs="25", run_nr="1"):
        """
        Query a single `tag` for one run. Tag refers to a part of the config or a metric.

        Data Layout
            data[dataset_name][config_id][epochs]
                ├── config          # flat dict, shared across runs
                ├── results[run_nr] # scalar metrics at end of training
                └── log[run_nr]     # per-epoch lists

        Parameters:
            dataset_name (`str`): Name of the dataset (top-level key)
            tag (`str`): Metric / field to retrieve
            config_id (`int | str`): The config index (second-level key)
            epochs (`int | str`): Budget / epoch key, either `5, `12`, `25` or `50`
            run_nr (`int | str`): Seed / repetition key, either `1`, `2` or `3`

        Returns:
            Any: The `tag` values. Can be any kind of data.
        """
        config_id = str(config_id)
        epochs = str(epochs)
        run_nr = str(run_nr)

        if dataset_name not in self.data:
            raise ValueError(f"Dataset '{dataset_name}' not found.")

        if config_id not in self.data[dataset_name]:
            raise ValueError(
                f"Config '{config_id}' not found for dataset '{dataset_name}'."
            )

        entry = self.data[dataset_name][config_id]

        if epochs not in entry:
            available = list(entry.keys())
            raise ValueError(
                f"Epoch key '{epochs}' not found for config '{config_id}'. "
                f"Available: {available}"
            )

        block = entry[epochs]

        # config is a flat dict — no run_nr dimension
        if tag == "config":
            return block["config"]
        if tag in block["config"]:
            return block["config"][tag]

        # results and log are keyed by run_nr
        if tag in block["results"].get(run_nr, {}):
            return block["results"][run_nr][tag]
        if tag in block["log"].get(run_nr, {}):
            return block["log"][run_nr][tag]

        raise ValueError(
            f"Tag '{tag}' not found for config '{config_id}', "
            f"epochs '{epochs}', run '{run_nr}' in dataset '{dataset_name}'."
        )

    def query_best(
        self, dataset_name, tag, criterion, epochs="25", run_nr="1", position=0
    ):
        """
        Return the `tag` value for the n-th best config (ranked by max value of `criterion`).

        Parameters:
            dataset_name (`str`): Name of the dataset (top-level key)
            tag (`str`): Metric / field to retrieve
            criterion (`str`): The metric used for comparison of the runs.
            epochs (`int | str`): Budget / epoch key, either `5, `12`, `25` or `50`
            run_nr (`int | str`): Seed / repetition key, either `1`, `2` or `3`
            position (`int`): The `position`-best value will be returned, e.g. 0 = best, 1 = second-best, etc.

        Returns:
            Any: The `tag` values. Can be any kind of data.
        """
        performances = []
        for config_id in self.data[dataset_name]:
            try:
                values = self.query(
                    dataset_name, criterion, config_id, epochs=epochs, run_nr=run_nr
                )
                # values can be a list (log) or a scalar (results)
                best_val = max(values) if isinstance(values, list) else values
                performances.append((config_id, best_val))
            except (ValueError, TypeError):
                pass  # skip configs that don't have this tag

        if not performances:
            raise ValueError(
                f"No configs found with criterion '{criterion}' "
                f"for dataset '{dataset_name}'."
            )

        performances.sort(key=lambda x: x[1], reverse=True)
        best_config_id = performances[position][0]
        return self.query(
            dataset_name, tag, best_config_id, epochs=epochs, run_nr=run_nr
        )

    ##### METADATA HELPERS #####

    def get_queriable_tags(
        self, dataset_name=None, config_id=None, epochs="25", run_nr="1"
    ):
        """
        Return all tags that can be queried. For completion, the path in the dataset json can be defined by the function params.
        Still, this should not affect the queriable tags. They should be always the same.

        Parameters:
            dataset_name (`str`): Name of the dataset in the path.
            config_id (`int | str`): Config id for the path.
            epochs (`int | str`): A run's epoch count for the path, either `5, `12`, `25` or `50`.
            run_id (`int | str`): Seed / repetition of the run, either `1`, `2` or `3`.

        Returns:
            list[str]: All queriable tags for the given path. Should usually be the same for all paths.
        """
        if dataset_name is None:
            dataset_name = self.dataset_names[0]
        if config_id is None:
            config_id = list(self.data[dataset_name].keys())[0]

        block = self.data[dataset_name][str(config_id)][str(epochs)]
        run_nr = str(run_nr)

        log_tags = list(block["log"].get(run_nr, {}).keys())
        result_tags = list(block["results"].get(run_nr, {}).keys())
        config_tags = list(block["config"].keys())
        return log_tags + result_tags + config_tags + ["config"]

    def get_dataset_names(self) -> list[str]:
        """Get all available dataset names."""
        return self.dataset_names

    def get_openml_task_ids(self) -> list[str]:
        """Returns the openml task ids that were used for data generation."""
        task_ids = []
        for dataset_name in self.dataset_names:
            config_id = list(self.data[dataset_name].keys())[0]
            epochs = list(self.data[dataset_name][config_id].keys())[0]
            run_nr = list(self.data[dataset_name][config_id][epochs]["results"].keys())[
                0
            ]
            task_ids.append(
                self.query(
                    dataset_name,
                    "OpenML_task_id",
                    config_id,
                    epochs=epochs,
                    run_nr=run_nr,
                )
            )
        return task_ids

    def get_number_of_configs(self, dataset_name: str) -> int:
        """
        Based on the selected dataset, provide the amount of configs that can be retrieved.

        Parameters:
            dataset_name (`str`): Name of the dataset to retrieve the amount of config ids for.

        Return:
            int: The list of config ids that can be retrieved.
        """
        if dataset_name not in self.dataset_names:
            raise ValueError("Dataset name not found.")

        return len(self.data[dataset_name])

    def get_config(self, dataset_name: str, config_id: int | str, epochs: int | str):
        """
        Get one training config based on dataset, config id and epoch count.

        Parameters:
            dataset_name (`str`): Name of the dataset to retrieve amount of config ids for.
            config_id (`int | str`): The run's config id.
            epochs (`int | str`): The run's epoch count, either `5, `12`, `25` or `50`.

        Return:
            dict: The dict containing the config settings.
        """
        if dataset_name not in self.dataset_names:
            raise ValueError("Dataset name not found.")
        return self.data[dataset_name][str(config_id)][str(epochs)]["config"]

    def get_config_ids(self, dataset_name: str) -> list[str]:
        """
        Based on the selected dataset, provide all available configs.

        Parameters:
            dataset_name (`str`): Name of the dataset to retrieve the config ids for.

        Return:
            list[str]: The list of config ids that can be retrieved.
        """
        if dataset_name not in self.dataset_names:
            raise ValueError(f"Dataset '{dataset_name}' not found.")
        return list(self.data[dataset_name].keys())

    ##### PLOTTING #####

    def plot_by_name(
        self,
        dataset_names: list[str],
        x_col: str,
        y_col: str,
        n_configs=10,
        show_best=False,
        xscale="linear",
        yscale="linear",
        epochs="25",
        run_nr="1",
        criterion=None,
    ):
        """
        Plot learning curves for one or more datasets.

        Parameters:
            dataset_names (`str`): Name of the dataset (top-level key)
            x_col (`str`): The value that will be plotted to the x axis.
            y_col (`str`): The value that will be plotted to the y axis.
            n_configs (`int`): The amount of configs considered.
            show_best (`bool`): Refer only to best runs.
            xscale (`str`): The value scale for the x axis, e.g. `log` or `linear`.
            yscale (`str`): The value scale for the x axis, e.g. `log` or `linear`.
            epochs (`int | str`): Budget / epoch key, either `5, `12`, `25` or `50`
            run_nr (`int | str`): Seed / repetition key, either `1`, `2` or `3`
            criterion (`str`): The metric used to rank runs if `show_best=True`, if not set, is set to `y_col`.

        Returns:
            Figure: The plot.
        """
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        if not isinstance(dataset_names, (list, np.ndarray)):
            raise ValueError("Pass a dataset name or list of names.")

        if criterion is None:
            criterion = y_col

        n_rows = len(dataset_names)
        fig, axes = plt.subplots(
            n_rows, 1, sharex=False, sharey=False, figsize=(10, 7 * n_rows)
        )
        loop = enumerate(axes.flatten()) if n_rows > 1 else [(0, axes)]

        for ind_ax, ax in loop:
            ds = dataset_names[ind_ax]
            # grab metadata from first available config
            first_cfg = list(self.data[ds].keys())[0]
            first_ep = list(self.data[ds][first_cfg].keys())[0]
            first_run = list(self.data[ds][first_cfg][first_ep]["results"].keys())[0]
            instances = int(
                self.query(
                    ds, "instances", first_cfg, epochs=first_ep, run_nr=first_run
                )
            )
            classes = int(
                self.query(ds, "classes", first_cfg, epochs=first_ep, run_nr=first_run)
            )
            features = int(
                self.query(ds, "features", first_cfg, epochs=first_ep, run_nr=first_run)
            )

            for ind in range(n_configs):
                try:
                    if show_best:
                        x = self.query_best(
                            ds,
                            x_col,
                            criterion,
                            epochs=epochs,
                            run_nr=run_nr,
                            position=ind,
                        )
                        y = self.query_best(
                            ds,
                            y_col,
                            criterion,
                            epochs=epochs,
                            run_nr=run_nr,
                            position=ind,
                        )
                    else:
                        x = self.query(ds, x_col, ind + 1, epochs=epochs, run_nr=run_nr)
                        y = self.query(ds, y_col, ind + 1, epochs=epochs, run_nr=run_nr)

                    ax.plot(x, y, "p-")
                except ValueError as e:
                    print(f"Run {ind} not found for dataset '{ds}': {e}")

            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set(xlabel=x_col, ylabel=y_col)
            ax.set_title(
                f"{ds} — features: {features}, classes: {classes}, "
                f"instances: {instances}"
            )

        plt.tight_layout()
        plt.close(fig)
        return fig

    ##### CACHING AND RETRIEVAL #####

    def _cache_data(self, data, cache_file):
        """
        Save the data as pickle file to disk.

        Parameters:
            data (`Any`): The data to be saved in cache as pickle.
            cache_file (`str`): Path where the pickle file should be saved.

        Returns:
            None
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        with gzip.open(cache_file, "wb") as f:
            pickle.dump(data, f)

    def _read_cached_data(self, cache_file):
        """
        Read a cached pickle file.

        Parameters:
            cache_file (`str`): Path to a pickle file to load.

        Returns:
            Any: Data in pickle file.
        """
        with gzip.open(cache_file, "rb") as f:
            return pickle.load(f)

    def _read_file_string(self, path):
        """
        Read a file string in 64MB chunks.

        Parameters:
            path (`str`): Path to a pickle file to load.

        Returns:
            str: The filestring.
        """
        file_str = ""
        with open(path, "r") as f:
            while True:
                block = f.read(64 * (1 << 20))
                if not block:
                    break
                file_str += block
        return file_str

    def _read_data(self, path):
        """
        Main logic for reading the dataset file. If cached version exists, read it, if not, ujse the json one.

        Parameters:
            path (`str`): Path to a pickle file to load.

        Returns:
            Any: Retrieved data.
        """
        # the cached file
        cache_file = os.path.join(
            self.cache_dir, os.path.basename(self.data_dir).replace(".json", ".pkl.gz")
        )
        # if the cached version exists, return cached version
        if os.path.exists(cache_file) and self.cache:
            print("==> Found cached data, loading...")
            return self._read_cached_data(cache_file)

        # if not, read json version
        print("==> No cached data found or cache set to False.")
        print("==> Reading json data...")
        data = json.loads(self._read_file_string(path))
        if self.cache:
            print("==> Caching data...")
            self._cache_data(data, cache_file)
        return data
