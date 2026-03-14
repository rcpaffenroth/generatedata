import pandas as pd
import pathlib
from pathlib import Path
import generatedata
import json
import requests
import generatedata.config
import random
import numpy as np

DATA_URL = generatedata.config.DATA_URL


def data_names(local: bool = False) -> list[str]:
    """List the names of the datasets that are available to load.

    Returns:
        list: the names of the datasets
    """
    if local:
        # The directory in which the notebook is located.
        base_dir = pathlib.Path(generatedata.__path__[0])
        # The directory where the data is stored.
        data_dir = base_dir / "../data/processed"
        with open(data_dir / "info.json", "r") as f:
            data_info = json.load(f)
    else:
        # Read the info json file from the URL DATA_URL+'/info.json'
        response = requests.get(DATA_URL + "/info.json")
        data_info = response.json()
    return list(data_info.keys())


def get_random_data_name(local: bool = False) -> str:
    """Return a random dataset name from the available datasets.

    Args:
        local (bool): If True, list datasets from the local processed data directory.

    Returns:
        str: A randomly chosen dataset name.
    """
    names = data_names(local=local)
    if not names:
        raise ValueError("No dataset names available to choose from.")
    return random.choice(names)


def load_data(name: str, local: bool = False, data_dir: Path | str | None = None) -> dict:
    """Load in the dataset with the given name.  This functions loads in a variety of datasets created by the
    `scripts/generate-data.py` script.

    Args:
        name (str): the name of the dataset

    Returns:
        dict: the start and target points of the dataset
    """
    valid_names = data_names(local=local)
    if name not in valid_names:
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {valid_names}")

    if local:
        # If a data_dir is provided, use it, otherwise use the default data directory
        if data_dir is not None:
            data_dir = pathlib.Path(data_dir)
        else:
            # The directory in which the notebook is located.
            base_dir = pathlib.Path(generatedata.__path__[0])
            # The directory where the data is stored.
            data_dir = base_dir / "../data/processed"

        # load in the info for the datasets
        with open(data_dir / "info.json", "r") as f:
            data_info = json.load(f)[name]

        # Read the start data
        z_start = pd.read_parquet(data_dir / f"{name}_start.parquet")
        # Read the target data
        z_target = pd.read_parquet(data_dir / f"{name}_target.parquet")
    else:
        # Read in the info for the datasets
        response = requests.get(DATA_URL + "/info.json")
        data_info = response.json()[name]
        # Read the start data
        z_start = pd.read_parquet(DATA_URL + f"/{name}_start.parquet")
        # Read the target data
        z_target = pd.read_parquet(DATA_URL + f"/{name}_target.parquet")

    return {"info": data_info, "start": z_start, "target": z_target}


def load_data_as_xy(name: str, local: bool = False, data_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load in the dataset with the given name and return it as a tuple of (X, Y).
    Note, the dataset must define the info json with the keys 'x_y_index', 'x_size', and 'y_size'.

    Args:
        name (str): the name of the dataset

    Returns:
        tuple: (X, Y) where X and Y are pandas DataFrames
    """
    data = load_data(name, local=local, data_dir=data_dir)
    info = data["info"]
    if "x_y_index" not in info or "x_size" not in info or "y_size" not in info:
        raise ValueError(
            f"Dataset {name} does not have the required keys in info.json: 'x_y_index', 'x_size', 'y_size'."
        )
    return data["target"].iloc[:, : info["x_y_index"]], data["target"].iloc[
        :, info["x_y_index"] :
    ]


def load_data_as_xy_onehot(name: str, local: bool = False, data_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load in the dataset with the given name and return it as a tuple of (X, Y).
    Note, the dataset must define the info json with the keys 'x_y_index', 'x_size', 'y_size', and 'onehot_y'. And the 'onehot_y' must be set to True.

    Args:
        name (str): the name of the dataset

    Returns:
        tuple: (X, Y) where X and Y are pandas DataFrames
    """
    data = load_data(name, local=local, data_dir=data_dir)
    info = data["info"]
    # Check if the required keys are present in the info.json
    if "onehot_y" not in info:
        raise ValueError(
            f"Dataset {name} does not have the required key 'onehot_y' in info.json."
        )
    if info["onehot_y"] != 1:
        raise ValueError(
            f"Dataset {name} does not have 'onehot_y' set to True in info.json."
        )
    # Check if the other required keys are present
    # 'x_y_index', 'x_size', and 'y_size'
    if "x_y_index" not in info or "x_size" not in info or "y_size" not in info:
        raise ValueError(
            f"Dataset {name} does not have the required keys in info.json: 'x_y_index', 'x_size', 'y_size'."
        )
    return data["target"].iloc[:, : info["x_y_index"]], data["target"].iloc[
        :, info["x_y_index"] :
    ]


def load_data_as_sequence(
    name: str,
    local: bool = False,
    data_dir: Path | str | None = None,
    label_every_step: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a sequence dataset and reshape it into (num_points, seq_len, step_size).

    Args:
        name (str): the name of the dataset
        local (bool): if True, load from local data directory
        data_dir (Path | str | None): override the default data directory
        label_every_step (bool): if True, broadcast labels across all timesteps
            and concatenate with pixel sequence; if False, return pixels only

    Returns:
        tuple: (X_seq, labels) where X_seq is a numpy array of shape
            (num_points, seq_len, step_size + label_dim) when label_every_step=True,
            or (num_points, seq_len, step_size) when label_every_step=False,
            and labels has shape (num_points, label_dim).
    """
    data = load_data(name, local=local, data_dir=data_dir)
    info = data["info"]

    if "seq_len" not in info or "step_size" not in info:
        raise ValueError(
            f"Dataset '{name}' has no sequence metadata (seq_len / step_size). "
            f"Use load_data_as_xy() instead, or load a sequence variant such as '{name}_seq1'."
        )

    seq_len = info["seq_len"]
    step_size = info["step_size"]
    x_y_index = info["x_y_index"]

    target_df = data["target"]
    num_points = len(target_df)

    pixels = target_df.iloc[:, :x_y_index].to_numpy()   # (num_points, total_pixels)
    labels = target_df.iloc[:, x_y_index:].to_numpy()   # (num_points, label_dim)
    label_dim = labels.shape[1]

    X_seq = pixels.reshape(num_points, seq_len, step_size)

    if label_every_step:
        labels_broadcast = np.broadcast_to(
            labels[:, np.newaxis, :], (num_points, seq_len, label_dim)
        ).copy()
        X_seq = np.concatenate([X_seq, labels_broadcast], axis=2)

    return X_seq, labels
