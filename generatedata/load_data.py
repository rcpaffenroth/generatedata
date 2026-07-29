import pandas as pd
import pathlib
from pathlib import Path
import generatedata
import io
import json
import requests
import generatedata.config
import random
import warnings
import numpy as np

DATA_URL = generatedata.config.DATA_URL

# Which array holds the labels for each `part` of a dataset stored as .npy.
_NPY_LABEL_ARRAYS = {
    "target": "final_means",
    "start": "ut_fixed",
    "all_layers": "all_layer_means",
}


def _resolve_data_dir(data_dir: Path | str | None) -> Path:
    """The local processed-data directory, defaulting to the one in the package."""
    if data_dir is not None:
        return pathlib.Path(data_dir)
    base_dir = pathlib.Path(generatedata.__path__[0])
    return base_dir / "../data/processed"


def _dataset_info(name: str, local: bool, data_dir: Path | str | None) -> dict:
    """Metadata for one dataset, read without touching its (possibly huge) data."""
    if local:
        with open(_resolve_data_dir(data_dir) / "info.json", "r") as f:
            all_info = json.load(f)
    else:
        all_info = requests.get(DATA_URL + "/info.json").json()
    if name not in all_info:
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {list(all_info.keys())}"
        )
    return all_info[name]


def _load_npy_array(
    name: str, array: str, local: bool, data_dir: Path | str | None
) -> np.ndarray:
    """Load one ``.npy`` array of a dataset stored that way.

    Local files are memory-mapped, so a dataset of hundreds of megabytes costs no
    RAM until rows are actually touched.  Call ``np.asarray`` on the result if you
    want it resident.
    """
    filename = f"{name}_{array}.npy"
    if local:
        return np.load(_resolve_data_dir(data_dir) / filename, mmap_mode="r")
    response = requests.get(f"{DATA_URL}/{filename}")
    response.raise_for_status()
    return np.load(io.BytesIO(response.content))


def data_names(local: bool = False, data_dir: Path | str | None = None) -> list[str]:
    """List the names of the datasets that are available to load.

    Args:
        local: If True, list datasets from a local processed data directory.
        data_dir: Override the default data directory.  Only meaningful with
            ``local``; the remote listing has one location.

    Returns:
        list: the names of the datasets
    """
    if local:
        with open(_resolve_data_dir(data_dir) / "info.json", "r") as f:
            data_info = json.load(f)
    else:
        # Read the info json file from the URL DATA_URL+'/info.json'
        response = requests.get(DATA_URL + "/info.json")
        data_info = response.json()
    return list(data_info.keys())


def dataset_info(name: str, local: bool = False, data_dir: Path | str | None = None) -> dict:
    """Return one dataset's metadata without loading its data.

    ``load_data`` also returns the metadata, but only for datasets that have a
    DataFrame form; this works for every dataset, and reads nothing but the
    ``info.json`` index — worth having when the data itself is a memory-mapped
    array of hundreds of megabytes.

    Args:
        name: the name of the dataset
        local: If True, read the local processed data directory.
        data_dir: Override the default data directory.

    Returns:
        dict: the dataset's entry from ``info.json``
    """
    return _dataset_info(name, local=local, data_dir=data_dir)


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
        local (bool): If True, read from a local processed data directory.
        data_dir: Override the default data directory.

    Returns:
        dict: the start and target points of the dataset
    """
    # No separate name check here.  There used to be a `data_names(local=local)`
    # membership test, which ignored `data_dir` and so read the *package-relative*
    # directory: with `data_dir` pointing anywhere else it raised FileNotFoundError
    # (or "Unknown dataset") before `data_dir` was ever consulted.  `_dataset_info`
    # below does honour `data_dir` and raises the identical ValueError for a name
    # that is not there, so the check was redundant as well as wrong -- and in the
    # remote case it cost a second fetch of info.json on every load.
    data_info = _dataset_info(name, local=local, data_dir=data_dir)

    # A few datasets are not a flat start/target parquet pair and so have no
    # DataFrame form: one row of the whest family holds an entire weight tensor.
    if data_info.get("storage") == "npy":
        raise ValueError(
            f"Dataset '{name}' (data_family={data_info.get('data_family')}) is stored as "
            f".npy arrays rather than a flat start/target parquet pair, so it has no "
            f"DataFrame representation. Load it with "
            f"load_data_as_sequence('{name}', local=True) instead."
        )

    if local:
        data_dir = _resolve_data_dir(data_dir)
        # Read the start data
        z_start = pd.read_parquet(data_dir / f"{name}_start.parquet")
        # Read the target data
        z_target = pd.read_parquet(data_dir / f"{name}_target.parquet")
    else:
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
    if info.get("is_sequence"):
        warnings.warn(
            f"Dataset '{name}' is a sequence dataset (data_family={info.get('data_family', 'unknown')}). "
            f"Loading as flat X/Y returns padded fixed-size data. "
            f"Consider using load_data_as_sequence() for dynamic-length sequence handling.",
            UserWarning,
            stacklevel=2,
        )
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
    if info.get("is_sequence"):
        warnings.warn(
            f"Dataset '{name}' is a sequence dataset (data_family={info.get('data_family', 'unknown')}). "
            f"Loading as flat X/Y returns padded fixed-size data. "
            f"Consider using load_data_as_sequence() for dynamic-length sequence handling.",
            UserWarning,
            stacklevel=2,
        )
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
    step_size: int | None = None,
    local: bool = False,
    data_dir: Path | str | None = None,
    label_every_step: bool | None = None,
    part: str = "target",
) -> tuple[np.ndarray, np.ndarray]:
    """Load any dataset with x_y_index and reshape it into a sequence.

    The sequence length is computed as x_y_index // step_size.  This allows
    any flat dataset to be treated as a time-series without storing sequence
    metadata in info.json.

    Args:
        name: Dataset name.
        step_size: Number of feature values per timestep.  When ``None``,
            the value is read from ``default_step_size`` in the dataset's
            metadata (set automatically for LRA and whest datasets).
        local: If True, load from local processed data directory.
        data_dir: Override the default data directory.
        label_every_step: If True, broadcast labels across all timesteps and
            concatenate with the feature sequence; if False, return features only.
            Defaults to the dataset's ``label_every_step_allowed`` metadata, which
            is True everywhere except where broadcasting the labels into the
            features would leak the answer (the whest family, whose per-layer
            means are what an estimator must predict, never something it is given).
        part: Which label block to return — ``"target"`` for the ground truth,
            ``"start"`` for the trajectory's starting point (for whest, the cheap
            ``UT_fixed(W)`` estimate, so ``target - start`` is the residual a
            corrector must learn), or ``"all_layers"`` for the per-layer mean
            stack, which only the whest family provides.

    Returns:
        (X_seq, labels) where X_seq has shape
        (num_points, seq_len, step_size [+ label_dim]) and
        labels has shape (num_points, label_dim) — or
        (num_points, depth, width) when ``part="all_layers"``.

    Raises:
        ValueError: If x_y_index is missing, step_size cannot be resolved,
            x_y_index is not divisible by step_size, ``part`` is unknown or
            unavailable, or ``label_every_step`` is requested where it would leak.
    """
    if part not in _NPY_LABEL_ARRAYS:
        raise ValueError(
            f"Unknown part '{part}'. Choose one of {sorted(_NPY_LABEL_ARRAYS)}."
        )

    info = _dataset_info(name, local=local, data_dir=data_dir)
    stored_as_npy = info.get("storage") == "npy"

    # Broadcasting labels into the feature axis is the library's denoising framing:
    # at inference you feed the `start` version, whose labels carry no answer.  The
    # whest family has no such counterpart on this axis, so allowing it there would
    # hand a model the very quantity it is supposed to predict.
    allowed = info.get("label_every_step_allowed", True)
    if label_every_step is None:
        label_every_step = allowed
    elif label_every_step and not allowed:
        raise ValueError(
            f"label_every_step=True is refused for dataset '{name}' "
            f"(data_family={info.get('data_family')}): it would concatenate the labels "
            f"onto every timestep of the features, leaking the target. Pass "
            f"label_every_step=False and use the labels returned separately."
        )

    if stored_as_npy:
        features = _load_npy_array(name, "weights", local, data_dir)
        features = features.reshape(features.shape[0], -1)
        labels = _load_npy_array(name, _NPY_LABEL_ARRAYS[part], local, data_dir)
    else:
        if part == "all_layers":
            raise ValueError(
                f"Dataset '{name}' has no per-layer label stack; part='all_layers' is "
                f"only available for datasets that store one (the whest family)."
            )
        data = load_data(name, local=local, data_dir=data_dir)

    if step_size is None:
        step_size = info.get("default_step_size")
        if step_size is None:
            raise ValueError(
                f"No step_size provided and dataset '{name}' has no "
                f"default_step_size in metadata. Please pass step_size explicitly."
            )

    if "x_y_index" not in info:
        raise ValueError(
            f"Dataset '{name}' has no x_y_index metadata. Cannot reshape as sequence."
        )

    x_y_index = info["x_y_index"]

    if x_y_index % step_size != 0:
        raise ValueError(
            f"x_y_index ({x_y_index}) is not evenly divisible by step_size ({step_size})."
        )

    seq_len = x_y_index // step_size

    if not stored_as_npy:
        block = data[part]
        features = block.iloc[:, :x_y_index].to_numpy()   # (num_points, x_y_index)
        labels = block.iloc[:, x_y_index:].to_numpy()     # (num_points, label_dim)

    num_points = features.shape[0]
    label_dim = labels.shape[-1]

    # A reshape of a memory-mapped array is a view, so an .npy-backed dataset stays
    # unread on disk until the caller indexes into it.
    X_seq = features.reshape(num_points, seq_len, step_size)

    if label_every_step:
        labels_broadcast = np.broadcast_to(
            labels[:, np.newaxis, :], (num_points, seq_len, label_dim)
        ).copy()
        X_seq = np.concatenate([X_seq, labels_broadcast], axis=2)

    return X_seq, labels
