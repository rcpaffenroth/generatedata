"""
Focused tests for load_data_as_sequence() in the v2 simplified API.

The v2 design:
  - step_size is a runtime parameter, not stored in metadata.
  - seq_len is computed as x_y_index // step_size.
  - Any dataset with x_y_index can be loaded as a sequence — no _seq variants needed.
"""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from generatedata.save_data import save_data
from generatedata.data_generators import compile_info_json
from generatedata.load_data import load_data_as_sequence


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_dataset(
    data_dir: Path,
    name: str,
    num_points: int,
    num_features: int,
    label_dim: int,
    x_y_index: int | None = None,
) -> None:
    """Write a synthetic flat dataset using save_data()."""
    rng = np.random.default_rng(42)
    total_cols = num_features + label_dim
    cols = {f"x{i}": rng.random(num_points) for i in range(total_cols)}
    save_data(
        data_dir, name, cols, cols,
        x_y_index=x_y_index,
        onehot_y=bool(x_y_index),
    )
    compile_info_json(data_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadDataAsSequence:
    NUM_POINTS = 100
    NUM_FEATURES = 784
    LABEL_DIM = 10

    @pytest.fixture()
    def mnist_dir(self, tmp_path):
        """Synthetic MNIST-like dataset: 784 features + 10 labels."""
        _make_dataset(
            tmp_path, "MNIST",
            num_points=self.NUM_POINTS,
            num_features=self.NUM_FEATURES,
            label_dim=self.LABEL_DIM,
            x_y_index=self.NUM_FEATURES,
        )
        return tmp_path

    @pytest.fixture()
    def mnist1d_dir(self, tmp_path):
        """Synthetic MNIST1D-like dataset: 40 features + 10 labels."""
        _make_dataset(
            tmp_path, "MNIST1D",
            num_points=self.NUM_POINTS,
            num_features=40,
            label_dim=self.LABEL_DIM,
            x_y_index=40,
        )
        return tmp_path

    @pytest.fixture()
    def no_xy_index_dir(self, tmp_path):
        """Synthetic geometric dataset without x_y_index."""
        rng = np.random.default_rng(0)
        cols = {f"x{i}": rng.random(self.NUM_POINTS) for i in range(2)}
        save_data(tmp_path, "circle", cols, cols)
        compile_info_json(tmp_path)
        return tmp_path

    def _load(self, name, data_dir, **kwargs):
        """Call load_data_as_sequence with data_names mocked to include name."""
        with patch("generatedata.load_data.data_names", return_value=[name]):
            return load_data_as_sequence(name, local=True, data_dir=data_dir, **kwargs)

    # ── shape tests ──────────────────────────────────────────────────────────

    def test_shape_step_size_1(self, mnist_dir):
        X, labels = self._load("MNIST", mnist_dir, step_size=1, label_every_step=True)
        assert X.shape == (self.NUM_POINTS, 784, 11)  # 1 pixel + 10 labels
        assert labels.shape == (self.NUM_POINTS, 10)

    def test_shape_step_size_28(self, mnist_dir):
        X, labels = self._load("MNIST", mnist_dir, step_size=28, label_every_step=True)
        assert X.shape == (self.NUM_POINTS, 28, 38)   # 28 pixels + 10 labels
        assert labels.shape == (self.NUM_POINTS, 10)

    def test_shape_step_size_784(self, mnist_dir):
        X, labels = self._load("MNIST", mnist_dir, step_size=784, label_every_step=True)
        assert X.shape == (self.NUM_POINTS, 1, 794)   # 784 pixels + 10 labels
        assert labels.shape == (self.NUM_POINTS, 10)

    def test_shape_label_every_step_false(self, mnist_dir):
        X, labels = self._load("MNIST", mnist_dir, step_size=1, label_every_step=False)
        assert X.shape == (self.NUM_POINTS, 784, 1)   # pixels only
        assert labels.shape == (self.NUM_POINTS, 10)

    # ── type & consistency tests ──────────────────────────────────────────────

    def test_returns_numpy_arrays(self, mnist_dir):
        X, labels = self._load("MNIST", mnist_dir, step_size=1)
        assert isinstance(X, np.ndarray)
        assert isinstance(labels, np.ndarray)

    def test_labels_consistent_across_modes(self, mnist_dir):
        """labels array is identical regardless of label_every_step."""
        _, labels_true = self._load("MNIST", mnist_dir, step_size=1, label_every_step=True)
        _, labels_false = self._load("MNIST", mnist_dir, step_size=1, label_every_step=False)
        np.testing.assert_array_equal(labels_true, labels_false)

    def test_labels_broadcast_into_x_seq(self, mnist_dir):
        """When label_every_step=True, label values appear in every timestep."""
        step_size = 1
        X, labels = self._load("MNIST", mnist_dir, step_size=step_size, label_every_step=True)
        seq_len = X.shape[1]
        for t in range(seq_len):
            np.testing.assert_array_almost_equal(X[:, t, step_size:], labels)

    # ── different dataset ─────────────────────────────────────────────────────

    def test_different_dataset_same_step_size(self, mnist1d_dir):
        """Works with MNIST1D-like data (40 features)."""
        X, labels = self._load("MNIST1D", mnist1d_dir, step_size=1, label_every_step=True)
        assert X.shape == (self.NUM_POINTS, 40, 11)   # 1 pixel + 10 labels
        assert labels.shape == (self.NUM_POINTS, 10)

    # ── error handling ────────────────────────────────────────────────────────

    def test_invalid_step_size_not_divisible(self, mnist_dir):
        """step_size=3 does not divide x_y_index=784 — must raise ValueError."""
        with pytest.raises(ValueError, match="not evenly divisible"):
            self._load("MNIST", mnist_dir, step_size=3)

    def test_dataset_without_x_y_index(self, no_xy_index_dir):
        """Dataset without x_y_index must raise ValueError."""
        with patch("generatedata.load_data.data_names", return_value=["circle"]):
            with pytest.raises(ValueError, match="no x_y_index metadata"):
                load_data_as_sequence(
                    "circle", step_size=1, local=True, data_dir=no_xy_index_dir
                )

    # ── seq_len computation ───────────────────────────────────────────────────

    @pytest.mark.parametrize("x_y_index,step_size", [
        (784, 1),
        (784, 28),
        (784, 784),
        (40, 1),
        (40, 10),
    ])
    def test_seq_len_computed_correctly(self, tmp_path, x_y_index, step_size):
        """seq_len == x_y_index // step_size for various (x_y_index, step_size) pairs."""
        name = f"ds_{x_y_index}_{step_size}"
        _make_dataset(
            tmp_path, name,
            num_points=50,
            num_features=x_y_index,
            label_dim=10,
            x_y_index=x_y_index,
        )
        with patch("generatedata.load_data.data_names", return_value=[name]):
            X, _ = load_data_as_sequence(
                name, step_size=step_size, local=True, data_dir=tmp_path
            )
        assert X.shape[1] == x_y_index // step_size

    def test_default_step_size_from_metadata(self, tmp_path):
        """When step_size is omitted, default_step_size from metadata is used."""
        name = "seq_ds"
        _make_dataset(
            tmp_path, name,
            num_points=50,
            num_features=40,
            label_dim=10,
            x_y_index=40,
        )
        # Patch the info JSON to include default_step_size
        import json
        info_path = tmp_path / f"{name}_info.json"
        with open(info_path) as f:
            info = json.load(f)
        info["default_step_size"] = 10
        info["is_sequence"] = True
        with open(info_path, "w") as f:
            json.dump(info, f)
        compile_info_json(tmp_path)

        with patch("generatedata.load_data.data_names", return_value=[name]):
            X, labels = load_data_as_sequence(
                name, local=True, data_dir=tmp_path
            )
        # step_size=10 from metadata, so 40/10 = 4 timesteps
        assert X.shape == (50, 4, 20)  # 10 features + 10 labels per step
        assert labels.shape == (50, 10)

    def test_no_step_size_no_default_raises(self, tmp_path):
        """Omitting step_size without default_step_size in metadata raises ValueError."""
        name = "plain_ds"
        _make_dataset(
            tmp_path, name,
            num_points=50,
            num_features=40,
            label_dim=10,
            x_y_index=40,
        )
        with patch("generatedata.load_data.data_names", return_value=[name]):
            with pytest.raises(ValueError, match="No step_size provided"):
                load_data_as_sequence(name, local=True, data_dir=tmp_path)
