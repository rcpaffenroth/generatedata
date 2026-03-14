"""
Tests for the three timeseries tasks:
  Task 1 - save_data() seq_len / step_size parameters
  Task 2 - generate_mnist_sequence() / generate_mnist1d_sequence()
  Task 3 - load_data_as_sequence()
"""
import json
import pickle
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from generatedata.save_data import save_data
from generatedata.data_generators import compile_info_json
from generatedata.load_data import load_data_as_sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seq_dataset(data_dir: Path, name: str, num_points: int,
                      total_pixels: int, label_dim: int,
                      seq_len: int, step_size: int) -> None:
    """Write a synthetic flat sequence dataset using save_data()."""
    rng = np.random.default_rng(42)
    pixel_cols = {f"x{i}": rng.random(num_points) for i in range(total_pixels)}
    label_cols = {f"x{total_pixels + j}": rng.random(num_points)
                  for j in range(label_dim)}
    target_data = {**pixel_cols, **label_cols}
    start_data = {**pixel_cols, **label_cols}

    save_data(
        data_dir, name, start_data, target_data,
        x_y_index=total_pixels, onehot_y=True,
        seq_len=seq_len, step_size=step_size,
    )
    compile_info_json(data_dir)


def _make_fake_torchvision_mnist(n: int = 20, img_size: int = 28,
                                 num_classes: int = 10) -> MagicMock:
    """Fake torchvision Dataset that looks like MNIST."""
    labels = torch.randint(0, num_classes, (n,))
    images = [
        (torch.rand(1, img_size, img_size), int(labels[i]))
        for i in range(n)
    ]
    mock_ds = MagicMock()
    mock_ds.targets = labels
    mock_ds.__len__ = MagicMock(return_value=n)
    mock_ds.__getitem__ = MagicMock(side_effect=lambda i: images[i % n])
    return mock_ds


def _make_fake_mnist1d_pickle(n: int = 20, dim: int = 40) -> bytes:
    """Pickle bytes that looks like the MNIST1D download."""
    data = {
        "x": np.random.rand(n, dim).astype(np.float32),
        "y": np.random.randint(0, 10, size=n),
    }
    return pickle.dumps(data)


# ---------------------------------------------------------------------------
# Task 1 — save_data() sequence metadata
# ---------------------------------------------------------------------------

class TestSaveDataSequenceMetadata:
    def test_seq_metadata_written_to_info_json(self, tmp_path):
        """seq_len and step_size appear in the per-dataset info file."""
        data = {f"x{i}": np.arange(5) for i in range(4)}
        save_data(tmp_path, "ds", data, data,
                  x_y_index=4, onehot_y=True, seq_len=2, step_size=2)

        with open(tmp_path / "ds_info.json") as f:
            info = json.load(f)

        assert info["seq_len"] == 2
        assert info["step_size"] == 2

    def test_no_seq_metadata_without_params(self, tmp_path):
        """Backward compat: seq_len / step_size absent when not supplied."""
        data = {"x0": np.arange(5), "x1": np.arange(5)}
        save_data(tmp_path, "ds", data, data, x_y_index=1)

        with open(tmp_path / "ds_info.json") as f:
            info = json.load(f)

        assert "seq_len" not in info
        assert "step_size" not in info

    def test_mismatch_raises_value_error(self, tmp_path):
        """seq_len * step_size != x_y_index must raise ValueError."""
        data = {f"x{i}": np.arange(5) for i in range(6)}
        with pytest.raises(ValueError, match="must equal x_y_index"):
            # 3 * 3 = 9 != 6
            save_data(tmp_path, "ds", data, data,
                      x_y_index=6, seq_len=3, step_size=3)

    def test_existing_keys_unaffected(self, tmp_path):
        """Adding seq metadata does not remove other existing keys."""
        data = {f"x{i}": np.arange(5) for i in range(4)}
        save_data(tmp_path, "ds", data, data,
                  x_y_index=4, onehot_y=True, seq_len=2, step_size=2)

        with open(tmp_path / "ds_info.json") as f:
            info = json.load(f)

        for key in ("num_points", "size", "x_y_index", "x_size", "y_size", "onehot_y"):
            assert key in info, f"Expected key '{key}' missing from info"

    def test_only_seq_len_provided_no_metadata_written(self, tmp_path):
        """Partial provision (only seq_len, no step_size) writes nothing."""
        data = {f"x{i}": np.arange(5) for i in range(4)}
        save_data(tmp_path, "ds", data, data, x_y_index=4, seq_len=4)

        with open(tmp_path / "ds_info.json") as f:
            info = json.load(f)

        assert "seq_len" not in info
        assert "step_size" not in info


# ---------------------------------------------------------------------------
# Task 2 — generator functions (mocked to avoid downloading real datasets)
# ---------------------------------------------------------------------------

class TestGenerateMnistSequence:
    """generate_mnist_sequence() with torchvision.datasets mocked out."""

    @patch("generatedata.data_generators.datasets")
    def test_creates_files_with_correct_shape(self, mock_datasets, tmp_path):
        n = 20
        mock_datasets.MNIST.return_value = _make_fake_torchvision_mnist(n)

        from generatedata.data_generators import generate_mnist_sequence
        generate_mnist_sequence(tmp_path, num_points=n, pixels_per_step=1)

        target = pd.read_parquet(tmp_path / "MNIST_seq1_target.parquet")
        assert target.shape == (n, 794)   # 784 pixels + 10 labels

    @patch("generatedata.data_generators.datasets")
    def test_info_contains_seq_metadata(self, mock_datasets, tmp_path):
        n = 20
        mock_datasets.MNIST.return_value = _make_fake_torchvision_mnist(n)

        from generatedata.data_generators import generate_mnist_sequence
        generate_mnist_sequence(tmp_path, num_points=n, pixels_per_step=1)

        with open(tmp_path / "MNIST_seq1_info.json") as f:
            info = json.load(f)

        assert info["seq_len"] == 784
        assert info["step_size"] == 1
        assert info["x_y_index"] == 784

    @patch("generatedata.data_generators.datasets")
    def test_naming_convention_pixels_per_step(self, mock_datasets, tmp_path):
        """Dataset name encodes pixels_per_step: FashionMNIST_seq28."""
        n = 20
        mock_datasets.FashionMNIST.return_value = _make_fake_torchvision_mnist(n)

        from generatedata.data_generators import generate_mnist_sequence
        generate_mnist_sequence(tmp_path, num_points=n,
                                pixels_per_step=28, dataset_name="FashionMNIST")

        assert (tmp_path / "FashionMNIST_seq28_target.parquet").exists()

    @patch("generatedata.data_generators.datasets")
    def test_standard_generate_mnist_no_seq_metadata(self, mock_datasets, tmp_path):
        """generate_mnist() must NOT write seq_len / step_size (backward compat)."""
        n = 20
        mock_datasets.MNIST.return_value = _make_fake_torchvision_mnist(n)

        from generatedata.data_generators import generate_mnist
        generate_mnist(tmp_path, num_points=n)

        with open(tmp_path / "MNIST_info.json") as f:
            info = json.load(f)

        assert "seq_len" not in info
        assert "step_size" not in info


class TestGenerateMnist1dSequence:
    """generate_mnist1d_sequence() with network I/O mocked out."""

    @patch("generatedata.data_generators.requests")
    def test_creates_files_with_correct_shape(self, mock_requests, tmp_path):
        n = 20
        mock_requests.get.return_value.content = _make_fake_mnist1d_pickle(n)

        from generatedata.data_generators import generate_mnist1d_sequence
        generate_mnist1d_sequence(tmp_path, num_points=n, pixels_per_step=1)

        target = pd.read_parquet(tmp_path / "MNIST1D_seq1_target.parquet")
        assert target.shape == (n, 50)   # 40 pixels + 10 labels

    @patch("generatedata.data_generators.requests")
    def test_info_contains_seq_metadata(self, mock_requests, tmp_path):
        n = 20
        mock_requests.get.return_value.content = _make_fake_mnist1d_pickle(n)

        from generatedata.data_generators import generate_mnist1d_sequence
        generate_mnist1d_sequence(tmp_path, num_points=n, pixels_per_step=1)

        with open(tmp_path / "MNIST1D_seq1_info.json") as f:
            info = json.load(f)

        assert info["seq_len"] == 40
        assert info["step_size"] == 1

    @patch("generatedata.data_generators.requests")
    def test_standard_generate_mnist1d_no_seq_metadata(self, mock_requests, tmp_path):
        """generate_mnist1d() must NOT write seq_len / step_size."""
        n = 20
        mock_requests.get.return_value.content = _make_fake_mnist1d_pickle(n)

        from generatedata.data_generators import generate_mnist1d
        generate_mnist1d(tmp_path, num_points=n)

        with open(tmp_path / "MNIST1D_info.json") as f:
            info = json.load(f)

        assert "seq_len" not in info
        assert "step_size" not in info


# ---------------------------------------------------------------------------
# Task 3 — load_data_as_sequence()
# ---------------------------------------------------------------------------

class TestLoadDataAsSequence:
    NUM_POINTS = 100
    SEQ_LEN = 784
    STEP_SIZE = 1
    LABEL_DIM = 10

    @pytest.fixture()
    def seq_data_dir(self, tmp_path):
        _make_seq_dataset(
            tmp_path, "MNIST_seq1",
            num_points=self.NUM_POINTS,
            total_pixels=self.SEQ_LEN * self.STEP_SIZE,
            label_dim=self.LABEL_DIM,
            seq_len=self.SEQ_LEN,
            step_size=self.STEP_SIZE,
        )
        return tmp_path

    @pytest.fixture()
    def plain_data_dir(self, tmp_path):
        """Dataset without sequence metadata."""
        data = {f"x{i}": np.random.rand(self.NUM_POINTS) for i in range(794)}
        save_data(tmp_path, "MNIST", data, data, x_y_index=784, onehot_y=True)
        compile_info_json(tmp_path)
        return tmp_path

    def _load(self, name, data_dir, **kwargs):
        """Call load_data_as_sequence with data_names mocked to include name."""
        with patch("generatedata.load_data.data_names", return_value=[name]):
            return load_data_as_sequence(name, local=True, data_dir=data_dir, **kwargs)

    def test_shape_label_every_step_true(self, seq_data_dir):
        X, labels = self._load("MNIST_seq1", seq_data_dir, label_every_step=True)
        assert X.shape == (self.NUM_POINTS, self.SEQ_LEN,
                           self.STEP_SIZE + self.LABEL_DIM)
        assert labels.shape == (self.NUM_POINTS, self.LABEL_DIM)

    def test_shape_label_every_step_false(self, seq_data_dir):
        X, labels = self._load("MNIST_seq1", seq_data_dir, label_every_step=False)
        assert X.shape == (self.NUM_POINTS, self.SEQ_LEN, self.STEP_SIZE)
        assert labels.shape == (self.NUM_POINTS, self.LABEL_DIM)

    def test_returns_numpy_arrays(self, seq_data_dir):
        X, labels = self._load("MNIST_seq1", seq_data_dir)
        assert isinstance(X, np.ndarray)
        assert isinstance(labels, np.ndarray)

    def test_raises_on_non_sequence_dataset(self, plain_data_dir):
        with patch("generatedata.load_data.data_names", return_value=["MNIST"]):
            with pytest.raises(ValueError, match="no sequence metadata"):
                load_data_as_sequence("MNIST", local=True, data_dir=plain_data_dir)

    def test_error_message_contains_dataset_name(self, plain_data_dir):
        with patch("generatedata.load_data.data_names", return_value=["MNIST"]):
            with pytest.raises(ValueError, match="'MNIST'"):
                load_data_as_sequence("MNIST", local=True, data_dir=plain_data_dir)

    def test_labels_consistent_across_modes(self, seq_data_dir):
        """labels array is identical whether label_every_step is True or False."""
        _, labels_true = self._load("MNIST_seq1", seq_data_dir, label_every_step=True)
        _, labels_false = self._load("MNIST_seq1", seq_data_dir, label_every_step=False)
        np.testing.assert_array_equal(labels_true, labels_false)

    def test_labels_broadcast_into_x_seq(self, seq_data_dir):
        """When label_every_step=True, label values appear in every timestep."""
        X, labels = self._load("MNIST_seq1", seq_data_dir, label_every_step=True)
        for t in range(self.SEQ_LEN):
            np.testing.assert_array_almost_equal(
                X[:, t, self.STEP_SIZE:], labels
            )

    def test_multi_step_size(self, tmp_path):
        """Works with step_size > 1 (e.g. 28 pixels per step for MNIST rows)."""
        seq_len, step_size, label_dim, n = 28, 28, 10, 50
        _make_seq_dataset(tmp_path, "MNIST_seq28",
                          num_points=n, total_pixels=seq_len * step_size,
                          label_dim=label_dim, seq_len=seq_len, step_size=step_size)

        X, labels = self._load("MNIST_seq28", tmp_path, label_every_step=True)
        assert X.shape == (n, seq_len, step_size + label_dim)
        assert labels.shape == (n, label_dim)
