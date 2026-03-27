"""Tests for Long Range Arena (LRA) dataset generators."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from generatedata.lra_generators import (
    _listops_evaluate,
    generate_lra_listops,
    generate_lra_pathfinder,
    generate_lra_pathx,
    generate_lra_image,
    generate_lra_text,
)
from generatedata.data_generators import compile_info_json
from generatedata.load_data import (
    load_data_as_xy_onehot,
    load_data_as_sequence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_dataset_files(data_dir, name, expected_seq_len, expected_num_classes, expected_num_points):
    """Verify parquet + info.json files for a generated LRA dataset."""
    start_path = data_dir / f"{name}_start.parquet"
    target_path = data_dir / f"{name}_target.parquet"
    info_path = data_dir / f"{name}_info.json"

    assert start_path.exists(), f"Missing {start_path}"
    assert target_path.exists(), f"Missing {target_path}"
    assert info_path.exists(), f"Missing {info_path}"

    start_df = pd.read_parquet(start_path)
    target_df = pd.read_parquet(target_path)

    with open(info_path) as f:
        info = json.load(f)

    total_cols = expected_seq_len + expected_num_classes

    # Shape checks
    assert start_df.shape == (expected_num_points, total_cols)
    assert target_df.shape == (expected_num_points, total_cols)

    # Info metadata checks
    assert info["num_points"] == expected_num_points
    assert info["size"] == total_cols
    assert info["x_y_index"] == expected_seq_len
    assert info["x_size"] == expected_seq_len
    assert info["y_size"] == expected_num_classes
    assert info["onehot_y"] == 1

    # Label checks: target labels should be valid one-hot
    target_labels = target_df.iloc[:, expected_seq_len:].to_numpy()
    assert np.allclose(target_labels.sum(axis=1), 1.0), "Target labels are not valid one-hot"
    assert set(np.unique(target_labels)).issubset({0.0, 1.0}), "Target labels contain non-binary values"

    # Label checks: start labels should be uniform
    start_labels = start_df.iloc[:, expected_seq_len:].to_numpy()
    expected_uniform = 1.0 / expected_num_classes
    assert np.allclose(start_labels, expected_uniform), "Start labels are not uniform"

    # Features should be identical between start and target
    start_features = start_df.iloc[:, :expected_seq_len].to_numpy()
    target_features = target_df.iloc[:, :expected_seq_len].to_numpy()
    assert np.allclose(start_features, target_features), "Features differ between start and target"

    # Sequence metadata checks — all LRA datasets are sequence datasets
    assert info.get("is_sequence") is True, "LRA datasets must have is_sequence=True"
    assert info.get("default_step_size") == 1, "LRA datasets must have default_step_size=1"

    return start_df, target_df, info


# ---------------------------------------------------------------------------
# ListOps unit tests
# ---------------------------------------------------------------------------

class TestListOpsEvaluator:
    """Test the ListOps expression evaluator."""

    def test_max(self):
        tokens = ["[", "MAX", "3", "5", "1", "]"]
        assert _listops_evaluate(tokens) == 5

    def test_min(self):
        tokens = ["[", "MIN", "7", "2", "9", "]"]
        assert _listops_evaluate(tokens) == 2

    def test_median(self):
        tokens = ["[", "MEDIAN", "1", "5", "3", "]"]
        assert _listops_evaluate(tokens) == 3

    def test_sum_mod(self):
        tokens = ["[", "SUM_MOD", "7", "8", "]"]
        assert _listops_evaluate(tokens) == 5  # (7+8) % 10 = 5

    def test_nested(self):
        # [MAX [MIN 3 5] 2] = MAX(MIN(3,5), 2) = MAX(3, 2) = 3
        tokens = ["[", "MAX", "[", "MIN", "3", "5", "]", "2", "]"]
        assert _listops_evaluate(tokens) == 3


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

NUM_POINTS = 50  # Small for fast tests


class TestListOps:
    def test_generate(self, tmp_path):
        generate_lra_listops(tmp_path, num_points=NUM_POINTS, seq_length=512)
        start_df, target_df, info = _check_dataset_files(
            tmp_path, "lra_listops", 512, 10, NUM_POINTS
        )
        # Token values should be in [0, vocab_size)
        features = target_df.iloc[:, :512].to_numpy()
        assert features.min() >= 0
        assert features.max() < 17  # LISTOPS_VOCAB_SIZE

        # Labels should be in [0, 10)
        label_cols = target_df.iloc[:, 512:].to_numpy()
        actual_labels = label_cols.argmax(axis=1)
        assert all(0 <= l < 10 for l in actual_labels)


class TestPathfinder:
    def test_generate(self, tmp_path):
        generate_lra_pathfinder(tmp_path, num_points=NUM_POINTS, image_size=32)
        start_df, target_df, info = _check_dataset_files(
            tmp_path, "lra_pathfinder", 1024, 2, NUM_POINTS
        )
        # Pixel values should be in [0, 1]
        features = target_df.iloc[:, :1024].to_numpy()
        assert features.min() >= 0.0
        assert features.max() <= 1.0

        # Check roughly balanced classes
        labels = target_df.iloc[:, 1024:].to_numpy().argmax(axis=1)
        assert len(set(labels)) == 2, "Both classes should be present"


class TestPathX:
    def test_generate(self, tmp_path):
        # Use tiny image size to keep test fast
        generate_lra_pathx(tmp_path, num_points=NUM_POINTS, image_size=16)
        start_df, target_df, info = _check_dataset_files(
            tmp_path, "lra_pathx", 256, 2, NUM_POINTS
        )
        features = target_df.iloc[:, :256].to_numpy()
        assert features.min() >= 0.0
        assert features.max() <= 1.0


class TestImage:
    def test_generate(self, tmp_path):
        generate_lra_image(tmp_path, num_points=NUM_POINTS)
        start_df, target_df, info = _check_dataset_files(
            tmp_path, "lra_image", 1024, 10, NUM_POINTS
        )
        # Grayscale pixel values should be in [0, 1]
        features = target_df.iloc[:, :1024].to_numpy()
        assert features.min() >= 0.0
        assert features.max() <= 1.0

        # All 10 classes might not be represented with only 50 samples,
        # but labels should be in range
        labels = target_df.iloc[:, 1024:].to_numpy().argmax(axis=1)
        assert all(0 <= l < 10 for l in labels)


class TestText:
    def test_generate(self, tmp_path):
        generate_lra_text(tmp_path, num_points=NUM_POINTS, seq_length=256)
        start_df, target_df, info = _check_dataset_files(
            tmp_path, "lra_text", 256, 2, NUM_POINTS
        )
        # Byte values should be in [0, 255]
        features = target_df.iloc[:, :256].to_numpy()
        assert features.min() >= 0
        assert features.max() <= 255

        # Binary labels
        labels = target_df.iloc[:, 256:].to_numpy().argmax(axis=1)
        assert set(np.unique(labels)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Integration: verify load_data_as_sequence compatibility
# ---------------------------------------------------------------------------

class TestSequenceLoading:
    """Verify that generated LRA data works with load_data_as_sequence."""

    def test_listops_as_sequence(self, tmp_path):
        seq_len = 128
        generate_lra_listops(tmp_path, num_points=20, seq_length=seq_len)
        # Manually replicate what load_data_as_sequence does
        target_df = pd.read_parquet(tmp_path / "lra_listops_target.parquet")
        x_y_index = seq_len
        step_size = 1
        features = target_df.iloc[:, :x_y_index].to_numpy()
        seq_len_actual = x_y_index // step_size
        X_seq = features.reshape(features.shape[0], seq_len_actual, step_size)
        assert X_seq.shape == (20, seq_len, 1)

    def test_pathfinder_as_sequence_row_by_row(self, tmp_path):
        generate_lra_pathfinder(tmp_path, num_points=20, image_size=32)
        target_df = pd.read_parquet(tmp_path / "lra_pathfinder_target.parquet")
        x_y_index = 1024
        step_size = 32  # one row at a time
        features = target_df.iloc[:, :x_y_index].to_numpy()
        seq_len_actual = x_y_index // step_size
        X_seq = features.reshape(features.shape[0], seq_len_actual, step_size)
        assert X_seq.shape == (20, 32, 32)


# ---------------------------------------------------------------------------
# Sequence-native loading behaviour
# ---------------------------------------------------------------------------

class TestSequenceNativeWarning:
    """Verify that loading LRA data as flat X/Y emits a warning."""

    def test_load_as_xy_onehot_warns(self, tmp_path):
        generate_lra_listops(tmp_path, num_points=20, seq_length=128)
        compile_info_json(tmp_path)
        with patch("generatedata.load_data.data_names", return_value=["lra_listops"]):
            with pytest.warns(UserWarning, match="sequence dataset"):
                load_data_as_xy_onehot("lra_listops", local=True, data_dir=tmp_path)


class TestDefaultStepSize:
    """Verify that LRA datasets can be loaded as sequences without explicit step_size."""

    def test_lra_default_step_size(self, tmp_path):
        """LRA dataset with default_step_size can omit step_size argument."""
        seq_len = 128
        generate_lra_listops(tmp_path, num_points=20, seq_length=seq_len)
        compile_info_json(tmp_path)
        with patch("generatedata.load_data.data_names", return_value=["lra_listops"]):
            X_seq, labels = load_data_as_sequence(
                "lra_listops", local=True, data_dir=tmp_path, label_every_step=False,
            )
        # default_step_size=1, so each timestep has 1 feature
        assert X_seq.shape == (20, seq_len, 1)
        assert labels.shape == (20, 10)

    def test_missing_step_size_and_no_default_raises(self, tmp_path):
        """Dataset without default_step_size must raise ValueError when step_size omitted."""
        from generatedata.save_data import save_data
        rng = np.random.default_rng(42)
        cols = {f"x{i}": rng.random(50) for i in range(20)}
        save_data(tmp_path, "plain", cols, cols, x_y_index=10, onehot_y=True)
        compile_info_json(tmp_path)
        with patch("generatedata.load_data.data_names", return_value=["plain"]):
            with pytest.raises(ValueError, match="No step_size provided"):
                load_data_as_sequence("plain", local=True, data_dir=tmp_path)
