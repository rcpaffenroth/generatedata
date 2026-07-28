"""Tests for the whest (ARC White-Box Estimation Challenge) dataset generators."""

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch

from generatedata.data_generators import compile_info_json
from generatedata.load_data import load_data, load_data_as_sequence, load_data_as_xy
from generatedata.whest_generators import (
    generate_whest,
    he_weights,
    mc_final_mean,
    ut_fixed_final_mean,
)

WIDTH, DEPTH = 8, 8
NUM_POINTS = 32          # small for fast tests
MC_SAMPLES = 8192        # small for fast tests; noise is checked, not assumed
SEED = 7


# ---------------------------------------------------------------------------
# The pieces: weights, ground truth, cheap estimator
# ---------------------------------------------------------------------------

class TestHeWeights:
    """W ~ N(0, 2/width), no biases, reproducible from the seed."""

    def test_shape_and_scale(self):
        weights = he_weights(2048, WIDTH, DEPTH, SEED)
        assert weights.shape == (2048, DEPTH, WIDTH, WIDTH)
        expected_std = math.sqrt(2.0 / WIDTH)
        assert abs(weights.std().item() - expected_std) < 0.01 * expected_std
        assert abs(weights.mean().item()) < 0.01 * expected_std

    def test_reproducible(self):
        first = he_weights(4, WIDTH, DEPTH, SEED)
        second = he_weights(4, WIDTH, DEPTH, SEED)
        third = he_weights(4, WIDTH, DEPTH, SEED + 1)
        assert torch.equal(first, second)
        assert not torch.equal(first, third)


class TestGroundTruth:
    """The Monte-Carlo label estimates F(W) = E_x[relu-chain(x)] without bias."""

    def test_matches_exact_single_layer_mean(self):
        """At depth 1 the target is analytic, which pins down the convention.

        With z = relu(x W) and x ~ N(0, I), the pre-activation of neuron j is
        x . W[:, j] ~ N(0, ||W[:, j]||^2), and E[relu(N(0, s^2))] = s / sqrt(2 pi).
        So F(W)_j = ||W[:, j]|| / sqrt(2 pi) -- note the COLUMN norm, which holds
        only for the `z @ W` convention (a `z @ W.T` forward would give the row
        norms instead).
        """
        weights = he_weights(16, WIDTH, 1, SEED)
        samples = 200_000
        mean, second_moment = mc_final_mean(weights, samples, SEED + 1, "cpu", 16)
        exact = (
            weights[:, 0].pow(2).sum(dim=1).sqrt() / math.sqrt(2 * math.pi)
        ).double()
        # Judge the error on the scale of its own standard error, which the
        # returned second moment supplies: sqrt(Var_x / samples) per neuron.
        standard_error = ((second_moment - mean ** 2) / samples).sqrt()
        z_scores = (mean - exact) / standard_error
        assert z_scores.abs().max().item() < 6.0
        assert abs(z_scores.mean().item()) < 0.5

    def test_narrow_deep_networks_are_dead(self):
        """A width-2 depth-32 ReLU net has F == 0 identically.

        A layer keeps an input only if some coordinate stays positive, and relu's
        positive homogeneity means no rescaling of W changes which coordinates
        those are -- so this is a property of the geometry, not the init.
        """
        weights = he_weights(8, 2, 32, SEED)
        mean, _ = mc_final_mean(weights, 2048, SEED + 1, "cpu", 8)
        assert torch.all(mean == 0)


class TestFixedUT:
    """The cheap deterministic estimator carried by the start data."""

    def test_matches_closed_form_single_layer(self):
        """At depth 1 the sigma-point average is exact arithmetic.

        The points are the rows of [r I; -r I], so the first layer gives
        relu(+-r W[i, :]) and averaging over all 2*width of them leaves
        (r / (2 width)) * sum_i |W[i, j]|.
        """
        weights = he_weights(16, WIDTH, 1, SEED)
        radius = math.sqrt(2.0) * math.exp(
            math.lgamma((WIDTH + 1) / 2) - math.lgamma(WIDTH / 2)
        )
        estimate = ut_fixed_final_mean(weights, "cpu", 16)
        closed_form = radius * weights[:, 0].abs().sum(dim=1) / (2 * WIDTH)
        assert torch.allclose(estimate, closed_form, atol=1e-6)

    def test_deterministic_in_the_weights(self):
        """No random rotations: repeated calls give bit-identical estimates."""
        weights = he_weights(8, WIDTH, DEPTH, SEED)
        assert torch.equal(
            ut_fixed_final_mean(weights, "cpu", 8),
            ut_fixed_final_mean(weights, "cpu", 3),      # also chunk-independent
        )


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------

def _generate(tmp_path, **kwargs):
    """Generate a small whest dataset and return (start_df, target_df, info)."""
    params = dict(
        width=WIDTH, depth=DEPTH, num_points=NUM_POINTS,
        mc_samples=MC_SAMPLES, seed=SEED,
    )
    params.update(kwargs)
    generate_whest(tmp_path, **params)
    name = f"whest_w{params['width']}_d{params['depth']}"
    start_df = pd.read_parquet(tmp_path / f"{name}_start.parquet")
    target_df = pd.read_parquet(tmp_path / f"{name}_target.parquet")
    with open(tmp_path / f"{name}_info.json") as f:
        info = json.load(f)
    return start_df, target_df, info


class TestGenerateWhest:
    def test_files_and_metadata(self, tmp_path):
        start_df, target_df, info = _generate(tmp_path)
        x_y_index = DEPTH * WIDTH * WIDTH
        total_columns = x_y_index + WIDTH

        assert start_df.shape == (NUM_POINTS, total_columns)
        assert target_df.shape == (NUM_POINTS, total_columns)
        assert info["num_points"] == NUM_POINTS
        assert info["size"] == total_columns
        assert info["x_y_index"] == x_y_index
        assert info["x_size"] == x_y_index
        assert info["y_size"] == WIDTH
        assert info["onehot_y"] == 0          # regression, not classification
        assert info["data_family"] == "whest"
        assert info["width"] == WIDTH and info["depth"] == DEPTH
        assert info["mc_samples"] == MC_SAMPLES
        # One width x width matrix per timestep, in layer order.
        assert info["default_step_size"] == WIDTH * WIDTH
        # Not padded, so the flat X/Y view is legitimate and must not warn.
        assert "is_sequence" not in info
        # Diagnostics: label noise, the cheap estimator's error, dead nets.
        assert 0.0 < info["label_mc_se2"] < 1.0
        assert info["ut_final_layer_mse"] > 0.0
        assert 0.0 <= info["dead_fraction"] <= 1.0

    def test_features_are_the_flattened_weights(self, tmp_path):
        """X is flatten(W) in layer-major order, identical in start and target."""
        start_df, target_df, _ = _generate(tmp_path)
        x_y_index = DEPTH * WIDTH * WIDTH
        features = target_df.iloc[:, :x_y_index].to_numpy()
        weights = he_weights(NUM_POINTS, WIDTH, DEPTH, SEED)
        assert np.array_equal(features, weights.reshape(NUM_POINTS, -1).numpy())
        assert np.array_equal(features, start_df.iloc[:, :x_y_index].to_numpy())

    def test_start_label_is_the_cheap_estimate(self, tmp_path):
        """start - target on the label block is the residual R(W) = F - UT_fixed."""
        start_df, target_df, _ = _generate(tmp_path)
        x_y_index = DEPTH * WIDTH * WIDTH
        weights = he_weights(NUM_POINTS, WIDTH, DEPTH, SEED)
        expected = ut_fixed_final_mean(weights, "cpu", NUM_POINTS).numpy()
        stored = start_df.iloc[:, x_y_index:].to_numpy()
        assert np.allclose(stored, expected, atol=1e-5)
        # The residual is not identically zero -- there is something to learn.
        residual = target_df.iloc[:, x_y_index:].to_numpy() - stored
        assert np.abs(residual).max() > 1e-4

    def test_target_label_is_the_mean_activation(self, tmp_path):
        """Recompute F for a few networks independently, in float64.

        Uses a fresh input sample and the `z <- relu(z @ W)` forward map, then
        checks agreement at the Monte-Carlo noise level of the stored label.
        """
        _, target_df, info = _generate(tmp_path)
        x_y_index = DEPTH * WIDTH * WIDTH
        weights = he_weights(NUM_POINTS, WIDTH, DEPTH, SEED).double()
        generator = torch.Generator().manual_seed(1234)
        x = torch.randn(400_000, WIDTH, generator=generator, dtype=torch.float64)
        # Both estimates are noisy, so compare against the sum of their variances.
        tolerance = 6.0 * math.sqrt(info["label_mc_se2"] + 0.5 / 400_000)
        for row in range(4):
            z = x
            for layer in range(DEPTH):
                z = torch.relu(z @ weights[row, layer])
            stored = target_df.iloc[row, x_y_index:].to_numpy().astype(np.float64)
            assert np.abs(stored - z.mean(dim=0).numpy()).max() < tolerance

    def test_geometry_is_configurable(self, tmp_path):
        _, target_df, info = _generate(tmp_path, width=4, depth=2, num_points=8)
        assert target_df.shape == (8, 2 * 4 * 4 + 4)
        assert info["x_y_index"] == 32 and info["y_size"] == 4
        assert info["default_step_size"] == 16
        assert info["weight_std"] == pytest.approx(math.sqrt(2.0 / 4))


class TestLoading:
    """The dataset must work with the existing load_data API, unchanged."""

    def test_load_as_xy(self, tmp_path):
        _generate(tmp_path)
        compile_info_json(tmp_path)
        with patch("generatedata.load_data.data_names", return_value=["whest_w8_d8"]):
            X, Y = load_data_as_xy("whest_w8_d8", local=True, data_dir=tmp_path)
        assert X.shape == (NUM_POINTS, DEPTH * WIDTH * WIDTH)
        assert Y.shape == (NUM_POINTS, WIDTH)

    def test_load_as_sequence_over_layers(self, tmp_path):
        """default_step_size = width^2 gives one weight matrix per timestep."""
        _generate(tmp_path)
        compile_info_json(tmp_path)
        with patch("generatedata.load_data.data_names", return_value=["whest_w8_d8"]):
            X_seq, labels = load_data_as_sequence(
                "whest_w8_d8", local=True, data_dir=tmp_path, label_every_step=False,
            )
        assert X_seq.shape == (NUM_POINTS, DEPTH, WIDTH * WIDTH)
        assert labels.shape == (NUM_POINTS, WIDTH)
        weights = he_weights(NUM_POINTS, WIDTH, DEPTH, SEED).numpy()
        assert np.array_equal(X_seq.reshape(NUM_POINTS, DEPTH, WIDTH, WIDTH), weights)

    def test_start_target_pair_loads(self, tmp_path):
        _generate(tmp_path)
        compile_info_json(tmp_path)
        with patch("generatedata.load_data.data_names", return_value=["whest_w8_d8"]):
            data = load_data("whest_w8_d8", local=True, data_dir=tmp_path)
        assert data["start"].shape == data["target"].shape
        assert data["info"]["data_family"] == "whest"


class TestCoreDataset:
    """whest_w8_d8 is part of the core set that a plain generate_all produces."""

    def test_present_in_the_session_data(self, generatedata_local_data):
        data_dir = generatedata_local_data
        with open(data_dir / "info.json") as f:
            info = json.load(f)["whest_w8_d8"]
        assert info["num_points"] == 10000
        assert info["width"] == 8 and info["depth"] == 8
        assert info["x_y_index"] == 512 and info["y_size"] == 8
