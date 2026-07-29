"""Tests for the whest (ARC White-Box Estimation Challenge) dataset generators."""

import json
import math

import numpy as np
import pytest
import torch

from generatedata.data_generators import compile_info_json, dataset_exists
from generatedata.load_data import load_data, load_data_as_sequence, load_data_as_xy
from generatedata.whest_generators import (
    WHEST_BUDGET_BYTES,
    WHEST_CORE_SPEC,
    WHEST_GRID,
    WHEST_OFFICIAL_SOURCES,
    WHEST_XL,
    WHEST_XL_BUDGET_BYTES,
    generate_whest,
    he_weights,
    mc_layer_means,
    ut_fixed_final_mean,
    whest_dataset_bytes,
    whest_name,
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
    """The Monte-Carlo labels estimate E_x[z_l] for every layer, without bias."""

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
        means, second_moment = mc_layer_means(weights, samples, SEED + 1, "cpu")
        assert means.shape == (16, 1, WIDTH)
        exact = (
            weights[:, 0].pow(2).sum(dim=1).sqrt() / math.sqrt(2 * math.pi)
        ).double()
        # Judge the error on the scale of its own standard error, which the
        # returned second moment supplies: sqrt(Var_x / samples) per coordinate.
        standard_error = ((second_moment[:, 0] - means[:, 0] ** 2) / samples).sqrt()
        z_scores = (means[:, 0] - exact) / standard_error
        assert z_scores.abs().max().item() < 6.0
        assert abs(z_scores.mean().item()) < 0.5

    def test_layer_stack_is_ordered(self):
        """Each layer's mean must come from propagating through that layer only."""
        weights = he_weights(8, WIDTH, 3, SEED)
        means, _ = mc_layer_means(weights, 20_000, SEED + 1, "cpu")
        assert means.shape == (8, 3, WIDTH)
        # The depth-1 mean of the full network equals the depth-1 network's mean.
        prefix, _ = mc_layer_means(weights[:, :1], 20_000, SEED + 1, "cpu")
        assert torch.allclose(means[:, 0], prefix[:, 0], atol=2e-2)

    def test_narrow_deep_networks_are_dead(self):
        """A width-2 depth-32 ReLU net has F == 0 identically.

        A layer keeps an input only if some coordinate stays positive, and relu's
        positive homogeneity means no rescaling of W changes which coordinates
        those are -- so this is a property of the geometry, not the init.
        """
        weights = he_weights(8, 2, 32, SEED)
        means, _ = mc_layer_means(weights, 2048, SEED + 1, "cpu")
        assert torch.all(means[:, -1] == 0)


class TestFixedUT:
    """The cheap deterministic estimator carried by the start labels."""

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
        estimate = ut_fixed_final_mean(weights, "cpu")
        closed_form = radius * weights[:, 0].abs().sum(dim=1) / (2 * WIDTH)
        assert torch.allclose(estimate, closed_form, atol=1e-6)

    def test_deterministic_in_the_weights(self):
        """No random rotations: repeated calls give bit-identical estimates."""
        weights = he_weights(8, WIDTH, DEPTH, SEED)
        assert torch.equal(
            ut_fixed_final_mean(weights, "cpu", net_chunk=8),
            ut_fixed_final_mean(weights, "cpu", net_chunk=3),   # chunk-independent
        )


# ---------------------------------------------------------------------------
# The dataset on disk
# ---------------------------------------------------------------------------

def _generate(tmp_path, **kwargs):
    """Generate a small whest dataset and return (name, info)."""
    params = dict(
        width=WIDTH, depth=DEPTH, num_points=NUM_POINTS,
        mc_samples=MC_SAMPLES, seed=SEED,
    )
    params.update(kwargs)
    generate_whest(tmp_path, **params)
    name = whest_name(params["width"], params["depth"], params.get("suffix", ""))
    with open(tmp_path / f"{name}_info.json") as f:
        info = json.load(f)
    compile_info_json(tmp_path)
    return name, info


class TestGenerateWhest:
    def test_arrays_and_metadata(self, tmp_path):
        name, info = _generate(tmp_path)
        x_y_index = DEPTH * WIDTH * WIDTH

        weights = np.load(tmp_path / f"{name}_weights.npy")
        all_layers = np.load(tmp_path / f"{name}_all_layer_means.npy")
        final = np.load(tmp_path / f"{name}_final_means.npy")
        ut_fixed = np.load(tmp_path / f"{name}_ut_fixed.npy")

        assert weights.shape == (NUM_POINTS, DEPTH, WIDTH, WIDTH)
        assert all_layers.shape == (NUM_POINTS, DEPTH, WIDTH)
        assert final.shape == ut_fixed.shape == (NUM_POINTS, WIDTH)
        assert weights.dtype == np.float32

        assert info["storage"] == "npy"
        assert info["data_family"] == "whest"
        assert info["source"] == "generated"
        assert info["num_points"] == NUM_POINTS
        assert info["x_y_index"] == x_y_index
        assert info["y_size"] == WIDTH
        assert info["onehot_y"] == 0                  # regression, not classification
        assert info["default_step_size"] == WIDTH * WIDTH
        assert info["label_every_step_allowed"] is False
        assert info["mc_samples"] == MC_SAMPLES
        assert 0.0 < info["label_mc_se2"] < 1.0
        assert info["ut_final_layer_mse"] > 0.0
        assert 0.0 <= info["dead_coordinate_fraction"] <= 1.0
        assert info["dead_network_fraction"] == 0.0   # width 8 depth 8 stays alive

    def test_weights_are_reproducible_from_the_seed(self, tmp_path):
        name, _ = _generate(tmp_path)
        stored = np.load(tmp_path / f"{name}_weights.npy")
        assert np.array_equal(stored, he_weights(NUM_POINTS, WIDTH, DEPTH, SEED).numpy())

    def test_final_means_is_the_last_layer(self, tmp_path):
        name, _ = _generate(tmp_path)
        all_layers = np.load(tmp_path / f"{name}_all_layer_means.npy")
        final = np.load(tmp_path / f"{name}_final_means.npy")
        assert np.array_equal(final, all_layers[:, -1])

    def test_start_labels_are_the_cheap_estimate(self, tmp_path):
        """target - start on the label block is the residual R(W) = F - UT_fixed."""
        name, _ = _generate(tmp_path)
        expected = ut_fixed_final_mean(
            he_weights(NUM_POINTS, WIDTH, DEPTH, SEED), "cpu"
        ).numpy()
        stored = np.load(tmp_path / f"{name}_ut_fixed.npy")
        assert np.allclose(stored, expected, atol=1e-5)
        residual = np.load(tmp_path / f"{name}_final_means.npy") - stored
        assert np.abs(residual).max() > 1e-4   # there is something to learn

    def test_target_labels_match_an_independent_recomputation(self, tmp_path):
        """Recompute F for a few networks in float64 with a fresh input sample."""
        name, info = _generate(tmp_path)
        weights = he_weights(NUM_POINTS, WIDTH, DEPTH, SEED).double()
        generator = torch.Generator().manual_seed(1234)
        x = torch.randn(400_000, WIDTH, generator=generator, dtype=torch.float64)
        final = np.load(tmp_path / f"{name}_final_means.npy")
        # Both estimates are noisy, so compare against the sum of their variances.
        tolerance = 6.0 * math.sqrt(info["label_mc_se2"] + 0.5 / 400_000)
        for row in range(4):
            z = x
            for layer in range(DEPTH):
                z = torch.relu(z @ weights[row, layer])
            assert np.abs(final[row] - z.mean(dim=0).numpy()).max() < tolerance

    def test_size_model_matches_the_files(self, tmp_path):
        """whest_dataset_bytes must predict what actually lands on disk."""
        name, _ = _generate(tmp_path)
        actual = sum(p.stat().st_size for p in tmp_path.glob(f"{name}_*.npy"))
        predicted = whest_dataset_bytes(WIDTH, DEPTH, NUM_POINTS)
        # Four .npy headers of 128 bytes each are the only unmodelled bytes.
        assert predicted <= actual <= predicted + 1024

    def test_geometry_is_configurable(self, tmp_path):
        name, info = _generate(tmp_path, width=4, depth=2, num_points=8)
        assert name == "whest_w4_d2"
        assert info["x_y_index"] == 32 and info["y_size"] == 4
        assert info["default_step_size"] == 16
        assert np.load(tmp_path / f"{name}_weights.npy").shape == (8, 2, 4, 4)

    def test_partial_writes_are_not_left_behind(self, tmp_path):
        name, _ = _generate(tmp_path)
        assert not list(tmp_path.glob("*.partial"))
        assert dataset_exists(tmp_path, name, {"num_points": NUM_POINTS})


class TestLoading:
    """The family is reachable only through load_data_as_sequence."""

    def test_load_as_sequence_over_layers(self, tmp_path):
        """default_step_size = width^2 gives one weight matrix per timestep."""
        name, _ = _generate(tmp_path)
        X_seq, labels = load_data_as_sequence(name, local=True, data_dir=tmp_path)
        assert X_seq.shape == (NUM_POINTS, DEPTH, WIDTH * WIDTH)
        assert labels.shape == (NUM_POINTS, WIDTH)
        weights = he_weights(NUM_POINTS, WIDTH, DEPTH, SEED).numpy()
        assert np.array_equal(
            np.asarray(X_seq).reshape(NUM_POINTS, DEPTH, WIDTH, WIDTH), weights
        )

    def test_part_selects_the_label_block(self, tmp_path):
        name, _ = _generate(tmp_path)
        _, target = load_data_as_sequence(name, local=True, data_dir=tmp_path)
        _, start = load_data_as_sequence(name, local=True, data_dir=tmp_path, part="start")
        _, layers = load_data_as_sequence(
            name, local=True, data_dir=tmp_path, part="all_layers"
        )
        assert np.array_equal(target, np.load(tmp_path / f"{name}_final_means.npy"))
        assert np.array_equal(start, np.load(tmp_path / f"{name}_ut_fixed.npy"))
        assert layers.shape == (NUM_POINTS, DEPTH, WIDTH)
        assert np.array_equal(layers[:, -1], target)

    def test_unknown_part_raises(self, tmp_path):
        name, _ = _generate(tmp_path)
        with pytest.raises(ValueError, match="Unknown part"):
            load_data_as_sequence(name, local=True, data_dir=tmp_path, part="nonsense")

    def test_label_every_step_is_refused(self, tmp_path):
        """Broadcasting the labels into the features would leak the target."""
        name, _ = _generate(tmp_path)
        with pytest.raises(ValueError, match="leaking the target"):
            load_data_as_sequence(
                name, local=True, data_dir=tmp_path, label_every_step=True
            )

    def test_default_does_not_broadcast_labels(self, tmp_path):
        """The default must be safe: features are weights only, no label columns."""
        name, _ = _generate(tmp_path)
        X_seq, _ = load_data_as_sequence(name, local=True, data_dir=tmp_path)
        assert X_seq.shape[2] == WIDTH * WIDTH      # not WIDTH*WIDTH + WIDTH

    def test_memory_mapped_when_local(self, tmp_path):
        name, _ = _generate(tmp_path)
        X_seq, _ = load_data_as_sequence(name, local=True, data_dir=tmp_path)
        assert isinstance(X_seq, np.memmap)

    def test_load_data_refuses_with_a_pointer(self, tmp_path):
        """The flat DataFrame API must say what to use instead, not fail obscurely."""
        name, _ = _generate(tmp_path)
        with pytest.raises(ValueError, match="load_data_as_sequence"):
            load_data(name, local=True, data_dir=tmp_path)
        with pytest.raises(ValueError, match="load_data_as_sequence"):
            load_data_as_xy(name, local=True, data_dir=tmp_path)

    def test_compile_info_json_keeps_npy_datasets(self, tmp_path):
        """A dataset with no parquet pair must not be silently dropped."""
        name, _ = _generate(tmp_path)
        with open(tmp_path / "info.json") as f:
            compiled = json.load(f)
        assert name in compiled and compiled[name]["storage"] == "npy"


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

class TestLadder:
    """Guard the ladder table arithmetically; generating it costs ~700 MB."""

    ALL = WHEST_GRID + WHEST_XL

    @pytest.mark.parametrize("spec", WHEST_GRID, ids=lambda s: whest_name(s["width"], s["depth"]))
    def test_grid_within_budget(self, spec):
        size = whest_dataset_bytes(spec["width"], spec["depth"], spec["num_points"])
        assert size <= WHEST_BUDGET_BYTES, f"{size / 1e6:.1f} MB"

    @pytest.mark.parametrize("spec", WHEST_XL, ids=lambda s: whest_name(s["width"], s["depth"], "_xl"))
    def test_xl_within_budget(self, spec):
        size = whest_dataset_bytes(spec["width"], spec["depth"], spec["num_points"])
        assert WHEST_BUDGET_BYTES < size <= WHEST_XL_BUDGET_BYTES, f"{size / 1e6:.1f} MB"

    @pytest.mark.parametrize("spec", ALL, ids=lambda s: f"w{s['width']}_d{s['depth']}")
    def test_no_degenerate_geometry(self, spec):
        """Deep narrow nets are half dead, which makes the target degenerate.

        At width 8, depth 32 roughly half the coordinates and 5% of whole networks
        have F == 0; width 16 is the shallowest that keeps every net alive.
        """
        assert not (spec["width"] < 16 and spec["depth"] > 8), "too narrow for its depth"

    @pytest.mark.parametrize("spec", ALL, ids=lambda s: f"w{s['width']}_d{s['depth']}")
    def test_official_rows_are_available(self, spec):
        if spec.get("official") is None:
            assert "mc_samples" in spec, "our own bakes must pin a Monte-Carlo budget"
            return
        source = WHEST_OFFICIAL_SOURCES[spec["official"]]
        assert spec["width"] == source["width"] and spec["depth"] == source["depth"]
        assert spec["row_start"] + spec["num_points"] <= source["num_available"]

    def test_names_are_unique(self):
        names = [whest_name(s["width"], s["depth"], s.get("suffix", "")) for s in self.ALL]
        assert len(names) == len(set(names))

    def test_official_row_ranges_are_disjoint(self):
        """Two datasets carved from one split must not share networks."""
        used: dict[str, list[range]] = {}
        for spec in self.ALL:
            if spec.get("official") is None:
                continue
            span = range(spec["row_start"], spec["row_start"] + spec["num_points"])
            for other in used.setdefault(spec["official"], []):
                assert not (span.start < other.stop and other.start < span.stop), \
                    f"overlapping official rows: {span} vs {other}"
            used[spec["official"]].append(span)

    def test_core_spec_is_in_the_grid_and_is_the_cheapest(self):
        assert WHEST_CORE_SPEC in WHEST_GRID
        cost = lambda s: whest_dataset_bytes(s["width"], s["depth"], s["num_points"])
        assert cost(WHEST_CORE_SPEC) == min(cost(s) for s in WHEST_GRID)
        # It must need no download, since generate_all builds it unprompted.
        assert WHEST_CORE_SPEC.get("official") is None


class TestSweepDriver:
    """generate_whest_all with a stand-in table, since the real one is ~700 MB."""

    TINY = (dict(width=4, depth=2, num_points=8, mc_samples=4096),)

    def test_builds_then_skips(self, tmp_path, monkeypatch, capsys):
        from generatedata import whest_generators

        monkeypatch.setattr(whest_generators, "WHEST_GRID", self.TINY)
        monkeypatch.setattr(whest_generators, "WHEST_XL", ())

        whest_generators.generate_whest_all(tmp_path)
        first = capsys.readouterr().out
        assert "whest_w4_d2" in first and "Generating" in first
        assert "total" in first                # the preflight reports sizes up front
        assert (tmp_path / "whest_w4_d2_weights.npy").exists()

        # info.json is compiled, so the dataset is loadable straight away
        X_seq, labels = load_data_as_sequence("whest_w4_d2", local=True, data_dir=tmp_path)
        assert X_seq.shape == (8, 2, 16) and labels.shape == (8, 4)

        # A second call must recognise it rather than rebuild it
        whest_generators.generate_whest_all(tmp_path)
        assert "Skipping whest_w4_d2" in capsys.readouterr().out

    def test_xl_is_opt_in(self, tmp_path, monkeypatch, capsys):
        from generatedata import whest_generators

        monkeypatch.setattr(whest_generators, "WHEST_GRID", ())
        monkeypatch.setattr(
            whest_generators, "WHEST_XL",
            (dict(width=4, depth=2, num_points=8, mc_samples=4096, suffix="_xl"),),
        )
        whest_generators.generate_whest_all(tmp_path)
        assert not list(tmp_path.glob("*_xl_*.npy"))
        assert "include_xl=True" in capsys.readouterr().out

        whest_generators.generate_whest_all(tmp_path, include_xl=True)
        assert (tmp_path / "whest_w4_d2_xl_weights.npy").exists()


class TestCoreDataset:
    """The one rung a plain generate_all produces."""

    def test_present_in_the_session_data(self, generatedata_local_data):
        data_dir = generatedata_local_data
        name = whest_name(WHEST_CORE_SPEC["width"], WHEST_CORE_SPEC["depth"])
        with open(data_dir / "info.json") as f:
            info = json.load(f)[name]
        assert info["num_points"] == WHEST_CORE_SPEC["num_points"]
        assert info["storage"] == "npy"
        X_seq, labels = load_data_as_sequence(name, local=True, data_dir=data_dir)
        assert X_seq.shape == (info["num_points"], info["depth"],
                               info["width"] * info["width"])
        assert labels.shape == (info["num_points"], info["width"])
