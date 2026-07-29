"""
ARC White-Box Estimation Challenge (whest) dataset generators.

The task: given only the weights of a deep ReLU MLP, predict the mean activation
of its layers under a standard normal input.  Writing one network's forward map as
a product of nonlinear layers,

    z_0 = x,        z_l = relu(z_{l-1} W_l),        l = 1 .. depth,

the target is the deterministic functional

    F(W) = E_{x ~ N(0, I_width)} [ z_depth ]  in  R^width,

together with its per-layer counterparts.  The input distribution is integrated
out, so ``F`` is a function of ``W`` alone.  Viewed as a dynamical system, each
network is a product of random matrices acting on a distribution and ``F(W)`` is a
functional of its terminal distribution; the heavy tail of ``F`` across random
``W`` is the large-deviation statistics of the finite-time Lyapunov exponent
(expanding vs contracting products), not noise.

Competition conventions reproduced here, verified against the official published
data (see ``docs`` in the ``whest`` project and its ``experiments/PRINCIPLES.md``):

  * weights are iid ``N(0, 2/width)`` (He initialisation at fan_in = width), with
    **no biases**;
  * the forward map is ``z <- relu(z @ W_l)`` — the row-vector convention, i.e.
    ``@W`` and *not* ``@W.T``.  Propagating the official weights this way
    reproduces the official means to within Monte-Carlo noise, while ``@W.T`` is
    wrong by O(1);
  * ``relu`` is applied at every layer, including the last;
  * accuracy is judged as raw mean-squared error, so ground truth is accumulated
    in float64 with TF32 disabled.

Why this family does not use the flat parquet format
----------------------------------------------------
Every other dataset here stores one scalar parquet column per feature.  A whest
row holds ``depth * width**2`` weights — 2,097,152 of them at the competition
geometry — and parquet spends ~570 bytes per column on page headers, column-chunk
metadata, and the column name, a cost per *column* that is independent of the
number of rows.  A 100 MB file is therefore exhausted by ~175,000 columns before a
single row is written, and past ~262,000 columns pyarrow writes a file it can no
longer read back ("TProtocolException: Exceeded size limit" while deserialising
the schema).  Compression does not help: He weights are maximum-entropy for their
variance, and byte-shuffle + LZMA reaches only 0.845x.

So the whest family stores plain ``.npy`` arrays (``storage: "npy"`` in its
metadata) and is reachable **only** through
:func:`generatedata.load_data.load_data_as_sequence`, which is the honest access
path anyway: the forward pass is an *ordered* product of operators, so the layer
axis is a sequence axis, not an exchangeable feature axis.  ``.npy`` also
memory-maps, so a 471 MB dataset costs no RAM until rows are touched.

Files written per dataset ``<name>``::

    <name>_weights.npy          (num_points, depth, width, width)  float32
    <name>_all_layer_means.npy  (num_points, depth, width)         float32
    <name>_final_means.npy      (num_points, width)                float32
    <name>_ut_fixed.npy         (num_points, width)                float32
    <name>_info.json

``final_means`` duplicates ``all_layer_means[:, -1]`` (as the official data does);
it costs ``4 * width`` bytes per row and saves every consumer an indexing rule.
``ut_fixed`` is a cheap deterministic estimate of ``F``, so ``final_means -
ut_fixed`` is the residual ``R(W)`` that a corrector has to learn.

Per-layer means are **labels, never inputs.** The estimator contract hands
``predict(mlp, budget)`` only ``width``, ``depth``, ``weights``, ``seed`` and
``name`` — no means of any kind — and requires it to *return* the ``(depth,
width)`` stack.  Feeding per-layer means in as features would collapse 32 layers
of error compounding into one and destroy the problem, which is why
``load_data_as_sequence`` refuses ``label_every_step=True`` for this family.
"""

import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from generatedata.hf_data import download_hf_parquet

# --------------------------------------------------------------------------- #
#  Monte-Carlo budgets                                                        #
# --------------------------------------------------------------------------- #

# Draws per network for the core dataset, which `generate_all` builds by default
# and therefore every test session pays for once.  10,000 networks at width 8 cost
# ~8 s on a GPU and ~8 min CPU-only at this value; raising it is what the gated
# tier is for.  Label noise here is ~1.5e-6 per neuron against Var(F) ~ 0.5.
WHEST_MC_SAMPLES = 262_144

# Draws per network for the gated tiers, where generation is opt-in and can take
# tens of minutes.  Label noise lands at 1e-8..3e-8 — about 100x below the error
# of the estimator being measured, and 100x above the official 1e9-sample floor of
# ~5e-11.
WHEST_MC_SAMPLES_GATED = 16_777_216

# Draws pushed through the networks per pass.  Bounds memory only; the estimate is
# the same for any value.
_SAMPLE_CHUNK = 16_384

# Activations held on the device at once, in floats, used to size the net chunk.
_DEVICE_FLOATS = 64_000_000


# --------------------------------------------------------------------------- #
#  The official published data                                                #
# --------------------------------------------------------------------------- #

# aicrowd/arc-whestbench-public-2026, licensed CC-BY-4.0.  Pinned to commit
# hashes rather than the `v1-phase1` / `v1-warmup` tags, matching how every other
# HuggingFace source in this library is pinned.
#
# Only the `mini` splits are used.  Both official rungs and the extra-large rung
# are drawn from them using disjoint row ranges, which keeps our datasets mutually
# disjoint while requiring ~1.07 GB of download instead of the 8.6 GB `full`
# split.  All 1,100 published networks are iid draws, so row ranges within `mini`
# are as independent as crossing splits would be.
WHEST_OFFICIAL_REPO = "aicrowd/arc-whestbench-public-2026"
WHEST_OFFICIAL_LICENSE = "cc-by-4.0"
WHEST_OFFICIAL_HOMEPAGE = (
    "https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026"
)
WHEST_OFFICIAL_GROUND_TRUTH_SAMPLES = 1_000_000_000

WHEST_OFFICIAL_SOURCES = {
    "phase1": {
        "tag": "v1-phase1",
        "revision": "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7",
        "width": 256,
        "depth": 32,
        "split": "mini",
        "num_available": 100,
        "shards": (
            "data/mini-00000-of-00003.parquet",
            "data/mini-00001-of-00003.parquet",
            "data/mini-00002-of-00003.parquet",
        ),
    },
    "warmup": {
        "tag": "v1-warmup",
        "revision": "59e261dc1ad56ba1c04212832c9be37afcd09eaf",
        "width": 256,
        "depth": 8,
        "split": "mini",
        "num_available": 100,
        "shards": ("data/mini-00000-of-00001.parquet",),
    },
}


# --------------------------------------------------------------------------- #
#  The dataset ladder                                                         #
# --------------------------------------------------------------------------- #

WHEST_BUDGET_BYTES = 100_000_000        # per dataset, for the grid
WHEST_XL_BUDGET_BYTES = 500_000_000     # per dataset, for the opt-in extra-large pair


def whest_dataset_bytes(width: int, depth: int, num_points: int) -> int:
    """Total bytes a whest dataset occupies: weights + the three label arrays.

    ``4 * (depth*width**2 + depth*width + 2*width)`` per network — float32
    weights, the per-layer mean stack, the final-layer means, and the cheap UT
    estimate.  Storage is raw bytes, so this is exact up to the ~500 byte ``.npy``
    headers and the metadata JSON.
    """
    per_net = depth * width * width + depth * width + 2 * width
    return 4 * per_net * num_points


# Each rung is one dataset.  `official` names a key of WHEST_OFFICIAL_SOURCES and
# `row_start` the first row taken from that split; `mc_samples` applies only to
# the rungs we bake ourselves.
#
# Depth is the regime variable and width is merely the dimension: the fraction of
# coordinates with F_j = 0 exactly is ~2% at depth 8, ~12-18% at depth 16 and
# ~25-33% at depth 32, nearly independent of width above width 32 (the cone's
# per-layer survival probability self-averages).  So the ladder holds depth fixed
# at the competition's 32 and sweeps width, with a depth-8 branch to isolate depth
# at fixed width.  Width 8 at depth 32 is deliberately absent: half its
# coordinates and 5% of whole networks are dead, which makes it degenerate.
WHEST_GRID = (
    # depth-32 spine: the competition regime, width as the only variable
    dict(width=256, depth=32, num_points=11, official="phase1", row_start=0),
    dict(width=128, depth=32, num_points=44, mc_samples=WHEST_MC_SAMPLES_GATED),
    dict(width=64, depth=32, num_points=178, mc_samples=WHEST_MC_SAMPLES_GATED),
    dict(width=32, depth=32, num_points=701, mc_samples=WHEST_MC_SAMPLES_GATED),
    dict(width=16, depth=32, num_points=2_700, mc_samples=WHEST_MC_SAMPLES_GATED),
    # depth-8 branch: isolates depth, and affordable at training scale
    dict(width=256, depth=8, num_points=45, official="warmup", row_start=0),
    dict(width=16, depth=8, num_points=10_000, mc_samples=WHEST_MC_SAMPLES_GATED),
    dict(width=8, depth=8, num_points=10_000, mc_samples=WHEST_MC_SAMPLES),
)

# The opt-in pair, at five times the budget.  They exist because the grid is
# statistically thin exactly where the problem is hardest: 11 networks give ~30%
# standard error on a measured MSE, 44 give ~15%.  These raise the two largest
# widths to ~13% and ~7%.  The 256x32 rung continues the official mini split from
# where the grid rung stopped, so the two are disjoint.
WHEST_XL = (
    dict(width=256, depth=32, num_points=56, official="phase1", row_start=11,
         suffix="_xl"),
    dict(width=128, depth=32, num_points=224, mc_samples=WHEST_MC_SAMPLES_GATED,
         suffix="_xl"),
)

# The one rung `generate_all` builds without being asked: cheap, needs no network,
# and keeps the .npy storage path and its loader guards under test on every run.
WHEST_CORE_SPEC = WHEST_GRID[-1]


def whest_name(width: int, depth: int, suffix: str = "") -> str:
    """Dataset name for a geometry, e.g. ``whest_w256_d32`` or ``whest_w128_d32_xl``."""
    return f"whest_w{width}_d{depth}{suffix}"


# --------------------------------------------------------------------------- #
#  Competition-faithful sampling and ground truth                             #
# --------------------------------------------------------------------------- #

def he_weights(num_points: int, width: int, depth: int, seed: int) -> torch.Tensor:
    """Draw ``num_points`` independent competition-faithful MLPs.

    Args:
        num_points: Number of networks.
        width: Layer width; every weight matrix is ``width x width``.
        depth: Number of layers.
        seed: Seed for the torch generator.

    Returns:
        Tensor of shape ``(num_points, depth, width, width)`` with iid
        ``N(0, 2/width)`` entries.  Filled in place so peak memory is one copy.
    """
    generator = torch.Generator().manual_seed(seed)
    weights = torch.empty(num_points, depth, width, width)
    weights.normal_(0.0, math.sqrt(2.0 / width), generator=generator)
    return weights


def _propagate(z: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Push a batch of inputs through a batch of networks: ``z <- relu(z @ W_l)``.

    Args:
        z: Inputs, shape ``(nets, points, width)``.
        weights: Weights, shape ``(nets, depth, width, width)``.

    Returns:
        Final-layer activations, shape ``(nets, points, width)``.
    """
    for layer in range(weights.shape[1]):
        z = torch.relu(torch.einsum("npi,nij->npj", z, weights[:, layer]))
    return z


def _net_chunk(width: int) -> int:
    """Networks to hold on the device at once, bounding activation memory."""
    return max(1, _DEVICE_FLOATS // (_SAMPLE_CHUNK * width))


def mc_layer_means(
    weights: torch.Tensor,
    mc_samples: int,
    seed: int,
    device: str,
    net_chunk: int | None = None,
    sample_chunk: int = _SAMPLE_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monte-Carlo estimate of the per-layer mean activations, plus second moments.

    Averages every layer's post-ReLU activation over ``mc_samples`` draws of
    ``x ~ N(0, I)``.  Sums accumulate in float64 so the float32 forward pass is
    the only source of rounding, and the estimate's own standard error is
    ``sqrt(Var_x / mc_samples)`` per coordinate, recoverable from the returned
    second moment.

    Args:
        weights: Shape ``(num_points, depth, width, width)``.
        mc_samples: Draws of ``x`` per network.
        seed: Seed for the sampling generator.
        device: Torch device on which to run the forward passes.
        net_chunk: Networks held on the device at once; defaults to a
            memory-bounded choice.
        sample_chunk: Draws processed per pass (bounds memory, not accuracy).

    Returns:
        ``(means, second_moments)``, both float64 of shape
        ``(num_points, depth, width)`` on the CPU.
    """
    num_points, depth, width, _ = weights.shape
    net_chunk = _net_chunk(width) if net_chunk is None else net_chunk
    generator = torch.Generator(device=device).manual_seed(seed)
    means, second_moments = [], []
    for start in range(0, num_points, net_chunk):
        chunk = weights[start:start + net_chunk].to(device)
        nets = chunk.shape[0]
        total = torch.zeros(nets, depth, width, device=device, dtype=torch.float64)
        total_sq = torch.zeros(nets, depth, width, device=device, dtype=torch.float64)
        drawn = 0
        while drawn < mc_samples:
            points = min(sample_chunk, mc_samples - drawn)
            z = torch.randn(nets, points, width, generator=generator, device=device)
            for layer in range(depth):
                z = torch.relu(torch.einsum("npi,nij->npj", z, chunk[:, layer]))
                total[:, layer] += z.sum(dim=1, dtype=torch.float64)
                total_sq[:, layer] += (z * z).sum(dim=1, dtype=torch.float64)
            drawn += points
            del z
        means.append((total / mc_samples).cpu())
        second_moments.append((total_sq / mc_samples).cpu())
        del chunk, total, total_sq
    return torch.cat(means), torch.cat(second_moments)


def ut_fixed_final_mean(
    weights: torch.Tensor, device: str, net_chunk: int | None = None
) -> torch.Tensor:
    """Fixed-quadrature unscented-transform estimate of ``F(W)``.

    Represents ``N(0, I_width)`` by the ``2*width`` sigma points ``+-r e_i`` — the
    rows of ``[r I; -r I]`` — propagates them through the network and averages.
    This symmetric set matches the input mean and covariance when
    ``r = sqrt(width)``, but the quantity of interest is a *first* radial moment:
    ``relu`` chains are positively homogeneous of degree one, so ``F(c x) = c
    F(x)`` and the right shell radius is

        r = E||x|| = sqrt(2) * Gamma((width+1)/2) / Gamma(width/2),

    which removes the ``sqrt(width)/E||x|| ~ 1 + 1/(4 width)`` bias that compounds
    with depth.  No random rotations are used, so this is a deterministic function
    of ``W`` and costs ``2*width`` forward passes rather than ``mc_samples``.

    Args:
        weights: Shape ``(num_points, depth, width, width)``.
        device: Torch device on which to run the forward passes.
        net_chunk: Networks held on the device at once.

    Returns:
        Float32 tensor of shape ``(num_points, width)`` on the CPU.
    """
    num_points, _, width, _ = weights.shape
    net_chunk = _net_chunk(width) if net_chunk is None else net_chunk
    radius = math.sqrt(2.0) * math.exp(
        math.lgamma((width + 1) / 2) - math.lgamma(width / 2)
    )
    axes = torch.eye(width, device=device)
    sigma_points = radius * torch.cat([axes, -axes], dim=0)      # (2*width, width)
    estimates = []
    for start in range(0, num_points, net_chunk):
        chunk = weights[start:start + net_chunk].to(device)
        z = sigma_points.expand(chunk.shape[0], 2 * width, width)
        estimates.append(_propagate(z, chunk).mean(dim=1).cpu())
        del chunk
    return torch.cat(estimates)


# --------------------------------------------------------------------------- #
#  Storage                                                                    #
# --------------------------------------------------------------------------- #

WHEST_ARRAYS = ("weights", "all_layer_means", "final_means", "ut_fixed")


def whest_array_path(data_dir: Path | str, name: str, array: str) -> Path:
    """Path of one of a whest dataset's ``.npy`` arrays."""
    return Path(data_dir) / f"{name}_{array}.npy"


def _save_npy(path: Path, array: np.ndarray) -> None:
    """Write one array, renaming into place only once it is complete.

    These files reach hundreds of megabytes, and an interrupted write would
    otherwise leave a truncated ``.npy`` that ``dataset_exists`` accepts as
    present — so generation would skip it forever while every load failed.
    """
    partial = path.with_name(path.name + ".partial")
    # Write through a file handle: given a path, np.save appends ".npy" to any name
    # that lacks it, which would silently produce "<name>.npy.partial.npy".
    with open(partial, "wb") as handle:
        np.save(handle, array, allow_pickle=False)
    partial.rename(path)


def _save_whest(
    data_dir: Path,
    name: str,
    weights: np.ndarray,
    all_layer_means: np.ndarray,
    ut_fixed: np.ndarray,
    info: dict,
) -> None:
    """Write a whest dataset's arrays and metadata.

    Args:
        data_dir: Output directory.
        name: Dataset name, used as the file prefix.
        weights: ``(num_points, depth, width, width)`` float32.
        all_layer_means: ``(num_points, depth, width)`` float32.
        ut_fixed: ``(num_points, width)`` float32.
        info: Metadata; the shape-derived keys are filled in here.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    num_points, depth, width, _ = weights.shape

    # The same keys the flat datasets record, so `load_data_as_sequence` computes
    # seq_len = x_y_index // step_size exactly as it does for every other dataset.
    x_y_index = depth * width * width
    info = {
        "num_points": num_points,
        "size": x_y_index + width,
        "x_y_index": x_y_index,
        "x_size": x_y_index,
        "y_size": width,
        "onehot_y": 0,
        "storage": "npy",
        "data_family": "whest",
        "width": width,
        "depth": depth,
        "default_step_size": width * width,
        "forward_convention": "z <- relu(z @ W_l), no biases",
        "flatten_order": "layer-major: W[layer, row, column]",
        "label_every_step_allowed": False,
        **info,
    }

    _save_npy(whest_array_path(data_dir, name, "weights"), weights)
    _save_npy(whest_array_path(data_dir, name, "all_layer_means"), all_layer_means)
    _save_npy(whest_array_path(data_dir, name, "final_means"),
              np.ascontiguousarray(all_layer_means[:, -1]))
    _save_npy(whest_array_path(data_dir, name, "ut_fixed"), ut_fixed)
    with open(data_dir / f"{name}_info.json", "w") as f:
        json.dump(info, f, indent=4)


# --------------------------------------------------------------------------- #
#  Datasets we bake ourselves                                                 #
# --------------------------------------------------------------------------- #

def generate_whest(
    data_dir: Path,
    width: int = 8,
    depth: int = 8,
    num_points: int = 10_000,
    mc_samples: int = WHEST_MC_SAMPLES,
    seed: int = 42,
    suffix: str = "",
) -> None:
    """Generate a whest dataset by sampling networks and estimating their means.

    Args:
        data_dir: Output directory.
        width: Layer width (also the label dimension).
        depth: Number of layers.
        num_points: Number of networks.
        mc_samples: Monte-Carlo draws per network for the ground-truth labels.
        seed: Base seed; weights use ``seed`` and the Monte-Carlo draws ``seed+1``.
            The weights are exactly reproducible; the labels are reproducible only
            up to the RNG stream, which differs between CPU and CUDA, so their
            residual noise is recorded as ``label_mc_se2``.
        suffix: Appended to the dataset name, e.g. ``"_xl"``.

    Note:
        Deep narrow ReLU nets die: a layer keeps an input only if some coordinate
        stays positive, and because ``relu`` is positively homogeneous no
        rescaling of the weights changes which coordinates those are.  The
        realised fraction of exactly-zero coordinates is recorded as
        ``dead_coordinate_fraction`` and of wholly dead networks as
        ``dead_network_fraction``.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TF32 truncates matmul inputs to 10 mantissa bits, a ~1e-3 relative error per
    # layer -- larger than the Monte-Carlo noise being paid for.  Restore the
    # caller's setting afterwards, since this is a library.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        weights = he_weights(num_points, width, depth, seed)
        means, second_moment = mc_layer_means(weights, mc_samples, seed + 1, device)
        ut_fixed = ut_fixed_final_mean(weights, device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    final = means[:, -1]
    _save_whest(
        data_dir,
        whest_name(width, depth, suffix),
        weights.numpy(),
        means.float().numpy(),
        ut_fixed.numpy(),
        info={
            "source": "generated",
            "seed": seed,
            "mc_samples": mc_samples,
            "label_mc_se2": ((second_moment[:, -1] - final ** 2) / mc_samples).mean().item(),
            "ut_final_layer_mse": ((ut_fixed.double() - final) ** 2).mean().item(),
            "dead_coordinate_fraction": (final == 0).double().mean().item(),
            "dead_network_fraction": (final.sum(dim=1) == 0).double().mean().item(),
        },
    )


# --------------------------------------------------------------------------- #
#  The official published data                                                #
# --------------------------------------------------------------------------- #

def _unnest(column, levels: int) -> np.ndarray:
    """Flatten a nested Arrow list column into a contiguous numpy array.

    The official data stores ``weights`` as a single ``Array3D`` column — one
    ``[depth, width, width]`` array per row, 8.4 MB at the competition geometry —
    rather than as scalar columns.  ``flatten()`` peels one list level at a time
    and respects the slice offsets of a sliced array.
    """
    array = column.combine_chunks()
    for _ in range(levels):
        array = array.flatten()
    return array.to_numpy(zero_copy_only=False)


def _read_official_rows(cache_dir: Path, key: str, row_start: int, num_points: int) -> dict:
    """Read a contiguous row range from a pinned official split.

    Shards are fetched only until the requested range is covered, so the grid's
    first 11 networks need one 309 MB shard rather than the whole split.

    Args:
        cache_dir: Directory for the cached parquet downloads.
        key: Key of :data:`WHEST_OFFICIAL_SOURCES`.
        row_start: First row of the split to take.
        num_points: Number of rows to take.

    Returns:
        Dict of numpy arrays plus the per-network identifiers.
    """
    source = WHEST_OFFICIAL_SOURCES[key]
    width, depth = source["width"], source["depth"]
    row_stop = row_start + num_points
    if row_stop > source["num_available"]:
        raise ValueError(
            f"official split '{key}/{source['split']}' has {source['num_available']} "
            f"networks; rows {row_start}..{row_stop} were requested"
        )

    pieces: list[dict] = []
    offset = 0
    for shard in source["shards"]:
        if offset >= row_stop:
            break
        path = download_hf_parquet(
            cache_dir, WHEST_OFFICIAL_REPO, source["revision"], shard
        )
        table = pq.read_table(path)
        rows = table.num_rows
        lo, hi = max(row_start, offset), min(row_stop, offset + rows)
        if lo < hi:
            piece = table.slice(lo - offset, hi - lo)
            pieces.append({
                "weights": _unnest(piece.column("weights"), 3).reshape(-1, depth, width, width),
                "all_layer_means": _unnest(piece.column("all_layer_means"), 2).reshape(-1, depth, width),
                "mlp_id": np.asarray(piece.column("mlp_id").to_pylist(), dtype=np.int32),
                "mlp_seed": np.asarray(piece.column("mlp_seed").to_pylist(), dtype=np.int64),
                "mlp_name": piece.column("mlp_name").to_pylist(),
                "avg_variance": np.asarray(piece.column("avg_variance").to_pylist(), dtype=np.float64),
            })
        offset += rows
        del table

    return {
        "weights": np.concatenate([p["weights"] for p in pieces]),
        "all_layer_means": np.concatenate([p["all_layer_means"] for p in pieces]),
        "mlp_id": np.concatenate([p["mlp_id"] for p in pieces]),
        "mlp_seed": np.concatenate([p["mlp_seed"] for p in pieces]),
        "mlp_name": [n for p in pieces for n in p["mlp_name"]],
        "avg_variance": np.concatenate([p["avg_variance"] for p in pieces]),
    }


def generate_whest_official(
    data_dir: Path,
    key: str = "phase1",
    num_points: int = 11,
    row_start: int = 0,
    suffix: str = "",
) -> None:
    """Build a whest dataset from the officially published competition data.

    Downloads a pinned revision of ``aicrowd/arc-whestbench-public-2026`` and
    keeps a contiguous row range of its ``mini`` split.  The labels are the
    organisers' own ground truth at 1,000,000,000 samples per network — a noise
    floor of ~5e-11, which nothing we can bake locally approaches — so these are
    the reference rungs of the ladder.  The cheap ``UT_fixed`` estimate is
    computed here from the published weights.

    The data is CC-BY-4.0; the required attribution is recorded in the dataset's
    metadata (``source``, ``source_repo``, ``source_revision``, ``license``).

    Args:
        data_dir: Output directory.
        key: Which official geometry: ``"phase1"`` (256x32) or ``"warmup"`` (256x8).
        num_points: Number of networks to keep.
        row_start: First row of the split to take, so that several datasets can be
            carved out of one split without overlapping.
        suffix: Appended to the dataset name, e.g. ``"_xl"``.
    """
    source = WHEST_OFFICIAL_SOURCES[key]
    width, depth = source["width"], source["depth"]
    cache_dir = Path(data_dir).parent.parent / "data" / "external" / "whestbench"
    rows = _read_official_rows(cache_dir, key, row_start, num_points)

    weights = torch.from_numpy(rows["weights"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ut_fixed = ut_fixed_final_mean(weights, device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    final = torch.from_numpy(rows["all_layer_means"][:, -1]).double()
    name = whest_name(width, depth, suffix)
    _save_npy(whest_array_path(data_dir, name, "mlp_ids"), rows["mlp_id"])
    _save_npy(whest_array_path(data_dir, name, "mlp_seeds"), rows["mlp_seed"])
    _save_whest(
        data_dir,
        name,
        rows["weights"],
        rows["all_layer_means"],
        ut_fixed.numpy(),
        info={
            "source": f"official:{key}",
            "source_repo": WHEST_OFFICIAL_REPO,
            "source_revision": source["revision"],
            "source_tag": source["tag"],
            "source_split": source["split"],
            "source_rows": [row_start, row_start + num_points],
            "license": WHEST_OFFICIAL_LICENSE,
            "homepage": WHEST_OFFICIAL_HOMEPAGE,
            "mc_samples": WHEST_OFFICIAL_GROUND_TRUTH_SAMPLES,
            # The organisers' own per-network activation variance gives the exact
            # noise floor of their bake: avg_variance / n_samples.
            "label_mc_se2": float(rows["avg_variance"].mean()) / WHEST_OFFICIAL_GROUND_TRUTH_SAMPLES,
            "ut_final_layer_mse": ((ut_fixed.double() - final) ** 2).mean().item(),
            "dead_coordinate_fraction": float((final == 0).double().mean()),
            "dead_network_fraction": float((final.sum(dim=1) == 0).double().mean()),
            "mlp_names": rows["mlp_name"],
            "statistical_note": (
                f"{num_points} networks give ~{100 / math.sqrt(num_points):.0f}% standard error on a "
                "measured final-layer MSE. For a properly powered evaluation use the "
                "full 1,000-network official split directly from the HuggingFace Hub."
            ),
        },
    )


# --------------------------------------------------------------------------- #
#  The gated sweep                                                            #
# --------------------------------------------------------------------------- #

def generate_whest_dataset(data_dir: Path, spec: dict) -> None:
    """Generate the single dataset described by a ladder entry."""
    spec = dict(spec)
    official = spec.pop("official", None)
    if official is not None:
        generate_whest_official(
            data_dir,
            key=official,
            num_points=spec["num_points"],
            row_start=spec.get("row_start", 0),
            suffix=spec.get("suffix", ""),
        )
    else:
        generate_whest(
            data_dir,
            width=spec["width"],
            depth=spec["depth"],
            num_points=spec["num_points"],
            mc_samples=spec["mc_samples"],
            suffix=spec.get("suffix", ""),
        )


def whest_expected_params(spec: dict) -> dict:
    """Metadata a already-generated dataset must match to be considered current."""
    params = {
        "num_points": spec["num_points"],
        "width": spec["width"],
        "depth": spec["depth"],
    }
    if spec.get("official") is None:
        params["mc_samples"] = spec["mc_samples"]
    else:
        params["source"] = f"official:{spec['official']}"
        params["source_rows"] = [spec["row_start"], spec["row_start"] + spec["num_points"]]
    return params


def generate_whest_all(data_dir: Path | str, include_xl: bool = False) -> None:
    """Generate the whest ladder.  Opt-in: nothing here runs from ``generate_all``.

    This tier is gated because, unlike the rest of the library, it wants a GPU,
    downloads about a gigabyte from the HuggingFace Hub, and writes hundreds of
    megabytes.  Datasets that already exist with matching parameters are skipped,
    so re-running only fills in what is missing.

    Args:
        data_dir: Output directory (normally ``data/processed``).
        include_xl: Also build the two extra-large datasets, which are five times
            the per-dataset budget and take tens of minutes to bake.
    """
    from generatedata.data_generators import compile_info_json, dataset_exists

    data_dir = Path(data_dir)
    specs = list(WHEST_GRID) + (list(WHEST_XL) if include_xl else [])

    print(f"whest ladder: {len(specs)} datasets")
    total = 0
    for spec in specs:
        name = whest_name(spec["width"], spec["depth"], spec.get("suffix", ""))
        size = whest_dataset_bytes(spec["width"], spec["depth"], spec["num_points"])
        total += size
        origin = spec.get("official", "generated")
        print(f"  {name:22s} {spec['num_points']:>6d} nets  {size / 1e6:7.1f} MB  {origin}")
    print(f"  {'total':22s} {'':>6s}       {total / 1e6:7.1f} MB  -> {data_dir}")
    if not include_xl:
        print("  (pass include_xl=True for the two 500 MB datasets)")

    for spec in specs:
        name = whest_name(spec["width"], spec["depth"], spec.get("suffix", ""))
        if dataset_exists(data_dir, name, whest_expected_params(spec)):
            print(f"Skipping {name} (already exists)")
        else:
            print(f"Generating {name} ...", flush=True)
            generate_whest_dataset(data_dir, spec)

    compile_info_json(data_dir)
