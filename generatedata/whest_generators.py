"""
ARC White-Box Estimation Challenge (whest) dataset generators.

The task: given only the weights of a deep ReLU MLP, predict the mean activation
of its final layer under a standard normal input.  Writing the forward map of one
network as a product of nonlinear layers,

    z_0 = x,        z_l = relu(z_{l-1} W_l),        l = 1 .. depth,

the target is the deterministic functional

    F(W) = E_{x ~ N(0, I_width)} [ z_depth ]  in  R^width.

The input distribution is integrated out, so ``F`` is a function of ``W`` alone —
a map ``R^(depth x width x width) -> R^width``.  Viewed as a dynamical system,
each network is a product of random matrices acting on a distribution, and
``F(W)`` is a functional of its terminal distribution; its heavy tail across
random ``W`` is the large-deviation statistics of the finite-time Lyapunov
exponent (expanding vs contracting products), not noise.

Competition conventions reproduced here (these matter — see the ``whest``
project's ``experiments/PRINCIPLES.md``):

  * weights are iid ``N(0, 2/width)`` (He initialisation at fan_in = width),
    with **no biases**;
  * the forward map is ``z <- relu(z @ W_l)`` — the row-vector convention, i.e.
    ``@W`` and *not* ``@W.T``;
  * ``relu`` is applied at every layer, including the last;
  * accuracy is judged as raw mean-squared error on ``F``, so the ground truth is
    accumulated in float64 with TF32 disabled.

Dataset layout (one row per network, the library's usual flat format)::

    row     = [ flatten(W) | mu ]                       size = depth*width^2 + width
    target  = [ flatten(W) | F(W)          ]            the Monte-Carlo ground truth
    start   = [ flatten(W) | UT_fixed(W)   ]            a cheap deterministic estimate

so that ``target - start`` is the residual ``R(W) = F(W) - UT_fixed(W)`` of the
cheap estimator.  Because the sigma-point set below is *fixed* (no random
rotations), ``UT_fixed`` is itself a deterministic function of ``W``, hence so is
``R``: correcting the cheap estimator is a well-posed regression problem, not a
variance-reduction problem.

``flatten(W)`` is in layer-major (C) order, so the metadata records
``default_step_size = width**2`` and ``load_data_as_sequence`` returns shape
``(num_points, depth, width**2)`` — one ``width x width`` matrix per timestep, in
layer order, which is the natural input for a recurrent model of the layer
dynamics.
"""

import math
from pathlib import Path

import torch

from generatedata.save_data import save_data

# Monte-Carlo draws per network for the ground-truth label.  At this value the
# label's own standard error is ~1e-3 per neuron (variance ~1.7e-6), three orders
# of magnitude below Var(F) ~ 0.5, while costing seconds per 10k networks on a
# GPU and a few minutes on a CPU.  Exposed as a constant so that `generate_all`
# can record it as the expected parameter without duplicating the number.
WHEST_MC_SAMPLES = 262_144

# Monte-Carlo draws pushed through the networks per pass.  Bounds memory only; the
# estimate is the same for any value.
_SAMPLE_CHUNK = 16_384


def he_weights(num_points: int, width: int, depth: int, seed: int) -> torch.Tensor:
    """Draw ``num_points`` independent competition-faithful MLPs.

    Args:
        num_points: Number of networks (dataset rows).
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


def mc_final_mean(
    weights: torch.Tensor,
    mc_samples: int,
    seed: int,
    device: str,
    net_chunk: int,
    sample_chunk: int = _SAMPLE_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monte-Carlo estimate of ``F(W)``, plus its second moment.

    Averages the final-layer activation over ``mc_samples`` draws of
    ``x ~ N(0, I)``.  Sums are accumulated in float64 so the float32 forward pass
    is the only source of rounding, and the estimate's own standard error is
    ``sqrt(Var_x / mc_samples)`` per neuron, obtainable from the returned second
    moment.

    Args:
        weights: Shape ``(num_points, depth, width, width)``.
        mc_samples: Draws of ``x`` per network.
        seed: Seed for the sampling generator.
        device: Torch device on which to run the forward passes.
        net_chunk: Networks held on the device at once.
        sample_chunk: Draws processed per pass (bounds memory, not accuracy).

    Returns:
        ``(mean, second_moment)``, both float64 of shape ``(num_points, width)``
        on the CPU: the per-neuron mean and mean-square of the final activation.
    """
    num_points, _, width, _ = weights.shape
    generator = torch.Generator(device=device).manual_seed(seed)
    means, second_moments = [], []
    for start in range(0, num_points, net_chunk):
        chunk = weights[start:start + net_chunk].to(device)
        nets = chunk.shape[0]
        total = torch.zeros(nets, width, device=device, dtype=torch.float64)
        total_sq = torch.zeros(nets, width, device=device, dtype=torch.float64)
        drawn = 0
        while drawn < mc_samples:
            points = min(sample_chunk, mc_samples - drawn)
            z = torch.randn(nets, points, width, generator=generator, device=device)
            z = _propagate(z, chunk)
            total += z.sum(dim=1, dtype=torch.float64)
            total_sq += (z * z).sum(dim=1, dtype=torch.float64)
            drawn += points
            del z
        means.append((total / mc_samples).cpu())
        second_moments.append((total_sq / mc_samples).cpu())
    return torch.cat(means), torch.cat(second_moments)


def ut_fixed_final_mean(
    weights: torch.Tensor, device: str, net_chunk: int
) -> torch.Tensor:
    """Fixed-quadrature unscented-transform estimate of ``F(W)``.

    Represents ``N(0, I_width)`` by the ``2*width`` sigma points ``+-r e_i`` — the
    columns of ``[r I; -r I]`` — propagates them through the network and averages.
    This symmetric set matches the input mean and covariance when
    ``r = sqrt(width)``, but the quantity of interest here is a *first* radial
    moment: ``relu`` chains are positively homogeneous of degree one, so
    ``F(c x) = c F(x)`` and the right shell radius is

        r = E||x|| = sqrt(2) * Gamma((width+1)/2) / Gamma(width/2),

    which removes the ``sqrt(width)/E||x|| ~ 1 + 1/(4 width)`` bias that
    compounds with depth.

    No random rotations are used, so this is a deterministic function of ``W``
    and costs ``2*width`` forward passes instead of ``mc_samples``.

    Args:
        weights: Shape ``(num_points, depth, width, width)``.
        device: Torch device on which to run the forward passes.
        net_chunk: Networks held on the device at once.

    Returns:
        Float32 tensor of shape ``(num_points, width)`` on the CPU.
    """
    num_points, _, width, _ = weights.shape
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
    return torch.cat(estimates)


def generate_whest(
    data_dir: Path,
    width: int = 8,
    depth: int = 8,
    num_points: int = 10_000,
    mc_samples: int = WHEST_MC_SAMPLES,
    seed: int = 42,
) -> None:
    """Generate a whest dataset: predict a ReLU MLP's final-layer mean from its weights.

    Each row is one random network: the features are its flattened weights and
    the labels are the ``width`` final-layer mean activations.  The target labels
    are the Monte-Carlo ground truth ``F(W)``; the start labels are the cheap
    deterministic ``UT_fixed(W)``, so the start-to-target displacement is the
    estimator's residual.

    Args:
        data_dir: Output directory for parquet / JSON files.
        width: Layer width (also the label dimension).
        depth: Number of layers.
        num_points: Number of networks to generate.
        mc_samples: Monte-Carlo draws per network for the ground-truth label.
        seed: Base seed; the weights use ``seed`` and the Monte-Carlo draws
            ``seed + 1``.  The weights are therefore exactly reproducible, while
            the labels are reproducible only up to the RNG stream, which differs
            between CPU and CUDA; their residual noise is recorded as
            ``label_mc_se2``.

    Note:
        Deep narrow ReLU nets die: a layer keeps an input only if some coordinate
        stays positive, and because ``relu`` is positively homogeneous no
        rescaling of the weights can change which coordinates those are.  At
        ``width = 2`` the surviving fraction halves every couple of layers and a
        depth-32 net has ``F == 0`` identically; ``width >= 8`` keeps the defaults
        here alive.  The realised fraction of dead networks is recorded as
        ``dead_fraction`` in the metadata.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TF32 truncates matmul inputs to 10 mantissa bits, a ~1e-3 relative error
    # per layer -- larger than the Monte-Carlo noise we are paying for.  Restore
    # the caller's setting afterwards, since this is a library.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        weights = he_weights(num_points, width, depth, seed)
        # Bound activations held on the device to ~64M floats per chunk.
        net_chunk = max(1, 64_000_000 // (_SAMPLE_CHUNK * width))
        truth, second_moment = mc_final_mean(
            weights, mc_samples, seed + 1, device, net_chunk
        )
        cheap = ut_fixed_final_mean(weights, device, net_chunk)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    features = weights.reshape(num_points, depth * width * width)
    target = torch.cat([features, truth.float()], dim=1)
    start = torch.cat([features, cheap], dim=1)

    # Diagnostics that let a user judge what the labels can support: the
    # Monte-Carlo variance of the labels themselves, and the raw MSE of the cheap
    # estimator that the start data encodes (the number a learned model must beat).
    label_mc_se2 = ((second_moment - truth ** 2) / mc_samples).mean().item()
    ut_final_layer_mse = ((cheap.double() - truth) ** 2).mean().item()
    dead_fraction = (truth.sum(dim=1) == 0).double().mean().item()

    x_y_index = depth * width * width
    total_columns = x_y_index + width
    start_data = {f"x{i}": start[:, i] for i in range(total_columns)}
    target_data = {f"x{i}": target[:, i] for i in range(total_columns)}

    save_data(
        data_dir,
        f"whest_w{width}_d{depth}",
        start_data,
        target_data,
        x_y_index=x_y_index,
        onehot_y=False,
        additional_info={
            "data_family": "whest",
            "width": width,
            "depth": depth,
            "weight_std": math.sqrt(2.0 / width),
            "forward_convention": "z <- relu(z @ W_l), no biases",
            "flatten_order": "layer-major: W[layer, row, column]",
            "default_step_size": width * width,
            "mc_samples": mc_samples,
            "label_mc_se2": label_mc_se2,
            "ut_final_layer_mse": ut_final_layer_mse,
            "dead_fraction": dead_fraction,
            "seed": seed,
        },
    )
