---
license: bsd-3-clause
tags:
  - dynamical-systems
  - benchmark
  - synthetic
---

# generatedata

Datasets produced by the [`generatedata`](https://github.com/rcpaffenroth/generatedata)
library. Every dataset is a **start/target pair**: `start` holds perturbed or
noisy points, `target` holds the corresponding clean points on the manifold of
interest. The pair defines a map — denoising, projection, or the flow of a
dynamical system — and the library exists to make that map the same shape across
very different data sources.

## Loading

```python
from generatedata import load_data

load_data.data_names()               # what is in this snapshot
data = load_data.load_data('MNIST')  # {'info': ..., 'start': df, 'target': df}
X, Y = load_data.load_data_as_xy('MNIST')
```

Nothing here requires a HuggingFace account or token.

## Layout

The repository is flat, and `info.json` is the index — a dict from dataset name
to that dataset's metadata (shapes, generating parameters, seeds, provenance).
Reading it is how the library discovers what exists, so an unreferenced file is
invisible to `load_data`.

| File | Contents |
| --- | --- |
| `info.json` | index: dataset name → metadata |
| `<name>_start.parquet` | start points, one row per sample |
| `<name>_target.parquet` | target points, one row per sample |
| `<name>_info.json` | that dataset's metadata on its own |
| `<name>_weights.npy` | whest only: the flattened MLP weights |
| `<name>_final_means.npy` | whest only: the labels |
| `<name>_ut_fixed.npy` | whest only: the cheap baseline estimate |
| `<name>_all_layer_means.npy` | whest only: per-layer mean activations |

The `whest_*` datasets are stored as `.npy` rather than parquet because a single
row is an entire weight tensor — hundreds of thousands of floats — which is not
a sensible DataFrame. The library memory-maps them, so a 90 MB array costs no
RAM until rows are touched.

## Versioning

Each upload is a commit, tagged `vYYYYMMDD_HHMMSS`. The library pins a commit
hash in `generatedata/config.py`, so a given release of the library always reads
exactly the same bytes. Older snapshots stay reachable by their hash or tag.

## Provenance and licensing

The library code is BSD-3-Clause. Individual datasets carry their own upstream
terms, recorded per dataset in `info.json`:

- MNIST, EMNIST, KMNIST, FashionMNIST, CIFAR-10 — standard research datasets,
  reprocessed into start/target form.
- [MNIST-1D](https://github.com/greydanus/mnist1d) — Greydanus.
- [Long Range Arena](https://github.com/google-research/long-range-arena) — the
  `lra_*` datasets are native reimplementations of the benchmark tasks.
- `whest_*` — the ARC White-Box Estimation Challenge. Rows sourced from
  [`aicrowd/arc-whestbench-public-2026`](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026)
  are marked `source: official:*` in `info.json`, with the pinned upstream
  revision and `license: cc-by-4.0`; the rest are generated from scratch.
