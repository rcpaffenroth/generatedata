## [v0.4.4] - 2026-07-28

### Fixed
- `load_data(name, local=True, data_dir=...)` now reads the directory it is given.  It
  checked the name against `data_names(local=local)` first, which ignores `data_dir` and
  reads the package-relative `data/processed`, so any other directory raised
  `FileNotFoundError` — or `Unknown dataset` — before `data_dir` was consulted at all.
  Everything downstream of it (`load_data_as_xy`, `load_data_as_xy_onehot`,
  `load_data_as_sequence` on parquet datasets) was unreachable outside the package
  directory for the same reason.  The check is gone rather than repaired:
  `_dataset_info` already validates the name, already honours `data_dir`, and raises the
  same `ValueError` — so the check was redundant as well as wrong, and in the remote
  case it fetched `info.json` a second time on every load.  Three tests in
  `test_load_data.py` cover it, including that the package directory is *not* consulted
  as a fallback.  Note that `tests/test_timeseries.py` and `tests/test_lra_generators.py`
  patch `data_names` in order to get past this check; those patches are now dead
- `data_names(local=True, data_dir=...)` takes a `data_dir` argument, which is what
  makes the above testable, and no longer re-implements `_resolve_data_dir` inline

### Added
- `lra_image_mnist` — MNIST in the same start/target sequence format as `lra_image`,
  giving an easy/hard pair of real image tasks that share a loader and a model config.
  Digits stay at their native 28×28 (sequence length 784) rather than being resized to
  CIFAR's 32×32: resampling would buy shape-compatibility with `lra_image` at the cost
  of no longer being MNIST's pixels.  Normalized to `[0, 1]` like the other LRA image
  tasks, not to the `[-1, 1]` that the older `MNIST` dataset uses.  Fetched through the
  same `ossci-datasets` S3 mirror and `data/external` cache as `generate_mnist`, so it
  costs no additional download
- `lra_toy_bw` — 8×8 images that are uniformly dark or uniformly bright, plus iid pixel
  noise; sequence length 64, two classes, exactly balanced.  Not a benchmark task: it
  exists because every real LRA task is either slow or hard, which makes them poor
  choices for checking that a model, a training loop, or the `load_data_as_sequence`
  path works at all.  Class levels 0.1 and 0.9 against the default noise scale 0.05 are
  16 standard deviations apart per pixel, so it is trivially separable — but the noise
  is deliberately nonzero so that every row is distinct and per-pixel variance is never
  exactly zero.  `noise_scale=0` recovers the fully degenerate two-distinct-rows version
- Both are in `generate_all` at 1,000 points each, matching the existing `MNIST` and
  `MNIST1D` entries.  They are not in the remote snapshot at `config.DATA_URL`, so like
  the rest of the `lra_*` family they need `local=True`
- `notebooks/5-rcp-whest-visualization.ipynb` (with a `tests/` copy, so nbmake runs it)
  — loads everything with `local=True`, since these datasets are not in the remote
  snapshot, and adapts to whichever of them are present rather than assuming the full
  ladder.  It re-derives `F(W)` from the stored weights to confirm the `z @ W`
  convention (z-scores standard normal over the live coordinates; the transposed
  convention is wrong by O(1)), then plots the heavy tail of `F`, the ReLU cone
  collapsing with depth, width scaling at fixed depth, and the residual structure of
  the cheap baseline — including that its error on dead coordinates is exactly zero
- `load_data.dataset_info(name, local=..., data_dir=...)` — public access to a
  dataset's metadata without loading its data.  Previously `load_data` was the only
  way to obtain it, which does not work for a family whose loader refuses the flat
  DataFrame path and whose arrays are hundreds of megabytes
- `README_whest.md` — standalone reference for this family: the competition
  conventions and the evidence for them, how each rung's size and network count were
  derived, per-rung label noise and baseline errors, the deadness structure that
  shapes the ladder, usage and training examples (all of which are run and verified),
  attribution for the official data, and the metadata schema.  `README.md` keeps a
  summary table and links to it, so the details live in one place
- `generatedata/whest_generators.py` — datasets for the ARC White-Box Estimation
  Challenge 2026: predict a deep ReLU MLP's final-layer mean activation
  `F(W) = E_{x~N(0,I)}[relu-chain(x)]` from its weights alone.  One row per random
  network: features are `flatten(W)` in layer-major order, labels are the `width`
  final-layer means.  `generate_whest(data_dir, width, depth, num_points,
  mc_samples, seed)` is the entry point; `he_weights`, `mc_final_mean`, and
  `ut_fixed_final_mean` are usable independently
- A **new storage backend** for this family: plain `.npy` arrays (`storage: "npy"` in
  the metadata) rather than a flat start/target parquet pair, holding the weight
  tensor, the per-layer mean stack, the final-layer means and the cheap `UT_fixed`
  baseline.  One scalar parquet column per weight is not a viable layout: parquet
  costs ~570 bytes per column regardless of row count
  (`file_bytes ≈ columns × (4…5 × num_points + 570)`), so a 100 MB file is exhausted
  by ~175,000 columns with *zero* rows, and past ~262,000 columns pyarrow writes a
  file it can no longer read back (`TProtocolException: Exceeded size limit` while
  deserialising the schema).  Compression is no escape — He weights are
  maximum-entropy for their variance, and byte-shuffle + LZMA reaches only 0.845×.
  Local arrays are memory-mapped, so a 471 MB dataset costs no RAM until rows are
  touched, and each array is written to a `.partial` file and renamed only on
  success so an interrupted write cannot leave a truncated file that
  `dataset_exists` accepts as present
- **The whest family is reachable only through `load_data_as_sequence`.**  `load_data`
  and `load_data_as_xy` raise a `ValueError` naming the function to use instead.
  That is the honest access path: the forward pass is an *ordered* product of
  operators, so the layer axis is a sequence axis and not an exchangeable feature
  axis.  With `default_step_size = width**2` the loader returns
  `(num_points, depth, width**2)` — one weight matrix per timestep, in layer order
- `load_data_as_sequence` gained a `part` argument — `"target"` (the ground truth),
  `"start"` (the cheap `UT_fixed(W)` estimate, so `target - start` is the residual a
  corrector must learn) or `"all_layers"` (the per-layer mean stack, which only the
  whest family provides).  It applies to flat datasets too, where `"start"` and
  `"target"` select the corresponding parquet block
- `load_data_as_sequence`'s `label_every_step` now defaults to `None`, resolved from
  the dataset's new `label_every_step_allowed` metadata; passing `True` where it is
  disallowed raises.  For the whest family broadcasting the labels onto every
  timestep of the features would hand a model the very quantity it must predict: the
  estimator contract passes only `width`, `depth`, `weights`, `seed` and `name` — no
  means of any kind — and having the per-layer means as inputs would collapse 32
  layers of error compounding into one.  Every existing dataset resolves `None` to
  `True`, so their behaviour is unchanged
- **The officially published competition data** is now ingestible:
  `generate_whest_official` carves row ranges out of
  `aicrowd/arc-whestbench-public-2026` (CC-BY-4.0, pinned to commit hashes), whose
  labels are ground truth at 1e9 samples per network — a noise floor of ~5e-11 that
  nothing bakeable locally approaches.  Attribution, the source revision and tag,
  the row range, `mlp_id` / `mlp_seed` / `mlp_name`, and a `statistical_note` warning
  that 11 networks is a smoke test rather than a benchmark are all recorded.  All
  official rungs are carved from the `mini` splits with disjoint row ranges, needing
  ~1.07 GB of download instead of the 8.6 GB `full` split
- `WHEST_GRID` and `WHEST_XL` in `whest_generators.py` are the ladder: eight datasets
  at ≤100 MB each and two at ≤500 MB.  Depth is held at the competition's 32 while
  width sweeps 256→16, with a depth-8 branch to isolate depth, because depth and not
  width is the regime variable — the fraction of exactly-zero coordinates is ~2% at
  depth 8, ~12–18% at depth 16 and ~25–33% at depth 32, nearly independent of width
  above width 32.  Width 8 at depth 32 is excluded as degenerate (half its
  coordinates and 5% of whole networks dead).  Each rung's `num_points` is set by the
  size budget via `whest_dataset_bytes`, and the tests hold the table to it
- `dataset_exists` and `compile_info_json` learned about non-parquet storage through
  a new `dataset_data_files` helper, so an `.npy`-backed dataset is neither
  regenerated on every call nor silently dropped from `info.json`
- Competition conventions are reproduced exactly: weights iid `N(0, 2/width)`
  (He init at fan-in `width`) with no biases, and the forward map `z ← relu(z @ W)`
  — `@W`, not `@W.T`.  Ground-truth sums are accumulated in float64 with TF32
  disabled (TF32's ~1e-3 relative error per layer would exceed the Monte-Carlo
  noise being paid for); the TF32 flag is restored afterwards rather than set
  globally at import
- The whest start/target split carries a baseline rather than noise: target labels
  are the Monte-Carlo ground truth `F(W)`, start labels are `UT_fixed(W)` — the
  `2·width` unscented-transform sigma points `±r·e_i` at radius `r = E‖x‖`, with no
  random rotations — so `target − start` is the deterministic residual
  `R(W) = F(W) − UT_fixed(W)`
- whest metadata records `mc_samples`, the measured label noise `label_mc_se2`, the
  cheap estimator's error `ut_final_layer_mse`, `dead_coordinate_fraction` and
  `dead_network_fraction`, `forward_convention`, and `flatten_order`
- Only `whest_w8_d8` is built by `generate_all`; the rest of the ladder is opt-in
  through `generate_whest_all(data_dir, include_xl=False)`, or
  `scripts/generatedata_local.py --whest` / `--whest-xl`.  `generate_all`'s signature
  is unchanged, so `tests/conftest.py` and any external caller are untouched, and
  `--all` remains CPU-only and self-contained.  The core rung stays in the default
  set deliberately: it keeps the `.npy` path and its loader guards under test on
  every run.  It also keeps the smaller Monte-Carlo budget (2¹⁸ draws, ~20 s on a
  GPU and ~10 min CPU-only) that a session fixture can afford, where the gated rungs
  use 2²⁴

## [v0.4.3] - 2026-07-27

### Fixed
- CI dependency install in `.github/workflows/python-tests.yml`: the step still ran
  `uv sync --extra dev`, but the dev dependencies had moved from
  `[project.optional-dependencies]` to `[dependency-groups]`, so no such extra
  existed and the install failed.  Now `uv sync --group dev`

### Changed
- Version bump to 0.4.3.  This is the first tag on `main` that actually contains
  the v0.4.1 changes below: they were developed on `release/v0.4.1` and only merged
  to `main` here, so despite the version ordering, v0.4.2 does not include them
- `.github/copilot-instructions.md` removed in favour of the global instructions
  file

## [v0.4.2] - 2026-07-26

### Changed
- Version bump only, `0.4.0` → `0.4.2` in `pyproject.toml`.  No functional change:
  the tag was cut from the `develop` line before the v0.4.1 work was merged, so its
  tree predates everything in the v0.4.1 entry below

## [v0.4.1] - 2026-07-25

### Added
- `requests` declared as an explicit dependency — it is imported by `load_data.py`
  and `data_generators.py` but was previously only available transitively
- `anywidget` dependency — plotly >=6 moved `go.FigureWidget` behind it, which
  `notebooks/4-rcp-timeseries-datasets.ipynb` requires
- `pillow` declared as an explicit dependency — now imported directly to decode
  the PNG-encoded images in HuggingFace parquet files, where previously it was
  only available transitively via `torchvision`
- KMNIST restored to the `all=True` dataset sweep, undoing its removal in v0.3.2.
  It now loads from a pinned HuggingFace mirror rather than `codh.rois.ac.jp`;
  the mirror was verified byte-identical to the official idx-ubyte files (same
  60000 rows in the same order, identical labels), and the resulting tensors are
  bit-exact against `torchvision.datasets.KMNIST`
- `generatedata/hf_data.py` — `download_hf_parquet()` for cached, revision-pinned
  downloads from the HuggingFace Hub, and `HFImageDataset`, a small
  torchvision-style wrapper over a HuggingFace image/label parquet file

### Changed
- CIFAR-10 and IMDB now download from pinned HuggingFace Hub revisions instead of
  `cs.toronto.edu` and `ai.stanford.edu`.  The CIFAR-10 page had stopped serving
  data entirely, which hung `generate_all()` (and therefore the whole test suite)
  indefinitely, since torchvision's downloader sets no socket timeout.  Cold
  generation of `lra_image` + `lra_text` now takes seconds rather than never
  completing.  Every dataset source is pinned to a commit hash for reproducibility
- Dataset downloads are written to a `.partial` file and renamed only on success,
  so an interrupted download can no longer leave behind a truncated file that
  later looks complete (the previous failure mode required manually clearing the
  cache before generation could recover)
- Loosened all dependency version ranges from tight minor-version pins to
  lower bound + next-major cap (e.g. `torch==2.5.1` → `torch>=2.5,<3`,
  `numpy>=2.1,<2.2` → `numpy>=2.1,<3`), so `generatedata` can be installed
  alongside other projects without version conflicts.  The test suite was
  verified at both the floor (`--resolution lowest-direct`) and the ceiling
  (torch 2.13, numpy 2.2, pandas 2.3, plotly 6.9, scikit-learn 1.7)
- The PyTorch CUDA index is now declared with `explicit = true` and routed only
  to `torch`/`torchvision` via `[tool.uv.sources]`.  Previously it was the
  default index for *every* package, which silently capped versions of anything
  that index happens to mirror (notably `numpy`)

## [v0.4.0] - 2026-03-16

### Added
- `load_data_as_sequence()` function in `load_data.py` — reshapes any flat dataset
  with `x_y_index` metadata into time-series format `(num_points, seq_len, step_size)`
  at load time, with configurable `step_size` and `label_every_step` options
- Random erasing transform (`random_erasing_prob`) for MNIST custom dataset generation,
  enabling new augmented MNIST variants
- MNIST downloads pinned to the `ossci-datasets` S3 mirror, replacing torchvision's
  default mirror list so the often-unavailable `yann.lecun.com` is never tried
- New notebook `notebooks/4-rcp-timeseries-datasets.ipynb` — interactive sequence
  builder with step-by-step pixel reveal, heatmap visualisation, and a complete
  LSTM classifier training example
- Comprehensive test suite for sequence loading (`tests/test_timeseries.py`) covering
  shape validation, label broadcasting, cross-dataset support, and error handling

### Changed
- Switched from `torchvision.transforms` to `torchvision.transforms.v2` in MNIST
  custom data generation
- Updated remote data URL to `20260316_115158` timestamp in `config.py`
- MNIST1D default dataset size reduced from 4000 to 1000 points for consistency

### Removed
- Legacy Poetry configuration files (`poetry.lock`, etc.) — `uv` is now the sole
  package manager

## [v0.3.2] - 2026-03-01

### Added
- Migrated from Poetry to `uv` for dependency management; added `uv.lock`
- Git LFS support via `.gitattributes` for binary data files
- Raw data files in `.npy` and `.parquet` formats for EM, LunarLander, and MassSpec datasets
- Conversion script `scripts/convert_em_pt_to_npy.py` to convert `.pt` files to `.npy`
- Dataset existence checks in `generate_all` — skips regeneration if dataset already exists
- Dataset name validation in `load_data` — raises `ValueError` for unknown dataset names
- Type hints throughout `data_generators.py` and related functions
- GitHub Actions CI workflow improvements and added `.github/CODEOWNERS`
- Task planning documentation for timeseries support (`tasks/timeseries_01_save_data_schema.md`, `timeseries_02_generators.md`, `timeseries_03_load_as_sequence.md`)
- Notebook tests renamed to `test_zzz_*` convention for proper pytest ordering

### Changed
- EM dataset loading updated to use `.npy` format with fixed path handling in `generate_emlocalization`
- EM data generation updated to write parquet files in addition to existing formats
- `load_data.py` refactored to use a context manager when opening `info.json`
- Updated README to reflect new datasets and installation procedure
- Changed MNIST data source repository

### Removed
- KMNIST dataset removed (upstream source appears broken)

### Fixed
- Restored `scripts/generatedata_local.py` after accidental breakage

## [v0.3.1] - 2025-12-17
### Changed
- Updated version number to 0.3.1
- Updated dependencies to latest compatible versions

### Fixed
- Added `data/` directory to `.gitignore` to prevent accidental commits of generated datasets

## [v0.3.0]
- Major refactor: cleaned up many interfaces
- More data types including several varieties of MNIST1D and MNIST
- The total produced data size is 1.3GB
