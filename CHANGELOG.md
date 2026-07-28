## [v0.4.4] - 2026-07-28

### Added
- `generatedata/whest_generators.py` — datasets for the ARC White-Box Estimation
  Challenge 2026: predict a deep ReLU MLP's final-layer mean activation
  `F(W) = E_{x~N(0,I)}[relu-chain(x)]` from its weights alone.  One row per random
  network: features are `flatten(W)` in layer-major order, labels are the `width`
  final-layer means.  `generate_whest(data_dir, width, depth, num_points,
  mc_samples, seed)` is the entry point; `he_weights`, `mc_final_mean`, and
  `ut_fixed_final_mean` are usable independently
- `whest_w8_d8` added to the core dataset set (10,000 networks, 512 features + 8
  labels, ~60 MB).  `all=True` additionally generates `whest_w16_d16` and
  `whest_w32_d8` (2,000 networks each); those are gated because `flatten(W)` has
  `depth × width²` columns.  The competition's own geometry (width 256, depth 32)
  would need 2,097,152 columns per row and is out of scope for the flat format
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
  cheap estimator's error `ut_final_layer_mse`, `dead_fraction` (networks with
  `F ≡ 0`), `weight_std`, `forward_convention`, and `flatten_order`
- whest datasets record `default_step_size = width**2`, so `load_data_as_sequence`
  returns `(num_points, depth, width**2)` — one weight matrix per timestep in layer
  order — with no change to `load_data.py`.  They deliberately do *not* set
  `is_sequence`, since their rows are not padded and the flat X/Y view is
  legitimate

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
