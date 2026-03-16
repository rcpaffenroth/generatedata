## [v0.4.0] - 2026-03-16

### Added
- `load_data_as_sequence()` function in `load_data.py` — reshapes any flat dataset
  with `x_y_index` metadata into time-series format `(num_points, seq_len, step_size)`
  at load time, with configurable `step_size` and `label_every_step` options
- Random erasing transform (`random_erasing_prob`) for MNIST custom dataset generation,
  enabling new augmented MNIST variants
- S3 mirror fallback for MNIST downloads to handle unreliable upstream server
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
