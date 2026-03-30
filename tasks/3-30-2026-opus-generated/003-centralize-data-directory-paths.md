# Centralize the default data directory path

## Problem

The path `base_dir / "../data/processed"` is computed in multiple places across the codebase:

- `generatedata/load_data.py` line 25 (in `data_names()`)
- `generatedata/load_data.py` line 72 (in `load_data()`)
- `tests/conftest.py` line 12

Similarly, the "external data" path `data_dir.parent.parent / 'data' / 'external'` appears in:

- `generatedata/data_generators.py` line 326 (in `generate_mnist()`)
- `generatedata/data_generators.py` line 373-375 (in `generate_mnist_custom()`)
- `generatedata/lra_generators.py` line 510 (in `generate_lra_image()`)
- `generatedata/lra_generators.py` line 643 (in `generate_lra_text()`)

And the "raw data" path `data_dir.parent / 'raw'` appears in:

- `generatedata/data_generators.py` lines 391, 410, 432

If the directory structure ever changes, every one of these must be updated. A student reading the code has to mentally resolve `../` to understand where data lives.

## Suggested Fix

Add helper functions to `generatedata/config.py`:

```python
from pathlib import Path
import generatedata

DATA_URL = 'http://users.wpi.edu/~rcpaffenroth/data/generatedata/20260316_115158'

def get_project_root() -> Path:
    """Return the root directory of the generatedata project."""
    return Path(generatedata.__path__[0]).parent

def get_processed_data_dir() -> Path:
    """Return the default directory for processed data."""
    return get_project_root() / "data" / "processed"

def get_external_data_dir() -> Path:
    """Return the directory for external/downloaded data."""
    return get_project_root() / "data" / "external"

def get_raw_data_dir() -> Path:
    """Return the directory for raw data."""
    return get_project_root() / "data" / "raw"
```

Then replace all the scattered path computations with these functions.

## Files to Modify

- `generatedata/config.py` (add the helper functions)
- `generatedata/load_data.py` (use `get_processed_data_dir()`)
- `generatedata/data_generators.py` (use `get_external_data_dir()` and `get_raw_data_dir()`)
- `generatedata/lra_generators.py` (use `get_external_data_dir()`)
- `tests/conftest.py` (use `get_processed_data_dir()`)

## Testing

- `uv run pytest` (full test suite should pass with no behavior change)
