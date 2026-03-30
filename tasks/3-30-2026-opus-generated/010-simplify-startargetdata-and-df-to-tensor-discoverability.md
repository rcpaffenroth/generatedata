# Improve discoverability of StartTargetData and df_to_tensor

## Problem

`StartTargetData` and `df_to_tensor` live in separate single-function files (`StartTargetData.py` and `df_to_tensor.py`) and are not exported from the package's `__init__.py` (which is empty). A student must know the exact import path:

```python
from generatedata.StartTargetData import StartTargetData
from generatedata.df_to_tensor import df_to_tensor
```

This is hard to discover. Most students will only find `load_data` by reading the README and won't know these utilities exist. Additionally, the file `StartTargetData.py` uses PascalCase (a class naming convention) as a module name, which is unconventional in Python.

## Suggested Fix

Export the key public API from `generatedata/__init__.py`:

```python
from generatedata.load_data import (
    data_names,
    get_random_data_name,
    load_data,
    load_data_as_xy,
    load_data_as_xy_onehot,
    load_data_as_sequence,
)
from generatedata.StartTargetData import StartTargetData
from generatedata.df_to_tensor import df_to_tensor
```

This lets students write:

```python
import generatedata
data = generatedata.load_data("MNIST", local=True)
# or
from generatedata import StartTargetData, df_to_tensor
```

This is a non-breaking change -- the old import paths still work.

## Files to Modify

- `generatedata/__init__.py`

## Testing

- `uv run pytest` (full suite to ensure no circular import issues)
- Verify that `from generatedata import load_data, StartTargetData, df_to_tensor` works in a Python REPL
