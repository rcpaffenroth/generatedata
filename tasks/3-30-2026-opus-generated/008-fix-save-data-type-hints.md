# Fix incorrect type hints in save_data()

## Problem

In `generatedata/save_data.py`, the `save_data()` function declares `start_data` and `target_data` as type `dict`:

```python
def save_data(
    data_dir: Path | str,
    name: str,
    start_data: dict,          # <-- says dict
    target_data: dict,         # <-- says dict
    ...
```

However, `generate_massspec()` in `data_generators.py` (lines 434-438) passes **pandas DataFrames** directly:

```python
start_data = mass_spec_df[new_order]    # This is a DataFrame, not a dict
target_data = mass_spec_df[new_order]   # This is a DataFrame, not a dict
...
save_data(data_dir, 'MassSpec', start_data, target_data, x_y_index=1433-512)
```

This works by accident because `pd.DataFrame(another_dataframe)` returns a copy, but the type hints are misleading. A student reading the function signature would think only dicts are accepted.

## Suggested Fix

Update the type hints to reflect what the function actually accepts:

```python
import pandas as pd
from typing import Union

def save_data(
    data_dir: Path | str,
    name: str,
    start_data: dict | pd.DataFrame,
    target_data: dict | pd.DataFrame,
    ...
```

Also update the docstring Args section to mention that both dicts and DataFrames are accepted.

## Files to Modify

- `generatedata/save_data.py`

## Testing

- `uv run pytest tests/test_save_data.py`
