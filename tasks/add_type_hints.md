# Task: Add Type Hints to Library Code

## Context

This is a Python research library (`generatedata`) that generates and loads synthetic/real-world datasets for ML experiments. The library consists of several low-level modules used directly by external code. Adding type hints to these modules improves IDE autocompletion, enables static analysis with `mypy`, and serves as inline documentation.

Type hints are most valuable in the **library core** (imported and called by users) and less necessary in top-level scripts or notebooks. Apply hints accordingly, prioritising by the order below.

---

## Files to Modify (in priority order)

### 1. `generatedata/df_to_tensor.py` — highest priority

The entire file has no type hints. Add them:

```python
import torch
import pandas as pd

def df_to_tensor(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(df.values, dtype=torch.float32)
```

---

### 2. `generatedata/StartTargetData.py` — highest priority

The class has no type hints on any method. Add them:

```python
import torch
from torch.utils.data import Dataset

class StartTargetData(Dataset):
    def __init__(self, z_start: torch.Tensor, z_target: torch.Tensor) -> None:
        ...
    def __len__(self) -> int:
        ...
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

---

### 3. `generatedata/save_data.py` — high priority

`save_data` has no type hints at all. The corrected signature:

```python
def save_data(
    data_dir: Path | str,
    name: str,
    start_data: dict,
    target_data: dict,
    x_y_index: int | None = None,
    onehot_y: bool = False,
    additional_info: dict | None = None,
) -> None:
```

---

### 4. `generatedata/load_data.py` — high priority

Several parameters and return types are untyped or imprecisely typed. Required changes:

| Function | Current signature | Target signature |
|---|---|---|
| `data_names` | `(local=False) -> list` | `(local: bool = False) -> list[str]` |
| `get_random_data_name` | `(local=False) -> str` | `(local: bool = False) -> str` |
| `load_data` | `(name: str, local=False, data_dir=None) -> dict` | `(name: str, local: bool = False, data_dir: Path \| str \| None = None) -> dict` |
| `load_data_as_xy` | `(name: str, local=False, data_dir=None) -> tuple` | `(name: str, local: bool = False, data_dir: Path \| str \| None = None) -> tuple[pd.DataFrame, pd.DataFrame]` |
| `load_data_as_xy_onehot` | `(name: str, local=False, data_dir=None) -> tuple` | `(name: str, local: bool = False, data_dir: Path \| str \| None = None) -> tuple[pd.DataFrame, pd.DataFrame]` |

Also add `from pathlib import Path` to the imports if it is not already present. The `pathlib` import is needed for the `Path` type in the signature of `load_data`.

---

### 5. `generatedata/data_generators.py` — medium priority

Two internal helper functions have no type hints:

**`mnist1d_save_data`** (line 122):
```python
def mnist1d_save_data(
    data_dir: Path,
    name: str,
    num_points: int,
    mnist1d_dataset: dict,
    vector_dim: int = 40,
    additional_info: dict | None = None,
) -> None:
```

**`mnist_save_data`** (line 198):
```python
def mnist_save_data(
    data_dir: Path,
    name: str,
    num_points: int,
    mnist_dataset,                        # torchvision Dataset — leave as untyped or use Any
    vector_dim: int = 28 * 28,
    additional_info: dict | None = None,
) -> None:
```

Also add missing parameter type hints for the `generate_mnist1d_custom` and `generate_mnist_custom` functions where float/tuple parameters currently have no annotations:

- `generate_mnist1d_custom`: `scale_coeff: float`, `max_translation: int`, `corr_noise_scale: float`, `iid_noise_scale: float`, `shear_scale: float`
- `generate_mnist_custom`: `dataset_name: str`, `degrees: tuple[int, int]`, `translate: tuple[float, float]`, `scale: tuple[float, float]`

---

## What NOT to annotate

- `scripts/generatedata_local.py` — top-level entry-point script; type hints add little value here.
- Jupyter notebooks in `notebooks/` — not applicable.
- The `generate_all` function's loop variables (`l1`, `l2`, etc.) — local variables inferred by static analysers automatically.

---

## Python Version Compatibility

The codebase uses Python 3.10+. You may use the modern union syntax `X | Y` (PEP 604) and built-in generic types like `list[str]` and `tuple[X, Y]` (PEP 585) without importing from `typing`.

Only add `from __future__ import annotations` if needed for forward references; it is not required for these changes.
