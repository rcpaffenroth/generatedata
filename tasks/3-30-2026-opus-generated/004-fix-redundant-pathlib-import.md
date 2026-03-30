# Fix redundant pathlib import in load_data.py

## Problem

In `generatedata/load_data.py`, lines 2-3 import pathlib twice:

```python
import pathlib
from pathlib import Path
```

Both are used: `pathlib.Path(...)` appears in several places, and `Path` is used in type hints. This is confusing for students -- pick one style and stick with it.

## Suggested Fix

Remove `import pathlib` (line 2) and replace all uses of `pathlib.Path(...)` with `Path(...)`, since `Path` is already imported from `pathlib`. This is the more Pythonic style for type hints and construction.

Instances to change:
- Line 23: `pathlib.Path(generatedata.__path__[0])` -> `Path(generatedata.__path__[0])`
- Line 67: `pathlib.Path(data_dir)` -> `Path(data_dir)`
- Line 70: `pathlib.Path(generatedata.__path__[0])` -> `Path(generatedata.__path__[0])`

## Files to Modify

- `generatedata/load_data.py`

## Testing

- `uv run pytest tests/test_load_data.py`
