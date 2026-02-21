# Task: Fix Unclosed File Handle in `data_names`

## Context

This is a Python research library (`generatedata`) whose data loading utilities live in `generatedata/load_data.py`. The function `data_names(local=False)` returns the list of available dataset names by reading an `info.json` metadata file.

## Problem

In the `local=True` branch of `data_names` (lines 18–23), the file is opened without a context manager:

```python
data_info = json.load(open(data_dir / "info.json", "r"))
```

This is a resource leak: the file handle is never explicitly closed. Python's garbage collector will eventually close it, but this pattern:

- Triggers linting warnings (e.g. `pylint` `consider-using-with`).
- Can cause issues under PyPy or in environments with limited file descriptors.
- Is inconsistent with the rest of the codebase — the `load_data` function on line 67 already uses the correct `with open(...) as f:` pattern.

## Fix Required

Replace the bare `open()` call in the `local=True` branch of `data_names` with a `with` statement:

**Before (lines 22–23):**
```python
data_info = json.load(open(data_dir / "info.json", "r"))
```

**After:**
```python
with open(data_dir / "info.json", "r") as f:
    data_info = json.load(f)
```

## File to Modify

`generatedata/load_data.py`, inside the `data_names` function, the `if local:` branch (around line 23).

## Constraints

- Change only the file-open pattern; do not alter the logic, return value, or any other lines.
- The fix is 2 lines replacing 1 line (net +1 line).
- Verify the function still returns `list(data_info.keys())` unchanged.
