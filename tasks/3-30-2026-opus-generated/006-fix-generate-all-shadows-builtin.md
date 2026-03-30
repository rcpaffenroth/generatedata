# Rename `all` parameter that shadows Python builtin

## Problem

In `generatedata/data_generators.py` line 440, the `generate_all()` function has a parameter named `all`:

```python
def generate_all(data_dir: Path, all: bool) -> None:
```

This shadows Python's built-in `all()` function. While this doesn't cause a bug in the current code (the builtin isn't used inside this function), it is a bad practice that confuses linters, IDEs, and students. If someone later adds code inside `generate_all()` that tries to use `all([...])`, they'll get a `TypeError` instead of the expected behavior.

The same parameter name is used in the caller `scripts/generatedata_local.py`.

## Suggested Fix

Rename the parameter to `full_sweep` (or `generate_variants`):

```python
def generate_all(data_dir: Path, full_sweep: bool = False) -> None:
```

Update all references inside the function body (lines 473, 504, 514) and in the caller script.

## Files to Modify

- `generatedata/data_generators.py` (function signature + body)
- `scripts/generatedata_local.py` (caller)
- `tests/conftest.py` (calls `generate_all(data_dir, all=False)`)

## Testing

- `uv run pytest` (full suite)
