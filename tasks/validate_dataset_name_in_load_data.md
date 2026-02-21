# Task: Add Dataset Name Validation to `load_data`

## Context

This is a Python research library (`generatedata`) that loads synthetic and real-world datasets for ML experiments. Users load datasets by name using the public API in `generatedata/load_data.py`.

The function `data_names(local=False)` returns a list of valid dataset names by reading `info.json` (either locally or from a remote URL).

The function `load_data(name, local=False, data_dir=None)` loads a dataset by name. It calls `requests.get(...)` or reads local Parquet files, then indexes into the parsed JSON with `data_info = response.json()[name]`.

## Problem

If a user passes an invalid or misspelled dataset name to `load_data`, they receive a cryptic `KeyError` with no guidance on what names are valid:

```python
load_data("manyfold")   # typo of "manifold"
# → KeyError: 'manyfold'
```

This is especially confusing because the error originates deep in a JSON dict lookup rather than at the call site, and the user has no indication of what names are available.

## Fix Required

Add a name validation check at the top of `load_data`, before any file I/O or HTTP requests are made. The check should:

1. Call `data_names(local=local)` to get the list of valid names.
2. If `name` is not in that list, raise a `ValueError` with a clear message that includes the list of available names.

Example of the desired error:

```
ValueError: Unknown dataset 'manyfold'. Available datasets: ['circle', 'manifold', 'pca_line', ...]
```

## File to Modify

`generatedata/load_data.py` — add approximately 4–5 lines at the start of the `load_data` function body (after line 56, before the `if local:` branch).

## Constraints

- Do not change the function signature or return type.
- Do not duplicate the HTTP/file I/O for `info.json`; simply call the existing `data_names()` helper.
- Keep the new code minimal — this should be a guard clause only.
