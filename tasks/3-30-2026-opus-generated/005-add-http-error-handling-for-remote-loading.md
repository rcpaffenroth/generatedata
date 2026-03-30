# Add HTTP error handling for remote data loading

## Problem

In `generatedata/load_data.py`, remote HTTP requests have no error handling:

- Line 30: `requests.get(DATA_URL + "/info.json")` in `data_names()`
- Line 84: `requests.get(DATA_URL + "/info.json")` in `load_data()`
- Lines 87-89: `pd.read_parquet(DATA_URL + ...)` in `load_data()`

If the remote server is down, returns a 404, or the network is unavailable, the user gets a confusing low-level error (e.g., `ConnectionError`, `JSONDecodeError`, or a cryptic pyarrow error). A graduate student unfamiliar with HTTP debugging will not know what went wrong.

## Suggested Fix

Add `response.raise_for_status()` after each `requests.get()` call, and wrap the remote loading in a try/except that gives a clear error message:

```python
# In data_names():
try:
    response = requests.get(DATA_URL + "/info.json")
    response.raise_for_status()
    data_info = response.json()
except requests.RequestException as e:
    raise ConnectionError(
        f"Failed to fetch dataset list from {DATA_URL}/info.json. "
        f"Check your internet connection or try local=True. Error: {e}"
    ) from e
```

Apply similar handling in `load_data()` for both the info.json fetch and the parquet file loads.

## Files to Modify

- `generatedata/load_data.py`

## Testing

- `uv run pytest tests/test_load_data.py`
- Existing remote tests (`test_data_names_remote`, `test_load_data_remote`) should still pass when the server is reachable
- Optionally add a test that mocks a failed HTTP request and verifies the clear error message
