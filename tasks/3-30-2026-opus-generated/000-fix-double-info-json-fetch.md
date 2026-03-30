# Fix double info.json fetch in load_data()

## Problem

In `generatedata/load_data.py`, the `load_data()` function calls `data_names()` on line 60 to validate the dataset name. `data_names()` fetches and parses `info.json` (either locally or via HTTP). Then `load_data()` immediately fetches `info.json` *again* on lines 75-76 (local) or line 84 (remote) to get the dataset's metadata.

This means every call to `load_data()` reads/downloads `info.json` **twice**. For remote usage this is two separate HTTP requests, which is slow and wasteful.

The same double-fetch problem affects `load_data_as_xy()`, `load_data_as_xy_onehot()`, and `load_data_as_sequence()` since they all call `load_data()` internally.

## Suggested Fix

Refactor `load_data()` to fetch `info.json` once and use it for both validation and metadata extraction. For example:

```python
def load_data(name: str, local: bool = False, data_dir: Path | str | None = None) -> dict:
    if local:
        if data_dir is not None:
            data_dir = pathlib.Path(data_dir)
        else:
            base_dir = pathlib.Path(generatedata.__path__[0])
            data_dir = base_dir / "../data/processed"

        with open(data_dir / "info.json", "r") as f:
            all_info = json.load(f)

        if name not in all_info:
            raise ValueError(f"Unknown dataset '{name}'. Available datasets: {list(all_info.keys())}")

        data_info = all_info[name]
        z_start = pd.read_parquet(data_dir / f"{name}_start.parquet")
        z_target = pd.read_parquet(data_dir / f"{name}_target.parquet")
    else:
        response = requests.get(DATA_URL + "/info.json")
        all_info = response.json()

        if name not in all_info:
            raise ValueError(f"Unknown dataset '{name}'. Available datasets: {list(all_info.keys())}")

        data_info = all_info[name]
        z_start = pd.read_parquet(DATA_URL + f"/{name}_start.parquet")
        z_target = pd.read_parquet(DATA_URL + f"/{name}_target.parquet")

    return {"info": data_info, "start": z_start, "target": z_target}
```

## Files to Modify

- `generatedata/load_data.py`

## Testing

- Run the existing tests: `uv run pytest tests/test_load_data.py`
- Verify behavior is unchanged (same return values, same error messages for invalid names)
