# Add real assertions to tests that only print

## Problem

Several tests in `tests/test_load_data.py` only `print()` and never assert anything:

```python
def test_data_names_local(generatedata_local_data):
    print(load_data.data_names(local=True))       # no assertion!

def test_load_data_local(generatedata_local_data):
    print(load_data.load_data('MNIST', local=True))  # no assertion!

def test_data_names_remote():
    print(load_data.data_names(local=False))       # no assertion!

def test_load_data_remote():
    print(load_data.load_data('MNIST', local=False))  # no assertion!
```

These tests will pass even if the functions return garbage data, an empty list, or wrong types. They only fail if the function raises an exception. A student looking at these tests won't learn what correct behavior looks like.

## Suggested Fix

Add meaningful assertions. For example:

```python
def test_data_names_local(generatedata_local_data):
    names = load_data.data_names(local=True)
    assert isinstance(names, list)
    assert len(names) > 0
    assert "MNIST" in names

def test_load_data_local(generatedata_local_data):
    data = load_data.load_data('MNIST', local=True)
    assert "info" in data
    assert "start" in data
    assert "target" in data
    assert isinstance(data["start"], pd.DataFrame)
    assert isinstance(data["target"], pd.DataFrame)
    assert data["start"].shape[0] > 0
    assert data["start"].shape == data["target"].shape
```

Apply similar assertions to the remote tests.

## Files to Modify

- `tests/test_load_data.py`

## Testing

- `uv run pytest tests/test_load_data.py -v`
