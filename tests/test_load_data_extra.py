"""
Unit tests for the generatedata.load_data module.
"""
import numpy as np
import pandas as pd
from generatedata import load_data, save_data
import pathlib

def test_load_data_returns_dict():
    # Create a dummy DataFrame and save as parquet
    df_dir = pathlib.Path('tests/data')
    df_start = pd.DataFrame({'x0': np.arange(10), 'x1': np.arange(10, 20)})
    df_target = pd.DataFrame({'x0': np.arange(20, 30), 'x1': np.arange(30, 40)})
    df_name = 'test_load_data'
    save_data.save_data(df_dir, df_name, df_start, df_target)
    
    # Test loading
    data = load_data.load_data(df_name, local=True,data_dir=df_dir)
    assert isinstance(data, dict)
    assert 'x0' in data['start'] and 'x1' in data['start']
    assert 'x0' in data['target'] and 'x1' in data['target']
    assert np.allclose(data['start']['x0'], np.arange(10))
    assert np.allclose(data['start']['x0'], np.arange(10))
    
    # Clean up
    import os
    os.remove(df_dir / f'{df_name}_start.parquet')
    os.remove(df_dir / f'{df_name}_target.parquet')
    # Also remove the info.json file if it exists
    info_path = df_dir / 'info.json'
    if info_path.exists():
        os.remove(info_path)
    # also remove the directory if it is empty
    if not os.listdir(df_dir):
        os.rmdir(df_dir)
