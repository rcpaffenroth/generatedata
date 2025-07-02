"""
Unit tests for generatedata.save_data
"""
import pandas as pd
import numpy as np
import torch
from generatedata.save_data import save_data

def test_save_data_creates_files(tmp_path):
    data_dir = tmp_path
    name = 'test_dataset'
    start_data = {'x0': np.arange(5), 'x1': np.arange(5, 10)}
    target_data = {'x0': np.arange(10, 15), 'x1': np.arange(15, 20)}
    save_data(data_dir, name, start_data, target_data)
    start_path = data_dir / f'{name}_start.parquet'
    target_path = data_dir / f'{name}_target.parquet'
    assert start_path.exists()
    assert target_path.exists()
    df_start = pd.read_parquet(start_path)
    df_target = pd.read_parquet(target_path)
    assert 'x0' in df_start.columns and 'x1' in df_start.columns
    assert 'x0' in df_target.columns and 'x1' in df_target.columns

def test_save_data_with_tensor(tmp_path):
    data_dir = tmp_path
    name = 'tensor_dataset'
    start_data = {'x0': torch.arange(5), 'x1': torch.arange(5, 10)}
    target_data = {'x0': torch.arange(10, 15), 'x1': torch.arange(15, 20)}
    save_data(data_dir, name, start_data, target_data)
    start_path = data_dir / f'{name}_start.parquet'
    df_start = pd.read_parquet(start_path)
    assert np.allclose(df_start['x0'], np.arange(5))
