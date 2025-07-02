"""
Unit tests for generatedata.df_to_tensor
"""
import numpy as np
import pandas as pd
import torch
from generatedata.df_to_tensor import df_to_tensor

def test_df_to_tensor_numpy():
    df = pd.DataFrame({'a': np.arange(3), 'b': np.arange(3, 6)})
    tensor = df_to_tensor(df)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 2)
    assert torch.allclose(tensor[:, 0], torch.tensor([0., 1., 2.]))
