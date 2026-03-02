from generatedata import load_data

import pytest
import pandas as pd

@pytest.mark.parametrize("name", ["MNIST", "MNIST1D", "regression_line", "regression_circle", 
                                  "EMlocalization", "LunarLander", "MassSpec"])
def test_load_data_as_xy_local(generatedata_local_data, name):
    # Only run if dataset is available locally
    X, Y = load_data.load_data_as_xy(name, local=True)
    assert isinstance(X, (pd.DataFrame, pd.Series))
    assert isinstance(Y, (pd.DataFrame, pd.Series))
    assert X.shape[0] == Y.shape[0]

@pytest.mark.parametrize("name", ["MNIST", "MNIST1D"])
def test_load_data_as_xy_onehot_local(generatedata_local_data, name):
    # Only run if dataset is available locally and supports onehot_y
    X, Y = load_data.load_data_as_xy_onehot(name, local=True)
    assert isinstance(X, (pd.DataFrame, pd.Series))
    assert isinstance(Y, (pd.DataFrame, pd.Series))
    assert X.shape[0] == Y.shape[0]

def test_data_names_local(generatedata_local_data):
    print(load_data.data_names(local=True))

def test_load_data_local(generatedata_local_data):
    print(load_data.load_data('MNIST', local=True))

def test_data_names_remote():
    print(load_data.data_names(local=False))

def test_load_data_remote():
    print(load_data.load_data('MNIST', local=False))


