from generatedata import load_data
from generatedata.data_generators import compile_info_json
from generatedata.save_data import save_data

import numpy as np
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


# ── data_dir is respected, not just accepted ────────────────────────────────
#
# These three go together.  `load_data` used to check the name against
# `data_names(local=local)`, which reads the package-relative directory whatever
# `data_dir` says, so a dataset outside the package was unreachable even though the
# parameter for reaching it existed.  A test that only loaded from `data_dir` would
# not have caught that -- the package directory also has the datasets it asks for --
# so the second and third assert that the *only* directory consulted is `data_dir`.

@pytest.fixture()
def elsewhere(tmp_path):
    """A processed-data directory holding one dataset that exists nowhere else."""
    rng = np.random.default_rng(0)
    cols = {f"x{i}": rng.random(20) for i in range(3)}
    save_data(tmp_path, "only_here", cols, cols)
    compile_info_json(tmp_path)
    return tmp_path


def test_load_data_honours_data_dir(elsewhere):
    data = load_data.load_data("only_here", local=True, data_dir=elsewhere)
    assert data["start"].shape == (20, 3)


def test_data_names_honours_data_dir(elsewhere):
    assert load_data.data_names(local=True, data_dir=elsewhere) == ["only_here"]


def test_load_data_does_not_fall_back_to_the_package_directory(elsewhere):
    # 'MNIST' is in the package's own data/processed, and must still be unknown
    # here: finding it would mean data_dir had been ignored.
    with pytest.raises(ValueError, match="Unknown dataset 'MNIST'"):
        load_data.load_data("MNIST", local=True, data_dir=elsewhere)


