import pathlib

import pytest

import generatedata
from generatedata.data_generators import generate_all


@pytest.fixture(scope="session", autouse=True)
def generatedata_local_data() -> pathlib.Path:
    base_dir = pathlib.Path(generatedata.__path__[0])
    data_dir = (base_dir / "../data/processed").resolve()
    generate_all(data_dir, all=False)
    return data_dir
