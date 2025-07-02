import pandas as pd
import pathlib
import generatedata
import json
import requests
import generatedata.config

DATA_URL = generatedata.config.DATA_URL

def data_names(local=False) -> list:
    """List the names of the datasets that are available to load.

    Returns:
        list: the names of the datasets
    """
    if local:
        # The directory in which the notebook is located.
        base_dir = pathlib.Path(generatedata.__path__[0])
        # The directory where the data is stored.
        data_dir = base_dir / '../data/processed'
        data_info = json.load(open(data_dir / 'info.json', 'r'))
    else:
        # Read the info json file from the URL DATA_URL+'/info.json'
        response = requests.get(DATA_URL+'/info.json')    
        data_info = response.json()
    return list(data_info.keys())

def load_data(name: str, local=False, data_dir=None) -> dict:
    """Load in the dataset with the given name.  This functions loads in a variety of datasets created by the
    `scripts/generate-data.py` script.

    Args:
        name (str): the name of the dataset

    Returns:
        dict: the start and target points of the dataset
    """
    if local:
        # If a data_dir is provided, use it, otherwise use the default data directory
        if data_dir is not None:
            data_dir = pathlib.Path(data_dir)
        else:
            # The directory in which the notebook is located.
            base_dir = pathlib.Path(generatedata.__path__[0])
            # The directory where the data is stored.
            data_dir = base_dir / '../data/processed'

        # load in the info for the datasets
        with open(data_dir / f'info.json', 'r') as f:
            data_info = json.load(f)[name]

        # Read the start data
        z_start = pd.read_parquet(data_dir / f'{name}_start.parquet')
        # Read the target data
        z_target = pd.read_parquet(data_dir / f'{name}_target.parquet')
    else:
        # Read in the info for the datasets
        response = requests.get(DATA_URL+'/info.json')    
        data_info = response.json()
        # Read the start data
        z_start = pd.read_parquet(DATA_URL+f'/{name}_start.parquet')
        # Read the target data
        z_target = pd.read_parquet(DATA_URL+f'/{name}_target.parquet')
        
    return {'info': data_info, 'start': z_start, 'target': z_target}