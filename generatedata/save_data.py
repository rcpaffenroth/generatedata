import pandas as pd
import json
from pathlib import Path

def save_data(data_dir, name, start_data, target_data, x_y_index=None, onehot_y=False, additional_info=None):
    """ Save data to parquet files and update info.json.  

    The start_data is the start of a trajectory and the target_data is the end of the trajectory.  
    Classically the target_data lay on the manifold of interest and is the "clean" data.
    Simlarly the start_data is the "noisy" data, for example an MNIST image with the classification label ramdomly chosen.

    Args: name (str): Name of the dataset.
            start_data (list): List of start data.
            target_data (list): List of target data.    

    Returns: None
    """
    data_dir = Path(data_dir)

    target_df = pd.DataFrame(target_data)
    start_df = pd.DataFrame(start_data)
    assert start_df.shape[1]==target_df.shape[1], 'shape mismatch'
    size = start_df.shape[1]

    data_info = {
        'num_points': start_df.shape[0],
        'size': size,       
    }

    if x_y_index is not None:
        data_info['x_y_index'] = x_y_index
        data_info['x_size'] = x_y_index
        data_info['y_size'] = size-x_y_index
        if onehot_y:
            data_info['onehot_y'] = 1
        else:
            data_info['onehot_y'] = 0

    if additional_info is not None:
        if isinstance(additional_info, dict):
            data_info.update(additional_info)
        else:
            raise ValueError("additional_info must be a dictionary")

    # Open or create info.json
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    info_path = data_dir / f'info.json'
    if not info_path.exists():
        with open(info_path, 'w') as f:
            json.dump({}, f)
    with open(info_path, 'r') as f:
        info = json.load(f)

    if name in info:
        info[name].update(data_info)
    else:
        info[name] = data_info

    with open(data_dir / f'info.json', 'w') as f:
        json.dump(info, f)

    start_df.to_parquet(data_dir / f'{name}_start.parquet')
    target_df.to_parquet(data_dir / f'{name}_target.parquet')

