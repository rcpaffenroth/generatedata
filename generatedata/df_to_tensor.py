import torch
import pandas as pd

# Turn a pandas dataframe into a pytorch tensor
def df_to_tensor(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(df.values, dtype=torch.float32)
