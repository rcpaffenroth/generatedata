import torch
from torch.utils.data import Dataset

# a dataloader which returns a batch of start and target data
class StartTargetData(Dataset):
    def __init__(self, z_start: torch.Tensor, z_target: torch.Tensor) -> None:
        self.z_start = z_start
        self.z_target = z_target
    def __len__(self) -> int:
        return len(self.z_start)
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.z_start[idx], self.z_target[idx]