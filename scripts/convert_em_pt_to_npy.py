"""
One-time conversion script: converts EM_X_train.pt and EM_Y_train.pt
to the more standard numpy .npy format.

The .pt format requires PyTorch to read and has security concerns with
pickle-based loading. The .npy format is a simple, well-documented standard
that can be read by any language with a numpy-compatible library.

Run with:
    uv run python scripts/convert_em_pt_to_npy.py
"""
import pathlib
import torch
import numpy as np

raw_dir = pathlib.Path(__file__).parent.parent / 'data' / 'raw'

for stem in ('EM_X_train', 'EM_Y_train'):
    pt_path = raw_dir / f'{stem}.pt'
    npy_path = raw_dir / f'{stem}.npy'

    if not pt_path.exists():
        print(f'Skipping {pt_path.name} (not found)')
        continue

    tensor = torch.load(pt_path, weights_only=True)
    np.save(npy_path, tensor.numpy())
    print(f'Converted {pt_path.name} -> {npy_path.name}  shape={tensor.shape}  dtype={tensor.dtype}')
