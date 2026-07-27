"""
Dataset downloads from the HuggingFace Hub.

Several of the datasets used here were originally published on a single academic
web server with no CDN (CIFAR-10 on ``cs.toronto.edu``, IMDB on
``ai.stanford.edu``, KMNIST on ``codh.rois.ac.jp``).  Those hosts are now
routinely slow or unreachable, so we fetch the same data from the HuggingFace
Hub instead, which is CDN-backed.

Each source is pinned to a specific repository revision (a git commit hash) so
that the bytes we download today are the bytes we download a year from now.
"""

import io
from pathlib import Path

import pyarrow.parquet as pq
import requests
import torch
from PIL import Image

_HF_DATASETS_URL = "https://huggingface.co/datasets"


def download_hf_parquet(
    cache_dir: Path,
    repo_id: str,
    revision: str,
    filename: str,
) -> Path:
    """Download one parquet file from a pinned HuggingFace dataset revision.

    The file is cached under ``cache_dir``; a cached copy is reused without
    contacting the network.  The download is written to a temporary path and
    renamed only on success, so an interrupted download can never leave behind
    a truncated file that later looks complete.

    Args:
        cache_dir: Directory in which to cache the downloaded file.
        repo_id: HuggingFace dataset repository, e.g. ``"uoft-cs/cifar10"``.
        revision: Commit hash pinning the repository contents.
        filename: Path of the parquet file within the repository.
    Returns:
        Path to the local copy of the parquet file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / f"{repo_id.replace('/', '__')}__{revision[:8]}__{Path(filename).name}"
    if local_path.exists():
        return local_path

    url = f"{_HF_DATASETS_URL}/{repo_id}/resolve/{revision}/{filename}"
    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    partial_path = local_path.with_name(local_path.name + ".partial")
    with open(partial_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    partial_path.rename(local_path)
    print("Download complete.")
    return local_path


class HFImageDataset:
    """A torchvision-style image dataset backed by a HuggingFace parquet file.

    HuggingFace image datasets store one row per example, holding a PNG-encoded
    image (as a ``{bytes, path}`` struct) and an integer class label.  This class
    exposes them through the same small interface that ``torchvision.datasets``
    provides, so it can be used directly by ``mnist_save_data``: ``len()``,
    indexing that returns a ``(transformed_image, label)`` pair, and a
    ``targets`` tensor of all labels.

    Args:
        parquet_path: Local parquet file, e.g. from :func:`download_hf_parquet`.
        transform: Optional torchvision transform applied to each PIL image.
        image_column: Name of the image column ("image" and "img" are both
            common in HuggingFace repositories).
    """

    def __init__(
        self,
        parquet_path: Path,
        transform=None,
        image_column: str = "image",
    ) -> None:
        table = pq.read_table(parquet_path, columns=[image_column, "label"])
        # The encoded images are kept as raw bytes and decoded lazily on access,
        # since callers typically sample only a small subset of the rows.
        self._images = table.column(image_column).to_pylist()
        self.targets = torch.tensor(table.column("label").to_pylist())
        self.transform = transform

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple:
        img = Image.open(io.BytesIO(self._images[index]["bytes"]))
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets[index])
