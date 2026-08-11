"""The one place that knows where dataset files physically live.

Every loader in `load_data` asks this module the same question -- *give me a
local filesystem path to the file named X* -- and then just reads that path.
Because the answer is always a `pathlib.Path`, `load_data.py` contains no
branch on local-vs-remote and no knowledge of HTTP or the Hub at all.

Three sources, one return type:

===========  ===========================================================
``local``    a directory of files you generated yourself
``hf``       a pinned revision of the HuggingFace dataset repo
``http``     a date-stamped snapshot directory on a web page (deprecated)
===========  ===========================================================

Both remote sources are content-immutable -- a commit hash and a date-stamped
directory each promise the bytes never change -- so a file, once downloaded, is
cached and reused without ever contacting the network again.

That caching is also why remote ``.npy`` arrays can be memory-mapped.  A
several-hundred-megabyte array of MLP weights costs no RAM until rows are
actually touched, which was previously true only for locally generated data.
"""

import os
import pathlib
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

import generatedata
import generatedata.config


def fetch(filename: str, local: bool = False, data_dir: Path | str | None = None) -> Path:
    """Return a local path to ``filename``, downloading it first if need be.

    Args:
        filename: A file's name within a dataset snapshot, e.g. ``"info.json"``
            or ``"whest_w8_d8_weights.npy"``.  Snapshots are flat, so this is a
            bare name, not a path.
        local: If True, read a locally generated snapshot instead of a remote one.
        data_dir: Override the local snapshot directory.  Only meaningful with
            ``local``; each remote backend has exactly one location.

    Returns:
        Path to the file on disk.
    """
    if local:
        return _resolve_data_dir(data_dir) / filename
    # Read config.BACKEND here rather than at import time, so a test (or a
    # notebook cell) can flip backends without re-importing the package.
    if generatedata.config.BACKEND == "hf":
        return _fetch_hf(filename)
    return _fetch_http(filename)


def _resolve_data_dir(data_dir: Path | str | None) -> Path:
    """The local processed-data directory, defaulting to the one in the package."""
    if data_dir is not None:
        return pathlib.Path(data_dir)
    base_dir = pathlib.Path(generatedata.__path__[0])
    return base_dir / "../data/processed"


def _fetch_hf(filename: str) -> Path:
    """Download one file from the pinned revision of the HuggingFace repo.

    `hf_hub_download` handles the cache, resumption and (for a private repo) the
    token.  When ``HF_REVISION`` is a full commit hash the cache lookup needs no
    network round trip at all, because a hash can never point at new bytes; a
    branch name such as ``"main"`` costs one HTTP request per call to revalidate.
    """
    return Path(
        hf_hub_download(
            repo_id=generatedata.config.HF_REPO_ID,
            revision=generatedata.config.HF_REVISION,
            filename=filename,
            repo_type="dataset",
        )
    )


# --- HTTP backend (deprecated) -- delete everything below to go HF-only -------

# Snapshots are cached per date-stamped directory, so two snapshots never
# collide and a stale file is never served for a fresh URL.
_HTTP_CACHE_ROOT = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "generatedata"


def _fetch_http(filename: str) -> Path:
    """Download one file from the date-stamped snapshot at ``config.DATA_URL``.

    The download goes to a temporary name and is renamed only once it completes,
    so an interrupted transfer can never leave behind a truncated file that a
    later run mistakes for a finished one.  (Same trick as `hf_data.py`, which
    fetches *source* datasets rather than these generated ones.)
    """
    url = generatedata.config.DATA_URL
    cache_dir = _HTTP_CACHE_ROOT / url.rstrip("/").rsplit("/", 1)[-1]
    local_path = cache_dir / filename
    if local_path.exists():
        return local_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}/{filename} ...")
    response = requests.get(f"{url}/{filename}", stream=True, timeout=300)
    response.raise_for_status()
    partial_path = local_path.with_name(local_path.name + ".partial")
    with open(partial_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    partial_path.rename(local_path)
    return local_path
