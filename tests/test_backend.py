"""The storage backend: does the Hub serve the bytes we put there?

A storage migration has exactly one obligation -- deliver the same bytes the old
path delivered -- so that is what these tests check, by hashing files rather than
by trusting the loaders.

What they deliberately do *not* check is that the `http` and `hf` backends agree
value-for-value.  They don't, and they can't: the generators are unseeded
(`generate_circle` calls `np.random.uniform` with no seed, and `seed` appears
once in all of data_generators.py), so every run of `generate_all` draws fresh
numbers.  The WPI snapshot is simply an older draw than the local directory the
Hub repo was uploaded from.  That is a property of the *generators*, not of the
storage, and asserting equality here would only mean the two snapshots happened
to be built from the same run.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from generatedata import backend, config, load_data


@pytest.fixture
def use_backend(monkeypatch):
    """Switch remote backends for the duration of one test.

    `backend.fetch` reads `config.BACKEND` at call time precisely so this works.
    """
    def _use(which: str):
        monkeypatch.setattr(config, "BACKEND", which)
    return _use


def test_config_names_both_backends():
    # Both pins must survive whatever the upload scripts do to config.py.
    assert isinstance(config.DATA_URL, str) and config.DATA_URL.startswith("http")
    assert isinstance(config.HF_REPO_ID, str) and "/" in config.HF_REPO_ID
    assert isinstance(config.HF_REVISION, str) and config.HF_REVISION


def test_fetch_returns_a_readable_local_path(use_backend):
    # The whole point of the seam: every backend puts bytes on local disk, so no
    # caller ever has to know which one it was talking to.
    use_backend("hf")
    path = backend.fetch("info.json")
    assert path.is_file()
    assert isinstance(json.loads(path.read_text()), dict)


@pytest.mark.parametrize(
    "filename", ["info.json", "circle_start.parquet", "circle_target.parquet"]
)
def test_hf_serves_the_local_bytes_verbatim(generatedata_local_data, use_backend, filename):
    """The migration's actual claim, stated as bytes rather than as DataFrames."""
    use_backend("hf")
    from_hub = backend.fetch(filename).read_bytes()
    from_disk = (generatedata_local_data / filename).read_bytes()
    assert hashlib.sha256(from_hub).hexdigest() == hashlib.sha256(from_disk).hexdigest()


def test_both_backends_load_the_same_shape(use_backend):
    """Different draws of the same generator, so only the *structure* can match.

    This is the weaker claim that survives unseeded generators, and it is still
    worth making: it catches a backend serving a differently-shaped or
    differently-columned file, which is what a broken migration looks like.
    """
    use_backend("http")
    from_http = load_data.load_data("circle")
    use_backend("hf")
    from_hf = load_data.load_data("circle")

    for part in ("start", "target"):
        assert list(from_http[part].columns) == list(from_hf[part].columns)
        assert from_http[part].shape == from_hf[part].shape
        assert from_http[part].dtypes.equals(from_hf[part].dtypes)


def test_hf_serves_the_npy_datasets_the_web_page_lacks(use_backend):
    # The whest families exist only on the Hub -- the web-page snapshot has no
    # .npy files at all -- so this exercises the path with no HTTP analogue.
    # whest_w8_d8 is the smallest at 20 MB; it is cached after the first run.
    use_backend("hf")
    X_seq, labels = load_data.load_data_as_sequence("whest_w8_d8")
    assert X_seq.shape[0] == labels.shape[0]
    # Memory-mapped, not read into RAM: a reshape of a memmap is a view of one.
    # Under the old code this array arrived through io.BytesIO and cost full RAM.
    assert isinstance(X_seq.base, np.memmap)
