"""Where `load_data` reads its datasets from.

Two remote storage backends exist while we migrate to the HuggingFace Hub.  Pick
one with the ``GENERATEDATA_BACKEND`` environment variable::

    GENERATEDATA_BACKEND=hf   uv run pytest        # the Hub
    GENERATEDATA_BACKEND=http uv run pytest        # the WPI web page (default)

Both are *content-immutable*: a pinned commit hash and a date-stamped directory
are each a promise that the bytes behind them never change.  That is what makes
a snapshot citable, and what lets `backend.py` cache both on disk forever.

To go HF-only later, delete the block marked below, drop ``config.BACKEND`` from
`backend.fetch`, and delete `_fetch_http` in `backend.py`.
"""

import os

# --- HTTP backend (deprecated) -- delete this block to go HF-only -------------
# A snapshot is a date-stamped directory written by scripts/copy_data_to_http.sh.
# Note this snapshot is NOT the same dataset collection as the Hub repo below:
# it has the full MNIST/EMNIST/KMNIST/FashionMNIST parameter sweeps but none of
# the whest .npy families, so `data_names()` differs between the two backends.
DATA_URL = 'http://users.wpi.edu/~rcpaffenroth/data/generatedata/20260726_082230'

BACKEND = os.environ.get("GENERATEDATA_BACKEND", "http")
# --- end HTTP backend block ---------------------------------------------------

# HuggingFace backend.  A snapshot is a commit; HF_REVISION pins it.  Both lines
# are rewritten in place by scripts/huggingface_upload.sh, which also pushes a
# date-stamped tag naming the same commit for humans reading the repo history.
HF_REPO_ID = 'rcpaffenroth/generatedata'
HF_REVISION = '251469953ae7e6cc50dccd9b9e952c5c2fc35426'  # tag v20260811_170337
