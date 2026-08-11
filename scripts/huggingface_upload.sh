#! /bin/bash
#
# Publish everything in data/processed/ as a new commit of the HuggingFace
# dataset repo, then pin generatedata/config.py to that commit.
#
# Re-run it as often as you like.  `hf upload` hashes each file and skips the
# unchanged ones, so uploading a handful of core datasets today and the full
# `generate_all(..., all=True)` sweep next month is the same command twice, and
# the second run transfers only what is new.
#
# The upload is additive: files already on the Hub with no local counterpart are
# left alone.  That is deliberate.  A half-generated data/processed/ is a normal
# state to be in, and it must never be able to delete a published dataset.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="$repo_root/data/processed"
config_py="$repo_root/generatedata/config.py"

# The repo id lives in config.py, so there is one place to change it.
HF_REPO_ID=$(sed -n "s/^HF_REPO_ID = '\(.*\)'/\1/p" "$config_py")

# Writing to the Hub needs a token; reading from a public repo does not, which
# is why nothing in generatedata/ ever asks for one.
: "${HF_TOKEN:=$(cat "$repo_root/do_not_commit/huggingface_token")}"
export HF_TOKEN

tag="v$(date +%Y%m%d_%H%M%S)"
echo "Uploading $data_dir -> $HF_REPO_ID (tag $tag)"

uv run hf repos create "$HF_REPO_ID" --repo-type dataset --public --exist-ok

# The dataset card is version-controlled here rather than in data/processed/,
# which is generated and gitignored.
uv run hf upload "$HF_REPO_ID" "$repo_root/scripts/hf_dataset_card.md" README.md \
    --repo-type dataset --commit-message "Dataset card"

uv run hf upload "$HF_REPO_ID" "$data_dir" . \
    --repo-type dataset --commit-message "Snapshot $tag"

# A tag is for humans reading the repo history; config.py pins the commit hash
# the tag names, because a hash cannot be moved to point at different bytes.
uv run hf repos tag create "$HF_REPO_ID" "$tag" --repo-type dataset
sha=$(uv run python -c "
from huggingface_hub import HfApi
print(HfApi().repo_info('$HF_REPO_ID', repo_type='dataset', revision='$tag').sha)
")

echo "Pinning HF_REVISION = '$sha' in $config_py"
sed -i "s|^HF_REVISION = .*|HF_REVISION = '$sha'  # tag $tag|" "$config_py"

echo
echo "Done.  Try it with:"
echo "  GENERATEDATA_BACKEND=hf uv run python -c \"from generatedata import load_data; print(load_data.data_names())\""
