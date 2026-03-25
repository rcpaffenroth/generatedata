"""
Long Range Arena (LRA) dataset generators.

Native Python/PyTorch implementations of the LRA benchmark tasks:
- ListOps: hierarchical expression evaluation (seq_len=2048, 10 classes)
- Text: IMDB byte-level sentiment classification (seq_len=4096, 2 classes)
- Image: CIFAR-10 grayscale sequential classification (seq_len=1024, 10 classes)
- Pathfinder: synthetic visual path connectivity (seq_len=1024, 2 classes)
- Path-X: extended pathfinder at higher resolution (seq_len=16384, 2 classes)

Reference: Tay et al., "Long Range Arena: A Benchmark for Efficient Transformers", 2020.
https://github.com/google-research/long-range-arena
"""

import io
import json
import tarfile
import random
from pathlib import Path

import numpy as np
import requests
import torch
from torchvision import datasets as tv_datasets

from generatedata.save_data import save_data


# =============================================================================
# Shared helper
# =============================================================================

def _lra_save_classification_data(
    data_dir: Path,
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    additional_info: dict | None = None,
) -> None:
    """Save LRA classification data in the start/target parquet format.

    Follows the same pattern as ``mnist_save_data`` in ``data_generators.py``:
    - **target**: sequence features + true one-hot label
    - **start**: same sequence features + uniform label probabilities

    Args:
        data_dir: Directory to write parquet / JSON files.
        name: Dataset name (used as file prefix).
        features: Numeric feature array of shape ``(N, seq_len)``.
        labels: Integer label array of shape ``(N,)`` with values in ``[0, num_classes)``.
        num_classes: Number of distinct classes.
        additional_info: Extra metadata stored in the info JSON.
    """
    num_points, seq_len = features.shape

    # Build one-hot labels
    one_hot = np.zeros((num_points, num_classes), dtype=np.float32)
    one_hot[np.arange(num_points), labels] = 1.0

    # target = features + true one-hot
    target_arr = np.concatenate(
        [features.astype(np.float32), one_hot], axis=1
    )
    # start = features + uniform label
    start_arr = target_arr.copy()
    start_arr[:, -num_classes:] = 1.0 / num_classes

    total_cols = seq_len + num_classes
    start_data = {f"x{i}": start_arr[:, i] for i in range(total_cols)}
    target_data = {f"x{i}": target_arr[:, i] for i in range(total_cols)}

    save_data(
        data_dir,
        name,
        start_data,
        target_data,
        x_y_index=seq_len,
        onehot_y=True,
        additional_info=additional_info,
    )


# =============================================================================
# ListOps
# =============================================================================

# Token vocabulary for ListOps
LISTOPS_OPERATORS = ["MIN", "MAX", "MEDIAN", "SUM_MOD"]
LISTOPS_OPEN = "["
LISTOPS_CLOSE = "]"
LISTOPS_PAD = "PAD"

# Build vocab: PAD=0, digits 0-9 -> 1-10, operators -> 11-14, [ -> 15, ] -> 16
_LISTOPS_VOCAB = {LISTOPS_PAD: 0}
for _d in range(10):
    _LISTOPS_VOCAB[str(_d)] = _d + 1
for _i, _op in enumerate(LISTOPS_OPERATORS):
    _LISTOPS_VOCAB[_op] = 11 + _i
_LISTOPS_VOCAB[LISTOPS_OPEN] = 15
_LISTOPS_VOCAB[LISTOPS_CLOSE] = 16
LISTOPS_VOCAB_SIZE = 17


def _listops_evaluate(tokens: list[str]) -> int:
    """Evaluate a ListOps expression represented as a flat token list.

    Returns an integer in [0, 9].
    """
    stack: list = []
    for tok in tokens:
        if tok == LISTOPS_CLOSE:
            # Pop arguments until we hit the operator
            args: list[int] = []
            while stack and stack[-1] not in LISTOPS_OPERATORS:
                args.append(stack.pop())
            if not stack:
                return 0
            op = stack.pop()
            # Also pop the opening bracket
            if stack and stack[-1] == LISTOPS_OPEN:
                stack.pop()
            args.reverse()
            if not args:
                stack.append(0)
                continue
            if op == "MIN":
                result = min(args)
            elif op == "MAX":
                result = max(args)
            elif op == "MEDIAN":
                sorted_args = sorted(args)
                mid = len(sorted_args) // 2
                result = sorted_args[mid]
            elif op == "SUM_MOD":
                result = sum(args) % 10
            else:
                result = 0
            stack.append(result)
        elif tok == LISTOPS_OPEN:
            stack.append(tok)
        elif tok in LISTOPS_OPERATORS:
            stack.append(tok)
        elif tok.isdigit():
            stack.append(int(tok))
        # skip PAD tokens
    return int(stack[0]) if stack else 0


def _listops_generate_expression(
    rng: np.random.Generator,
    depth: int,
    max_depth: int,
    max_args: int,
) -> list[str]:
    """Recursively generate a single ListOps expression as a token list."""
    if depth >= max_depth:
        # Leaf: return a single digit
        return [str(rng.integers(0, 10))]

    op = LISTOPS_OPERATORS[rng.integers(0, len(LISTOPS_OPERATORS))]
    num_args = int(rng.integers(2, max_args + 1))

    tokens = [LISTOPS_OPEN, op]
    for _ in range(num_args):
        # Decide whether to recurse or emit a leaf
        if rng.random() < 0.5 and depth + 1 < max_depth:
            tokens.extend(
                _listops_generate_expression(rng, depth + 1, max_depth, max_args)
            )
        else:
            tokens.append(str(rng.integers(0, 10)))
    tokens.append(LISTOPS_CLOSE)
    return tokens


def _listops_generate_sample(
    rng: np.random.Generator,
    seq_length: int,
    max_depth: int = 6,
    max_args: int = 5,
) -> tuple[np.ndarray, int]:
    """Generate a single ListOps sample: (token_ids, label).

    Retries until a valid expression of appropriate length is produced.
    """
    for _ in range(200):
        tokens = _listops_generate_expression(rng, 0, max_depth, max_args)
        if len(tokens) <= seq_length:
            label = _listops_evaluate(tokens)
            # Convert to integer token IDs and pad
            ids = [_LISTOPS_VOCAB.get(t, 0) for t in tokens]
            ids = ids[:seq_length]
            ids += [0] * (seq_length - len(ids))
            return np.array(ids, dtype=np.float32), label
    # Fallback: simple expression
    tokens = [LISTOPS_OPEN, "MAX", "3", "5", LISTOPS_CLOSE]
    label = _listops_evaluate(tokens)
    ids = [_LISTOPS_VOCAB.get(t, 0) for t in tokens]
    ids = ids[:seq_length]
    ids += [0] * (seq_length - len(ids))
    return np.array(ids, dtype=np.float32), label


def generate_lra_listops(
    data_dir: Path,
    num_points: int = 10000,
    seq_length: int = 2048,
    max_depth: int = 6,
    max_args: int = 5,
    seed: int = 42,
) -> None:
    """Generate the LRA ListOps dataset.

    Produces hierarchical mathematical expressions using MIN, MAX, MEDIAN,
    and SUM_MOD operators over single-digit integers.  Each expression is
    serialized to a token sequence, padded/truncated to ``seq_length``, and
    the label is the evaluated result (0-9).

    Args:
        data_dir: Output directory for parquet / JSON files.
        num_points: Number of samples to generate.
        seq_length: Fixed sequence length (padded with 0).
        max_depth: Maximum nesting depth of expressions.
        max_args: Maximum number of arguments per operator.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    features = np.empty((num_points, seq_length), dtype=np.float32)
    labels = np.empty(num_points, dtype=np.int64)

    for i in range(num_points):
        features[i], labels[i] = _listops_generate_sample(
            rng, seq_length, max_depth, max_args
        )

    _lra_save_classification_data(
        data_dir,
        "lra_listops",
        features,
        labels,
        num_classes=10,
        additional_info={
            "data_family": "LRA",
            "lra_task": "listops",
            "sequence_length": seq_length,
            "num_classes": 10,
            "vocabulary_size": LISTOPS_VOCAB_SIZE,
            "max_depth": max_depth,
            "max_args": max_args,
            "seed": seed,
        },
    )


# =============================================================================
# Pathfinder / Path-X
# =============================================================================


def _draw_circle(img: np.ndarray, cy: int, cx: int, radius: int, value: float = 1.0) -> None:
    """Draw a filled circle on a 2-D array."""
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    img[mask] = value


def _draw_bezier_curve(
    img: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    thickness: float = 1.0,
    value: float = 1.0,
    num_steps: int = 200,
) -> None:
    """Draw a cubic Bezier curve on a 2-D array with given thickness."""
    h, w = img.shape
    t = np.linspace(0, 1, num_steps).reshape(-1, 1)
    # Cubic Bezier: B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
    pts = (
        (1 - t) ** 3 * np.array(p0)
        + 3 * (1 - t) ** 2 * t * np.array(p1)
        + 3 * (1 - t) * t ** 2 * np.array(p2)
        + t ** 3 * np.array(p3)
    )
    for y, x in pts:
        iy, ix = int(round(y)), int(round(x))
        # Draw a small square for thickness
        r = max(0, int(thickness))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = iy + dy, ix + dx
                if 0 <= ny < h and 0 <= nx < w:
                    img[ny, nx] = value


def _generate_pathfinder_sample(
    rng: np.random.Generator,
    image_size: int,
    num_distractors: int = 4,
) -> tuple[np.ndarray, int]:
    """Generate a single Pathfinder sample: (flattened_image, label).

    label=1 means the two dots are connected, label=0 means not.
    """
    img = np.zeros((image_size, image_size), dtype=np.float32)
    margin = max(3, image_size // 8)
    dot_radius = max(1, image_size // 16)

    # Place two endpoint dots far enough apart
    for _ in range(100):
        y0 = int(rng.integers(margin, image_size - margin))
        x0 = int(rng.integers(margin, image_size - margin))
        y1 = int(rng.integers(margin, image_size - margin))
        x1 = int(rng.integers(margin, image_size - margin))
        dist = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
        if dist > image_size * 0.3:
            break

    connected = int(rng.integers(0, 2))

    if connected:
        # Draw a Bezier curve connecting the two dots
        # Control points are random perturbations between the endpoints
        mid_y = (y0 + y1) / 2
        mid_x = (x0 + x1) / 2
        spread = image_size * 0.3
        cp1 = (
            float(np.clip(mid_y + rng.uniform(-spread, spread), 1, image_size - 2)),
            float(np.clip(mid_x + rng.uniform(-spread, spread), 1, image_size - 2)),
        )
        cp2 = (
            float(np.clip(mid_y + rng.uniform(-spread, spread), 1, image_size - 2)),
            float(np.clip(mid_x + rng.uniform(-spread, spread), 1, image_size - 2)),
        )
        curve_steps = max(200, image_size * 4)
        _draw_bezier_curve(
            img,
            (float(y0), float(x0)),
            cp1,
            cp2,
            (float(y1), float(x1)),
            thickness=0.0,
            value=0.5,
            num_steps=curve_steps,
        )

    # Draw distractor curves
    for _ in range(num_distractors):
        dy0 = int(rng.integers(1, image_size - 1))
        dx0 = int(rng.integers(1, image_size - 1))
        dy1 = int(rng.integers(1, image_size - 1))
        dx1 = int(rng.integers(1, image_size - 1))
        dcp1 = (
            float(rng.integers(1, image_size - 1)),
            float(rng.integers(1, image_size - 1)),
        )
        dcp2 = (
            float(rng.integers(1, image_size - 1)),
            float(rng.integers(1, image_size - 1)),
        )
        _draw_bezier_curve(
            img,
            (float(dy0), float(dx0)),
            dcp1,
            dcp2,
            (float(dy1), float(dx1)),
            thickness=0.0,
            value=0.3,
            num_steps=max(100, image_size * 2),
        )

    # Draw the dots on top so they are clearly visible
    _draw_circle(img, y0, x0, dot_radius, value=1.0)
    _draw_circle(img, y1, x1, dot_radius, value=1.0)

    return img.flatten(), connected


def generate_lra_pathfinder(
    data_dir: Path,
    num_points: int = 10000,
    image_size: int = 32,
    num_distractors: int = 4,
    seed: int = 42,
) -> None:
    """Generate the LRA Pathfinder dataset.

    Creates binary images with two highlighted dots.  Positive samples have
    a smooth curve connecting the dots; negative samples do not.  Images are
    flattened in raster-scan order to produce sequences of length
    ``image_size ** 2``.

    Args:
        data_dir: Output directory.
        num_points: Number of samples.
        image_size: Width/height of the square image (default 32).
        num_distractors: Number of distractor curves per image.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    seq_length = image_size * image_size
    features = np.empty((num_points, seq_length), dtype=np.float32)
    labels = np.empty(num_points, dtype=np.int64)

    for i in range(num_points):
        features[i], labels[i] = _generate_pathfinder_sample(
            rng, image_size, num_distractors
        )

    _lra_save_classification_data(
        data_dir,
        "lra_pathfinder",
        features,
        labels,
        num_classes=2,
        additional_info={
            "data_family": "LRA",
            "lra_task": "pathfinder",
            "sequence_length": seq_length,
            "image_size": image_size,
            "num_classes": 2,
            "num_distractors": num_distractors,
            "seed": seed,
        },
    )


def generate_lra_pathx(
    data_dir: Path,
    num_points: int = 2000,
    image_size: int = 128,
    num_distractors: int = 8,
    seed: int = 42,
) -> None:
    """Generate the LRA Path-X dataset (extended Pathfinder at 128x128).

    Same algorithm as Pathfinder but at higher resolution, producing
    sequences of length 16384.  Default ``num_points`` is lower because
    each sample is substantially larger.

    Args:
        data_dir: Output directory.
        num_points: Number of samples.
        image_size: Width/height (default 128).
        num_distractors: Number of distractor curves per image.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    seq_length = image_size * image_size
    features = np.empty((num_points, seq_length), dtype=np.float32)
    labels = np.empty(num_points, dtype=np.int64)

    for i in range(num_points):
        features[i], labels[i] = _generate_pathfinder_sample(
            rng, image_size, num_distractors
        )

    _lra_save_classification_data(
        data_dir,
        "lra_pathx",
        features,
        labels,
        num_classes=2,
        additional_info={
            "data_family": "LRA",
            "lra_task": "pathx",
            "sequence_length": seq_length,
            "image_size": image_size,
            "num_classes": 2,
            "num_distractors": num_distractors,
            "seed": seed,
        },
    )


# =============================================================================
# Image (CIFAR-10)
# =============================================================================


def generate_lra_image(
    data_dir: Path,
    num_points: int = 10000,
    seed: int = 42,
) -> None:
    """Generate the LRA Image dataset from CIFAR-10.

    Downloads CIFAR-10 via torchvision, converts to grayscale, flattens
    each 32x32 image in raster-scan order to produce sequences of length
    1024 with 10-class labels.

    Args:
        data_dir: Output directory.
        num_points: Number of samples.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)

    # Download CIFAR-10 (raw PIL images, no transform)
    external_dir = str(Path(data_dir).parent.parent / "data" / "external")
    cifar = tv_datasets.CIFAR10(root=external_dir, train=True, download=True)

    seq_length = 32 * 32
    features = np.empty((num_points, seq_length), dtype=np.float32)
    labels = np.empty(num_points, dtype=np.int64)

    indices = rng.integers(0, len(cifar), size=num_points)
    for i, idx in enumerate(indices):
        img, label = cifar[int(idx)]
        # img is a PIL Image (RGB) — convert to numpy
        img_arr = np.array(img, dtype=np.float32)  # (32, 32, 3)
        # Convert to grayscale: 0.2989*R + 0.5870*G + 0.1140*B
        gray = 0.2989 * img_arr[:, :, 0] + 0.5870 * img_arr[:, :, 1] + 0.1140 * img_arr[:, :, 2]
        # Normalize to [0, 1]
        gray = gray / 255.0
        features[i] = gray.flatten()
        labels[i] = label

    _lra_save_classification_data(
        data_dir,
        "lra_image",
        features,
        labels,
        num_classes=10,
        additional_info={
            "data_family": "LRA",
            "lra_task": "image",
            "sequence_length": seq_length,
            "image_size": 32,
            "num_classes": 10,
            "source": "CIFAR-10",
            "seed": seed,
        },
    )


# =============================================================================
# Text (IMDB)
# =============================================================================

_IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def _download_and_parse_imdb(
    cache_dir: Path,
    rng: np.random.Generator,
    num_points: int,
    seq_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Download and parse the IMDB dataset into byte-level sequences.

    Returns (features, labels) where features has shape (num_points, seq_length)
    and labels has shape (num_points,).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = cache_dir / "aclImdb_v1.tar.gz"

    # Download if not cached
    if not tarball_path.exists():
        print(f"Downloading IMDB dataset from {_IMDB_URL} ...")
        resp = requests.get(_IMDB_URL, stream=True, timeout=300)
        resp.raise_for_status()
        with open(tarball_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        print("Download complete.")

    # Parse the tarball: read pos/ and neg/ review text files from train split
    texts: list[str] = []
    text_labels: list[int] = []

    with tarfile.open(tarball_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # Use both train and test splits to have enough data
            parts = member.name.split("/")
            # Expect: aclImdb/{train,test}/{pos,neg}/XXXX_Y.txt
            if len(parts) < 4:
                continue
            split = parts[1]  # train or test
            sentiment = parts[2]  # pos or neg
            if split not in ("train", "test"):
                continue
            if sentiment not in ("pos", "neg"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            text = f.read().decode("utf-8", errors="replace")
            texts.append(text)
            text_labels.append(1 if sentiment == "pos" else 0)

    # Shuffle and subsample
    combined = list(zip(texts, text_labels))
    rng.shuffle(combined)
    combined = combined[:num_points]

    features = np.zeros((len(combined), seq_length), dtype=np.float32)
    labels = np.empty(len(combined), dtype=np.int64)

    for i, (text, label) in enumerate(combined):
        # Encode as bytes (UTF-8), truncate/pad to seq_length
        byte_seq = list(text.encode("utf-8"))[:seq_length]
        features[i, : len(byte_seq)] = byte_seq
        labels[i] = label

    return features, labels


def generate_lra_text(
    data_dir: Path,
    num_points: int = 10000,
    seq_length: int = 4096,
    seed: int = 42,
) -> None:
    """Generate the LRA Text classification dataset from IMDB reviews.

    Downloads the IMDB movie review dataset, encodes each review as a
    byte-level sequence (each character mapped to its UTF-8 byte value
    0-255), and pads/truncates to ``seq_length``.

    Args:
        data_dir: Output directory.
        num_points: Number of samples.
        seq_length: Fixed sequence length in bytes (default 4096).
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    cache_dir = Path(data_dir).parent.parent / "data" / "external" / "imdb"

    features, labels = _download_and_parse_imdb(
        cache_dir, rng, num_points, seq_length
    )

    _lra_save_classification_data(
        data_dir,
        "lra_text",
        features,
        labels,
        num_classes=2,
        additional_info={
            "data_family": "LRA",
            "lra_task": "text",
            "sequence_length": seq_length,
            "num_classes": 2,
            "source": "IMDB",
            "encoding": "byte",
            "seed": seed,
        },
    )
