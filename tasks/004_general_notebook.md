# Task 4 of 4 — General-purpose sequence notebook

## Series context

This is the last task.  It creates (or replaces) the notebook
`notebooks/4-rcp-timeseries-datasets.ipynb` with a streamlined version that uses the
simplified v2 `load_data_as_sequence(name, step_size, ...)` API.  The notebook should
demonstrate that **any** dataset with `x_y_index` can be loaded as a sequence — no
special `_seq` variant datasets are needed.

## Design goals (from `RCP-better-way`)

- **Generic** — the user picks any dataset and specifies `step_size` for that dataset.
  No hardcoded references to `MNIST_seq1`, `MNIST1D_seq1`, etc.
- **No MNIST1D ↔ MNIST duplication** — remove the separate per-dataset visualization
  sections.  One set of cells that works for any dataset.
- **Keep sections 7 & 8** — the interactive sequence builder (section 7) and the LSTM
  training section (section 8) from v1 are retained and generalised.
- **Simpler front matter** — the API reference should reflect the v2 API (no mention of
  `seq_len`/`step_size` in `save_data()`, no `generate_*_sequence()` functions).

## Notebook outline

The notebook should have the following sections.  Cell numbers are approximate; use your
judgement for logical breaks.

### Section 1 — Title & imports

Markdown cell:

```markdown
# Sequence View of `generatedata` Datasets

Any dataset that has an `x_y_index` (i.e. a pixel/label split) can be reshaped
into a time-series by calling `load_data_as_sequence(name, step_size=...)`.

- `step_size` controls how many feature values form one timestep.
- `seq_len` is computed automatically as `x_y_index // step_size`.
- No special datasets need to be generated — the reshaping happens at load time.
```

Code cell (imports):

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

from generatedata.load_data import load_data, load_data_as_sequence, data_names
```

### Section 2 — API reference

A single markdown cell documenting the v2 `load_data_as_sequence()` function:

```python
load_data_as_sequence(
    name: str,
    step_size: int,
    local: bool = False,
    data_dir: Path | str | None = None,
    label_every_step: bool = True,
) -> tuple[np.ndarray, np.ndarray]
```

Key points to document:
- `step_size` is a runtime parameter (not stored in metadata).
- `seq_len = x_y_index // step_size`.
- Raises `ValueError` if `x_y_index` is missing or not divisible by `step_size`.
- Return shapes for `label_every_step=True` and `False`.

### Section 3 — Discovering compatible datasets

Code cell that lists all datasets with `x_y_index` (compatible with sequence loading):

```python
all_names = data_names(local=True)
print(f"All datasets ({len(all_names)}):")
for n in sorted(all_names):
    try:
        info = load_data(n, local=True)["info"]
        if "x_y_index" in info:
            print(f"  {n:50s}  x_y_index={info['x_y_index']}  y_size={info['y_size']}")
        else:
            print(f"  {n:50s}  (no x_y_index — cannot load as sequence)")
    except Exception:
        print(f"  {n:50s}  (error loading)")
```

### Section 4 — Choosing a dataset and step_size

Markdown explaining the relationship between `step_size`, `seq_len`, and the data
dimensions.  Then a code cell:

```python
# ── Pick your dataset and step_size ─────────────────────────────────────────
DATASET   = "MNIST"      # any dataset with x_y_index
STEP_SIZE = 1            # must divide x_y_index evenly

X_seq, labels = load_data_as_sequence(DATASET, step_size=STEP_SIZE, local=True,
                                       label_every_step=True)
info = load_data(DATASET, local=True)["info"]

print(f"Dataset      : {DATASET}")
print(f"step_size    : {STEP_SIZE}")
print(f"seq_len      : {X_seq.shape[1]}  (= x_y_index / step_size = {info['x_y_index']} / {STEP_SIZE})")
print(f"X_seq shape  : {X_seq.shape}   (num_points, seq_len, step_size + label_dim)")
print(f"labels shape : {labels.shape}")
```

### Section 5 — Basic visualisations

A general set of visualisations that work for **any** dataset:

1. **Time-series signal plot** — plot a few samples' pixel values across timesteps.
   Works for any dataset regardless of dimensionality.
2. **Sequence heatmap** — `(seq_len, step_size)` heatmap for a single sample.
3. **Label distribution** — bar chart of class frequencies.

If `x_y_index == 784` (MNIST), optionally show the 28×28 image reconstruction.
Use a simple conditional, not a whole separate section.

### Section 6 — Error handling

Show what happens with an invalid `step_size` (not a divisor) and with a dataset
that has no `x_y_index`.

```python
# Invalid step_size
try:
    load_data_as_sequence("MNIST", step_size=3, local=True)
except ValueError as e:
    print(f"ValueError: {e}")

# Dataset without x_y_index
try:
    load_data_as_sequence("circle", step_size=1, local=True)
except ValueError as e:
    print(f"ValueError: {e}")
```

### Section 7 — Interactive Sequence Builder (keep from v1)

This is the interactive widget section from v1.  It should be adapted to use the v2
API:

- **The raw-data cache** (`get_raw_data` function) can stay largely the same — it loads
  flat data via `load_data()` and splits at `x_y_index`.  It does **not** call
  `load_data_as_sequence()` because the reshaping is done on-the-fly as the user
  changes `pixels_per_step` in the widget.
- **Dataset discovery** should find all datasets with `x_y_index` (not just `_seq`
  variants).
- **Widgets**: dataset dropdown, sample slider, pixels_per_step selector (divisors of
  `x_y_index`), up-to-step slider, show-image checkbox.
- **Three panels**: revealed pixel signal (left), heatmap (centre), image or label bar
  (right).
- The logic is identical to v1's section 7 — just ensure it doesn't reference
  `seq_len`/`step_size` from `info.json` (compute them from `x_y_index` and
  `pixels_per_step`).

### Section 8 — LSTM Training (keep from v1)

Adapted to use the v2 API.  Key changes:

- Use `load_data_as_sequence(DATASET, step_size=PIXELS_PER_STEP, local=True,
  label_every_step=False)` to get `(X_seq, labels)`.
- Or use the `get_raw_data` cache from section 7 and reshape manually (as v1 does).
- Keep the same structure: config cell, data loading cell, model definition cell,
  training loop cell, visualisation cell (loss/accuracy curves + confusion matrix).
- The config variables should be: `DATASET`, `PIXELS_PER_STEP` (= `step_size`),
  `HIDDEN_SIZE`, `NUM_LAYERS`, `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `VAL_FRAC`.

## File to create

`notebooks/4-rcp-timeseries-datasets.ipynb`

## What NOT to change

- Source files — those are covered by Tasks 1–3.
- Other notebooks (`1-rcp-visualize-data.ipynb`, etc.).

## Verification

1. Run the notebook end-to-end in a kernel where `generate_all(all=False)` has been
   executed (so MNIST, MNIST1D, and the basic datasets are available locally).
2. The interactive sequence builder should work with any dataset from the dropdown.
3. The LSTM training section should train successfully with the chosen dataset and
   `PIXELS_PER_STEP`.
4. No references to `_seq` variant dataset names, `generate_*_sequence()`, or
   `seq_len`/`step_size` metadata in `info.json`.
