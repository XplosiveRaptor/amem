# amem

Replicating and exploring methods and results from associative memory papers.

The end goal is a Jupyter notebook where images can be loaded, processed and
recalled interactively, with the backend code living in plain `.py` modules at the
repo root that the notebook imports.

## Pipeline

1. **Load** — pull the Yale face dataset from the Hugging Face API (`loadyale.py`).
2. **Process** — build the orthogonal vector basis and store it, so it can be
   loaded later without recreating it.
3. **Recall** — use the basis to recall images from keys, viewing them in the
   notebook and saving them as standard image files.

Step 1 is implemented. Steps 2 and 3 are next: `basis.py`, `recall.py`, `viz.py`
and the notebook.

## 1. Loading the data

[`loadyale.py`](loadyale.py) pulls
[`AIOmarRehan/Cropped_Yale_Faces`](https://huggingface.co/datasets/AIOmarRehan/Cropped_Yale_Faces)
(CC0-1.0) — the cropped and aligned Extended Yale Face Database B — via
`huggingface_hub`.

```bash
pip install huggingface_hub pyarrow numpy pillow
python loadyale.py
```

That writes 2414 grayscale PNGs (168×192) for 38 subjects:

```
data/yale/<subject>/<subject>_<illum>.png     e.g. data/yale/yaleB01/yaleB01_P00A+000E+00.png
data/yale/manifest.csv                        index,subject,label_id,relpath,source_name,illum,width,height
```

`data/` is gitignored. Reruns are no-ops; `--force` rewrites, `--limit N` does a
quick smoke run, and `--revision` pins the dataset commit. The 38 `*_Ambient`
frames (no single light source, and some uncropped at 640×480) are excluded by
default — pass `--include-ambient` to keep them.

### From the notebook

```python
from loadyale import load_images

faces = load_images(frontal_only=True)   # one frontal-lit image per subject
faces.data      # (38, 32256) float64 — one flattened image per row
faces.shape     # (192, 168) — reshape with this rather than a hardcoded constant
faces.labels    # ['yaleB01_P00A+000E+00', ...]
faces.subjects  # ['yaleB01', ...]

subject = load_images(subjects="yaleB05", flatten=False)   # (64, 192, 168)
```

`load_images` downloads on first use, filters on the manifest (`subjects`,
`illum`, `frontal_only`, `limit`), and refuses to stack a selection with mixed
image sizes. No preprocessing happens here — resizing, flattening choices and any
Laplacian step belong to the basis stage.
