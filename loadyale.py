"""Pull the Yale face images from the Hugging Face Hub onto local disk.

Source: AIOmarRehan/Cropped_Yale_Faces (CC0-1.0) -- the cropped/aligned Extended
Yale Face Database B: 38 subjects photographed under many single light-source
directions, shipped as one parquet file.

One-time download from the command line:

    python loadyale.py

which writes

    data/yale/<subject>/<name>.png
    data/yale/manifest.csv   index,subject,label_id,relpath,source_name,illum,width,height

From a notebook, load vectors straight into memory (downloading on first use):

    from loadyale import load_images
    faces = load_images(frontal_only=True)
    faces.data     # (38, 32256) float64, one flattened image per row
    faces.shape    # (192, 168) -- reshape with this, never a hardcoded shape

Images are saved unmodified beyond a grayscale conversion; resizing, flattening
and any Laplacian preprocessing belong to the later pipeline steps.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "AIOmarRehan/Cropped_Yale_Faces"
PARQUET = "data/train-00000-of-00001.parquet"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "yale"
MANIFEST_FIELDS = [
    "index",
    "subject",
    "label_id",
    "relpath",
    "source_name",
    "illum",
    "width",
    "height",
]
INT_FIELDS = ("index", "label_id", "width", "height")

# Uncropped 640x480 whole-scene frames; they would make the image vectors ragged.
AMBIENT_MARKER = "_Ambient"
# Frontal illumination: exactly one image per subject, the canonical "normal" view.
FRONTAL = "P00A+000E+00"

PROGRESS_EVERY = 250


class YaleImages(NamedTuple):
    """A loaded selection of Yale faces.

    data:     (n, h*w) when flattened, else (n, h, w)
    labels:   source image name per row, e.g. "yaleB01_P00A+000E+00"
    subjects: subject id per row, e.g. "yaleB01"
    shape:    (h, w) of a single image -- use this to reshape, not a constant
    """

    data: np.ndarray
    labels: list[str]
    subjects: list[str]
    shape: tuple[int, int]


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #


def download_parquet(revision: str = "main") -> Path:
    """Fetch the dataset's parquet file into the local Hugging Face cache."""
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=PARQUET,
            repo_type="dataset",
            revision=revision,
        )
    except Exception as exc:  # network, auth, missing revision, ...
        raise RuntimeError(
            f"Failed to download {PARQUET} from {REPO_ID!r} (revision {revision!r}): "
            f"{type(exc).__name__}: {exc}\n"
            "Check your network connection and try again; if the repo moved, see "
            f"https://huggingface.co/datasets/{REPO_ID}"
        ) from exc
    return Path(path)


def _label_names(table) -> list[str]:
    """Recover the class-label names from the parquet's Hugging Face metadata."""
    raw = (table.schema.metadata or {}).get(b"huggingface")
    if raw:
        try:
            names = json.loads(raw)["info"]["features"]["label"]["names"]
            if isinstance(names, list) and names:
                return [str(name) for name in names]
        except (ValueError, KeyError, TypeError):
            pass
    print(
        "warning: could not read label names from parquet metadata; "
        "falling back to numeric subject ids",
        file=sys.stderr,
    )
    return []


def _subject_for(label_id: int, names: Sequence[str]) -> str:
    if 0 <= label_id < len(names):
        return names[label_id]
    return f"label{label_id:02d}"


def _illum_for(stem: str, subject: str) -> str:
    """The lighting/pose part of a filename, e.g. "P00A+000E+00"."""
    prefix = f"{subject}_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def _write_manifest(out: Path, rows: list[dict]) -> Path:
    """Write manifest.csv atomically, so an interrupted run leaves no partial file."""
    target = out / "manifest.csv"
    tmp = out / "manifest.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, target)
    return target


def fetch_yale(
    out: Path | str = DEFAULT_OUT,
    revision: str = "main",
    limit: int | None = None,
    force: bool = False,
    include_ambient: bool = False,
    quiet: bool = False,
) -> Path:
    """Download the dataset and unpack it to PNGs plus a manifest.

    Idempotent: existing PNGs are kept (and still listed in the manifest) unless
    `force` is set. Returns the manifest path.
    """
    out = Path(out)

    def log(message: str) -> None:
        if not quiet:
            print(message)

    log(f"downloading {REPO_ID} :: {PARQUET} (revision {revision}) ...")
    parquet_path = download_parquet(revision)
    log(f"  cached at {parquet_path}")

    table = pq.read_table(parquet_path)
    names = _label_names(table)
    images = table.column("image").to_pylist()
    labels = table.column("label").to_pylist()

    total = len(images)
    log(f"{total} rows in table -> {out}")
    out.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    seen: set[Path] = set()
    written = skipped = ambient = duplicate = 0

    for index in range(total):
        if limit is not None and len(manifest) >= limit:
            break

        image = images[index] or {}
        source_name = str(image.get("path") or "")
        stem = Path(source_name).stem or f"img_{index:05d}"

        if AMBIENT_MARKER in stem and not include_ambient:
            ambient += 1
            continue

        label_id = int(labels[index])
        subject = _subject_for(label_id, names)
        target = out / subject / f"{stem}.png"

        if target in seen:  # same stem twice: the dataset has one .pgm/.png pair
            duplicate += 1
            continue
        seen.add(target)

        if target.exists() and not force:
            with Image.open(target) as img:
                size = img.size
            skipped += 1
        else:
            data = image.get("bytes")
            if not data:
                print(f"warning: row {index} has no image bytes, skipping", file=sys.stderr)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(io.BytesIO(data)) as img:
                grayscale = img.convert("L")
                size = grayscale.size
                grayscale.save(target, format="PNG")
            written += 1

        manifest.append(
            {
                "index": len(manifest),
                "subject": subject,
                "label_id": label_id,
                "relpath": target.relative_to(out).as_posix(),
                "source_name": stem,
                "illum": _illum_for(stem, subject),
                "width": size[0],
                "height": size[1],
            }
        )

        if not quiet and len(manifest) % PROGRESS_EVERY == 0:
            log(f"  {len(manifest)} images")

    manifest_path = _write_manifest(out, manifest)
    subjects = len({row["subject"] for row in manifest})
    log(
        f"done: {written} written, {skipped} already present, "
        f"{ambient} ambient skipped, {duplicate} duplicates skipped"
    )
    log(f"  {len(manifest)} images across {subjects} subjects in {out}")
    log(f"  manifest: {manifest_path}")
    return manifest_path


# --------------------------------------------------------------------------- #
# loading (the notebook-facing side)
# --------------------------------------------------------------------------- #


def load_manifest(out: Path | str = DEFAULT_OUT) -> list[dict]:
    """Read manifest.csv, coercing the numeric columns to int."""
    path = Path(out) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"no manifest at {path}; download the dataset first with "
            "`python loadyale.py` or call loadyale.fetch_yale()"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in INT_FIELDS:
            row[field] = int(row[field])
    return rows


def select(
    rows: Iterable[dict],
    subjects: Sequence[str] | str | None = None,
    illum: str | None = None,
    frontal_only: bool = False,
    limit: int | None = None,
) -> list[dict]:
    """Filter manifest rows by subject, illumination code, and count."""
    if isinstance(subjects, str):
        subjects = [subjects]
    wanted = set(subjects) if subjects is not None else None

    picked: list[dict] = []
    for row in rows:
        if wanted is not None and row["subject"] not in wanted:
            continue
        if frontal_only and row["illum"] != FRONTAL:
            continue
        if illum is not None and illum not in row["illum"]:
            continue
        picked.append(row)
        if limit is not None and len(picked) >= limit:
            break
    return picked


def load_images(
    out: Path | str = DEFAULT_OUT,
    subjects: Sequence[str] | str | None = None,
    illum: str | None = None,
    frontal_only: bool = False,
    limit: int | None = None,
    flatten: bool = True,
    dtype: str = "float64",
    download: bool = True,
) -> YaleImages:
    """Load a selection of Yale faces into one array.

    Downloads the dataset on first use unless `download=False`. The returned
    `shape` field carries the image dimensions so downstream code never has to
    hardcode them.
    """
    out = Path(out)
    if not (out / "manifest.csv").exists():
        if not download:
            raise FileNotFoundError(
                f"no manifest at {out / 'manifest.csv'}; run `python loadyale.py` "
                "or call load_images(..., download=True)"
            )
        fetch_yale(out=out)

    rows = select(
        load_manifest(out),
        subjects=subjects,
        illum=illum,
        frontal_only=frontal_only,
        limit=limit,
    )
    if not rows:
        raise ValueError(
            "no images matched the selection "
            f"(subjects={subjects!r}, illum={illum!r}, frontal_only={frontal_only})"
        )

    sizes = {(row["height"], row["width"]) for row in rows}
    if len(sizes) > 1:
        offenders = sorted(
            f"{row['relpath']} ({row['width']}x{row['height']})" for row in rows
        )[:5]
        raise ValueError(
            f"selection mixes image sizes {sorted(sizes)}, so it cannot form one "
            f"array; narrow the selection. First few: {', '.join(offenders)}"
        )
    shape = sizes.pop()

    arrays = []
    for row in rows:
        with Image.open(out / row["relpath"]) as img:
            arrays.append(np.asarray(img.convert("L"), dtype=dtype))

    data = np.stack(arrays)
    if flatten:
        data = data.reshape(len(arrays), -1)

    return YaleImages(
        data=data,
        labels=[row["source_name"] for row in rows],
        subjects=[row["subject"] for row in rows],
        shape=shape,
    )


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the cropped Extended Yale Face Database B to PNGs."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="stop after N images (useful for a smoke run)",
    )
    parser.add_argument("--revision", default="main", help="dataset revision (default: main)")
    parser.add_argument("--force", action="store_true", help="re-write PNGs that already exist")
    parser.add_argument(
        "--include-ambient",
        action="store_true",
        help="also save the 28 uncropped 640x480 *_Ambient frames",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        fetch_yale(
            out=args.out,
            revision=args.revision,
            limit=args.limit,
            force=args.force,
            include_ambient=args.include_ambient,
        )
    except RuntimeError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
