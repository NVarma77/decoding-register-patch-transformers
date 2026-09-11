"""Rebuild the audit cohorts from ImageNet validation parquet, one shard at a time.

The original runs drew their cohorts with ``datasets`` streaming
(``shuffle(seed=20260822, buffer_size=512).take(n)``).  Streaming is ~23 s per
image in this environment, which is unusable inside a GPU run, so cohorts are
instead rebuilt from parquet and verified by content hash: every cohort record
carries ``source_id``, a sha256 over the label and the raw RGB bytes.  Matching
on that hash is a stronger guarantee than replaying the shuffle would be -- it
proves pixel identity rather than assuming the sampler is reproducible.

Shards are fetched one at a time and the local parquet file is deleted after
scanning. Hugging Face's own content cache is never deleted by this script.
Matched images are retained as encoded JPEG bytes, not decoded arrays, keeping
a 512-image cohort near 60 MB instead of ~290 MB.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

VALIDATION_SHARDS = 14
REPO_ID = "ILSVRC/imagenet-1k"


def image_source_id(image: Image.Image, label: int) -> str:
    """Identical to scripts/full_sae_register_audit.py:image_source_id."""
    digest = hashlib.sha256()
    digest.update(str(label).encode("ascii"))
    digest.update(image.convert("RGB").tobytes())
    return f"sha256:{digest.hexdigest()}"


def _fetch_shard(index: int, workdir: Path) -> Path:
    """Download one validation shard into ``workdir`` (not the shared HF cache)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        REPO_ID,
        f"data/validation-{index:05d}-of-{VALIDATION_SHARDS:05d}.parquet",
        repo_type="dataset",
        local_dir=str(workdir),
    )
    return Path(path)


def iter_validation(
    workdir: Path, shards: Iterator[int] | None = None
) -> Iterator[tuple[bytes, int]]:
    """Yield ``(jpeg_bytes, label)`` for the validation split, deleting as we go."""
    import pyarrow.parquet as pq

    workdir.mkdir(parents=True, exist_ok=True)
    for shard_index in shards if shards is not None else range(VALIDATION_SHARDS):
        shard_path = _fetch_shard(shard_index, workdir)
        try:
            table = pq.read_table(shard_path, columns=["image", "label"])
            images = table.column("image").to_pylist()
            labels = table.column("label").to_pylist()
            logging.info("shard %d: %d rows", shard_index, len(labels))
            for payload, label in zip(images, labels):
                yield payload["bytes"], int(label)
            del table, images, labels
        finally:
            try:
                os.remove(shard_path)
            except OSError:
                logging.warning("could not delete %s", shard_path)
            logging.info("shard %d local parquet deleted", shard_index)


def _connect_cache(path: Path) -> sqlite3.Connection:
    """Open the local SQLite cache and create its schema if needed.

    The cache is a data-only format. In particular, this helper never unpickles
    caller-provided files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS images "
        "(source_id TEXT PRIMARY KEY, jpeg BLOB NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS fresh_cohorts ("
        "cache_key TEXT NOT NULL, position INTEGER NOT NULL, "
        "source_id TEXT NOT NULL, label INTEGER NOT NULL, split TEXT NOT NULL, "
        "jpeg BLOB NOT NULL, PRIMARY KEY (cache_key, position))"
    )
    return connection


def resolve_cohorts(
    image_ids_paths: list[Path], workdir: Path, cache: Path, allow_missing: int = 0
) -> dict[str, bytes]:
    """Find the JPEG bytes for every source_id across all given manifests.

    ``datasets.shuffle(seed, buffer_size)`` permutes *shard order* as well as
    filling a shuffle buffer, so a cohort drawn from the head of the shuffled
    stream is not in shard 0 -- it is in whichever shard the permutation placed
    first.  Rather than replicate that permutation, scan shards until every
    wanted hash is matched.  Results are cached so this is paid once.
    """
    wanted: set[str] = set()
    for path in image_ids_paths:
        for record in json.loads(Path(path).read_text(encoding="utf-8")):
            wanted.add(record["source_id"])

    found: dict[str, bytes] = {}
    with _connect_cache(cache) as connection:
        for source_id, jpeg in connection.execute("SELECT source_id, jpeg FROM images"):
            if source_id in wanted:
                found[str(source_id)] = bytes(jpeg)
    logging.info("cache hit: %d/%d images already resolved", len(found), len(wanted))
    # Honour the same tolerance the completeness check uses; otherwise a cohort
    # that is permanently short by one image can never satisfy the cache and every
    # invocation re-scans the whole split.
    if len(wanted) - len(found) <= allow_missing:
        return found

    scanned = 0
    for jpeg, label in iter_validation(workdir):
        scanned += 1
        if len(found) == len(wanted):
            break
        source_id = image_source_id(Image.open(io.BytesIO(jpeg)), label)
        if source_id in wanted and source_id not in found:
            found[source_id] = jpeg
            if len(found) % 128 == 0:
                logging.info("matched %d/%d (scanned %d)", len(found), len(wanted), scanned)

    # Cache whatever was resolved *before* judging completeness, so a shortfall
    # never costs a second full scan of the split. INSERT OR REPLACE preserves
    # entries previously resolved for other manifests.
    with _connect_cache(cache) as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO images(source_id, jpeg) VALUES (?, ?)",
            found.items(),
        )
        cache_size = int(connection.execute("SELECT COUNT(*) FROM images").fetchone()[0])
    logging.info(
        "resolved %d images (scanned %d rows); cache now holds %d",
        len(found),
        scanned,
        cache_size,
    )

    missing = sorted(wanted - set(found))
    if len(missing) > allow_missing:
        raise RuntimeError(
            f"{len(missing)} of {len(wanted)} cohort images not found in the "
            f"validation split (tolerance {allow_missing}); first missing: {missing[0]}"
        )
    if missing:
        logging.warning(
            "%d of %d cohort images unmatched and tolerated: %s. A handful of "
            "ImageNet JPEGs decode to different RGB bytes through different "
            "readers; any run using this cohort must report the reduced n.",
            len(missing), len(wanted), ", ".join(m[:23] for m in missing),
        )
    return found


def load_cohort(
    image_ids_path: Path, workdir: Path, cache: Path | None = None, allow_missing: int = 0
) -> list[dict[str, Any]]:
    """Return the cohort described by ``image_ids_path``, in recorded order.

    Raises if any recorded image is never found, so a silently truncated or
    reordered cohort is impossible.
    """
    image_ids_path = Path(image_ids_path)
    cache = cache or (Path(workdir) / "cohort_cache.sqlite3")
    recorded = json.loads(image_ids_path.read_text(encoding="utf-8"))
    if len({r["source_id"] for r in recorded}) != len(recorded):
        raise ValueError("Duplicate source_id in cohort manifest")

    found = resolve_cohorts([image_ids_path], Path(workdir), cache, allow_missing)

    cohort: list[dict[str, Any]] = []
    dropped = 0
    for record in recorded:
        if record["source_id"] not in found:
            dropped += 1
            continue
        image = Image.open(io.BytesIO(found[record["source_id"]])).convert("RGB")
        if image_source_id(image, int(record["label"])) != record["source_id"]:
            raise AssertionError(f"Hash mismatch after reload for {record['source_id']}")
        cohort.append({**record, "image": image})
    logging.info(
        "cohort verified: %d images, all source_id hashes match (%d dropped)", len(cohort), dropped
    )
    return cohort


def fresh_cohort(
    n_images: int, seed: int, workdir: Path, cache: Path, shard: int = 0
) -> list[dict[str, Any]]:
    """Draw a fresh, deterministic cohort from one validation shard.

    Fallback for when the recorded cohorts cannot be recovered by hash.  A fresh
    cohort is fine for the geometry sweep (which does not depend on which images
    are used) and for the subspace control (whose decisive comparisons are all
    *within* a single run, across conditions on the same images).  It is not
    interchangeable with the paper's cohorts, so anything reported from it must
    say so, and the reproduced clean-top5 effect should be checked against the
    paper's value as a distributional sanity check rather than assumed equal.
    """
    key = f"fresh:{shard}:{seed}:{n_images}"
    with _connect_cache(cache) as connection:
        cached = list(
            connection.execute(
                "SELECT source_id, label, split, jpeg FROM fresh_cohorts "
                "WHERE cache_key = ? ORDER BY position",
                (key,),
            )
        )
    if len(cached) == n_images:
        logging.info("fresh cohort cache hit (%s)", key)
        return [
            {
                "stream_position": position,
                "source_id": str(source_id),
                "label": int(label),
                "split": str(split),
                "jpeg": bytes(jpeg),
                "image": Image.open(io.BytesIO(jpeg)).convert("RGB"),
            }
            for position, (source_id, label, split, jpeg) in enumerate(cached)
        ]

    rows: list[tuple[bytes, int]] = list(iter_validation(Path(workdir), iter([shard])))
    order = np.random.default_rng(seed).permutation(len(rows))[:n_images]
    records = []
    for position, row_index in enumerate(order):
        jpeg, label = rows[int(row_index)]
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        records.append({
            "stream_position": position,
            "source_id": image_source_id(image, label),
            "label": int(label),
            "split": "validation",
            "jpeg": jpeg,
        })
    with _connect_cache(cache) as connection:
        connection.execute("DELETE FROM fresh_cohorts WHERE cache_key = ?", (key,))
        connection.executemany(
            "INSERT INTO fresh_cohorts"
            "(cache_key, position, source_id, label, split, jpeg) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    key,
                    int(record["stream_position"]),
                    str(record["source_id"]),
                    int(record["label"]),
                    str(record["split"]),
                    record["jpeg"],
                )
                for record in records
            ],
        )
    logging.info("drew fresh cohort of %d from shard %d (seed %d)", len(records), shard, seed)
    return [{**r, "image": Image.open(io.BytesIO(r["jpeg"])).convert("RGB")} for r in records]


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-ids", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()

    cohort = load_cohort(args.image_ids, args.workdir)
    logging.info("VERIFIED %d images", len(cohort))
