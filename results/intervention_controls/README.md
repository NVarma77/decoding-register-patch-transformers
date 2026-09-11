# P0 intervention controls

This directory is a complete fixed-cohort result bundle for the layer-8
reconstruction and matched-direction controls. The primary result and proposed
paper replacement paragraph are in `OUTCOME_MEMO.md`.

Primary command:

`.venv/bin/python scripts/intervention_controls.py --allow-inferred-legacy-bottom --n-images 512 --device cpu --dtype float32 --streaming-shuffle-buffer 512`

Key artifacts: `config.json` records model/SAE revisions, hash, preprocessing,
token indexing, hardware, seeds, and the checkpoint-position provenance note;
`image_ids.json` records the ordered ImageNet validation cohort; the
per-image CSV/Parquet files contain all conditions and metrics;
`selection_audit.json` contains selected coordinates and random-control draws;
`validation_checks.json` summarizes the runtime assertions; and
`reproducibility_check.json` records the two-image same-seed replay.

The historical bottom-five source implementation was not recoverable. It is
explicitly marked as an inferred smallest-positive-active rule and must not be
presented as an exact legacy replication.
