# Corrective full-SAE true-register audit

This audit tests whether the prior true-register null was solely caused by applying a terminal-patch SAE out of distribution. The all-token SAE is a valid reconstruction/intervention control for true registers, but it is not evidence for register-only features.

## Cohorts and execution

- `primary/` exactly matches the completed P0 512-image ImageNet-1k validation cohort.
- `holdout/` is the next nonoverlapping 256 images in the same deterministic streaming shuffle, with seed `20260822` and shuffle buffer `512`.
- Model: `facebook/dinov2-with-registers-small`, revision `0d9846e56b43a21fa46d7f3f5070f0506a5795a9`.
- Hook: output of Hugging Face encoder block index 8 (zero-based).
- True registers: `[1:5)`; mislabeled checkpoint source positions: `[257:261)`; final-patch metric: `[5:261)`.
- Hardware/precision: CPU float32, 2 logical CPUs, PyTorch 2.5.1+cu124, CUDA unavailable.

## Reproduction

```bash
.venv/bin/python scripts/audit_registers_only_provenance.py \
  --output-dir outputs/experiments/registers_only_provenance_audit
.venv/bin/python scripts/full_sae_register_audit.py \
  --cohort primary --random-draws 8 --intervention-batch-size 8 --device cpu --dtype float32
.venv/bin/python scripts/full_sae_register_audit.py \
  --cohort holdout --random-draws 8 --intervention-batch-size 8 --device cpu --dtype float32
.venv/bin/python scripts/full_sae_register_audit.py --aggregate \
  --root-output outputs/experiments/full_sae_register_audit
```

`primary/` and `holdout/` contain exact image IDs, raw per-image CSV/Parquet metrics, fidelity data, selected coordinates, every random draw, seeds, validation audits, and summaries. The aggregate outputs are `summary.json`, `OUTCOME_MEMO.md`, `PAPER_OUTCOME_MEMO.md`, `reconstruction_target_audit_table.csv`, and the three compact figures. The strict provenance command `python scripts/audit_registers_only_provenance.py --strict` is intentionally expected to raise for the historical checkpoints.
