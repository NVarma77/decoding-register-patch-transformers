# Decoding the Functional Roles of Register and High-Norm Patch Tokens in Vision Transformers

This repository contains the code, paper, compact results, and provenance
records for an anonymized research artifact studying null controls for sparse
autoencoder (SAE) ablations at DINOv2 register tokens.

> **Release status:** the repository is private while the manuscript remains
> double-blind. Keep the repository and draft release private until the review
> process permits disclosure; final citation metadata can then be added.

## What is included

- `scripts/`: intervention, control, geometry, robustness, and figure scripts.
- `dictionary_learning/`: the minimal checkpoint-compatible TopK SAE loader and
  the historical vision-token selector needed for the provenance audit.
- `utils/`: DINOv2 model-loading helpers.
- `saes/`: checkpoint configs and evaluation metadata. Weight files are kept in
  the versioned release artifact, not Git history.
- `results/`: compact summaries, configs, cohort identifiers, figures, and
  outcome memos. Bulky per-image records and logs are kept in the release
  artifact.
- `paper/`: the anonymized manuscript, bibliography, style file, and principal
  figure.

No ImageNet samples, Hugging Face base-model weights, credentials, local cache,
or virtual environment are distributed.

## Setup

Python 3.10 through 3.12 and [`uv`](https://docs.astral.sh/uv/) are supported.

```bash
uv sync --extra test --frozen
uv run pytest -q
```

The experiment scripts download base models through Hugging Face. Full reruns
also require authorized access to ImageNet-1k and suitable GPU resources.

## Fetch the full artifacts

After release `v0.1.0` has been created, authenticated GitHub CLI users can
download, verify, and unpack both archives with:

```bash
bash scripts/fetch_release_assets.sh v0.1.0
```

This restores ignored files into their expected locations:

- checkpoint weights under `saes/**/trainer_0/ae.pt`;
- raw per-image metrics, selection audits, reconstruction tables, and run logs
  under `results/`.

The draft release also provides a single
`null-problem-sae-ablation-complete-v0.1.0.zip` for offline transfer. It contains
the exact tracked GitHub snapshot, a restorable Git bundle, all compact and raw
results, and all 15 SAE checkpoint weight files. It does not contain
credentials, virtual environments, caches, ImageNet samples, or Hugging Face
base-model weights.

See [`docs/artifacts.md`](docs/artifacts.md) for the Git/release boundary and
verification procedure.

## Provenance warning

The released checkpoints named `registers_only` preserve a historical selector
that selected the final four sequence positions `[257:261)`, while Hugging Face
DINOv2-with-registers places its true register tokens at `[1:5)`. This mismatch
is a finding of the study, not a behavior to silently correct. The all-token SAE
used for the corrective true-register analysis was trained over all 261 tokens.

Run the executable audit with:

```bash
uv run python scripts/audit_registers_only_provenance.py \
  --saes-root saes \
  --output-dir outputs/experiments/registers_only_provenance_audit
```

Passing `--strict` is intentionally expected to raise for the historical
`registers_only` checkpoints.

## Reproducing analyses

Every experiment is exposed as a command-line program. Start with its `--help`
output, then use the exact cohorts and configurations stored under `results/`.
For example:

```bash
uv run python scripts/full_sae_register_audit.py --help
uv run python scripts/terminal_patch_matched_controls.py --help
uv run python scripts/register_subspace_geometry.py --help
uv run python scripts/llm_sink_participation_sweep.py --help
```

The paper's numerical claims are linked to machine-readable summary files; the
full row-level evidence is checksum-pinned in the release archives.

## Licenses and attribution

Original code is distributed under the [MIT License](LICENSE). Except where
otherwise noted, original paper and research-result material, including the
project-produced SAE checkpoint weights, is distributed under
[CC BY 4.0](LICENSE-PAPER-DATA.md). Third-party notices and preserved license
text are in [`NOTICE.md`](NOTICE.md) and [`LICENSES/`](LICENSES/).
