# Artifact layout

The release is split deliberately so routine Git clones stay small while the
full row-level evidence and exact checkpoint weights remain versioned and
checksum-verifiable.

## Tracked in Git

- experiment and analysis source code;
- paper source, bibliography, principal figure, and compiled manuscript;
- SAE training configs and available evaluation metadata;
- result configs, summaries, cohort identifiers, figures, validation records,
  provenance manifests, and outcome memos.

## Release archive: results

`null-problem-sae-ablation-results-v0.1.0.tar.zst` restores files ignored from
ordinary Git history:

- `results/**/*.log`;
- `results/**/per_image_metrics.csv`;
- `results/**/selection_audit.json`;
- `results/**/reconstruction_fidelity.csv`.

These are generated, high-volume records. The compact summaries used to check
paper claims remain directly browsable in Git.

## Release archive: checkpoints

`null-problem-sae-ablation-checkpoints-v0.1.0.tar.zst` contains the `ae.pt`
weight file corresponding to every checkpoint config under `saes/`. There are
15 checkpoint files in release `v0.1.0`; the archive restores each weight beside
its tracked `config.json`.

The archives do not contain Hugging Face base-model weights or dataset samples.

## Build and verify

Maintainers with access to the source workspace can build deterministic
archives with:

```bash
bash scripts/build_release_assets.sh v0.1.0
```

The command emits the two archives, an inventory, and `SHA256SUMS` under the
ignored `release-assets/` directory. Verify before upload with:

```bash
cd release-assets
sha256sum -c SHA256SUMS
```

Consumers can download, verify, and extract a published release with:

```bash
bash scripts/fetch_release_assets.sh v0.1.0
```
