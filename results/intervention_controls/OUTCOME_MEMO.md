# P0 intervention-controls outcome

Interpretation: **reconstruction-confounded result**.

Reconstruction alone produced a 9.27% mean final-patch cosine
drop (95% paired bootstrap CI 9.07 to
9.47; n=512). The legacy top-five
replacement added only 0.027 percentage points beyond
reconstruction (95% CI -0.007 to
0.062), so that increment is not separated from
zero. It must not be presented as a feature-specific causal effect.

The clean top-five intervention exceeded clean bottom-five by
2.14 points (95% CI 2.08 to
2.21), but it was smaller than the matched random
decoder-span control by 0.18 points (top minus span:
-0.18, 95% CI -0.24 to
-0.12). The active-coordinate comparison is based on
136 valid image-level matches because 376
images had zero candidate norm at a target register position; its top-minus-
active contrast was 0.16 (95% CI
0.10 to 0.23).

This run used `facebook/dinov2-with-registers-small`, the supplied SAE at
output of Hugging Face encoder block index 8 (zero-based),
the fixed seeded 512-image validation cohort, CPU float32, and one random draw
per image. The historical bottom-five source implementation is unavailable;
the run uses the explicitly inferred smallest-positive-active rule and records
that provenance limitation in `config.json` and `selection_audit.json`.

Command: `python scripts/intervention_controls.py --allow-inferred-legacy-bottom
--n-images 512 --device cpu --dtype float32 --streaming-shuffle-buffer 512`.
Elapsed time: 1151.7 seconds. Hardware:
{'platform': 'Linux-6.12.95-124.187.amzn2023.x86_64-x86_64-with-glibc2.39', 'cpu_count': 2, 'torch_version': '2.5.1+cu124', 'cuda_available': False, 'cuda_device_count': 0}.

Additional checkpoint provenance concern: the model inserts register tokens at
positions `[1:5]`, while the repository training helper selected the final four
positions for `registers_only`. Results above use the actual model register
positions and should be interpreted as a correction, not a confirmation of the
older register-SAE claim.

Proposed replacement ablation paragraph:

> On a fixed 512-image ImageNet validation cohort, replacing layer-8 register
> activations with their TopK-SAE reconstruction caused a 9.27% mean drop in
> final patch-representation cosine stability (95% paired bootstrap CI
> [9.07, 9.47]). Removing the legacy top-five SAE coordinates from that
> reconstruction changed the drop by only 0.03 percentage points relative to
> reconstruction alone (95% CI [-0.01, 0.06]). We therefore do not interpret
> the earlier large replacement effect as a semantic feature-specific causal
> effect. Clean-stream top-five removal was rank-sensitive relative to the
> bottom-five coordinates, but it did not exceed matched random decoder-span
> perturbations; this remains a reconstruction- and direction-control-limited
> observation rather than evidence for selected-feature semantics.
