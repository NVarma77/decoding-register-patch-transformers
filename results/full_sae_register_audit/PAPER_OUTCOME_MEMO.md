# Manuscript outcome memo

**Interpretation:** generic-perturbation result after correcting SAE distribution shift.

The full-token SAE resolves the reconstruction provenance failure but does not support selected-feature semantic causality: on the 512-image cohort, clean top-five removal is 6.792 percentage points less disruptive than the matched random-span control (95% image CI [-7.043, -6.535]) and 5.364 points less disruptive than the support/magnitude-matched decoder-permutation control ([-5.589, -5.141]). The same reversed ordering holds on the independent 256-image holdout. Because the corrective SAE was trained on all 261 tokens, it is not evidence for register-only features.

## Proposed ablation replacement paragraph

An executable provenance audit showed that every released checkpoint labeled `registers_only` selected terminal patch positions `[257:261)`, whereas the four Hugging Face DINOv2 register tokens are `[1:5)`. This explains the prior reconstruction failure: the mislabeled SAE has FVE 0.007 on true registers but 0.661 at the terminal patches where it was trained. An all-token SAE that saw true registers during training reconstructs them at FVE 0.999, and reconstruction injection alone produces only a 0.224% final-patch cosine drop. However, clean removal of the top five active contributions is less disruptive than both per-register energy-matched random decoder-span and support/magnitude-matched decoder-permutation controls, with reversed paired effects on both a 512-image primary cohort and a fixed nonoverlapping 256-image holdout. We therefore report generic sensitivity to energetic directions in the learned register-reconstructing subspace, not selected-feature semantic causality or a register-only feature hierarchy.
