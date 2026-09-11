# Corrective full-SAE register audit

Interpretation: **generic-perturbation result after correcting SAE distribution shift**.

The target-versus-mistargeted reconstruction table is in `reconstruction_target_audit_table.csv` and `figure_reconstruction_target_audit.pdf`. The full-token SAE was trained on all 261 token positions, so its true-register reconstruction is a corrective distribution-shift check, not evidence for register-only features.

Primary full-SAE reconstruction injection produced a 0.22% mean final-patch cosine drop (95% image-level CI [0.21, 0.24]).

| Cohort | Top5 minus random span (95% CI) | Top5 minus decoder permutation (95% CI) |
| --- | --- | --- |
| P0-matched 512 | -6.792 [-7.043, -6.535] | -5.364 [-5.589, -5.141] |
| Separate 256 holdout | -6.670 [-7.032, -6.309] | -5.016 [-5.321, -4.733] |

Even with an in-distribution full-token SAE on true registers, clean top-five removal was consistently less disruptive than both matched controls on both cohorts. The result does not support selected-feature semantic causality or a register-only feature claim.
