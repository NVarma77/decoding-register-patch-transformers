# Corrective full-SAE register audit

Interpretation: **generic-perturbation result after correcting SAE distribution shift**.

The target-versus-mistargeted reconstruction table is in `reconstruction_target_audit_table.csv` and `figure_reconstruction_target_audit.pdf`. The full-token SAE was trained on all 261 token positions, so its true-register reconstruction is a corrective distribution-shift check, not evidence for register-only features.

Primary full-SAE reconstruction injection produced a 0.30% mean final-patch cosine drop (95% image-level CI [0.28, 0.32]).

| Cohort | Top5 minus random span (95% CI) | Top5 minus decoder permutation (95% CI) |
| --- | --- | --- |
| P0-matched 512 | -4.807 [-5.100, -4.516] | -7.444 [-7.790, -7.108] |
| Separate 256 holdout | -4.680 [-5.120, -4.244] | -6.960 [-7.441, -6.493] |

Even with an in-distribution full-token SAE on true registers, clean top-five removal was consistently less disruptive than both matched controls on both cohorts. The result does not support selected-feature semantic causality or a register-only feature claim.
