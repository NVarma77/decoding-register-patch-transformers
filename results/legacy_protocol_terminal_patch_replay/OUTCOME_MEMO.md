# Legacy-protocol replay at the checkpoint's own operating point ([257:261))

Interpretation: same-cohort, same-protocol replay of the mistargeted
`registers_only` checkpoint's historical top-five/bottom-five ablation,
evaluated at terminal patches [257:261) -- the population its training
selector actually drew from -- rather than at true registers [1:5).

Reconstruction alone (no latent removed) produced a 0.134%
mean final-patch cosine drop (95% paired bootstrap CI
[0.127, 0.143];
n=512). Legacy top-five removal added
0.139 points beyond reconstruction (95% CI
[0.129, 0.149]); legacy
bottom-five (inferred rule) added 0.016 points beyond
reconstruction (95% CI [0.014, 0.017]).
Top-five minus bottom-five is 0.123 points (95% CI
[0.113, 0.133]).

This uses the same fixed 512-image cohort (seed 20260822), the
same checkpoint, and the same legacy_top5 / (inferred) legacy_bottom5
selection rules as the true-register replay in
`outputs/experiments/intervention_controls`. The historical bottom-five
source implementation remains unrecoverable; `legacy_bottom5` here is the
same explicitly inferred smallest-positive-mean-active-latent rule, not a
verified reproduction of the original notebook's algorithm.

Command: `python scripts/legacy_protocol_terminal_patch_replay.py --n-images 512
--device cpu --dtype float32 --streaming-shuffle-buffer 512`.
Elapsed time: 398.5 seconds.
