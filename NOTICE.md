# Third-party notices

`dictionary_learning/trainers/top_k.py` is an inference-only adaptation of the
TopK autoencoder in
[`saprmarks/dictionary_learning`](https://github.com/saprmarks/dictionary_learning),
developed by Samuel Marks, Adam Karvonen, and Aaron Mueller. The upstream code
is distributed under the MIT License; its license text is preserved in
`LICENSES/dictionary-learning-MIT.txt`.

`dictionary_learning/buffer.py` contains only the historical vision-token
selector required to execute the checkpoint-provenance audit. Its terminal-token
selection behavior is intentionally preserved because that behavior is itself
an audited result of this study.

The `paper/neurips_2026.sty` file is the official NeurIPS 2026 style file and
retains its original notices.

No third-party model weights or dataset samples are redistributed here.

Original project code is licensed under MIT as described in `LICENSE`.
Except where otherwise noted, original material under `paper/` and `results/`
and the project-produced SAE checkpoint weights are licensed under CC BY 4.0
as described in `LICENSE-PAPER-DATA.md`.
