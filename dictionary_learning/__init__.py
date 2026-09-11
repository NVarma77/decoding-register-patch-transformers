"""Minimal checkpoint-compatible subset of ``dictionary_learning``.

Only the TopK autoencoder needed by this artifact is included.  See
``NOTICE.md`` and ``LICENSES/dictionary-learning-MIT.txt`` for provenance.
"""

from .trainers.top_k import AutoEncoderTopK

__all__ = ["AutoEncoderTopK"]
