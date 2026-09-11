#!/usr/bin/env python3
"""Audit the token provenance of every released ``registers_only`` checkpoint.

The audit executes the currently shipped selector on an index tensor rather
than inferring behavior from names.  ``--strict`` intentionally raises when
the selected positions do not equal the true DINOv2-with-registers positions.
This makes it safe to use in CI before assigning register-only scope to a new
checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dictionary_learning.buffer import _select_vision_tokens


DEFAULT_OUTPUT = REPO / "outputs/experiments/registers_only_provenance_audit"
SEQUENCE_LENGTH = 261


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def actual_register_positions(num_register_tokens: int) -> list[int]:
    """DINOv2-with-registers places registers directly after the CLS token."""
    return list(range(1, 1 + num_register_tokens))


def selected_positions(model_name: str, num_register_tokens: int, sequence_length: int) -> list[int]:
    index_tensor = torch.arange(sequence_length, dtype=torch.float32).reshape(1, sequence_length, 1)
    cfg = SimpleNamespace(model_name=model_name, token_subset="registers_only", num_register_tokens=num_register_tokens)
    selected = _select_vision_tokens(index_tensor, cfg, training=True)
    return [int(value) for value in selected.reshape(-1).tolist()]


def interval(indices: list[int]) -> str:
    if not indices:
        return "[]"
    if indices == list(range(indices[0], indices[-1] + 1)):
        return f"[{indices[0]}:{indices[-1] + 1})"
    return "[" + ", ".join(str(value) for value in indices) + "]"


def checkpoint_records(saes_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for config_path in sorted(saes_root.rglob("config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run = config.get("run", {})
        if run.get("token_subset") != "registers_only":
            continue
        trainer = config.get("trainer", {})
        buffer = config.get("buffer", {})
        model_name = str(trainer.get("lm_name", ""))
        num_register_tokens = int(run.get("num_register_tokens", 4))
        actual = actual_register_positions(num_register_tokens)
        selected = selected_positions(model_name, num_register_tokens, SEQUENCE_LENGTH)
        checkpoint = config_path.parent / "ae.pt"
        record = {
            "checkpoint_dir": str(config_path.parent.relative_to(REPO)),
            "checkpoint_path": str(checkpoint.relative_to(REPO)),
            "checkpoint_exists": checkpoint.exists(),
            "checkpoint_sha256": sha256_file(checkpoint),
            "model_name": model_name,
            "model_size": model_name.rsplit("-", 1)[-1],
            "hook": trainer.get("submodule_name"),
            "hook_block_index_zero_based": trainer.get("layer"),
            "architecture_sequence_length": SEQUENCE_LENGTH,
            "actual_register_positions": actual,
            "actual_register_interval": interval(actual),
            "selected_positions": selected,
            "selected_interval": interval(selected),
            "selector_matches_architecture": selected == actual,
            "checkpoint_context_length_after_selection": buffer.get("ctx_len"),
            "selector_source": "dictionary_learning.buffer._select_vision_tokens(training=True, token_subset='registers_only')",
            "selector_expression": "hidden_states[:, -num_register_tokens:, :]",
        }
        records.append(record)
    return records


def assert_selector_matches_architecture(records: list[dict[str, Any]]) -> None:
    mismatches = [record for record in records if not record["selector_matches_architecture"]]
    if mismatches:
        examples = [f"{record['checkpoint_dir']}: selected {record['selected_interval']} != actual {record['actual_register_interval']}" for record in mismatches[:3]]
        raise AssertionError("registers_only selector provenance mismatch: " + "; ".join(examples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saes-root", type=Path, default=REPO / "saes")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true", help="Raise if selected positions disagree with the architecture")
    args = parser.parse_args()
    records = checkpoint_records(args.saes_root)
    if not records:
        raise RuntimeError(f"No registers_only checkpoint configs under {args.saes_root}")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    mismatch_count = sum(not record["selector_matches_architecture"] for record in records)
    payload = {
        "n_checkpoints": len(records),
        "n_selector_architecture_mismatches": mismatch_count,
        "architecture_rule": "For Hugging Face DINOv2-with-registers, CLS is position 0 and the four register tokens are positions [1:5).",
        "selected_positions_are_executed_from_source": True,
        "strict_assertion_command": "python scripts/audit_registers_only_provenance.py --strict",
        "records": records,
    }
    save_json(output / "registers_only_checkpoint_manifest.json", payload)
    fieldnames = list(records[0].keys())
    with (output / "registers_only_checkpoint_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in record.items()})
    assertion = {
        "status": "passed" if mismatch_count == 0 else "failed",
        "n_mismatches": mismatch_count,
        "strict_mode_raises_on_mismatch": True,
    }
    save_json(output / "selector_architecture_assertion.json", assertion)
    (output / "README.md").write_text(
        "This audit enumerates every released registers_only checkpoint config and executes the shipped selector on a 261-position index tensor. `--strict` raises when selected positions differ from true DINOv2 register positions [1:5).\n",
        encoding="utf-8",
    )
    print(f"Audited {len(records)} registers_only checkpoints; selector/architecture mismatches: {mismatch_count}")
    if args.strict:
        assert_selector_matches_architecture(records)


if __name__ == "__main__":
    main()
