# Released `registers_only` checkpoint provenance audit

The audit enumerates every released `registers_only` checkpoint configuration and executes `dictionary_learning.buffer._select_vision_tokens` on a 261-position index tensor. It does not infer scope from a directory name.

For Hugging Face DINOv2-with-registers, CLS is position 0 and the four true registers are `[1:5)`. The shipped selector returns `[257:261)` for every audited record, the final four patch positions. The manifest contains model size, hook, exact selected/actual positions, checkpoint hash, and selector expression for all 13 released configurations.

```bash
.venv/bin/python scripts/audit_registers_only_provenance.py \
  --output-dir outputs/experiments/registers_only_provenance_audit

# Expected to raise AssertionError while the released selector is mismatched.
.venv/bin/python scripts/audit_registers_only_provenance.py --strict
```

`selector_architecture_assertion.json` records the observed failed status and `strict_mode_raises_on_mismatch: true`; this is an intentional release guard, not a passing test for the historical checkpoints.
