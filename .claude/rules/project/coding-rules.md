---
name: coding-rules
description: Quy ước code Python repo train-avm — file không được đụng, header mỗi file, docstring Process + comment đánh số cho hàm nghiệp vụ, comment WHY. LAZY, load khi chạm src/tests/scripts.
status: live
updated: 2026-10-29
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
  - "scripts/*.py"
metadata:
  type: reference
---

# Coding Rules — train-avm

Adapted from the agent-team standard (`v-nexus-extensions/.claude/config/coding-rules.md`), Python only.
Applies to every `.py` under `src/`, `tests/` and `scripts/`. Out of scope: `scripts/delegate/**` (harness-managed, overwritten on re-sync).

---

## 1. Protected files — never edit

Do not read or modify:

| Category | Examples |
|---|---|
| Environment secrets | `.env`, `.env.*` except the templates below |
| Credentials & keys | `*.pem`, `*.key`, `secrets.json`, `service-account.json` |
| Repo/IDE settings | `.git/config`, `.vscode/settings.json`, `.idea/` unless the task targets them |

Allowed: `.env.example`, `.env.sample`, `.env.template` (placeholder values only).

DB access is by `AVM_DB_*` environment variables only (see `extract/vnexus_db.py`). Never write a credential into code, config, a test, a log or a docstring. If a task needs a real secret value, stop and ask the user for it.

---

## 2. File header

Every `.py` file carries this docstring as its first statement, before any import (including `from __future__`):

```python
"""
File:        <filename>.py
Description: <one-line description>

Created at:  YYYY-MM-DD
Created by:  <author>
Updated at:  YYYY-MM-DD
Updated by:  <author>

<optional longer notes: design, invariants, caveats>
"""
```

- `Created by` / `Updated by` = the agent persona that did the work (`claude`) or the developer's handle.
- Bump `Updated at` / `Updated by` when you change a file in a way a reviewer should notice (behaviour, signature, new function). Pure typo fixes do not need it.
- The optional notes keep what the module needs a reader to know (why the design is what it is); the one-line `Description` is not a substitute for them.
- Empty `__init__.py` files get the header with `Description: Package marker for <package>.`

---

## 3. Business function comments

A function holding business logic (pipeline stage, gate, cleaning rule, encoding, calibration — not pure glue) with **3 or more logical steps** gets:

1. A docstring with a `Process:` list of every major step.
2. Matching numbered inline comments in the body.

```python
def run(cfg: dict, args=None) -> None:
    """Runs stage `evaluate`: metrics, unseen-ward check, Test B, then the gate.

    Process:
        1. Load models, conformal offsets and feature schema; refuse a config-hash mismatch.
        2. Score val and test_a and compute the metric battery.
        3. Refit per ward-fold for the unseen-ward check.
        4. Score Test B from the V-Nexus DB when configured.
        5. Evaluate hard and soft gate checks; write metrics.json.
        6. Raise GateError if a hard check failed.

    Args:
        cfg: resolved run config.
        args: parsed CLI args (unused by this stage).

    Raises:
        GateError: a hard gate failed or inputs were built with a different config.
    """
    # 1. Load models and verify they match this config.
    ...
```

Rules:

- List every major step in the docstring; no hidden logic.
- Mark each step in the body with `# <N>. <action>`, matching the list number. Comment only the first line of a step's block.
- Fewer than 3 steps and obvious from the name: a one-line docstring is enough, no numbered list.
- Use `Args:` / `Returns:` / `Raises:` (Google style) for parameters, return and errors when they are not obvious from names and type hints.

---

## 4. Inline comments

- Comment the **WHY**, never the WHAT. Good: `# lightgbm 4.7.0 on Windows crashes in multi-threaded predict; one thread is stable.` Bad: `i += 1  # increment i`.
- Single-line `#` only. No block comments inside function bodies; the numbered step markers (rule 3) are the exception.
- Do not reference task, spec, PR or issue ids in comments; state the local rule instead.

---

## 5. Hygiene

- No dead code and no commented-out blocks.
- No unexplained TODO. Out-of-scope work goes in the hand-off note, not in the code.
- Match the surrounding style (imports, naming, type hints).
- Before declaring done: `.venv/Scripts/python.exe -m pytest -q` is green. Tests use synthetic data only; never real listings or credentials.
