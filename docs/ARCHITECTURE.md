# AVM Training — Architecture

Source plan: `v-nexus-backend/specs/avm/002-model-training/plan.md` (§1–§9). Python, offline, outside the Spring app.
Output = artifact dir registered by `AvmModelAdminService` (BE). BE loads `model_*.txt` with `lightgbm4j`.

## Pipeline (one stage = one CLI command, resumable, output under `data/<run_id>/`)

```
fetch ─► crosswalk ─► clean ─► split ─► features ─► baseline
                                          │
                                          └─► train(point,q05,q95) ─► CQR ─► evaluate+gate ─► export
```

| Stage | Module | In | Out |
|---|---|---|---|
| fetch | `extract/hf.py` | pinned HF revision, crosswalk CSV commit | raw parquet |
| crosswalk | `extract/crosswalk.py` | raw | `ward_code_new`, `ward_confidence`, `crosswalk_report.json` (stop if match < 95% or ward codes ≠ DB) |
| clean | `clean/rules.py` | crosswalked | cleaned parquet (HCM-new, price/area bounds, dedup, MAD) |
| split | `split.py` | cleaned | train/val/test_a + GroupKFold ids |
| features | `features/*` | split | X matrix, `feature_schema.json`, `*_price_ref` per fold |
| baseline | `train/baseline.py` | split | baseline metrics |
| train | `train/lgbm.py`, `train/cqr.py` | X, y=log(price) | 3 LightGBM models + `conformal.json` |
| evaluate | `evaluate/*` | models, Test A, folds, Test B | `metrics.json`, gate verdict |
| export | `export/artifacts.py` | all | `artifacts/<version>/` |

## Invariants

1. **Parity with Java**: `features/normalize.py` and the category tables in `feature_schema.json` are the only encoding authority; `golden.csv` proves Python == Java.
2. **No leakage**: `*_price_ref` computed inside each fold from its train part; Test B (V-Nexus DB) never enters training.
3. **Provenance**: `manifest.json` records revisions, seed, git hash, sha256; HF-derived runs carry `NON_PRODUCTION` and BE rejects them for production.
4. **Config over code**: bounds, split dates, gates live in `configs/<run>.yaml`; unpinned HF revision/commit → `fetch` refuses.
5. **Read-only DB**: `extract/vnexus_db.py` uses a replica, never writes.
6. `data/` and `artifacts/` are gitignored (licence + PII).

## Phases → code

- T0/v0: `configs/v0_hf.yaml`, stages above on HF data.
- T1+: add `configs/v1_vnexus.yaml`; `vnexus_db.py` replaces `hf.py` + `crosswalk.py`; features gain `loc_*` (point-in-time snapshots) and the category set from plan §2.
- T3: `export` also emits `oof_predictions.csv`. T4: rerun with same config + new window.
