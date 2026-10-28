# train-avm-opportunity

Offline AVM training pipeline (LightGBM point + quantile + CQR). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"
avm-train fetch --config configs/v0_hf.yaml
```
