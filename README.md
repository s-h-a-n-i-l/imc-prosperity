# IMC Prosperity EDA Scaffold

This workspace is set up for quick exploratory analysis against the tutorial CSV and parquet files in `TUTORIAL_ROUND_1/`.

## What is included

- `pyproject.toml`: `uv`-managed Python environment with the standard EDA stack.
- `src/imc_eda/data.py`: small helpers to load prices and trades into `pandas`.
- `output/jupyter-notebook/tutorial-round-1-eda.ipynb`: starter notebook wired to the existing dataset.
- `data/processed/`: derived tables you want to save.
- `reports/figures/`: exported charts and visuals.

## Quick start

```bash
uv sync
uv run jupyter lab
```

If you want to jump straight into the scaffolded notebook:

```bash
uv run jupyter lab output/jupyter-notebook/tutorial-round-1-eda.ipynb
```

## Notes

- The raw tutorial files stay where they are. The helper module reads from `TUTORIAL_ROUND_1/` by default and supports both `csv` and `parquet` via the `file_format` argument.
- Trade files do not contain a `day` column, so the loader infers it from the filename.
