# ASH_COATED_OSMIUM Local Backtest

This repo now contains a notebook-first local backtest workflow for `ASH_COATED_OSMIUM` built around a small reusable Python package in [backtest](</Users/shanilshah/Desktop/coding/imc prosperity/output/jupyter-notebook/ash coated osmium trading/backtest>). It is designed for quick idea rejection and rough parameter range-finding, not for matching the official IMC website exactly.

## Structure
- [ash_backtest.ipynb](</Users/shanilshah/Desktop/coding/imc prosperity/output/jupyter-notebook/ash coated osmium trading/ash_backtest.ipynb>): primary rerun surface
- [backtest/data.py](</Users/shanilshah/Desktop/coding/imc prosperity/output/jupyter-notebook/ash coated osmium trading/backtest/data.py>): Round 1 loaders and feature prep
- [backtest/strategy.py](</Users/shanilshah/Desktop/coding/imc prosperity/output/jupyter-notebook/ash coated osmium trading/backtest/strategy.py>): local adapter for the three-regime market maker used by the submission trader
- [backtest/engine.py](</Users/shanilshah/Desktop/coding/imc prosperity/output/jupyter-notebook/ash coated osmium trading/backtest/engine.py>): replay loop, fills, accounting, sweeps, sanity checks
- [backtest/reporting.py](</Users/shanilshah/Desktop/coding/imc prosperity/output/jupyter-notebook/ash coated osmium trading/backtest/reporting.py>): CSV outputs and plots
- [tests](</Users/shanilshah/Desktop/coding/imc prosperity/output/jupyter-notebook/ash coated osmium trading/tests>): engine invariants and real-data smoke coverage

## Setup
```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run The Backtest
Notebook-first workflow:

```bash
.venv/bin/python -m ipykernel install --user --name ash-osmium-backtest
.venv/bin/jupyter lab
```

Open [ash_backtest.ipynb](</Users/shanilshah/Desktop/coding/imc prosperity/output/jupyter-notebook/ash coated osmium trading/ash_backtest.ipynb>) and run top to bottom. The first config cell contains all named parameters in one place:
- `PRODUCT`
- `BASE_PARAMS`
- `EXECUTION_PROFILES`
- `SWEEP_GRID`

The notebook writes outputs to `outputs/ash_coated_osmium/<run_label>/`.

## What To Inspect Before Strategy Work
Prioritize these in order:
1. `strict/summary.csv`: net pnl, day consistency, passive share, max and near-limit inventory pressure
2. `strict/daily_pnl.csv`: whether performance depends on one day only
3. `strict/fills.csv`: buy versus sell balance, forced flatten turnover, signed post-fill mid move
4. `strict/plots/adverse_selection.png`: whether fills age well after `+1`, `+5`, and `+10`
5. `sweep_summary.csv`: robust parameter regions ranked by worst-case pnl, not just loose-profile upside
6. `loose/summary.csv`: upside check only after the strict profile still looks credible

## Run Tests
```bash
.venv/bin/python -m pytest
```

## Limitations
- The official IMC website backtest is the source of truth.
- Local passive fills are heuristic because the public data does not expose true queue position or official matching logic.
- The strict profile is intentionally conservative and should drive tuning decisions.
- Day-end force-flat is a research convenience for comparing parameter sets, not a claim about official engine behavior.
