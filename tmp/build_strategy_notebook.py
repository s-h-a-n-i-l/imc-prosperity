from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(r"f:\Projects\imc\imc-prosperity")
NOTEBOOK_PATH = ROOT / "output" / "jupyter-notebook" / "strategy_exploration.ipynb"
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata.update(
    {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
)

cells: list = []

cells.append(
    nbf.v4.new_markdown_cell(
        """# Round 1 Strategy Exploration

This notebook turns the Round 1 EDA into a reusable signal -> strategy -> backtest workflow for:

- `ASH_COATED_OSMIUM`
- `INTARIAN_PEPPER_ROOT`

The focus is structural and interpretable:

- fixed fair vs dynamic fair
- imbalance and microprice validation
- explicit `take -> clear -> make` behavior
- side-by-side execution assumptions
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """from __future__ import annotations

import json
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from imc_eda.round1 import load_prices, load_trades
from imc_eda.round1.strategy import (
    DynamicFairMeanReverter,
    FixedFairMarketMaker,
    ImbalanceMicropriceStrategy,
    align_trades_to_quotes,
    compute_features,
    estimate_fill_probabilities,
    estimate_quote_ev,
    evaluate_trade_impact,
    make_future_targets,
    prepare_execution_context,
    prepare_quotes,
    run_backtest,
    validate_mean_reversion,
    validate_signal_monotonicity,
)

sns.set_theme(style="whitegrid", context="notebook")
pd.options.display.max_columns = 200
FIGURES_DIR = ROOT / "reports" / "figures" / "strategy-exploration"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """PRODUCTS = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]
FIXED_FAIRS = {"ASH_COATED_OSMIUM": 10000.0}
HORIZONS = (1, 5, 10, 20)
FILL_LOOKAHEAD = 5
EXECUTION_STYLES = ("aggressive", "passive", "improved")
HARD_POSITION_LIMITS = {product: 20 for product in PRODUCTS}
SOFT_POSITION_LIMITS = {product: 15 for product in PRODUCTS}

FIXED_GRID = list(product([1, 2, 3], [0, 1, 2], [2, 3, 4], [0.0, 0.25, 0.5], [1, 3, 5]))
DYNAMIC_GRID = list(product(["wall_mid", "book_mid_ema_50", "wall_mid_ema_50"], [1.0, 1.5, 2.0], [0.0, 0.25, 0.5], [1, 3, 5]))
IMBALANCE_GRID = list(product(["imbalance", "microprice_minus_mid"], [0.6, 0.7, 0.8], [1, 5, 10], [1, 3, 5]))

SHORTLIST_PER_FAMILY = 3
VALIDATION_FAIR_COLUMNS = [
    "wall_mid",
    "book_mid_rolling_mean_50",
    "book_mid_ema_50",
    "wall_mid_ema_50",
]
VALIDATION_SIGNAL_COLUMNS = ["imbalance", "microprice_minus_mid"]
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Setup

Load Round 1 quotes and trades, sort them, clean missing top-of-book values, and prepare reusable feature tables.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """raw_prices = load_prices(file_format="csv")
raw_trades = load_trades(file_format="csv")

quotes = prepare_quotes(raw_prices)
features = make_future_targets(compute_features(quotes), horizons=HORIZONS)

for product, fair in FIXED_FAIRS.items():
    features[f"{product.lower()}_fixed_fair_edge"] = np.where(
        features["product"] == product,
        features["book_mid"] - fair,
        np.nan,
    )

aligned_trades = align_trades_to_quotes(features, raw_trades)

inventory = pd.DataFrame(
    {
        "frame": ["quotes", "features", "aligned_trades"],
        "rows": [len(quotes), len(features), len(aligned_trades)],
        "products": [
            sorted(quotes["product"].unique().tolist()),
            sorted(features["product"].unique().tolist()),
            sorted(aligned_trades["product"].unique().tolist()),
        ],
    }
)
display(inventory)

quote_health = (
    features.groupby("product")
    .agg(
        rows=("book_mid", "size"),
        mean_spread=("spread", "mean"),
        missing_microprice=("microprice", lambda s: float(s.isna().mean())),
        missing_wall_mid=("wall_mid", lambda s: float(s.isna().mean())),
        quote_timestamps=("timestamp", "nunique"),
    )
    .round(4)
)
display(quote_health)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Feature Engineering

The feature set below is entirely built from reusable functions in `imc_eda.round1.strategy`.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """feature_preview_columns = [
    "day",
    "timestamp",
    "product",
    "bid_price_1",
    "ask_price_1",
    "book_mid",
    "spread",
    "microprice",
    "imbalance",
    "wall_mid",
    "book_mid_rolling_mean_50",
    "book_mid_ema_50",
    "wall_mid_ema_50",
    "future_return_5",
]

display(features[feature_preview_columns].head())

feature_summary = (
    features.groupby("product")
    .agg(
        mean_mid=("book_mid", "mean"),
        mid_std=("book_mid", "std"),
        mean_microprice_minus_mid=("microprice_minus_mid", "mean"),
        mean_imbalance=("imbalance", "mean"),
        mean_wall_minus_mid=("wall_mid_minus_mid", "mean"),
    )
    .round(4)
)
display(feature_summary)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Signal Validation

We validate fair-value edges, imbalance, and microprice before turning them into strategies.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """mean_reversion_summary, mean_reversion_buckets = validate_mean_reversion(
    features,
    fair_columns=VALIDATION_FAIR_COLUMNS,
    horizons=HORIZONS,
)

signal_summary, signal_buckets = validate_signal_monotonicity(
    features,
    signal_columns=VALIDATION_SIGNAL_COLUMNS,
    horizons=HORIZONS,
)

future_mid_comparison_rows = []
for product_name in PRODUCTS:
    product_frame = features[features["product"] == product_name].copy()
    for horizon in HORIZONS:
        future_mid_column = f"future_mid_{horizon}"
        subset = product_frame[["book_mid", "microprice", future_mid_column]].dropna()
        if subset.empty:
            continue
        future_mid_comparison_rows.append(
            {
                "product": product_name,
                "horizon": horizon,
                "mid_to_future_mid_corr": subset["book_mid"].corr(subset[future_mid_column]),
                "microprice_to_future_mid_corr": subset["microprice"].corr(subset[future_mid_column]),
            }
        )

future_mid_comparison = pd.DataFrame(future_mid_comparison_rows)
trade_impact_summary = evaluate_trade_impact(aligned_trades, horizons=HORIZONS)

display(mean_reversion_summary.sort_values(["product", "horizon", "fair_column"]).round(4))
display(signal_summary.sort_values(["product", "horizon", "signal_column"]).round(4))
display(future_mid_comparison.round(4))
display(trade_impact_summary.round(4))

mean_reversion_scores = (
    mean_reversion_summary.assign(score=lambda df: -df["correlation"])
    .sort_values(["product", "score"], ascending=[True, False])
    .reset_index(drop=True)
)
signal_scores = (
    signal_summary.assign(score=lambda df: df["correlation"].abs())
    .sort_values(["product", "score"], ascending=[True, False])
    .reset_index(drop=True)
)
display(mean_reversion_scores.groupby("product").head(5).round(4))
display(signal_scores.groupby("product").head(5).round(4))
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """selected_fairs = {
    product_name: mean_reversion_scores.loc[mean_reversion_scores["product"] == product_name, "fair_column"].head(2).tolist()
    for product_name in PRODUCTS
}
selected_signals = {
    product_name: signal_scores.loc[signal_scores["product"] == product_name, "signal_column"].head(2).tolist()
    for product_name in PRODUCTS
}

fig, axes = plt.subplots(len(PRODUCTS), 2, figsize=(15, 8), constrained_layout=True)
for row_index, product_name in enumerate(PRODUCTS):
    fair_name = selected_fairs[product_name][0]
    horizon = int(
        mean_reversion_scores.loc[
            (mean_reversion_scores["product"] == product_name) & (mean_reversion_scores["fair_column"] == fair_name),
            "horizon",
        ].iloc[0]
    )
    bucket_data = mean_reversion_buckets[
        (mean_reversion_buckets["product"] == product_name)
        & (mean_reversion_buckets["fair_column"] == fair_name)
        & (mean_reversion_buckets["horizon"] == horizon)
    ]
    sns.lineplot(data=bucket_data, x="avg_edge", y="avg_future_return", marker="o", ax=axes[row_index, 0])
    axes[row_index, 0].set_title(f"{product_name}: edge vs future return ({fair_name}, h={horizon})")
    axes[row_index, 0].set_xlabel("Average edge")
    axes[row_index, 0].set_ylabel("Average future return")

    signal_name = selected_signals[product_name][0]
    horizon_signal = int(
        signal_scores.loc[
            (signal_scores["product"] == product_name) & (signal_scores["signal_column"] == signal_name),
            "horizon",
        ].iloc[0]
    )
    bucket_signal = signal_buckets[
        (signal_buckets["product"] == product_name)
        & (signal_buckets["signal_column"] == signal_name)
        & (signal_buckets["horizon"] == horizon_signal)
    ]
    sns.lineplot(data=bucket_signal, x="avg_signal", y="avg_future_return", marker="o", ax=axes[row_index, 1])
    axes[row_index, 1].set_title(f"{product_name}: signal monotonicity ({signal_name}, h={horizon_signal})")
    axes[row_index, 1].set_xlabel("Average signal")
    axes[row_index, 1].set_ylabel("Average future return")

validation_plot_path = FIGURES_DIR / "signal-validation-overview.png"
fig.savefig(validation_plot_path, dpi=150, bbox_inches="tight")
plt.show()
validation_plot_path
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Execution Diagnostics

Estimate fill rates and quote EV under aggressive, passive, and improved quoting assumptions.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """fill_probabilities = estimate_fill_probabilities(features, aligned_trades, lookahead=FILL_LOOKAHEAD)

ev_frames = []
for product_name in PRODUCTS:
    product_frame = features[features["product"] == product_name].copy()
    if product_name in FIXED_FAIRS:
        product_frame["edge_for_ev"] = product_frame["book_mid"] - FIXED_FAIRS[product_name]
    else:
        fair_name = selected_fairs[product_name][0]
        product_frame["edge_for_ev"] = product_frame["book_mid"] - product_frame[fair_name]
    ev_frame = estimate_quote_ev(product_frame, fill_probabilities, edge_column="edge_for_ev")
    ev_frames.append(ev_frame.assign(edge_source="fixed_or_best_fair"))

execution_ev = pd.concat(ev_frames, ignore_index=True)
display(fill_probabilities.round(4))
display(execution_ev.round(4))

fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
sns.barplot(data=fill_probabilities, x="product", y="fill_probability", hue="execution_style", ax=axes[0])
axes[0].set_title("Estimated fill probabilities")
axes[0].tick_params(axis="x", rotation=15)

sns.barplot(data=execution_ev, x="product", y="ev", hue="execution_style", ax=axes[1])
axes[1].set_title("Estimated EV by execution style")
axes[1].tick_params(axis="x", rotation=15)

execution_plot_path = FIGURES_DIR / "execution-diagnostics.png"
fig.savefig(execution_plot_path, dpi=150, bbox_inches="tight")
plt.show()
execution_plot_path
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Strategy Definitions

Use explicit parameter grids, then shortlist the most plausible configs per family before full backtests so the notebook stays interactive.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def build_fixed_candidates(product_name: str) -> list[BaseException]:
    fair_value = FIXED_FAIRS.get(product_name, float(features.loc[features["product"] == product_name, "book_mid"].mean()))
    grid = pd.DataFrame(
        FIXED_GRID,
        columns=["take_width", "clear_width", "make_width", "inventory_skew", "order_size"],
    )
    grid["heuristic"] = (
        -abs(grid["take_width"] - 2)
        -abs(grid["clear_width"] - 1)
        -abs(grid["make_width"] - 3)
        -abs(grid["order_size"] - 3) / 2
        -abs(grid["inventory_skew"] - 0.25)
    )
    candidates = []
    for row in grid.sort_values("heuristic", ascending=False).head(SHORTLIST_PER_FAMILY).itertuples(index=False):
        config = {
            "fair_value": fair_value,
            "take_width": int(row.take_width),
            "clear_width": int(row.clear_width),
            "make_width": int(row.make_width),
            "soft_position_limit": SOFT_POSITION_LIMITS[product_name],
            "hard_position_limit": HARD_POSITION_LIMITS[product_name],
            "inventory_skew": float(row.inventory_skew),
            "order_size": int(row.order_size),
        }
        candidates.append(
            FixedFairMarketMaker(
                name=f"{product_name.lower()}_fixed_tw{row.take_width}_cw{row.clear_width}_mw{row.make_width}_s{row.order_size}",
                config=config,
            )
        )
    return candidates


def build_dynamic_candidates(product_name: str) -> list[BaseException]:
    product_scores = mean_reversion_scores[mean_reversion_scores["product"] == product_name].copy()
    fair_priority = product_scores.groupby("fair_column")["score"].mean().to_dict()
    grid = pd.DataFrame(
        DYNAMIC_GRID,
        columns=["fair_source", "entry_threshold", "inventory_skew", "order_size"],
    )
    grid["fair_score"] = grid["fair_source"].map(fair_priority).fillna(0.0)
    grid["heuristic"] = grid["fair_score"] - abs(grid["entry_threshold"] - 1.5) - abs(grid["order_size"] - 3) / 2 - abs(grid["inventory_skew"] - 0.25)
    candidates = []
    for row in grid.sort_values("heuristic", ascending=False).head(SHORTLIST_PER_FAMILY).itertuples(index=False):
        config = {
            "fair_source": row.fair_source,
            "entry_threshold": float(row.entry_threshold),
            "exit_threshold": float(row.entry_threshold) * 0.5,
            "soft_position_limit": SOFT_POSITION_LIMITS[product_name],
            "hard_position_limit": HARD_POSITION_LIMITS[product_name],
            "inventory_skew": float(row.inventory_skew),
            "order_size": int(row.order_size),
            "make_width": 1.0,
        }
        candidates.append(
            DynamicFairMeanReverter(
                name=f"{product_name.lower()}_dynamic_{row.fair_source}_et{row.entry_threshold}_s{row.order_size}",
                config=config,
            )
        )
    return candidates


def build_imbalance_candidates(product_name: str, reference_frame: pd.DataFrame) -> list[BaseException]:
    product_scores = signal_scores[signal_scores["product"] == product_name].copy()
    signal_priority = product_scores.groupby("signal_column")["score"].mean().to_dict()
    best_horizon_by_signal = (
        product_scores.sort_values("score", ascending=False)
        .drop_duplicates("signal_column")
        .set_index("signal_column")["horizon"]
        .to_dict()
    )
    grid = pd.DataFrame(
        IMBALANCE_GRID,
        columns=["signal_source", "quantile", "holding_horizon", "order_size"],
    )
    grid["signal_score"] = grid["signal_source"].map(signal_priority).fillna(0.0)
    grid["preferred_horizon"] = grid["signal_source"].map(best_horizon_by_signal).fillna(grid["holding_horizon"])
    grid["heuristic"] = grid["signal_score"] - abs(grid["holding_horizon"] - grid["preferred_horizon"]) / 10 - abs(grid["quantile"] - 0.7) - abs(grid["order_size"] - 3) / 2
    candidates = []
    for row in grid.sort_values("heuristic", ascending=False).head(SHORTLIST_PER_FAMILY).itertuples(index=False):
        product_frame = reference_frame[reference_frame["product"] == product_name]
        threshold = float(product_frame[row.signal_source].abs().quantile(float(row.quantile)))
        config = {
            "signal_source": row.signal_source,
            "signal_threshold": threshold,
            "holding_horizon": int(row.holding_horizon),
            "soft_position_limit": SOFT_POSITION_LIMITS[product_name],
            "hard_position_limit": HARD_POSITION_LIMITS[product_name],
            "order_size": int(row.order_size),
        }
        candidates.append(
            ImbalanceMicropriceStrategy(
                name=f"{product_name.lower()}_signal_{row.signal_source}_q{row.quantile}_h{row.holding_horizon}_s{row.order_size}",
                config=config,
            )
        )
    return candidates


def build_candidates(product_name: str, reference_frame: pd.DataFrame) -> list:
    return (
        build_fixed_candidates(product_name)
        + build_dynamic_candidates(product_name)
        + build_imbalance_candidates(product_name, reference_frame)
    )
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Backtests

Tune on days `-2` and `-1`, evaluate winners on day `0`, and also compare aggregate all-day performance.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def choose_best(summary_frame: pd.DataFrame) -> pd.Series:
    ranked = summary_frame.sort_values(
        ["total_pnl", "simple_sharpe", "max_drawdown", "trade_count"],
        ascending=[False, False, True, False],
    )
    return ranked.iloc[0]


def evaluate_candidates(product_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    product_features = features[features["product"] == product_name].copy()
    product_trades = aligned_trades[aligned_trades["product"] == product_name].copy()

    train_features = product_features[product_features["day"].isin([-2, -1])].copy()
    holdout_features = product_features[product_features["day"] == 0].copy()

    train_context = prepare_execution_context(train_features, product_trades[product_trades["day"].isin([-2, -1])], lookahead=FILL_LOOKAHEAD)
    holdout_context = prepare_execution_context(holdout_features, product_trades[product_trades["day"] == 0], lookahead=FILL_LOOKAHEAD)
    aggregate_context = prepare_execution_context(product_features, product_trades, lookahead=FILL_LOOKAHEAD)

    candidates = build_candidates(product_name, train_features)
    candidate_lookup = {candidate.name: candidate for candidate in candidates}

    tuning_rows = []
    aggregate_rows = []
    diagnostics = {}

    for candidate in candidates:
        for execution_style in EXECUTION_STYLES:
            train_result = run_backtest(train_context, product_trades, candidate, execution_style, hard_position_limit=HARD_POSITION_LIMITS[product_name], fill_horizon=FILL_LOOKAHEAD)
            train_summary = train_result["summary"] | {
                "product": product_name,
                "candidate_name": candidate.name,
                "family": candidate.__class__.__name__,
                "params": json.dumps(candidate.config, sort_keys=True),
                "evaluation_mode": "train",
            }
            tuning_rows.append(train_summary)

            aggregate_result = run_backtest(aggregate_context, product_trades, candidate, execution_style, hard_position_limit=HARD_POSITION_LIMITS[product_name], fill_horizon=FILL_LOOKAHEAD)
            aggregate_summary = aggregate_result["summary"] | {
                "product": product_name,
                "candidate_name": candidate.name,
                "family": candidate.__class__.__name__,
                "params": json.dumps(candidate.config, sort_keys=True),
                "evaluation_mode": "aggregate",
            }
            aggregate_rows.append(aggregate_summary)
            diagnostics[(product_name, candidate.name, execution_style, "aggregate")] = aggregate_result

    tuning_frame = pd.DataFrame(tuning_rows)
    aggregate_frame = pd.DataFrame(aggregate_rows)

    best_train = choose_best(tuning_frame)
    best_candidate = candidate_lookup[best_train["candidate_name"]]
    holdout_result = run_backtest(holdout_context, product_trades[product_trades["day"] == 0], best_candidate, best_train["execution_style"], hard_position_limit=HARD_POSITION_LIMITS[product_name], fill_horizon=FILL_LOOKAHEAD)
    holdout_summary = pd.DataFrame(
        [
            holdout_result["summary"]
            | {
                "product": product_name,
                "candidate_name": best_candidate.name,
                "family": best_candidate.__class__.__name__,
                "params": json.dumps(best_candidate.config, sort_keys=True),
                "evaluation_mode": "holdout",
            }
        ]
    )
    diagnostics[(product_name, best_candidate.name, best_train["execution_style"], "holdout")] = holdout_result

    best_aggregate = pd.DataFrame([choose_best(aggregate_frame)])
    return holdout_summary, best_aggregate, diagnostics


holdout_winners = []
aggregate_winners = []
diagnostic_runs = {}
for product_name in PRODUCTS:
    holdout_summary, aggregate_summary, diagnostics = evaluate_candidates(product_name)
    holdout_winners.append(holdout_summary)
    aggregate_winners.append(aggregate_summary)
    diagnostic_runs.update(diagnostics)

holdout_winners = pd.concat(holdout_winners, ignore_index=True)
aggregate_winners = pd.concat(aggregate_winners, ignore_index=True)

display(holdout_winners.round(4))
display(aggregate_winners.round(4))
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Diagnostics

Inspect PnL, position, and entry/exit behavior for the winning strategies.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """fig, axes = plt.subplots(len(PRODUCTS), 3, figsize=(18, 8), constrained_layout=True)
for row_index, product_name in enumerate(PRODUCTS):
    holdout_row = holdout_winners.loc[holdout_winners["product"] == product_name].iloc[0]
    aggregate_row = aggregate_winners.loc[aggregate_winners["product"] == product_name].iloc[0]

    holdout_run = diagnostic_runs[(product_name, holdout_row["candidate_name"], holdout_row["execution_style"], "holdout")]
    aggregate_run = diagnostic_runs[(product_name, aggregate_row["candidate_name"], aggregate_row["execution_style"], "aggregate")]

    holdout_curve = holdout_run["equity_curve"]
    aggregate_curve = aggregate_run["equity_curve"]
    aggregate_trades = aggregate_run["trades"]

    sns.lineplot(data=aggregate_curve, x="timestamp", y="marked_pnl", ax=axes[row_index, 0])
    axes[row_index, 0].set_title(f"{product_name}: aggregate PnL")
    axes[row_index, 0].set_xlabel("Timestamp")
    axes[row_index, 0].set_ylabel("Marked PnL")

    sns.lineplot(data=aggregate_curve, x="timestamp", y="position", ax=axes[row_index, 1])
    axes[row_index, 1].set_title(f"{product_name}: aggregate inventory")
    axes[row_index, 1].set_xlabel("Timestamp")
    axes[row_index, 1].set_ylabel("Position")

    price_frame = features[(features["product"] == product_name) & (features["day"] == 0)].copy()
    sns.lineplot(data=price_frame, x="timestamp", y="book_mid", ax=axes[row_index, 2], label="book_mid")
    if not holdout_run["trades"].empty:
        sns.scatterplot(
            data=holdout_run["trades"],
            x="timestamp",
            y="fill_price",
            hue="side",
            style="phase",
            ax=axes[row_index, 2],
        )
    axes[row_index, 2].set_title(f"{product_name}: holdout entries and exits")
    axes[row_index, 2].set_xlabel("Timestamp")
    axes[row_index, 2].set_ylabel("Price")

diagnostic_plot_path = FIGURES_DIR / "best-strategy-diagnostics.png"
fig.savefig(diagnostic_plot_path, dpi=150, bbox_inches="tight")
plt.show()
diagnostic_plot_path
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Output

Summarize the best holdout and aggregate strategies per product with their signals, execution choices, and tuned parameters.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """final_summary = pd.concat([holdout_winners, aggregate_winners], ignore_index=True)
final_summary = final_summary[
    [
        "evaluation_mode",
        "product",
        "family",
        "candidate_name",
        "execution_style",
        "total_pnl",
        "simple_sharpe",
        "max_drawdown",
        "trade_count",
        "avg_inventory_usage",
        "fraction_near_limit",
        "params",
    ]
].sort_values(["product", "evaluation_mode"]).reset_index(drop=True)

display(final_summary.round(4))

for product_name in PRODUCTS:
    product_summary = final_summary[final_summary["product"] == product_name]
    print(f"\\n{product_name}")
    for _, row in product_summary.iterrows():
        pnl_message = "profitable" if row["total_pnl"] > 0 else "not profitable under these assumptions"
        print(
            f\"  {row['evaluation_mode']}: {row['family']} | execution={row['execution_style']} | pnl={row['total_pnl']:.2f} | sharpe={row['simple_sharpe']:.4f} | drawdown={row['max_drawdown']:.2f} | trades={int(row['trade_count'])} | {pnl_message}\"
        )
        print(f\"    params={row['params']}\")
"""
    )
)

nb.cells = cells
nbf.write(nb, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
