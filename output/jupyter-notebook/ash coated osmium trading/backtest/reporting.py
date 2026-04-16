from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.models import BacktestResult


def _save_empty_plot(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_daily_pnl(result: BacktestResult, path: Path) -> None:
    daily = result.daily_pnl
    if daily.empty:
        _save_empty_plot(path, "Daily PnL", "No daily rows available.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = np.where(daily["net_pnl"] >= 0, "#2e7d32", "#c62828")
    ax.bar(daily["day"].astype(str), daily["net_pnl"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(f"{result.profile.name}: net PnL by day")
    ax.set_xlabel("day")
    ax.set_ylabel("net pnl")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_inventory_path(result: BacktestResult, path: Path) -> None:
    inventory = result.inventory_path
    if inventory.empty:
        _save_empty_plot(path, "Inventory Path", "No inventory rows available.")
        return

    plot_frame = inventory.copy()
    plot_frame["step"] = np.arange(len(plot_frame))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for day, day_frame in plot_frame.groupby("day", observed=True):
        ax.plot(day_frame["step"], day_frame["inventory"], label=f"day {day}")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(result.profile.inventory_limit, color="#999999", linewidth=1, linestyle="--")
    ax.axhline(-result.profile.inventory_limit, color="#999999", linewidth=1, linestyle="--")
    ax.set_title(f"{result.profile.name}: inventory path")
    ax.set_xlabel("simulation step")
    ax.set_ylabel("inventory")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_fill_mix(result: BacktestResult, path: Path) -> None:
    fills = result.fills
    if fills.empty:
        _save_empty_plot(path, "Fill Mix", "No fills recorded.")
        return

    grouped = (
        fills.groupby(["liquidity", "side"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=["passive", "taker"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped.plot(kind="bar", ax=ax, color=["#1565c0", "#ef6c00"])
    ax.set_title(f"{result.profile.name}: fill count by liquidity and side")
    ax.set_xlabel("liquidity")
    ax.set_ylabel("fill count")
    ax.legend(title="side")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_adverse_selection(result: BacktestResult, path: Path) -> None:
    fills = result.fills
    if fills.empty:
        _save_empty_plot(path, "Adverse Selection", "No fills recorded.")
        return

    grouped = (
        fills.groupby("liquidity", observed=True)
        .apply(
            lambda frame: pd.Series(
                {
                    "h1": np.average(frame["signed_mid_move_1"], weights=frame["qty"]) if frame["signed_mid_move_1"].notna().any() else np.nan,
                    "h5": np.average(frame["signed_mid_move_5"], weights=frame["qty"]) if frame["signed_mid_move_5"].notna().any() else np.nan,
                    "h10": np.average(frame["signed_mid_move_10"], weights=frame["qty"]) if frame["signed_mid_move_10"].notna().any() else np.nan,
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.22
    x = np.arange(len(grouped))
    ax.bar(x - width, grouped["h1"], width=width, label="+1")
    ax.bar(x, grouped["h5"], width=width, label="+5")
    ax.bar(x + width, grouped["h10"], width=width, label="+10")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["liquidity"])
    ax.set_title(f"{result.profile.name}: signed post-fill mid move")
    ax.set_xlabel("liquidity")
    ax.set_ylabel("signed mid move")
    ax.legend(title="horizon")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_sweep_summary(sweep_df: pd.DataFrame, path: Path) -> None:
    if sweep_df.empty:
        _save_empty_plot(path, "Sweep Robustness", "No sweep rows available.")
        return

    plot_frame = sweep_df.head(15).copy()
    plot_frame["label"] = (
        "anchor="
        + plot_frame["anchor_lookback"].astype(str)
        + " | half="
        + plot_frame["base_half_spread"].astype(str)
        + " | skew="
        + plot_frame["inventory_skew"].astype(str)
        + " | disloc="
        + plot_frame["dislocation_threshold"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(plot_frame["label"], plot_frame["worst_case_pnl"], color="#455a64")
    ax.set_title("Sweep robustness ranked by worst-case PnL")
    ax.set_xlabel("parameter set")
    ax.set_ylabel("worst-case pnl")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report(result_or_sweep: BacktestResult | pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(result_or_sweep, BacktestResult):
        result = result_or_sweep
        summary_path = out_dir / "summary.csv"
        daily_path = out_dir / "daily_pnl.csv"
        fills_path = out_dir / "fills.csv"
        inventory_path = out_dir / "inventory_path.csv"
        sanity_path = out_dir / "sanity_checks.csv"

        result.summary.to_csv(summary_path, index=False)
        result.daily_pnl.to_csv(daily_path, index=False)
        result.fills.to_csv(fills_path, index=False)
        result.inventory_path.to_csv(inventory_path, index=False)
        result.sanity_checks.to_csv(sanity_path, index=False)

        _plot_daily_pnl(result, plots_dir / "daily_pnl.png")
        _plot_inventory_path(result, plots_dir / "inventory_path.png")
        _plot_fill_mix(result, plots_dir / "fill_mix.png")
        _plot_adverse_selection(result, plots_dir / "adverse_selection.png")

        return {
            "summary": summary_path,
            "daily_pnl": daily_path,
            "fills": fills_path,
            "inventory_path": inventory_path,
            "sanity_checks": sanity_path,
            "plots_dir": plots_dir,
        }

    sweep_df = result_or_sweep
    sweep_path = out_dir / "sweep_summary.csv"
    sweep_df.to_csv(sweep_path, index=False)
    _plot_sweep_summary(sweep_df, plots_dir / "sweep_robustness.png")
    return {"sweep_summary": sweep_path, "plots_dir": plots_dir}
