from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from datamodel import Observation, OrderDepth, Trade, TradingState

sns.set_theme(style="whitegrid", context="talk")
pd.options.display.max_columns = 200

# Edit these in your IDE before running the script directly.
CONFIG_ROOT_NAME = "root 3"
CONFIG_RUN_ID = None
CONFIG_TRADER_NAME = "root_trader_5.py"
CONFIG_OUTPUT_DIR = None
CONFIG_LABEL = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated result analysis for a root trader result bundle.")
    parser.add_argument("--result-json", help="Path to the result json file.")
    parser.add_argument("--result-log", help="Path to the result log file. Defaults to same stem as --result-json.")
    parser.add_argument("--trader-path", help="Path to the trader .py file used for replay.")
    parser.add_argument(
        "--output-dir",
        help="Directory for generated figures and tables. Defaults to reports/figures/<result-stem>-analysis.",
    )
    parser.add_argument("--label", help="Label used in titles/report headings.")
    return parser.parse_args()


def nonzero_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for column in columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].where(cleaned[column].notna() & (cleaned[column] != 0), np.nan)
    return cleaned


def build_order_depth(row: pd.Series) -> OrderDepth:
    depth = OrderDepth()
    for level in (1, 2, 3):
        bid_price = getattr(row, f"bid_price_{level}", np.nan)
        bid_volume = getattr(row, f"bid_volume_{level}", np.nan)
        ask_price = getattr(row, f"ask_price_{level}", np.nan)
        ask_volume = getattr(row, f"ask_volume_{level}", np.nan)
        if pd.notna(bid_price) and pd.notna(bid_volume):
            depth.buy_orders[int(bid_price)] = int(bid_volume)
        if pd.notna(ask_price) and pd.notna(ask_volume):
            depth.sell_orders[int(ask_price)] = -int(ask_volume)
    return depth


def load_trader_class(trader_path: Path):
    spec = importlib.util.spec_from_file_location("root_submission_trader_module", trader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trader from {trader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Trader


def resolve_result_json_path(results_dir: Path, configured_run_id: str | None) -> Path:
    if configured_run_id:
        return results_dir / f"{configured_run_id}.json"

    numeric_result_files = []
    for path in results_dir.glob("*.json"):
        try:
            numeric_result_files.append((int(path.stem), path))
        except ValueError:
            continue

    if not numeric_result_files:
        raise FileNotFoundError(f"No numeric result json files found in {results_dir}")

    _, highest_result_path = max(numeric_result_files, key=lambda item: item[0])
    return highest_result_path


def resolve_trader_path(result_json_path: Path, explicit_trader_path: str | None) -> Path:
    if explicit_trader_path:
        return Path(explicit_trader_path)

    result_matched_trader_path = result_json_path.with_suffix(".py")
    if result_matched_trader_path.exists():
        return result_matched_trader_path

    return ROOT / "src" / "imc_eda" / "round1" / "root" / CONFIG_TRADER_NAME


def replay_trader(
    activities: pd.DataFrame,
    submission_trades: pd.DataFrame,
    trader_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    Trader = load_trader_class(trader_path)
    replay_trader = Trader()
    trader_data = ""
    replayed_rows: list[dict] = []

    submission_trade_rows = submission_trades.to_dict("records")
    own_trades_by_ts = {}
    position_by_ts = {}
    running_position = 0
    for ts in sorted(activities["timestamp"].unique().tolist()):
        fills_this_ts = []
        for row in submission_trade_rows:
            if row["timestamp"] != ts:
                continue
            fills_this_ts.append(
                Trade(
                    symbol=row["symbol"],
                    price=int(row["price"]),
                    quantity=int(row["quantity"]),
                    buyer=row["buyer"],
                    seller=row["seller"],
                    timestamp=int(row["timestamp"]),
                )
            )
            running_position += int(row["signed_qty"])
        own_trades_by_ts[ts] = {"INTARIAN_PEPPER_ROOT": fills_this_ts} if fills_this_ts else {}
        position_by_ts[ts] = {"INTARIAN_PEPPER_ROOT": running_position} if running_position else {}

    for ts, group in activities.groupby("timestamp", sort=True):
        order_depths = {}
        mid_price = np.nan
        for row in group.itertuples(index=False):
            order_depths[row.product] = build_order_depth(row)
            if row.product == "INTARIAN_PEPPER_ROOT":
                mid_price = row.mid_price

        state = TradingState(
            trader_data,
            int(ts),
            {},
            order_depths,
            own_trades_by_ts.get(ts, {}),
            {},
            position_by_ts.get(ts, {}),
            Observation({}, {}),
        )
        orders_by_product, _, trader_data = replay_trader.run(state)
        memory = json.loads(trader_data)
        pepper = memory.get("pepper", {})
        orders = orders_by_product.get("INTARIAN_PEPPER_ROOT", [])

        replayed_rows.append(
            {
                "timestamp": int(ts),
                "mid_price": mid_price,
                "signal": pepper.get("last_signal", 0),
                "action": pepper.get("last_action", "hold"),
                "sigma": pepper.get("last_sigma"),
                "residual": pepper.get("last_residual"),
                "adjusted_residual": pepper.get("last_adjusted_residual"),
                "fair_value": pepper.get("last_fair_value"),
                "upper_threshold": pepper.get("last_upper_threshold"),
                "lower_threshold": pepper.get("last_lower_threshold"),
                "buy_k": pepper.get("last_buy_k"),
                "sell_k": pepper.get("last_sell_k"),
                "buy_aggression_level": pepper.get("buy_aggression_level", 0),
                "sell_aggression_level": pepper.get("sell_aggression_level", 0),
                "buy_miss_count": pepper.get("buy_miss_count", 0),
                "sell_miss_count": pepper.get("sell_miss_count", 0),
                "position": position_by_ts.get(ts, {}).get("INTARIAN_PEPPER_ROOT", 0),
                "order_count": len(orders),
                "buy_orders": sum(1 for order in orders if order.quantity > 0),
                "sell_orders": sum(1 for order in orders if order.quantity < 0),
                "submitted_prices": [order.price for order in orders],
                "submitted_quantities": [order.quantity for order in orders],
            }
        )

    replayed = pd.DataFrame(replayed_rows)
    replay_summary = pd.DataFrame(
        [
            {
                "ticks": len(replayed),
                "ticks_with_orders": int((replayed["order_count"] > 0).sum()),
                "submitted_orders": int(replayed["order_count"].sum()),
                "buy_signal_share": float((replayed["signal"] > 0).mean()),
                "sell_signal_share": float((replayed["signal"] < 0).mean()),
                "realized_fills": len(submission_trades),
                "fill_per_submitted_order": (
                    float(len(submission_trades) / replayed["order_count"].sum())
                    if replayed["order_count"].sum() > 0
                    else 0.0
                ),
                "max_short_position": int(replayed["position"].min()),
            }
        ]
    ).round(4)

    return replayed, replay_summary


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def list_contains_side(value: object, side: str) -> bool:
    if pd.isna(value):
        return False
    return side in str(value).split(", ")


def first_order_price(submitted_prices: object, side: str, submitted_quantities: object) -> float:
    if not isinstance(submitted_prices, list) or not isinstance(submitted_quantities, list):
        return np.nan
    side_prices = [
        price
        for price, quantity in zip(submitted_prices, submitted_quantities)
        if (side == "buy" and quantity > 0) or (side == "sell" and quantity < 0)
    ]
    if not side_prices:
        return np.nan
    return float(max(side_prices) if side == "buy" else min(side_prices))


def classify_signal_reason(row: pd.Series) -> str:
    signal_side = row["threshold_cross"]
    if signal_side not in {"buy", "sell"}:
        return "no_signal"

    action = row["action"]
    filled = list_contains_side(row["realized_fill_action"], signal_side)
    position = float(row.get("position", 0) or 0)
    fair_value = row.get("fair_value")
    best_bid = row.get("bid_price_1")
    best_ask = row.get("ask_price_1")
    order_distance = row.get("distance_to_fill")

    if action == signal_side and filled:
        return f"{signal_side}_signal_filled"

    if signal_side == "buy":
        if action != "buy":
            if position >= 80:
                return "buy_signal_blocked_position_limit"
            if pd.notna(best_bid) and pd.notna(fair_value) and best_bid + 1 > fair_value:
                return "buy_signal_blocked_quote_above_fair"
            return "buy_signal_blocked_other_logic"
        if pd.notna(order_distance) and order_distance > 0:
            return "buy_signal_unfilled_distance_to_ask"
        return "buy_signal_unfilled_other"

    if action != "sell":
        if position <= 0:
            return "sell_signal_blocked_no_inventory"
        if pd.notna(best_ask) and pd.notna(fair_value) and best_ask - 1 < fair_value:
            return "sell_signal_blocked_quote_below_fair"
        return "sell_signal_blocked_other_logic"
    if pd.notna(order_distance) and order_distance > 0:
        return "sell_signal_unfilled_distance_to_bid"
    return "sell_signal_unfilled_other"


def write_interactive_quote_chart(replayed: pd.DataFrame, label: str, output_path: Path) -> None:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=replayed["timestamp"], y=replayed["fair_value"], mode="lines", name="fair value")
    )
    figure.add_trace(
        go.Scatter(x=replayed["timestamp"], y=replayed["bid_price_1"], mode="lines", name="best bid")
    )
    figure.add_trace(
        go.Scatter(x=replayed["timestamp"], y=replayed["ask_price_1"], mode="lines", name="best ask")
    )
    figure.add_trace(
        go.Scatter(
            x=replayed["timestamp"],
            y=replayed["submitted_buy_price"],
            mode="markers",
            name="our bid",
            marker={"color": "seagreen", "size": 7, "symbol": "triangle-up"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=replayed["timestamp"],
            y=replayed["submitted_sell_price"],
            mode="markers",
            name="our ask",
            marker={"color": "crimson", "size": 7, "symbol": "triangle-down"},
        )
    )
    figure.update_layout(
        title=f"{label} quotes vs fair value",
        xaxis_title="Timestamp",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
    )
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)


def write_report(report_path: Path, label: str, summary: dict, replay_summary: pd.DataFrame, findings: list[str]) -> None:
    lines = [
        f"# {label} Result Analysis",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Reported profit: `{summary['reported_profit']}`",
        f"- Submission trades: `{summary['submission_trades']}`",
        f"- Submission symbols: `{summary['submission_symbols']}`",
        f"- Terminal position: `{summary['terminal_position']}`",
        f"- Replay trader: `{summary['replay_trader_path']}`",
        "",
        "## Replay Summary",
        "",
    ]
    for key, value in replay_summary.iloc[0].to_dict().items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Findings", ""])
    lines.extend(f"- {item}" for item in findings)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    default_results_dir = ROOT / "data" / "round 1" / "results" / CONFIG_ROOT_NAME
    result_json_path = (
        Path(args.result_json)
        if args.result_json
        else resolve_result_json_path(default_results_dir, CONFIG_RUN_ID)
    )
    result_log_path = (
        Path(args.result_log)
        if args.result_log
        else result_json_path.with_suffix(".log")
    )
    trader_path = resolve_trader_path(result_json_path, args.trader_path)
    default_label = f"{result_json_path.parent.name} {result_json_path.stem}"
    label = args.label or CONFIG_LABEL or default_label
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(CONFIG_OUTPUT_DIR)
        if CONFIG_OUTPUT_DIR is not None
        else ROOT / "reports" / "figures" / f"{result_json_path.parent.name.replace(' ', '-').lower()}-{result_json_path.stem}-analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    result_json = json.loads(result_json_path.read_text(encoding="utf-8"))
    result_log = json.loads(result_log_path.read_text(encoding="utf-8"))

    activities = (
        pd.read_csv(StringIO(result_json["activitiesLog"]), sep=";")
        .sort_values(["timestamp", "product"])
        .reset_index(drop=True)
    )
    graph = pd.read_csv(StringIO(result_json["graphLog"]), sep=";").rename(columns={"value": "pnl"})
    trade_history = pd.DataFrame(result_log["tradeHistory"]).sort_values(["timestamp", "symbol", "price"]).reset_index(drop=True)
    submission_trades = trade_history[
        (trade_history["buyer"] == "SUBMISSION") | (trade_history["seller"] == "SUBMISSION")
    ].copy()
    submission_trades["side"] = np.where(submission_trades["buyer"] == "SUBMISSION", "buy", "sell")
    submission_trades["signed_qty"] = np.where(
        submission_trades["buyer"] == "SUBMISSION",
        submission_trades["quantity"],
        -submission_trades["quantity"],
    )
    submission_trades["cash_flow"] = -submission_trades["signed_qty"] * submission_trades["price"]
    submission_trades["cum_position"] = submission_trades["signed_qty"].cumsum()

    summary = {
        "status": result_json["status"],
        "reported_profit": float(result_json["profit"]),
        "total_market_trades": int(len(trade_history)),
        "submission_trades": int(len(submission_trades)),
        "submission_symbols": ", ".join(sorted(submission_trades["symbol"].unique().tolist())),
        "terminal_position": int(submission_trades["signed_qty"].sum()),
        "replay_trader_path": str(trader_path.relative_to(ROOT) if trader_path.is_relative_to(ROOT) else trader_path),
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.lineplot(data=graph, x="timestamp", y="pnl", ax=ax, linewidth=2)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"{label} PnL path")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("PnL (XIRECS)")
    fig.savefig(output_dir / "pnl-path.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    quote_columns = ["timestamp", "product", "bid_price_1", "ask_price_1", "mid_price", "profit_and_loss"]
    executed = submission_trades.rename(columns={"symbol": "product"}).merge(
        activities[quote_columns], on=["timestamp", "product"], how="left"
    )
    executed["fill_vs_bid"] = executed["price"] - executed["bid_price_1"]
    executed["fill_vs_ask"] = executed["ask_price_1"] - executed["price"]

    pepper_quotes = activities[activities["product"] == "INTARIAN_PEPPER_ROOT"].copy()
    pepper_quotes_plot = nonzero_frame(pepper_quotes, ["bid_price_1", "ask_price_1", "mid_price"])
    executed_plot = nonzero_frame(executed, ["price", "bid_price_1", "ask_price_1", "mid_price"])
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.lineplot(data=pepper_quotes_plot, x="timestamp", y="mid_price", ax=ax, label="mid price", linewidth=2, color="red")
    if not executed_plot.empty:
        sns.scatterplot(data=executed_plot, x="timestamp", y="price", hue="side", style="side", s=110, ax=ax)
    ax.set_title("INTARIAN_PEPPER_ROOT price with realized submission fills")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    fig.savefig(output_dir / "fills-vs-price.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    inventory_path = executed[["timestamp", "signed_qty"]].copy()
    inventory_path["position"] = inventory_path["signed_qty"].cumsum()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    if not inventory_path.empty:
        axes[0].step(inventory_path["timestamp"], inventory_path["position"], where="post", linewidth=2)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Submission inventory path")
    axes[0].set_ylabel("Position")
    sns.lineplot(data=graph, x="timestamp", y="pnl", ax=axes[1], linewidth=2)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("PnL path")
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel("PnL")
    fig.savefig(output_dir / "inventory-and-pnl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    replayed, replay_summary = replay_trader(activities, submission_trades, trader_path)
    fill_action_by_ts = (
        executed.groupby("timestamp")["side"].agg(lambda values: ", ".join(sorted(pd.unique(values)))).rename("realized_fill_action")
        if not executed.empty
        else pd.Series(dtype="object", name="realized_fill_action")
    )
    replayed = replayed.merge(fill_action_by_ts, on="timestamp", how="left")
    replayed["realized_fill_action"] = replayed["realized_fill_action"].fillna("none")
    replayed = replayed.merge(
        pepper_quotes[["timestamp", "bid_price_1", "ask_price_1"]],
        on="timestamp",
        how="left",
    )
    replayed["submitted_buy_price"] = replayed.apply(
        lambda row: first_order_price(row["submitted_prices"], "buy", row["submitted_quantities"]),
        axis=1,
    )
    replayed["submitted_sell_price"] = replayed.apply(
        lambda row: first_order_price(row["submitted_prices"], "sell", row["submitted_quantities"]),
        axis=1,
    )
    replayed["buy_distance_to_fill"] = np.where(
        replayed["buy_orders"] > 0,
        np.maximum(replayed["ask_price_1"] - replayed["submitted_buy_price"], 0),
        np.nan,
    )
    replayed["sell_distance_to_fill"] = np.where(
        replayed["sell_orders"] > 0,
        np.maximum(replayed["submitted_sell_price"] - replayed["bid_price_1"], 0),
        np.nan,
    )
    replayed["distance_to_fill"] = np.where(
        replayed["action"] == "buy",
        replayed["buy_distance_to_fill"],
        np.where(replayed["action"] == "sell", replayed["sell_distance_to_fill"], np.nan),
    )

    for column in ["residual", "adjusted_residual", "lower_threshold", "upper_threshold", "buy_k", "sell_k", "sigma"]:
        replayed[column] = pd.to_numeric(replayed[column], errors="coerce")

    threshold_ready = replayed[["adjusted_residual", "lower_threshold", "upper_threshold"]].notna().all(axis=1)
    replayed["threshold_cross"] = "hold"
    replayed.loc[threshold_ready & (replayed["adjusted_residual"] < replayed["lower_threshold"]), "threshold_cross"] = "buy"
    replayed.loc[threshold_ready & (replayed["adjusted_residual"] > replayed["upper_threshold"]), "threshold_cross"] = "sell"
    replayed["decision_match"] = replayed["threshold_cross"] == replayed["action"]

    replayed_plot = nonzero_frame(replayed, ["mid_price", "fair_value"])
    buy_points_plot = nonzero_frame(executed[executed["side"] == "buy"], ["price"])
    sell_points_plot = nonzero_frame(executed[executed["side"] == "sell"], ["price"])
    def save_signal_diagnostics(
        residual_column: str,
        residual_label: str,
        chart_title: str,
        output_name: str,
    ) -> None:
        action_points = replayed.loc[replayed["action"] != "hold", ["timestamp", residual_column, "action"]].copy()

        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
        sns.lineplot(data=replayed_plot, x="timestamp", y="mid_price", ax=axes[0], label="mid price", linewidth=2)
        sns.lineplot(data=replayed_plot, x="timestamp", y="fair_value", ax=axes[0], label="fair value", linewidth=2)
        if not buy_points_plot.empty:
            sns.scatterplot(data=buy_points_plot, x="timestamp", y="price", color="seagreen", s=110, ax=axes[0], label="realized buys")
        if not sell_points_plot.empty:
            sns.scatterplot(data=sell_points_plot, x="timestamp", y="price", color="crimson", s=110, ax=axes[0], label="realized sells")
        axes[0].set_title("Mid price, fair value, and realized fills")
        axes[0].set_ylabel("Price")

        sns.lineplot(data=replayed, x="timestamp", y=residual_column, ax=axes[1], linewidth=1.8, label=residual_label)
        sns.lineplot(data=replayed, x="timestamp", y="upper_threshold", ax=axes[1], linestyle="--", linewidth=1.4, label="sell threshold")
        sns.lineplot(data=replayed, x="timestamp", y="lower_threshold", ax=axes[1], linestyle="--", linewidth=1.4, label="buy threshold")
        if not action_points.empty:
            sns.scatterplot(
                data=action_points,
                x="timestamp",
                y=residual_column,
                hue="action",
                palette={"buy": "seagreen", "sell": "crimson"},
                s=60,
                ax=axes[1],
                legend=True,
            )
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set_title(chart_title)
        axes[1].set_ylabel("Residual")

        distance_plot = replayed.loc[replayed["action"] != "hold", ["timestamp", "distance_to_fill", "action"]].copy()
        if not distance_plot.empty:
            sns.scatterplot(
                data=distance_plot,
                x="timestamp",
                y="distance_to_fill",
                hue="action",
                palette={"buy": "seagreen", "sell": "crimson"},
                s=55,
                ax=axes[2],
            )
        axes[2].axhline(0, color="black", linewidth=1)
        axes[2].set_title("Submitted order distance to fill")
        axes[2].set_xlabel("Timestamp")
        axes[2].set_ylabel("Ticks from opposite top of book")
        fig.savefig(output_dir / output_name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    save_signal_diagnostics(
        residual_column="residual",
        residual_label="raw residual",
        chart_title="Raw residual vs live thresholds with submitted action",
        output_name="signal-diagnostics-raw-residual.png",
    )
    save_signal_diagnostics(
        residual_column="adjusted_residual",
        residual_label="adjusted residual",
        chart_title="Adjusted residual vs live thresholds with submitted action",
        output_name="signal-diagnostics-adjusted-residual.png",
    )
    replayed["signal_reason"] = replayed.apply(classify_signal_reason, axis=1)
    signal_reason_summary = (
        replayed.loc[replayed["threshold_cross"].isin(["buy", "sell"]), ["threshold_cross", "signal_reason"]]
        .value_counts()
        .rename("count")
        .reset_index()
        .sort_values(["threshold_cross", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    write_interactive_quote_chart(replayed, label, output_dir / "interactive-quotes.html")

    pepper_path = pepper_quotes[["timestamp", "mid_price"]].copy().reset_index(drop=True)
    timestamp_to_index = {timestamp: index for index, timestamp in enumerate(pepper_path["timestamp"])}
    post_fill_rows = []
    for row in executed.itertuples(index=False):
        idx = timestamp_to_index[row.timestamp]
        entry = {
            "timestamp": row.timestamp,
            "side": row.side,
            "price": row.price,
            "quantity": row.quantity,
            "mid_price": row.mid_price,
            "signed_qty": row.signed_qty,
        }
        for horizon in (1, 5, 10, 20):
            future_idx = min(idx + horizon, len(pepper_path) - 1)
            future_mid = float(pepper_path.iloc[future_idx]["mid_price"])
            move = future_mid - row.mid_price
            entry[f"mid_change_{horizon}"] = move
            entry[f"mtm_pnl_{horizon}"] = -row.signed_qty * move
        post_fill_rows.append(entry)
    post_fill = pd.DataFrame(post_fill_rows)
    if not post_fill.empty:
        plot_post_fill = post_fill.melt(
            id_vars=["timestamp", "side", "price", "quantity"],
            value_vars=["mtm_pnl_1", "mtm_pnl_5", "mtm_pnl_10", "mtm_pnl_20"],
            var_name="horizon",
            value_name="mtm_pnl",
        )
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.barplot(data=plot_post_fill, x="timestamp", y="mtm_pnl", hue="horizon", ax=ax)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Post-fill mark-to-market PnL by horizon")
        ax.set_xlabel("Fill timestamp")
        ax.set_ylabel("Pnl from held inventory")
        fig.savefig(output_dir / "post-fill-pnl.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    comparison_columns = [
        "timestamp",
        "residual",
        "adjusted_residual",
        "lower_threshold",
        "upper_threshold",
        "threshold_cross",
        "buy_k",
        "sell_k",
        "signal",
        "action",
        "realized_fill_action",
        "decision_match",
        "buy_orders",
        "sell_orders",
        "submitted_prices",
        "submitted_quantities",
        "submitted_buy_price",
        "submitted_sell_price",
        "buy_distance_to_fill",
        "sell_distance_to_fill",
        "distance_to_fill",
        "signal_reason",
    ]
    action_decision_view = replayed.loc[
        (replayed["order_count"] > 0) | (replayed["realized_fill_action"] != "none"),
        comparison_columns,
    ].copy()
    for column in ["residual", "adjusted_residual", "lower_threshold", "upper_threshold", "buy_k", "sell_k"]:
        if column in action_decision_view.columns:
            action_decision_view[column] = action_decision_view[column].round(4)

    fill_decision_view = replayed.loc[replayed["realized_fill_action"] != "none", comparison_columns].copy()
    for column in ["residual", "adjusted_residual", "lower_threshold", "upper_threshold", "buy_k", "sell_k"]:
        if column in fill_decision_view.columns:
            fill_decision_view[column] = fill_decision_view[column].round(4)

    findings = [
        f"The run finished with reported profit `{summary['reported_profit']}` and `{summary['submission_trades']}` submission fills.",
        f"All submission activity was in `{summary['submission_symbols']}` with terminal position `{summary['terminal_position']}`.",
        f"Replay submitted `{int(replay_summary.iloc[0]['submitted_orders'])}` orders with fill-per-order `{float(replay_summary.iloc[0]['fill_per_submitted_order'])}`.",
        "The signal diagnostics now include submitted-order distance to fill instead of miss counters.",
        "A signal reason summary table is saved for every buy and sell signal that was blocked, unfilled, or filled.",
        "An interactive Plotly quote chart with fair value, best bid/ask, and submitted prices is saved as `interactive-quotes.html`.",
    ]

    save_table(pd.DataFrame([summary]), output_dir / "summary.csv")
    save_table(replay_summary, output_dir / "replay_summary.csv")
    save_table(executed, output_dir / "executed_fills.csv")
    save_table(replayed, output_dir / "replayed_with_diagnostics.csv")
    save_table(signal_reason_summary, output_dir / "signal_reason_summary.csv")
    save_table(action_decision_view, output_dir / "action_decision_view.csv")
    save_table(fill_decision_view, output_dir / "fill_decision_view.csv")
    if not post_fill.empty:
        save_table(post_fill, output_dir / "post_fill.csv")

    write_report(output_dir / "report.md", label, summary, replay_summary, findings)

    print(f"Analysis complete: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
