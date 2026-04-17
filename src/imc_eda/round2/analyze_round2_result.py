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

def resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "datamodel.py").exists() and (parent / "src").exists():
            return parent
    raise FileNotFoundError(f"Could not locate project root from {current}")


ROOT = resolve_project_root()
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from datamodel import Observation, OrderDepth, Trade, TradingState

sns.set_theme(style="whitegrid", context="talk")
pd.options.display.max_columns = 200

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

PRODUCT_CONFIG = {
    PEPPER: {
        "slug": "roots",
        "label": PEPPER,
        "fair_label": "fair value",
        "side_palette": {"buy": "seagreen", "sell": "crimson"},
    },
    OSMIUM: {
        "slug": "osmium",
        "label": OSMIUM,
        "fair_label": "anchor price",
        "side_palette": {"buy": "seagreen", "sell": "crimson"},
    },
}

CONFIG_RUN_ID = None
CONFIG_RESULTS_DIR = ROOT / "data" / "round 2" / "results"
CONFIG_OUTPUT_DIR = None
CONFIG_LABEL = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run round 2 result analysis with per-product output folders.")
    parser.add_argument("--run-id", help="Numeric run id to analyze. Uses the highest run id in the results folder if omitted.")
    parser.add_argument("--results-dir", help="Directory containing <run-id>.json/.log/.py result bundles.")
    parser.add_argument("--result-json", help="Explicit path to the round 2 result json file. Overrides --run-id and --results-dir.")
    parser.add_argument("--result-log", help="Explicit path to the round 2 result log file. Defaults to same stem as the chosen result json.")
    parser.add_argument("--trader-path", help="Explicit path to the trader .py file used for replay. Defaults to <run-id>.py in the results folder.")
    parser.add_argument(
        "--output-dir",
        help="Directory for generated figures and tables. Defaults to reports/figures/round2/<run-id>.",
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
    spec = importlib.util.spec_from_file_location("round2_submission_trader_module", trader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trader from {trader_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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
    candidate = result_json_path.with_suffix(".py")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find trader script for run {result_json_path.stem}. "
        f"Expected {candidate}. Pass --trader-path to override."
    )


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


def submitted_action(quantities: list[int]) -> str:
    has_buy = any(quantity > 0 for quantity in quantities)
    has_sell = any(quantity < 0 for quantity in quantities)
    if has_buy and has_sell:
        return "quote_both"
    if has_buy:
        return "buy"
    if has_sell:
        return "sell"
    return "hold"


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def write_interactive_quote_chart(
    replayed: pd.DataFrame,
    label: str,
    output_path: Path,
    fair_column: str,
    fair_label: str,
) -> None:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=replayed["timestamp"], y=replayed[fair_column], mode="lines", name=fair_label)
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
        title=f"{label} quotes vs {fair_label}",
        xaxis_title="Timestamp",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
    )
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)


def build_replay_row(
    product: str,
    timestamp: int,
    quote_row: pd.Series,
    orders: list,
    memory: dict,
    position: int,
) -> dict:
    submitted_prices = [int(order.price) for order in orders]
    submitted_quantities = [int(order.quantity) for order in orders]
    row = {
        "timestamp": timestamp,
        "product": product,
        "bid_price_1": quote_row.get("bid_price_1"),
        "ask_price_1": quote_row.get("ask_price_1"),
        "mid_price": quote_row.get("mid_price"),
        "profit_and_loss": quote_row.get("profit_and_loss"),
        "position": position,
        "order_count": len(orders),
        "buy_orders": sum(1 for order in orders if int(order.quantity) > 0),
        "sell_orders": sum(1 for order in orders if int(order.quantity) < 0),
        "submitted_prices": submitted_prices,
        "submitted_quantities": submitted_quantities,
        "submitted_action": submitted_action(submitted_quantities),
    }

    if product == PEPPER:
        pepper = memory.get("pepper", {})
        row.update(
            {
                "signal": pepper.get("last_signal", "hold"),
                "action": pepper.get("last_action", "hold"),
                "sigma": pepper.get("last_sigma"),
                "residual": pepper.get("last_residual"),
                "fair_value": pepper.get("last_fair_value"),
                "trend_slope": pepper.get("last_slope"),
                "trend_intercept": pepper.get("last_intercept"),
                "buy_aggression_level": pepper.get("buy_aggression_level", 0),
                "sell_aggression_level": pepper.get("sell_aggression_level", 0),
                "buy_miss_count": pepper.get("buy_miss_count", 0),
                "sell_miss_count": pepper.get("sell_miss_count", 0),
            }
        )
    else:
        osmium = memory.get("osmium", {})
        row.update(
            {
                "signal": osmium.get("last_regime", "standby"),
                "action": submitted_action(submitted_quantities),
                "fair_value": osmium.get("last_anchor_price"),
                "anchor_price": osmium.get("last_anchor_price"),
                "reservation_price": osmium.get("last_reservation_price"),
                "imbalance": osmium.get("last_imbalance"),
                "regime": osmium.get("last_regime", "standby"),
                "book_state": osmium.get("last_book_state", "empty"),
                "top_bid_depth": osmium.get("last_top_bid_depth", 0.0),
                "top_ask_depth": osmium.get("last_top_ask_depth", 0.0),
                "total_bid_depth": osmium.get("last_total_bid_depth", 0.0),
                "total_ask_depth": osmium.get("last_total_ask_depth", 0.0),
                "planned_bid": osmium.get("last_planned_bid"),
                "planned_ask": osmium.get("last_planned_ask"),
                "passive_buy_size": osmium.get("last_passive_buy_size", 0.0),
                "passive_sell_size": osmium.get("last_passive_sell_size", 0.0),
                "taker_side": osmium.get("last_taker_side"),
                "taker_qty": osmium.get("last_taker_qty", 0.0),
            }
        )
    return row


def replay_trader(
    activities: pd.DataFrame,
    submission_trades: pd.DataFrame,
    trader_path: Path,
) -> pd.DataFrame:
    Trader = load_trader_class(trader_path)
    replay_trader_instance = Trader()
    trader_data = ""
    replayed_rows: list[dict] = []

    submission_trade_rows = submission_trades.to_dict("records")
    own_trades_by_ts: dict[int, dict[str, list[Trade]]] = {}
    position_by_ts: dict[int, dict[str, int]] = {}
    running_position = {PEPPER: 0, OSMIUM: 0}

    for ts in sorted(activities["timestamp"].unique().tolist()):
        own_trades_by_ts[ts] = {}
        position_by_ts[ts] = {}
        for product in (PEPPER, OSMIUM):
            fills_this_ts = []
            for row in submission_trade_rows:
                if row["timestamp"] != ts or row["symbol"] != product:
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
                running_position[product] += int(row["signed_qty"])
            if fills_this_ts:
                own_trades_by_ts[ts][product] = fills_this_ts
            if running_position[product]:
                position_by_ts[ts][product] = running_position[product]

    for ts, group in activities.groupby("timestamp", sort=True):
        order_depths = {}
        quote_rows: dict[str, pd.Series] = {}
        for row in group.itertuples(index=False):
            order_depths[row.product] = build_order_depth(row)
            quote_rows[row.product] = pd.Series(row._asdict())

        state = TradingState(
            trader_data,
            int(ts),
            {},
            order_depths,
            own_trades_by_ts.get(int(ts), {}),
            {},
            position_by_ts.get(int(ts), {}),
            Observation({}, {}),
        )
        orders_by_product, _, trader_data = replay_trader_instance.run(state)
        memory = json.loads(trader_data)

        for product, quote_row in quote_rows.items():
            replayed_rows.append(
                build_replay_row(
                    product=product,
                    timestamp=int(ts),
                    quote_row=quote_row,
                    orders=orders_by_product.get(product, []),
                    memory=memory,
                    position=int(position_by_ts.get(int(ts), {}).get(product, 0)),
                )
            )

    return pd.DataFrame(replayed_rows).sort_values(["product", "timestamp"]).reset_index(drop=True)


def build_submission_summary(
    product: str,
    submission_trades: pd.DataFrame,
    trade_history: pd.DataFrame,
    trader_path: Path,
    result_json: dict,
) -> dict:
    product_submission = submission_trades[submission_trades["symbol"] == product].copy()
    return {
        "product": product,
        "status": result_json["status"],
        "reported_profit": float(result_json["profit"]),
        "total_market_trades": int((trade_history["symbol"] == product).sum()),
        "submission_trades": int(len(product_submission)),
        "terminal_position": int(product_submission["signed_qty"].sum()) if not product_submission.empty else 0,
        "replay_trader_path": str(trader_path.relative_to(ROOT) if trader_path.is_relative_to(ROOT) else trader_path),
    }


def build_replay_summary(replayed: pd.DataFrame, executed: pd.DataFrame) -> pd.DataFrame:
    ticks = len(replayed)
    submitted_orders = int(replayed["order_count"].sum()) if ticks else 0
    with_orders = int((replayed["order_count"] > 0).sum()) if ticks else 0
    buy_order_share = float((replayed["buy_orders"] > 0).mean()) if ticks else 0.0
    sell_order_share = float((replayed["sell_orders"] > 0).mean()) if ticks else 0.0
    replay_summary = {
        "ticks": ticks,
        "ticks_with_orders": with_orders,
        "submitted_orders": submitted_orders,
        "buy_order_share": buy_order_share,
        "sell_order_share": sell_order_share,
        "realized_fills": int(len(executed)),
        "fill_per_submitted_order": float(len(executed) / submitted_orders) if submitted_orders > 0 else 0.0,
        "max_long_position": int(replayed["position"].max()) if ticks else 0,
        "max_short_position": int(replayed["position"].min()) if ticks else 0,
    }
    if "regime" in replayed.columns:
        for regime in ("standby", "defensive", "normal", "dislocation"):
            replay_summary[f"{regime}_share"] = float((replayed["regime"] == regime).mean())
    return pd.DataFrame([replay_summary]).round(4)


def prepare_product_frames(
    product: str,
    activities: pd.DataFrame,
    submission_trades: pd.DataFrame,
    replayed_all: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quotes = activities[activities["product"] == product].copy().reset_index(drop=True)
    quote_columns = ["timestamp", "product", "bid_price_1", "ask_price_1", "mid_price", "profit_and_loss"]
    executed = submission_trades[submission_trades["symbol"] == product].rename(columns={"symbol": "product"}).merge(
        quotes[quote_columns],
        on=["timestamp", "product"],
        how="left",
    )
    executed["fill_vs_bid"] = executed["price"] - executed["bid_price_1"]
    executed["fill_vs_ask"] = executed["ask_price_1"] - executed["price"]

    replayed = replayed_all[replayed_all["product"] == product].copy().reset_index(drop=True)
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
        replayed["buy_orders"] > 0,
        replayed["buy_distance_to_fill"],
        np.where(replayed["sell_orders"] > 0, replayed["sell_distance_to_fill"], np.nan),
    )

    fill_action_by_ts = (
        executed.groupby("timestamp")["side"].agg(lambda values: ", ".join(sorted(pd.unique(values)))).rename("realized_fill_action")
        if not executed.empty
        else pd.Series(dtype="object", name="realized_fill_action")
    )
    replayed = replayed.merge(fill_action_by_ts, on="timestamp", how="left")
    replayed["realized_fill_action"] = replayed["realized_fill_action"].fillna("none")
    return quotes, executed, replayed


def save_common_plots(
    product: str,
    label: str,
    product_dir: Path,
    quotes: pd.DataFrame,
    executed: pd.DataFrame,
    replayed: pd.DataFrame,
    graph: pd.DataFrame,
) -> pd.DataFrame:
    quotes_plot = nonzero_frame(quotes, ["bid_price_1", "ask_price_1", "mid_price"])
    executed_plot = nonzero_frame(executed, ["price", "bid_price_1", "ask_price_1", "mid_price"])

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.lineplot(data=quotes_plot, x="timestamp", y="mid_price", ax=ax, label="mid price", linewidth=2, color="royalblue")
    if not replayed.empty and "fair_value" in replayed.columns:
        fair_plot = nonzero_frame(replayed, ["fair_value"])
        sns.lineplot(data=fair_plot, x="timestamp", y="fair_value", ax=ax, label=PRODUCT_CONFIG[product]["fair_label"], linewidth=2, color="darkorange")
    if not executed_plot.empty:
        sns.scatterplot(data=executed_plot, x="timestamp", y="price", hue="side", style="side", s=110, ax=ax)
    ax.set_title(f"{label} price with realized submission fills")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    fig.savefig(product_dir / "fills-vs-price.png", dpi=150, bbox_inches="tight")
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
    fig.savefig(product_dir / "inventory-and-pnl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return inventory_path


def save_pepper_diagnostics(label: str, product_dir: Path, replayed: pd.DataFrame, executed: pd.DataFrame) -> pd.DataFrame:
    replayed_plot = nonzero_frame(replayed, ["mid_price", "fair_value", "residual", "sigma"])
    buy_points_plot = nonzero_frame(executed[executed["side"] == "buy"], ["price"])
    sell_points_plot = nonzero_frame(executed[executed["side"] == "sell"], ["price"])
    action_points = replayed.loc[replayed["action"] != "hold", ["timestamp", "residual", "action"]].copy()
    action_palette = {
        "buy": "seagreen",
        "sell": "crimson",
        "quote_both": "slateblue",
        "quote_bid": "teal",
        "quote_ask": "darkred",
        "take_ask": "darkgreen",
        "take_bid": "firebrick",
    }

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    sns.lineplot(data=replayed_plot, x="timestamp", y="mid_price", ax=axes[0], label="mid price", linewidth=2)
    sns.lineplot(data=replayed_plot, x="timestamp", y="fair_value", ax=axes[0], label="fair value", linewidth=2)
    if not buy_points_plot.empty:
        sns.scatterplot(data=buy_points_plot, x="timestamp", y="price", color="seagreen", s=110, ax=axes[0], label="realized buys")
    if not sell_points_plot.empty:
        sns.scatterplot(data=sell_points_plot, x="timestamp", y="price", color="crimson", s=110, ax=axes[0], label="realized sells")
    axes[0].set_title(f"{label} mid price, fair value, and realized fills")
    axes[0].set_ylabel("Price")

    sns.lineplot(data=replayed_plot, x="timestamp", y="residual", ax=axes[1], linewidth=1.8, label="residual")
    if "sigma" in replayed_plot.columns:
        sns.lineplot(data=replayed_plot, x="timestamp", y="sigma", ax=axes[1], linestyle="--", linewidth=1.4, label="sigma")
        sigma_band = replayed_plot[["timestamp", "sigma"]].dropna().copy()
        sigma_band["neg_sigma"] = -sigma_band["sigma"]
        sns.lineplot(data=sigma_band, x="timestamp", y="neg_sigma", ax=axes[1], linestyle="--", linewidth=1.4, label="-sigma")
    if not action_points.empty:
        sns.scatterplot(
            data=action_points,
            x="timestamp",
            y="residual",
            hue="action",
            palette=action_palette,
            s=60,
            ax=axes[1],
        )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Residual vs sigma with submitted action")
    axes[1].set_ylabel("Residual")

    distance_plot = replayed.loc[replayed["submitted_action"] != "hold", ["timestamp", "distance_to_fill", "submitted_action"]].copy()
    if not distance_plot.empty:
        sns.scatterplot(
            data=distance_plot,
            x="timestamp",
            y="distance_to_fill",
            hue="submitted_action",
            s=55,
            ax=axes[2],
        )
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_title("Submitted order distance to fill")
    axes[2].set_xlabel("Timestamp")
    axes[2].set_ylabel("Ticks from opposite top of book")
    fig.savefig(product_dir / "signal-diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return replayed.loc[
        (replayed["order_count"] > 0) | (replayed["realized_fill_action"] != "none"),
        [
            "timestamp",
            "mid_price",
            "fair_value",
            "residual",
            "sigma",
            "action",
            "submitted_action",
            "realized_fill_action",
            "buy_aggression_level",
            "sell_aggression_level",
            "buy_distance_to_fill",
            "sell_distance_to_fill",
            "submitted_prices",
            "submitted_quantities",
        ],
    ].copy()


def save_osmium_diagnostics(label: str, product_dir: Path, replayed: pd.DataFrame, executed: pd.DataFrame) -> pd.DataFrame:
    replayed_plot = nonzero_frame(
        replayed,
        [
            "mid_price",
            "anchor_price",
            "reservation_price",
            "planned_bid",
            "planned_ask",
            "imbalance",
            "top_bid_depth",
            "top_ask_depth",
        ],
    )
    buy_points_plot = nonzero_frame(executed[executed["side"] == "buy"], ["price"])
    sell_points_plot = nonzero_frame(executed[executed["side"] == "sell"], ["price"])

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    sns.lineplot(data=replayed_plot, x="timestamp", y="mid_price", ax=axes[0], label="mid price", linewidth=2)
    sns.lineplot(data=replayed_plot, x="timestamp", y="anchor_price", ax=axes[0], label="anchor", linewidth=2)
    sns.lineplot(data=replayed_plot, x="timestamp", y="reservation_price", ax=axes[0], label="reservation", linewidth=2)
    if not buy_points_plot.empty:
        sns.scatterplot(data=buy_points_plot, x="timestamp", y="price", color="seagreen", s=110, ax=axes[0], label="realized buys")
    if not sell_points_plot.empty:
        sns.scatterplot(data=sell_points_plot, x="timestamp", y="price", color="crimson", s=110, ax=axes[0], label="realized sells")
    axes[0].set_title(f"{label} mid, anchor, reservation, and realized fills")
    axes[0].set_ylabel("Price")

    sns.lineplot(data=replayed_plot, x="timestamp", y="imbalance", ax=axes[1], linewidth=1.8, label="imbalance")
    axes[1].axhline(0, color="black", linewidth=1)
    regime_points = replayed.loc[replayed["regime"].notna(), ["timestamp", "imbalance", "regime"]].copy()
    if not regime_points.empty:
        sns.scatterplot(
            data=regime_points,
            x="timestamp",
            y="imbalance",
            hue="regime",
            palette={
                "standby": "gray",
                "defensive": "goldenrod",
                "normal": "steelblue",
                "dislocation": "crimson",
            },
            s=28,
            ax=axes[1],
        )
    axes[1].set_title("Imbalance and inferred regime")
    axes[1].set_ylabel("Imbalance")

    sns.lineplot(data=replayed_plot, x="timestamp", y="top_bid_depth", ax=axes[2], label="top bid depth", linewidth=1.6)
    sns.lineplot(data=replayed_plot, x="timestamp", y="top_ask_depth", ax=axes[2], label="top ask depth", linewidth=1.6)
    distance_plot = replayed.loc[replayed["submitted_action"] != "hold", ["timestamp", "distance_to_fill", "submitted_action"]].copy()
    if not distance_plot.empty:
        sns.scatterplot(data=distance_plot, x="timestamp", y="distance_to_fill", hue="submitted_action", s=55, ax=axes[2])
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_title("Top-of-book depth and submitted order distance to fill")
    axes[2].set_xlabel("Timestamp")
    axes[2].set_ylabel("Depth / ticks")
    fig.savefig(product_dir / "signal-diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, constrained_layout=True)
    sns.lineplot(data=replayed_plot, x="timestamp", y="planned_bid", ax=axes[0], label="planned bid", linewidth=1.8)
    sns.lineplot(data=replayed_plot, x="timestamp", y="planned_ask", ax=axes[0], label="planned ask", linewidth=1.8)
    sns.scatterplot(data=replayed, x="timestamp", y="submitted_buy_price", ax=axes[0], color="seagreen", s=35, label="submitted bid")
    sns.scatterplot(data=replayed, x="timestamp", y="submitted_sell_price", ax=axes[0], color="crimson", s=35, label="submitted ask")
    axes[0].set_title("Quote plan vs submitted prices")
    axes[0].set_ylabel("Price")

    sns.lineplot(data=replayed, x="timestamp", y="passive_buy_size", ax=axes[1], label="passive buy size", linewidth=1.8)
    sns.lineplot(data=replayed, x="timestamp", y="passive_sell_size", ax=axes[1], label="passive sell size", linewidth=1.8)
    if "taker_qty" in replayed.columns:
        sns.lineplot(data=replayed, x="timestamp", y="taker_qty", ax=axes[1], label="taker qty", linewidth=1.8)
    axes[1].set_title("Planned sizing")
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel("Quantity")
    fig.savefig(product_dir / "quote-plan-diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return replayed.loc[
        (replayed["order_count"] > 0) | (replayed["realized_fill_action"] != "none"),
        [
            "timestamp",
            "mid_price",
            "anchor_price",
            "reservation_price",
            "imbalance",
            "regime",
            "book_state",
            "planned_bid",
            "planned_ask",
            "passive_buy_size",
            "passive_sell_size",
            "taker_side",
            "taker_qty",
            "submitted_action",
            "realized_fill_action",
            "buy_distance_to_fill",
            "sell_distance_to_fill",
            "submitted_prices",
            "submitted_quantities",
        ],
    ].copy()


def save_post_fill_plot(product_dir: Path, quotes: pd.DataFrame, executed: pd.DataFrame) -> pd.DataFrame:
    if executed.empty:
        return pd.DataFrame()

    path = quotes[["timestamp", "mid_price"]].copy().reset_index(drop=True)
    timestamp_to_index = {timestamp: index for index, timestamp in enumerate(path["timestamp"])}
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
            future_idx = min(idx + horizon, len(path) - 1)
            future_mid = float(path.iloc[future_idx]["mid_price"])
            move = future_mid - row.mid_price
            entry[f"mid_change_{horizon}"] = move
            entry[f"mtm_pnl_{horizon}"] = -row.signed_qty * move
        post_fill_rows.append(entry)

    post_fill = pd.DataFrame(post_fill_rows)
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
    ax.set_ylabel("PnL from held inventory")
    fig.savefig(product_dir / "post-fill-pnl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return post_fill


def write_report(
    report_path: Path,
    label: str,
    summary: dict,
    replay_summary: pd.DataFrame,
    findings: list[str],
) -> None:
    lines = [
        f"# {label} Result Analysis",
        "",
        "## Summary",
        "",
        f"- Product: `{summary['product']}`",
        f"- Status: `{summary['status']}`",
        f"- Reported profit: `{summary['reported_profit']}`",
        f"- Submission trades: `{summary['submission_trades']}`",
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


def analyze_product(
    product: str,
    label: str,
    product_dir: Path,
    activities: pd.DataFrame,
    trade_history: pd.DataFrame,
    submission_trades: pd.DataFrame,
    replayed_all: pd.DataFrame,
    graph: pd.DataFrame,
    trader_path: Path,
    result_json: dict,
) -> dict:
    product_dir.mkdir(parents=True, exist_ok=True)
    quotes, executed, replayed = prepare_product_frames(product, activities, submission_trades, replayed_all)
    inventory_path = save_common_plots(product, label, product_dir, quotes, executed, replayed, graph)
    replay_summary = build_replay_summary(replayed, executed)
    summary = build_submission_summary(product, submission_trades, trade_history, trader_path, result_json)
    post_fill = save_post_fill_plot(product_dir, quotes, executed)

    if product == PEPPER:
        action_decision_view = save_pepper_diagnostics(label, product_dir, replayed, executed)
    else:
        action_decision_view = save_osmium_diagnostics(label, product_dir, replayed, executed)

    write_interactive_quote_chart(
        replayed,
        label,
        product_dir / "interactive-quotes.html",
        fair_column="fair_value",
        fair_label=PRODUCT_CONFIG[product]["fair_label"],
    )

    findings = [
        f"The run finished with reported profit `{summary['reported_profit']}` and `{summary['submission_trades']}` submission fills in `{product}`.",
        f"Replay submitted `{int(replay_summary.iloc[0]['submitted_orders'])}` orders with fill-per-order `{float(replay_summary.iloc[0]['fill_per_submitted_order'])}`.",
        f"Peak inventory ranged from `{int(replay_summary.iloc[0]['max_short_position'])}` to `{int(replay_summary.iloc[0]['max_long_position'])}`.",
        "The folder includes an interactive quote chart with the model fair price, top of book, and submitted prices.",
    ]
    if product == OSMIUM:
        findings.append(
            f"Regime shares were normal `{float(replay_summary.iloc[0].get('normal_share', 0.0))}`, defensive `{float(replay_summary.iloc[0].get('defensive_share', 0.0))}`, and dislocation `{float(replay_summary.iloc[0].get('dislocation_share', 0.0))}`."
        )
    else:
        realized_buy_share = float((executed["side"] == "buy").mean()) if not executed.empty else 0.0
        findings.append(f"Realized fill mix was buy-heavy `{realized_buy_share:.2%}` of the time.")

    save_table(pd.DataFrame([summary]), product_dir / "summary.csv")
    save_table(replay_summary, product_dir / "replay_summary.csv")
    save_table(executed, product_dir / "executed_fills.csv")
    save_table(replayed, product_dir / "replayed_with_diagnostics.csv")
    save_table(action_decision_view, product_dir / "action_decision_view.csv")
    save_table(inventory_path, product_dir / "inventory_path.csv")
    if not post_fill.empty:
        save_table(post_fill, product_dir / "post_fill.csv")

    write_report(product_dir / "report.md", label, summary, replay_summary, findings)
    return summary


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else Path(CONFIG_RESULTS_DIR)
    result_json_path = (
        Path(args.result_json)
        if args.result_json
        else resolve_result_json_path(results_dir, args.run_id or CONFIG_RUN_ID)
    )
    result_log_path = Path(args.result_log) if args.result_log else result_json_path.with_suffix(".log")
    trader_path = resolve_trader_path(result_json_path, args.trader_path)
    run_id = result_json_path.stem
    default_label = f"round 2 {run_id}"
    label = args.label or CONFIG_LABEL or default_label
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(CONFIG_OUTPUT_DIR)
        if CONFIG_OUTPUT_DIR is not None
        else ROOT / "reports" / "figures" / "round2" / run_id
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
    submission_trades["cum_position"] = submission_trades.groupby("symbol")["signed_qty"].cumsum()

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.lineplot(data=graph, x="timestamp", y="pnl", ax=ax, linewidth=2)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"{label} PnL path")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("PnL (XIRECS)")
    fig.savefig(output_dir / "pnl-path.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    replayed_all = replay_trader(activities, submission_trades, trader_path)

    summaries = []
    for product in (PEPPER, OSMIUM):
        product_dir = output_dir / PRODUCT_CONFIG[product]["slug"]
        product_label = f"{label} {PRODUCT_CONFIG[product]['label']}"
        summaries.append(
            analyze_product(
                product=product,
                label=product_label,
                product_dir=product_dir,
                activities=activities,
                trade_history=trade_history,
                submission_trades=submission_trades,
                replayed_all=replayed_all,
                graph=graph,
                trader_path=trader_path,
                result_json=result_json,
            )
        )

    save_table(pd.DataFrame(summaries), output_dir / "product_summary.csv")
    print(f"Analysis complete: {output_dir}")
    print(json.dumps({"run_id": run_id, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
