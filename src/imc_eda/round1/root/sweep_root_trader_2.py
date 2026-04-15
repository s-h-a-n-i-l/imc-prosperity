from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from datamodel import Observation, OrderDepth, Trade, TradingState
from imc_eda.round1 import load_prices, load_trades
from imc_eda.round1.vibe.strategy import align_trades_to_quotes, prepare_execution_context, prepare_quotes

PRODUCT = "INTARIAN_PEPPER_ROOT"
TRADER_PATH = Path(__file__).resolve().with_name("root_trader_2.py")
OUTPUT_DIR = ROOT / "output" / "root-trader-2-sweeps"


def log(message: str) -> None:
    print(message, flush=True)


def default_worker_count() -> int:
    cpu_total = os.cpu_count() or 2
    return max(1, cpu_total - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coarse-to-fine parameter sweep for root_trader_2.py")
    parser.add_argument("--days", nargs="*", type=int, default=[-2, -1, 0], help="Days to replay.")
    parser.add_argument("--fill-horizon", type=int, default=1, help="Conservative passive fill lookahead in ticks.")
    parser.add_argument("--file-format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--stage1-top-k", type=int, default=5)
    parser.add_argument("--stage2-top-k", type=int, default=3)
    parser.add_argument("--max-combos-per-stage", type=int, default=0, help="For smoke tests. 0 means no limit.")
    parser.add_argument("--inventory-penalty", type=float, default=2.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.25)
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Number of worker processes per stage. Defaults to CPU count minus one.",
    )
    return parser.parse_args()


def load_trader_class():
    spec = importlib.util.spec_from_file_location("root_trader_2_module", TRADER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trader from {TRADER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Trader


def load_trader_class_from_path(trader_path: str):
    path = Path(trader_path)
    spec = importlib.util.spec_from_file_location(f"root_trader_2_module_{os.getpid()}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trader from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Trader


def load_context(days: list[int], file_format: str, fill_horizon: int) -> pd.DataFrame:
    log(f"[data] Loading prices/trades for days={days} format={file_format} horizon={fill_horizon}")
    prices = load_prices(file_format=file_format)
    trades = load_trades(file_format=file_format)

    prices = prices[(prices["product"] == PRODUCT) & (prices["day"].isin(days))].copy()
    trades = trades[(trades["symbol"] == PRODUCT) & (trades["day"].isin(days))].copy()

    quotes = prepare_quotes(prices)
    aligned_trades = align_trades_to_quotes(quotes, trades)
    context = prepare_execution_context(quotes, aligned_trades, lookahead=fill_horizon)
    context["book_mid"] = (context["bid_price_1"] + context["ask_price_1"]) / 2
    context = context.sort_values(["day", "timestamp"]).reset_index(drop=True)

    log(
        f"[data] Prepared {len(context):,} quote rows and {len(aligned_trades):,} aligned trade rows "
        f"for {context['day'].nunique()} day(s)"
    )
    return context


def build_day_records(context: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for day, group in context.groupby("day", sort=True):
        grouped[int(day)] = group.reset_index(drop=True).to_dict("records")
    return grouped


def build_order_depth(row: dict[str, Any]) -> OrderDepth:
    depth = OrderDepth()
    for level in (1, 2, 3):
        bid_price = row.get(f"bid_price_{level}")
        bid_volume = row.get(f"bid_volume_{level}")
        ask_price = row.get(f"ask_price_{level}")
        ask_volume = row.get(f"ask_volume_{level}")
        if pd.notna(bid_price) and pd.notna(bid_volume):
            depth.buy_orders[int(bid_price)] = int(bid_volume)
        if pd.notna(ask_price) and pd.notna(ask_volume):
            depth.sell_orders[int(ask_price)] = -int(ask_volume)
    return depth


def resolve_immediate_fill(row: dict[str, Any], price: int, quantity: int) -> tuple[bool, float]:
    if quantity > 0 and price >= int(row["ask_price_1"]):
        return True, float(row["ask_price_1"])
    if quantity < 0 and price <= int(row["bid_price_1"]):
        return True, float(row["bid_price_1"])
    return False, float(price)


def find_future_fill_index(
    day_records: list[dict[str, Any]],
    current_index: int,
    side: str,
    target_price: int,
    fill_horizon: int,
) -> int | None:
    upper_bound = min(current_index + fill_horizon, len(day_records) - 1)
    for future_index in range(current_index + 1, upper_bound + 1):
        future_row = day_records[future_index]
        if side == "buy":
            ask_hit = pd.notna(future_row.get("ask_price_1")) and float(future_row["ask_price_1"]) <= target_price
            sell_trade_hit = pd.notna(future_row.get("sell_trade_price_at_quote")) and float(
                future_row["sell_trade_price_at_quote"]
            ) <= target_price
            if ask_hit or sell_trade_hit:
                return future_index
        else:
            bid_hit = pd.notna(future_row.get("bid_price_1")) and float(future_row["bid_price_1"]) >= target_price
            buy_trade_hit = pd.notna(future_row.get("buy_trade_price_at_quote")) and float(
                future_row["buy_trade_price_at_quote"]
            ) >= target_price
            if bid_hit or buy_trade_hit:
                return future_index
    return None


def apply_params(trader: Any, params: dict[str, Any]) -> None:
    for key, value in params.items():
        setattr(trader, key, value)


def score_result(summary: dict[str, Any], inventory_penalty: float, drawdown_penalty: float) -> float:
    return float(
        summary["total_pnl"]
        - inventory_penalty * summary["max_abs_position"]
        - drawdown_penalty * summary["max_drawdown"]
    )


def replay_day(
    trader_class: Any,
    day: int,
    day_records: list[dict[str, Any]],
    params: dict[str, Any],
    fill_horizon: int,
) -> dict[str, Any]:
    trader = trader_class()
    apply_params(trader, params)

    trader_data = ""
    position = 0
    cash = 0.0

    scheduled_fills: dict[int, list[dict[str, Any]]] = defaultdict(list)
    signal_counts = {-1: 0, 0: 0, 1: 0}
    submitted_buy_orders = 0
    submitted_sell_orders = 0
    realized_buy_fills = 0
    realized_sell_fills = 0
    equity_rows: list[dict[str, Any]] = []

    for index, row in enumerate(day_records):
        own_trades: list[Trade] = []
        for fill in scheduled_fills.pop(index, []):
            signed_quantity = fill["quantity"] if fill["side"] == "buy" else -fill["quantity"]
            position += signed_quantity
            cash -= signed_quantity * fill["price"]
            own_trades.append(
                Trade(
                    symbol=PRODUCT,
                    price=int(fill["price"]),
                    quantity=int(fill["quantity"]),
                    buyer="SUBMISSION" if fill["side"] == "buy" else "OTHER",
                    seller="OTHER" if fill["side"] == "buy" else "SUBMISSION",
                    timestamp=int(row["timestamp"]),
                )
            )
            if fill["side"] == "buy":
                realized_buy_fills += 1
            else:
                realized_sell_fills += 1

        state = TradingState(
            trader_data,
            int(row["timestamp"]),
            {},
            {PRODUCT: build_order_depth(row)},
            {PRODUCT: own_trades} if own_trades else {},
            {},
            {PRODUCT: position} if position else {},
            Observation({}, {}),
        )
        orders_by_product, _, trader_data = trader.run(state)
        memory = json.loads(trader_data).get("pepper", {})
        signal = int(memory.get("last_signal", 0))
        signal_counts[signal] = signal_counts.get(signal, 0) + 1

        for order in orders_by_product.get(PRODUCT, []):
            side = "buy" if order.quantity > 0 else "sell"
            quantity = abs(int(order.quantity))
            if side == "buy":
                submitted_buy_orders += 1
            else:
                submitted_sell_orders += 1

            immediate_fill, fill_price = resolve_immediate_fill(row, int(order.price), int(order.quantity))
            if immediate_fill:
                signed_quantity = int(order.quantity)
                position += signed_quantity
                cash -= signed_quantity * fill_price
                if side == "buy":
                    realized_buy_fills += 1
                else:
                    realized_sell_fills += 1
                continue

            future_fill_index = find_future_fill_index(
                day_records=day_records,
                current_index=index,
                side=side,
                target_price=int(order.price),
                fill_horizon=fill_horizon,
            )
            if future_fill_index is not None:
                scheduled_fills[future_fill_index].append(
                    {
                        "side": side,
                        "quantity": quantity,
                        "price": float(order.price),
                    }
                )

        marked_pnl = cash + position * float(row["book_mid"])
        equity_rows.append(
            {
                "day": day,
                "timestamp": int(row["timestamp"]),
                "position": int(position),
                "cash": float(cash),
                "marked_pnl": float(marked_pnl),
            }
        )

    equity_curve = pd.DataFrame(equity_rows)
    pnl_changes = equity_curve["marked_pnl"].diff().fillna(0.0)
    pnl_std = float(pnl_changes.std(ddof=0))
    running_max = equity_curve["marked_pnl"].cummax()
    drawdown = running_max - equity_curve["marked_pnl"]

    return {
        "day": int(day),
        "total_pnl": float(equity_curve["marked_pnl"].iloc[-1]),
        "simple_sharpe": float(pnl_changes.mean() / pnl_std) if pnl_std > 0 else np.nan,
        "max_drawdown": float(drawdown.max()),
        "submitted_buy_orders": int(submitted_buy_orders),
        "submitted_sell_orders": int(submitted_sell_orders),
        "realized_buy_fills": int(realized_buy_fills),
        "realized_sell_fills": int(realized_sell_fills),
        "signal_buy_ticks": int(signal_counts.get(1, 0)),
        "signal_sell_ticks": int(signal_counts.get(-1, 0)),
        "signal_neutral_ticks": int(signal_counts.get(0, 0)),
        "final_position": int(position),
        "max_abs_position": int(equity_curve["position"].abs().max()),
        "avg_abs_position": float(equity_curve["position"].abs().mean()),
        "equity_curve_rows": int(len(equity_curve)),
    }


def evaluate_params(
    trader_class: Any,
    day_records_map: dict[int, list[dict[str, Any]]],
    params: dict[str, Any],
    fill_horizon: int,
    inventory_penalty: float,
    drawdown_penalty: float,
) -> dict[str, Any]:
    day_summaries = [
        replay_day(trader_class, day, day_records_map[day], params=params, fill_horizon=fill_horizon)
        for day in sorted(day_records_map)
    ]
    frame = pd.DataFrame(day_summaries)

    summary = {
        **params,
        "days": ",".join(str(day) for day in sorted(day_records_map)),
        "total_pnl": float(frame["total_pnl"].sum()),
        "mean_daily_pnl": float(frame["total_pnl"].mean()),
        "simple_sharpe_mean": float(frame["simple_sharpe"].dropna().mean()) if frame["simple_sharpe"].notna().any() else np.nan,
        "max_drawdown": float(frame["max_drawdown"].max()),
        "submitted_buy_orders": int(frame["submitted_buy_orders"].sum()),
        "submitted_sell_orders": int(frame["submitted_sell_orders"].sum()),
        "submitted_orders": int(frame["submitted_buy_orders"].sum() + frame["submitted_sell_orders"].sum()),
        "realized_buy_fills": int(frame["realized_buy_fills"].sum()),
        "realized_sell_fills": int(frame["realized_sell_fills"].sum()),
        "realized_fills": int(frame["realized_buy_fills"].sum() + frame["realized_sell_fills"].sum()),
        "signal_buy_ticks": int(frame["signal_buy_ticks"].sum()),
        "signal_sell_ticks": int(frame["signal_sell_ticks"].sum()),
        "signal_neutral_ticks": int(frame["signal_neutral_ticks"].sum()),
        "final_position_sum": int(frame["final_position"].sum()),
        "max_abs_position": int(frame["max_abs_position"].max()),
        "avg_abs_position": float(frame["avg_abs_position"].mean()),
    }
    summary["fill_rate"] = (
        float(summary["realized_fills"] / summary["submitted_orders"]) if summary["submitted_orders"] > 0 else 0.0
    )
    summary["score"] = score_result(summary, inventory_penalty=inventory_penalty, drawdown_penalty=drawdown_penalty)
    return summary


def evaluate_combo_task(
    combo_index: int,
    stage_name: str,
    params: dict[str, Any],
    trader_path: str,
    day_records_map: dict[int, list[dict[str, Any]]],
    fill_horizon: int,
    inventory_penalty: float,
    drawdown_penalty: float,
) -> dict[str, Any]:
    combo_started_at = time.perf_counter()
    trader_class = load_trader_class_from_path(trader_path)
    result = evaluate_params(
        trader_class=trader_class,
        day_records_map=day_records_map,
        params=params,
        fill_horizon=fill_horizon,
        inventory_penalty=inventory_penalty,
        drawdown_penalty=drawdown_penalty,
    )
    result["stage"] = stage_name
    result["combo_index"] = combo_index
    result["elapsed_seconds"] = time.perf_counter() - combo_started_at
    return result


def unique_preserve_order(combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, Any], ...]] = set()
    unique: list[dict[str, Any]] = []
    for combo in combos:
        key = tuple(sorted((name, round(value, 6) if isinstance(value, float) else value) for name, value in combo.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(combo)
    return unique


def stage1_combos(defaults: dict[str, Any]) -> list[dict[str, Any]]:
    combos = []
    for base_k, min_sigma, linear_window in product(
        [0.8, 1.1, 1.4, 1.7, 2.0],
        [0.25, 0.5, 0.75, 1.0, 1.25],
        [10, 20, 30, 40, 60],
    ):
        combos.append(
            {
                **defaults,
                "BASE_K": float(base_k),
                "MIN_SIGMA": float(min_sigma),
                "LINEAR_WINDOW": int(linear_window),
            }
        )
    return combos


def stage2_combos(defaults: dict[str, Any], seeds: pd.DataFrame) -> list[dict[str, Any]]:
    combos = []
    for row in seeds.to_dict("records"):
        seed = {
            "BASE_K": float(row["BASE_K"]),
            "MIN_SIGMA": float(row["MIN_SIGMA"]),
            "LINEAR_WINDOW": int(row["LINEAR_WINDOW"]),
            "MAX_AGGRESSION": int(row["MAX_AGGRESSION"]),
            "INVENTORY_SKEW": float(row["INVENTORY_SKEW"]),
        }
        for max_aggression, inventory_skew in product([0, 1, 2, 3], [0.0, 0.1, 0.25, 0.4, 0.6]):
            combos.append(
                {
                    **defaults,
                    **seed,
                    "MAX_AGGRESSION": int(max_aggression),
                    "INVENTORY_SKEW": float(inventory_skew),
                }
            )
    return unique_preserve_order(combos)


def stage3_combos(defaults: dict[str, Any], seeds: pd.DataFrame) -> list[dict[str, Any]]:
    combos = []
    for row in seeds.to_dict("records"):
        base_k_values = sorted({max(0.4, round(float(row["BASE_K"]) + delta, 2)) for delta in (-0.2, 0.0, 0.2)})
        min_sigma_values = sorted({max(0.1, round(float(row["MIN_SIGMA"]) + delta, 2)) for delta in (-0.1, 0.0, 0.1)})
        linear_window_values = sorted({max(5, int(row["LINEAR_WINDOW"]) + delta) for delta in (-10, 0, 10)})
        max_aggression_values = sorted({min(4, max(0, int(row["MAX_AGGRESSION"]) + delta)) for delta in (-1, 0, 1)})
        inventory_skew_values = sorted({max(0.0, round(float(row["INVENTORY_SKEW"]) + delta, 2)) for delta in (-0.1, 0.0, 0.1)})

        for base_k, min_sigma, linear_window, max_aggression, inventory_skew in product(
            base_k_values,
            min_sigma_values,
            linear_window_values,
            max_aggression_values,
            inventory_skew_values,
        ):
            combos.append(
                {
                    **defaults,
                    "BASE_K": float(base_k),
                    "MIN_SIGMA": float(min_sigma),
                    "LINEAR_WINDOW": int(linear_window),
                    "MAX_AGGRESSION": int(max_aggression),
                    "INVENTORY_SKEW": float(inventory_skew),
                }
            )
    return unique_preserve_order(combos)


def run_stage(
    stage_name: str,
    combos: list[dict[str, Any]],
    trader_class: Any,
    day_records_map: dict[int, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if args.max_combos_per_stage > 0:
        combos = combos[: args.max_combos_per_stage]

    log(f"[{stage_name}] Starting {len(combos)} combination(s)")
    started_at = time.perf_counter()
    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_row: dict[str, Any] | None = None

    worker_count = max(1, min(int(args.workers), len(combos)))
    log(f"[{stage_name}] Using {worker_count} worker process(es)")

    future_to_meta: dict[concurrent.futures.Future, tuple[int, dict[str, Any]]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        for combo_index, params in enumerate(combos, start=1):
            log(f"[{stage_name}] Queueing combo {combo_index}/{len(combos)} params={params}")
            future = executor.submit(
                evaluate_combo_task,
                combo_index,
                stage_name,
                params,
                str(TRADER_PATH),
                day_records_map,
                args.fill_horizon,
                args.inventory_penalty,
                args.drawdown_penalty,
            )
            future_to_meta[future] = (combo_index, params)

        completed = 0
        for future in concurrent.futures.as_completed(future_to_meta):
            combo_index, params = future_to_meta[future]
            completed += 1
            result = future.result()
            rows.append(result)

            if result["score"] > best_score:
                best_score = result["score"]
                best_row = result

            log(
                f"[{stage_name}] Finished combo {combo_index}/{len(combos)} "
                f"({completed} completed) "
                f"score={result['score']:.2f} pnl={result['total_pnl']:.2f} "
                f"fills={result['realized_fills']}/{result['submitted_orders']} "
                f"elapsed={result['elapsed_seconds']:.2f}s "
                f"best_score={best_score:.2f}"
            )

    frame = pd.DataFrame(rows).sort_values(["score", "total_pnl"], ascending=[False, False]).reset_index(drop=True)
    stage_elapsed = time.perf_counter() - started_at
    log(f"[{stage_name}] Complete in {stage_elapsed:.2f}s")
    if best_row is not None:
        log(f"[{stage_name}] Best params={{{', '.join(f'{k}={best_row[k]}' for k in ['BASE_K','MIN_SIGMA','LINEAR_WINDOW','MAX_AGGRESSION','INVENTORY_SKEW'])}}}")
    return frame


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trader_class = load_trader_class()
    defaults = {
        "BASE_K": float(trader_class.BASE_K),
        "MIN_SIGMA": float(trader_class.MIN_SIGMA),
        "LINEAR_WINDOW": int(trader_class.LINEAR_WINDOW),
        "MAX_AGGRESSION": int(trader_class.MAX_AGGRESSION),
        "INVENTORY_SKEW": float(trader_class.INVENTORY_SKEW),
    }
    log(f"[setup] Trader defaults={defaults}")

    context = load_context(days=args.days, file_format=args.file_format, fill_horizon=args.fill_horizon)
    day_records_map = build_day_records(context)

    stage1 = run_stage("stage1_coarse", stage1_combos(defaults), trader_class, day_records_map, args)
    stage1_path = OUTPUT_DIR / "stage1_coarse_results.csv"
    stage1.to_csv(stage1_path, index=False)
    log(f"[stage1_coarse] Wrote {stage1_path}")

    stage2 = run_stage("stage2_targeted", stage2_combos(defaults, stage1.head(args.stage1_top_k)), trader_class, day_records_map, args)
    stage2_path = OUTPUT_DIR / "stage2_targeted_results.csv"
    stage2.to_csv(stage2_path, index=False)
    log(f"[stage2_targeted] Wrote {stage2_path}")

    stage3 = run_stage("stage3_refine", stage3_combos(defaults, stage2.head(args.stage2_top_k)), trader_class, day_records_map, args)
    stage3_path = OUTPUT_DIR / "stage3_refine_results.csv"
    stage3.to_csv(stage3_path, index=False)
    log(f"[stage3_refine] Wrote {stage3_path}")

    combined = pd.concat([stage1, stage2, stage3], ignore_index=True)
    combined = combined.sort_values(["score", "total_pnl"], ascending=[False, False]).reset_index(drop=True)
    combined_path = OUTPUT_DIR / "combined_results.csv"
    combined.to_csv(combined_path, index=False)
    log(f"[done] Wrote combined results to {combined_path}")

    top_columns = [
        "stage",
        "score",
        "total_pnl",
        "max_drawdown",
        "realized_fills",
        "submitted_orders",
        "fill_rate",
        "signal_buy_ticks",
        "signal_sell_ticks",
        "max_abs_position",
        "BASE_K",
        "MIN_SIGMA",
        "LINEAR_WINDOW",
        "MAX_AGGRESSION",
        "INVENTORY_SKEW",
    ]
    log("[done] Top 10 parameter sets:")
    log(combined[top_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
