from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from backtest.models import BacktestResult, ExecutionProfile, MMParams
from backtest.strategy import OrderIntent, StrategyState, generate_orders, prepare_strategy_frame


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def _passive_move_through(next_row: pd.Series | None, side: str, order_price: float) -> bool:
    if next_row is None or pd.isna(order_price):
        return False
    if side == "buy":
        return (
            pd.notna(next_row["best_bid"])
            and next_row["best_bid"] < order_price
            and pd.notna(next_row["clean_mid"])
            and next_row["clean_mid"] <= order_price
        )
    return (
        pd.notna(next_row["best_ask"])
        and next_row["best_ask"] > order_price
        and pd.notna(next_row["clean_mid"])
        and next_row["clean_mid"] >= order_price
    )


def _execute_taker(active_row: pd.Series, order: OrderIntent, profile: ExecutionProfile) -> dict[str, object] | None:
    prefix = "ask" if order.side == "buy" else "bid"
    levels: list[tuple[float, float]] = []
    for level in (1, 2, 3):
        price = active_row.get(f"{prefix}_price_{level}")
        volume = active_row.get(f"{prefix}_volume_{level}")
        if pd.notna(price) and pd.notna(volume) and float(volume) > 0:
            levels.append((float(price), float(volume)))

    if not levels:
        return None

    remaining = float(order.requested_qty)
    notional = 0.0
    filled = 0.0
    for price, volume in levels:
        trade_qty = min(remaining, volume)
        notional += trade_qty * price
        remaining -= trade_qty
        filled += trade_qty
        if remaining <= 0:
            break

    if filled <= 0:
        return None

    gross_price = notional / filled
    net_price = gross_price + profile.taker_penalty_ticks if order.side == "buy" else gross_price - profile.taker_penalty_ticks
    top_price = levels[0][0]
    slippage = gross_price - top_price if order.side == "buy" else top_price - gross_price

    fill = order.as_dict()
    fill.update(
        {
            "day": int(active_row["day"]),
            "timestamp": int(active_row["timestamp"]),
            "qty": filled,
            "fill_price_gross": gross_price,
            "fill_price_net": net_price,
            "fill_price": net_price,
            "visible_depth_used": filled,
            "visible_depth_available": sum(volume for _, volume in levels),
            "slippage_ticks": slippage,
            "book_state_at_fill": active_row["book_state"],
            "best_bid_at_fill": active_row["best_bid"],
            "best_ask_at_fill": active_row["best_ask"],
            "clean_mid_at_fill": active_row["clean_mid"],
            "fill_source": "sweep_visible_book",
        }
    )
    return fill


def _execute_passive(
    active_row: pd.Series,
    next_row: pd.Series | None,
    order: OrderIntent,
    profile: ExecutionProfile,
) -> dict[str, object] | None:
    price_column = "best_bid" if order.side == "buy" else "best_ask"
    depth_column = "top_bid_depth" if order.side == "buy" else "top_ask_depth"
    trade_qty_column = "trade_qty_at_bid" if order.side == "buy" else "trade_qty_at_ask"

    order_price = active_row.get(price_column)
    top_depth = float(active_row.get(depth_column, 0.0) or 0.0)
    if pd.isna(order_price) or top_depth <= 0 or active_row["book_state"] == "empty":
        return None

    queue_ahead = profile.queue_ahead_frac * top_depth
    traded_qty = float(active_row.get(trade_qty_column, 0.0) or 0.0)
    fill_from_trade = max(0.0, traded_qty - queue_ahead)
    fill_qty = min(order.requested_qty, fill_from_trade)
    fill_source = "trade_touch" if fill_qty > 0 else "no_fill"

    remaining = order.requested_qty - fill_qty
    if profile.allow_move_through and remaining > 0 and _passive_move_through(next_row, order.side, float(order_price)):
        available_after_queue = max(0.0, top_depth - queue_ahead)
        fill_from_move = min(remaining, available_after_queue)
        if fill_from_move > 0:
            fill_qty += fill_from_move
            fill_source = "trade_and_move" if fill_source == "trade_touch" else "move_through"

    if fill_qty <= 0:
        return None

    fill = order.as_dict()
    fill.update(
        {
            "day": int(active_row["day"]),
            "timestamp": int(active_row["timestamp"]),
            "qty": float(fill_qty),
            "fill_price_gross": float(order_price),
            "fill_price_net": float(order_price),
            "fill_price": float(order_price),
            "visible_depth_used": float(fill_qty),
            "visible_depth_available": top_depth,
            "slippage_ticks": 0.0,
            "book_state_at_fill": active_row["book_state"],
            "best_bid_at_fill": active_row["best_bid"],
            "best_ask_at_fill": active_row["best_ask"],
            "clean_mid_at_fill": active_row["clean_mid"],
            "fill_source": fill_source,
        }
    )
    return fill


def _apply_fill(fill: dict[str, object], inventory: float, cash_gross: float, cash_net: float) -> tuple[float, float, float, dict[str, object]]:
    direction = 1.0 if fill["side"] == "buy" else -1.0
    qty = float(fill["qty"])
    gross_price = float(fill["fill_price_gross"])
    net_price = float(fill["fill_price_net"])
    inventory += direction * qty
    cash_gross -= direction * gross_price * qty
    cash_net -= direction * net_price * qty

    reference_price = fill["reference_price"]
    edge_ticks = direction * (float(reference_price) - net_price) if not pd.isna(reference_price) else np.nan

    fill["inventory_after"] = inventory
    fill["cash_gross_after"] = cash_gross
    fill["cash_net_after"] = cash_net
    fill["edge_ticks"] = edge_ticks
    return inventory, cash_gross, cash_net, fill


def _record_inventory(
    records: list[dict[str, object]],
    strategy: str,
    profile: ExecutionProfile,
    row: pd.Series,
    inventory: float,
    cash_gross: float,
    cash_net: float,
) -> None:
    clean_mid = float(row["clean_mid"]) if pd.notna(row["clean_mid"]) else 0.0
    records.append(
        {
            "strategy": strategy,
            "profile": profile.name,
            "day": int(row["day"]),
            "timestamp": int(row["timestamp"]),
            "inventory": inventory,
            "cash_gross": cash_gross,
            "cash_net": cash_net,
            "clean_mid": row["clean_mid"],
            "equity_gross": cash_gross + inventory * clean_mid,
            "equity_net": cash_net + inventory * clean_mid,
            "book_state": row["book_state"],
            "near_limit": abs(inventory) >= profile.near_limit_frac * profile.inventory_limit,
        }
    )


def _force_flat(
    day_frame: pd.DataFrame,
    strategy: str,
    inventory: float,
    cash_gross: float,
    cash_net: float,
    profile: ExecutionProfile,
    trade_records: list[dict[str, object]],
) -> tuple[float, float, float]:
    if not profile.force_flat_at_day_end or abs(inventory) < 1e-9:
        return inventory, cash_gross, cash_net

    side = "sell" if inventory > 0 else "buy"
    qty = abs(inventory)
    for _, row in day_frame.sort_values("timestamp", ascending=False).iterrows():
        order = OrderIntent(
            strategy=strategy,
            signal_day=int(row["day"]),
            signal_timestamp=int(row["timestamp"]),
            side=side,
            liquidity="taker",
            requested_qty=qty,
            reference_price=float(row["clean_mid"]) if pd.notna(row["clean_mid"]) else np.nan,
            reason="force_flat",
            regime="force_flat",
            anchor_price=float(row["clean_mid"]) if pd.notna(row["clean_mid"]) else np.nan,
            order_price=None,
        )
        fill = _execute_taker(row, order, profile)
        if fill is None:
            continue
        fill["forced_flat"] = True
        inventory, cash_gross, cash_net, fill = _apply_fill(fill, inventory, cash_gross, cash_net)
        trade_records.append(fill)
        qty = abs(inventory)
        if qty <= 1e-9:
            break
    return inventory, cash_gross, cash_net


def _enrich_fill_diagnostics(fills: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return fills.copy()

    diag_cols = ["day", "timestamp", "fwd_mid_move_1", "fwd_mid_move_5", "fwd_mid_move_10"]
    merged = fills.merge(feature_df[diag_cols], on=["day", "timestamp"], how="left")
    direction = np.where(merged["side"] == "buy", 1.0, -1.0)
    for horizon in (1, 5, 10):
        merged[f"signed_mid_move_{horizon}"] = direction * merged[f"fwd_mid_move_{horizon}"]
    return merged


def _summarize_daily(
    inventory_path: pd.DataFrame,
    fills: pd.DataFrame,
    order_attempts: int,
) -> pd.DataFrame:
    inventory = inventory_path.copy()
    inventory["running_peak"] = inventory["equity_net"].cummax()
    inventory["drawdown"] = inventory["running_peak"] - inventory["equity_net"]

    daily = (
        inventory.groupby("day", observed=True)
        .agg(
            gross_equity_end=("equity_gross", "last"),
            net_equity_end=("equity_net", "last"),
            mean_abs_inventory=("inventory", lambda series: float(series.abs().mean())),
            max_abs_inventory=("inventory", lambda series: float(series.abs().max())),
            near_limit_share=("near_limit", lambda series: float(series.mean())),
            max_drawdown=("drawdown", "max"),
        )
        .reset_index()
        .sort_values("day")
        .reset_index(drop=True)
    )
    daily["gross_pnl"] = daily["gross_equity_end"].diff().fillna(daily["gross_equity_end"])
    daily["net_pnl"] = daily["net_equity_end"].diff().fillna(daily["net_equity_end"])

    if fills.empty:
        daily["fill_count"] = 0
        daily["buy_fills"] = 0
        daily["sell_fills"] = 0
        daily["passive_fills"] = 0
        daily["taker_fills"] = 0
        daily["turnover"] = 0.0
        daily["forced_flat_turnover"] = 0.0
        daily["avg_edge_per_fill"] = np.nan
        daily["avg_signed_mid_move_1"] = np.nan
        daily["avg_signed_mid_move_5"] = np.nan
        daily["avg_signed_mid_move_10"] = np.nan
    else:
        grouped = fills.groupby("day", observed=True)
        daily_fills = grouped.apply(
            lambda frame: pd.Series(
                {
                    "fill_count": int(frame.shape[0]),
                    "buy_fills": int((frame["side"] == "buy").sum()),
                    "sell_fills": int((frame["side"] == "sell").sum()),
                    "passive_fills": int((frame["liquidity"] == "passive").sum()),
                    "taker_fills": int((frame["liquidity"] == "taker").sum()),
                    "turnover": float(frame["qty"].sum()),
                    "forced_flat_turnover": float(frame.loc[frame["forced_flat"], "qty"].sum()),
                    "avg_edge_per_fill": _weighted_mean(frame["edge_ticks"], frame["qty"]),
                    "avg_signed_mid_move_1": _weighted_mean(frame["signed_mid_move_1"], frame["qty"]),
                    "avg_signed_mid_move_5": _weighted_mean(frame["signed_mid_move_5"], frame["qty"]),
                    "avg_signed_mid_move_10": _weighted_mean(frame["signed_mid_move_10"], frame["qty"]),
                }
            ),
            include_groups=False,
        ).reset_index()
        daily = daily.merge(daily_fills, on="day", how="left")
        fill_defaults = {
            "fill_count": 0,
            "buy_fills": 0,
            "sell_fills": 0,
            "passive_fills": 0,
            "taker_fills": 0,
            "turnover": 0.0,
            "forced_flat_turnover": 0.0,
        }
        daily = daily.fillna(fill_defaults)

    days = max(1, len(daily))
    daily["fill_rate"] = float(daily["fill_count"].sum() / order_attempts) if order_attempts else 0.0
    daily["attempts_per_day"] = order_attempts / days
    return daily


def _summarize_result(
    strategy: str,
    profile: ExecutionProfile,
    params: MMParams,
    daily: pd.DataFrame,
    fills: pd.DataFrame,
    inventory_path: pd.DataFrame,
    order_attempts: int,
) -> pd.DataFrame:
    total_net = float(daily["net_pnl"].sum()) if not daily.empty else 0.0
    total_gross = float(daily["gross_pnl"].sum()) if not daily.empty else 0.0
    fill_count = int(fills.shape[0])
    summary = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "profile": profile.name,
                "total_pnl": total_net,
                "gross_pnl": total_gross,
                "net_pnl": total_net,
                "pnl_per_day": float(daily["net_pnl"].mean()) if not daily.empty else 0.0,
                "daily_hit_rate": float((daily["net_pnl"] > 0).mean()) if not daily.empty else 0.0,
                "fill_count": fill_count,
                "buy_fills": int((fills["side"] == "buy").sum()) if fill_count else 0,
                "sell_fills": int((fills["side"] == "sell").sum()) if fill_count else 0,
                "passive_fills": int((fills["liquidity"] == "passive").sum()) if fill_count else 0,
                "taker_fills": int((fills["liquidity"] == "taker").sum()) if fill_count else 0,
                "turnover": float(fills["qty"].sum()) if fill_count else 0.0,
                "forced_flat_turnover": float(fills.loc[fills["forced_flat"], "qty"].sum()) if fill_count else 0.0,
                "passive_share": float((fills["liquidity"] == "passive").mean()) if fill_count else 0.0,
                "order_attempts": order_attempts,
                "fill_rate": float(fill_count / order_attempts) if order_attempts else 0.0,
                "mean_abs_inventory": float(inventory_path["inventory"].abs().mean()) if not inventory_path.empty else 0.0,
                "max_abs_inventory": float(inventory_path["inventory"].abs().max()) if not inventory_path.empty else 0.0,
                "near_limit_share": float(inventory_path["near_limit"].mean()) if not inventory_path.empty else 0.0,
                "worst_day_pnl": float(daily["net_pnl"].min()) if not daily.empty else 0.0,
                "max_drawdown": float(daily["max_drawdown"].max()) if not daily.empty else 0.0,
                "avg_edge_per_fill": _weighted_mean(fills["edge_ticks"], fills["qty"]) if fill_count else np.nan,
                "avg_signed_mid_move_1": _weighted_mean(fills["signed_mid_move_1"], fills["qty"]) if fill_count else np.nan,
                "avg_signed_mid_move_5": _weighted_mean(fills["signed_mid_move_5"], fills["qty"]) if fill_count else np.nan,
                "avg_signed_mid_move_10": _weighted_mean(fills["signed_mid_move_10"], fills["qty"]) if fill_count else np.nan,
                "strictness_focus_note": "Tune against strict first; loose is upside only.",
                "anchor_lookback": params.anchor_lookback,
                "base_half_spread": params.base_half_spread,
                "inventory_skew": params.inventory_skew,
                "imbalance_skew": params.imbalance_skew,
                "dislocation_threshold": params.dislocation_threshold,
                "defensive_widening_multiplier": params.defensive_widening_multiplier,
            }
        ]
    )
    return summary


def _build_sanity_checks(
    fills: pd.DataFrame,
    inventory_path: pd.DataFrame,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    profile: ExecutionProfile,
) -> pd.DataFrame:
    net_pnl = float(summary["net_pnl"].iloc[0]) if not summary.empty else 0.0
    trade_log_pnl = 0.0
    if not fills.empty:
        trade_log_pnl = float(
            np.where(fills["side"] == "buy", -fills["fill_price_net"] * fills["qty"], fills["fill_price_net"] * fills["qty"]).sum()
        )

    checks = [
        {
            "check": "No lookahead fills",
            "passed": bool(
                fills.loc[~fills["forced_flat"], "timestamp"].gt(fills.loc[~fills["forced_flat"], "signal_timestamp"]).all()
                if not fills.empty
                else True
            ),
        },
        {
            "check": "Taker fills stay within visible 3-level depth",
            "passed": bool(
                (
                    fills.loc[fills["liquidity"] == "taker", "qty"]
                    <= fills.loc[fills["liquidity"] == "taker", "visible_depth_available"] + 1e-9
                ).all()
                if not fills.empty
                else True
            ),
        },
        {
            "check": "Passive fills never occur in empty books",
            "passed": bool(
                fills.loc[fills["liquidity"] == "passive", "book_state_at_fill"].ne("empty").all()
                if not fills.empty
                else True
            ),
        },
        {
            "check": "Inventory never breaches limit",
            "passed": bool(
                inventory_path["inventory"].abs().max() <= profile.inventory_limit + 1e-9 if not inventory_path.empty else True
            ),
        },
        {
            "check": "Each day ends flat",
            "passed": bool(
                inventory_path.groupby("day", observed=True)["inventory"].last().abs().max() <= 1e-9
                if profile.force_flat_at_day_end and not inventory_path.empty
                else True
            ),
        },
        {
            "check": "Summary PnL matches daily and trade-log totals",
            "passed": bool(
                np.isclose(net_pnl, float(daily["net_pnl"].sum()) if not daily.empty else 0.0)
                and np.isclose(net_pnl, trade_log_pnl)
            ),
        },
    ]
    return pd.DataFrame(checks)


def run_backtest(feature_df: pd.DataFrame, params: MMParams, execution_profile: ExecutionProfile) -> BacktestResult:
    if execution_profile.signal_delay_steps < 1:
        raise ValueError("signal_delay_steps must be >= 1")

    frame = prepare_strategy_frame(feature_df, params)
    trade_records: list[dict[str, object]] = []
    inventory_records: list[dict[str, object]] = []
    inventory = 0.0
    cash_gross = 0.0
    cash_net = 0.0
    order_attempts = 0

    for _, day_frame in frame.groupby("day", sort=True):
        day_frame = day_frame.sort_values("timestamp").reset_index(drop=True)
        state = StrategyState()
        max_index = len(day_frame) - execution_profile.signal_delay_steps
        for idx in range(max(0, max_index)):
            signal_row = day_frame.iloc[idx]
            active_idx = idx + execution_profile.signal_delay_steps
            active_row = day_frame.iloc[active_idx]
            next_row = day_frame.iloc[active_idx + 1] if active_idx + 1 < len(day_frame) else None

            orders, state = generate_orders(signal_row, inventory, params, execution_profile, state)
            order_attempts += len(orders)
            for order in orders:
                if order.liquidity == "taker":
                    fill = _execute_taker(active_row, order, execution_profile)
                else:
                    fill = _execute_passive(active_row, next_row, order, execution_profile)
                if fill is None:
                    continue
                fill["forced_flat"] = False
                fill["profile"] = execution_profile.name
                inventory, cash_gross, cash_net, fill = _apply_fill(fill, inventory, cash_gross, cash_net)
                trade_records.append(fill)

            _record_inventory(inventory_records, params.strategy_name, execution_profile, active_row, inventory, cash_gross, cash_net)

        inventory, cash_gross, cash_net = _force_flat(
            day_frame,
            params.strategy_name,
            inventory,
            cash_gross,
            cash_net,
            execution_profile,
            trade_records,
        )
        final_row = day_frame.iloc[-1]
        _record_inventory(inventory_records, params.strategy_name, execution_profile, final_row, inventory, cash_gross, cash_net)

    fills = pd.DataFrame(trade_records)
    if fills.empty:
        fills = pd.DataFrame(
            columns=[
                "strategy",
                "signal_day",
                "signal_timestamp",
                "side",
                "liquidity",
                "requested_qty",
                "reference_price",
                "reason",
                "regime",
                "anchor_price",
                "order_price",
                "day",
                "timestamp",
                "qty",
                "fill_price_gross",
                "fill_price_net",
                "fill_price",
                "visible_depth_used",
                "visible_depth_available",
                "slippage_ticks",
                "book_state_at_fill",
                "best_bid_at_fill",
                "best_ask_at_fill",
                "clean_mid_at_fill",
                "fill_source",
                "forced_flat",
                "profile",
                "inventory_after",
                "cash_gross_after",
                "cash_net_after",
                "edge_ticks",
            ]
        )

    inventory_path = pd.DataFrame(inventory_records)
    fills = _enrich_fill_diagnostics(fills, frame)
    daily = _summarize_daily(inventory_path, fills, order_attempts)
    summary = _summarize_result(params.strategy_name, execution_profile, params, daily, fills, inventory_path, order_attempts)
    sanity_checks = _build_sanity_checks(fills, inventory_path, summary, daily, execution_profile)
    return BacktestResult(
        profile=execution_profile,
        params=params,
        summary=summary,
        daily_pnl=daily,
        fills=fills,
        inventory_path=inventory_path,
        sanity_checks=sanity_checks,
    )


def run_sweep(
    feature_df: pd.DataFrame,
    base_params: MMParams,
    sweep_grid: dict[str, Iterable[object]],
    execution_profiles: dict[str, ExecutionProfile],
) -> pd.DataFrame:
    allowed_keys = {
        "anchor_lookback",
        "base_half_spread",
        "inventory_skew",
        "dislocation_threshold",
        "imbalance_skew",
        "defensive_widening_multiplier",
    }
    unexpected = set(sweep_grid) - allowed_keys
    if unexpected:
        raise ValueError(f"Unsupported sweep parameters: {sorted(unexpected)}")

    parameter_names = list(sweep_grid)
    parameter_values = [list(sweep_grid[name]) for name in parameter_names]
    if not parameter_names:
        raise ValueError("sweep_grid must include at least one parameter")

    rows: list[dict[str, object]] = []
    for values in product(*parameter_values):
        update_map = {name: value for name, value in zip(parameter_names, values, strict=True)}
        params = replace(
            base_params,
            anchor_lookback=int(update_map.get("anchor_lookback", base_params.anchor_lookback)),
            base_half_spread=float(update_map.get("base_half_spread", base_params.base_half_spread)),
            inventory_skew=float(update_map.get("inventory_skew", base_params.inventory_skew)),
            dislocation_threshold=float(update_map.get("dislocation_threshold", base_params.dislocation_threshold)),
            imbalance_skew=float(update_map.get("imbalance_skew", base_params.imbalance_skew)),
            defensive_widening_multiplier=float(
                update_map.get("defensive_widening_multiplier", base_params.defensive_widening_multiplier)
            ),
        )

        row: dict[str, object] = {
            "strategy": params.strategy_name,
            "anchor_lookback": params.anchor_lookback,
            "base_half_spread": params.base_half_spread,
            "inventory_skew": params.inventory_skew,
            "dislocation_threshold": params.dislocation_threshold,
            "imbalance_skew": params.imbalance_skew,
            "defensive_widening_multiplier": params.defensive_widening_multiplier,
        }
        profile_pnls: list[float] = []
        for profile_name, profile in execution_profiles.items():
            result = run_backtest(feature_df, params, profile)
            summary_row = result.summary.iloc[0]
            row[f"{profile_name}_net_pnl"] = float(summary_row["net_pnl"])
            row[f"{profile_name}_daily_hit_rate"] = float(summary_row["daily_hit_rate"])
            row[f"{profile_name}_max_abs_inventory"] = float(summary_row["max_abs_inventory"])
            row[f"{profile_name}_near_limit_share"] = float(summary_row["near_limit_share"])
            row[f"{profile_name}_adverse_5"] = float(summary_row["avg_signed_mid_move_5"])
            row[f"{profile_name}_fill_count"] = int(summary_row["fill_count"])
            profile_pnls.append(float(summary_row["net_pnl"]))

        row["worst_case_pnl"] = min(profile_pnls) if profile_pnls else np.nan
        rows.append(row)

    sweep = pd.DataFrame(rows)
    return sweep.sort_values(
        ["worst_case_pnl", "strict_max_abs_inventory", "strict_near_limit_share", "strict_net_pnl", "loose_net_pnl"],
        ascending=[False, True, True, False, False],
    ).reset_index(drop=True)
