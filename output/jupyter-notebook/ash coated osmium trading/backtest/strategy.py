from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.data import rolling_median_with_fallback
from backtest.models import ExecutionProfile, MMParams
from trader import BookSnapshot, build_quote_plan


@dataclass
class StrategyState:
    last_regime: str = "standby"


@dataclass
class OrderIntent:
    strategy: str
    signal_day: int
    signal_timestamp: int
    side: str
    liquidity: str
    requested_qty: float
    reference_price: float
    reason: str
    regime: str
    anchor_price: float
    order_price: float | None = None

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "strategy": self.strategy,
            "signal_day": self.signal_day,
            "signal_timestamp": self.signal_timestamp,
            "side": self.side,
            "liquidity": self.liquidity,
            "requested_qty": self.requested_qty,
            "reference_price": self.reference_price,
            "reason": self.reason,
            "regime": self.regime,
            "anchor_price": self.anchor_price,
            "order_price": self.order_price,
        }


def prepare_strategy_frame(feature_df: pd.DataFrame, params: MMParams) -> pd.DataFrame:
    frame = feature_df.copy().sort_values(["day", "timestamp"]).reset_index(drop=True)
    frame["anchor_price"] = frame.groupby("day")["clean_mid"].transform(
        lambda series: rolling_median_with_fallback(series, params.anchor_lookback)
    )
    return frame


def _snapshot_from_row(row: pd.Series) -> BookSnapshot:
    mid_price = float(row["clean_mid"]) if pd.notna(row["clean_mid"]) else None
    return BookSnapshot(
        best_bid=int(row["best_bid"]) if pd.notna(row["best_bid"]) else None,
        best_ask=int(row["best_ask"]) if pd.notna(row["best_ask"]) else None,
        top_bid_depth=float(row.get("top_bid_depth", 0.0) or 0.0),
        top_ask_depth=float(row.get("top_ask_depth", 0.0) or 0.0),
        total_bid_depth=float(row.get("total_bid_depth_3lvl", 0.0) or 0.0),
        total_ask_depth=float(row.get("total_ask_depth_3lvl", 0.0) or 0.0),
        mid_price=mid_price,
        book_state=str(row["book_state"]),
    )


def _make_order(
    row: pd.Series,
    params: MMParams,
    side: str,
    liquidity: str,
    qty: float,
    reference_price: float,
    reason: str,
    regime: str,
    anchor_price: float,
    order_price: float | None,
) -> OrderIntent:
    return OrderIntent(
        strategy=params.strategy_name,
        signal_day=int(row["day"]),
        signal_timestamp=int(row["timestamp"]),
        side=side,
        liquidity=liquidity,
        requested_qty=float(qty),
        reference_price=float(reference_price),
        reason=reason,
        regime=regime,
        anchor_price=float(anchor_price),
        order_price=None if order_price is None else float(order_price),
    )


def generate_orders(
    row: pd.Series,
    inventory: float,
    params: MMParams,
    profile: ExecutionProfile,
    state: StrategyState,
) -> tuple[list[OrderIntent], StrategyState]:
    anchor_price = row.get("anchor_price")
    if pd.isna(anchor_price):
        return [], state

    snapshot = _snapshot_from_row(row)
    plan = build_quote_plan(snapshot, inventory, float(anchor_price), params)
    state.last_regime = plan.regime

    orders: list[OrderIntent] = []
    passive_cap = float(profile.passive_order_size)
    taker_cap = float(profile.taker_order_size)

    if (
        plan.planned_bid is not None
        and pd.notna(row["best_bid"])
        and plan.planned_bid >= float(row["best_bid"])
        and plan.passive_buy_size > 0.0
    ):
        qty = min(plan.passive_buy_size, passive_cap)
        if qty > 0.0:
            orders.append(
                _make_order(
                    row=row,
                    params=params,
                    side="buy",
                    liquidity="passive",
                    qty=qty,
                    reference_price=float(plan.reservation_price),
                    reason=f"{plan.regime}_passive_buy",
                    regime=plan.regime,
                    anchor_price=float(plan.anchor_price),
                    order_price=float(plan.planned_bid),
                )
            )

    if (
        plan.planned_ask is not None
        and pd.notna(row["best_ask"])
        and plan.planned_ask <= float(row["best_ask"])
        and plan.passive_sell_size > 0.0
    ):
        qty = min(plan.passive_sell_size, passive_cap)
        if qty > 0.0:
            orders.append(
                _make_order(
                    row=row,
                    params=params,
                    side="sell",
                    liquidity="passive",
                    qty=qty,
                    reference_price=float(plan.reservation_price),
                    reason=f"{plan.regime}_passive_sell",
                    regime=plan.regime,
                    anchor_price=float(plan.anchor_price),
                    order_price=float(plan.planned_ask),
                )
            )

    if plan.taker_side is not None and plan.taker_qty > 0.0:
        qty = min(plan.taker_qty, taker_cap)
        if qty > 0.0:
            orders.append(
                _make_order(
                    row=row,
                    params=params,
                    side=plan.taker_side,
                    liquidity="taker",
                    qty=qty,
                    reference_price=float(plan.reservation_price),
                    reason=f"{plan.regime}_taker_{plan.taker_side}",
                    regime=plan.regime,
                    anchor_price=float(plan.anchor_price),
                    order_price=None,
                )
            )

    return orders, state
