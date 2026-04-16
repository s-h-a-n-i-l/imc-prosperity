from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from statistics import median
from typing import Any, Protocol

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    @dataclass
    class Order:
        symbol: str
        price: int
        quantity: int

    @dataclass
    class OrderDepth:
        buy_orders: dict[int, int] = field(default_factory=dict)
        sell_orders: dict[int, int] = field(default_factory=dict)

    @dataclass
    class TradingState:
        traderData: str = ""
        timestamp: int = 0
        order_depths: dict[str, OrderDepth] = field(default_factory=dict)
        position: dict[str, int] = field(default_factory=dict)


PRODUCT = "ASH_COATED_OSMIUM"


class SupportsRegimeMMParams(Protocol):
    strategy_name: str
    anchor_lookback: int
    base_half_spread: float
    inventory_skew: float
    imbalance_skew: float
    dislocation_threshold: float
    defensive_widening_multiplier: float
    max_quote_size: float
    inventory_soft_limit: float
    inventory_hard_limit: float
    thin_depth_threshold: float
    defensive_size_fraction: float
    strong_dislocation_buffer: float
    aggressive_size: float
    enable_dislocation_takers: bool
    dislocation_one_sided_only: bool


@dataclass(frozen=True)
class RegimeMMConfig:
    strategy_name: str = "ASH_COATED_OSMIUM_Candidate_D"
    anchor_lookback: int = 30
    base_half_spread: float = 5.0
    inventory_skew: float = 0.20
    imbalance_skew: float = 1.0
    dislocation_threshold: float = 5.0
    defensive_widening_multiplier: float = 1.5
    max_quote_size: float = 10.0
    inventory_soft_limit: float = 24.0
    inventory_hard_limit: float = 40.0
    thin_depth_threshold: float = 8.0
    defensive_size_fraction: float = 0.5
    strong_dislocation_buffer: float = 2.0
    aggressive_size: float = 0.0
    enable_dislocation_takers: bool = False
    dislocation_one_sided_only: bool = True


@dataclass(frozen=True)
class BookSnapshot:
    best_bid: int | None
    best_ask: int | None
    top_bid_depth: float
    top_ask_depth: float
    total_bid_depth: float
    total_ask_depth: float
    mid_price: float | None
    book_state: str

    @property
    def min_top_depth(self) -> float:
        return min(self.top_bid_depth, self.top_ask_depth)


@dataclass(frozen=True)
class QuotePlan:
    regime: str
    anchor_price: float | None
    reservation_price: float | None
    imbalance: float
    mid_price: float | None
    planned_bid: int | None
    planned_ask: int | None
    passive_buy_size: float
    passive_sell_size: float
    taker_side: str | None
    taker_qty: float


def _sorted_levels(book: dict[int, int] | dict[float, float], *, reverse: bool) -> list[tuple[int, float]]:
    levels = [(int(price), abs(float(volume))) for price, volume in book.items() if float(volume) != 0.0]
    return sorted(levels, key=lambda item: item[0], reverse=reverse)


def build_book_snapshot(order_depth: OrderDepth, fallback_mid: float | None = None) -> BookSnapshot:
    buy_orders = getattr(order_depth, "buy_orders", {}) or {}
    sell_orders = getattr(order_depth, "sell_orders", {}) or {}

    bid_levels = _sorted_levels(buy_orders, reverse=True)
    ask_levels = _sorted_levels(sell_orders, reverse=False)

    best_bid = bid_levels[0][0] if bid_levels else None
    best_ask = ask_levels[0][0] if ask_levels else None
    top_bid_depth = bid_levels[0][1] if bid_levels else 0.0
    top_ask_depth = ask_levels[0][1] if ask_levels else 0.0
    total_bid_depth = sum(volume for _, volume in bid_levels[:3])
    total_ask_depth = sum(volume for _, volume in ask_levels[:3])

    if best_bid is not None and best_ask is not None:
        mid_price = (best_bid + best_ask) / 2.0
        book_state = "both_sides"
    elif best_bid is not None:
        mid_price = fallback_mid
        book_state = "bid_only"
    elif best_ask is not None:
        mid_price = fallback_mid
        book_state = "ask_only"
    else:
        mid_price = fallback_mid
        book_state = "empty"

    return BookSnapshot(
        best_bid=best_bid,
        best_ask=best_ask,
        top_bid_depth=top_bid_depth,
        top_ask_depth=top_ask_depth,
        total_bid_depth=total_bid_depth,
        total_ask_depth=total_ask_depth,
        mid_price=mid_price,
        book_state=book_state,
    )


def rolling_anchor(mid_history: list[float], lookback: int) -> float | None:
    if not mid_history:
        return None
    window = mid_history[-lookback:]
    return float(median(window))


def compute_imbalance(snapshot: BookSnapshot) -> float:
    bid_depth = snapshot.total_bid_depth if snapshot.total_bid_depth > 0 else snapshot.top_bid_depth
    ask_depth = snapshot.total_ask_depth if snapshot.total_ask_depth > 0 else snapshot.top_ask_depth
    total_depth = bid_depth + ask_depth
    if total_depth <= 0:
        return 0.0
    return float((bid_depth - ask_depth) / total_depth)


def capacity_for_side(position: float, side: str, hard_limit: float) -> float:
    return max(0.0, hard_limit - position) if side == "buy" else max(0.0, position + hard_limit)


def classify_regime(snapshot: BookSnapshot, anchor_price: float | None, params: SupportsRegimeMMParams) -> str:
    if anchor_price is None:
        return "defensive"
    if snapshot.book_state != "both_sides" or snapshot.min_top_depth <= params.thin_depth_threshold:
        return "defensive"
    if snapshot.mid_price is None:
        return "defensive"
    if abs(snapshot.mid_price - anchor_price) >= params.dislocation_threshold:
        return "dislocation"
    return "normal"


def _round_quotes(reservation_price: float, bid_offset: float, ask_offset: float) -> tuple[int, int]:
    bid_price = math.floor(reservation_price - bid_offset)
    ask_price = math.ceil(reservation_price + ask_offset)
    if bid_price >= ask_price:
        bid_price = math.floor(reservation_price) - 1
        ask_price = math.ceil(reservation_price) + 1
        if bid_price >= ask_price:
            ask_price = bid_price + 1
    return bid_price, ask_price


def build_quote_plan(
    snapshot: BookSnapshot,
    position: float,
    anchor_price: float | None,
    params: SupportsRegimeMMParams,
) -> QuotePlan:
    if anchor_price is None:
        return QuotePlan(
            regime="standby",
            anchor_price=None,
            reservation_price=None,
            imbalance=0.0,
            mid_price=snapshot.mid_price,
            planned_bid=None,
            planned_ask=None,
            passive_buy_size=0.0,
            passive_sell_size=0.0,
            taker_side=None,
            taker_qty=0.0,
        )

    mid_price = snapshot.mid_price if snapshot.mid_price is not None else anchor_price
    imbalance = compute_imbalance(snapshot)
    reservation_price = anchor_price + params.imbalance_skew * imbalance - params.inventory_skew * position
    regime = classify_regime(snapshot, anchor_price, params)

    bid_offset = float(params.base_half_spread)
    ask_offset = float(params.base_half_spread)
    passive_size = float(params.max_quote_size)

    if regime == "dislocation":
        deviation = mid_price - anchor_price
        lean_ticks = min(2.0, max(1.0, math.floor(abs(deviation) / 2.0)))
        if deviation > 0:
            bid_offset += lean_ticks
            ask_offset = max(1.0, ask_offset - lean_ticks)
        elif deviation < 0:
            bid_offset = max(1.0, bid_offset - lean_ticks)
            ask_offset += lean_ticks
    elif regime == "defensive":
        bid_offset *= params.defensive_widening_multiplier
        ask_offset *= params.defensive_widening_multiplier
        passive_size *= params.defensive_size_fraction

    planned_bid, planned_ask = _round_quotes(reservation_price, bid_offset, ask_offset)

    buy_capacity = capacity_for_side(position, "buy", params.inventory_hard_limit)
    sell_capacity = capacity_for_side(position, "sell", params.inventory_hard_limit)
    allow_buy = buy_capacity > 0.0
    allow_sell = sell_capacity > 0.0

    if position >= params.inventory_soft_limit:
        allow_buy = False
    if position <= -params.inventory_soft_limit:
        allow_sell = False

    if regime == "defensive":
        if snapshot.book_state == "bid_only":
            allow_sell = False
            allow_buy = allow_buy and position < 0
        elif snapshot.book_state == "ask_only":
            allow_buy = False
            allow_sell = allow_sell and position > 0
        elif snapshot.book_state == "empty":
            allow_buy = False
            allow_sell = False
    elif regime == "dislocation" and params.dislocation_one_sided_only:
        if mid_price > anchor_price:
            allow_buy = False
        elif mid_price < anchor_price:
            allow_sell = False

    passive_buy_size = min(passive_size, buy_capacity) if allow_buy else 0.0
    passive_sell_size = min(passive_size, sell_capacity) if allow_sell else 0.0

    taker_side: str | None = None
    taker_qty = 0.0
    if params.enable_dislocation_takers and regime == "dislocation" and abs(position) < params.inventory_soft_limit:
        strong_threshold = params.dislocation_threshold + params.strong_dislocation_buffer
        if abs(mid_price - anchor_price) >= strong_threshold:
            if mid_price > anchor_price and sell_capacity > 0.0:
                taker_side = "sell"
                taker_qty = min(float(params.aggressive_size), sell_capacity)
            elif mid_price < anchor_price and buy_capacity > 0.0:
                taker_side = "buy"
                taker_qty = min(float(params.aggressive_size), buy_capacity)

    return QuotePlan(
        regime=regime,
        anchor_price=anchor_price,
        reservation_price=reservation_price,
        imbalance=imbalance,
        mid_price=snapshot.mid_price,
        planned_bid=planned_bid if passive_buy_size > 0.0 else None,
        planned_ask=planned_ask if passive_sell_size > 0.0 else None,
        passive_buy_size=passive_buy_size,
        passive_sell_size=passive_sell_size,
        taker_side=taker_side,
        taker_qty=taker_qty,
    )


def _load_trader_state(raw_state: str) -> dict[str, Any]:
    if not raw_state:
        return {"mid_history": {PRODUCT: []}}
    try:
        decoded = json.loads(raw_state)
    except json.JSONDecodeError:
        return {"mid_history": {PRODUCT: []}}
    if not isinstance(decoded, dict):
        return {"mid_history": {PRODUCT: []}}
    decoded.setdefault("mid_history", {})
    decoded["mid_history"].setdefault(PRODUCT, [])
    return decoded


def _dump_trader_state(state: dict[str, Any]) -> str:
    return json.dumps(state, separators=(",", ":"))


class Trader:
    def __init__(self, params: SupportsRegimeMMParams | None = None) -> None:
        self.params = params or RegimeMMConfig()

    def _round_quantity(self, qty: float) -> int:
        return int(math.floor(qty + 1e-9))

    def _orders_from_plan(self, snapshot: BookSnapshot, plan: QuotePlan) -> list[Order]:
        orders: list[Order] = []

        buy_qty = self._round_quantity(plan.passive_buy_size)
        if plan.planned_bid is not None and buy_qty > 0:
            orders.append(Order(PRODUCT, int(plan.planned_bid), buy_qty))

        sell_qty = self._round_quantity(plan.passive_sell_size)
        if plan.planned_ask is not None and sell_qty > 0:
            orders.append(Order(PRODUCT, int(plan.planned_ask), -sell_qty))

        taker_qty = self._round_quantity(plan.taker_qty)
        if plan.taker_side == "buy" and taker_qty > 0 and snapshot.best_ask is not None:
            orders.append(Order(PRODUCT, int(snapshot.best_ask), taker_qty))
        elif plan.taker_side == "sell" and taker_qty > 0 and snapshot.best_bid is not None:
            orders.append(Order(PRODUCT, int(snapshot.best_bid), -taker_qty))

        return orders

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        trader_state = _load_trader_state(getattr(state, "traderData", ""))
        mid_history = [float(value) for value in trader_state["mid_history"].get(PRODUCT, [])]
        fallback_mid = mid_history[-1] if mid_history else None

        order_depth = state.order_depths.get(PRODUCT)
        if order_depth is None:
            return {PRODUCT: []}, 0, _dump_trader_state(trader_state)

        snapshot = build_book_snapshot(order_depth, fallback_mid=fallback_mid)
        if snapshot.book_state == "both_sides" and snapshot.mid_price is not None:
            mid_history.append(float(snapshot.mid_price))
            mid_history = mid_history[-self.params.anchor_lookback :]
            snapshot = build_book_snapshot(order_depth, fallback_mid=mid_history[-1])

        trader_state["mid_history"][PRODUCT] = mid_history
        anchor_price = rolling_anchor(mid_history, self.params.anchor_lookback)
        position = float(state.position.get(PRODUCT, 0))
        plan = build_quote_plan(snapshot, position, anchor_price, self.params)
        orders = self._orders_from_plan(snapshot, plan)
        return {PRODUCT: orders}, 0, _dump_trader_state(trader_state)
