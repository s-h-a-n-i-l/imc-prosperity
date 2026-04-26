from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from statistics import median
from typing import Any, Optional, Protocol

try:
    from datamodel import Order, OrderDepth, Trade, TradingState
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
    class Trade:
        symbol: str
        price: int
        quantity: int
        buyer: str | None = None
        seller: str | None = None
        timestamp: int = 0

    @dataclass
    class TradingState:
        traderData: str = ""
        timestamp: int = 0
        order_depths: dict[str, OrderDepth] = field(default_factory=dict)
        own_trades: dict[str, list[Trade]] = field(default_factory=dict)
        position: dict[str, int] = field(default_factory=dict)


GEL = "HYDROGEL_PACK"
FRUIT = "VELVETFRUIT_EXTRACT"

GEL_EXCHANGE_LIMIT = 200
FRUIT_EXCHANGE_LIMIT = 200
GEL_REFERENCE_MID = 10000.0
FRUIT_REFERENCE_MID = 10000.0


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
    strategy_name: str = "Candidate_G"
    anchor_lookback: int = 30
    base_half_spread: float = 5.0
    inventory_skew: float = 0.10
    imbalance_skew: float = 1.0
    dislocation_threshold: float = 5.0
    defensive_widening_multiplier: float = 1.25
    max_quote_size: float = 15.0
    inventory_soft_limit: float = 36.0
    inventory_hard_limit: float = 200.0
    thin_depth_threshold: float = 8.0
    defensive_size_fraction: float = 0.5
    strong_dislocation_buffer: float = 4.0
    aggressive_size: float = 2.0
    enable_dislocation_takers: bool = True
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

def _wall_mid(order_depth: OrderDepth) -> Optional[int]:
    if (not order_depth.buy_orders) or (not order_depth.sell_orders):
        return None

    # assume the "walls" are at the prices with maximum volume
    buy_wall = max(order_depth.buy_orders.items(), key=lambda x: x[1])[0]
    sell_wall = max(order_depth.sell_orders.items(), key=lambda x: abs(x[1]))[0]
    return (buy_wall + sell_wall) / 2

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
        mid_price = _wall_mid(order_depth)
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
    return float(median(mid_history[-lookback:]))


def compute_imbalance(snapshot: BookSnapshot) -> float:
    bid_depth = snapshot.total_bid_depth if snapshot.total_bid_depth > 0 else snapshot.top_bid_depth
    ask_depth = snapshot.total_ask_depth if snapshot.total_ask_depth > 0 else snapshot.top_ask_depth
    total_depth = bid_depth + ask_depth
    if total_depth <= 0:
        return 0.0
    return float((bid_depth - ask_depth) / total_depth)


def capacity_for_side(position: float, side: str, hard_limit: float) -> float:
    if side == "buy":
        return max(0.0, hard_limit - position)
    return max(0.0, position + hard_limit)


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
    if snapshot.best_ask is not None:
        planned_ask = snapshot.best_ask - 1
    if snapshot.best_bid is not None:
        planned_bid = snapshot.best_bid + 1

    # override the calculated prices if there are market orders we can beat
    if snapshot.best_ask is not None:
        planned_ask = snapshot.best_ask - 1 
    if snapshot.best_bid is not None:
        planned_bid = snapshot.best_bid + 1

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


class Trader:
    def __init__(
        self,
        gel_params: SupportsRegimeMMParams | None = None,
        fruit_params: SupportsRegimeMMParams | None = None,
    ) -> None:
        # Separate configuration profiles
        self.gel_params = gel_params or RegimeMMConfig(strategy_name="HYDROGEL_PACK_Candidate")
        self.fruit_params = fruit_params or RegimeMMConfig(strategy_name="VELVETFRUIT_EXTRACT_Candidate")

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        memory = self._load_memory(getattr(state, "traderData", ""))
        mids = self._current_mids(state, memory)

        result: dict[str, list[Order]] = {}

        # Trade GEL
        gel_depth = state.order_depths.get(GEL)
        if gel_depth is not None:
            gel_orders, gel_memory = self._trade_product(
                product=GEL,
                order_depth=gel_depth,
                position=float(state.position.get(GEL, 0)),
                memory=memory.get("gel", {}),
                effective_limit=GEL_EXCHANGE_LIMIT,
                base_params=self.gel_params,
            )
            result[GEL] = gel_orders
            memory["gel"] = gel_memory
            memory["portfolio"]["limits"][GEL] = GEL_EXCHANGE_LIMIT

        # Trade FRUIT
        fruit_depth = state.order_depths.get(FRUIT)
        if fruit_depth is not None:
            fruit_orders, fruit_memory = self._trade_product(
                product=FRUIT,
                order_depth=fruit_depth,
                position=float(state.position.get(FRUIT, 0)),
                memory=memory.get("fruit", {}),
                effective_limit=FRUIT_EXCHANGE_LIMIT,
                base_params=self.fruit_params,
            )
            result[FRUIT] = fruit_orders
            memory["fruit"] = fruit_memory
            memory["portfolio"]["limits"][FRUIT] = FRUIT_EXCHANGE_LIMIT

        return result, 0, json.dumps(memory, separators=(",", ":"))

    def _trade_product(
        self,
        product: str,
        order_depth: OrderDepth,
        position: float,
        memory: dict[str, Any],
        effective_limit: int,
        base_params: SupportsRegimeMMParams,
    ) -> tuple[list[Order], dict[str, Any]]:
        if effective_limit <= 0:
            return [], memory

        mid_history = [float(value) for value in memory.get("mid_history", [])]
        fallback_mid = mid_history[-1] if mid_history else None
        snapshot = build_book_snapshot(order_depth, fallback_mid=fallback_mid)

        if snapshot.book_state == "both_sides" and snapshot.mid_price is not None:
            mid_history.append(float(snapshot.mid_price))
            mid_history = mid_history[-base_params.anchor_lookback :]
            snapshot = build_book_snapshot(order_depth, fallback_mid=mid_history[-1])

        memory["mid_history"] = mid_history
        anchor_price = rolling_anchor(mid_history, base_params.anchor_lookback)

        scale = max(min(effective_limit / float(base_params.inventory_hard_limit), 1.0), 0.0)
        params = RegimeMMConfig(
            strategy_name=base_params.strategy_name,
            anchor_lookback=base_params.anchor_lookback,
            base_half_spread=base_params.base_half_spread,
            inventory_skew=base_params.inventory_skew,
            imbalance_skew=base_params.imbalance_skew,
            dislocation_threshold=base_params.dislocation_threshold,
            defensive_widening_multiplier=base_params.defensive_widening_multiplier,
            max_quote_size=max(1.0, base_params.max_quote_size * max(scale, 0.25)),
            inventory_soft_limit=max(1.0, math.floor(effective_limit * 0.6)),
            inventory_hard_limit=float(effective_limit),
            thin_depth_threshold=base_params.thin_depth_threshold,
            defensive_size_fraction=base_params.defensive_size_fraction,
            strong_dislocation_buffer=base_params.strong_dislocation_buffer,
            aggressive_size=max(1.0, base_params.aggressive_size * max(scale, 0.5)),
            enable_dislocation_takers=base_params.enable_dislocation_takers,
            dislocation_one_sided_only=base_params.dislocation_one_sided_only,
        )

        plan = build_quote_plan(snapshot, position, anchor_price, params)
        orders = self._orders_from_plan(product, snapshot, plan)

        memory["effective_limit"] = effective_limit
        memory["last_fair_value"] = anchor_price
        memory["last_anchor_price"] = anchor_price
        memory["last_mid_price"] = snapshot.mid_price
        memory["last_best_bid"] = snapshot.best_bid
        memory["last_best_ask"] = snapshot.best_ask
        memory["last_top_bid_depth"] = snapshot.top_bid_depth
        memory["last_top_ask_depth"] = snapshot.top_ask_depth
        memory["last_total_bid_depth"] = snapshot.total_bid_depth
        memory["last_total_ask_depth"] = snapshot.total_ask_depth
        memory["last_book_state"] = snapshot.book_state
        memory["last_regime"] = plan.regime
        memory["last_reservation_price"] = plan.reservation_price
        memory["last_imbalance"] = plan.imbalance
        memory["last_planned_bid"] = plan.planned_bid
        memory["last_planned_ask"] = plan.planned_ask
        memory["last_passive_buy_size"] = plan.passive_buy_size
        memory["last_passive_sell_size"] = plan.passive_sell_size
        memory["last_taker_side"] = plan.taker_side
        memory["last_taker_qty"] = plan.taker_qty
        return orders, memory

    def _orders_from_plan(self, product: str, snapshot: BookSnapshot, plan: QuotePlan) -> list[Order]:
        orders: list[Order] = []

        buy_qty = self._round_quantity(plan.passive_buy_size)
        if plan.planned_bid is not None and buy_qty > 0:
            orders.append(Order(product, int(plan.planned_bid), buy_qty))

        sell_qty = self._round_quantity(plan.passive_sell_size)
        if plan.planned_ask is not None and sell_qty > 0:
            orders.append(Order(product, int(plan.planned_ask), -sell_qty))

        taker_qty = self._round_quantity(plan.taker_qty)
        if plan.taker_side == "buy" and taker_qty > 0 and snapshot.best_ask is not None:
            orders.append(Order(product, int(snapshot.best_ask), taker_qty))
        elif plan.taker_side == "sell" and taker_qty > 0 and snapshot.best_bid is not None:
            orders.append(Order(product, int(snapshot.best_bid), -taker_qty))

        return orders

    def _round_quantity(self, qty: float) -> int:
        return int(math.floor(qty + 1e-9))

    def _load_memory(self, trader_data: str) -> dict[str, Any]:
        if not trader_data:
            return self._default_memory()

        try:
            data = json.loads(trader_data)
        except json.JSONDecodeError:
            data = {}

        default = self._default_memory()
        data.setdefault("gel", default["gel"])
        data.setdefault("fruit", default["fruit"])
        data.setdefault("portfolio", default["portfolio"])

        for product_key in ("gel", "fruit"):
            product_memory = data[product_key]
            product_memory.setdefault("mid_history", [])
            product_memory.setdefault("effective_limit", GEL_EXCHANGE_LIMIT) 
            product_memory.setdefault("last_fair_value", None)
            product_memory.setdefault("last_anchor_price", None)
            product_memory.setdefault("last_mid_price", None)
            product_memory.setdefault("last_best_bid", None)
            product_memory.setdefault("last_best_ask", None)
            product_memory.setdefault("last_top_bid_depth", 0.0)
            product_memory.setdefault("last_top_ask_depth", 0.0)
            product_memory.setdefault("last_total_bid_depth", 0.0)
            product_memory.setdefault("last_total_ask_depth", 0.0)
            product_memory.setdefault("last_book_state", "empty")
            product_memory.setdefault("last_regime", "standby")
            product_memory.setdefault("last_reservation_price", None)
            product_memory.setdefault("last_imbalance", 0.0)
            product_memory.setdefault("last_planned_bid", None)
            product_memory.setdefault("last_planned_ask", None)
            product_memory.setdefault("last_passive_buy_size", 0.0)
            product_memory.setdefault("last_passive_sell_size", 0.0)
            product_memory.setdefault("last_taker_side", None)
            product_memory.setdefault("last_taker_qty", 0.0)

        portfolio = data["portfolio"]
        portfolio.setdefault("reference_mids", {GEL: GEL_REFERENCE_MID, FRUIT: FRUIT_REFERENCE_MID})
        portfolio.setdefault("limits", {GEL: GEL_EXCHANGE_LIMIT, FRUIT: FRUIT_EXCHANGE_LIMIT})
        portfolio.setdefault("last_total_pnl", 0.0)
        return data

    def _default_memory(self) -> dict[str, Any]:
        base_product_memory = {
            "mid_history": [],
            "effective_limit": GEL_EXCHANGE_LIMIT,
            "last_fair_value": None,
            "last_anchor_price": None,
            "last_mid_price": None,
            "last_best_bid": None,
            "last_best_ask": None,
            "last_top_bid_depth": 0.0,
            "last_top_ask_depth": 0.0,
            "last_total_bid_depth": 0.0,
            "last_total_ask_depth": 0.0,
            "last_book_state": "empty",
            "last_regime": "standby",
            "last_reservation_price": None,
            "last_imbalance": 0.0,
            "last_planned_bid": None,
            "last_planned_ask": None,
            "last_passive_buy_size": 0.0,
            "last_passive_sell_size": 0.0,
            "last_taker_side": None,
            "last_taker_qty": 0.0,
        }

        return {
            "gel": dict(base_product_memory),
            "fruit": dict(base_product_memory),
            "portfolio": {
                "reference_mids": {GEL: GEL_REFERENCE_MID, FRUIT: FRUIT_REFERENCE_MID},
                "limits": {GEL: GEL_EXCHANGE_LIMIT, FRUIT: FRUIT_EXCHANGE_LIMIT},
                "last_total_pnl": 0.0,
            },
        }

    def _current_mids(self, state: TradingState, memory: dict[str, Any]) -> dict[str, float]:
        reference_mids = memory["portfolio"]["reference_mids"]
        mids = {GEL: float(reference_mids.get(GEL, GEL_REFERENCE_MID)),
                FRUIT: float(reference_mids.get(FRUIT, FRUIT_REFERENCE_MID))}

        for product in (GEL, FRUIT):
            order_depth = state.order_depths.get(product)
            if order_depth is None:
                continue
            mid = _wall_mid(order_depth)
            if mid is not None:
                mids[product] = mid
                reference_mids[product] = mids[product]
        return mids