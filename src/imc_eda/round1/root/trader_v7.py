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


PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

# Starting cash allocation across products.
PEPPER_STARTING_CASH_PROPORTION = 0.80
OSMIUM_STARTING_CASH_PROPORTION = 0.20

PEPPER_EXCHANGE_LIMIT = 80
OSMIUM_EXCHANGE_LIMIT = 80
PEPPER_REFERENCE_MID = 10400.0
OSMIUM_REFERENCE_MID = 10000.0


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
    strategy_name: str = "ASH_COATED_OSMIUM_Candidate_G"
    anchor_lookback: int = 30
    base_half_spread: float = 5.0
    inventory_skew: float = 0.20
    imbalance_skew: float = 1.0
    dislocation_threshold: float = 5.0
    defensive_widening_multiplier: float = 1.25
    max_quote_size: float = 15.0
    inventory_soft_limit: float = 36.0
    inventory_hard_limit: float = float(OSMIUM_EXCHANGE_LIMIT)
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
    sell_wall = max(order_depth.sell_orders.items(), key=lambda x: x[1])[0]
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
    LINEAR_WINDOW = 40
    RESIDUAL_WINDOW = 25
    MIN_SIGMA = 0.5

    # --- NEW TUNABLE PARAMETERS FOR PEPPER ---
    # The fraction of your max limit to hold passively for trend growth (e.g., 0.75 * 80 = 60)
    PEPPER_CORE_FRACTION = 0.90
    # The minimum edge from fair value to place a passive quote
    PEPPER_MIN_MARGIN = 1
    # The distance above fair value to trigger dumping your ENTIRE position (spike catching)
    PEPPER_SPIKE_MARGIN = 5.0
    # -----------------------------------------

    def __init__(self, params: SupportsRegimeMMParams | None = None) -> None:
        self.osmium_params = params or RegimeMMConfig()

        # ... [Keep run, _trade_osmium, _orders_from_plan, etc. exactly the same] ...
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        memory = self._load_memory(getattr(state, "traderData", ""))
        self._update_cash_ledger(memory, state)

        mids = self._current_mids(state, memory)
        self._initialize_starting_cash(memory, mids)

        result: dict[str, list[Order]] = {}

        pepper_depth = state.order_depths.get(PEPPER)
        if pepper_depth is not None:
            pepper_orders, pepper_memory = self._trade_pepper(
                order_depth=pepper_depth,
                position=int(state.position.get(PEPPER, 0)),
                own_trades=list(getattr(state, "own_trades", {}).get(PEPPER, [])),
                memory=memory.get("pepper", {}),
                timestamp=int(state.timestamp),
                max_position=PEPPER_EXCHANGE_LIMIT,
            )
            result[PEPPER] = pepper_orders
            memory["pepper"] = pepper_memory
            memory["portfolio"]["limits"][PEPPER] = PEPPER_EXCHANGE_LIMIT

        osmium_depth = state.order_depths.get(OSMIUM)
        if osmium_depth is not None:
            osmium_orders, osmium_memory = self._trade_osmium(
                order_depth=osmium_depth,
                position=float(state.position.get(OSMIUM, 0)),
                memory=memory.get("osmium", {}),
                effective_limit=OSMIUM_EXCHANGE_LIMIT,
            )
            result[OSMIUM] = osmium_orders
            memory["osmium"] = osmium_memory
            memory["portfolio"]["limits"][OSMIUM] = OSMIUM_EXCHANGE_LIMIT

        return result, 0, json.dumps(memory, separators=(",", ":"))

    def _trade_pepper(
        self,
        order_depth: OrderDepth,
        position: int,
        own_trades: list[Trade],
        memory: dict[str, Any],
        timestamp: int,
        max_position: int,
    ) -> tuple[list[Order], dict[str, Any]]:
        best_bid, best_ask = self._top_of_book(order_depth)
        if best_bid is None or best_ask is None or max_position <= 0:
            return [], memory # TODO: we can still do some things if there aren't bids or asks... like trying to market make (but not worth it really)

        current_price = _wall_mid(order_depth)
        fair_value, slope, intercept = self._update_linear_fair_value(memory, current_price)

        residual = current_price - fair_value
        residual_history = list(memory.get("residual_history", []))
        residual_history.append(residual)
        residual_history = residual_history[-self.RESIDUAL_WINDOW :]
        memory["residual_history"] = residual_history

        sigma = self._rolling_std(residual_history)

        # --- PHASE 1 & 2: Prediction and Capacities ---
        # Project fair value one tick into the future
        predicted_fair = fair_value + slope
        z_score = residual / sigma if sigma > 0 else 0

        # Determine how much we want to hold permanently vs market make
        core_target = int(max_position * self.PEPPER_CORE_FRACTION)

        buy_capacity = self._buy_capacity(position, max_position)
        # Only allow passive selling if we exceed our "Core" trend-following position
        sell_capacity = max(0, position - core_target)

        orders: list[Order] = []
        action = "hold"

        # --- PHASE 3: Pennying Logic (Passive Bidding) ---
        # Step inside the spread if it's wide enough, otherwise join the best quote
        my_bid = best_bid + 1 if best_ask - best_bid > 1 else best_bid
        my_ask = best_ask - 1 if best_ask - best_bid > 1 else best_ask

        # Build inventory passively: only bid if it's NOT on the "wrong" side of fair value
        if buy_capacity > 0 and my_bid <= (predicted_fair - self.PEPPER_MIN_MARGIN):
            orders.append(Order(PEPPER, int(my_bid), buy_capacity))
            action = "quote_both" if sell_capacity > 0 else "quote_bid"
            memory["pending_buy_order"] = {
                "side": "buy",
                "timestamp": timestamp,
                "price": my_bid,
                "position_before": position,
            }

        # Cycle the top inventory passively: only ask if it's NOT on the "wrong" side
        if sell_capacity > 0 and my_ask >= (predicted_fair + self.PEPPER_MIN_MARGIN):
            orders.append(Order(PEPPER, int(my_ask), -sell_capacity))
            action = "quote_both" if buy_capacity > 0 else "quote_ask"
            memory["pending_sell_order"] = {
                "side": "sell",
                "timestamp": timestamp,
                "price": my_ask,
                "position_before": position,
            }

        # --- PHASE 4: Aggressive Sniping (Override Passive) ---
        # If the market asks are actively cheaper than our predicted fair value, or we aren't holding enough peppers, hit them!
        if (best_ask < predicted_fair and buy_capacity > 0) or (buy_capacity > 10):
            # Remove any passive buys, replace with aggressive take
            orders = [o for o in orders if o.quantity < 0]
            orders.append(Order(PEPPER, int(best_ask), buy_capacity))
            action = "take_ask"

        # If the market bids spike abnormally high, dump our ENTIRE position to lock in profit
        elif best_bid > (predicted_fair + self.PEPPER_SPIKE_MARGIN) and position > 0:
            # Remove any passive sells, replace with aggressive full dump
            orders = [o for o in orders if o.quantity > 0]
            orders.append(Order(PEPPER, int(best_bid), -position))
            action = "take_bid_dump"

        # (Leave your memory tracking intact so aggressiveness counters don't crash)
        buy_aggression, buy_miss_count = self._update_aggressiveness(
            memory, position, own_trades, timestamp, "buy"
        )
        sell_aggression, sell_miss_count = self._update_aggressiveness(
            memory, position, own_trades, timestamp, "sell"
        )

        memory["max_position"] = max_position
        memory["last_position"] = position
        memory["last_timestamp"] = timestamp
        memory["last_signal"] = action
        memory["last_action"] = action
        memory["last_sigma"] = sigma
        memory["last_residual"] = residual
        memory["last_adjusted_residual"] = residual
        memory["buy_aggression_level"] = buy_aggression
        memory["sell_aggression_level"] = sell_aggression
        memory["buy_miss_count"] = buy_miss_count
        memory["sell_miss_count"] = sell_miss_count
        memory["last_fair_value"] = fair_value
        memory["last_slope"] = slope
        memory["last_intercept"] = intercept

        return orders, memory


    def _trade_osmium(
        self,
        order_depth: OrderDepth,
        position: float,
        memory: dict[str, Any],
        effective_limit: int,
    ) -> tuple[list[Order], dict[str, Any]]:
        if effective_limit <= 0:
            return [], memory

        mid_history = [float(value) for value in memory.get("mid_history", [])]
        fallback_mid = mid_history[-1] if mid_history else None
        snapshot = build_book_snapshot(order_depth, fallback_mid=fallback_mid)

        if snapshot.book_state == "both_sides" and snapshot.mid_price is not None:
            mid_history.append(float(snapshot.mid_price))
            mid_history = mid_history[-self.osmium_params.anchor_lookback :]
            snapshot = build_book_snapshot(order_depth, fallback_mid=mid_history[-1])

        memory["mid_history"] = mid_history
        anchor_price = rolling_anchor(mid_history, self.osmium_params.anchor_lookback)

        scale = max(min(effective_limit / float(OSMIUM_EXCHANGE_LIMIT), 1.0), 0.0)
        params = RegimeMMConfig(
            strategy_name=self.osmium_params.strategy_name,
            anchor_lookback=self.osmium_params.anchor_lookback,
            base_half_spread=self.osmium_params.base_half_spread,
            inventory_skew=self.osmium_params.inventory_skew,
            imbalance_skew=self.osmium_params.imbalance_skew,
            dislocation_threshold=self.osmium_params.dislocation_threshold,
            defensive_widening_multiplier=self.osmium_params.defensive_widening_multiplier,
            max_quote_size=max(1.0, self.osmium_params.max_quote_size * max(scale, 0.25)),
            inventory_soft_limit=max(1.0, math.floor(effective_limit * 0.6)),
            inventory_hard_limit=float(effective_limit),
            thin_depth_threshold=self.osmium_params.thin_depth_threshold,
            defensive_size_fraction=self.osmium_params.defensive_size_fraction,
            strong_dislocation_buffer=self.osmium_params.strong_dislocation_buffer,
            aggressive_size=max(1.0, self.osmium_params.aggressive_size * max(scale, 0.5)),
            enable_dislocation_takers=self.osmium_params.enable_dislocation_takers,
            dislocation_one_sided_only=self.osmium_params.dislocation_one_sided_only,
        )

        plan = build_quote_plan(snapshot, position, anchor_price, params)
        orders = self._orders_from_plan(snapshot, plan)

        memory["effective_limit"] = effective_limit
        memory["last_anchor_price"] = anchor_price
        memory["last_mid_price"] = snapshot.mid_price
        memory["last_regime"] = plan.regime
        return orders, memory

    def _orders_from_plan(self, snapshot: BookSnapshot, plan: QuotePlan) -> list[Order]:
        orders: list[Order] = []

        buy_qty = self._round_quantity(plan.passive_buy_size)
        if plan.planned_bid is not None and buy_qty > 0:
            orders.append(Order(OSMIUM, int(plan.planned_bid), buy_qty))

        sell_qty = self._round_quantity(plan.passive_sell_size)
        if plan.planned_ask is not None and sell_qty > 0:
            orders.append(Order(OSMIUM, int(plan.planned_ask), -sell_qty))

        taker_qty = self._round_quantity(plan.taker_qty)
        if plan.taker_side == "buy" and taker_qty > 0 and snapshot.best_ask is not None:
            orders.append(Order(OSMIUM, int(snapshot.best_ask), taker_qty))
        elif plan.taker_side == "sell" and taker_qty > 0 and snapshot.best_bid is not None:
            orders.append(Order(OSMIUM, int(snapshot.best_bid), -taker_qty))

        return orders

    def _round_quantity(self, qty: float) -> int:
        return int(math.floor(qty + 1e-9))

    def _update_linear_fair_value(self, memory: dict[str, Any], price: float) -> tuple[float, float, float]:
        price_history = list(memory.get("price_history", []))
        price_history.append(float(price))
        price_history = price_history[-self.LINEAR_WINDOW :]
        memory["price_history"] = price_history

        sample_count = len(price_history)
        if sample_count <= 10: # when we don't have many samples, linear prediction is unreliable
            fair_value = float(price_history[-1])
            slope = 0.0
            intercept = fair_value
        else:
            x_values = [float(index) for index in range(sample_count)]
            x_mean = sum(x_values) / sample_count
            y_mean = sum(price_history) / sample_count
            denominator = sum((x_value - x_mean) ** 2 for x_value in x_values)
            if denominator <= 0:
                slope = 0.0
            else:
                numerator = sum(
                    (x_value - x_mean) * (y_value - y_mean)
                    for x_value, y_value in zip(x_values, price_history)
                )
                slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            fair_value = slope * x_values[-1] + intercept

        memory["fair_value"] = fair_value
        memory["trend_slope"] = slope
        memory["trend_intercept"] = intercept
        return fair_value, slope, intercept

    def _update_aggressiveness(
        self,
        memory: dict[str, Any],
        position: int,
        own_trades: list[Trade],
        timestamp: int,
        side: str,
    ) -> tuple[int, int]:
        pending_key = f"pending_{side}_order"
        miss_key = f"{side}_miss_count"
        aggression_key = f"{side}_aggression_level"

        pending_order = memory.get(pending_key, {})
        miss_count = int(memory.get(miss_key, 0))
        if not pending_order:
            return min(int(memory.get(aggression_key, 0)), 2), miss_count

        filled = self._pending_order_filled(
            pending_order=pending_order,
            position=position,
            own_trades=own_trades,
            timestamp=timestamp,
            side=side,
        )
        if filled:
            memory[pending_key] = {}
            reduced_miss_count = max(miss_count - 1, 0)
            return min(reduced_miss_count, 2), reduced_miss_count

        if timestamp > int(pending_order.get("timestamp", timestamp)):
            miss_count += 1
            memory[pending_key] = {}

        aggression_level = min(miss_count, 2)
        return aggression_level, miss_count

    def _pending_order_filled(
        self,
        pending_order: dict[str, Any],
        position: int,
        own_trades: list[Trade],
        timestamp: int,
        side: str,
    ) -> bool:
        previous_position = int(pending_order.get("position_before", position))
        if side == "buy" and position > previous_position:
            return True
        if side == "sell" and position < previous_position:
            return True

        for trade in own_trades:
            if int(getattr(trade, "timestamp", 0)) != timestamp:
                continue
            if side == "buy" and getattr(trade, "buyer", None) == "SUBMISSION":
                return True
            if side == "sell" and getattr(trade, "seller", None) == "SUBMISSION":
                return True
        return False

    def _rolling_std(self, values: list[float]) -> float:
        if len(values) < 2:
            return self.MIN_SIGMA
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
        return max(variance**0.5, self.MIN_SIGMA)

    def _top_of_book(self, order_depth: OrderDepth) -> tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        return best_bid, best_ask

    def _buy_capacity(self, position: int, max_position: int) -> int:
        return max(max_position - position, 0)

    def _sell_capacity(self, position: int, max_position: int) -> int:
        return max(min(max_position, position), 0)

    def _load_memory(self, trader_data: str) -> dict[str, Any]:
        if not trader_data:
            return self._default_memory()

        try:
            data = json.loads(trader_data)
        except json.JSONDecodeError:
            data = {}

        default = self._default_memory()
        data.setdefault("pepper", default["pepper"])
        data.setdefault("osmium", default["osmium"])
        data.setdefault("portfolio", default["portfolio"])

        pepper_memory = data["pepper"]
        pepper_memory.setdefault("fair_value", None)
        pepper_memory.setdefault("price_history", [])
        pepper_memory.setdefault("residual_history", [])
        pepper_memory.setdefault("pending_buy_order", {})
        pepper_memory.setdefault("pending_sell_order", {})
        pepper_memory.setdefault("buy_miss_count", 0)
        pepper_memory.setdefault("sell_miss_count", 0)
        pepper_memory.setdefault("buy_aggression_level", 0)
        pepper_memory.setdefault("sell_aggression_level", 0)
        pepper_memory.setdefault("last_position", 0)
        pepper_memory.setdefault("last_timestamp", 0)
        pepper_memory.setdefault("trend_slope", 0.0)
        pepper_memory.setdefault("trend_intercept", None)
        pepper_memory.setdefault("max_position", PEPPER_EXCHANGE_LIMIT)

        osmium_memory = data["osmium"]
        osmium_memory.setdefault("mid_history", [])
        osmium_memory.setdefault("effective_limit", OSMIUM_EXCHANGE_LIMIT)

        portfolio = data["portfolio"]
        portfolio.setdefault("starting_cash", None)
        portfolio.setdefault("cash", {PEPPER: 0.0, OSMIUM: 0.0})
        portfolio.setdefault("last_trade_timestamp", {PEPPER: -1, OSMIUM: -1})
        portfolio.setdefault("reference_mids", {PEPPER: PEPPER_REFERENCE_MID, OSMIUM: OSMIUM_REFERENCE_MID})
        portfolio.setdefault("limits", {PEPPER: PEPPER_EXCHANGE_LIMIT, OSMIUM: OSMIUM_EXCHANGE_LIMIT})
        portfolio.setdefault("last_total_pnl", 0.0)
        return data

    def _default_memory(self) -> dict[str, Any]:
        return {
            "pepper": {
                "fair_value": None,
                "price_history": [],
                "residual_history": [],
                "pending_buy_order": {},
                "pending_sell_order": {},
                "buy_miss_count": 0,
                "sell_miss_count": 0,
                "buy_aggression_level": 0,
                "sell_aggression_level": 0,
                "last_position": 0,
                "last_timestamp": 0,
                "trend_slope": 0.0,
                "trend_intercept": None,
                "max_position": PEPPER_EXCHANGE_LIMIT,
            },
            "osmium": {
                "mid_history": [],
                "effective_limit": OSMIUM_EXCHANGE_LIMIT,
            },
            "portfolio": {
                "starting_cash": None,
                "cash": {PEPPER: 0.0, OSMIUM: 0.0},
                "last_trade_timestamp": {PEPPER: -1, OSMIUM: -1},
                "reference_mids": {PEPPER: PEPPER_REFERENCE_MID, OSMIUM: OSMIUM_REFERENCE_MID},
                "limits": {PEPPER: PEPPER_EXCHANGE_LIMIT, OSMIUM: OSMIUM_EXCHANGE_LIMIT},
                "last_total_pnl": 0.0,
            },
        }

    def _update_cash_ledger(self, memory: dict[str, Any], state: TradingState) -> None:
        portfolio = memory["portfolio"]
        own_trades = getattr(state, "own_trades", {}) or {}

        for product in (PEPPER, OSMIUM):
            trades = list(own_trades.get(product, []))
            last_timestamp = int(portfolio["last_trade_timestamp"].get(product, -1))
            for trade in trades:
                trade_timestamp = int(getattr(trade, "timestamp", -1))
                if trade_timestamp <= last_timestamp:
                    continue

                quantity = int(abs(getattr(trade, "quantity", 0)))
                price = float(getattr(trade, "price", 0))
                if getattr(trade, "buyer", None) == "SUBMISSION":
                    portfolio["cash"][product] -= price * quantity
                elif getattr(trade, "seller", None) == "SUBMISSION":
                    portfolio["cash"][product] += price * quantity

                last_timestamp = max(last_timestamp, trade_timestamp)

            portfolio["last_trade_timestamp"][product] = last_timestamp

    def _current_mids(self, state: TradingState, memory: dict[str, Any]) -> dict[str, float]:
        reference_mids = memory["portfolio"]["reference_mids"]
        mids = {PEPPER: float(reference_mids.get(PEPPER, PEPPER_REFERENCE_MID)),
                OSMIUM: float(reference_mids.get(OSMIUM, OSMIUM_REFERENCE_MID))}

        for product in (PEPPER, OSMIUM):
            order_depth = state.order_depths.get(product)
            if order_depth is None:
                continue
            mid = _wall_mid(order_depth)
            if mid is not None:
                mids[product] = mid
                reference_mids[product] = mids[product]
        return mids

    def _initialize_starting_cash(self, memory: dict[str, Any], mids: dict[str, float]) -> None:
        portfolio = memory["portfolio"]
        if portfolio["starting_cash"] is not None:
            return

        pepper_mid = mids.get(PEPPER, PEPPER_REFERENCE_MID)
        portfolio["starting_cash"] = pepper_mid * PEPPER_EXCHANGE_LIMIT / PEPPER_STARTING_CASH_PROPORTION
