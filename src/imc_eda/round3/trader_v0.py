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

# Tradable products
GEL = "HYDROGEL_PACK"
FRUIT = "VELVETFRUIT_EXTRACT"
FRUIT_OPT = "VEV_{n}" # fstring for a fruit option at n price, e.g. VEV_5000
# The vouchers are labeled VEV_4000, VEV_4500, VEV_5000, VEV_5100, VEV_5200, VEV_5300, VEV_5400, VEV_5500, VEV_6000, VEV_6500
OPT_PRICES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
# NOTE: The vouchers (options) expire in 7 days SINCE THE START OF ROUND 1! 
# so at the start of Round 2 options expire in 6 days, and so on. 

# Volume limits (max hold/short) for each product
GEL_LIMIT = 200
FRUIT_LIMIT = 200
OPT_LIMIT = 300 # for each of the 10 options

# Assumed fair values 
GEL_FAIR_VAL = 10000

# GEL market making settings
EXPOSURE_SHIFT = 10 # how many units of position do we need to shift our bid by 1
SIGNIFICANT_RESI = 10000 # how much accumulated residual do we count as significant

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

class Trader:
    def run(self, state: TradingState):
        memory = self._load_memory(getattr(state, "traderData", ""))
        
        # Array of orders for each product
        trades : dict[str, list[Order]] = {} 
        
        # For every option price available, calculate the implied volatility. 
        trades[GEL], memory[GEL] = self.trade_gel(state, memory[GEL])
        trades[FRUIT], memory[FRUIT] = self.trade_fruit(state, memory[FRUIT])
        trades[FRUIT_OPT], memory = self.trade_fruit_opt(state, memory)

        traderData = json.dumps(memory, separators=(",", ":"))
        return trades, 0, traderData
    
    def trade_gel(self, state: TradingState, memory: dict[str, Any]) -> list[Order] | dict[str, Any]:
        # Gel is a sort of mean-reverting product about 10,000. 
        mid = _wall_mid(state.order_depths[GEL])
        if mid is None:
            mid = memory["mid_price"]
        if mid is None:
            mid = GEL_FAIR_VAL # last resort, should only ever happen at start
            
        memory["mid_price"] = mid

        memory["cum_residual"] += mid - GEL_FAIR_VAL # integrate the deviation from a known fair value
        # if this deviation is very positive, we look to shorting (and vice versa)

        book_snapshot = build_book_snapshot(state.order_depths[GEL], fallback_mid=mid)
        
        
        orders = []
        position = state.position.get(GEL, 0)
        
        # Stupid trading strategy: Always submit an order one better than the best, on both sides, with some volume
        vol_buy = 20
        vol_sell = 20 # TODO: set these dynamically, and nudge prices too
        
        # The bots will NEVER BUY if we just offer the same best_bid and best_ask as the bots. We have to be better
        # Calculate the price we're willing to buy and sell for by considering appropriate factors available.
        # Explain the rationale in code comments. 
        halfdiff = 7 # half the "margin" between our buy and sell
        shift = 0 # if our price is shifted "up" or "down"

        if book_snapshot.best_bid > mid:
            # be concerned
            shift += 1

        if book_snapshot.best_ask < mid:
            shift -= 1
        
        if book_snapshot.book_state == "ask_only":
            # directional cue, going down
            shift -= 2
        
        if book_snapshot.book_state == "bid_only":
            # directional cue, going up 
            shift += 2
        
        # If memory["cum_residual"] exceeds a certain absolute size, it means the product is overdue to return to the mean, 
        # so increase/decrease our bid/ask accordingly (in the opposite direction to the accumulated diff)
        if abs(memory["cum_residual"]) > SIGNIFICANT_RESI:
            shift -= memory["cum_residual"]/SIGNIFICANT_RESI

        # Inventory management: lean against our current position
        shift -= (position / EXPOSURE_SHIFT) # if position is positive, lower prices (and vice versa)
            
        
        our_buy = mid - halfdiff + shift
        our_sell = mid + halfdiff + shift

        price_buy = max(book_snapshot.best_bid + 1, our_buy)
        price_sell = min(book_snapshot.best_ask - 1, our_sell)
        if book_snapshot.best_ask is not None:
            orders.append(Order(GEL, int(price_sell), -vol_sell))
        if book_snapshot.best_bid is not None:
            orders.append(Order(GEL, int(price_buy), vol_buy))

        # if the price has "crossed" the fair value, clear the integrated residual 
        if memory["cum_residual"] > SIGNIFICANT_RESI and mid < GEL_FAIR_VAL:
            memory["cum_residual"] = 0
        if memory["cum_residual"] < -SIGNIFICANT_RESI and mid > GEL_FAIR_VAL:
            memory["cum_residual"] = 0

        return orders, memory
    
    def trade_fruit(self, state: TradingState, memory: dict[str, Any]):
        return [], memory
    
    def trade_fruit_opt(self, state: TradingState, memory: dict[str, Any]):
        return [], memory
    
    def _load_memory(self, traderData: str) -> dict[str, Any]:
        if not traderData:
            return self._default_memory()

        try:
            data = json.loads(traderData)
        except json.JSONDecodeError:
            return self._default_memory()  

        return data
    
    def _default_memory(self) -> dict[str, Any]:
        return {
            "HYDROGEL_PACK": {
                "mid_price": None,
                "cum_residual": 0,
            },
            "VELVETFRUIT_EXTRACT": {
                "mid_price": None,
            },
        }