from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from statistics import median
from typing import Any, Optional

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
OPT_PRICES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

GEL_LIMIT = 200
FRUIT_LIMIT = 200
OPT_LIMIT = 300 

@dataclass(frozen=True)
class TraderParams:
    # GEL Params
    gel_fair_val: float = 10000.0
    gel_half_spread: float = 4.0
    gel_exposure_shift: float = 30.0 # Higher = allows holding more inventory before shifting quotes
    gel_obi_mult: float = 2.0 
    gel_snipe_threshold: float = 0.6 
    gel_trade_qty: int = 20

    # FRUIT Params
    fruit_half_spread: float = 3.0
    fruit_exposure_shift: float = 30.0 
    fruit_ema_alpha: float = 0.10 
    fruit_obi_mult: float = 2.5 
    fruit_snipe_threshold: float = 0.55 
    fruit_trade_qty: int = 20

    # Option Params
    opt_edge_required: float = 6.0 # Increased safety margin so we don't bleed on the spread
    opt_trade_qty_limit: int = 30 
    opt_risk_free_rate: float = 0.0

# --- BLACK SCHOLES ---
def norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x: float) -> float:
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * norm_pdf(d1) * math.sqrt(T)

def implied_volatility(target_price: float, S: float, K: float, T: float, r: float) -> float:
    if target_price <= 0: return 0.001
    sigma = 0.5  
    for _ in range(100): 
        price = bs_call_price(S, K, T, r, sigma)
        diff = price - target_price
        if abs(diff) < 1e-4: break
        vega = bs_vega(S, K, T, r, sigma)
        if vega < 1e-8: break
        sigma -= diff / vega
        sigma = max(sigma, 0.001) 
    return sigma

@dataclass(frozen=True)
class BookSnapshot:
    best_bid: int | None
    best_ask: int | None
    top_bid_depth: float
    top_ask_depth: float
    mid_price: float | None

def _sorted_levels(book: dict[int, int] | dict[float, float], *, reverse: bool) -> list[tuple[int, float]]:
    levels = [(int(price), abs(float(volume))) for price, volume in book.items() if float(volume) != 0.0]
    return sorted(levels, key=lambda item: item[0], reverse=reverse)

def _wall_mid(order_depth: OrderDepth) -> Optional[float]:
    if not order_depth.buy_orders or not order_depth.sell_orders: return None
    buy_wall = max(order_depth.buy_orders.items(), key=lambda x: x[1])[0]
    sell_wall = max(order_depth.sell_orders.items(), key=lambda x: abs(x[1]))[0]
    return (buy_wall + sell_wall) / 2.0

def build_book_snapshot(order_depth: OrderDepth, fallback_mid: float | None = None) -> BookSnapshot:
    buy_orders = getattr(order_depth, "buy_orders", {}) or {}
    sell_orders = getattr(order_depth, "sell_orders", {}) or {}

    bid_levels = _sorted_levels(buy_orders, reverse=True)
    ask_levels = _sorted_levels(sell_orders, reverse=False)

    best_bid = bid_levels[0][0] if bid_levels else None
    best_ask = ask_levels[0][0] if ask_levels else None
    top_bid_depth = bid_levels[0][1] if bid_levels else 0.0
    top_ask_depth = ask_levels[0][1] if ask_levels else 0.0

    if best_bid is not None and best_ask is not None:
        mid_price = _wall_mid(order_depth)
    else:
        mid_price = fallback_mid

    return BookSnapshot(
        best_bid=best_bid, best_ask=best_ask,
        top_bid_depth=top_bid_depth, top_ask_depth=top_ask_depth,
        mid_price=mid_price
    )

def calc_top_imbalance(book: BookSnapshot) -> float:
    """Returns -1.0 (all asks) to +1.0 (all bids) based on top of book volume."""
    total_top_vol = book.top_bid_depth + book.top_ask_depth
    if total_top_vol == 0: return 0.0
    return (book.top_bid_depth - book.top_ask_depth) / total_top_vol

class Trader:
    def __init__(self, params: TraderParams | None = None):
        self.p = params or TraderParams()

    def run(self, state: TradingState):
        memory = self._load_memory(getattr(state, "traderData", ""))
        trades: dict[str, list[Order]] = {} 
        
        opt_trades, memory = self.trade_fruit_opts(state, memory)
        trades.update(opt_trades)

        if GEL in state.order_depths:
            trades[GEL], memory[GEL] = self.trade_gel(state, memory[GEL])
            
        if FRUIT in state.order_depths:
            trades[FRUIT], memory[FRUIT] = self.trade_fruit(state, memory[FRUIT])

        traderData = json.dumps(memory, separators=(",", ":"))
        return trades, 0, traderData
    
    def trade_gel(self, state: TradingState, memory: dict[str, Any]) -> tuple[list[Order], dict[str, Any]]:
        mid = _wall_mid(state.order_depths[GEL]) or memory["mid_price"] or self.p.gel_fair_val
        memory["mid_price"] = mid
        book = build_book_snapshot(state.order_depths[GEL], fallback_mid=mid)
        
        orders = []
        position = state.position.get(GEL, 0)
        buy_cap = min(self.p.gel_trade_qty, GEL_LIMIT - position)
        sell_cap = min(self.p.gel_trade_qty, position + GEL_LIMIT)
        
        obi = calc_top_imbalance(book)
        
        # 1. SNIPER LOGIC
        if obi > self.p.gel_snipe_threshold and buy_cap > 0 and book.best_ask is not None:
            take_qty = min(buy_cap, int(book.top_ask_depth))
            if take_qty > 0:
                orders.append(Order(GEL, book.best_ask, take_qty))
                buy_cap -= take_qty
                
        elif obi < -self.p.gel_snipe_threshold and sell_cap > 0 and book.best_bid is not None:
            take_qty = min(sell_cap, int(book.top_bid_depth))
            if take_qty > 0:
                orders.append(Order(GEL, book.best_bid, -take_qty))
                sell_cap -= take_qty

        # 2. MAKER LOGIC
        shift = obi * self.p.gel_obi_mult 
        shift -= (position / self.p.gel_exposure_shift) 
            
        our_buy = math.floor(self.p.gel_fair_val - self.p.gel_half_spread + shift)
        our_sell = math.ceil(self.p.gel_fair_val + self.p.gel_half_spread + shift)

        # STRICT GUARDRAIL: Never passively cross the market spread
        if book.best_ask is not None:
            our_buy = min(our_buy, book.best_ask - 1)
        if book.best_bid is not None:
            our_sell = max(our_sell, book.best_bid + 1)
            
        if our_buy >= our_sell:
            our_buy = book.best_bid if book.best_bid else our_buy
            our_sell = book.best_ask if book.best_ask else our_sell

        if book.best_ask is not None and sell_cap > 0:
            orders.append(Order(GEL, int(our_sell), -sell_cap))
        if book.best_bid is not None and buy_cap > 0:
            orders.append(Order(GEL, int(our_buy), buy_cap))

        return orders, memory
    
    def trade_fruit(self, state: TradingState, memory: dict[str, Any]) -> tuple[list[Order], dict[str, Any]]:
        mid = _wall_mid(state.order_depths[FRUIT]) or memory["mid_price"] or 5250.0
        memory["mid_price"] = mid
        
        ema = memory.get("ema_fair_val", mid)
        ema = (self.p.fruit_ema_alpha * mid) + ((1 - self.p.fruit_ema_alpha) * ema)
        memory["ema_fair_val"] = ema

        book = build_book_snapshot(state.order_depths[FRUIT], fallback_mid=mid)
        orders = []
        position = state.position.get(FRUIT, 0)
        
        buy_cap = min(self.p.fruit_trade_qty, FRUIT_LIMIT - position)
        sell_cap = min(self.p.fruit_trade_qty, position + FRUIT_LIMIT)
        
        obi = calc_top_imbalance(book)
        
        # 1. SNIPER LOGIC
        if obi > self.p.fruit_snipe_threshold and buy_cap > 0 and book.best_ask is not None:
            take_qty = min(buy_cap, int(book.top_ask_depth))
            if take_qty > 0:
                orders.append(Order(FRUIT, book.best_ask, take_qty))
                buy_cap -= take_qty
                
        elif obi < -self.p.fruit_snipe_threshold and sell_cap > 0 and book.best_bid is not None:
            take_qty = min(sell_cap, int(book.top_bid_depth))
            if take_qty > 0:
                orders.append(Order(FRUIT, book.best_bid, -take_qty))
                sell_cap -= take_qty

        # 2. MAKER LOGIC
        shift = obi * self.p.fruit_obi_mult 
        shift -= (position / self.p.fruit_exposure_shift) 

        our_buy = math.floor(ema - self.p.fruit_half_spread + shift)
        our_sell = math.ceil(ema + self.p.fruit_half_spread + shift)

        # STRICT GUARDRAIL: Never passively cross the market spread
        if book.best_ask is not None:
            our_buy = min(our_buy, book.best_ask - 1)
        if book.best_bid is not None:
            our_sell = max(our_sell, book.best_bid + 1)
            
        if our_buy >= our_sell:
            our_buy = book.best_bid if book.best_bid else our_buy
            our_sell = book.best_ask if book.best_ask else our_sell

        if book.best_ask is not None and sell_cap > 0:
            orders.append(Order(FRUIT, int(our_sell), -sell_cap))
        if book.best_bid is not None and buy_cap > 0:
            orders.append(Order(FRUIT, int(our_buy), buy_cap))

        return orders, memory
    
    def trade_fruit_opts(self, state: TradingState, memory: dict[str, Any]) -> tuple[dict[str, list[Order]], dict[str, Any]]:
        orders_dict = {}
        fruit_mid = _wall_mid(state.order_depths.get(FRUIT, OrderDepth())) or memory.get("last_fruit_mid", 5250.0)
        memory["last_fruit_mid"] = fruit_mid

        days_left = 7.0 - (state.timestamp / 1_000_000.0)
        T = max(days_left / 365.0, 0.0001) 

        ivs = {}
        for K in OPT_PRICES:
            symbol = f"VEV_{K}"
            if symbol not in state.order_depths: continue
            opt_mid = _wall_mid(state.order_depths[symbol])
            if opt_mid is None: continue
            
            # Avoid calculating IV on deep ITM options to prevent noise
            if K < fruit_mid - 200: continue 
            
            ivs[symbol] = implied_volatility(opt_mid, fruit_mid, K, T, self.p.opt_risk_free_rate)

        if not ivs: return {}, memory

        median_iv = median(ivs.values())

        for K in OPT_PRICES:
            symbol = f"VEV_{K}"
            if symbol not in state.order_depths: continue
            book = build_book_snapshot(state.order_depths[symbol])
            position = state.position.get(symbol, 0)
            fair_price = bs_call_price(fruit_mid, K, T, self.p.opt_risk_free_rate, median_iv)

            my_orders = []
            buy_cap = OPT_LIMIT - position
            sell_cap = position + OPT_LIMIT

            if book.best_ask is not None and fair_price > book.best_ask + self.p.opt_edge_required:
                vol = min(buy_cap, self.p.opt_trade_qty_limit) 
                if vol > 0: my_orders.append(Order(symbol, book.best_ask, vol))
            elif book.best_bid is not None and fair_price < book.best_bid - self.p.opt_edge_required:
                vol = min(sell_cap, self.p.opt_trade_qty_limit)
                if vol > 0: my_orders.append(Order(symbol, book.best_bid, -vol))

            if my_orders: orders_dict[symbol] = my_orders

        return orders_dict, memory
    
    def _load_memory(self, traderData: str) -> dict[str, Any]:
        if not traderData: return self._default_memory()
        try: data = json.loads(traderData)
        except json.JSONDecodeError: return self._default_memory()  
        default = self._default_memory()
        for k in default.keys():
            if k not in data: data[k] = default[k]
        return data
    
    def _default_memory(self) -> dict[str, Any]:
        return {
            "HYDROGEL_PACK": {"mid_price": None},
            "VELVETFRUIT_EXTRACT": {"mid_price": None, "ema_fair_val": None},
            "options": {"last_fruit_mid": 5250.0}
        }