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

class Trader:
    def run(self, state: TradingState):
        # Check our positions, and if we aren't already, buy a single gel at whatever price is available and hold it forever
        # so IMC backtester tells us the true value 
        if state.position.get(GEL, 0) == 0:
            getattr(state.order_depths[GEL], "sell_orders", {}) or {}
        else:
            orders = {}
        return orders, 0, ""
