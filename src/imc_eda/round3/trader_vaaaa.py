from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


class Trader:
    def run(self, state: TradingState):
        trades: dict[str, list[Order]] = {} 
        
        # Iterate over EVERY product the exchange sends us (GEL, FRUIT, and all VEV Options)
        for product, order_depth in state.order_depths.items():
            orders = []
            
            # 1. Figure out limits based on the product name
            limit = 300 if "VEV" in product else 200
            
            # Get current position
            position = state.position.get(product, 0)
            
            # 2. Find the top of the book
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            
            # If the book is empty on either side, skip trading it this tick
            if best_bid is None or best_ask is None:
                continue
                
            # 3. Calculate basic penny prices
            spread = best_ask - best_bid
            
            # If spread > 1, we can step inside it. Otherwise, we just join the top quotes.
            my_bid = best_bid + 1 if spread > 1 else best_bid
            my_ask = best_ask - 1 if spread > 1 else best_ask
            
            # 4. Inventory Management (Avoid holding too much)
            # If we hold more than half our limit, back off the quotes to skew inventory
            safe_threshold = limit * 0.5
            
            if position > safe_threshold:
                # Too much inventory! Lower our bid so we stop buying, and lower our ask to sell faster.
                my_bid -= 1
                my_ask -= 1
            elif position < -safe_threshold:
                # Too short! Raise our ask so we stop selling, and raise our bid to buy faster.
                my_bid += 1
                my_ask += 1
                
            # STRICT GUARDRAIL: Never let our bid equal or exceed our ask (self-matching)
            if my_bid >= my_ask:
                my_bid = best_bid
                my_ask = best_ask

            # 5. Order Sizing
            # Take small bites (e.g., 20) so we don't wipe out the book, bounded by our actual limits
            chunk_size = 20
            buy_qty = min(chunk_size, limit - position)
            sell_qty = min(chunk_size, position + limit)
            
            # 6. Submit Orders
            if sell_qty > 0:
                orders.append(Order(product, my_ask, -sell_qty))
            if buy_qty > 0:
                orders.append(Order(product, my_bid, buy_qty))
                
            trades[product] = orders

        # We aren't tracking anything across ticks, so traderData stays empty!
        return trades, 0, ""