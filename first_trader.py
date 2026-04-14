from typing import List
import string
from prosperity3bt.datamodel import OrderDepth, UserId, TradingState, Order

class Trader:

    def run(self, state: TradingState):
        result = {}

        # 1. Handle EMERALDS (Market Making Strategy)
        if "EMERALDS" in state.order_depths:
            product = "EMERALDS"
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Constants for Emeralds
            fair_value = 10000
            position_limit = 20 # Adjust to 80 if your specific round allows it
            current_pos = state.position.get(product, 0)

            # Spread settings: How far from fair value do we want to post?
            # Buying at 9998 and selling at 10002 gives a spread of 4.
            half_spread = 2

            bid_price = fair_value - half_spread
            ask_price = fair_value + half_spread

            # Calculate how much room we have to buy/sell
            # If current_pos is 10, we can only buy 10 more (20 - 10)
            # but we could sell 30 (20 + 10) to reach -20.
            max_buy_qty = position_limit - current_pos
            max_sell_qty = -position_limit - current_pos # This will be negative

            if max_buy_qty > 0:
                print(f"BID {product}: {max_buy_qty}x{bid_price}")
                orders.append(Order(product, bid_price, max_buy_qty))

            if max_sell_qty < 0:
                print(f"ASK {product}: {max_sell_qty}x{ask_price}")
                orders.append(Order(product, ask_price, max_sell_qty))

            result[product] = orders

        # 2. Handle TOMATOES (VWAP Midpoint Calculation Skeleton)
        if "TOMATOES" in state.order_depths:
            product = "TOMATOES"
            order_depth: OrderDepth = state.order_depths[product]

            # We want: (BidPrice * BidVol + AskPrice * AskVol) / (Total Vol)
            # Note: sell_orders volumes are usually negative in the API, use abs()

            total_volume = 0
            weighted_sum = 0

            for price, vol in order_depth.buy_orders.items():
                weighted_sum += price * abs(vol)
                total_volume += abs(vol)

            for price, vol in order_depth.sell_orders.items():
                weighted_sum += price * abs(vol)
                total_volume += abs(vol)

            if total_volume > 0:
                vwap_midpoint = weighted_sum / total_volume
                print(f"TOMATOES VWAP Midpoint: {vwap_midpoint}")

            # TODO: Add trading logic based on vwap_midpoint
            result[product] = []

        traderData = "SAMPLE"
        conversions = 1
        return result, conversions, traderData
