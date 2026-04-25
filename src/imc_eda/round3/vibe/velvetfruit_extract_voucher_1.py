from __future__ import annotations

import json
import math
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState


UNDERLYING = "VELVETFRUIT_EXTRACT"
VOUCHERS = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}
LIMIT = 300
ROUND3_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}

try:
    import prosperity4bt.data as _bt_data

    _bt_data.LIMITS.update(ROUND3_LIMITS)
except Exception:
    pass


class Trader:
    BASE_IV = 0.235
    TARGET_PRODUCTS = ("VEV_5300", "VEV_6000", "VEV_6500")
    IV_BIAS = {
        "VEV_5300": -0.008,
        "VEV_5400": 0.017,
    }
    ENTRY_EDGE = 2.5
    EXIT_EDGE = 0.25
    ORDER_SIZE = 24
    SOFT_LIMIT = 220
    MIN_PRICE = 1

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        result: dict[str, list[Order]] = {}

        underlying_mid = self._mid_price(state.order_depths.get(UNDERLYING))
        if underlying_mid is None:
            underlying_mid = float(memory.get("last_underlying_mid", 5250.0))
        memory["last_underlying_mid"] = underlying_mid

        tte_years = 7.0 / 365.0

        voucher_memory = memory.get("vouchers", {})
        for product in self.TARGET_PRODUCTS:
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            orders, product_memory = self._trade_voucher(
                product=product,
                depth=depth,
                position=state.position.get(product, 0),
                underlying_mid=underlying_mid,
                tte_years=tte_years,
                memory=voucher_memory.get(product, {}),
            )
            result[product] = orders
            voucher_memory[product] = product_memory

        memory["vouchers"] = voucher_memory
        return result, 0, json.dumps(memory, separators=(",", ":"))

    def _trade_voucher(
        self,
        product: str,
        depth: OrderDepth,
        position: int,
        underlying_mid: float,
        tte_years: float,
        memory: dict[str, Any],
    ) -> tuple[list[Order], dict[str, Any]]:
        best_bid, best_bid_volume, best_ask, best_ask_volume = self._top_of_book(depth)
        if best_bid is None or best_ask is None:
            return [], memory

        if product in ("VEV_6000", "VEV_6500"):
            qty = self._capacity("buy", position, LIMIT, min(self.ORDER_SIZE, self.SOFT_LIMIT - max(position, 0)))
            orders = [Order(product, best_bid, qty)] if qty > 0 and best_bid == 0 else []
            memory["last_fair"] = 0.5
            memory["last_underlying_mid"] = underlying_mid
            memory["last_position"] = position
            memory["last_order_count"] = len(orders)
            return orders, memory

        strike = VOUCHERS[product]
        fair = self._call_price(underlying_mid, strike, tte_years, self.BASE_IV + self.IV_BIAS.get(product, 0.0))
        fair = max(fair, max(underlying_mid - strike, 0.0), self.MIN_PRICE)

        orders: list[Order] = []
        working_position = position

        if best_ask <= fair - self.ENTRY_EDGE:
            qty = self._capacity("buy", working_position, LIMIT, self.ORDER_SIZE)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                working_position += qty

        if best_bid >= fair + self.ENTRY_EDGE:
            qty = self._capacity("sell", working_position, LIMIT, self.ORDER_SIZE)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                working_position -= qty

        if working_position > 0 and best_bid >= fair - self.EXIT_EDGE:
            qty = min(working_position, self.ORDER_SIZE)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                working_position -= qty

        if working_position < 0 and best_ask <= fair + self.EXIT_EDGE:
            qty = min(-working_position, self.ORDER_SIZE)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                working_position += qty

        spread = best_ask - best_bid
        if spread >= 2 and abs(position) < self.SOFT_LIMIT:
            bid_price = min(best_bid + 1, math.floor(fair - self.EXIT_EDGE))
            ask_price = max(best_ask - 1, math.ceil(fair + self.EXIT_EDGE))
            if bid_price < ask_price:
                buy_qty = self._capacity("buy", working_position, LIMIT, min(self.ORDER_SIZE, self.SOFT_LIMIT - max(working_position, 0)))
                sell_qty = self._capacity("sell", working_position, LIMIT, min(self.ORDER_SIZE, self.SOFT_LIMIT + min(working_position, 0)))
                if buy_qty > 0 and bid_price < fair - self.EXIT_EDGE:
                    orders.append(Order(product, int(bid_price), buy_qty))
                if sell_qty > 0 and ask_price > fair + self.EXIT_EDGE:
                    orders.append(Order(product, int(ask_price), -sell_qty))

        memory["last_fair"] = fair
        memory["last_underlying_mid"] = underlying_mid
        memory["last_position"] = position
        memory["last_order_count"] = len(orders)
        return orders, memory

    def _load_memory(self, trader_data: str) -> dict[str, Any]:
        if not trader_data:
            return {"vouchers": {}}
        try:
            data = json.loads(trader_data)
        except json.JSONDecodeError:
            data = {}
        data.setdefault("vouchers", {})
        return data

    def _mid_price(self, depth: OrderDepth | None) -> float | None:
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2

    def _top_of_book(self, depth: OrderDepth) -> tuple[int | None, int, int | None, int]:
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        best_bid_volume = abs(depth.buy_orders[best_bid]) if best_bid is not None else 0
        best_ask_volume = abs(depth.sell_orders[best_ask]) if best_ask is not None else 0
        return best_bid, best_bid_volume, best_ask, best_ask_volume

    def _call_price(self, spot: float, strike: float, tte_years: float, volatility: float) -> float:
        if tte_years <= 0 or volatility <= 0:
            return max(spot - strike, 0.0)
        sigma_sqrt_t = volatility * math.sqrt(tte_years)
        if sigma_sqrt_t <= 0:
            return max(spot - strike, 0.0)
        d1 = (math.log(max(spot, 1e-9) / strike) + 0.5 * volatility * volatility * tte_years) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t
        return spot * self._norm_cdf(d1) - strike * self._norm_cdf(d2)

    def _norm_cdf(self, value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    def _capacity(self, side: str, position: int, limit: int, preferred: int) -> int:
        preferred = max(int(preferred), 0)
        if side == "buy":
            return max(min(preferred, limit - position), 0)
        return max(min(preferred, limit + position), 0)
