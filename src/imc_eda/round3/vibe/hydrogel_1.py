from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState


HYDROGEL = "HYDROGEL_PACK"
LIMIT = 200
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
    FAIR_VALUE = 10000.0
    TAKE_EDGE = 8
    CLEAR_EDGE = 3
    QUOTE_EDGE = 7
    ORDER_SIZE = 28
    SOFT_LIMIT = 170
    INVENTORY_SKEW = 0.06

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        result: dict[str, list[Order]] = {}

        depth = state.order_depths.get(HYDROGEL)
        if depth is not None:
            orders, product_memory = self._trade(depth, state.position.get(HYDROGEL, 0), memory.get("hydrogel", {}))
            result[HYDROGEL] = orders
            memory["hydrogel"] = product_memory

        return result, 0, json.dumps(memory, separators=(",", ":"))

    def _trade(self, depth: OrderDepth, position: int, memory: dict[str, Any]) -> tuple[list[Order], dict[str, Any]]:
        best_bid, best_bid_volume, best_ask, best_ask_volume = self._top_of_book(depth)
        if best_bid is None or best_ask is None:
            return [], memory

        fair = self.FAIR_VALUE - self.INVENTORY_SKEW * position
        orders: list[Order] = []
        working_position = position

        if best_ask <= fair - self.TAKE_EDGE:
            qty = self._capacity("buy", working_position, LIMIT, self.ORDER_SIZE)
            if qty > 0:
                orders.append(Order(HYDROGEL, best_ask, qty))
                working_position += qty

        if best_bid >= fair + self.TAKE_EDGE:
            qty = self._capacity("sell", working_position, LIMIT, self.ORDER_SIZE)
            if qty > 0:
                orders.append(Order(HYDROGEL, best_bid, -qty))
                working_position -= qty

        if working_position > 0 and best_bid >= fair - self.CLEAR_EDGE:
            qty = min(working_position, self.ORDER_SIZE)
            if qty > 0:
                orders.append(Order(HYDROGEL, best_bid, -qty))
                working_position -= qty

        if working_position < 0 and best_ask <= fair + self.CLEAR_EDGE:
            qty = min(-working_position, self.ORDER_SIZE)
            if qty > 0:
                orders.append(Order(HYDROGEL, best_ask, qty))
                working_position += qty

        bid_price = min(best_bid + 1, int(fair - self.QUOTE_EDGE))
        ask_price = max(best_ask - 1, int(fair + self.QUOTE_EDGE))
        if bid_price < ask_price:
            buy_size = min(self.ORDER_SIZE, max(self.SOFT_LIMIT - max(working_position, 0), 0))
            buy_qty = self._capacity("buy", working_position, LIMIT, buy_size)
            if buy_qty > 0 and bid_price < fair:
                orders.append(Order(HYDROGEL, int(bid_price), buy_qty))

            sell_size = min(self.ORDER_SIZE, max(self.SOFT_LIMIT + min(working_position, 0), 0))
            sell_qty = self._capacity("sell", working_position, LIMIT, sell_size)
            if sell_qty > 0 and ask_price > fair:
                orders.append(Order(HYDROGEL, int(ask_price), -sell_qty))

        memory["last_fair"] = fair
        memory["last_best_bid"] = best_bid
        memory["last_best_ask"] = best_ask
        memory["last_position"] = position
        memory["last_order_count"] = len(orders)
        return orders, memory

    def _load_memory(self, trader_data: str) -> dict[str, Any]:
        if not trader_data:
            return {"hydrogel": {}}
        try:
            data = json.loads(trader_data)
        except json.JSONDecodeError:
            data = {}
        data.setdefault("hydrogel", {})
        return data

    def _top_of_book(self, depth: OrderDepth) -> tuple[int | None, int, int | None, int]:
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        best_bid_volume = abs(depth.buy_orders[best_bid]) if best_bid is not None else 0
        best_ask_volume = abs(depth.sell_orders[best_ask]) if best_ask is not None else 0
        return best_bid, best_bid_volume, best_ask, best_ask_volume

    def _capacity(self, side: str, position: int, limit: int, preferred: int) -> int:
        preferred = max(int(preferred), 0)
        if side == "buy":
            return max(min(preferred, limit - position), 0)
        return max(min(preferred, limit + position), 0)
