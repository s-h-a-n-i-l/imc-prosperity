from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState


VELVET = "VELVETFRUIT_EXTRACT"
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
    EMA_ALPHA = 2 / 31
    TREND_WINDOW = 12
    IMBALANCE_SCALE = 2.0
    TREND_SCALE = 2.0
    QUOTE_EDGE = 1.0
    ORDER_SIZE = 24
    SOFT_LIMIT = 170
    INVENTORY_SKEW = 0.04

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        result: dict[str, list[Order]] = {}

        depth = state.order_depths.get(VELVET)
        if depth is not None:
            orders, product_memory = self._trade(depth, state.position.get(VELVET, 0), memory.get("velvet", {}))
            result[VELVET] = orders
            memory["velvet"] = product_memory

        return result, 0, json.dumps(memory, separators=(",", ":"))

    def _trade(self, depth: OrderDepth, position: int, memory: dict[str, Any]) -> tuple[list[Order], dict[str, Any]]:
        best_bid, best_bid_volume, best_ask, best_ask_volume = self._top_of_book(depth)
        if best_bid is None or best_ask is None:
            return [], memory

        mid = (best_bid + best_ask) / 2
        ema = memory.get("ema")
        ema = mid if ema is None else self.EMA_ALPHA * mid + (1 - self.EMA_ALPHA) * float(ema)

        mids = [float(value) for value in memory.get("mids", [])]
        mids.append(mid)
        mids = mids[-self.TREND_WINDOW :]
        trend = 0.0 if len(mids) < 2 else (mids[-1] - mids[0]) / (len(mids) - 1)

        imbalance = self._imbalance(best_bid_volume, best_ask_volume)
        signal = self.IMBALANCE_SCALE * imbalance + self.TREND_SCALE * trend
        fair = ema + signal - self.INVENTORY_SKEW * position
        orders: list[Order] = []
        working_position = position

        spread = best_ask - best_bid
        improved_bid = best_bid + 1 if spread >= 5 else best_bid
        improved_ask = best_ask - 1 if spread >= 5 else best_ask

        if improved_bid <= fair - self.QUOTE_EDGE and working_position < self.SOFT_LIMIT:
            qty = self._capacity("buy", working_position, LIMIT, min(self.ORDER_SIZE, self.SOFT_LIMIT - working_position))
            if qty > 0:
                orders.append(Order(VELVET, improved_bid, qty))

        if improved_ask >= fair + self.QUOTE_EDGE and working_position > -self.SOFT_LIMIT:
            qty = self._capacity("sell", working_position, LIMIT, min(self.ORDER_SIZE, self.SOFT_LIMIT + working_position))
            if qty > 0:
                orders.append(Order(VELVET, improved_ask, -qty))

        memory["ema"] = ema
        memory["mids"] = mids
        memory["last_mid"] = mid
        memory["last_trend"] = trend
        memory["last_imbalance"] = imbalance
        memory["last_signal"] = signal
        memory["last_position"] = position
        memory["last_order_count"] = len(orders)
        return orders, memory

    def _load_memory(self, trader_data: str) -> dict[str, Any]:
        if not trader_data:
            return {"velvet": {}}
        try:
            data = json.loads(trader_data)
        except json.JSONDecodeError:
            data = {}
        data.setdefault("velvet", {})
        return data

    def _top_of_book(self, depth: OrderDepth) -> tuple[int | None, int, int | None, int]:
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        best_bid_volume = abs(depth.buy_orders[best_bid]) if best_bid is not None else 0
        best_ask_volume = abs(depth.sell_orders[best_ask]) if best_ask is not None else 0
        return best_bid, best_bid_volume, best_ask, best_ask_volume

    def _imbalance(self, bid_volume: int, ask_volume: int) -> float:
        total = bid_volume + ask_volume
        if total <= 0:
            return 0.0
        return (bid_volume - ask_volume) / total

    def _capacity(self, side: str, position: int, limit: int, preferred: int) -> int:
        preferred = max(int(preferred), 0)
        if side == "buy":
            return max(min(preferred, limit - position), 0)
        return max(min(preferred, limit + position), 0)
