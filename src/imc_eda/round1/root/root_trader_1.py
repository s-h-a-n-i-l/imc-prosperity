from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState


class Trader:
    PEPPER = "INTARIAN_PEPPER_ROOT"

    MAX_POSITION = 20
    ORDER_SIZE = 5

    LINEAR_WINDOW = 40
    RESIDUAL_WINDOW = 25
    BASE_K = 1.6
    MIN_SIGMA = 1.0
    INVENTORY_SKEW = 0.25

    MAX_AGGRESSION = 1
    K_TIGHTEN_PER_MISS = 0.15

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}

        if self.PEPPER in state.order_depths:
            pepper_orders, pepper_memory = self._trade_pepper(
                order_depth=state.order_depths[self.PEPPER],
                position=state.position.get(self.PEPPER, 0),
                own_trades=state.own_trades.get(self.PEPPER, []),
                memory=memory.get("pepper", {}),
                timestamp=state.timestamp,
            )
            result[self.PEPPER] = pepper_orders
            memory["pepper"] = pepper_memory

        trader_data = json.dumps(memory, separators=(",", ":"))
        conversions = 0
        return result, conversions, trader_data

    def _trade_pepper(
        self,
        order_depth: OrderDepth,
        position: int,
        own_trades: List,
        memory: dict,
        timestamp: int,
    ) -> Tuple[List[Order], dict]:
        best_bid, best_ask = self._top_of_book(order_depth)
        if best_bid is None or best_ask is None:
            return [], memory

        current_price = (best_bid + best_ask) / 2
        fair_value, slope, intercept = self._update_linear_fair_value(memory, current_price)

        residual = current_price - fair_value
        residual_history = list(memory.get("residual_history", []))
        residual_history.append(residual)
        residual_history = residual_history[-self.RESIDUAL_WINDOW :]
        memory["residual_history"] = residual_history

        sigma = self._rolling_std(residual_history)
        buy_aggression, buy_miss_count = self._update_aggressiveness(
            memory=memory,
            position=position,
            own_trades=own_trades,
            timestamp=timestamp,
            side="buy",
        )
        sell_aggression, sell_miss_count = self._update_aggressiveness(
            memory=memory,
            position=position,
            own_trades=own_trades,
            timestamp=timestamp,
            side="sell",
        )

        adjusted_residual = residual - self.INVENTORY_SKEW * position
        buy_k = max(self.BASE_K - buy_miss_count * self.K_TIGHTEN_PER_MISS, 0.6)
        sell_k = max(self.BASE_K - sell_miss_count * self.K_TIGHTEN_PER_MISS, 0.6)
        upper = sell_k * sigma
        lower = -buy_k * sigma

        orders: List[Order] = []
        signal_side = 0
        if adjusted_residual < lower:
            signal_side = 1
        elif adjusted_residual > upper:
            signal_side = -1

        target_price: Optional[int] = None
        if signal_side > 0:
            quantity = self._buy_capacity(position, self.ORDER_SIZE)
            if quantity > 0:
                target_price = min(best_bid + buy_aggression, best_ask)
                orders.append(Order(self.PEPPER, target_price, quantity))
                memory["pending_buy_order"] = {
                    "side": "buy",
                    "timestamp": timestamp,
                    "price": target_price,
                    "position_before": position,
                }
        elif signal_side < 0:
            quantity = self._sell_capacity(position, self.ORDER_SIZE)
            if quantity > 0:
                target_price = max(best_ask - sell_aggression, best_bid)
                orders.append(Order(self.PEPPER, target_price, -quantity))
                memory["pending_sell_order"] = {
                    "side": "sell",
                    "timestamp": timestamp,
                    "price": target_price,
                    "position_before": position,
                }

        memory["last_position"] = position
        memory["last_timestamp"] = timestamp
        memory["last_signal"] = signal_side
        memory["last_sigma"] = sigma
        memory["last_residual"] = residual
        memory["last_adjusted_residual"] = adjusted_residual
        memory["buy_aggression_level"] = buy_aggression
        memory["sell_aggression_level"] = sell_aggression
        memory["buy_miss_count"] = buy_miss_count
        memory["sell_miss_count"] = sell_miss_count
        memory["last_fair_value"] = fair_value
        memory["last_slope"] = slope
        memory["last_intercept"] = intercept

        return orders, memory

    def _update_linear_fair_value(self, memory: dict, price: float) -> Tuple[float, float, float]:
        price_history = list(memory.get("price_history", []))
        price_history.append(float(price))
        price_history = price_history[-self.LINEAR_WINDOW :]
        memory["price_history"] = price_history

        sample_count = len(price_history)
        if sample_count == 1:
            slope = 0.0
            intercept = float(price_history[0])
            fair_value = intercept
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
        memory: dict,
        position: int,
        own_trades: List,
        timestamp: int,
        side: str,
    ) -> Tuple[int, int]:
        pending_key = f"pending_{side}_order"
        miss_key = f"{side}_miss_count"
        aggression_key = f"{side}_aggression_level"

        pending_order = memory.get(pending_key, {})
        miss_count = int(memory.get(miss_key, 0))

        if not pending_order:
            return min(int(memory.get(aggression_key, 0)), self.MAX_AGGRESSION), miss_count

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
            reduced_aggression = min(reduced_miss_count, self.MAX_AGGRESSION)
            return reduced_aggression, reduced_miss_count

        if timestamp > int(pending_order.get("timestamp", timestamp)):
            miss_count += 1
            memory[pending_key] = {}

        aggression_level = min(miss_count, self.MAX_AGGRESSION)
        return aggression_level, miss_count

    def _pending_order_filled(
        self,
        pending_order: dict,
        position: int,
        own_trades: List,
        timestamp: int,
        side: str,
    ) -> bool:
        previous_position = int(pending_order.get("position_before", position))

        if side == "buy" and position > previous_position:
            return True
        if side == "sell" and position < previous_position:
            return True

        for trade in own_trades:
            if trade.timestamp != timestamp:
                continue
            if side == "buy" and trade.buyer == "SUBMISSION":
                return True
            if side == "sell" and trade.seller == "SUBMISSION":
                return True
        return False

    def _rolling_std(self, values: List[float]) -> float:
        if len(values) < 2:
            return self.MIN_SIGMA
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
        return max(variance**0.5, self.MIN_SIGMA)

    def _top_of_book(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        return best_bid, best_ask

    def _buy_capacity(self, position: int, preferred_size: int) -> int:
        return max(min(preferred_size, self.MAX_POSITION - position), 0)

    def _sell_capacity(self, position: int, preferred_size: int) -> int:
        return max(min(preferred_size, self.MAX_POSITION + position), 0)

    def _load_memory(self, trader_data: str) -> dict:
        if not trader_data:
            return {"pepper": self._default_pepper_memory()}
        try:
            data = json.loads(trader_data)
        except json.JSONDecodeError:
            data = {}
        data.setdefault("pepper", self._default_pepper_memory())
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
        return data

    def _default_pepper_memory(self) -> dict:
        return {
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
        }
