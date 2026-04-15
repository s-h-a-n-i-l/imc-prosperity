from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState


class Trader:
    RESIN = "ASH_COATED_OSMIUM"
    PEPPER = "INTARIAN_PEPPER_ROOT"

    HARD_LIMITS = {
        RESIN: 20,
        PEPPER: 20,
    }
    SOFT_LIMITS = {
        RESIN: 15,
        PEPPER: 15,
    }

    RESIN_EMA_ALPHA = 2 / 51
    RESIN_ENTRY_THRESHOLD_MULT = 1.5
    RESIN_EXIT_THRESHOLD_MULT = 0.75
    RESIN_ORDER_SIZE = 3
    RESIN_INVENTORY_SKEW = 0.25

    PEPPER_ORDER_SIZE = 3
    PEPPER_HOLD_STEPS = 10

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}

        if self.RESIN in state.order_depths:
            resin_orders, resin_memory = self._trade_resin(
                state.order_depths[self.RESIN],
                state.position.get(self.RESIN, 0),
                memory.get("resin", {}),
            )
            result[self.RESIN] = resin_orders
            memory["resin"] = resin_memory

        if self.PEPPER in state.order_depths:
            pepper_orders, pepper_memory = self._trade_pepper(
                state.order_depths[self.PEPPER],
                state.position.get(self.PEPPER, 0),
                memory.get("pepper", {}),
            )
            result[self.PEPPER] = pepper_orders
            memory["pepper"] = pepper_memory

        trader_data = json.dumps(memory, separators=(",", ":"))
        conversions = 0
        return result, conversions, trader_data

    def _trade_resin(
        self,
        order_depth: OrderDepth,
        position: int,
        memory: dict,
    ) -> Tuple[List[Order], dict]:
        best_bid, best_bid_volume, best_ask, best_ask_volume = self._top_of_book(order_depth)
        if best_bid is None or best_ask is None:
            return [], memory

        spread = max(best_ask - best_bid, 1)
        book_mid = (best_bid + best_ask) / 2
        wall_mid = self._wall_mid(order_depth, fallback=book_mid)

        previous_ema = memory.get("wall_mid_ema")
        fair_value = wall_mid if previous_ema is None else (
            self.RESIN_EMA_ALPHA * wall_mid + (1 - self.RESIN_EMA_ALPHA) * previous_ema
        )
        memory["wall_mid_ema"] = fair_value

        entry_threshold = self.RESIN_ENTRY_THRESHOLD_MULT * spread
        exit_threshold = self.RESIN_EXIT_THRESHOLD_MULT * spread
        skew = self.RESIN_INVENTORY_SKEW * (position / self.SOFT_LIMITS[self.RESIN])

        orders: List[Order] = []
        working_position = position

        if best_ask <= fair_value - entry_threshold:
            buy_quantity = self._bounded_quantity(
                side="buy",
                position=working_position,
                hard_limit=self.HARD_LIMITS[self.RESIN],
                preferred_size=self.RESIN_ORDER_SIZE,
            )
            if buy_quantity > 0:
                orders.append(Order(self.RESIN, int(best_ask), buy_quantity))
                working_position += buy_quantity

        if best_bid >= fair_value + entry_threshold:
            sell_quantity = self._bounded_quantity(
                side="sell",
                position=working_position,
                hard_limit=self.HARD_LIMITS[self.RESIN],
                preferred_size=self.RESIN_ORDER_SIZE,
            )
            if sell_quantity > 0:
                orders.append(Order(self.RESIN, int(best_bid), -sell_quantity))
                working_position -= sell_quantity

        if working_position > 0 and best_bid >= fair_value - exit_threshold:
            clear_quantity = min(working_position, self.RESIN_ORDER_SIZE)
            if clear_quantity > 0:
                orders.append(Order(self.RESIN, int(best_bid), -clear_quantity))
                working_position -= clear_quantity

        if working_position < 0 and best_ask <= fair_value + exit_threshold:
            clear_quantity = min(-working_position, self.RESIN_ORDER_SIZE)
            if clear_quantity > 0:
                orders.append(Order(self.RESIN, int(best_ask), clear_quantity))
                working_position += clear_quantity

        if abs(book_mid - fair_value) < entry_threshold:
            improved_bid = best_bid + 1 if best_ask - best_bid >= 2 else best_bid
            improved_ask = best_ask - 1 if best_ask - best_bid >= 2 else best_ask
            fair_with_skew = fair_value - skew

            maker_buy_capacity = self._bounded_quantity(
                side="buy",
                position=working_position,
                hard_limit=self.HARD_LIMITS[self.RESIN],
                preferred_size=min(self.RESIN_ORDER_SIZE, self.SOFT_LIMITS[self.RESIN] - max(working_position, 0)),
            )
            if maker_buy_capacity > 0 and improved_bid < fair_with_skew:
                orders.append(Order(self.RESIN, int(improved_bid), maker_buy_capacity))

            maker_sell_capacity = self._bounded_quantity(
                side="sell",
                position=working_position,
                hard_limit=self.HARD_LIMITS[self.RESIN],
                preferred_size=min(self.RESIN_ORDER_SIZE, self.SOFT_LIMITS[self.RESIN] + min(working_position, 0)),
            )
            if maker_sell_capacity > 0 and improved_ask > fair_with_skew:
                orders.append(Order(self.RESIN, int(improved_ask), -maker_sell_capacity))

        return orders, memory

    def _trade_pepper(
        self,
        order_depth: OrderDepth,
        position: int,
        memory: dict,
    ) -> Tuple[List[Order], dict]:
        best_bid, best_bid_volume, best_ask, best_ask_volume = self._top_of_book(order_depth)
        if best_bid is None or best_ask is None:
            return [], memory

        imbalance = self._imbalance(best_bid_volume, best_ask_volume)
        signal_side = 0
        if imbalance > 0:
            signal_side = 1
        elif imbalance < 0:
            signal_side = -1

        hold_side = int(memory.get("hold_side", 0))
        hold_steps = int(memory.get("hold_steps", 0))
        if signal_side != 0:
            hold_side = signal_side
            hold_steps = self.PEPPER_HOLD_STEPS
        elif hold_steps > 0:
            hold_steps -= 1
            signal_side = hold_side
        else:
            hold_side = 0
            signal_side = 0

        memory["hold_side"] = hold_side
        memory["hold_steps"] = hold_steps

        orders: List[Order] = []
        working_position = position

        if signal_side > 0:
            if working_position < 0:
                clear_quantity = min(-working_position, self.PEPPER_ORDER_SIZE)
                if clear_quantity > 0:
                    orders.append(Order(self.PEPPER, int(best_bid), clear_quantity))
                    working_position += clear_quantity

            buy_quantity = self._bounded_quantity(
                side="buy",
                position=working_position,
                hard_limit=self.HARD_LIMITS[self.PEPPER],
                preferred_size=min(self.PEPPER_ORDER_SIZE, self.SOFT_LIMITS[self.PEPPER] - max(working_position, 0)),
            )
            if buy_quantity > 0:
                orders.append(Order(self.PEPPER, int(best_bid), buy_quantity))

        elif signal_side < 0:
            if working_position > 0:
                clear_quantity = min(working_position, self.PEPPER_ORDER_SIZE)
                if clear_quantity > 0:
                    orders.append(Order(self.PEPPER, int(best_ask), -clear_quantity))
                    working_position -= clear_quantity

            sell_quantity = self._bounded_quantity(
                side="sell",
                position=working_position,
                hard_limit=self.HARD_LIMITS[self.PEPPER],
                preferred_size=min(self.PEPPER_ORDER_SIZE, self.SOFT_LIMITS[self.PEPPER] + min(working_position, 0)),
            )
            if sell_quantity > 0:
                orders.append(Order(self.PEPPER, int(best_ask), -sell_quantity))

        else:
            if working_position > 0:
                clear_quantity = min(working_position, self.PEPPER_ORDER_SIZE)
                orders.append(Order(self.PEPPER, int(best_ask), -clear_quantity))
            elif working_position < 0:
                clear_quantity = min(-working_position, self.PEPPER_ORDER_SIZE)
                orders.append(Order(self.PEPPER, int(best_bid), clear_quantity))

        return orders, memory

    def _load_memory(self, trader_data: str) -> dict:
        if not trader_data:
            return {"resin": {}, "pepper": {"hold_side": 0, "hold_steps": 0}}
        try:
            data = json.loads(trader_data)
        except json.JSONDecodeError:
            data = {}
        data.setdefault("resin", {})
        data.setdefault("pepper", {"hold_side": 0, "hold_steps": 0})
        return data

    def _top_of_book(
        self,
        order_depth: OrderDepth,
    ) -> Tuple[Optional[int], int, Optional[int], int]:
        best_bid = None
        best_bid_volume = 0
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            best_bid_volume = abs(order_depth.buy_orders[best_bid])

        best_ask = None
        best_ask_volume = 0
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            best_ask_volume = abs(order_depth.sell_orders[best_ask])

        return best_bid, best_bid_volume, best_ask, best_ask_volume

    def _wall_mid(self, order_depth: OrderDepth, fallback: float) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return fallback

        wall_bid = max(order_depth.buy_orders.items(), key=lambda item: (abs(item[1]), item[0]))[0]
        wall_ask = min(order_depth.sell_orders.items(), key=lambda item: (-abs(item[1]), item[0]))[0]
        return (wall_bid + wall_ask) / 2

    def _imbalance(self, bid_volume: int, ask_volume: int) -> float:
        denominator = bid_volume + ask_volume
        if denominator <= 0:
            return 0.0
        return (bid_volume - ask_volume) / denominator

    def _bounded_quantity(
        self,
        side: str,
        position: int,
        hard_limit: int,
        preferred_size: int,
    ) -> int:
        preferred_size = max(int(preferred_size), 0)
        if preferred_size <= 0:
            return 0
        if side == "buy":
            return max(min(preferred_size, hard_limit - position), 0)
        return max(min(preferred_size, hard_limit + position), 0)
