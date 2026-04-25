from __future__ import annotations

import json
import math
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState


HYDROGEL = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"
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
ROUND3_LIMITS = {
    HYDROGEL: 200,
    VELVET: 200,
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
    HYDROGEL_FAIR_VALUE = 10000.0
    HYDROGEL_TAKE_EDGE = 8
    HYDROGEL_CLEAR_EDGE = 3
    HYDROGEL_QUOTE_EDGE = 7
    HYDROGEL_ORDER_SIZE = 28
    HYDROGEL_SOFT_LIMIT = 170
    HYDROGEL_INVENTORY_SKEW = 0.06

    VELVET_EMA_ALPHA = 2 / 31
    VELVET_TREND_WINDOW = 12
    VELVET_IMBALANCE_SCALE = 2.0
    VELVET_TREND_SCALE = 2.0
    VELVET_QUOTE_EDGE = 1.0
    VELVET_ORDER_SIZE = 24
    VELVET_SOFT_LIMIT = 170
    VELVET_INVENTORY_SKEW = 0.04

    VOUCHER_BASE_IV = 0.235
    VOUCHER_TARGET_PRODUCTS = ("VEV_5300", "VEV_6000", "VEV_6500")
    VOUCHER_IV_BIAS = {"VEV_5300": -0.008, "VEV_5400": 0.017}
    VOUCHER_ENTRY_EDGE = 2.5
    VOUCHER_EXIT_EDGE = 0.25
    VOUCHER_ORDER_SIZE = 24
    VOUCHER_SOFT_LIMIT = 220
    VOUCHER_MIN_PRICE = 1

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        result: dict[str, list[Order]] = {}

        hydrogel_depth = state.order_depths.get(HYDROGEL)
        if hydrogel_depth is not None:
            orders, product_memory = self._trade_hydrogel(
                hydrogel_depth,
                state.position.get(HYDROGEL, 0),
                memory.get("hydrogel", {}),
            )
            result[HYDROGEL] = orders
            memory["hydrogel"] = product_memory

        velvet_depth = state.order_depths.get(VELVET)
        if velvet_depth is not None:
            orders, product_memory = self._trade_velvet(
                velvet_depth,
                state.position.get(VELVET, 0),
                memory.get("velvet", {}),
            )
            result[VELVET] = orders
            memory["velvet"] = product_memory

        underlying_mid = self._mid_price(velvet_depth)
        if underlying_mid is None:
            underlying_mid = float(memory.get("last_underlying_mid", 5250.0))
        memory["last_underlying_mid"] = underlying_mid

        tte_years = 7.0 / 365.0
        voucher_memory = memory.get("vouchers", {})
        for product in self.VOUCHER_TARGET_PRODUCTS:
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            orders, product_memory = self._trade_voucher(
                product,
                depth,
                state.position.get(product, 0),
                underlying_mid,
                tte_years,
                voucher_memory.get(product, {}),
            )
            result[product] = orders
            voucher_memory[product] = product_memory
        memory["vouchers"] = voucher_memory

        return result, 0, json.dumps(memory, separators=(",", ":"))

    def _trade_hydrogel(self, depth: OrderDepth, position: int, memory: dict[str, Any]) -> tuple[list[Order], dict[str, Any]]:
        best_bid, _, best_ask, _ = self._top_of_book(depth)
        if best_bid is None or best_ask is None:
            return [], memory

        fair = self.HYDROGEL_FAIR_VALUE - self.HYDROGEL_INVENTORY_SKEW * position
        orders: list[Order] = []
        working_position = position

        if best_ask <= fair - self.HYDROGEL_TAKE_EDGE:
            qty = self._capacity("buy", working_position, ROUND3_LIMITS[HYDROGEL], self.HYDROGEL_ORDER_SIZE)
            if qty > 0:
                orders.append(Order(HYDROGEL, best_ask, qty))
                working_position += qty

        if best_bid >= fair + self.HYDROGEL_TAKE_EDGE:
            qty = self._capacity("sell", working_position, ROUND3_LIMITS[HYDROGEL], self.HYDROGEL_ORDER_SIZE)
            if qty > 0:
                orders.append(Order(HYDROGEL, best_bid, -qty))
                working_position -= qty

        if working_position > 0 and best_bid >= fair - self.HYDROGEL_CLEAR_EDGE:
            qty = min(working_position, self.HYDROGEL_ORDER_SIZE)
            if qty > 0:
                orders.append(Order(HYDROGEL, best_bid, -qty))
                working_position -= qty
        elif working_position < 0 and best_ask <= fair + self.HYDROGEL_CLEAR_EDGE:
            qty = min(-working_position, self.HYDROGEL_ORDER_SIZE)
            if qty > 0:
                orders.append(Order(HYDROGEL, best_ask, qty))
                working_position += qty

        bid_price = min(best_bid + 1, int(fair - self.HYDROGEL_QUOTE_EDGE))
        ask_price = max(best_ask - 1, int(fair + self.HYDROGEL_QUOTE_EDGE))
        if bid_price < ask_price:
            buy_qty = self._capacity(
                "buy",
                working_position,
                ROUND3_LIMITS[HYDROGEL],
                min(self.HYDROGEL_ORDER_SIZE, max(self.HYDROGEL_SOFT_LIMIT - max(working_position, 0), 0)),
            )
            sell_qty = self._capacity(
                "sell",
                working_position,
                ROUND3_LIMITS[HYDROGEL],
                min(self.HYDROGEL_ORDER_SIZE, max(self.HYDROGEL_SOFT_LIMIT + min(working_position, 0), 0)),
            )
            if buy_qty > 0 and bid_price < fair:
                orders.append(Order(HYDROGEL, int(bid_price), buy_qty))
            if sell_qty > 0 and ask_price > fair:
                orders.append(Order(HYDROGEL, int(ask_price), -sell_qty))

        memory["last_fair"] = fair
        memory["last_order_count"] = len(orders)
        return orders, memory

    def _trade_velvet(self, depth: OrderDepth, position: int, memory: dict[str, Any]) -> tuple[list[Order], dict[str, Any]]:
        best_bid, best_bid_volume, best_ask, best_ask_volume = self._top_of_book(depth)
        if best_bid is None or best_ask is None:
            return [], memory

        mid = (best_bid + best_ask) / 2
        ema = memory.get("ema")
        ema = mid if ema is None else self.VELVET_EMA_ALPHA * mid + (1 - self.VELVET_EMA_ALPHA) * float(ema)

        mids = [float(value) for value in memory.get("mids", [])]
        mids.append(mid)
        mids = mids[-self.VELVET_TREND_WINDOW :]
        trend = 0.0 if len(mids) < 2 else (mids[-1] - mids[0]) / (len(mids) - 1)
        imbalance = self._imbalance(best_bid_volume, best_ask_volume)
        signal = self.VELVET_IMBALANCE_SCALE * imbalance + self.VELVET_TREND_SCALE * trend
        fair = ema + signal - self.VELVET_INVENTORY_SKEW * position

        spread = best_ask - best_bid
        improved_bid = best_bid + 1 if spread >= 5 else best_bid
        improved_ask = best_ask - 1 if spread >= 5 else best_ask

        orders: list[Order] = []
        if improved_bid <= fair - self.VELVET_QUOTE_EDGE and position < self.VELVET_SOFT_LIMIT:
            qty = self._capacity("buy", position, ROUND3_LIMITS[VELVET], min(self.VELVET_ORDER_SIZE, self.VELVET_SOFT_LIMIT - position))
            if qty > 0:
                orders.append(Order(VELVET, improved_bid, qty))
        if improved_ask >= fair + self.VELVET_QUOTE_EDGE and position > -self.VELVET_SOFT_LIMIT:
            qty = self._capacity("sell", position, ROUND3_LIMITS[VELVET], min(self.VELVET_ORDER_SIZE, self.VELVET_SOFT_LIMIT + position))
            if qty > 0:
                orders.append(Order(VELVET, improved_ask, -qty))

        memory["ema"] = ema
        memory["mids"] = mids
        memory["last_fair"] = fair
        memory["last_signal"] = signal
        memory["last_order_count"] = len(orders)
        return orders, memory

    def _trade_voucher(
        self,
        product: str,
        depth: OrderDepth,
        position: int,
        underlying_mid: float,
        tte_years: float,
        memory: dict[str, Any],
    ) -> tuple[list[Order], dict[str, Any]]:
        best_bid, _, best_ask, _ = self._top_of_book(depth)
        if best_bid is None or best_ask is None:
            return [], memory

        if product in ("VEV_6000", "VEV_6500"):
            qty = self._capacity("buy", position, ROUND3_LIMITS[product], min(self.VOUCHER_ORDER_SIZE, self.VOUCHER_SOFT_LIMIT - max(position, 0)))
            orders = [Order(product, best_bid, qty)] if qty > 0 and best_bid == 0 else []
            memory["last_fair"] = 0.5
            memory["last_order_count"] = len(orders)
            return orders, memory

        strike = VOUCHERS[product]
        fair = self._call_price(
            underlying_mid,
            strike,
            tte_years,
            self.VOUCHER_BASE_IV + self.VOUCHER_IV_BIAS.get(product, 0.0),
        )
        fair = max(fair, max(underlying_mid - strike, 0.0), self.VOUCHER_MIN_PRICE)

        orders: list[Order] = []
        working_position = position
        if best_ask <= fair - self.VOUCHER_ENTRY_EDGE:
            qty = self._capacity("buy", working_position, ROUND3_LIMITS[product], self.VOUCHER_ORDER_SIZE)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                working_position += qty
        if best_bid >= fair + self.VOUCHER_ENTRY_EDGE:
            qty = self._capacity("sell", working_position, ROUND3_LIMITS[product], self.VOUCHER_ORDER_SIZE)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                working_position -= qty

        if working_position > 0 and best_bid >= fair - self.VOUCHER_EXIT_EDGE:
            qty = min(working_position, self.VOUCHER_ORDER_SIZE)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
        elif working_position < 0 and best_ask <= fair + self.VOUCHER_EXIT_EDGE:
            qty = min(-working_position, self.VOUCHER_ORDER_SIZE)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))

        memory["last_fair"] = fair
        memory["last_order_count"] = len(orders)
        return orders, memory

    def _load_memory(self, trader_data: str) -> dict[str, Any]:
        if not trader_data:
            return {"hydrogel": {}, "velvet": {}, "vouchers": {}}
        try:
            data = json.loads(trader_data)
        except json.JSONDecodeError:
            data = {}
        data.setdefault("hydrogel", {})
        data.setdefault("velvet", {})
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

    def _imbalance(self, bid_volume: int, ask_volume: int) -> float:
        total = bid_volume + ask_volume
        if total <= 0:
            return 0.0
        return (bid_volume - ask_volume) / total

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
