from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

TOP_LEVEL_BOOK_COLUMNS = [
    "bid_price_1",
    "bid_volume_1",
    "ask_price_1",
    "ask_volume_1",
]
BOOK_COLUMNS = [
    "bid_price_1",
    "bid_volume_1",
    "bid_price_2",
    "bid_volume_2",
    "bid_price_3",
    "bid_volume_3",
    "ask_price_1",
    "ask_volume_1",
    "ask_price_2",
    "ask_volume_2",
    "ask_price_3",
    "ask_volume_3",
]


def _group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("product", "day") if column in frame.columns]


def _top_volume_denominator(frame: pd.DataFrame) -> pd.Series:
    return (frame["bid_volume_1"].abs() + frame["ask_volume_1"].abs()).replace(0, np.nan)


def _bucketize(series: pd.Series, bucket_count: int = 10) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    ranked = valid.rank(method="first")
    bucketed = pd.qcut(ranked, q=min(bucket_count, len(valid)), labels=False, duplicates="drop")
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    result.loc[valid.index] = bucketed.astype(float)
    return result


def _select_wall_price(row: pd.Series, side: str) -> float:
    prices: list[float] = []
    volumes: list[float] = []
    for level in range(1, 4):
        price = row.get(f"{side}_price_{level}")
        volume = row.get(f"{side}_volume_{level}")
        if pd.notna(price) and pd.notna(volume):
            prices.append(float(price))
            volumes.append(abs(float(volume)))
    if not prices:
        return np.nan
    return prices[int(np.argmax(volumes))]


def prepare_quotes(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    sort_columns = [column for column in ("product", "day", "timestamp") if column in frame.columns]
    frame = frame.sort_values(sort_columns).reset_index(drop=True)

    fill_columns = [column for column in BOOK_COLUMNS if column in frame.columns]
    frame[fill_columns] = frame.groupby(_group_columns(frame))[fill_columns].ffill()
    frame = frame.dropna(subset=["bid_price_1", "ask_price_1"]).reset_index(drop=True)
    return frame


def compute_features(
    quotes: pd.DataFrame,
    rolling_windows: tuple[int, ...] = (20, 50),
    ema_windows: tuple[int, ...] = (20, 50),
) -> pd.DataFrame:
    frame = quotes.copy()
    frame["book_mid"] = (frame["bid_price_1"] + frame["ask_price_1"]) / 2
    frame["spread"] = frame["ask_price_1"] - frame["bid_price_1"]

    denominator = _top_volume_denominator(frame)
    frame["microprice"] = (
        frame["ask_price_1"] * frame["bid_volume_1"].abs() + frame["bid_price_1"] * frame["ask_volume_1"].abs()
    ) / denominator
    frame["imbalance"] = (frame["bid_volume_1"].abs() - frame["ask_volume_1"].abs()) / denominator

    total_bid_volume = (
        frame[[column for column in ("bid_volume_1", "bid_volume_2", "bid_volume_3") if column in frame.columns]]
        .fillna(0)
        .abs()
        .sum(axis=1)
    )
    total_ask_volume = (
        frame[[column for column in ("ask_volume_1", "ask_volume_2", "ask_volume_3") if column in frame.columns]]
        .fillna(0)
        .abs()
        .sum(axis=1)
    )
    total_denominator = (total_bid_volume + total_ask_volume).replace(0, np.nan)
    frame["book_imbalance"] = (total_bid_volume - total_ask_volume) / total_denominator

    frame["wall_bid_price"] = frame.apply(_select_wall_price, axis=1, side="bid")
    frame["wall_ask_price"] = frame.apply(_select_wall_price, axis=1, side="ask")
    frame["wall_mid"] = (frame["wall_bid_price"] + frame["wall_ask_price"]) / 2
    frame["microprice_minus_mid"] = frame["microprice"] - frame["book_mid"]
    frame["wall_mid_minus_mid"] = frame["wall_mid"] - frame["book_mid"]

    group_keys = _group_columns(frame)
    grouped = frame.groupby(group_keys, sort=False)

    for window in rolling_windows:
        frame[f"book_mid_rolling_mean_{window}"] = grouped["book_mid"].transform(
            lambda series: series.rolling(window, min_periods=max(3, window // 4)).mean()
        )
        frame[f"wall_mid_rolling_mean_{window}"] = grouped["wall_mid"].transform(
            lambda series: series.rolling(window, min_periods=max(3, window // 4)).mean()
        )

    for window in ema_windows:
        frame[f"book_mid_ema_{window}"] = grouped["book_mid"].transform(
            lambda series: series.ewm(span=window, adjust=False, min_periods=max(3, window // 4)).mean()
        )
        frame[f"wall_mid_ema_{window}"] = grouped["wall_mid"].transform(
            lambda series: series.ewm(span=window, adjust=False, min_periods=max(3, window // 4)).mean()
        )

    return frame


def make_future_targets(frame: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 10, 20)) -> pd.DataFrame:
    result = frame.copy()
    grouped = result.groupby(_group_columns(result), sort=False)
    for horizon in horizons:
        result[f"future_mid_{horizon}"] = grouped["book_mid"].shift(-horizon)
        result[f"future_return_{horizon}"] = result[f"future_mid_{horizon}"] - result["book_mid"]
        result[f"future_bid_price_1_{horizon}"] = grouped["bid_price_1"].shift(-horizon)
        result[f"future_ask_price_1_{horizon}"] = grouped["ask_price_1"].shift(-horizon)
    return result


def align_trades_to_quotes(quotes: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    quote_reference = quotes.copy().sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    quote_reference["quote_index"] = quote_reference.index

    trade_reference = trades.rename(columns={"symbol": "product"}).copy()
    trade_reference = trade_reference.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)

    aligned_groups: list[pd.DataFrame] = []
    quote_columns = ["timestamp", "quote_index"] + [column for column in quote_reference.columns if column not in {"product", "day", "timestamp"}]
    for (product_name, day), trade_group in trade_reference.groupby(["product", "day"], sort=False):
        quote_group = quote_reference[
            (quote_reference["product"] == product_name) & (quote_reference["day"] == day)
        ][quote_columns]
        if quote_group.empty:
            continue
        aligned = pd.merge_asof(
            trade_group.sort_values("timestamp"),
            quote_group.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        aligned["product"] = product_name
        aligned["day"] = day
        aligned_groups.append(aligned)

    if not aligned_groups:
        return pd.DataFrame()

    aligned_trades = pd.concat(aligned_groups, ignore_index=True)
    aligned_trades["dist_to_bid"] = (aligned_trades["price"] - aligned_trades["bid_price_1"]).abs()
    aligned_trades["dist_to_ask"] = (aligned_trades["price"] - aligned_trades["ask_price_1"]).abs()
    aligned_trades["inferred_side"] = "mid_or_unknown"
    aligned_trades.loc[aligned_trades["dist_to_ask"] < aligned_trades["dist_to_bid"], "inferred_side"] = "buy"
    aligned_trades.loc[aligned_trades["dist_to_bid"] < aligned_trades["dist_to_ask"], "inferred_side"] = "sell"
    aligned_trades["impact_sign"] = np.where(
        aligned_trades["inferred_side"] == "buy",
        1,
        np.where(aligned_trades["inferred_side"] == "sell", -1, 0),
    )
    return aligned_trades.sort_values(["product", "day", "timestamp", "price"]).reset_index(drop=True)


def validate_mean_reversion(
    frame: pd.DataFrame,
    fair_columns: list[str] | tuple[str, ...],
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    price_column: str = "book_mid",
    bucket_count: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    bucketed_frames: list[pd.DataFrame] = []

    for fair_column, horizon in product(fair_columns, horizons):
        future_column = f"future_return_{horizon}"
        subset = frame[["product", price_column, fair_column, future_column]].dropna().copy()
        if subset.empty:
            continue
        subset["edge"] = subset[price_column] - subset[fair_column]
        subset["bucket"] = _bucketize(subset["edge"], bucket_count=bucket_count)

        summary = (
            subset.groupby("product", as_index=False)
            .agg(
                correlation=("edge", lambda series: series.corr(subset.loc[series.index, future_column])),
                observations=("edge", "size"),
                edge_std=("edge", "std"),
                future_return_std=(future_column, "std"),
            )
            .assign(fair_column=fair_column, horizon=horizon)
        )
        summaries.append(summary)

        bucketed = (
            subset.groupby(["product", "bucket"], as_index=False)
            .agg(
                avg_edge=("edge", "mean"),
                avg_future_return=(future_column, "mean"),
                observations=("edge", "size"),
            )
            .assign(fair_column=fair_column, horizon=horizon)
        )
        bucketed["monotonicity"] = bucketed.groupby("product")["avg_edge"].transform(
            lambda series: series.corr(bucketed.loc[series.index, "avg_future_return"])
        )
        bucketed_frames.append(bucketed)

    summary_frame = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    bucketed_frame = pd.concat(bucketed_frames, ignore_index=True) if bucketed_frames else pd.DataFrame()
    return summary_frame, bucketed_frame


def validate_signal_monotonicity(
    frame: pd.DataFrame,
    signal_columns: list[str] | tuple[str, ...],
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    bucket_count: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    bucketed_frames: list[pd.DataFrame] = []

    for signal_column, horizon in product(signal_columns, horizons):
        future_column = f"future_return_{horizon}"
        subset = frame[["product", signal_column, future_column]].dropna().copy()
        if subset.empty:
            continue
        subset["bucket"] = _bucketize(subset[signal_column], bucket_count=bucket_count)

        summary = (
            subset.groupby("product", as_index=False)
            .agg(
                correlation=(signal_column, lambda series: series.corr(subset.loc[series.index, future_column])),
                observations=(signal_column, "size"),
                signal_std=(signal_column, "std"),
                future_return_std=(future_column, "std"),
            )
            .assign(signal_column=signal_column, horizon=horizon)
        )
        summaries.append(summary)

        bucketed = (
            subset.groupby(["product", "bucket"], as_index=False)
            .agg(
                avg_signal=(signal_column, "mean"),
                avg_future_return=(future_column, "mean"),
                observations=(signal_column, "size"),
            )
            .assign(signal_column=signal_column, horizon=horizon)
        )
        bucketed["monotonicity"] = bucketed.groupby("product")["avg_signal"].transform(
            lambda series: series.corr(bucketed.loc[series.index, "avg_future_return"])
        )
        bucketed_frames.append(bucketed)

    summary_frame = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    bucketed_frame = pd.concat(bucketed_frames, ignore_index=True) if bucketed_frames else pd.DataFrame()
    return summary_frame, bucketed_frame


def evaluate_trade_impact(
    aligned_trades: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []
    for horizon in horizons:
        future_mid_column = f"future_mid_{horizon}"
        if future_mid_column not in aligned_trades.columns or "book_mid" not in aligned_trades.columns:
            continue
        subset = aligned_trades[["product", "inferred_side", "book_mid", future_mid_column, "quantity"]].dropna().copy()
        if subset.empty:
            continue
        subset["impact"] = subset[future_mid_column] - subset["book_mid"]
        summary = (
            subset.groupby(["product", "inferred_side"], as_index=False)
            .agg(
                trades=("impact", "size"),
                mean_impact=("impact", "mean"),
                median_impact=("impact", "median"),
                mean_quantity=("quantity", "mean"),
            )
            .assign(horizon=horizon)
        )
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def _forward_window_stat(values: np.ndarray, lookahead: int, mode: str) -> np.ndarray:
    result = np.full(len(values), np.nan)
    for index in range(len(values)):
        future_slice = values[index + 1 : index + 1 + lookahead]
        if len(future_slice) == 0:
            continue
        valid = future_slice[~np.isnan(future_slice)]
        if len(valid) == 0:
            continue
        if mode == "min":
            result[index] = valid.min()
        elif mode == "max":
            result[index] = valid.max()
        else:
            raise ValueError(f"Unsupported mode {mode!r}")
    return result


def _build_execution_context(quotes: pd.DataFrame, aligned_trades: pd.DataFrame, lookahead: int = 5) -> pd.DataFrame:
    context = quotes.copy().sort_values(["product", "day", "timestamp"]).reset_index(drop=True)

    if aligned_trades.empty:
        context["buy_trade_price_at_quote"] = np.nan
        context["sell_trade_price_at_quote"] = np.nan
    else:
        trade_reference = aligned_trades.copy()
        trade_agg = (
            trade_reference.groupby(["product", "day", "timestamp", "inferred_side"], as_index=False)
            .agg(
                buy_trade_price=("price", "max"),
                sell_trade_price=("price", "min"),
            )
        )
        buy_side = (
            trade_agg[trade_agg["inferred_side"] == "buy"][["product", "day", "timestamp", "buy_trade_price"]]
            .rename(columns={"buy_trade_price": "buy_trade_price_at_quote"})
        )
        sell_side = (
            trade_agg[trade_agg["inferred_side"] == "sell"][["product", "day", "timestamp", "sell_trade_price"]]
            .rename(columns={"sell_trade_price": "sell_trade_price_at_quote"})
        )
        context = context.merge(buy_side, on=["product", "day", "timestamp"], how="left")
        context = context.merge(sell_side, on=["product", "day", "timestamp"], how="left")

    frames: list[pd.DataFrame] = []
    for _, group in context.groupby(["product", "day"], sort=False):
        working = group.copy()
        working[f"future_min_ask_price_1_{lookahead}"] = _forward_window_stat(
            working["ask_price_1"].to_numpy(dtype=float), lookahead=lookahead, mode="min"
        )
        working[f"future_max_bid_price_1_{lookahead}"] = _forward_window_stat(
            working["bid_price_1"].to_numpy(dtype=float), lookahead=lookahead, mode="max"
        )
        working[f"future_min_sell_trade_price_{lookahead}"] = _forward_window_stat(
            working["sell_trade_price_at_quote"].to_numpy(dtype=float), lookahead=lookahead, mode="min"
        )
        working[f"future_max_buy_trade_price_{lookahead}"] = _forward_window_stat(
            working["buy_trade_price_at_quote"].to_numpy(dtype=float), lookahead=lookahead, mode="max"
        )
        frames.append(working)
    return pd.concat(frames, ignore_index=True) if frames else context


def prepare_execution_context(quotes: pd.DataFrame, aligned_trades: pd.DataFrame, lookahead: int = 5) -> pd.DataFrame:
    return _build_execution_context(quotes, aligned_trades, lookahead=lookahead)


def estimate_fill_probabilities(
    quotes: pd.DataFrame,
    aligned_trades: pd.DataFrame,
    lookahead: int = 5,
) -> pd.DataFrame:
    context = _build_execution_context(quotes, aligned_trades, lookahead=lookahead)
    context["passive_buy_price"] = context["bid_price_1"]
    context["passive_sell_price"] = context["ask_price_1"]
    context["improved_buy_price"] = np.where(context["spread"] >= 2, context["bid_price_1"] + 1, context["bid_price_1"])
    context["improved_sell_price"] = np.where(context["spread"] >= 2, context["ask_price_1"] - 1, context["ask_price_1"])

    context["passive_buy_fill"] = (
        (context[f"future_min_ask_price_1_{lookahead}"] <= context["passive_buy_price"])
        | (context[f"future_min_sell_trade_price_{lookahead}"] <= context["passive_buy_price"])
    )
    context["passive_sell_fill"] = (
        (context[f"future_max_bid_price_1_{lookahead}"] >= context["passive_sell_price"])
        | (context[f"future_max_buy_trade_price_{lookahead}"] >= context["passive_sell_price"])
    )
    context["improved_buy_fill"] = (
        (context[f"future_min_ask_price_1_{lookahead}"] <= context["improved_buy_price"])
        | (context[f"future_min_sell_trade_price_{lookahead}"] <= context["improved_buy_price"])
    )
    context["improved_sell_fill"] = (
        (context[f"future_max_bid_price_1_{lookahead}"] >= context["improved_sell_price"])
        | (context[f"future_max_buy_trade_price_{lookahead}"] >= context["improved_sell_price"])
    )

    rows = [
        ("aggressive", "buy", np.ones(len(context), dtype=float)),
        ("aggressive", "sell", np.ones(len(context), dtype=float)),
        ("passive", "buy", context["passive_buy_fill"].astype(float).to_numpy()),
        ("passive", "sell", context["passive_sell_fill"].astype(float).to_numpy()),
        ("improved", "buy", context["improved_buy_fill"].astype(float).to_numpy()),
        ("improved", "sell", context["improved_sell_fill"].astype(float).to_numpy()),
    ]

    probability_frames: list[pd.DataFrame] = []
    for execution_style, side, values in rows:
        working = context[["product", "spread"]].copy()
        working["execution_style"] = execution_style
        working["side"] = side
        working["fill_indicator"] = values
        probability_frames.append(working)

    fill_frame = pd.concat(probability_frames, ignore_index=True)
    return (
        fill_frame.groupby(["product", "execution_style", "side"], as_index=False)
        .agg(fill_probability=("fill_indicator", "mean"), samples=("fill_indicator", "size"), mean_spread=("spread", "mean"))
        .sort_values(["product", "execution_style", "side"])
        .reset_index(drop=True)
    )


def estimate_quote_ev(
    frame: pd.DataFrame,
    fill_probabilities: pd.DataFrame,
    edge_column: str,
    bucket_count: int = 6,
) -> pd.DataFrame:
    subset = frame[["product", edge_column]].dropna().copy()
    subset["abs_edge"] = subset[edge_column].abs()
    subset = subset[subset["abs_edge"] > 0]
    if subset.empty:
        return pd.DataFrame()

    subset["edge_bucket"] = subset.groupby("product")["abs_edge"].transform(
        lambda series: _bucketize(series, bucket_count=bucket_count)
    )
    edge_summary = (
        subset.groupby(["product", "edge_bucket"], as_index=False)
        .agg(mean_abs_edge=("abs_edge", "mean"), observations=("abs_edge", "size"))
    )

    fill_summary = (
        fill_probabilities.groupby(["product", "execution_style"], as_index=False)
        .agg(fill_probability=("fill_probability", "mean"))
    )
    expected_value = edge_summary.merge(fill_summary, on="product", how="left")
    expected_value["ev"] = expected_value["mean_abs_edge"] * expected_value["fill_probability"]
    return expected_value.sort_values(["product", "execution_style", "edge_bucket"]).reset_index(drop=True)


@dataclass
class StrategyAction:
    phase: str
    side: str
    quantity: int
    price: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseStrategy:
    name: str
    config: dict[str, Any]

    def _inventory_skew(self, position: int, soft_limit: int, skew_strength: float) -> float:
        if soft_limit <= 0:
            return 0.0
        return skew_strength * (position / soft_limit)

    def _capacity(self, position: int, hard_limit: int, side: str) -> int:
        if side == "buy":
            return max(hard_limit - position, 0)
        return max(hard_limit + position, 0)

    def generate_actions(self, row: pd.Series, state: dict[str, Any]) -> list[StrategyAction]:
        raise NotImplementedError


@dataclass
class FixedFairMarketMaker(BaseStrategy):
    def generate_actions(self, row: pd.Series, state: dict[str, Any]) -> list[StrategyAction]:
        fair = float(self.config["fair_value"])
        take_width = float(self.config["take_width"])
        clear_width = float(self.config["clear_width"])
        make_width = float(self.config["make_width"])
        order_size = int(self.config["order_size"])
        hard_limit = int(self.config["hard_position_limit"])
        soft_limit = int(self.config["soft_position_limit"])
        skew = self._inventory_skew(state["position"], soft_limit, float(self.config["inventory_skew"]))

        actions: list[StrategyAction] = []
        buy_capacity = min(order_size, self._capacity(state["position"], hard_limit, "buy"))
        sell_capacity = min(order_size, self._capacity(state["position"], hard_limit, "sell"))

        if pd.notna(row["ask_price_1"]) and row["ask_price_1"] <= fair - take_width and buy_capacity > 0:
            actions.append(StrategyAction("take", "buy", buy_capacity, float(row["ask_price_1"])))
        if pd.notna(row["bid_price_1"]) and row["bid_price_1"] >= fair + take_width and sell_capacity > 0:
            actions.append(StrategyAction("take", "sell", sell_capacity, float(row["bid_price_1"])))

        if state["position"] > 0 and pd.notna(row["bid_price_1"]) and row["bid_price_1"] >= fair - clear_width:
            actions.append(StrategyAction("clear", "sell", min(state["position"], order_size), float(row["bid_price_1"])))
        if state["position"] < 0 and pd.notna(row["ask_price_1"]) and row["ask_price_1"] <= fair + clear_width:
            actions.append(StrategyAction("clear", "buy", min(abs(state["position"]), order_size), float(row["ask_price_1"])))

        quote_bid = int(np.floor(fair - make_width - skew))
        quote_ask = int(np.ceil(fair + make_width - skew))
        if quote_bid < quote_ask and buy_capacity > 0:
            actions.append(StrategyAction("make", "buy", buy_capacity, float(quote_bid)))
        if quote_ask > quote_bid and sell_capacity > 0:
            actions.append(StrategyAction("make", "sell", sell_capacity, float(quote_ask)))
        return actions


@dataclass
class DynamicFairMeanReverter(BaseStrategy):
    def generate_actions(self, row: pd.Series, state: dict[str, Any]) -> list[StrategyAction]:
        fair_value = row[self.config["fair_source"]]
        if pd.isna(fair_value):
            return []
        fair = float(fair_value)
        spread_scale = max(float(row["spread"]), 1.0)
        entry_threshold = float(self.config["entry_threshold"]) * spread_scale
        exit_threshold = float(self.config["exit_threshold"]) * spread_scale
        order_size = int(self.config["order_size"])
        hard_limit = int(self.config["hard_position_limit"])
        soft_limit = int(self.config["soft_position_limit"])
        skew = self._inventory_skew(state["position"], soft_limit, float(self.config["inventory_skew"]))
        deviation = float(row["book_mid"] - fair)

        actions: list[StrategyAction] = []
        buy_capacity = min(order_size, self._capacity(state["position"], hard_limit, "buy"))
        sell_capacity = min(order_size, self._capacity(state["position"], hard_limit, "sell"))

        if deviation <= -entry_threshold and buy_capacity > 0:
            actions.append(StrategyAction("take", "buy", buy_capacity, float(row["ask_price_1"])))
        elif deviation >= entry_threshold and sell_capacity > 0:
            actions.append(StrategyAction("take", "sell", sell_capacity, float(row["bid_price_1"])))
        else:
            if state["position"] > 0 and deviation >= -exit_threshold:
                actions.append(StrategyAction("clear", "sell", min(state["position"], order_size), float(row["bid_price_1"])))
            if state["position"] < 0 and deviation <= exit_threshold:
                actions.append(StrategyAction("clear", "buy", min(abs(state["position"]), order_size), float(row["ask_price_1"])))

        make_width = max(1.0, float(self.config.get("make_width", 1.0)) * spread_scale / 2)
        quote_bid = int(np.floor(fair - make_width - skew))
        quote_ask = int(np.ceil(fair + make_width - skew))
        if abs(deviation) < entry_threshold and buy_capacity > 0:
            actions.append(StrategyAction("make", "buy", buy_capacity, float(quote_bid)))
        if abs(deviation) < entry_threshold and sell_capacity > 0:
            actions.append(StrategyAction("make", "sell", sell_capacity, float(quote_ask)))
        return actions


@dataclass
class ImbalanceMicropriceStrategy(BaseStrategy):
    def generate_actions(self, row: pd.Series, state: dict[str, Any]) -> list[StrategyAction]:
        signal_value = row[self.config["signal_source"]]
        if pd.isna(signal_value):
            return []
        signal = float(signal_value)
        threshold = float(self.config["signal_threshold"])
        order_size = int(self.config["order_size"])
        hard_limit = int(self.config["hard_position_limit"])
        actions: list[StrategyAction] = []

        buy_capacity = min(order_size, self._capacity(state["position"], hard_limit, "buy"))
        sell_capacity = min(order_size, self._capacity(state["position"], hard_limit, "sell"))

        if signal >= threshold and buy_capacity > 0:
            actions.append(StrategyAction("take", "buy", buy_capacity, float(row["ask_price_1"])))
        elif signal <= -threshold and sell_capacity > 0:
            actions.append(StrategyAction("take", "sell", sell_capacity, float(row["bid_price_1"])))
        else:
            if state["position"] > 0 and signal <= threshold / 2:
                actions.append(StrategyAction("clear", "sell", min(state["position"], order_size), float(row["bid_price_1"])))
            if state["position"] < 0 and signal >= -threshold / 2:
                actions.append(StrategyAction("clear", "buy", min(abs(state["position"]), order_size), float(row["ask_price_1"])))
        return actions


def _resolve_fill(
    row: pd.Series,
    action: StrategyAction,
    execution_style: str,
    fill_horizon: int,
) -> tuple[bool, float]:
    if execution_style == "aggressive":
        if action.phase == "make":
            return False, float(action.price)
        fill_price = float(row["ask_price_1"] if action.side == "buy" else row["bid_price_1"])
        return True, fill_price

    if execution_style == "passive":
        target_price = float(row["bid_price_1"] if action.side == "buy" else row["ask_price_1"])
    elif execution_style == "improved":
        if action.side == "buy":
            target_price = float(row["bid_price_1"] + 1 if row["spread"] >= 2 else row["bid_price_1"])
        else:
            target_price = float(row["ask_price_1"] - 1 if row["spread"] >= 2 else row["ask_price_1"])
    else:
        raise ValueError(f"Unsupported execution style {execution_style!r}")

    if action.side == "buy":
        future_min_ask = row.get(f"future_min_ask_price_1_{fill_horizon}")
        future_min_sell = row.get(f"future_min_sell_trade_price_{fill_horizon}")
        filled = (pd.notna(future_min_ask) and future_min_ask <= target_price) or (
            pd.notna(future_min_sell) and future_min_sell <= target_price
        )
    else:
        future_max_bid = row.get(f"future_max_bid_price_1_{fill_horizon}")
        future_max_buy = row.get(f"future_max_buy_trade_price_{fill_horizon}")
        filled = (pd.notna(future_max_bid) and future_max_bid >= target_price) or (
            pd.notna(future_max_buy) and future_max_buy >= target_price
        )

    return bool(filled), target_price


def run_backtest(
    quotes: pd.DataFrame,
    aligned_trades: pd.DataFrame,
    strategy: BaseStrategy,
    execution_style: str,
    hard_position_limit: int = 20,
    fill_horizon: int = 5,
    initial_cash: float = 0.0,
) -> dict[str, Any]:
    required_context_columns = [
        f"future_min_ask_price_1_{fill_horizon}",
        f"future_max_bid_price_1_{fill_horizon}",
        f"future_min_sell_trade_price_{fill_horizon}",
        f"future_max_buy_trade_price_{fill_horizon}",
    ]
    if all(column in quotes.columns for column in required_context_columns):
        context = quotes.copy()
    else:
        context = _build_execution_context(quotes, aligned_trades, lookahead=fill_horizon)
    state = {"position": 0, "cash": float(initial_cash), "hard_position_limit": hard_position_limit}

    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for _, row in context.iterrows():
        actions = strategy.generate_actions(row, state)
        for action in actions:
            if action.quantity <= 0:
                continue
            if action.side == "buy":
                executable_quantity = min(action.quantity, max(hard_position_limit - state["position"], 0))
            else:
                executable_quantity = min(action.quantity, max(hard_position_limit + state["position"], 0))
            if executable_quantity <= 0:
                continue

            filled, fill_price = _resolve_fill(row, action, execution_style=execution_style, fill_horizon=fill_horizon)
            if not filled:
                continue

            signed_quantity = executable_quantity if action.side == "buy" else -executable_quantity
            state["position"] += signed_quantity
            state["cash"] -= signed_quantity * fill_price

            trade_rows.append(
                {
                    "product": row["product"],
                    "day": row["day"],
                    "timestamp": row["timestamp"],
                    "phase": action.phase,
                    "side": action.side,
                    "execution_style": execution_style,
                    "quantity": executable_quantity,
                    "fill_price": fill_price,
                    "book_mid": row["book_mid"],
                    "position_after": state["position"],
                    "strategy": strategy.name,
                }
            )

        marked_pnl = state["cash"] + state["position"] * float(row["book_mid"])
        equity_rows.append(
            {
                "product": row["product"],
                "day": row["day"],
                "timestamp": row["timestamp"],
                "book_mid": float(row["book_mid"]),
                "cash": float(state["cash"]),
                "position": int(state["position"]),
                "marked_pnl": float(marked_pnl),
                "strategy": strategy.name,
                "execution_style": execution_style,
            }
        )

    equity_curve = pd.DataFrame(equity_rows)
    trades_frame = pd.DataFrame(trade_rows)

    if equity_curve.empty:
        summary = {
            "strategy": strategy.name,
            "execution_style": execution_style,
            "total_pnl": 0.0,
            "simple_sharpe": np.nan,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "avg_inventory_usage": 0.0,
            "max_inventory_usage": 0.0,
            "fraction_near_limit": 0.0,
        }
        return {"summary": summary, "equity_curve": equity_curve, "trades": trades_frame}

    pnl_changes = equity_curve["marked_pnl"].diff().fillna(0.0)
    pnl_std = float(pnl_changes.std(ddof=0))
    running_max = equity_curve["marked_pnl"].cummax()
    drawdown = running_max - equity_curve["marked_pnl"]
    inventory_usage = equity_curve["position"].abs() / hard_position_limit if hard_position_limit else 0.0

    summary = {
        "strategy": strategy.name,
        "execution_style": execution_style,
        "total_pnl": float(equity_curve["marked_pnl"].iloc[-1]),
        "simple_sharpe": float(pnl_changes.mean() / pnl_std) if pnl_std > 0 else np.nan,
        "max_drawdown": float(drawdown.max()),
        "trade_count": int(len(trades_frame)),
        "avg_inventory_usage": float(inventory_usage.mean()),
        "max_inventory_usage": float(inventory_usage.max()),
        "fraction_near_limit": float((inventory_usage >= 0.8).mean()),
    }
    return {"summary": summary, "equity_curve": equity_curve, "trades": trades_frame}


__all__ = [
    "DynamicFairMeanReverter",
    "FixedFairMarketMaker",
    "ImbalanceMicropriceStrategy",
    "align_trades_to_quotes",
    "compute_features",
    "estimate_fill_probabilities",
    "estimate_quote_ev",
    "evaluate_trade_impact",
    "make_future_targets",
    "prepare_execution_context",
    "prepare_quotes",
    "run_backtest",
    "validate_mean_reversion",
    "validate_signal_monotonicity",
]
