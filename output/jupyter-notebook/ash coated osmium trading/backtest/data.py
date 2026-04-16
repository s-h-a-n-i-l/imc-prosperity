from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


DAY_PATTERN = re.compile(r"day_(-?\d+)")
FORWARD_HORIZONS = (1, 5, 10)


def infer_day_from_filename(path: Path) -> int:
    match = DAY_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not infer day from filename: {path}")
    return int(match.group(1))


def rolling_median_with_fallback(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window=window, min_periods=1).median()
    expanding = series.expanding(min_periods=1).median()
    return rolling.fillna(expanding)


def _load_price_files(paths: list[Path], product: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(paths):
        frame = pd.read_csv(path, sep=";")
        if "day" not in frame.columns:
            frame["day"] = infer_day_from_filename(path)
        frame = frame.loc[frame["product"] == product].copy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["day", "timestamp"]).reset_index(drop=True)


def _load_trade_files(paths: list[Path], product: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(paths):
        frame = pd.read_csv(path, sep=";")
        frame["day"] = infer_day_from_filename(path)
        frame = frame.loc[frame["symbol"] == product].copy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["day", "timestamp", "price"]).reset_index(drop=True)


def load_round1_product(data_dir: str | Path, product: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir)
    price_paths = list(base.glob("prices_round_1_day_*.csv"))
    trade_paths = list(base.glob("trades_round_1_day_*.csv"))

    if not price_paths:
        raise FileNotFoundError(f"No price files found under {base}")
    if not trade_paths:
        raise FileNotFoundError(f"No trade files found under {base}")

    quotes = _load_price_files(price_paths, product)
    trades = _load_trade_files(trade_paths, product)

    if quotes.empty:
        raise ValueError(f"No quote rows found for product={product}")
    return quotes, trades


def _clean_mid_series(series: pd.Series) -> pd.Series:
    filled = series.ffill()
    expanding = series.expanding(min_periods=1).median()
    return filled.fillna(expanding)


def build_feature_frame(quotes_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    quotes = quotes_df.copy().sort_values(["day", "timestamp"]).reset_index(drop=True)
    bid_price_cols = [f"bid_price_{level}" for level in (1, 2, 3)]
    ask_price_cols = [f"ask_price_{level}" for level in (1, 2, 3)]
    bid_volume_cols = [f"bid_volume_{level}" for level in (1, 2, 3)]
    ask_volume_cols = [f"ask_volume_{level}" for level in (1, 2, 3)]

    quotes["best_bid"] = quotes["bid_price_1"]
    quotes["best_ask"] = quotes["ask_price_1"]
    quotes["top_bid_depth"] = quotes["bid_volume_1"].fillna(0.0)
    quotes["top_ask_depth"] = quotes["ask_volume_1"].fillna(0.0)
    quotes["total_bid_depth_3lvl"] = quotes[bid_volume_cols].fillna(0.0).sum(axis=1)
    quotes["total_ask_depth_3lvl"] = quotes[ask_volume_cols].fillna(0.0).sum(axis=1)

    quotes["both_sides"] = quotes["best_bid"].notna() & quotes["best_ask"].notna()
    quotes["bid_only"] = quotes["best_bid"].notna() & quotes["best_ask"].isna()
    quotes["ask_only"] = quotes["best_bid"].isna() & quotes["best_ask"].notna()
    quotes["book_empty"] = quotes["best_bid"].isna() & quotes["best_ask"].isna()
    quotes["book_state"] = np.select(
        [quotes["both_sides"], quotes["bid_only"], quotes["ask_only"], quotes["book_empty"]],
        ["both_sides", "bid_only", "ask_only", "empty"],
        default="unknown",
    )

    quotes["mid_from_touch"] = np.where(quotes["both_sides"], (quotes["best_bid"] + quotes["best_ask"]) / 2.0, np.nan)
    quotes["clean_mid"] = quotes.groupby("day")["mid_from_touch"].transform(_clean_mid_series)
    quotes["spread"] = quotes["best_ask"] - quotes["best_bid"]
    quotes["top_depth"] = quotes["top_bid_depth"] + quotes["top_ask_depth"]
    quotes["imbalance_top"] = np.where(
        quotes["top_depth"] > 0,
        (quotes["top_bid_depth"] - quotes["top_ask_depth"]) / quotes["top_depth"],
        np.nan,
    )
    total_depth = quotes["total_bid_depth_3lvl"] + quotes["total_ask_depth_3lvl"]
    quotes["imbalance_3lvl"] = np.where(
        total_depth > 0,
        (quotes["total_bid_depth_3lvl"] - quotes["total_ask_depth_3lvl"]) / total_depth,
        np.nan,
    )

    trade_map = (
        trades_df.groupby(["day", "timestamp", "price"], observed=True)["quantity"].sum().to_dict()
        if not trades_df.empty
        else {}
    )

    def trade_qty(day: int, timestamp: int, price: float) -> float:
        if pd.isna(price):
            return 0.0
        return float(trade_map.get((day, timestamp, float(price)), 0.0))

    quotes["trade_qty_at_bid"] = [
        trade_qty(int(row.day), int(row.timestamp), row.best_bid) for row in quotes.itertuples(index=False)
    ]
    quotes["trade_qty_at_ask"] = [
        trade_qty(int(row.day), int(row.timestamp), row.best_ask) for row in quotes.itertuples(index=False)
    ]

    for column in bid_price_cols + ask_price_cols + bid_volume_cols + ask_volume_cols + ["best_bid", "best_ask", "clean_mid"]:
        quotes[f"next_{column}"] = quotes.groupby("day")[column].shift(-1)

    for horizon in FORWARD_HORIZONS:
        quotes[f"fwd_clean_mid_{horizon}"] = quotes.groupby("day")["clean_mid"].shift(-horizon)
        quotes[f"fwd_mid_move_{horizon}"] = quotes[f"fwd_clean_mid_{horizon}"] - quotes["clean_mid"]

    quotes["day_end"] = quotes.groupby("day")["timestamp"].shift(-1).isna()
    return quotes
