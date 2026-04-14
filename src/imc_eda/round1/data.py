from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = ROOT / "data" / "round 1"

PRICE_PATTERNS = {
    "csv": "prices_round_*_day_*.csv",
    "parquet": "prices_round_*_day_*.parquet",
}
TRADE_PATTERNS = {
    "csv": "trades_round_*_day_*.csv",
    "parquet": "trades_round_*_day_*.parquet",
}
DAY_PATTERN = re.compile(r"day_(-?\d+)")


def _resolve_files(data_dir: Path | str, pattern: str) -> list[Path]:
    base_dir = Path(data_dir)
    files = sorted(base_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern!r} in {base_dir}")
    return files


def _infer_day(path: Path) -> int:
    match = DAY_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Could not infer day from filename: {path.name}")
    return int(match.group(1))


def _normalize_file_format(file_format: str) -> str:
    normalized = file_format.lower()
    if normalized not in PRICE_PATTERNS:
        supported = ", ".join(sorted(PRICE_PATTERNS))
        raise ValueError(f"Unsupported file format {file_format!r}. Expected one of: {supported}")
    return normalized


def _read_table(path: Path, file_format: str) -> pd.DataFrame:
    if file_format == "csv":
        return pd.read_csv(path, sep=";")
    if file_format == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format {file_format!r}")


def _select_wall_price(row: pd.Series, side: str) -> float:
    prices: list[float] = []
    volumes: list[float] = []

    for level in range(1, 4):
        price_key = f"{side}_price_{level}"
        volume_key = f"{side}_volume_{level}"
        price = row.get(price_key)
        volume = row.get(volume_key)
        if pd.notna(price) and pd.notna(volume):
            prices.append(float(price))
            volumes.append(abs(float(volume)))

    if not prices:
        return np.nan

    wall_index = int(np.argmax(volumes))
    return prices[wall_index]


def load_prices(data_dir: Path | str = DEFAULT_DATA_DIR, file_format: str = "csv") -> pd.DataFrame:
    file_format = _normalize_file_format(file_format)
    frames: list[pd.DataFrame] = []
    for path in _resolve_files(data_dir, PRICE_PATTERNS[file_format]):
        frame = _read_table(path, file_format)
        frame["source_file"] = path.name
        if "day" not in frame.columns:
            frame["day"] = _infer_day(path)
        frames.append(frame)

    prices = pd.concat(frames, ignore_index=True)
    return prices.sort_values(["day", "timestamp", "product"]).reset_index(drop=True)


def load_trades(data_dir: Path | str = DEFAULT_DATA_DIR, file_format: str = "csv") -> pd.DataFrame:
    file_format = _normalize_file_format(file_format)
    frames: list[pd.DataFrame] = []
    for path in _resolve_files(data_dir, TRADE_PATTERNS[file_format]):
        frame = _read_table(path, file_format)
        frame["source_file"] = path.name
        if "day" not in frame.columns:
            frame["day"] = _infer_day(path)
        frames.append(frame)

    trades = pd.concat(frames, ignore_index=True)
    trades["notional"] = trades["price"] * trades["quantity"]
    return trades.sort_values(["day", "timestamp", "symbol", "price"]).reset_index(drop=True)


def build_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame["spread"] = frame["ask_price_1"] - frame["bid_price_1"]
    frame["book_mid"] = (frame["ask_price_1"] + frame["bid_price_1"]) / 2

    bid_volume_cols = [column for column in ("bid_volume_1", "bid_volume_2", "bid_volume_3") if column in frame]
    ask_volume_cols = [column for column in ("ask_volume_1", "ask_volume_2", "ask_volume_3") if column in frame]

    frame["total_bid_volume"] = frame[bid_volume_cols].fillna(0).sum(axis=1)
    frame["total_ask_volume"] = frame[ask_volume_cols].fillna(0).sum(axis=1)

    denominator = (frame["total_bid_volume"] + frame["total_ask_volume"]).replace(0, np.nan)
    frame["book_imbalance"] = (frame["total_bid_volume"] - frame["total_ask_volume"]) / denominator

    frame["wall_bid_price"] = frame.apply(_select_wall_price, axis=1, side="bid")
    frame["wall_ask_price"] = frame.apply(_select_wall_price, axis=1, side="ask")
    frame["wall_mid"] = (frame["wall_bid_price"] + frame["wall_ask_price"]) / 2
    frame["wall_mid_minus_book_mid"] = frame["wall_mid"] - frame["book_mid"]

    return frame
