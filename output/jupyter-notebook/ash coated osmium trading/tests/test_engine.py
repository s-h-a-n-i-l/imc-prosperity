from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import ExecutionProfile, MMParams, build_feature_frame, run_backtest


PRICE_COLUMNS = [
    "day",
    "timestamp",
    "product",
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
    "mid_price",
    "profit_and_loss",
]


def _quote_row(day: int, timestamp: int, bid: float | None, ask: float | None, bid_vol: float, ask_vol: float) -> dict[str, object]:
    mid_price = np.nan if bid is None or ask is None else (bid + ask) / 2.0
    return {
        "day": day,
        "timestamp": timestamp,
        "product": "ASH_COATED_OSMIUM",
        "bid_price_1": bid,
        "bid_volume_1": bid_vol if bid is not None else np.nan,
        "bid_price_2": np.nan,
        "bid_volume_2": np.nan,
        "bid_price_3": np.nan,
        "bid_volume_3": np.nan,
        "ask_price_1": ask,
        "ask_volume_1": ask_vol if ask is not None else np.nan,
        "ask_price_2": np.nan,
        "ask_volume_2": np.nan,
        "ask_price_3": np.nan,
        "ask_volume_3": np.nan,
        "mid_price": mid_price,
        "profit_and_loss": 0.0,
    }


def _base_params() -> MMParams:
    return MMParams(
        strategy_name="TestMM",
        anchor_lookback=2,
        base_half_spread=1.0,
        inventory_skew=0.4,
        dislocation_threshold=99.0,
        thin_depth_threshold=0.0,
        max_quote_size=5.0,
    )


def _strict_profile() -> ExecutionProfile:
    return ExecutionProfile(
        name="strict",
        queue_ahead_frac=0.50,
        allow_move_through=False,
        taker_penalty_ticks=1.0,
    )


def _loose_profile() -> ExecutionProfile:
    return ExecutionProfile(
        name="loose",
        queue_ahead_frac=0.25,
        allow_move_through=True,
        taker_penalty_ticks=0.5,
    )


def test_engine_sanity_checks_pass_on_synthetic_book() -> None:
    quotes = pd.DataFrame(
        [
            _quote_row(0, 0, 99.0, 101.0, 10.0, 10.0),
            _quote_row(0, 100, 99.0, 101.0, 10.0, 10.0),
            _quote_row(0, 200, 99.0, None, 10.0, 0.0),
            _quote_row(0, 300, 99.0, 101.0, 10.0, 10.0),
        ],
        columns=PRICE_COLUMNS,
    )
    trades = pd.DataFrame(
        [
            {
                "timestamp": 100,
                "buyer": "",
                "seller": "",
                "symbol": "ASH_COATED_OSMIUM",
                "currency": "XIRECS",
                "price": 99.0,
                "quantity": 10,
                "day": 0,
            }
        ]
    )

    feature_df = build_feature_frame(quotes, trades)
    result = run_backtest(feature_df, _base_params(), _strict_profile())

    assert not result.fills.empty
    assert result.sanity_checks["passed"].all()
    assert result.inventory_path["inventory"].abs().max() <= result.profile.inventory_limit + 1e-9
    assert result.inventory_path.groupby("day")["inventory"].last().abs().max() <= 1e-9
    assert np.isclose(result.summary["net_pnl"].iloc[0], result.daily_pnl["net_pnl"].sum())


def test_loose_profile_can_fill_on_move_through_when_strict_cannot() -> None:
    quotes = pd.DataFrame(
        [
            _quote_row(0, 0, 99.0, 101.0, 10.0, 10.0),
            _quote_row(0, 100, 99.0, 101.0, 10.0, 10.0),
            _quote_row(0, 200, 98.0, 100.0, 10.0, 10.0),
            _quote_row(0, 300, 98.0, 100.0, 10.0, 10.0),
        ],
        columns=PRICE_COLUMNS,
    )
    trades = pd.DataFrame(
        columns=["timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity", "day"]
    )

    feature_df = build_feature_frame(quotes, trades)
    strict_result = run_backtest(feature_df, _base_params(), _strict_profile())
    loose_result = run_backtest(feature_df, _base_params(), _loose_profile())

    assert int(strict_result.summary["fill_count"].iloc[0]) < int(loose_result.summary["fill_count"].iloc[0])
    assert loose_result.fills["fill_source"].isin(["move_through", "trade_and_move"]).any()
