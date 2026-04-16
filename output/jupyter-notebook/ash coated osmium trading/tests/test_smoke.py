from __future__ import annotations

from pathlib import Path

from backtest import (
    ExecutionProfile,
    MMParams,
    build_feature_frame,
    build_report,
    load_round1_product,
    run_backtest,
)


def test_round1_ash_smoke(tmp_path: Path) -> None:
    quotes, trades = load_round1_product("data", "ASH_COATED_OSMIUM")
    assert quotes["day"].nunique() == 3

    feature_df = build_feature_frame(quotes, trades)
    params = MMParams(
        strategy_name="SmokeMM",
        anchor_lookback=30,
        base_half_spread=7.0,
        inventory_skew=0.4,
        dislocation_threshold=5.0,
    )
    profiles = [
        ExecutionProfile(
            name="strict",
            queue_ahead_frac=0.50,
            allow_move_through=False,
            taker_penalty_ticks=1.0,
        ),
        ExecutionProfile(
            name="loose",
            queue_ahead_frac=0.25,
            allow_move_through=True,
            taker_penalty_ticks=0.5,
        ),
    ]

    for profile in profiles:
        result = run_backtest(feature_df, params, profile)
        assert not result.summary.empty
        assert not result.daily_pnl.empty
        assert not result.inventory_path.empty

        report_dir = tmp_path / profile.name
        report_paths = build_report(result, report_dir)
        assert report_paths["summary"].exists()
        assert report_paths["daily_pnl"].exists()
        assert (report_paths["plots_dir"] / "daily_pnl.png").exists()
