from __future__ import annotations

from backtest import MMParams
from trader import BookSnapshot, build_quote_plan, classify_regime


def _params() -> MMParams:
    return MMParams(
        strategy_name="StrategyUnitTest",
        anchor_lookback=30,
        base_half_spread=4.0,
        inventory_skew=0.4,
        imbalance_skew=1.0,
        dislocation_threshold=5.0,
        defensive_widening_multiplier=2.0,
        max_quote_size=5.0,
        inventory_soft_limit=12.0,
        inventory_hard_limit=20.0,
        thin_depth_threshold=8.0,
        defensive_size_fraction=0.5,
        strong_dislocation_buffer=2.0,
        aggressive_size=0.0,
        enable_dislocation_takers=False,
        dislocation_one_sided_only=True,
    )


def _snapshot(
    *,
    best_bid: int | None = 99,
    best_ask: int | None = 101,
    top_bid_depth: float = 10.0,
    top_ask_depth: float = 10.0,
    total_bid_depth: float = 30.0,
    total_ask_depth: float = 30.0,
    mid_price: float | None = 100.0,
    book_state: str = "both_sides",
) -> BookSnapshot:
    return BookSnapshot(
        best_bid=best_bid,
        best_ask=best_ask,
        top_bid_depth=top_bid_depth,
        top_ask_depth=top_ask_depth,
        total_bid_depth=total_bid_depth,
        total_ask_depth=total_ask_depth,
        mid_price=mid_price,
        book_state=book_state,
    )


def test_regime_classifier_covers_normal_dislocation_and_defensive() -> None:
    params = _params()

    assert classify_regime(_snapshot(), 100.0, params) == "normal"
    assert classify_regime(_snapshot(mid_price=106.0), 100.0, params) == "dislocation"
    assert classify_regime(_snapshot(top_bid_depth=5.0, top_ask_depth=5.0), 100.0, params) == "defensive"
    assert classify_regime(_snapshot(best_ask=None, book_state="bid_only", mid_price=100.0), 100.0, params) == "defensive"


def test_inventory_soft_limit_disables_further_buys() -> None:
    params = _params()
    plan = build_quote_plan(_snapshot(), position=12.0, anchor_price=100.0, params=params)

    assert plan.passive_buy_size == 0.0
    assert plan.passive_sell_size > 0.0


def test_inventory_hard_limit_only_allows_reducing_side() -> None:
    params = _params()
    plan = build_quote_plan(_snapshot(), position=20.0, anchor_price=100.0, params=params)

    assert plan.passive_buy_size == 0.0
    assert plan.passive_sell_size > 0.0


def test_one_sided_book_only_quotes_inventory_reducing_side() -> None:
    params = _params()
    bid_only = _snapshot(best_ask=None, top_ask_depth=0.0, total_ask_depth=0.0, mid_price=100.0, book_state="bid_only")

    short_plan = build_quote_plan(bid_only, position=-5.0, anchor_price=100.0, params=params)
    flat_plan = build_quote_plan(bid_only, position=0.0, anchor_price=100.0, params=params)

    assert short_plan.passive_buy_size > 0.0
    assert short_plan.passive_sell_size == 0.0
    assert flat_plan.passive_buy_size == 0.0
    assert flat_plan.passive_sell_size == 0.0


def test_dislocation_above_anchor_quotes_sell_side_only() -> None:
    params = _params()
    plan = build_quote_plan(_snapshot(mid_price=106.0), position=0.0, anchor_price=100.0, params=params)

    assert plan.regime == "dislocation"
    assert plan.passive_buy_size == 0.0
    assert plan.passive_sell_size > 0.0
    assert plan.planned_bid is None
    assert plan.planned_ask is not None


def test_dislocation_below_anchor_quotes_buy_side_only() -> None:
    params = _params()
    plan = build_quote_plan(_snapshot(mid_price=94.0), position=0.0, anchor_price=100.0, params=params)

    assert plan.regime == "dislocation"
    assert plan.passive_buy_size > 0.0
    assert plan.passive_sell_size == 0.0
    assert plan.planned_bid is not None
    assert plan.planned_ask is None


def test_strong_dislocation_emits_no_taker_when_disabled() -> None:
    params = _params()
    plan = build_quote_plan(_snapshot(mid_price=108.0), position=0.0, anchor_price=100.0, params=params)

    assert plan.regime == "dislocation"
    assert plan.taker_side is None
    assert plan.taker_qty == 0.0
