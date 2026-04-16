from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    inventory_limit: float = 20.0
    passive_order_size: float = 5.0
    taker_order_size: float = 5.0
    queue_ahead_frac: float = 0.50
    allow_move_through: bool = False
    taker_penalty_ticks: float = 1.0
    signal_delay_steps: int = 1
    near_limit_frac: float = 0.75
    force_flat_at_day_end: bool = True


@dataclass(frozen=True)
class MMParams:
    strategy_name: str = "ASH_COATED_OSMIUM_RegimeMM"
    anchor_lookback: int = 30
    base_half_spread: float = 4.0
    inventory_skew: float = 0.40
    imbalance_skew: float = 1.0
    dislocation_threshold: float = 5.0
    defensive_widening_multiplier: float = 2.0
    max_quote_size: float = 5.0
    inventory_soft_limit: float = 12.0
    inventory_hard_limit: float = 20.0
    thin_depth_threshold: float = 8.0
    defensive_size_fraction: float = 0.5
    strong_dislocation_buffer: float = 2.0
    aggressive_size: float = 0.0
    enable_dislocation_takers: bool = False
    dislocation_one_sided_only: bool = True


@dataclass
class BacktestResult:
    profile: ExecutionProfile
    params: MMParams
    summary: pd.DataFrame
    daily_pnl: pd.DataFrame
    fills: pd.DataFrame
    inventory_path: pd.DataFrame
    sanity_checks: pd.DataFrame
