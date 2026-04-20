from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots

from manual_trading.round2_manual_trading_simulator import (
    MAX_SPEED_INVESTMENT,
    MIN_SPEED_INVESTMENT,
    TOTAL_BUDGET,
)


SCALE_UNIT = 7 / 100
DEFAULT_SPEED_FIELD = (
    (5_000, 20, 3),
    (5_000, 40, 3),
    (5_000, 60, 3),
)


@dataclass(frozen=True)
class SpeedFieldGroup:
    player_count: int
    mean_pct: float
    variance: float

    @property
    def label(self) -> str:
        return f"N({self.mean_pct:g}%, var {self.variance:g})"


@dataclass(frozen=True)
class UniformSpeedFieldGroup:
    player_count: int
    low_pct: float
    high_pct: float

    @property
    def label(self) -> str:
        return f"U({self.low_pct:g}%, {self.high_pct:g}%)"


def _as_groups(groups: Iterable[tuple[int, float, float]]) -> list[SpeedFieldGroup]:
    return [SpeedFieldGroup(*group) for group in groups]


def _research_value(research_pct: int) -> float:
    if research_pct <= 0:
        return 0.0
    return 200_000 * np.log1p(research_pct) / np.log(101)


def _rank_to_multiplier(rank: int, total_players: int) -> float:
    if total_players <= 1:
        return 0.9
    return 0.9 - (0.8 * (rank - 1) / (total_players - 1))


def _compute_best_scenario(speed_pct: int, speed_multiplier: float) -> dict[str, float | int]:
    cap = 100 - speed_pct
    best: dict[str, float | int] = {
        "pnl": float("-inf"),
        "gross": 0.0,
        "research_pct": 0,
        "scale_pct": 0,
        "used_budget": TOTAL_BUDGET * speed_pct / 100,
    }

    for research_pct in range(cap + 1):
        research_value = _research_value(research_pct)
        max_scale_pct = cap - research_pct
        gross_per_scale_pct = research_value * SCALE_UNIT * speed_multiplier
        scale_pct = max_scale_pct if gross_per_scale_pct > TOTAL_BUDGET / 100 else 0
        gross = research_value * (scale_pct * SCALE_UNIT) * speed_multiplier
        used_budget = TOTAL_BUDGET * (research_pct + scale_pct + speed_pct) / 100
        pnl = gross - used_budget

        if pnl > best["pnl"]:
            best = {
                "pnl": pnl,
                "gross": gross,
                "research_pct": research_pct,
                "scale_pct": scale_pct,
                "used_budget": used_budget,
            }

    return best


def sample_speed_field(
    groups: Iterable[tuple[int, float, float]] = DEFAULT_SPEED_FIELD,
    uniform_groups: Iterable[tuple[int, float, float]] = ((5_000, 23, 63),),
    seed: int = 42,
    min_speed: int = MIN_SPEED_INVESTMENT,
    max_speed: int = MAX_SPEED_INVESTMENT,
) -> pd.DataFrame:
    """Build the requested sampled speed-investment field.

    Gaussian variance is interpreted literally, so each normal distribution
    uses ``sqrt(variance)`` as its standard deviation. Samples are rounded to
    integer percentage investments because the manual simulator works in
    integer percentage points.
    """
    rng = np.random.default_rng(seed)
    records = []
    player_id = 1

    for group in _as_groups(groups):
        raw_values = rng.normal(group.mean_pct, np.sqrt(group.variance), group.player_count)
        rounded_values = np.rint(raw_values).astype(int)
        speed_values = np.clip(rounded_values, min_speed, max_speed)

        for raw_speed, speed_pct in zip(raw_values, speed_values, strict=True):
            records.append(
                {
                    "player_id": player_id,
                    "cohort": group.label,
                    "raw_speed_investment": raw_speed,
                    "speed_investment": int(speed_pct),
                }
            )
            player_id += 1

    for group in (UniformSpeedFieldGroup(*group) for group in uniform_groups):
        raw_values = rng.uniform(group.low_pct, group.high_pct, group.player_count)
        rounded_values = np.rint(raw_values).astype(int)
        speed_values = np.clip(rounded_values, min_speed, max_speed)

        for raw_speed, speed_pct in zip(raw_values, speed_values, strict=True):
            records.append(
                {
                    "player_id": player_id,
                    "cohort": group.label,
                    "raw_speed_investment": raw_speed,
                    "speed_investment": int(speed_pct),
                }
            )
            player_id += 1

    return pd.DataFrame.from_records(records)


def build_speed_pnl_curve(
    players: pd.DataFrame,
    min_speed: int = MIN_SPEED_INVESTMENT,
    max_speed: int = MAX_SPEED_INVESTMENT,
) -> pd.DataFrame:
    total_players = len(players)
    speed_values = players["speed_investment"].to_numpy()
    records = []

    for speed_pct in range(min_speed, max_speed + 1):
        faster_players = int(np.count_nonzero(speed_values > speed_pct))
        rank = 1 + faster_players
        speed_multiplier = _rank_to_multiplier(rank, total_players)
        best = _compute_best_scenario(speed_pct, speed_multiplier)
        records.append(
            {
                "speed_investment": speed_pct,
                "rank": rank,
                "speed_multiplier": speed_multiplier,
                **best,
            }
        )

    return pd.DataFrame.from_records(records)


def attach_player_pnl(players: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    outcome_columns = [
        "speed_investment",
        "rank",
        "speed_multiplier",
        "pnl",
        "gross",
        "research_pct",
        "scale_pct",
        "used_budget",
    ]
    return players.merge(curve[outcome_columns], on="speed_investment", how="left")


def build_speed_pnl_figure(
    players: pd.DataFrame,
    curve: pd.DataFrame,
    seed: int = 42,
) -> go.Figure:
    plot_players = players.copy()
    rng = np.random.default_rng(seed + 1)
    plot_players["plot_speed_investment"] = (
        plot_players["speed_investment"] + rng.uniform(-0.18, 0.18, len(plot_players))
    )

    counts = (
        plot_players.groupby("speed_investment", as_index=False)
        .size()
        .rename(columns={"size": "player_count"})
    )
    best = curve.loc[curve["pnl"].idxmax()]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=("Player PnL by speed investment", "Sampled player distribution"),
    )

    colors = ["#2450d3", "#e76f51", "#2a9d8f", "#8a4fff"]
    for color, (cohort, cohort_players) in zip(colors, plot_players.groupby("cohort"), strict=False):
        fig.add_trace(
            go.Scattergl(
                x=cohort_players["plot_speed_investment"],
                y=cohort_players["pnl"],
                mode="markers",
                name=cohort,
                marker={"size": 5, "opacity": 0.28, "color": color},
                customdata=np.stack(
                    [
                        cohort_players["speed_investment"],
                        cohort_players["rank"],
                        cohort_players["speed_multiplier"],
                        cohort_players["research_pct"],
                        cohort_players["scale_pct"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Speed %{customdata[0]}%<br>"
                    "PnL %{y:,.0f}<br>"
                    "Rank %{customdata[1]:,.0f}<br>"
                    "Multiplier %{customdata[2]:.3f}<br>"
                    "Best research %{customdata[3]}%<br>"
                    "Best scale %{customdata[4]}%"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=curve["speed_investment"],
            y=curve["pnl"],
            mode="lines",
            name="PnL curve",
            line={"color": "#111827", "width": 3},
            hovertemplate="Speed %{x}%<br>PnL %{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[best["speed_investment"]],
            y=[best["pnl"]],
            mode="markers+text",
            name="Best speed",
            text=[f"Best {best['speed_investment']:.0f}%"],
            textposition="top center",
            marker={"size": 12, "color": "#ffb000", "line": {"color": "#111827", "width": 1}},
            hovertemplate="Best speed %{x}%<br>PnL %{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=counts["speed_investment"],
            y=counts["player_count"],
            name="Players",
            marker={"color": "#7c8da6"},
            hovertemplate="Speed %{x}%<br>Players %{y:,}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        title=(
            "Round 2 PnL vs speed investment for 20,000 sampled-field players"
            "<br><sup>5,000 each from N(25%, var 3), N(37%, var 3), N(60%, var 3), and U(23%, 63%)</sup>"
        ),
        height=780,
        margin={"l": 80, "r": 30, "t": 95, "b": 70},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(title_text="Speed investment (% of total budget)", row=2, col=1)
    fig.update_yaxes(title_text="Best net PnL", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="Players", tickformat=",", row=2, col=1)

    return fig


def display_clustered_speed_investment_pnl_graph(
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, go.Figure]:
    players = sample_speed_field(seed=seed)
    curve = build_speed_pnl_curve(players)
    players_with_pnl = attach_player_pnl(players, curve)
    fig = build_speed_pnl_figure(players_with_pnl, curve, seed=seed)
    display(fig)
    return players_with_pnl, curve, fig
