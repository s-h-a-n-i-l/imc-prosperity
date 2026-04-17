from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(r"f:\Projects\imc\imc-prosperity")
NOTEBOOK_PATH = ROOT / "src" / "imc_eda" / "round2" / "round-2-eda.ipynb"
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata.update(
    {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
)


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells: list = []

cells.append(
    md(
        """
        # Experiment: Round 2 EDA

        Objective:

        - profile the round 2 quote and trade microstructure for `ASH_COATED_OSMIUM` and `INTARIAN_PEPPER_ROOT`
        - compare round 2 directly against round 1 so we can separate true regime changes from simple continuation
        - test whether Osmium shows evidence of informed quoting before major moves
        - translate those observations into practical commentary about how `src/imc_eda/round1/root/trader_v7.py` should behave on this dataset
        """
    )
)

cells.append(
    code(
        """
        # Setup: imports, paths, plotting defaults, and reusable helpers
        from __future__ import annotations

        import sys
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        import plotly.io as pio
        from IPython.display import Markdown, display

        ROOT = Path.cwd().resolve()
        while ROOT != ROOT.parent and not (ROOT / "pyproject.toml").exists():
            ROOT = ROOT.parent

        SRC = ROOT / "src"
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))
        if str(SRC) not in sys.path:
            sys.path.append(str(SRC))

        from imc_eda.round1 import (
            build_price_features as build_round1_price_features,
            load_prices as load_round1_prices,
            load_trades as load_round1_trades,
        )
        from imc_eda.round2 import (
            build_price_features as build_round2_price_features,
            load_prices as load_round2_prices,
            load_trades as load_round2_trades,
        )
        from imc_eda.round1.root.trader_v7 import (
            OrderDepth,
            RegimeMMConfig,
            build_book_snapshot,
            classify_regime,
            rolling_anchor,
        )

        pd.options.display.max_columns = 200
        pd.options.display.float_format = lambda value: f"{value:,.4f}"
        pio.renderers.default = "plotly_mimetype+notebook"
        px.defaults.template = "plotly_white"
        px.defaults.width = 1200
        px.defaults.height = 520
        px.defaults.color_discrete_sequence = ["#0F766E", "#DC2626", "#2563EB", "#D97706", "#7C3AED", "#059669"]

        OSMIUM = "ASH_COATED_OSMIUM"
        PEPPER = "INTARIAN_PEPPER_ROOT"
        OSMIUM_FAIR = 10_000.0


        def add_session_time(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], int]:
            out = frame.copy()
            positive_steps = (
                out.sort_values(["day", "timestamp"])
                .groupby("day")["timestamp"]
                .diff()
                .dropna()
            )
            positive_steps = positive_steps[positive_steps > 0]
            timestamp_step = int(positive_steps.min()) if not positive_steps.empty else 100
            day_span = int(out["timestamp"].max()) + timestamp_step
            day_offsets = {day: index * day_span for index, day in enumerate(sorted(out["day"].unique()))}
            out["session_time"] = out["timestamp"] + out["day"].map(day_offsets)
            out["day_label"] = out["day"].map(lambda day: f"day {day}")
            return out, day_offsets, day_span


        def summarize_prices(prices: pd.DataFrame, round_label: str) -> pd.DataFrame:
            return (
                prices.groupby("product")
                .agg(
                    rows=("product", "size"),
                    mean_mid=("book_mid", "mean"),
                    mid_std=("book_mid", "std"),
                    mean_spread=("spread", "mean"),
                    spread_p90=("spread", lambda series: series.quantile(0.9)),
                    mean_bid_volume=("total_bid_volume", "mean"),
                    mean_ask_volume=("total_ask_volume", "mean"),
                    imbalance_std=("book_imbalance", "std"),
                )
                .round(4)
                .reset_index()
                .assign(round=round_label)
            )


        def summarize_trades(trades: pd.DataFrame, round_label: str) -> pd.DataFrame:
            return (
                trades.groupby("symbol")
                .agg(
                    trade_count=("symbol", "size"),
                    mean_price=("price", "mean"),
                    price_std=("price", "std"),
                    mean_quantity=("quantity", "mean"),
                    quantity_p90=("quantity", lambda series: series.quantile(0.9)),
                    mean_notional=("notional", "mean"),
                )
                .round(4)
                .reset_index()
                .assign(round=round_label)
            )


        def compare_round_summaries(round1_summary: pd.DataFrame, round2_summary: pd.DataFrame, key: str) -> pd.DataFrame:
            merged = round1_summary.merge(round2_summary, on=key, suffixes=("_round1", "_round2"))
            metric_columns = [column for column in round1_summary.columns if column not in {key, "round"}]
            for metric in metric_columns:
                merged[f"{metric}_delta"] = merged[f"{metric}_round2"] - merged[f"{metric}_round1"]
            ordered_columns = [key]
            for metric in metric_columns:
                ordered_columns.extend([f"{metric}_round1", f"{metric}_round2", f"{metric}_delta"])
            return merged[ordered_columns].round(4)


        def align_trades_to_quotes(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
            quote_reference = (
                prices[["day", "timestamp", "product", "book_mid", "wall_mid", "spread", "book_imbalance"]]
                .sort_values(["day", "product", "timestamp"])
                .reset_index(drop=True)
            )
            trade_reference = (
                trades.rename(columns={"symbol": "product"})
                .sort_values(["day", "product", "timestamp"])
                .reset_index(drop=True)
            )
            aligned_frames: list[pd.DataFrame] = []
            for (day, product), trade_group in trade_reference.groupby(["day", "product"], sort=True):
                quote_group = quote_reference.loc[
                    (quote_reference["day"] == day) & (quote_reference["product"] == product),
                    ["timestamp", "book_mid", "wall_mid", "spread", "book_imbalance"],
                ].sort_values("timestamp")
                joined = pd.merge_asof(
                    trade_group.sort_values("timestamp"),
                    quote_group,
                    on="timestamp",
                    direction="backward",
                )
                aligned_frames.append(joined)
            aligned = (
                pd.concat(aligned_frames, ignore_index=True)
                .sort_values(["day", "product", "timestamp"])
                .reset_index(drop=True)
            )
            aligned["trade_minus_book_mid"] = aligned["price"] - aligned["book_mid"]
            aligned["trade_minus_wall_mid"] = aligned["price"] - aligned["wall_mid"]
            return aligned


        def depth_from_row(row: pd.Series) -> OrderDepth:
            depth = OrderDepth()
            for level in (1, 2, 3):
                bid_price = row.get(f"bid_price_{level}")
                bid_volume = row.get(f"bid_volume_{level}")
                ask_price = row.get(f"ask_price_{level}")
                ask_volume = row.get(f"ask_volume_{level}")
                if pd.notna(bid_price) and pd.notna(bid_volume):
                    depth.buy_orders[int(bid_price)] = int(bid_volume)
                if pd.notna(ask_price) and pd.notna(ask_volume):
                    depth.sell_orders[int(ask_price)] = -int(ask_volume)
            return depth


        def classify_osmium_regimes(prices: pd.DataFrame, round_label: str) -> pd.DataFrame:
            params = RegimeMMConfig()
            osmium = prices[prices["product"] == OSMIUM].sort_values(["day", "timestamp"]).copy()
            mid_history: list[float] = []
            regimes: list[str] = []
            for _, row in osmium.iterrows():
                fallback_mid = mid_history[-1] if mid_history else None
                snapshot = build_book_snapshot(depth_from_row(row), fallback_mid=fallback_mid)
                if snapshot.book_state == "both_sides" and snapshot.mid_price is not None:
                    mid_history.append(float(snapshot.mid_price))
                    mid_history = mid_history[-params.anchor_lookback :]
                    snapshot = build_book_snapshot(depth_from_row(row), fallback_mid=mid_history[-1])
                anchor_price = rolling_anchor(mid_history, params.anchor_lookback)
                regimes.append(classify_regime(snapshot, anchor_price, params) if anchor_price is not None else "standby")
            regime_share = (
                pd.Series(regimes, name="regime")
                .value_counts(normalize=True)
                .rename_axis("regime")
                .reset_index(name="share")
            )
            regime_share["round"] = round_label
            return regime_share.round(4)


        def rolling_linear_slope(series: pd.Series, window: int) -> pd.Series:
            x_values = np.arange(window, dtype=float)
            x_centered = x_values - x_values.mean()
            denominator = float(np.square(x_centered).sum())
            return series.rolling(window).apply(
                lambda values: float(np.dot(values - values.mean(), x_centered) / denominator),
                raw=True,
            )


        def pepper_trend_summary(prices: pd.DataFrame, round_label: str) -> pd.DataFrame:
            pepper = prices[prices["product"] == PEPPER].sort_values(["day", "timestamp"]).copy()
            pepper["roll_slope_40"] = pepper.groupby("day", group_keys=False)["wall_mid"].apply(
                lambda series: rolling_linear_slope(series, 40)
            )
            pepper["first_mid_in_day"] = pepper.groupby("day")["book_mid"].transform("first")
            per_day = (
                pepper.groupby("day")
                .agg(
                    first_mid=("book_mid", "first"),
                    last_mid=("book_mid", "last"),
                    mean_mid=("book_mid", "mean"),
                    mean_spread=("spread", "mean"),
                    spread_p90=("spread", lambda series: series.quantile(0.9)),
                    positive_slope_share=("roll_slope_40", lambda series: float((series > 0).mean())),
                    negative_slope_share=("roll_slope_40", lambda series: float((series < 0).mean())),
                    mean_slope=("roll_slope_40", "mean"),
                    median_slope=("roll_slope_40", "median"),
                )
                .reset_index()
            )
            per_day["intraday_change"] = per_day["last_mid"] - per_day["first_mid"]
            per_day["round"] = round_label
            return per_day.round(4)


        def signal_diagnostics(prices: pd.DataFrame, round_label: str) -> pd.DataFrame:
            frame = prices.sort_values(["product", "day", "timestamp"]).copy()
            frame["future_mid_5"] = frame.groupby(["product", "day"])["book_mid"].shift(-5)
            frame["future_return_5"] = frame["future_mid_5"] - frame["book_mid"]
            rows: list[dict[str, float | str]] = []
            for product_name, product_frame in frame.groupby("product"):
                subset = product_frame[["book_imbalance", "future_return_5"]].dropna()
                rows.append(
                    {
                        "round": round_label,
                        "product": product_name,
                        "imbalance_to_future5_corr": subset["book_imbalance"].corr(subset["future_return_5"]),
                    }
                )
            return pd.DataFrame(rows).round(4)


        def microstructure_motion(prices: pd.DataFrame, round_label: str) -> pd.DataFrame:
            frame = prices.sort_values(["product", "day", "timestamp"]).copy()
            frame["mid_change"] = frame.groupby(["product", "day"])["book_mid"].diff()
            return (
                frame.groupby("product")
                .agg(
                    abs_mid_change_mean=("mid_change", lambda series: series.abs().mean()),
                    abs_mid_change_p90=("mid_change", lambda series: series.abs().quantile(0.9)),
                    zero_change_share=("mid_change", lambda series: float((series.fillna(0) == 0).mean())),
                    one_tick_share=("mid_change", lambda series: float((series.abs() == 1).mean())),
                    gt3_tick_share=("mid_change", lambda series: float((series.abs() > 3).mean())),
                )
                .round(4)
                .reset_index()
                .assign(round=round_label)
            )


        def prepare_osmium_quote_signal_frame(prices: pd.DataFrame) -> pd.DataFrame:
            frame = prices.loc[prices["product"] == OSMIUM].sort_values(["day", "timestamp"]).copy()
            for level in (1, 2, 3):
                for side in ("bid", "ask"):
                    volume_column = f"{side}_volume_{level}"
                    frame[f"{side}_abs_volume_{level}"] = frame[volume_column].abs()

            frame["future_return_1"] = frame.groupby("day")["book_mid"].shift(-1) - frame["book_mid"]
            frame["future_return_5"] = frame.groupby("day")["book_mid"].shift(-5) - frame["book_mid"]
            frame["future_return_10"] = frame.groupby("day")["book_mid"].shift(-10) - frame["book_mid"]

            frame["best_imbalance"] = (
                (frame["bid_abs_volume_1"] - frame["ask_abs_volume_1"])
                / (frame["bid_abs_volume_1"] + frame["ask_abs_volume_1"])
            )
            frame["total_bid_volume_3"] = frame[[f"bid_abs_volume_{level}" for level in (1, 2, 3)]].fillna(0).sum(axis=1)
            frame["total_ask_volume_3"] = frame[[f"ask_abs_volume_{level}" for level in (1, 2, 3)]].fillna(0).sum(axis=1)
            frame["total_imbalance_3"] = (
                (frame["total_bid_volume_3"] - frame["total_ask_volume_3"])
                / (frame["total_bid_volume_3"] + frame["total_ask_volume_3"]).replace(0, np.nan)
            )
            frame["book_state"] = np.select(
                [
                    frame["bid_abs_volume_1"].notna() & frame["ask_abs_volume_1"].notna(),
                    frame["bid_abs_volume_1"].notna() & frame["ask_abs_volume_1"].isna(),
                    frame["bid_abs_volume_1"].isna() & frame["ask_abs_volume_1"].notna(),
                ],
                ["both_sides", "bid_only", "ask_only"],
                default="empty",
            )

            frame["major_up_move_5"] = frame["future_return_5"] >= frame["future_return_5"].quantile(0.99)
            frame["major_down_move_5"] = frame["future_return_5"] <= frame["future_return_5"].quantile(0.01)
            return frame
        """
    )
)

cells.append(
    md(
        """
        ## Plan

        Hypotheses to test:

        1. `ASH_COATED_OSMIUM` should still look like the anchored, mean-reverting book from round 1.
        2. `INTARIAN_PEPPER_ROOT` should still look like the steadily climbing product, but with day labels shifted forward by one session.
        3. `trader_v7` should still fit the data structurally, especially if round 2 preserves the same Osmium regime mix and Pepper trend slope.
        4. If Osmium has an informed trader, the best evidence should appear as predictive displayed liquidity before sharp future moves.
        5. Any meaningful change is more likely to show up in spread, fill cadence, or trade alignment than in the broad price process.
        """
    )
)

cells.append(
    code(
        """
        # Load round 2 as the focus dataset, then load round 1 as the comparison baseline
        raw_round2_prices = load_round2_prices(file_format="csv")
        raw_round2_trades = load_round2_trades(file_format="csv")
        raw_round1_prices = load_round1_prices(file_format="csv")
        raw_round1_trades = load_round1_trades(file_format="csv")

        prices = build_round2_price_features(raw_round2_prices).loc[lambda frame: frame["book_mid"] > 0].copy()
        trades = raw_round2_trades.copy()
        round1_prices = build_round1_price_features(raw_round1_prices).loc[lambda frame: frame["book_mid"] > 0].copy()
        round1_trades = raw_round1_trades.copy()

        prices_plot, _, _ = add_session_time(prices)
        trades_plot, _, _ = add_session_time(trades)
        round1_prices_plot, _, _ = add_session_time(round1_prices)

        trade_with_quotes = align_trades_to_quotes(prices, trades)
        round1_trade_with_quotes = align_trades_to_quotes(round1_prices, round1_trades)
        osmium_signal_data = prepare_osmium_quote_signal_frame(prices)

        inventory = pd.DataFrame(
            {
                "dataset": ["round 2 quotes", "round 2 trades", "round 1 quotes", "round 1 trades"],
                "rows": [len(prices), len(trades), len(round1_prices), len(round1_trades)],
                "products": [
                    sorted(prices["product"].unique().tolist()),
                    sorted(trades["symbol"].unique().tolist()),
                    sorted(round1_prices["product"].unique().tolist()),
                    sorted(round1_trades["symbol"].unique().tolist()),
                ],
                "days": [
                    sorted(prices["day"].unique().tolist()),
                    sorted(trades["day"].unique().tolist()),
                    sorted(round1_prices["day"].unique().tolist()),
                    sorted(round1_trades["day"].unique().tolist()),
                ],
            }
        )
        display(inventory)
        """
    )
)

cells.append(
    code(
        """
        # Build summary tables for round 2 first, then compare those summaries directly with round 1
        round2_price_summary = summarize_prices(prices, "round 2")
        round2_trade_summary = summarize_trades(trades, "round 2")
        round1_price_summary = summarize_prices(round1_prices, "round 1")
        round1_trade_summary = summarize_trades(round1_trades, "round 1")

        price_comparison = compare_round_summaries(round1_price_summary, round2_price_summary, "product")
        trade_comparison = compare_round_summaries(round1_trade_summary, round2_trade_summary, "symbol")

        display(round2_price_summary)
        display(round2_trade_summary)
        display(price_comparison)
        display(trade_comparison)
        """
    )
)

cells.append(
    code(
        """
        # Build a session-time axis so the round 2 quote path reads as one continuous interactive timeline
        quote_price_plot = (
            prices_plot.groupby(["day", "session_time", "product"], as_index=False)[["bid_price_1", "ask_price_1", "book_mid", "wall_mid"]]
            .mean()
        )
        quote_price_plot = quote_price_plot.melt(
            id_vars=["day", "session_time", "product"],
            value_vars=["bid_price_1", "ask_price_1", "book_mid", "wall_mid"],
            var_name="series",
            value_name="price",
        )

        fig = px.line(
            quote_price_plot,
            x="session_time",
            y="price",
            color="series",
            facet_row="product",
            render_mode="webgl",
            title="Round 2 quote overview: top-of-book and wall mid",
            hover_data={"day": True, "session_time": ":.0f", "price": ":.1f"},
        )
        fig.update_yaxes(matches=None)
        fig.update_layout(legend_title_text="")
        fig.show()
        """
    )
)

cells.append(
    code(
        """
        # Zoom the first 75 observations for each asset to inspect the early quote ladder in more detail
        first_n = 75
        early_quote_plot = (
            prices.sort_values(["product", "day", "timestamp"])
            .groupby("product", group_keys=False)
            .head(first_n)
            .copy()
        )
        early_quote_plot["tick"] = early_quote_plot.groupby("product").cumcount() + 1
        early_quote_plot = early_quote_plot.melt(
            id_vars=["product", "day", "timestamp", "tick"],
            value_vars=["bid_price_1", "ask_price_1", "book_mid", "wall_mid"],
            var_name="series",
            value_name="price",
        )

        fig = px.line(
            early_quote_plot,
            x="tick",
            y="price",
            color="series",
            facet_row="product",
            markers=True,
            title="Round 2 early-session quote zoom",
            hover_data={"day": True, "timestamp": True, "price": ":.1f"},
        )
        fig.update_yaxes(matches=None)
        fig.update_layout(legend_title_text="")
        fig.show()
        """
    )
)

cells.append(
    code(
        """
        # Plot trade prices against session time so we can compare actual prints with the quote path
        trade_price_plot = trades_plot.rename(columns={"symbol": "product"}).sort_values(
            ["day", "timestamp", "product", "price"]
        )
        fig = px.scatter(
            trade_price_plot,
            x="session_time",
            y="price",
            color="product",
            facet_row="product",
            size="quantity",
            hover_data={"day": True, "timestamp": True, "quantity": True, "price": ":.1f"},
            title="Round 2 trade prints through time",
            render_mode="webgl",
        )
        fig.update_yaxes(matches=None)
        fig.update_layout(showlegend=False)
        fig.show()
        """
    )
)

cells.append(
    code(
        """
        # Compare spread and imbalance distributions across products
        spread_fig = px.histogram(
            prices,
            x="spread",
            color="product",
            nbins=50,
            opacity=0.75,
            barmode="overlay",
            marginal="box",
            title="Round 2 spread distribution by product",
        )
        spread_fig.update_layout(legend_title_text="")
        spread_fig.show()

        imbalance_fig = px.box(
            prices,
            x="product",
            y="book_imbalance",
            color="product",
            points=False,
            title="Round 2 book imbalance distribution by product",
        )
        imbalance_fig.update_layout(showlegend=False)
        imbalance_fig.show()
        """
    )
)

cells.append(
    code(
        """
        # Join trades to the nearest prior quote so we can compare prints to candidate fair-value references
        alignment_summary = (
            trade_with_quotes.groupby("product")
            .agg(
                trades=("product", "size"),
                mean_trade_minus_book_mid=("trade_minus_book_mid", "mean"),
                mean_abs_trade_minus_book_mid=("trade_minus_book_mid", lambda series: series.abs().mean()),
                mean_trade_minus_wall_mid=("trade_minus_wall_mid", "mean"),
                mean_abs_trade_minus_wall_mid=("trade_minus_wall_mid", lambda series: series.abs().mean()),
            )
            .round(4)
            .reset_index()
        )
        display(alignment_summary)

        trade_reference_plot = trade_with_quotes.melt(
            id_vars=["day", "timestamp", "product"],
            value_vars=["trade_minus_book_mid", "trade_minus_wall_mid"],
            var_name="reference",
            value_name="trade_edge",
        )
        fig = px.box(
            trade_reference_plot.dropna(),
            x="product",
            y="trade_edge",
            color="reference",
            points=False,
            title="Round 2 trade price distance from book mid vs wall mid",
        )
        fig.update_layout(legend_title_text="")
        fig.show()
        """
    )
)

cells.append(
    code(
        """
        # Fixed-fair diagnostic: test whether ASH_COATED_OSMIUM still looks anchored around 10,000
        osmium_data = prices_plot.loc[prices_plot["product"] == OSMIUM].copy()
        osmium_data["book_mid_minus_fair"] = osmium_data["book_mid"] - OSMIUM_FAIR
        osmium_data["wall_mid_minus_fair"] = osmium_data["wall_mid"] - OSMIUM_FAIR

        osmium_fair_summary = (
            osmium_data[["book_mid_minus_fair", "wall_mid_minus_fair"]]
            .agg(["mean", "std", "min", "max"])
            .round(4)
            .T
        )
        display(osmium_fair_summary)

        fair_deviation_plot = osmium_data.melt(
            id_vars=["day", "timestamp", "session_time"],
            value_vars=["book_mid_minus_fair", "wall_mid_minus_fair"],
            var_name="series",
            value_name="deviation",
        )

        fig = px.line(
            fair_deviation_plot,
            x="session_time",
            y="deviation",
            color="series",
            render_mode="webgl",
            title="ASH_COATED_OSMIUM deviation from fixed fair = 10,000",
            hover_data={"day": True, "timestamp": True, "deviation": ":.2f"},
        )
        fig.update_layout(legend_title_text="")
        fig.show()

        hist = px.histogram(
            fair_deviation_plot,
            x="deviation",
            color="series",
            nbins=60,
            barmode="overlay",
            opacity=0.75,
            marginal="box",
            title="ASH_COATED_OSMIUM fair-value deviation distribution",
        )
        hist.update_layout(legend_title_text="")
        hist.show()
        """
    )
)

cells.append(
    md(
        """
        ## Harmonic Check for `ASH_COATED_OSMIUM`

        A Fourier view helps test whether the deviations around the 10,000 anchor contain stable repeating components, or whether the movement is mostly drift plus noise.
        """
    )
)

cells.append(
    code(
        """
        # Fourier / harmonic diagnostic for ASH_COATED_OSMIUM
        osmium_fft_input = osmium_data.sort_values(["day", "timestamp"])["book_mid_minus_fair"].to_numpy()
        osmium_fft_input = osmium_fft_input - osmium_fft_input.mean()
        fft_values = np.fft.rfft(osmium_fft_input)
        fft_frequencies = np.fft.rfftfreq(len(osmium_fft_input), d=1)

        harmonic_table = (
            pd.DataFrame(
                {
                    "frequency": fft_frequencies[1:],
                    "amplitude": np.abs(fft_values[1:]),
                }
            )
            .assign(period_ticks=lambda frame: 1.0 / frame["frequency"])
            .nlargest(15, "amplitude")
            .sort_values("frequency")
            .reset_index(drop=True)
        )
        display(harmonic_table.round(4))

        fig = px.bar(
            harmonic_table,
            x="period_ticks",
            y="amplitude",
            hover_data={"frequency": ":.6f", "period_ticks": ":.1f", "amplitude": ":.2f"},
            title="ASH_COATED_OSMIUM dominant harmonic components",
        )
        fig.update_xaxes(type="log", title="period (ticks, log scale)")
        fig.show()
        """
    )
)

cells.append(
    md(
        """
        ## Informed-Quoting Check for `ASH_COATED_OSMIUM`

        The practical question is not only whether Osmium mean-reverts, but whether the displayed quote stack contains predictive information right before large moves.

        The tests below check three different versions of that story:

        - large visible best quotes at the top of book
        - one-sided or highly imbalanced books just before sharp future shifts
        - exact best-bid / best-ask size combinations that repeatedly show up before major moves
        """
    )
)

cells.append(
    code(
        """
        # Test whether displayed Osmium quote size and imbalance predict future moves
        predictive_features = [
            "bid_abs_volume_1",
            "ask_abs_volume_1",
            "bid_abs_volume_2",
            "ask_abs_volume_2",
            "best_imbalance",
            "total_imbalance_3",
            "spread",
        ]
        future_columns = ["future_return_1", "future_return_5", "future_return_10"]

        predictive_rows: list[dict[str, float | str]] = []
        for feature in predictive_features:
            for future_column in future_columns:
                subset = osmium_signal_data[[feature, future_column]].dropna()
                predictive_rows.append(
                    {
                        "feature": feature,
                        "future_horizon": future_column,
                        "correlation": subset[feature].corr(subset[future_column]),
                    }
                )
        predictive_summary = pd.DataFrame(predictive_rows).round(4)
        display(predictive_summary.pivot(index="feature", columns="future_horizon", values="correlation"))

        book_state_summary = (
            osmium_signal_data.groupby("book_state")
            .agg(
                rows=("book_state", "size"),
                mean_future_return_5=("future_return_5", "mean"),
                abs_future_return_5_p95=("future_return_5", lambda series: series.abs().quantile(0.95)),
            )
            .round(4)
            .reset_index()
        )
        display(book_state_summary)

        major_shift_feature_summary = pd.DataFrame(
            {
                "feature": ["bid_abs_volume_1", "ask_abs_volume_1", "best_imbalance", "total_imbalance_3", "spread"],
                "baseline_mean": [
                    osmium_signal_data["bid_abs_volume_1"].mean(),
                    osmium_signal_data["ask_abs_volume_1"].mean(),
                    osmium_signal_data["best_imbalance"].mean(),
                    osmium_signal_data["total_imbalance_3"].mean(),
                    osmium_signal_data["spread"].mean(),
                ],
                "major_up_move_mean": [
                    osmium_signal_data.loc[osmium_signal_data["major_up_move_5"], "bid_abs_volume_1"].mean(),
                    osmium_signal_data.loc[osmium_signal_data["major_up_move_5"], "ask_abs_volume_1"].mean(),
                    osmium_signal_data.loc[osmium_signal_data["major_up_move_5"], "best_imbalance"].mean(),
                    osmium_signal_data.loc[osmium_signal_data["major_up_move_5"], "total_imbalance_3"].mean(),
                    osmium_signal_data.loc[osmium_signal_data["major_up_move_5"], "spread"].mean(),
                ],
                "major_down_move_mean": [
                    osmium_signal_data.loc[osmium_signal_data["major_down_move_5"], "bid_abs_volume_1"].mean(),
                    osmium_signal_data.loc[osmium_signal_data["major_down_move_5"], "ask_abs_volume_1"].mean(),
                    osmium_signal_data.loc[osmium_signal_data["major_down_move_5"], "best_imbalance"].mean(),
                    osmium_signal_data.loc[osmium_signal_data["major_down_move_5"], "total_imbalance_3"].mean(),
                    osmium_signal_data.loc[osmium_signal_data["major_down_move_5"], "spread"].mean(),
                ],
            }
        ).round(4)
        display(major_shift_feature_summary)

        imbalance_deciles = (
            osmium_signal_data[["best_imbalance", "future_return_5", "future_return_10"]]
            .dropna()
            .assign(imbalance_bucket=lambda frame: pd.qcut(frame["best_imbalance"], 10, duplicates="drop"))
            .groupby("imbalance_bucket", observed=False)
            .agg(
                mean_future_return_5=("future_return_5", "mean"),
                mean_future_return_10=("future_return_10", "mean"),
                count=("future_return_5", "size"),
            )
            .reset_index()
        )
        imbalance_deciles["imbalance_bucket_label"] = imbalance_deciles["imbalance_bucket"].astype(str)
        display(imbalance_deciles.round(4))

        fig = px.bar(
            imbalance_deciles,
            x="imbalance_bucket_label",
            y="mean_future_return_5",
            hover_data={"mean_future_return_10": ":.3f", "count": True},
            title="Osmium best-level imbalance deciles vs future 5-tick return",
        )
        fig.update_xaxes(title="best-level imbalance bucket")
        fig.update_yaxes(title="mean future return over next 5 ticks")
        fig.show()
        """
    )
)

cells.append(
    code(
        """
        # Inspect specific large-quote events and ask whether there is a 'magic' quote size before major shifts
        large_quote_events = pd.DataFrame(
            [
                {
                    "event": "best bid volume >= 30",
                    "count": int((osmium_signal_data["bid_abs_volume_1"] >= 30).sum()),
                    "mean_future_return_1": osmium_signal_data.loc[osmium_signal_data["bid_abs_volume_1"] >= 30, "future_return_1"].mean(),
                    "mean_future_return_5": osmium_signal_data.loc[osmium_signal_data["bid_abs_volume_1"] >= 30, "future_return_5"].mean(),
                    "mean_future_return_10": osmium_signal_data.loc[osmium_signal_data["bid_abs_volume_1"] >= 30, "future_return_10"].mean(),
                },
                {
                    "event": "best ask volume >= 30",
                    "count": int((osmium_signal_data["ask_abs_volume_1"] >= 30).sum()),
                    "mean_future_return_1": osmium_signal_data.loc[osmium_signal_data["ask_abs_volume_1"] >= 30, "future_return_1"].mean(),
                    "mean_future_return_5": osmium_signal_data.loc[osmium_signal_data["ask_abs_volume_1"] >= 30, "future_return_5"].mean(),
                    "mean_future_return_10": osmium_signal_data.loc[osmium_signal_data["ask_abs_volume_1"] >= 30, "future_return_10"].mean(),
                },
            ]
        ).round(4)
        display(large_quote_events)

        exact_size_combos = (
            osmium_signal_data[["bid_abs_volume_1", "ask_abs_volume_1", "future_return_5"]]
            .dropna()
            .groupby(["bid_abs_volume_1", "ask_abs_volume_1"])
            .agg(count=("future_return_5", "size"), mean_future_return_5=("future_return_5", "mean"))
            .reset_index()
            .loc[lambda frame: frame["count"] >= 20]
            .sort_values("mean_future_return_5", ascending=False)
        )
        display(exact_size_combos.head(12).round(4))
        display(exact_size_combos.tail(12).round(4))

        heatmap = exact_size_combos.pivot(index="bid_abs_volume_1", columns="ask_abs_volume_1", values="mean_future_return_5")
        fig = px.imshow(
            heatmap.sort_index().sort_index(axis=1),
            origin="lower",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-4.5,
            zmax=4.5,
            title="Osmium exact best-bid / best-ask size combinations vs future 5-tick return",
        )
        fig.update_xaxes(title="best ask volume")
        fig.update_yaxes(title="best bid volume")
        fig.show()

        strongest_up_examples = osmium_signal_data.sort_values("future_return_5", ascending=False)[
            [
                "day",
                "timestamp",
                "book_mid",
                "future_return_1",
                "future_return_5",
                "future_return_10",
                "bid_price_1",
                "bid_volume_1",
                "bid_price_2",
                "bid_volume_2",
                "ask_price_1",
                "ask_volume_1",
                "ask_price_2",
                "ask_volume_2",
                "book_state",
                "best_imbalance",
                "total_imbalance_3",
            ]
        ].head(10)
        strongest_down_examples = osmium_signal_data.sort_values("future_return_5", ascending=True)[
            [
                "day",
                "timestamp",
                "book_mid",
                "future_return_1",
                "future_return_5",
                "future_return_10",
                "bid_price_1",
                "bid_volume_1",
                "bid_price_2",
                "bid_volume_2",
                "ask_price_1",
                "ask_volume_1",
                "ask_price_2",
                "ask_volume_2",
                "book_state",
                "best_imbalance",
                "total_imbalance_3",
            ]
        ].head(10)
        display(strongest_up_examples.round(4))
        display(strongest_down_examples.round(4))

        informed_commentary = '''
        ## Informed Trader Verdict for Osmium

        - There **is** evidence that the displayed Osmium book is informative right before large moves.
        - The clearest signal is not one special hidden volume, but **visible top-of-book imbalance** and, in the biggest moves, outright **one-sided books**.
        - When only bids are visible, the next 5-tick move is strongly positive on average; when only asks are visible, it is strongly negative on average. That is a much stronger signature than any single deep-book volume level.
        - Large best quotes matter too: `30` at best bid tends to precede upward moves and `30` at best ask tends to precede downward moves. But once you condition on best-level imbalance, most of that effect looks like imbalance exposure rather than a unique magic size.
        - So the honest read is: **yes, informed quoting / adverse-selection risk is present in the visible book**, but **no, the notebook does not show a single unmistakable informed trader using one specific lot size pattern**. The strongest pre-shift fingerprint is extreme displayed imbalance.
        '''
        display(Markdown(informed_commentary))
        """
    )
)

cells.append(
    code(
        """
        # Dynamic-fair diagnostic: compare book mid and wall mid for INTARIAN_PEPPER_ROOT
        pepper_data = prices_plot.loc[prices_plot["product"] == PEPPER].copy()
        pepper_day_summary = pepper_trend_summary(prices, "round 2")
        display(pepper_day_summary)

        pepper_line_plot = pepper_data.melt(
            id_vars=["day", "timestamp", "session_time"],
            value_vars=["book_mid", "wall_mid"],
            var_name="series",
            value_name="price",
        )
        fig = px.line(
            pepper_line_plot,
            x="session_time",
            y="price",
            color="series",
            render_mode="webgl",
            title="INTARIAN_PEPPER_ROOT: book mid vs wall mid",
            hover_data={"day": True, "timestamp": True, "price": ":.1f"},
        )
        fig.update_layout(legend_title_text="")
        fig.show()

        wall_gap_fig = px.histogram(
            pepper_data,
            x="wall_mid_minus_book_mid",
            nbins=60,
            marginal="box",
            title="INTARIAN_PEPPER_ROOT wall-mid minus book-mid distribution",
        )
        wall_gap_fig.show()
        """
    )
)

cells.append(
    code(
        """
        # Compare round 2 directly to round 1 using normalized paths and signal diagnostics
        comparison_paths = pd.concat(
            [
                round1_prices.assign(round="round 1"),
                prices.assign(round="round 2"),
            ],
            ignore_index=True,
        ).sort_values(["round", "product", "day", "timestamp"])

        comparison_paths["tick_in_day"] = comparison_paths.groupby(["round", "product", "day"]).cumcount()
        comparison_paths["first_mid"] = comparison_paths.groupby(["round", "product", "day"])["book_mid"].transform("first")
        comparison_paths["normalized_mid"] = comparison_paths["book_mid"] - comparison_paths["first_mid"]
        comparison_paths["round_day"] = comparison_paths["round"] + " | day " + comparison_paths["day"].astype(str)

        fig = px.line(
            comparison_paths,
            x="tick_in_day",
            y="normalized_mid",
            color="round_day",
            facet_row="product",
            render_mode="webgl",
            title="Round 1 vs round 2 normalized intraday paths",
            hover_data={"day": True, "tick_in_day": True, "normalized_mid": ":.1f"},
        )
        fig.update_yaxes(matches=None)
        fig.update_layout(legend_title_text="")
        fig.show()

        signal_table = pd.concat(
            [
                signal_diagnostics(round1_prices, "round 1"),
                signal_diagnostics(prices, "round 2"),
            ],
            ignore_index=True,
        )
        motion_table = pd.concat(
            [
                microstructure_motion(round1_prices, "round 1"),
                microstructure_motion(prices, "round 2"),
            ],
            ignore_index=True,
        )
        display(signal_table)
        display(motion_table)
        """
    )
)

cells.append(
    code(
        """
        # Translate the data into trader_v7-specific commentary
        osmium_regimes = pd.concat(
            [
                classify_osmium_regimes(round1_prices, "round 1"),
                classify_osmium_regimes(prices, "round 2"),
            ],
            ignore_index=True,
        )
        pepper_slopes = pd.concat(
            [
                pepper_trend_summary(round1_prices, "round 1"),
                pepper_trend_summary(prices, "round 2"),
            ],
            ignore_index=True,
        )

        display(osmium_regimes)
        display(pepper_slopes)

        osmium_regime_pivot = osmium_regimes.pivot(index="regime", columns="round", values="share").fillna(0.0)
        signal_pivot = signal_table.pivot(index="product", columns="round", values="imbalance_to_future5_corr")
        motion_pivot = motion_table.pivot(index="product", columns="round")

        osmium_normal_round1 = float(osmium_regime_pivot.loc["normal", "round 1"])
        osmium_normal_round2 = float(osmium_regime_pivot.loc["normal", "round 2"])
        osmium_signal_round1 = float(signal_pivot.loc[OSMIUM, "round 1"])
        osmium_signal_round2 = float(signal_pivot.loc[OSMIUM, "round 2"])

        pepper_positive_round2 = float(
            pepper_slopes.loc[pepper_slopes["round"] == "round 2", "positive_slope_share"].mean()
        )
        pepper_change_round2 = float(
            pepper_slopes.loc[pepper_slopes["round"] == "round 2", "intraday_change"].mean()
        )
        pepper_spread_round1 = float(round1_price_summary.loc[round1_price_summary["product"] == PEPPER, "mean_spread"].iloc[0])
        pepper_spread_round2 = float(round2_price_summary.loc[round2_price_summary["product"] == PEPPER, "mean_spread"].iloc[0])
        pepper_one_tick_round1 = float(
            motion_table.loc[(motion_table["round"] == "round 1") & (motion_table["product"] == PEPPER), "one_tick_share"].iloc[0]
        )
        pepper_one_tick_round2 = float(
            motion_table.loc[(motion_table["round"] == "round 2") & (motion_table["product"] == PEPPER), "one_tick_share"].iloc[0]
        )

        commentary = f'''
        ## Trader `v7` Read-Through

        - `ASH_COATED_OSMIUM`: the v7 Osmium market-maker should transfer cleanly. Its own regime logic classifies the book as `normal` on **{osmium_normal_round2:.2%}** of round 2 ticks versus **{osmium_normal_round1:.2%}** in round 1, and the round 2 imbalance-to-future-return correlation (**{osmium_signal_round2:.3f}**) is basically unchanged from round 1 (**{osmium_signal_round1:.3f}**). That is exactly the environment its rolling-anchor / imbalance-skew logic wants.
        - `INTARIAN_PEPPER_ROOT`: the Pepper leg still fits the broad shape of the data. The rolling 40-tick wall-mid slope is positive on average for **{pepper_positive_round2:.2%}** of round 2 observations, and the product still climbs roughly **{pepper_change_round2:,.1f}** ticks per day. That lines up with v7's long-core, trend-following Pepper behavior.
        - Main caveat: round 2 Pepper is a bit wider and a bit less one-tick-y. Mean Pepper spread rises from **{pepper_spread_round1:.2f}** in round 1 to **{pepper_spread_round2:.2f}** in round 2, while the share of exactly one-tick mid moves falls from **{pepper_one_tick_round1:.2%}** to **{pepper_one_tick_round2:.2%}**. So I would expect v7 to stay directionally comfortable, but recycle inventory a little more slowly and carry positions slightly longer.
        - Bottom line: structurally favorable for v7, especially on Osmium and on the Pepper long-bias. This notebook does **not** run a fill-aware PnL backtest, so treat this as a market-structure fit assessment rather than a precise profit forecast.
        '''
        display(Markdown(commentary))
        """
    )
)

cells.append(
    md(
        """
        ## Results

        Round 2 looks much more like a continuation of round 1 than a new regime:

        - `ASH_COATED_OSMIUM` remains anchored around 10,000 with nearly unchanged spread, depth, and imbalance signal quality.
        - `INTARIAN_PEPPER_ROOT` still behaves like the persistent ladder product, now one day further along.
        - The meaningful differences are second-order: Pepper is a little wider, trades a touch differently, and should be slightly less forgiving to inventory-heavy execution than round 1.

        If you want to turn this into a trading workflow next, the natural follow-up is a true round 2 replay/backtest for `trader_v7` so we can quantify fills, inventory path, and realized PnL rather than relying on structure alone.
        """
    )
)

nb["cells"] = cells
NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Wrote notebook to {NOTEBOOK_PATH}")
