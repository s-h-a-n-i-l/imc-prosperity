from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(r"f:\Projects\imc\imc-prosperity")
NOTEBOOK_PATH = ROOT / "src" / "imc_eda" / "round3" / "round-3-eda.ipynb"
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
        # Experiment: Round 3 EDA

        Objective:

        - profile the round 3 quote and trade microstructure for `HYDROGEL_PACK`, `VELVETFRUIT_EXTRACT`, and the ten `VEV_*` vouchers
        - identify whether `HYDROGEL_PACK` behaves like an anchored mean-reverter, whether `VELVETFRUIT_EXTRACT` behaves like a drifting underlying, and how the voucher ladder maps that underlying
        - measure where voucher liquidity and trade activity concentrate across strike space
        - turn those observations into practical strategy guidance for the final-round trader we build next
        """
    )
)

cells.append(
    code(
        """
        # Setup: imports, paths, plotting defaults, and reusable helpers
        from __future__ import annotations

        import re
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

        from imc_eda.round3 import build_price_features as build_round3_price_features
        from imc_eda.round3 import load_prices as load_round3_prices
        from imc_eda.round3 import load_trades as load_round3_trades

        pd.options.display.max_columns = 200
        pd.options.display.float_format = lambda value: f"{value:,.4f}"
        pio.renderers.default = "plotly_mimetype+notebook"
        px.defaults.template = "plotly_white"
        px.defaults.width = 1200
        px.defaults.height = 520
        px.defaults.color_discrete_sequence = ["#0F766E", "#DC2626", "#2563EB", "#D97706", "#7C3AED", "#059669"]

        HYDROGEL = "HYDROGEL_PACK"
        UNDERLYING = "VELVETFRUIT_EXTRACT"
        VOUCHER_PREFIX = "VEV_"
        DELTA1_PRODUCTS = [HYDROGEL, UNDERLYING]
        HYDROGEL_FAIR = 10_000.0
        STRIKE_PATTERN = re.compile(r"(\\d+)$")


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
            aligned["signed_trade_minus_book_mid"] = np.sign(aligned["trade_minus_book_mid"]) * aligned["trade_minus_book_mid"].abs()
            aligned["abs_trade_minus_book_mid"] = aligned["trade_minus_book_mid"].abs()
            return aligned


        def extract_strike(product: str) -> float:
            match = STRIKE_PATTERN.search(product)
            return float(match.group(1)) if match else np.nan


        def rolling_linear_slope(series: pd.Series, window: int) -> pd.Series:
            x_values = np.arange(window, dtype=float)
            x_centered = x_values - x_values.mean()
            denominator = float(np.square(x_centered).sum())
            return series.rolling(window).apply(
                lambda values: float(np.dot(values - values.mean(), x_centered) / denominator),
                raw=True,
            )


        def safe_corr(left: pd.Series, right: pd.Series) -> float:
            pair = pd.concat(
                [
                    pd.Series(left, copy=False, name="left"),
                    pd.Series(right, copy=False, name="right"),
                ],
                axis=1,
            ).dropna()
            if pair.empty:
                return np.nan

            left_std = pair["left"].std()
            right_std = pair["right"].std()
            if pd.isna(left_std) or pd.isna(right_std) or left_std <= 0 or right_std <= 0:
                return np.nan
            return float(pair["left"].corr(pair["right"]))


        def safe_corr_matrix(frame: pd.DataFrame) -> pd.DataFrame:
            columns = list(frame.columns)
            output = pd.DataFrame(np.nan, index=columns, columns=columns, dtype=float)
            for idx, left_column in enumerate(columns):
                left_std = frame[left_column].dropna().std()
                if pd.notna(left_std) and left_std > 0:
                    output.loc[left_column, left_column] = 1.0
                for right_column in columns[idx + 1 :]:
                    correlation = safe_corr(frame[left_column], frame[right_column])
                    output.loc[left_column, right_column] = correlation
                    output.loc[right_column, left_column] = correlation
            return output


        def prepare_delta1_quote_signal_frame(prices: pd.DataFrame, product_name: str) -> pd.DataFrame:
            frame = prices.loc[prices["product"] == product_name].sort_values(["day", "timestamp"]).copy()
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


        def annotate_option_frame(prices: pd.DataFrame) -> pd.DataFrame:
            underlying = (
                prices.loc[prices["product"] == UNDERLYING, ["day", "timestamp", "book_mid"]]
                .rename(columns={"book_mid": "underlying_mid"})
                .sort_values(["day", "timestamp"])
            )
            options = prices.loc[prices["product"].str.startswith(VOUCHER_PREFIX)].copy()
            options["strike"] = options["product"].map(extract_strike)
            options["tte_days"] = 8 - options["day"]
            options = options.merge(underlying, on=["day", "timestamp"], how="left")
            options["voucher_mid"] = options["book_mid"]
            options["intrinsic_value"] = (options["underlying_mid"] - options["strike"]).clip(lower=0.0)
            options["option_minus_intrinsic"] = options["voucher_mid"] - options["intrinsic_value"]
            options["moneyness"] = options["underlying_mid"] - options["strike"]
            options["normalized_moneyness"] = options["moneyness"] / options["underlying_mid"]
            options["voucher_return_1"] = options.groupby(["product", "day"])["voucher_mid"].diff()
            options["voucher_return_5"] = options.groupby(["product", "day"])["voucher_mid"].diff(5)
            return options.sort_values(["day", "timestamp", "strike"]).reset_index(drop=True)


        def option_sensitivity_summary(option_prices: pd.DataFrame, underlying_prices: pd.DataFrame) -> pd.DataFrame:
            underlying_returns = (
                underlying_prices.loc[:, ["day", "timestamp", "book_mid"]]
                .rename(columns={"book_mid": "underlying_mid"})
                .sort_values(["day", "timestamp"])
            )
            underlying_returns["underlying_return_1"] = underlying_returns.groupby("day")["underlying_mid"].diff()

            merged = option_prices.merge(
                underlying_returns[["day", "timestamp", "underlying_return_1"]],
                on=["day", "timestamp"],
                how="left",
            )

            rows: list[dict[str, float | str]] = []
            for product, frame in merged.groupby("product", sort=True):
                valid = frame[["voucher_return_1", "underlying_return_1"]].dropna()
                variance = valid["underlying_return_1"].var()
                voucher_std = valid["voucher_return_1"].std()
                underlying_std = valid["underlying_return_1"].std()
                beta = np.nan
                correlation = np.nan
                if not valid.empty and pd.notna(variance) and variance > 0:
                    beta = valid["voucher_return_1"].cov(valid["underlying_return_1"]) / variance
                    if pd.notna(voucher_std) and pd.notna(underlying_std) and voucher_std > 0 and underlying_std > 0:
                        correlation = safe_corr(valid["voucher_return_1"], valid["underlying_return_1"])
                rows.append(
                    {
                        "product": product,
                        "strike": extract_strike(product),
                        "obs": len(valid),
                        "beta_1tick": beta,
                        "corr_1tick": correlation,
                        "mean_mid": frame["voucher_mid"].mean(),
                        "mean_spread": frame["spread"].mean(),
                        "mean_option_minus_intrinsic": frame["option_minus_intrinsic"].mean(),
                        "negative_intrinsic_gap_share": (frame["option_minus_intrinsic"] < 0).mean(),
                    }
                )
            return pd.DataFrame(rows).sort_values("strike").round(4).reset_index(drop=True)
        """
    )
)

cells.append(
    md(
        """
        ## Plan

        Hypotheses to test:

        1. `HYDROGEL_PACK` should look like an anchored, mean-reverting delta-1 product around 10,000.
        2. `VELVETFRUIT_EXTRACT` should behave like the true underlying and carry the main directional information for the voucher complex.
        3. Voucher prices should be monotone in strike, with empirical sensitivity falling as strike rises.
        4. Trade activity should cluster in the strikes that still move enough to matter, rather than the very deep OTM vouchers pinned near zero.
        5. The most useful trading edge will likely be a hybrid of simple delta-1 trading in the underlyings plus relative-value quoting across the voucher ladder.
        """
    )
)

cells.append(
    code(
        """
        # Load round 3 data, build reusable feature frames, and inventory the dataset
        raw_prices = load_round3_prices(file_format="csv")
        raw_trades = load_round3_trades(file_format="csv")

        prices = build_round3_price_features(raw_prices).loc[lambda frame: frame["book_mid"] > 0].copy()
        trades = raw_trades.copy()

        prices_plot, _, _ = add_session_time(prices)
        trades_plot, _, _ = add_session_time(trades)

        option_prices = annotate_option_frame(prices)
        delta1_prices = prices.loc[prices["product"].isin(DELTA1_PRODUCTS)].copy()
        delta1_prices_plot = prices_plot.loc[prices_plot["product"].isin(DELTA1_PRODUCTS)].copy()
        delta1_signal_frames = {
            product_name: prepare_delta1_quote_signal_frame(prices, product_name)
            for product_name in DELTA1_PRODUCTS
        }
        trade_with_quotes = align_trades_to_quotes(prices, trades)

        inventory = pd.DataFrame(
            {
                "dataset": ["round 3 quotes", "round 3 trades"],
                "rows": [len(prices), len(trades)],
                "products": [
                    sorted(prices["product"].unique().tolist()),
                    sorted(trades["symbol"].unique().tolist()),
                ],
                "days": [
                    sorted(prices["day"].unique().tolist()),
                    sorted(trades["day"].unique().tolist()),
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
        # Build high-level quote and trade summaries, then summarize the voucher ladder separately
        price_summary = summarize_prices(prices, "round 3").sort_values("product").reset_index(drop=True)
        trade_summary = summarize_trades(trades, "round 3").sort_values("symbol").reset_index(drop=True)
        option_summary = option_sensitivity_summary(option_prices, prices.loc[prices["product"] == UNDERLYING]).copy()

        trade_summary["strike"] = trade_summary["symbol"].map(extract_strike)
        trade_summary["asset_class"] = np.where(
            trade_summary["symbol"].str.startswith(VOUCHER_PREFIX),
            "voucher",
            "delta_1",
        )

        display(price_summary)
        display(trade_summary)
        display(option_summary)
        """
    )
)

cells.append(
    code(
        """
        # Visual overview of the two delta-1 products across the full three-day history
        delta1_line_data = delta1_prices_plot.melt(
            id_vars=["day", "timestamp", "session_time", "product"],
            value_vars=["bid_price_1", "ask_price_1", "book_mid", "wall_mid"],
            var_name="series",
            value_name="price",
        )
        fig = px.line(
            delta1_line_data,
            x="session_time",
            y="price",
            color="series",
            facet_row="product",
            render_mode="webgl",
            title="Round 3 delta-1 products: best quotes, book mid, and wall mid",
            hover_data={"day": True, "timestamp": True, "price": ":.1f"},
        )
        fig.update_yaxes(matches=None)
        fig.update_layout(legend_title_text="")
        fig.show()

        delta1_trade_alignment = (
            trade_with_quotes.loc[trade_with_quotes["product"].isin(DELTA1_PRODUCTS)]
            .groupby("product")
            .agg(
                trade_count=("product", "size"),
                mean_trade_minus_book_mid=("trade_minus_book_mid", "mean"),
                mean_abs_trade_minus_book_mid=("abs_trade_minus_book_mid", "mean"),
                mean_spread=("spread", "mean"),
            )
            .round(4)
            .reset_index()
        )
        display(delta1_trade_alignment)
        """
    )
)

cells.append(
    md(
        """
        ## Cross-Product Correlation

        This section checks how the round 3 products move relative to one another:

        - contemporaneous 1-tick return correlation across the full product set
        - lead/lag relationships between the two delta-1 products and the voucher ladder
        - whether `HYDROGEL_PACK` behaves independently while the voucher complex tracks `VELVETFRUIT_EXTRACT`
        """
    )
)

cells.append(
    code(
        """
        # Cross-product correlation analysis for Hydrogel, Velvetfruit Extract, and the voucher ladder
        correlation_frame = prices.sort_values(["product", "day", "timestamp"]).copy()
        correlation_frame["return_1"] = correlation_frame.groupby(["product", "day"])["book_mid"].diff()
        correlation_frame["return_5"] = correlation_frame.groupby(["product", "day"])["book_mid"].diff(5)
        correlation_frame["future_return_5"] = correlation_frame.groupby(["product", "day"])["book_mid"].shift(-5) - correlation_frame["book_mid"]

        contemporaneous_return_1 = (
            correlation_frame.pivot_table(
                index=["day", "timestamp"],
                columns="product",
                values="return_1",
            )
            .sort_index(axis=1)
        )
        contemporaneous_corr = safe_corr_matrix(contemporaneous_return_1).round(4)
        display(contemporaneous_corr)

        fig = px.imshow(
            contemporaneous_corr,
            origin="lower",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Contemporaneous 1-tick return correlation across all round 3 products",
        )
        fig.show()

        extract_lead_input = (
            correlation_frame.loc[:, ["day", "timestamp", "product", "return_1"]]
            .pivot_table(index=["day", "timestamp"], columns="product", values="return_1")
            .reset_index()
            .sort_values(["day", "timestamp"])
        )
        extract_lead_input["extract_future_return_5"] = (
            extract_lead_input.groupby("day")[UNDERLYING].shift(-5)
            .rolling(5)
            .sum()
            .reset_index(level=0, drop=True)
        )

        lead_lag_rows: list[dict[str, float | str]] = []
        for product_name in sorted(prices["product"].unique()):
            if product_name == UNDERLYING:
                subset = extract_lead_input[[UNDERLYING, "extract_future_return_5"]].dropna()
                corr_with_extract_return_1 = 1.0
                corr_with_extract_future5 = safe_corr(subset[UNDERLYING], subset["extract_future_return_5"])
            else:
                subset = extract_lead_input[[product_name, UNDERLYING, "extract_future_return_5"]].dropna()
                corr_with_extract_return_1 = safe_corr(subset[product_name], subset[UNDERLYING])
                corr_with_extract_future5 = safe_corr(subset[product_name], subset["extract_future_return_5"])
            lead_lag_rows.append(
                {
                    "product": product_name,
                    "corr_with_extract_return_1": corr_with_extract_return_1,
                    "corr_with_extract_future5": corr_with_extract_future5,
                }
            )
        lead_lag_summary = pd.DataFrame(lead_lag_rows).sort_values("product").round(4).reset_index(drop=True)

        hydrogel_returns = (
            correlation_frame.loc[correlation_frame["product"] == HYDROGEL, ["day", "timestamp", "return_1"]]
            .rename(columns={"return_1": "hydrogel_return_1"})
        )
        underlying_returns = (
            correlation_frame.loc[correlation_frame["product"] == UNDERLYING, ["day", "timestamp", "return_1"]]
            .rename(columns={"return_1": "underlying_return_1"})
        )
        voucher_return_panel = option_prices.loc[:, ["day", "timestamp", "product", "strike", "voucher_return_1"]].copy()
        voucher_return_panel = voucher_return_panel.merge(underlying_returns, on=["day", "timestamp"], how="left")
        voucher_return_panel = voucher_return_panel.merge(hydrogel_returns, on=["day", "timestamp"], how="left")
        voucher_cross_summary = (
            voucher_return_panel.groupby("product")
            .agg(
                strike=("strike", "first"),
                corr_with_underlying_return_1=("voucher_return_1", lambda series: safe_corr(
                    series,
                    voucher_return_panel.loc[series.index, "underlying_return_1"]
                )),
                corr_with_hydrogel_return_1=("voucher_return_1", lambda series: safe_corr(
                    series,
                    voucher_return_panel.loc[series.index, "hydrogel_return_1"]
                )),
            )
            .round(4)
            .reset_index()
            .sort_values("strike")
        )

        display(lead_lag_summary)
        display(voucher_cross_summary)

        voucher_corr_plot = px.line(
            voucher_cross_summary,
            x="strike",
            y=["corr_with_underlying_return_1", "corr_with_hydrogel_return_1"],
            markers=True,
            title="Voucher return correlation vs underlying and Hydrogel",
        )
        voucher_corr_plot.update_layout(legend_title_text="")
        voucher_corr_plot.show()
        """
    )
)

cells.append(
    code(
        """
        # Hydrogel diagnostic: quantify anchoring around 10,000 and test for short-horizon mean reversion
        hydrogel = prices_plot.loc[prices_plot["product"] == HYDROGEL].sort_values(["day", "timestamp"]).copy()
        hydrogel["mid_minus_fair"] = hydrogel["book_mid"] - HYDROGEL_FAIR
        hydrogel["return_1"] = hydrogel.groupby("day")["book_mid"].diff()
        hydrogel["future_return_1"] = hydrogel.groupby("day")["book_mid"].shift(-1) - hydrogel["book_mid"]
        hydrogel["future_return_5"] = hydrogel.groupby("day")["book_mid"].shift(-5) - hydrogel["book_mid"]

        hydrogel_day_summary = (
            hydrogel.groupby("day")
            .agg(
                first_mid=("book_mid", "first"),
                last_mid=("book_mid", "last"),
                mean_mid=("book_mid", "mean"),
                mid_std=("book_mid", "std"),
                mean_spread=("spread", "mean"),
                mean_mid_minus_fair=("mid_minus_fair", "mean"),
            )
            .reset_index()
        )
        hydrogel_day_summary["intraday_change"] = hydrogel_day_summary["last_mid"] - hydrogel_day_summary["first_mid"]

        hydrogel["deviation_bucket"] = pd.cut(
            hydrogel["mid_minus_fair"],
            bins=[-np.inf, -40, -20, -10, 10, 20, 40, np.inf],
            labels=["<= -40", "(-40, -20]", "(-20, -10]", "(-10, 10]", "(10, 20]", "(20, 40]", "> 40"],
        )
        hydrogel_reversion = (
            hydrogel.groupby("deviation_bucket", observed=False)
            .agg(
                rows=("deviation_bucket", "size"),
                mean_future_return_1=("future_return_1", "mean"),
                mean_future_return_5=("future_return_5", "mean"),
            )
            .round(4)
            .reset_index()
        )

        display(hydrogel_day_summary.round(4))
        display(hydrogel_reversion)

        fig = px.line(
            hydrogel,
            x="session_time",
            y="mid_minus_fair",
            color="day_label",
            render_mode="webgl",
            title="HYDROGEL_PACK deviation from the 10,000 anchor",
            hover_data={"day": True, "timestamp": True, "mid_minus_fair": ":.1f"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#111827")
        fig.show()

        fig = px.histogram(
            hydrogel,
            x="mid_minus_fair",
            nbins=70,
            marginal="box",
            title="HYDROGEL_PACK deviation-from-anchor distribution",
        )
        fig.show()
        """
    )
)

cells.append(
    code(
        """
        # Velvetfruit Extract diagnostic: inspect trend, spread, and the difference between book and wall mids
        extract = prices_plot.loc[prices_plot["product"] == UNDERLYING].sort_values(["day", "timestamp"]).copy()
        extract["roll_slope_40"] = extract.groupby("day", group_keys=False)["book_mid"].apply(
            lambda series: rolling_linear_slope(series, 40)
        )
        extract_day_summary = (
            extract.groupby("day")
            .agg(
                first_mid=("book_mid", "first"),
                last_mid=("book_mid", "last"),
                mean_mid=("book_mid", "mean"),
                mid_std=("book_mid", "std"),
                mean_spread=("spread", "mean"),
                positive_slope_share=("roll_slope_40", lambda series: float((series > 0).mean())),
                negative_slope_share=("roll_slope_40", lambda series: float((series < 0).mean())),
            )
            .reset_index()
        )
        extract_day_summary["intraday_change"] = extract_day_summary["last_mid"] - extract_day_summary["first_mid"]

        display(extract_day_summary.round(4))

        fig = px.line(
            extract,
            x="session_time",
            y="book_mid",
            color="day_label",
            render_mode="webgl",
            title="VELVETFRUIT_EXTRACT mid-price path",
            hover_data={"day": True, "timestamp": True, "book_mid": ":.1f"},
        )
        fig.show()

        wall_gap_fig = px.histogram(
            extract,
            x="wall_mid_minus_book_mid",
            nbins=70,
            marginal="box",
            title="VELVETFRUIT_EXTRACT wall-mid minus book-mid distribution",
        )
        wall_gap_fig.show()
        """
    )
)

cells.append(
    md(
        """
        ## Informed-Quoting Check for Delta-1 Products

        Round 2's Osmium notebook looked for evidence that the displayed book itself predicts short-horizon future moves.

        The same tests here ask:

        - whether visible best-level size and imbalance predict future returns
        - whether large 5-tick moves are preceded by skewed displayed liquidity
        - whether there is a repeated "magic" best-bid / best-ask size pattern before sharp moves
        """
    )
)

cells.append(
    code(
        """
        # Reuse the round 2 informed-trading workflow for both round 3 delta-1 products
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
        book_state_rows: list[pd.DataFrame] = []
        major_shift_rows: list[pd.DataFrame] = []
        imbalance_decile_frames: list[pd.DataFrame] = []
        informed_summary_rows: list[dict[str, float | str]] = []

        for product_name, signal_frame in delta1_signal_frames.items():
            for feature in predictive_features:
                for future_column in future_columns:
                    subset = signal_frame[[feature, future_column]].dropna()
                    predictive_rows.append(
                        {
                            "product": product_name,
                            "feature": feature,
                            "future_horizon": future_column,
                            "correlation": safe_corr(subset[feature], subset[future_column]),
                        }
                    )

            book_state_summary = (
                signal_frame.groupby("book_state")
                .agg(
                    rows=("book_state", "size"),
                    mean_future_return_5=("future_return_5", "mean"),
                    abs_future_return_5_p95=("future_return_5", lambda series: series.abs().quantile(0.95)),
                )
                .round(4)
                .reset_index()
            )
            book_state_summary["product"] = product_name
            book_state_rows.append(book_state_summary)

            major_shift_feature_summary = pd.DataFrame(
                {
                    "feature": ["bid_abs_volume_1", "ask_abs_volume_1", "best_imbalance", "total_imbalance_3", "spread"],
                    "baseline_mean": [
                        signal_frame["bid_abs_volume_1"].mean(),
                        signal_frame["ask_abs_volume_1"].mean(),
                        signal_frame["best_imbalance"].mean(),
                        signal_frame["total_imbalance_3"].mean(),
                        signal_frame["spread"].mean(),
                    ],
                    "major_up_move_mean": [
                        signal_frame.loc[signal_frame["major_up_move_5"], "bid_abs_volume_1"].mean(),
                        signal_frame.loc[signal_frame["major_up_move_5"], "ask_abs_volume_1"].mean(),
                        signal_frame.loc[signal_frame["major_up_move_5"], "best_imbalance"].mean(),
                        signal_frame.loc[signal_frame["major_up_move_5"], "total_imbalance_3"].mean(),
                        signal_frame.loc[signal_frame["major_up_move_5"], "spread"].mean(),
                    ],
                    "major_down_move_mean": [
                        signal_frame.loc[signal_frame["major_down_move_5"], "bid_abs_volume_1"].mean(),
                        signal_frame.loc[signal_frame["major_down_move_5"], "ask_abs_volume_1"].mean(),
                        signal_frame.loc[signal_frame["major_down_move_5"], "best_imbalance"].mean(),
                        signal_frame.loc[signal_frame["major_down_move_5"], "total_imbalance_3"].mean(),
                        signal_frame.loc[signal_frame["major_down_move_5"], "spread"].mean(),
                    ],
                }
            ).round(4)
            major_shift_feature_summary["product"] = product_name
            major_shift_rows.append(major_shift_feature_summary)

            imbalance_deciles = (
                signal_frame[["best_imbalance", "future_return_5", "future_return_10"]]
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
            imbalance_deciles["product"] = product_name
            imbalance_decile_frames.append(imbalance_deciles)

            best_corr = (
                pd.DataFrame([row for row in predictive_rows if row["product"] == product_name])
                .pivot(index="feature", columns="future_horizon", values="correlation")
            )
            informed_summary_rows.append(
                {
                    "product": product_name,
                    "best_imbalance_to_future5_corr": best_corr.loc["best_imbalance", "future_return_5"],
                    "total_imbalance_3_to_future5_corr": best_corr.loc["total_imbalance_3", "future_return_5"],
                    "spread_to_future5_corr": best_corr.loc["spread", "future_return_5"],
                    "major_up_best_imbalance": major_shift_feature_summary.loc[
                        major_shift_feature_summary["feature"] == "best_imbalance", "major_up_move_mean"
                    ].iloc[0],
                    "major_down_best_imbalance": major_shift_feature_summary.loc[
                        major_shift_feature_summary["feature"] == "best_imbalance", "major_down_move_mean"
                    ].iloc[0],
                    "bottom_decile_future5": imbalance_deciles["mean_future_return_5"].iloc[0],
                    "top_decile_future5": imbalance_deciles["mean_future_return_5"].iloc[-1],
                }
            )

        predictive_summary = pd.DataFrame(predictive_rows).round(4)
        book_state_summary_all = pd.concat(book_state_rows, ignore_index=True)
        major_shift_feature_summary_all = pd.concat(major_shift_rows, ignore_index=True)
        imbalance_deciles_all = pd.concat(imbalance_decile_frames, ignore_index=True)
        informed_delta1_summary = pd.DataFrame(informed_summary_rows).round(4)

        display(predictive_summary.pivot(index=["product", "feature"], columns="future_horizon", values="correlation"))
        display(book_state_summary_all)
        display(major_shift_feature_summary_all)
        display(informed_delta1_summary)

        fig = px.bar(
            imbalance_deciles_all,
            x="imbalance_bucket_label",
            y="mean_future_return_5",
            color="product",
            barmode="group",
            hover_data={"mean_future_return_10": ":.3f", "count": True},
            title="Delta-1 best-level imbalance deciles vs future 5-tick return",
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
        # Inspect whether specific best-size combinations repeatedly show up before large delta-1 moves
        delta1_size_combo_frames: list[pd.DataFrame] = []

        for product_name, signal_frame in delta1_signal_frames.items():
            exact_size_combos = (
                signal_frame[["bid_abs_volume_1", "ask_abs_volume_1", "future_return_5"]]
                .dropna()
                .groupby(["bid_abs_volume_1", "ask_abs_volume_1"])
                .agg(count=("future_return_5", "size"), mean_future_return_5=("future_return_5", "mean"))
                .reset_index()
                .loc[lambda frame: frame["count"] >= 20]
                .sort_values("mean_future_return_5", ascending=False)
            )
            exact_size_combos["product"] = product_name
            delta1_size_combo_frames.append(exact_size_combos)

            display(exact_size_combos.head(10).round(4))
            display(exact_size_combos.tail(10).round(4))

            heatmap = exact_size_combos.pivot(
                index="bid_abs_volume_1",
                columns="ask_abs_volume_1",
                values="mean_future_return_5",
            )
            fig = px.imshow(
                heatmap.sort_index().sort_index(axis=1),
                origin="lower",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title=f"{product_name} exact best-bid / best-ask size combinations vs future 5-tick return",
            )
            fig.update_xaxes(title="best ask volume")
            fig.update_yaxes(title="best bid volume")
            fig.show()

        delta1_size_combo_summary = pd.concat(delta1_size_combo_frames, ignore_index=True)

        informed_commentary = f'''
        ## Delta-1 Informed Trader Verdict

        - `HYDROGEL_PACK`: there is modest evidence that the displayed best quote is informative. Best-level imbalance has a positive correlation of **{informed_delta1_summary.loc[informed_delta1_summary["product"] == HYDROGEL, "best_imbalance_to_future5_corr"].iloc[0]:.3f}** with future 5-tick returns, and the top imbalance bucket is followed by roughly **{informed_delta1_summary.loc[informed_delta1_summary["product"] == HYDROGEL, "top_decile_future5"].iloc[0]:.2f}** ticks over the next 5 updates. But the exact size-combo table is narrow and repetitive, so this looks more like broad top-of-book adverse selection than one special informed lot size.
        - `VELVETFRUIT_EXTRACT`: the informed-book signature is clearer. Best-level imbalance has a future 5-tick correlation of **{informed_delta1_summary.loc[informed_delta1_summary["product"] == UNDERLYING, "best_imbalance_to_future5_corr"].iloc[0]:.3f}**, and strong bid-heavy versus ask-heavy best quotes line up with future moves in the expected direction. The top and bottom exact size combinations are materially asymmetric, which is much closer to the round 2 informed-quoting story.
        - A subtle but important wrinkle is that 3-level imbalance is less useful than best-level imbalance and even points the other way on this dataset. That suggests the actionable signal is concentrated at the very top of book, while deeper displayed size may reflect replenishment or passive support rather than informed urgency.
        '''
        display(Markdown(informed_commentary))
        """
    )
)

cells.append(
    md(
        """
        ## Voucher Surface

        The round 3 novelty is the voucher ladder. Here the useful questions are:

        - does the cross-section stay ordered by strike?
        - which strikes actually carry sensitivity to the underlying?
        - where do trades cluster?
        - how often do quoted mids sit above or below simple intrinsic value?

        The competition description implies these historical files line up with time-to-expiry values `8d`, `7d`, and `6d` for `day 0`, `day 1`, and `day 2`, so the notebook uses `tte_days = 8 - day` as a convenient label.
        """
    )
)

cells.append(
    code(
        """
        # Summarize the voucher ladder across strike and day, including monotonicity and intrinsic-gap diagnostics
        voucher_surface_summary = (
            option_prices.groupby(["day", "tte_days", "strike"])
            .agg(
                mean_voucher_mid=("voucher_mid", "mean"),
                mean_intrinsic_value=("intrinsic_value", "mean"),
                mean_option_minus_intrinsic=("option_minus_intrinsic", "mean"),
                mean_spread=("spread", "mean"),
                mean_book_imbalance=("book_imbalance", "mean"),
            )
            .round(4)
            .reset_index()
        )

        strike_monotonicity = (
            option_prices.sort_values(["day", "timestamp", "strike"])
            .groupby(["day", "timestamp"])["voucher_mid"]
            .diff()
            .gt(0)
            .fillna(False)
            .sum()
        )
        negative_intrinsic_gap = int((option_prices["option_minus_intrinsic"] < 0).sum())

        voucher_health = pd.DataFrame(
            {
                "check": [
                    "strike-monotonicity violations",
                    "rows with option_mid < intrinsic_value",
                    "share with option_mid < intrinsic_value",
                ],
                "value": [
                    int(strike_monotonicity),
                    negative_intrinsic_gap,
                    float((option_prices["option_minus_intrinsic"] < 0).mean()),
                ],
            }
        )

        display(voucher_surface_summary.head(30))
        display(voucher_health.round(4))
        """
    )
)

cells.append(
    code(
        """
        # Plot the average voucher surface by strike and time to expiry, and inspect intrinsic-gap behavior
        avg_mid_by_strike = (
            voucher_surface_summary
            .pivot(index="strike", columns="tte_days", values="mean_voucher_mid")
            .sort_index()
        )
        avg_gap_by_strike = (
            voucher_surface_summary
            .pivot(index="strike", columns="tte_days", values="mean_option_minus_intrinsic")
            .sort_index()
        )

        fig = px.imshow(
            avg_mid_by_strike,
            origin="lower",
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Average voucher mid by strike and time to expiry",
            labels={"x": "time to expiry (days)", "y": "strike", "color": "avg voucher mid"},
        )
        fig.show()

        fig = px.imshow(
            avg_gap_by_strike,
            origin="lower",
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Average voucher mid minus intrinsic value by strike and time to expiry",
            labels={"x": "time to expiry (days)", "y": "strike", "color": "avg mid - intrinsic"},
        )
        fig.show()

        strike_lines = px.line(
            voucher_surface_summary,
            x="strike",
            y="mean_voucher_mid",
            color="tte_days",
            markers=True,
            title="Average voucher mid by strike for each time-to-expiry bucket",
            hover_data={"day": True, "mean_voucher_mid": ":.2f"},
        )
        strike_lines.update_layout(legend_title_text="tte_days")
        strike_lines.show()
        """
    )
)

cells.append(
    code(
        """
        # Empirical voucher sensitivity: estimate how strongly each strike moves with the underlying
        sensitivity = option_sensitivity_summary(option_prices, prices.loc[prices["product"] == UNDERLYING]).copy()
        sensitivity["beta_bucket"] = pd.cut(
            sensitivity["beta_1tick"],
            bins=[-np.inf, 0.05, 0.15, 0.35, 0.6, np.inf],
            labels=["near_zero", "tiny", "small", "medium", "high"],
        )
        display(sensitivity)

        beta_fig = px.bar(
            sensitivity,
            x="strike",
            y="beta_1tick",
            color="corr_1tick",
            text="beta_1tick",
            title="Empirical 1-tick voucher beta vs underlying",
            color_continuous_scale="Blues",
            hover_data={
                "product": True,
                "corr_1tick": ":.3f",
                "mean_spread": ":.3f",
                "mean_option_minus_intrinsic": ":.3f",
                "negative_intrinsic_gap_share": ":.3f",
            },
        )
        beta_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        beta_fig.update_yaxes(title="beta to 1-tick underlying move")
        beta_fig.show()

        moneyness_scatter = px.scatter(
            sensitivity,
            x="strike",
            y="beta_1tick",
            size="mean_mid",
            color="mean_option_minus_intrinsic",
            title="Strike vs empirical beta, sized by average mid",
            hover_data={"product": True, "corr_1tick": ":.3f", "mean_mid": ":.2f"},
            color_continuous_scale="RdBu",
        )
        moneyness_scatter.show()
        """
    )
)

cells.append(
    code(
        """
        # Trade alignment: see which strikes actually trade and how trades print relative to quoted mids
        voucher_trades = (
            trade_with_quotes.loc[trade_with_quotes["product"].str.startswith(VOUCHER_PREFIX)]
            .assign(strike=lambda frame: frame["product"].map(extract_strike))
            .sort_values(["strike", "day", "timestamp"])
        )
        voucher_trade_summary = (
            voucher_trades.groupby("product")
            .agg(
                strike=("strike", "first"),
                trade_count=("product", "size"),
                mean_price=("price", "mean"),
                mean_quantity=("quantity", "mean"),
                mean_trade_minus_book_mid=("trade_minus_book_mid", "mean"),
                mean_abs_trade_minus_book_mid=("abs_trade_minus_book_mid", "mean"),
            )
            .round(4)
            .reset_index()
            .sort_values("strike")
        )
        display(voucher_trade_summary)

        trade_count_fig = px.bar(
            voucher_trade_summary,
            x="strike",
            y="trade_count",
            text="trade_count",
            title="Voucher trade activity by strike",
            hover_data={"mean_quantity": ":.2f", "mean_price": ":.2f", "mean_abs_trade_minus_book_mid": ":.3f"},
        )
        trade_count_fig.update_traces(textposition="outside")
        trade_count_fig.show()

        trade_edge_fig = px.bar(
            voucher_trade_summary,
            x="strike",
            y="mean_trade_minus_book_mid",
            title="Average trade price minus quoted mid by strike",
            hover_data={"trade_count": True, "mean_abs_trade_minus_book_mid": ":.3f"},
        )
        trade_edge_fig.add_hline(y=0, line_dash="dash", line_color="#111827")
        trade_edge_fig.show()
        """
    )
)

cells.append(
    code(
        """
        # Convert the notebook's measurements into practical strategy guidance
        hydrogel_reversion_core = hydrogel_reversion.loc[hydrogel_reversion["deviation_bucket"] == "(-10, 10]"]
        hydrogel_large_positive = hydrogel_reversion.loc[hydrogel_reversion["deviation_bucket"] == "> 40", "mean_future_return_5"].iloc[0]
        hydrogel_large_negative = hydrogel_reversion.loc[hydrogel_reversion["deviation_bucket"] == "<= -40", "mean_future_return_5"].iloc[0]

        extract_positive_share = float(extract_day_summary["positive_slope_share"].mean())
        extract_mean_change = float(extract_day_summary["intraday_change"].mean())
        hydrogel_informed_corr = float(
            informed_delta1_summary.loc[
                informed_delta1_summary["product"] == HYDROGEL,
                "best_imbalance_to_future5_corr",
            ].iloc[0]
        )
        extract_informed_corr = float(
            informed_delta1_summary.loc[
                informed_delta1_summary["product"] == UNDERLYING,
                "best_imbalance_to_future5_corr",
            ].iloc[0]
        )

        top_trade_strike = float(voucher_trade_summary.sort_values("trade_count", ascending=False)["strike"].iloc[0])
        max_beta_strike = float(sensitivity.sort_values("beta_1tick", ascending=False)["strike"].iloc[0])
        dead_strikes = sensitivity.loc[sensitivity["beta_1tick"].fillna(0) <= 0.01, "strike"].astype(int).tolist()

        commentary = f'''
        ## Strategy Translation

        - `HYDROGEL_PACK`: the data supports an anchored market-making / mean-reversion stance. When Hydrogel is far below the 10,000 anchor, subsequent 5-tick returns turn positive on average; when it is far above the anchor, those forward returns turn negative. That is a clean fit for inventory-aware quoting around a fixed fair.
        - `HYDROGEL_PACK` informed-flow check: the best-level imbalance signal is real but moderate, with future-5 correlation **{hydrogel_informed_corr:.3f}**. I would use it as a skew input on top of the fixed-fair strategy, not as the whole trading logic.
        - `VELVETFRUIT_EXTRACT`: the underlying carries the real directional signal. Its rolling slope is positive on average for **{extract_positive_share:.2%}** of the sample, with mean intraday change of **{extract_mean_change:,.2f}** ticks, and its best-level imbalance has future-5 correlation **{extract_informed_corr:.3f}**. That suggests a lighter-touch directional bias plus imbalance-based quote skew is more appropriate than hard mean reversion.
        - Vouchers: the most useful strikes are the middle of the ladder, not the tails. The highest empirical 1-tick sensitivity shows up around strike **{max_beta_strike:,.0f}**, while actual trade activity peaks around **{top_trade_strike:,.0f}**. Those are the best candidates for active quoting and relative-value checks.
        - Deep OTM vouchers are effectively pinned. Strikes **{dead_strikes}** show near-zero empirical beta, so they are poor vehicles for directional expression unless the market gifts us mispriced optionality.
        - One caution: the dataset contains many rows where quoted voucher mids sit below simple intrinsic value. I would treat `mid - intrinsic` as a market diagnostic rather than a strict no-arbitrage signal, and only lean on it when the discount is large and tradable in the actual book.
        '''
        display(Markdown(commentary))
        """
    )
)

cells.append(
    md(
        """
        ## Results

        The round 3 structure is encouraging:

        - `HYDROGEL_PACK` looks like the clean fixed-fair product in the set and should support an anchored quoting strategy.
        - `VELVETFRUIT_EXTRACT` behaves like the underlying state variable the voucher ladder reacts to.
        - Voucher prices are orderly by strike, their empirical sensitivity falls as strike rises, and trading concentrates in the middle strikes rather than the dead tail.

        The natural next step after this notebook is to convert these observations into a fill-aware trader:

        - mean-reverting market making in `HYDROGEL_PACK`
        - modest directional / inventory-aware trading in `VELVETFRUIT_EXTRACT`
        - voucher quoting focused on the liquid middle strikes plus simple cross-strike consistency checks
        """
    )
)

nb["cells"] = cells
NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Wrote notebook to {NOTEBOOK_PATH}")
