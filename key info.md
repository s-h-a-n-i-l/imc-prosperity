# Key Info

This document consolidates the main knowledge in `sources/` that matters for:

- understanding the competition setup and market structure,
- deciding what research to do next,
- applying data science effectively,
- choosing trading techniques,
- selecting strategies by product and round.

It is being built from the READMEs and research notebooks across the repository and will emphasize durable insights over repo-specific implementation details.

## How To Use This Document

- Read the `Research Workflow` section first.
- Use `Strategy Playbook` when designing or revising a trader.
- Use `Round/Product Research Priorities` to decide where to spend notebook time.
- Treat repo-specific notes as supporting evidence, not as unquestioned truth.

## In Progress

This file is currently being populated from:

- Prosperity 2 repos and manual notebooks,
- Prosperity 3 repos, notebooks, and backtester docs,
- round-by-round strategy notes,
- manual trading research where it reveals state-space structure, equilibrium behavior, or exploitable constraints.

## Core Mental Model

Across the repos, the most important repeated lesson is that Prosperity is not primarily a forecasting contest. It is a market microstructure contest with some forecasting subproblems layered on top.

What consistently matters most:

- infer the simulator's fair-value logic,
- infer which fills are real opportunities versus queue traps,
- exploit structural relationships between related products,
- manage inventory so you keep risk capacity for the next edge,
- avoid overfitting when there are only a few days of data,
- use execution quality to convert a known signal into actual PnL.

The best repositories repeatedly avoid unnecessarily complex prediction models unless the product truly demands one. They usually win by:

- estimating fair value better than the crowd,
- measuring deviation from fair value,
- mapping deviation to position or quote placement,
- using inventory-aware exits,
- backtesting honestly against fill assumptions and limit constraints.

## Research Workflow

This is the research workflow implied by the strongest READMEs and notebooks in `sources/`.

### 1. Start With Market Structure, Not ML

Before fitting anything:

- determine position limits,
- determine whether the product has a stable fair price, slow moving fair price, mean reversion, or cross-asset dependence,
- determine whether the book itself reveals fair value better than trades do,
- determine whether there are conversion constraints, fees, storage costs, or trader-identity signals,
- determine whether you should trade the underlying, the derivative, or only one side of a pair.

Questions to answer first:

- What defines fair value?
- What creates temporary mispricing?
- What determines fill probability?
- What creates inventory risk?
- Is edge structural, statistical, or signal-driven?

### 2. Build The Right Derived Series

The notebooks repeatedly create these derived features:

- `mid_price`,
- wall mid / max-volume mid,
- synthetic basket value,
- basket premium or spread,
- implied bid / implied ask for conversion products,
- implied volatility for option-like products,
- moneyness and time-to-expiry,
- rolling mean, EMA, rolling std, z-score,
- lagged returns, lead returns, spread change, regime labels,
- trader-conditioned signal series in Round 5.

If you do not compute these cleanly, later analysis becomes noisy and misleading.

### 3. Visualize Before Optimizing

The repos emphasize dashboards, scatter plots of order book levels, spread charts, IV smiles, and trader overlays because visual inspection reveals:

- queue traps,
- discrete price bands,
- mean-reversion regions,
- persistent premium levels,
- whether fills happen at one level or several nearby levels,
- whether a named trader is early, lucky, or actually informed.

Use plots to answer:

- Is this stationary enough for z-scores?
- Are thresholds symmetric?
- Does the spread cluster around multiple regimes?
- Does a trader systematically buy lows / sell highs?

### 4. Only Then Tune Thresholds

Threshold tuning is useful only after the signal definition is stable. The notebooks use:

- manual threshold sweeps,
- grid search,
- rolling-band strategies,
- inverse-CDF position mapping,
- conservative parameter selection when sample size is tiny.

Common pattern:

- define signal,
- define entry threshold,
- define softer exit threshold,
- test inventory clearing rules,
- test layered execution versus single-price execution,
- prefer robustness over peak backtest PnL.

### 5. Treat Backtests As Approximate

The backtester README and writeups make this very clear:

- matching assumptions matter,
- order depth gets priority over market trades,
- trade matching can be optimistic or pessimistic depending on configuration,
- limits are enforced before fills,
- conversions may be unsupported in some tooling,
- end-of-round data and submission data can differ.

So your research should include:

- strategy PnL under multiple fill assumptions,
- sensitivity to threshold changes,
- sensitivity to entry/exit lag,
- sensitivity to order placement depth,
- stress tests for inventory getting stuck.

## Data Science Approach

The right data science for this codebase is pragmatic, not flashy.

### Use Data Science To Answer These 5 Questions

1. What is fair value now?
2. How far is the market from fair value?
3. Is the deviation likely to revert, trend, or persist?
4. What size should I trade?
5. At which price levels should I execute?

### Preferred Techniques

- descriptive statistics and market microstructure analysis,
- rolling features and regime detection,
- z-score and band-based mean reversion,
- EMA/SMA smoothing for noisy fair values,
- linear regression or ridge regression for synthetic pricing only when justified,
- Kalman filtering when hedge ratios may drift,
- Black-Scholes inversion plus IV-surface fitting for voucher products,
- Monte Carlo / simulation for manual rounds,
- game-theoretic fixed-point / Nash-style reasoning for crowd-choice puzzles,
- counterparty-conditioned event studies in Round 5.

### Techniques To Use Carefully

- full ML pipelines on tiny datasets,
- highly parameterized predictors,
- deep models without structural justification,
- strategies whose PnL disappears under slightly different fill assumptions,
- aggressive cross-product hedging when one leg historically dominates profits.

The repeated warning from the repos is simple: if there are only a few days of data, complexity often hurts more than it helps.

## Repeated Cross-Year Lessons

Several themes recur across Prosperity 1, 2, and 3 material in `sources/`.

### Fair Value Often Hides In The Book

Older repositories repeatedly discovered that:

- a large market maker or max-volume level can define the simulator's internal fair,
- naive midpoint can be noisier than the right book-derived mid,
- verifying fair-value logic via simple hold-one-unit experiments is extremely valuable.

This is the origin of the "wall mid" / "popular mid" / "max-volume mid" idea that shows up again in Prosperity 3.

### Structural Alpha Beats Feature Fishing

Prosperity 2 ORCHIDS research is especially instructive:

- many feature-correlation and regression attempts on sunlight, humidity, and tariffs were weak or spurious,
- the robust edge came from understanding the market mechanics and conversion economics,
- once the structural local-versus-foreign arbitrage was identified, it dominated the predictive-feature approach.

General rule:

- if a product has clear structural pricing constraints, exploit those before trying predictive ML.

### Hedge Only If The Hedge Helps

Across gift baskets, coconuts/coupons, and Prosperity 3 baskets/vouchers:

- theoretical delta neutrality is not automatically optimal,
- position limits, execution friction, and leg asymmetry can make partial or no hedge better,
- often one leg carries most of the signal and most of the PnL.

Research implication:

- always compare `signal leg only` versus `fully hedged` versus `partially hedged`.

### Simple Models Need Better Execution, Not More Complexity

Counterparty-signal strategies in older and newer repos show the same thing:

- once a signal is publicly inferable, the remaining edge comes from execution,
- layered quoting, inventory awareness, and selecting the right leg often matter more than marginal model improvements.

## Strategy Archetypes

### 1. Fixed-Fair Market Making

Use when:

- true price is effectively constant,
- market takers regularly cross that fair value,
- adverse selection is negligible.

Main products:

- `RAINFOREST_RESIN`,
- historically analogous products like Prosperity 1 pearls / Prosperity 2 amethysts.

Execution rules:

- immediately take obviously favorable quotes,
- otherwise improve the best quote by one tick if that still leaves positive edge,
- clear inventory at or near fair value when needed,
- avoid sitting behind queue if queue priority makes fills unlikely.

### 2. Dynamic-Fair Market Making

Use when:

- price moves slowly,
- short-horizon prediction is weak,
- current order book reveals best fair estimate.

Main products:

- `KELP`,
- analogous products like bananas / starfruit when prediction is weak.

Execution rules:

- estimate fair via wall mid or max-volume mid,
- make and take around that fair,
- add inventory-aware skew,
- do not overcomplicate with regression unless it demonstrably adds edge.

### 3. Mean Reversion Around A Smoothed Baseline

Use when:

- product experiences bursts away from a slow mean,
- jumps are often retraced,
- pure market making is too noisy.

Main products:

- `SQUID_INK`,
- sometimes `VOLCANIC_ROCK`,
- some basket spreads.

Techniques:

- EMA / rolling mean,
- threshold or z-score trigger,
- separate entry and exit thresholds,
- conservative sizing because jump risk is real.

### 4. Basket / Synthetic Statistical Arbitrage

Use when:

- one product is a deterministic combination of others,
- spread around synthetic value is stationary or regime-like.

Main products:

- `PICNIC_BASKET1`,
- `PICNIC_BASKET2`,
- older gift-basket style products from Prosperity 2.

Core formulas:

- `PB1 = 6*CROISSANTS + 3*JAMS + 1*DJEMBES`
- `PB2 = 4*CROISSANTS + 2*JAMS`
- also analyze cross-basket relationships, e.g. `2*PB1 - 3*PB2 - 2*DJEMBES`

Techniques:

- compute synthetic mid, bid, and ask,
- analyze premium distribution and stationarity,
- trade premium z-score or threshold bands,
- consider EMA-smoothed synthetic values,
- sometimes prefer trading only the basket side if component hedges add more risk than value,
- consider dynamic hedge ratios only if coefficients appear unstable.

### 5. Derivative / Volatility Arbitrage

Use when:

- products are option-like,
- fair value depends on underlying, strike, and time-to-expiry,
- volatility mispricing is more stable than raw price mispricing.

Main products:

- `VOLCANIC_ROCK_VOUCHER_9500`
- `VOLCANIC_ROCK_VOUCHER_9750`
- `VOLCANIC_ROCK_VOUCHER_10000`
- `VOLCANIC_ROCK_VOUCHER_10250`
- `VOLCANIC_ROCK_VOUCHER_10500`

Techniques:

- compute Black-Scholes fair values,
- invert to implied volatility,
- plot IV versus moneyness,
- fit a parabola to the smile,
- track fitted ATM IV through time,
- only trade IV mean reversion where vega is meaningful,
- use moneyness to decide whether a voucher behaves more like the underlying or more like a volatility instrument.

### 6. Conversion / Location Arbitrage

Use when:

- local and external venue prices are linked by fees and tariffs,
- conversion creates bounded arbitrage opportunities,
- storage costs or position asymmetry matter.

Main products:

- `MAGNIFICENT_MACARONS`,
- historical ORCHIDS-like analogs in Prosperity 2.

Techniques:

- compute implied import and export prices,
- trade only when local bid exceeds external adjusted ask or vice versa,
- include transport fees, tariffs, and storage cost explicitly,
- be careful with conversion size limits,
- favor the leg with clearer positive edge if one direction is rarely profitable.

### 7. Counterparty-Signal Trading

Use when:

- counterparty IDs are visible,
- certain traders repeatedly buy lows / sell highs,
- the signal horizon is short and execution-sensitive.

Main products:

- Round 5 `SQUID_INK`,
- `CROISSANTS`,
- `KELP`,
- sometimes baskets or specific vouchers depending on trader.

Techniques:

- label trades by counterparty,
- measure mark-to-market PnL by trader and product,
- test whether a trader buys near rolling lows / sells near rolling highs,
- condition on product and trade direction,
- react immediately with layered orders across nearby levels rather than one blunt quote.

## Product Playbook

### RAINFOREST_RESIN

What research to do:

- verify fair price is fixed at 10,000,
- verify queue-joining versus pennying / scalping economics,
- study irrational fills when best quotes cross fair value.

Best techniques:

- fixed-fair arbitrage,
- one-tick quote improvement,
- inventory clearing at fair.

Avoid:

- predictive modeling,
- fancy inventory models unless they improve risk usage.

### KELP

What research to do:

- estimate fair via wall mid or max-volume bid/ask midpoint,
- test whether any short-horizon forecasting beats current fair,
- inspect top-layer depth skew and gap patterns,
- compare joining, pennying, and crossing.

Best techniques:

- dynamic-fair market making,
- order-book gap arbitrage,
- inventory-aware quote skew,
- fallback market making when no special pattern fires.

Avoid:

- overcommitting to linear regression without strong evidence.

### SQUID_INK

What research to do:

- characterize jump frequency and average reversion speed,
- compare raw mid, EMA, and rolling-average baselines,
- test whether named-trader flow dominates pure price-based edge in Round 5.

Best techniques:

- mean reversion with smoothed baseline,
- conservative position sizing,
- signal following from informed counterparties when available.

Avoid:

- passive market making if volatility swamps spread capture,
- stubbornly trading it if it remains structurally unprofitable.

### CROISSANTS / JAMS / DJEMBES / PICNIC_BASKET1 / PICNIC_BASKET2

What research to do:

- test basket-vs-synthetic spread stationarity,
- test cross-basket spread stationarity,
- identify whether one side of the arb drives most of the PnL,
- test direct threshold rules versus z-score rules,
- test whether smoothing synthetic value improves execution.

Best techniques:

- basket premium trading,
- cross-basket arbitrage,
- ridge / linear regression only as a hedge-ratio diagnostic,
- Kalman filter only if hedge ratios drift materially,
- layered order placement around likely fill levels.

Important repo-level insight:

- several strong teams found that trading only the basket leg can outperform full multi-leg hedging once fill quality and noise are accounted for.
- older Prosperity 2 gift-basket research also found stable premium centers around a nonzero constant rather than zero, so always estimate the premium mean empirically.

### VOLCANIC_ROCK and VOUCHERS

What research to do:

- compute IV per timestamp and strike,
- plot IV smile versus moneyness,
- fit parabola and extract ATM IV path,
- classify each voucher by moneyness over time,
- estimate where vega is high enough to justify vol trading,
- test mean reversion directly on deep ITM / OTM instruments if they mostly co-move with rock.

Best techniques:

- IV z-score trading for near-ATM vouchers,
- rock-like mean reversion for deep ITM vouchers,
- selective product universe as vouchers drift from ATM to OTM,
- avoid trading strikes whose liquidity, vega, or stability is poor.

Important repo-level insight:

- not all vouchers should share one strategy; strategy must depend on moneyness and vega.

### MAGNIFICENT_MACARONS

What research to do:

- compute adjusted import/export economics,
- identify sunlight threshold / CSI regime behavior,
- measure storage-cost drag on long inventory,
- test whether the reverse conversion leg is ever worth it after export tariffs.

Best techniques:

- conservative one-leg arbitrage when local bid is clearly rich to external adjusted ask,
- regime-aware trading only if sunlight threshold behavior is clearly validated,
- extremely small and disciplined inventory.

Avoid:

- holding longs casually because storage cost compounds every timestamp,
- trusting unstable regression models without strong out-of-sample evidence.

### Legacy Analog: ORCHIDS

Older Prosperity 2 ORCHIDS research is worth treating as a template for Macarons-like work:

- compute foreign implied bid/ask after fees,
- distinguish executable arb from noisy environmental variables,
- adapt quoted edge based on observed fill rate,
- understand whether you can convert and re-open exposure in the same iteration.

## Execution Principles

The notebooks and writeups repeatedly reinforce these rules:

- Execution quality matters as much as signal quality.
- Layered orders often beat single-price orders.
- Quote placement should depend on queue position and historical fill behavior.
- When position limits bind, freeing inventory is alpha because it restores optionality.
- A strategy with high paper edge but poor fills is not a real strategy.

Good execution research:

- measure fill rate by distance from touch,
- measure expected edge by quote level,
- test all-in aggressive hit versus layered passive posting,
- compare basket-only execution to full hedge execution,
- log every reason an order was placed.

Extra lesson from older repos:

- if a product has one dominant, highly reliable execution path, do not dilute it with speculative side strategies unless they survive robustness tests.

## Backtesting Checklist

- Verify position-limit handling.
- Verify whether market trades are matched optimistically or pessimistically.
- Compare merged and per-day PnL.
- Re-run with alternate thresholds to test robustness.
- Check whether PnL is concentrated in a tiny number of events.
- Check if end-of-day inventory assumptions matter.
- For conversion products, verify the backtester supports the mechanics you rely on.
- Prefer strategies that survive small perturbations.

## Manual-Round Research Principles

The manual notebooks are useful because they model uncertainty and crowd behavior cleanly.

Repeated themes:

- brute force is fine when the search space is small,
- Monte Carlo is useful when distributions are explicit,
- game theory matters when payout depends on what others choose,
- expected value must include crowd dilution and opening costs,
- optimization under nonlinear fees is often the real problem.

By manual round type:

- FX / path search: brute-force paths and maximize product of rates.
- Container / suitcase rounds: simulate crowd choices, estimate equilibrium or mixed strategies, compare one-pick vs multi-pick after fees.
- Reserve-price rounds: simulate reserve distributions and optimize bid pairs against assumed market average.
- News-trading rounds: map view on each asset into an optimization problem with transaction fee schedule.

Useful mathematical patterns from the manual notebooks:

- replace raw intuition with an explicit objective function whenever possible,
- solve discrete search spaces exactly by brute force if they are small,
- use simulation only when an exact solution is cumbersome or crowd behavior is uncertain,
- for crowd-allocation problems, study profitability thresholds first before doing full equilibrium simulation,
- for fee-based portfolio allocation, formulate the problem as constrained optimization instead of guessing weights.

Concrete examples repeated in the notebooks:

- reserve-price rounds: expected-profit maximization over bid pairs,
- FX rounds: exact path enumeration,
- crowd-choice rounds: expected value after dilution and fees,
- news rounds: integer portfolio optimization under quadratic trading costs.

## Practical Research Priorities

If you were continuing work from these repos, the highest-value next steps would be:

1. Build one unified research pipeline that computes fair values, spreads, z-scores, IV, conversion edges, and trader-conditioned event studies from raw price/trade files.
2. Standardize robust backtests with multiple fill models, not just one optimistic assumption.
3. Add strategy diagnostics that decompose PnL into signal quality, execution quality, and inventory cost.
4. For baskets, compare basket-only, full-hedge, and partial-hedge implementations on identical data.
5. For vouchers, classify instruments by moneyness regime and switch strategy family accordingly.
6. For Round 5 data, rank counterparties by predictive quality per product and reaction horizon.
7. For conversion products, explicitly simulate storage-cost accumulation and limited conversion throughput.
8. For all products, test whether the simulator's internal mark price is closer to naive mid, wall mid, or another book-derived fair.
9. Add side-by-side diagnostics for queue-joining, pennying, and aggressive taking.

## Common Mistakes To Avoid

- Treating every product as a forecasting problem.
- Using one strategy family for all vouchers.
- Joining queues when pennying or taking is clearly better.
- Ignoring inventory-clearing logic.
- Using raw spread instead of executable spread.
- Assuming synthetic hedges help when they may just add noise.
- Overfitting thresholds to three days of data.
- Believing a backtest without challenging its fill assumptions.
- Trading structurally difficult products just because they are available.
- Assuming all environmental features are useful just because the game provides them.
- Forcing full hedges because they look theoretically cleaner.
- Confusing signal discovery with profit capture; execution is often the missing step.
