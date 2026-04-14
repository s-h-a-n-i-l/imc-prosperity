# Trading Strategy

This document analyzes the strategy implementations found in the `sources/` repos.

It focuses on:

- what signal each implementation acts on,
- how the strategy executes,
- how it appears to have been tuned,
- what the code suggests about risk management and design choices,
- what is repo-level versus round-level versus experimental.

## Scope

Analyzed as trading implementations:

- `imc-prosperity-2`
- `imc-prosperity-2-1`
- `IMC-Prosperity-2023-Stanford-Cardinal`
- `imc-prosperity-3`
- `imc-prosperity-3-1`
- `imc-prosperity-3-2`
- `imc-prosperity-3-3`
- `imc-prosperity3-trading`

Not trading implementations, so only noted briefly:

- `IMC-Prosperity-2-Manual`
- `imc-prosperity-3-backtester`
- `imc-prosperity-3-visualizer`

## How To Read This

- `Signal`: what market information the strategy is reacting to.
- `Execution`: how orders are actually placed.
- `Tuning`: where parameters, thresholds, and fitted constants show up.
- `Takeaway`: the main implementation lesson from that repo.

## Repo Analysis

### `imc-prosperity-2`

Representative implementation files:

- `round1/round_1_v6.py`
- `round2/round_2_v3.py`
- `round3/round3_v2.py`
- `round4/round4_v4_roses.py`
- `round5/round5_v1.py`

#### Round 1

Signal:

- `AMETHYSTS`: fixed fair value at 10,000.
- `STARFRUIT`: market-maker mid / filtered large-volume book mid, then a slight mean-reversion adjustment using `reversion_beta`.

Execution:

- split into three stages: `take`, `clear`, `make`.
- immediately hit quotes better than fair by `take_width`.
- clear inventory near fair to restore capacity.
- market make with configurable edge rules like `disregard_edge`, `join_edge`, and `default_edge`.
- avoid adverse selection on starfruit by ignoring small-volume levels and using filtered book information.

Tuning:

- explicit parameter dictionary with take widths, clearing widths, adverse volume filters, join/default edge, soft position limits.
- starfruit fair-value logic is tuned through filtered-volume thresholds and the reversion coefficient.

Takeaway:

- this repo is a clean example of research being translated into reusable execution primitives.
- the main edge is not forecasting; it is fair-value estimation plus disciplined market making around that fair.

#### Round 2

Signal:

- `ORCHIDS`: local-versus-foreign conversion arbitrage rather than environmental-feature prediction.
- `STARFRUIT` and `AMETHYSTS`: continuation of round-1 fair-value market making.

Execution:

- orchids logic uses implied bid/ask from conversion economics.
- adaptive edge logic changes quoting aggressiveness based on observed fill volumes.
- the code clearly distinguishes taking obvious arb, then making around the implied foreign price.

Tuning:

- orchids parameters include `make_edge`, `make_min_edge`, `make_probability`, `volume_avg_timestamp`, `volume_bar`, `dec_edge_discount`, `step_size`.
- this strongly suggests online edge adaptation based on fill-rate feedback.

Takeaway:

- the implementation shows a shift from prediction to structural microstructure exploitation.
- tuning is mostly about quote placement and fill-quality adaptation, not feature engineering.

#### Round 3

Signal:

- basket spread / premium between `GIFT_BASKET` and synthetic constituents.
- rolling or regime-aware mean reversion on the spread.

Execution:

- basket spread is transformed into target positions.
- the repo experiments with Bollinger-band and modified z-score style logic.
- execution is conservative because realized slippage mattered.

Tuning:

- hardcoded mean premium and rolling-std style thresholds.
- multiple variants indicate threshold optimization and regime labeling were important.

Takeaway:

- the code supports the repo’s broader lesson: spread trading worked, but hedging every leg was fragile and easier to overfit.

#### Round 4

Signal:

- `COCONUT_COUPON`: implied-volatility deviation from rolling mean using Black-Scholes.
- `COCONUT`: mainly used as hedge via delta.
- `ORCHIDS`: hybrid of conversion arb plus predictive overlay from `GIFT_BASKET` returns in some versions.
- `ROSES`: added later as directional signal product in `round4_v4_roses.py`.

Execution:

- compute coupon IV, vega, delta, then trade coupon when vol z-score is large.
- hedge coupon position in `COCONUT` using delta-based target position.
- orchids implementation can switch between pure conversion arb behavior and target-position logic.

Tuning:

- z-score thresholds for coupon vol trading.
- rolling windows for past vol.
- `gift_basket_beta`, returns thresholds, and clearing thresholds for orchids overlay.

Takeaway:

- this is a strong example of a repo mixing structural and statistical signals in one codebase.
- it is also a warning that extra predictors can be layered onto a structural strategy without being the true core edge.

#### Round 5

Signal:

- named-trader flow, especially pairs like `Vladimir/Remy` and `Rihanna/Rhianna/Vinnie`, plus carryover from earlier rounds.
- coconut/coupon pricing continues from Black-Scholes logic.

Execution:

- trader-name events trigger direct directional trades on the relevant product.
- execution is intentionally simple because the signal itself is already strong.

Tuning:

- signal thresholds are often event-defined rather than continuous.
- some round-5 logic also contains overfit historical parameters inherited from earlier rounds.

Takeaway:

- this repo’s final round implementation shows that once trader IDs arrive, event detection can be more important than continuous modeling.

### `imc-prosperity-2-1`

Representative implementation files:

- `src/submissions/round1.py`
- `src/submissions/round2.py`
- `src/submissions/round3.py`
- `src/submissions/round4.py`
- `src/submissions/round5.py`

Signal:

- mostly structural signals consistent with the README:
- round 1 fixed-fair / wall-mid market making,
- round 2 orchids conversion arb,
- round 3 basket premium thresholds,
- round 4 coconut coupon fair value from Black-Scholes,
- round 5 named-trader directional signals.

Execution:

- this repo is simpler and more production-like than `imc-prosperity-2`.
- it tends to pick one strong edge per product and implement it directly instead of keeping as many experimental branches.
- for basket-style products, the code and analysis strongly imply preference for trading the main signal leg rather than every possible hedge leg.

Tuning:

- threshold choices are more static and repo-wide lessons suggest grid search and backtest comparison were used outside the final submission files.
- round 5 seems tuned through event selection rather than dense parameterization.

Takeaway:

- this repo is a good example of pruning complexity down to the strongest robust signals.
- compared with `imc-prosperity-2`, it feels more selective and less research-lab-like.

### `IMC-Prosperity-2023-Stanford-Cardinal`

Representative implementation file:

- `trader.py`

Signal:

- `PEARLS`: fixed fair around 10,000.
- `BANANAS`: linear regression on recent price history via `bananas_cache` and `calc_next_price_bananas`.
- `COCONUTS/PINA_COLADAS`: pair-trading around the `15/8` ratio.
- `PICNIC_BASKET`: premium over synthetic constituents.
- `DIVING_GEAR`: dolphin-sighting directional signal.
- `BERRIES`: time-of-day / schedule-based directional curve.
- `UKULELE` and `BERRIES`: Olivia-based copy trading in later logic.

Execution:

- product-specific handlers like `compute_orders_pearls`, `compute_orders_regression`, `compute_orders_c_and_pc`, `compute_orders_basket`, `compute_orders_dg`, `compute_orders_br`.
- execution style is generally aggressive once the product-specific condition is met.
- for fixed-fair and regression products, the code mixes taking favorable quotes and market making around an acceptable bid/ask band.

Tuning:

- regression coefficients are hardcoded.
- pair-trading ratio is hardcoded.
- basket threshold uses `basket_std`.
- multiple stateful flags control time-regime strategies.

Takeaway:

- this repo is a classic “many product-specific alphas in one trader” implementation.
- it is less abstracted than later repos, but very clear about mapping one signal family to one product family.

### `imc-prosperity-3`

Representative implementation file:

- `FrankfurtHedgehogs_polished.py`

Signal:

- `RAINFOREST_RESIN`: static fair.
- `KELP`: wall-mid-based dynamic fair.
- `PICNIC_BASKET1/2`: spread between basket and synthetic constituents, with running premium updates.
- `VOLCANIC_ROCK` and vouchers: mixed framework using IV, vega, smile logic, and mean reversion depending on product.
- `MAGNIFICENT_MACARONS`: conversion economics.
- `Olivia`: informed trader signal used to shift ETF thresholds and likely other directional logic.

Execution:

- very modular design with `ProductTrader` plus strategy-specific trader wrappers.
- basket implementation trades baskets first, then partial-hedges constituents via `ETF_HEDGE_FACTOR`.
- execution uses actual wall / book structure instead of theoretical mid only.
- closes positions at zero or near-zero edge when configured.

Tuning:

- explicit global constants:
- `BASKET_THRESHOLDS`,
- `INITIAL_ETF_PREMIUMS`,
- `ETF_THR_INFORMED_ADJS`,
- `ETF_HEDGE_FACTOR`,
- `THR_OPEN`, `THR_CLOSE`,
- `IV_SCALPING_THR`,
- mean-reversion windows for underlying and options.

Takeaway:

- this is one of the strongest examples of turning notebook research into a coordinated multi-product production trader.
- the implementation is notable because it combines:
- fair-value modeling,
- partial hedging,
- informed-trader overlays,
- and product-by-product strategy switching.

### `imc-prosperity-3-1`

Representative implementation files:

- `ROUND 1/final_round_1_trader.py`
- `ROUND 2/FINAL_FRENCH_GUY.py`
- `ROUND 3/big_volcano_man.py`
- `ROUND 4/algo run for round 4.py`
- `ROUND5/OLIVIA IS THE GOAT.py`

Signal:

- round-specific and very experimental.
- `ROUND5/OLIVIA IS THE GOAT.py` strongly centers:
- Olivia counterparty flow,
- basket trading with intentionally reduced size limits,
- Black-Scholes / implied volatility for vouchers.

Execution:

- this repo tends to implement narrow, round-specific strategies rather than one polished unified engine.
- round-5 file uses small hand-set caps like `BASKET1_LIMIT = 10`, `BASKET2_LIMIT = 10`, `SQUID_LIMIT = 15`, which implies deliberate throttling for risk control or execution realism.
- voucher logic computes IV, delta, gamma, vega and then trades with explicit option math.

Tuning:

- many parameters appear to have been tuned manually in notebooks and then hardcoded into round-specific files.
- reduced limits in the final round are an important tuning choice: they show the team preferred controlled exposure over max-limit aggression.

Takeaway:

- this repo is best read as a research-to-prototype repo.
- its implementations are informative because they show the intermediate stage between notebook discovery and a fully generalized production trader.

### `imc-prosperity-3-2`

Representative implementation file:

- `trader.py`

Signal:

- `RAINFOREST_RESIN`, `KELP`, `SQUID_INK`: fair-value / market-making plus mean-reversion logic.
- baskets: synthetic fair values using hardcoded linear coefficients and intercepts.
- vouchers: implied vol by strike plus rolling-vol mean logic and cross-voucher arb checks.
- `MAGNIFICENT_MACARONS`: implied import/export prices plus local/foreign arb.
- Olivia copying for `SQUID_INK` and `CROISSANTS`.

Execution:

- this trader mixes several execution styles in one class:
- standard make/take for simple products,
- basket arb using synthetic order depths,
- Macaron conversion `take`, `make`, and `clear`,
- voucher directional trading and optional close-out logic,
- direct copy-trading of Olivia.

Tuning:

- basket coefficients are explicitly set, including both regression-derived and hand-fixed formula versions.
- product activation flags show that the team chose a selective trading universe rather than trading everything.
- widths, timespans, volatility windows, arbitrage thresholds, and conversion config values are all hardcoded.

Notable design choice:

- this code is unusually willing to mix multiple hypotheses in one trader: stat arb, direct arb, options pricing, and copy trading.

Takeaway:

- this repo is a hybrid system.
- its strongest lesson is that the final implementation does not need one philosophy as long as each product has a clean local logic.

### `imc-prosperity-3-3`

Representative implementation files:

- `submissions/round1.py`
- `submissions/round2.py`
- `submissions/round3.py`
- `submissions/round4.py`
- `submissions/round5.py`

#### Round 1

Signal:

- resin fixed fair.
- kelp VWAP / max-volume fair.
- squid based on short mid-history and running-pressure style inference.

Execution:

- resin and kelp use clean make/take/clear logic.
- squid uses directional signal logic with explicit target-volume behavior.

Tuning:

- short rolling windows and hand-tuned take widths.

#### Round 2

Signal:

- basket index arb via `basket.mid - synthetic`.
- `SQUID_INK` uses a custom online logistic regressor with features like z-score, velocity, acceleration, spread, order-book imbalance, running pressure, and MACD.
- `JAMS` has its own directional logic.

Execution:

- basket trades are simple threshold-based basket-side execution.
- squid is more sophisticated: infer action from model probability, then size into a soft position limit.

Tuning:

- squid parameters include learning window, MACD windows, epochs, learning rate, and volatility-sensitive decision thresholds.
- basket thresholds are hardcoded around values like `48` and `59`.

#### Round 3

Signal:

- vouchers `10000` and `10250` traded via implied-volatility z-score versus mean vol.
- squid still retained as a standalone strategy.
- baskets remain threshold / mean-reversion products.

Execution:

- voucher implementation is much more direct than some other repos:
- calculate IV,
- compare to mean volatility,
- trade when z-score exceeds threshold.

Tuning:

- `mean_volatility`, `std_window`, and `zscore_threshold` are all per-voucher.

#### Round 4

Signal:

- round-4 file keeps the voucher system and adds `MAGNIFICENT_MACARONS`.
- macarons logic appears to be more conservative than the option logic.

Execution:

- options continue to be treated as quantifiable fair-value instruments.
- conversions are integrated but not overcomplicated.

#### Round 5

Signal:

- continuation of vouchers plus trader-ID logic.
- code architecture suggests strong reuse rather than complete rewrite.

Execution:

- this repo favors explicit, reusable strategy modules over a giant monolithic handler.

Takeaway:

- `imc-prosperity-3-3` is one of the clearest “round evolution” repos.
- you can see signals becoming more complex while the execution scaffolding stays reusable.

### `imc-prosperity3-trading`

Representative implementation file:

- `Round_5/ROUND5_SUBMISSION.py`

Signal:

- round-1 to round-5 notebooks indicate:
- resin fixed fair,
- kelp wall-mid / max-volume fair,
- squid slow-EMA reversion,
- baskets via synthetic spread and inter-basket relationships,
- vouchers via IV smile / parabola / ATM-IV tracking,
- macarons via CSI / sunlight regime plus conversion economics,
- round-5 trader IDs, especially Olivia.

Execution:

- the final code appears to reflect the notebook-heavy style of the repo:
- signal discovery first,
- direct implementation second.
- compared with `imc-prosperity-3`, execution seems more tied to the research narrative of each round rather than one uniform abstraction layer.

Tuning:

- the notebooks show threshold tuning, EMA selection, and strike-specific voucher handling.
- this repo is especially explicit about not treating all mispricings the same:
- some are traded as reversion,
- some as informed-flow events,
- some as vol deviations.

Takeaway:

- this repo is a good bridge between explanation and implementation.
- its main strength is that the code closely follows the research logic, making the signal-to-execution mapping easier to interpret.

## Non-Trading Repos

### `IMC-Prosperity-2-Manual`

- this repo contains manual-round optimization notebooks, not deployable algorithmic traders.
- still useful for understanding their optimization style: brute force, Monte Carlo, equilibrium reasoning, and constrained optimization.

### `imc-prosperity-3-backtester`

- tooling repo, not strategy repo.
- still important because backtester assumptions shape how strategies were tuned and validated.

### `imc-prosperity-3-visualizer`

- tooling / visualization repo.
- not a strategy implementation, but very relevant to how several teams discovered signals.

## Cross-Repo Patterns

### Signals That Show Up Repeatedly

- fixed fair-value mean reversion / market making,
- wall-mid or max-volume fair estimation,
- basket-minus-synthetic premium,
- Black-Scholes implied-volatility deviation,
- conversion implied bid/ask arbitrage,
- informed-trader / named-counterparty following.

### Execution Patterns That Show Up Repeatedly

- separate `take`, `clear`, and `make` phases,
- use of inventory-clearing trades to restore capacity,
- basket-first then hedge-constituents execution,
- partial hedging instead of full theoretical hedging,
- selective universe activation rather than trading every product,
- layered execution when following counterparty signals.

### Tuning Patterns That Show Up Repeatedly

- parameter dictionaries at top of file,
- hardcoded coefficients from notebook research,
- grid-searched thresholds,
- rolling-window sizes chosen empirically,
- risk throttles via soft position limits or reduced custom limits,
- per-product strategy selection instead of one global optimizer.

## Highest-Signal Takeaways

1. The best implementations are usually reacting to structural signals, not trying to predict everything.
2. Execution logic is often as important as signal logic.
3. Several strong repos deliberately avoid trading every leg of a theoretically correct hedge.
4. Named-trader signals become extremely important once IDs are revealed.
5. The most mature repos separate reusable execution primitives from product-specific signal generation.
6. The most experimental repos are still very useful because they show how ideas were tuned before being simplified.

## If You Were Choosing What To Study First

For clean production-style implementations:

- `imc-prosperity-3/FrankfurtHedgehogs_polished.py`
- `imc-prosperity-2/round1/round_1_v6.py`
- `imc-prosperity-2/round2/round_2_v3.py`
- `imc-prosperity-2/round4/round4_v4_roses.py`

For round-by-round evolution:

- `imc-prosperity-3-3/submissions/round1.py` through `round5.py`

For research-to-code translation:

- `imc-prosperity3-trading`
- `imc-prosperity-3-1`

For classic older multi-product logic:

- `IMC-Prosperity-2023-Stanford-Cardinal/trader.py`
