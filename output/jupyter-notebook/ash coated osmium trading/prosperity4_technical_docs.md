# IMC Prosperity 4 Technical Notes

## Source note
The public Prosperity 4 wiki link appears to return a 404 from outside the platform login flow. These notes therefore combine:
- official IMC Prosperity pages for the confirmed competition structure;
- public community mirrors of the starter API/data model;
- public open-source Prosperity backtesters for local replay mechanics.

## 1) Officially confirmed challenge structure
Prosperity 4 is a team-based virtual trading simulation with 5 trading rounds.
Each round has:
- 1 algorithmic challenge;
- 1 manual challenge.

Teams upload a Python program for the algorithmic side. Manual and algorithmic results are evaluated independently. At the end of each round, the trading engine evaluates submissions and the round PnL is recorded on the leaderboard.

The tutorial round exists specifically so you can practice programming your first Python program before the scored rounds.

## 2) What your trader is expected to do
The public starter-code mirrors consistently show the same basic interface:
- you implement a class called `Trader`;
- the engine calls `Trader.run(state)` repeatedly;
- `state` contains the current market snapshot and your current internal state;
- your function returns orders to send, plus (in newer interfaces) a conversion request and a `traderData` string for persistent state.

### Core state objects seen in public mirrors
Common mirrored fields in `TradingState`:
- `traderData`: string persisted from your previous invocation;
- `timestamp`;
- `listings`;
- `order_depths`;
- `own_trades`;
- `market_trades`;
- `position`;
- `observations`.

Common mirrored objects:
- `Order(symbol, price, quantity)`;
- `OrderDepth.buy_orders` and `OrderDepth.sell_orders`, usually maps from price to volume;
- `Trade(symbol, price, quantity, buyer, seller, timestamp)`.

In newer Prosperity-style mirrors, `observations` can also contain conversion-related observations, not just plain numeric observations.

## 3) How local backtesting works in practice
A local backtest replays historical timestamps from a round/day dataset.
At each timestamp the backtester:
1. constructs the `TradingState`;
2. calls your `Trader.run(state)`;
3. simulates matching of your returned orders;
4. updates positions/trades/PnL;
5. advances to the next timestamp.

This is the core loop you should assume when building your own research harness as well.

## 4) Widely used Prosperity 4 backtester behaviour (unofficial)
A commonly used open-source tool for Prosperity 4 is `prosperity4btest`.

### Installation and basic usage
```bash
pip install -U prosperity4btest
prosperity4btest your_trader.py 0
prosperity4btest your_trader.py 1 --merge-pnl
prosperity4btest your_trader.py 1-0 --print
```

### Matching model used by that backtester
Its documented matching logic is:
- your orders are first matched against the order book (`order_depths`) for that timestamp;
- if the order can be filled completely from the order book, market trades are not used;
- otherwise it can match against the timestamp's market trades;
- market-trade matching can be configured with `--match-trades all|worse|none`.

The same backtester also documents two important details:
- limits are enforced before matching, so an order set that would breach the configured limit can be cancelled;
- fills are clamped during matching so the position never exceeds the configured limit.

### Current limitations of that backtester
- conversions are not supported;
- it sets `PROSPERITY4BT_ROUND` and `PROSPERITY4BT_DAY` during local runs, but those variables do not exist in the official submission environment;
- known product limits are only bundled for some currently known rounds, and the tool falls back to defaults for unknown products unless you override them.

## 5) Recommended research/backtest workflow
### A. Data understanding
For each product:
- plot mid-price;
- inspect spread;
- inspect book imbalance;
- inspect traded volume by timestamp;
- inspect whether the series looks stationary, drifting, basket-linked, derivative-linked, or observation-linked.

### B. Separate fair value from execution
Build your strategy in layers:
1. fair value / signal model;
2. quoting / taking logic;
3. inventory control;
4. risk limits;
5. product-level and portfolio-level sizing.

Do not mix all of that into one giant if-statement if you can avoid it.

### C. Backtest under multiple fill assumptions
For the same strategy, test at least:
- optimistic fill assumptions;
- book-only or conservative assumptions;
- day-by-day PnL, not only merged PnL.

That matters because unofficial backtesters are approximations, not the official engine.

### D. Track more than final PnL
Track at minimum:
- final PnL;
- PnL by product;
- max long / short inventory;
- turnover;
- number of fills;
- adverse-selection behaviour after fills;
- contribution of passive vs aggressive fills.

## 6) What the official docs do and do not tell you
Officially confirmed:
- competition structure;
- algorithmic/manual separation;
- end-of-round engine evaluation;
- leaderboard based on round PnL;
- tutorial round for practice.

Not publicly specified in the official pages I could access:
- the exact fill model used by the production evaluator;
- the exact order-priority rules of the official engine;
- whether public historical data is identical to final evaluation data for every round;
- any full public specification of conversion handling for Prosperity 4.

So any local backtester should be treated as a calibration tool, not as a ground-truth oracle.

## 7) Practical engineering guidance
- Keep strategy state compact and serialisable in `traderData`.
- Never depend on local-only environment variables in submitted code.
- Keep round/product configuration explicit so new products can be added quickly.
- Treat position limits as hard constraints and add softer internal limits before the hard cap.
- Build a switchable execution model so you can compare passive quoting versus aggressive taking.
- Make it easy to run per-product ablations.
- Save research outputs separately from submission code.

## 8) A sensible mental model for Prosperity
Treat the challenge as three linked problems:
1. infer fair value from partial market information;
2. decide how aggressively to express that view;
3. survive inventory and regime changes long enough for the edge to realise.

That framing is more useful than thinking of the task as “just make PnL go up”.

## 9) Short version: how to backtest well
1. Start from the tutorial data.
2. Build one product at a time.
3. Replay data timestamp by timestamp.
4. Verify that your simulated positions and fills make sense.
5. Compare multiple matching assumptions.
6. Validate day-by-day, not only merged.
7. Keep submission code environment-agnostic.
8. Expect each round to introduce new mechanics, so optimise for adaptation speed.
