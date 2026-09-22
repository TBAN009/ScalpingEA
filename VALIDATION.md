# Execution and post-loss safeguards — validation required

## Status and scope

This branch is a partial implementation, not a live-ready or proven-profitable release. Only `scalping_ea_v2.py` is changed so far. `ScalpingEA.mq5`, `ScalpingEA.py`, and `scalping_ea_exness.py` remain unchanged. The two larger Python files could not be fully read through the available connector response, so they were not edited. The MQL5 port remains pending separate tests and MetaEditor compilation.

No live trades are authorized by these commits. Nothing is merged to master. The Deriv repository is untouched. No unit tests, lint, type checking, backtests, or terminal compilation have been executed in this chat; the connector supports repository operations, not a Python/MT5 execution environment.

## Behavior changes in version 2

The original EMA crossover/RSI conditions, ATR smoothing, configured risk percentage, symbol names, public function signatures, and ATR stop/target multipliers remain. These defaults are not optimized or recommended risk settings.

Latest closed bars now come from `copy_rates_from_pos(..., 1, n)`. Previously the timestamp-based request could retrieve older bars. This correction changes signals and requires fresh baseline comparison. The loop polls once per second rather than ten seconds, but full indicator work occurs once per newly closed M1 candle. It does not use unfinished-candle signals, increase order frequency intentionally, or guarantee faster fills. A signal is considered only once per bar; a rejected signal waits for a new bar/crossover.

After a completed EA-owned losing position, the default cooldown is 900 seconds. Net result includes entry and exit commission, swap, and deal fees. Manual exits of identified owned positions are included. New entries subsequently require three wholly post-loss M1 candles, the original crossover signal, bounded ATR relative to its preceding 20-bar median, stricter spread/ATR, EMA separation, directional RSI, and three aligned closed M5 trend bars with a directional slow-EMA slope. These are technical filters, not comprehensive news/fundamental analysis. Thresholds are provisional and must be validated out of sample.

Recovery continues until a completed profitable position is observed. Break-even does not clear it. State is reconstructed from broker deal history over a configurable 30-day lookback on each eligible signal; losses before that window are not retained. Increase the window or implement tested persistent state before claiming indefinite restart recovery. Missing history and mixed-magic netting/reversal histories block entries. Separate broker charges not attached to position deals are not included.

Sizing uses tick size and loss tick value, respects broker volume step/limits, and skips entries when the minimum volume exceeds the configured price-risk budget. It no longer falls back to an unconditional minimum lot. Commission, slippage, gaps, and currency conversion changes can increase actual loss beyond the sizing budget. There is no added daily/equity kill switch in this partial version.

Order preflight checks and broker-compatible FOK/IOC selection replace unconditional IOC. Unsupported fill modes skip entry. Quote validation, pending-order checks, and account-wide per-symbol position caps fail closed. Orders are submitted synchronously once. Uncertain, rejected, or non-final outcomes block that symbol in memory until an operator reconciles broker orders/positions and restarts. The block is not durable across crashes: never automatically restart after an uncertain submission without reconciliation. Do not run multiple instances with the same magic or combine this EA with other EAs on the same symbol in a netting account.

Version 2 did not have trailing stops; none were added. Broker-side SL/TP already attached to open positions remain in force while new entries are paused, subject to broker execution and market gap risks. This does not promise continuous active management if the program/terminal is offline.

## Offline validation — run before any terminal connection

Use a clean checkout of this branch, not the live deployment directory. The tests replace MetaTrader5 with a mock and do not connect to a terminal.

```bash
python -m pip install pandas numpy ruff mypy pandas-stubs
python -m unittest discover -s tests -v
python -m py_compile scalping_ea_v2.py tests/test_scalping_eas.py
python -m ruff check --select E9,F63,F7,F82 scalping_ea_v2.py tests/test_scalping_eas.py
python -m mypy --ignore-missing-imports scalping_ea_v2.py
```

Record the commands, Python/dependency versions, and complete outputs. Resolve every failure before proceeding. Type-check results may require explicit MT5 stubs or annotations; no passing result is claimed. Characterization tests were committed separately before the production change; those tests can also be run at their original commit to check baseline behavior.

Tests cover import isolation, indicator input preservation, missing symbol/initialization failures, tick-size sizing, invalid risk inputs, minimum-volume rejection, failed position/history queries, closed-bar retrieval, entry commissions/manual exits, uncertain submission blocking, preflight rejection, stale ticks, excessive recovery spread, and active cooldown. Tests do not establish profitability, broker compatibility, latency, or complete lifecycle correctness.

## Required integration and performance validation

Use a demo account with the exact broker symbol specifications. Verify buy/sell prices, tick-grid stops, minimum-stop distance, FOK/IOC, margin rejection, partial fills, close-by exits, manual closes, netting rejection, market closure, missing history, disconnections, and restart after a losing position. Confirm all post-loss gates pass in a controlled positive case as well as fail in negative cases. Add fixtures for profitable reset, break-even retention, partial closures, and mixed ownership before merging.

Replay identical historical data for the unmodified and modified Python strategy with spread, commission, swap, realistic slippage, and execution delays. MetaTrader Strategy Tester does not directly execute this Python bot; use a validated Python replay harness for strategy comparisons, then forward-test via the MT5 demo terminal. The native MQL5 EA requires its own MetaEditor compilation and Strategy Tester runs when ported.

Use separate training/validation/out-of-sample periods and multiple volatility regimes. Report net expectancy, profit factor, maximum drawdown, trade count, exposure, costs, and confidence intervals. Do not optimize to win rate alone or select cooldown thresholds using the final test period. Compare p50/p95 signal-to-submit and submit-to-confirm latency; fewer local calculations do not establish lower broker execution latency.

Do not merge or deploy to live trading until code review, automated tests, broker integration, out-of-sample analysis, and a sufficient demo forward test have passed. Roll back by using the unchanged master version in a separate environment, with trading disabled during transition and broker positions/orders reconciled first.
