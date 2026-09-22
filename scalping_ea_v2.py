#!/usr/bin/env python3
"""Demo-first MT5 scalper. See VALIDATION.md before running.
Preserves EMA/RSI entry rules and ATR exits; adds fail-closed execution guards.
No profitability or broker-latency guarantee. Run only one instance per magic.
"""
import time
import math
import logging
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import pandas as pd

SYMBOLS = ["XAUUSD", "BTCUSD", "USTEC"]
TIMEFRAME = mt5.TIMEFRAME_M1
FAST_EMA = 5
SLOW_EMA = 21
RSI_PERIOD = 14
RSI_MIN = 35
RSI_MAX = 70
ATR_PERIOD = 14
ATR_MULT = 1.2
TP_ATR_MULT = 1.5
RISK_PER_TRADE = 0.02
MIN_LOT = 0.01
MAX_POS_PER_SYMBOL = 1
CHECK_INTERVAL = 1.0
DEVIATION = 20
MAGIC = 123456
LOG_LEVEL = logging.INFO
POST_LOSS_COOLDOWN_SECONDS = 900
HISTORY_LOOKBACK_DAYS = 30
MAX_TICK_AGE_SECONDS = 5
MAX_SPREAD_ATR_RATIO = 0.15
RECOVERY_SPREAD_ATR_RATIO = 0.10
RECOVERY_MIN_EMA_GAP_ATR = 0.10
RECOVERY_MIN_ATR_RATIO = 0.5
RECOVERY_MAX_ATR_RATIO = 2.0

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
_last_bar = {}
_blocked_symbols = set()


def compute_indicators(df):
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=FAST_EMA, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=SLOW_EMA, adjust=False).mean()
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    ma_down = down.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = ma_up / ma_down
    df['rsi'] = 100 - (100 / (1 + rs))
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - df['close'].shift()).abs(),
                    (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    return df


def mt5_connect():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    account = mt5.account_info()
    if account is None:
        mt5.shutdown()
        raise RuntimeError("No account info; check MT5 login")
    return account


def get_rates(symbol, timeframe, n=200):
    # Start at bar 1: latest CLOSED bars, never the developing candle.
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, n)
    if rates is None or len(rates) < n:
        raise RuntimeError(f"Insufficient closed bars for {symbol}")
    df = pd.DataFrame(rates).sort_values('time').reset_index(drop=True)
    if df['time'].duplicated().any():
        raise RuntimeError("Duplicate bar timestamps")
    for column in ('open', 'high', 'low', 'close'):
        if not df[column].map(lambda x: math.isfinite(x) and x > 0).all():
            raise RuntimeError("Invalid OHLC data")
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    duration = 60 if timeframe == mt5.TIMEFRAME_M1 else 300
    age = (datetime.now(timezone.utc) - df.iloc[-1]['time']).total_seconds()
    if age < duration or age > duration * 3:
        raise RuntimeError("Closed bars are stale or incomplete")
    return df


def get_symbol_info_or_raise(symbol):
    si = mt5.symbol_info(symbol)
    if si is None:
        raise RuntimeError(f"Symbol {symbol} not found in Market Watch")
    return si


def calculate_lot(symbol, stop_loss_price_diff, account_balance):
    """Fail closed; never raise volume to the broker minimum beyond budget."""
    si = mt5.symbol_info(symbol)
    if si is None:
        return 0.0
    tick_value = getattr(si, 'trade_tick_value_loss', 0) or si.trade_tick_value
    values = (stop_loss_price_diff, account_balance, tick_value,
              si.trade_tick_size, si.volume_step, si.volume_min, si.volume_max)
    if not all(math.isfinite(x) and x > 0 for x in values):
        return 0.0
    if not 0 < RISK_PER_TRADE <= 1:
        return 0.0
    budget = account_balance * RISK_PER_TRADE
    loss_per_lot = stop_loss_price_diff / si.trade_tick_size * tick_value
    raw = min(budget / loss_per_lot, si.volume_max)
    lot = round(math.floor(raw / si.volume_step) * si.volume_step, 8)
    if lot < max(MIN_LOT, si.volume_min) or lot * loss_per_lot > budget:
        return 0.0
    return lot


def latest_loss_time(symbol):
    """Reconstruct recovery state from the latest owned completed position.
    Include entry/exit costs and manual closes. Mixed-magic positions block.
    A profitable completed position clears recovery; break-even does not.
    History window is explicitly bounded; see VALIDATION.md.
    """
    now = datetime.now(timezone.utc)
    deals = mt5.history_deals_get(now - timedelta(days=HISTORY_LOOKBACK_DAYS), now)
    if deals is None:
        raise RuntimeError("Deal history unavailable; entries blocked")
    owned = {d.position_id for d in deals if d.symbol == symbol and d.magic == MAGIC}
    exits = [d for d in deals if d.symbol == symbol and d.position_id in owned
             and d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY)]
    # Walk backwards through break-even positions to retain the last loss.
    visited = set()
    for closing in sorted(exits, key=lambda d: (d.time_msc, d.ticket), reverse=True):
        if closing.position_id in visited:
            continue
        visited.add(closing.position_id)
        full = mt5.history_deals_get(position=closing.position_id)
        if not full:
            raise RuntimeError("Position history unavailable")
        entries = [d for d in full if d.entry == mt5.DEAL_ENTRY_IN]
        if not entries or any(d.magic != MAGIC for d in entries):
            raise RuntimeError("Mixed or incomplete position ownership")
        # Reversals/netting mixing require manual reconciliation.
        allowed = (mt5.DEAL_ENTRY_IN, mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY)
        if any(d.entry not in allowed for d in full):
            raise RuntimeError("Unsupported position reversal in history")
        opened = sum(d.volume for d in entries)
        closed = sum(d.volume for d in full if d.entry != mt5.DEAL_ENTRY_IN)
        if not math.isclose(opened, closed, abs_tol=1e-8):
            raise RuntimeError("Position not fully closed or history incomplete")
        net = sum(d.profit + d.commission + d.swap + getattr(d, 'fee', 0) for d in full)
        if not math.isfinite(net):
            raise RuntimeError("Invalid deal P/L")
        if net < -1e-8:
            return closing.time_msc / 1000.0
        if net > 1e-8:
            return None
    return None


def recovery_checks(symbol, direction, df, spread):
    """Defined technical reassessment, not predictive AI or news analysis."""
    last = df.iloc[-1]
    atr = last['atr']
    baseline = df['atr'].iloc[-21:-1].median()
    if not math.isfinite(baseline) or baseline <= 0:
        return False
    ratio = atr / baseline
    if not RECOVERY_MIN_ATR_RATIO <= ratio <= RECOVERY_MAX_ATR_RATIO:
        return False
    if spread > atr * RECOVERY_SPREAD_ATR_RATIO:
        return False
    if direction * (last['ema_fast'] - last['ema_slow']) < atr * RECOVERY_MIN_EMA_GAP_ATR:
        return False
    if not (50 < last['rsi'] < 70 if direction == 1 else 30 < last['rsi'] < 50):
        return False
    higher = compute_indicators(get_rates(symbol, mt5.TIMEFRAME_M5, 200))
    recent = higher.iloc[-3:]
    if not ((recent['ema_fast'] - recent['ema_slow']) * direction > 0).all():
        return False
    return direction * (higher.iloc[-1]['ema_slow'] - higher.iloc[-3]['ema_slow']) > 0


def fresh_tick(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError("No tick available")
    if not all(math.isfinite(v) and v > 0 for v in (tick.ask, tick.bid)) or tick.ask < tick.bid:
        raise RuntimeError("Invalid quote")
    age = time.time() - tick.time_msc / 1000.0
    if not 0 <= age <= MAX_TICK_AGE_SECONDS:
        raise RuntimeError("Stale or future tick")
    return tick


def place_order(symbol, order_type, volume, price, sl, tp, deviation=DEVIATION, comment=None):
    # Single synchronous submission. Never blindly retry an uncertain outcome.
    if symbol in _blocked_symbols:
        return None
    si = get_symbol_info_or_raise(symbol)
    if not all(math.isfinite(v) and v > 0 for v in (volume, price, sl, tp)):
        return None
    if not si.volume_min <= volume <= si.volume_max:
        return None
    if si.volume_step <= 0 or not math.isclose(volume / si.volume_step, round(volume / si.volume_step), abs_tol=1e-7):
        return None
    tick = fresh_tick(symbol)
    current = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    if order_type not in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL):
        return None
    if abs(current - price) > si.point * deviation:
        return None
    # Reject shifted quotes instead of widening stops or silently increasing risk.
    minimum = si.trade_stops_level * si.point
    if order_type == mt5.ORDER_TYPE_BUY:
        valid = sl < tick.bid and tp > tick.ask and tick.bid - sl >= minimum and tp - tick.bid >= minimum
    else:
        valid = sl > tick.ask and tp < tick.bid and sl - tick.ask >= minimum and tick.ask - tp >= minimum
    if not valid:
        return None
    # SYMBOL_FILLING_MODE uses flags, not ORDER_FILLING enum values.
    if si.filling_mode & 1:
        filling = mt5.ORDER_FILLING_FOK
    elif si.filling_mode & 2:
        filling = mt5.ORDER_FILLING_IOC
    else:
        logging.warning("%s: no supported FOK/IOC filling policy", symbol)
        return None
    request = {'action': mt5.TRADE_ACTION_DEAL, 'symbol': symbol,
               'volume': float(volume), 'type': order_type, 'price': float(price),
               'sl': float(sl), 'tp': float(tp), 'deviation': deviation,
               'magic': MAGIC, 'comment': comment or 'py_scalper',
               'type_time': mt5.ORDER_TIME_GTC, 'type_filling': filling}
    check = mt5.order_check(request)
    if check is None or check.retcode != 0:
        logging.warning("%s: preflight rejected", symbol)
        return None
    _blocked_symbols.add(symbol)
    start = time.perf_counter()
    try:
        result = mt5.order_send(request)
    except Exception:
        logging.exception("%s: uncertain submission; reconcile manually before restart", symbol)
        return None
    logging.info("%s: submission latency %.1f ms", symbol, (time.perf_counter() - start) * 1000)
    if result is not None and result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
        _blocked_symbols.discard(symbol)
        logging.info("%s: filled, retcode=%s volume=%s", symbol, result.retcode, result.volume)
    else:
        logging.error("%s: unconfirmed/rejected submission; entries blocked until manual reconciliation", symbol)
    return result


def evaluate_and_maybe_trade(symbol, account_balance):
    try:
        if symbol in _blocked_symbols:
            return
        si = get_symbol_info_or_raise(symbol)
        if not si.visible and not mt5.symbol_select(symbol, True):
            return
        # Keep the original account-wide symbol limit; also block pending orders.
        positions = mt5.positions_get(symbol=symbol)
        pending = mt5.orders_get(symbol=symbol)
        if positions is None or pending is None or pending or len(positions) >= MAX_POS_PER_SYMBOL:
            return
        tick = fresh_tick(symbol)
        bars = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 1, 1)
        if bars is None or len(bars) != 1:
            return
        stamp = int(bars[0]['time'])
        if _last_bar.get(symbol) == stamp:
            return
        df = compute_indicators(get_rates(symbol, TIMEFRAME, 300))
        if int(df.iloc[-1]['time'].timestamp()) != stamp:
            return  # A candle closed during retrieval; retry with one consistent snapshot.
        _last_bar[symbol] = stamp
        last, prev = df.iloc[-1], df.iloc[-2]
        values = [last[k] for k in ('ema_fast', 'ema_slow', 'rsi', 'atr')]
        if not all(math.isfinite(v) for v in values) or last['atr'] <= 0:
            return
        up = prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']
        down = prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']
        direction = 1 if up and RSI_MIN < last['rsi'] < 90 else -1 if down and 10 < last['rsi'] < RSI_MAX else 0
        if not direction:
            return
        spread = tick.ask - tick.bid
        if spread > last['atr'] * MAX_SPREAD_ATR_RATIO:
            return
        loss_time = latest_loss_time(symbol)
        if loss_time is not None:
            if time.time() - loss_time < POST_LOSS_COOLDOWN_SECONDS:
                logging.info("%s: post-loss cooldown active", symbol)
                return
            # Require at least three wholly post-loss M1 candles.
            if df.iloc[-3]['time'].timestamp() <= loss_time:
                return
            if not recovery_checks(symbol, direction, df, spread):
                logging.info("%s: post-loss reassessment rejected entry", symbol)
                return
        # Refresh quote after potentially expensive history/M5 reads.
        tick = fresh_tick(symbol)
        limit = RECOVERY_SPREAD_ATR_RATIO if loss_time is not None else MAX_SPREAD_ATR_RATIO
        if tick.ask - tick.bid > last['atr'] * limit:
            return
        price = tick.ask if direction == 1 else tick.bid
        step = si.trade_tick_size
        if not math.isfinite(step) or step <= 0:
            return
        sl_raw = price - direction * ATR_MULT * last['atr']
        tp_raw = price + direction * TP_ATR_MULT * last['atr']
        # Round stop away from entry; size using the resulting actual distance.
        sl = round((math.floor(sl_raw / step) if direction == 1 else math.ceil(sl_raw / step)) * step, si.digits)
        tp = round((math.floor(tp_raw / step) if direction == 1 else math.ceil(tp_raw / step)) * step, si.digits)
        lot = calculate_lot(symbol, abs(price - sl), account_balance)
        if lot <= 0:
            return
        # Recheck account-wide symbol exposure immediately before submission.
        positions = mt5.positions_get(symbol=symbol)
        pending = mt5.orders_get(symbol=symbol)
        if positions is None or pending is None or pending or len(positions) >= MAX_POS_PER_SYMBOL:
            return
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        place_order(symbol, order_type, lot, price, sl, tp)
    except Exception:
        logging.exception("%s: evaluation failed; no entry", symbol)


def main_loop():
    mt5_connect()
    try:
        while True:
            try:
                account = mt5.account_info()
                if account is None or not math.isfinite(account.balance) or account.balance <= 0:
                    raise RuntimeError("Invalid account state")
                for symbol in SYMBOLS:
                    evaluate_and_maybe_trade(symbol, account.balance)
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                break
            except Exception:
                logging.exception("Main loop failed; pausing entries")
                time.sleep(10)
    finally:
        mt5.shutdown()


if __name__ == '__main__':
    main_loop()
