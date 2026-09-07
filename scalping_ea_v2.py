#!/usr/bin/env python3
"""
MT5 scalper skeleton for XAUUSD, BTCUSD, USTEC using MetaTrader5 Python API.
- EMA crossover + RSI filter
- ATR-based SL/TP
- Risk-based lot calculation (best-effort using symbol tick_value), fallback to fixed lot
- Designed as a starting point: backtest and tune heavily before live usage.

Requirements:
    pip install MetaTrader5 pandas numpy

Run:
    - Start MT5 desktop, log into Exness account, enable automated trading / allow DLLs if needed.
    - Run: python mt5_scalper.py
"""

import time
import math
import logging
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# === USER CONFIGURATION ===
SYMBOLS = ["XAUUSD", "BTCUSD", "USTEC"]   # Confirm exact symbol names in MT5 Market Watch
TIMEFRAME = mt5.TIMEFRAME_M1              # M1 scalping
FAST_EMA = 5
SLOW_EMA = 21
RSI_PERIOD = 14
RSI_MIN = 35      # lower bound for buys
RSI_MAX = 70      # upper bound for sells
ATR_PERIOD = 14
ATR_MULT = 1.2    # SL = ATR * ATR_MULT
TP_ATR_MULT = 1.5 # TP distance in ATR units
RISK_PER_TRADE = 0.02  # 2% of account balance per trade (adjust with caution)
MIN_LOT = 0.01
MAX_POS_PER_SYMBOL = 1
CHECK_INTERVAL = 10  # seconds between checks
DEVIATION = 20       # allowed slippage in points
MAGIC = 123456
LOG_LEVEL = logging.INFO

# === Logging ===
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")


# === Indicator helpers ===
def compute_indicators(df):
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=FAST_EMA, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=SLOW_EMA, adjust=False).mean()
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    ma_down = down.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = ma_up / ma_down
    df['rsi'] = 100 - (100 / (1 + rs))
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    return df


# === MT5 helpers ===
def mt5_connect():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed, error code: {mt5.last_error()}")
    logging.info("MT5 initialized")
    account_info = mt5.account_info()
    if account_info is None:
        raise RuntimeError("No account info available; is MT5 logged in?")
    logging.info(f"Logged in as: {account_info.login}, balance={account_info.balance}, leverage={account_info.leverage}")
    return account_info


def get_rates(symbol, timeframe, n=200):
    utc_from = datetime.now() - timedelta(minutes=n * (1 if timeframe == mt5.TIMEFRAME_M1 else 5))
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, n)
    if rates is None:
        raise RuntimeError(f"Failed to fetch rates for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    # convert time
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def get_symbol_info_or_raise(symbol):
    si = mt5.symbol_info(symbol)
    if si is None:
        raise RuntimeError(f"Symbol {symbol} not found in Market Watch. Add it to Market Watch and confirm the name.")
    return si


def calculate_lot(symbol, stop_loss_price_diff, account_balance):
    """
    Try to compute lot based on RISK_PER_TRADE using symbol tick value if available.
    stop_loss_price_diff is in price units (price - SL price absolute).
    If not possible, return MIN_LOT.
    """
    si = mt5.symbol_info(symbol)
    if si is None:
        return MIN_LOT
    # Risk amount in account currency
    risk_amount = account_balance * RISK_PER_TRADE
    # Try to get tick value per 1 lot (tick_value is value of one tick for 1 lot)
    tick_value = getattr(si, "trade_tick_value", None) or getattr(si, "trade_tick_value", None)
    point = si.point
    if tick_value and point and stop_loss_price_diff > 0:
        # Number of points in SL
        points = abs(stop_loss_price_diff) / point
        # Value per point for 1 lot: tick_value / tick_size might be needed; we assume trade_tick_value corresponds to 1 point per lot
        # Some brokers report trade_tick_value as tick value for tick_size; we still attempt:
        value_per_point_per_lot = tick_value
        # Compute volume:
        lot = risk_amount / (points * value_per_point_per_lot)
        # clamp and round to symbol's volume step
        lot = max(lot, MIN_LOT)
        # Round to lot step
        step = si.volume_step if si.volume_step else 0.01
        lot = math.floor(lot / step) * step
        if lot < MIN_LOT:
            return MIN_LOT
        return round(lot, 2)
    else:
        logging.warning(f"Could not compute lot via tick_value for {symbol}; using MIN_LOT fallback")
        return MIN_LOT


def place_order(symbol, order_type, volume, price, sl, tp, deviation=DEVIATION, comment=None):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": float(price),
        "sl": float(sl) if sl else 0.0,
        "tp": float(tp) if tp else 0.0,
        "deviation": deviation,
        "magic": MAGIC,
        "comment": comment or "py_scalper",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None:
        logging.error(f"order_send returned None for {symbol}: {mt5.last_error()}")
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Order failed for {symbol}: retcode={result.retcode}, comment={result.comment}")
    else:
        logging.info(f"Order placed for {symbol}: ticket={result.order}, volume={volume}, sl={sl}, tp={tp}")
    return result


# === Trading logic ===
def evaluate_and_maybe_trade(symbol, account_balance):
    try:
        si = get_symbol_info_or_raise(symbol)
        if not si.visible:
            logging.info(f"Symbol {symbol} not visible; trying to add to Market Watch")
            mt5.symbol_select(symbol, True)

        df = get_rates(symbol, TIMEFRAME, n=300)
        df = compute_indicators(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # quick checks: no existing positions > limit
        positions = mt5.positions_get(symbol=symbol)
        if positions is not None and len(positions) >= MAX_POS_PER_SYMBOL:
            logging.debug(f"{symbol}: existing positions {len(positions)} >= limit {MAX_POS_PER_SYMBOL}")
            return

        ask = mt5.symbol_info_tick(symbol).ask
        bid = mt5.symbol_info_tick(symbol).bid
        price = ask if True else bid

        # EMA crossover detection (simple)
        ema_cross_up = (prev['ema_fast'] <= prev['ema_slow']) and (last['ema_fast'] > last['ema_slow'])
        ema_cross_down = (prev['ema_fast'] >= prev['ema_slow']) and (last['ema_fast'] < last['ema_slow'])

        # RSI filter
        rsi = last['rsi']
        atr = last['atr']
        if atr <= 0 or math.isnan(atr):
            logging.debug(f"{symbol}: ATR invalid, skipping")
            return

        if ema_cross_up and rsi > RSI_MIN and rsi < 90:
            # Buy
            sl_price = price - ATR_MULT * atr
            tp_price = price + TP_ATR_MULT * atr
            stop_diff = abs(price - sl_price)
            lot = calculate_lot(symbol, stop_diff, account_balance)
            logging.info(f"{symbol} BUY signal: price={price:.5f}, sl={sl_price:.5f}, tp={tp_price:.5f}, lot={lot}")
            place_order(symbol, mt5.ORDER_TYPE_BUY, lot, ask, sl_price, tp_price)
        elif ema_cross_down and rsi < RSI_MAX and rsi > 10:
            # Sell
            price = bid
            sl_price = price + ATR_MULT * atr
            tp_price = price - TP_ATR_MULT * atr
            stop_diff = abs(price - sl_price)
            lot = calculate_lot(symbol, stop_diff, account_balance)
            logging.info(f"{symbol} SELL signal: price={price:.5f}, sl={sl_price:.5f}, tp={tp_price:.5f}, lot={lot}")
            place_order(symbol, mt5.ORDER_TYPE_SELL, lot, bid, sl_price, tp_price)
        else:
            logging.debug(f"{symbol}: no signal (ema_up={ema_cross_up}, ema_down={ema_cross_down}, rsi={rsi:.2f})")
    except Exception as e:
        logging.exception(f"Error evaluating {symbol}: {e}")


def main_loop():
    account_info = mt5_connect()
    while True:
        try:
            # Refresh account info
            account_info = mt5.account_info()
            balance = account_info.balance
            for sym in SYMBOLS:
                evaluate_and_maybe_trade(sym, balance)
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            logging.info("Interrupted by user, exiting")
            break
        except Exception:
            logging.exception("Unexpected error in main loop")
            time.sleep(10)
    mt5.shutdown()


if __name__ == "__main__":
    main_loop()