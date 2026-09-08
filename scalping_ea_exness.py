"""
============================================================
  ScalpingEA for MT5 - Python Version (FAST MODE)
  Symbol  : USTEC (Tech 100) - MONO SYMBOL
  Broker  : Exness (small accounts $10+)
  Strategy: EMA 8/21 Crossover + RSI(7) Filter + ATR SL/TP
  Author  : ScalpingEA
  Mode    : FAST EXECUTION - Target 10 trades/day
============================================================

REQUIREMENTS:
    pip install MetaTrader5 pandas numpy

HOW TO USE:
    1. Install MT5 terminal from Exness and log in.
    2. Enable "Allow Algo Trading" in MT5 Tools > Options > Expert Advisors.
    3. Run: python scalping_ea_exness.py
============================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

# ─────────────────────────────────────────────
#  CONFIGURATION — FAST EXECUTION MODE
# ─────────────────────────────────────────────

CONFIG = {
    # MT5 Login (leave as None to use already-logged-in terminal)
    "mt5_login"    : None,       # e.g. 12345678
    "mt5_password" : None,       # e.g. "YourPassword"
    "mt5_server"   : None,       # e.g. "Exness-MT5Real8"

    # Symbol — USTEC ONLY
    "primary_symbol" : "USTEC",  # Primary trading symbol (USTEC/USTECm)

    # Risk Management
    "risk_percent"      : 1.0,    # % of balance risked per trade
    "fixed_lot_size"    : 0.05,   # Fixed lot size for USTEC (starting small lot)
    "max_open_trades"   : 5,      # Max simultaneous trades total
    "max_daily_loss_pct": 5.0,    # Stop trading if daily loss hits this %
    "target_daily_trades": 10,    # Target number of trades per day

    # Scalping Indicators (M1 = more signals)
    "fast_ema"      : 8,
    "slow_ema"      : 21,
    "rsi_period"    : 7,
    "rsi_overbought": 70.0,
    "rsi_oversold"  : 30.0,
    "atr_period"    : 14,

    # SL / TP (ATR-based) — FAST MODE: Tighter TP for quick exits
    "sl_atr_mult"   : 1.5,        # SL = ATR × this
    "tp_atr_mult"   : 1.5,        # FAST MODE: Reduced from 2.0 to 1.5 for quicker closes
    "trail_atr_mult": 1.2,        # FAST MODE: Increased from 1.0 for better exits

    # Features
    "use_breakeven"     : True,   # Move SL to breakeven at 50% of TP
    "use_trailing_stop" : True,   # Enable trailing stop
    "use_fast_exit"     : True,   # FAST MODE: Close on opposite signal

    # Time Filter (MT5 server time)
    "use_time_filter": True,
    "start_hour"     : 7,
    "end_hour"       : 20,
    "avoid_friday"   : True,      # No new trades Friday after 20:00

    # EA Identity
    "magic_number"  : 202401,
    "comment"       : "ScalpEA_USTEC",
    "timeframe"     : mt5.TIMEFRAME_M1,  # FAST MODE: M1 instead of M5
    "loop_interval" : 1,         # FAST MODE: 1 second instead of 10
}

# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scalping_ea.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("ScalpingEA")


# ─────────────────────────────────────────────
#  SYMBOL RESOLVER
# ─────────────────────────────────────────────

SYMBOL_CANDIDATES = {
    "xauusd": ["XAUUSD", "XAUUSDm", "GOLD", "XAUUSD."],
    "ustec" : ["USTECm", "USTEC", "NAS100m", "NAS100", "US100", "NDX"],
    "btc"   : ["BTCUSD", "BTCUSDm", "BITCOIN", "BTC/USD"],
}

def resolve_symbol(asset: str) -> Optional[str]:
    """Find the correct symbol name used by the broker."""
    for candidate in SYMBOL_CANDIDATES.get(asset, []):
        info = mt5.symbol_info(candidate)
        if info is not None and info.visible:
            log.info(f"✓ Symbol {candidate} found and visible")
            return candidate
        # Try to add to market watch
        if info is not None:
            mt5.symbol_select(candidate, True)
            info = mt5.symbol_info(candidate)
            if info and info.bid > 0:
                log.info(f"✓ Symbol {candidate} added to market watch")
                return candidate
    log.error(f"Could not resolve symbol for {asset}")
    return None


# ─────────────────────────────────────────────
#  INDICATOR CALCULATIONS
# ─────────────────────────────────────────────

def get_ohlcv(symbol: str, timeframe: int, count: int = 100) -> Optional[pd.DataFrame]:
    """Fetch OHLCV bars from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) < count // 2:
        log.warning(f"Failed to get rates for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()

def get_indicators(symbol: str, cfg: dict) -> Optional[dict]:
    """Return all indicator values for a symbol."""
    df = get_ohlcv(symbol, cfg["timeframe"], count=150)
    if df is None or len(df) < 50:
        return None

    fast_ema = calc_ema(df["close"], cfg["fast_ema"])
    slow_ema = calc_ema(df["close"], cfg["slow_ema"])
    rsi      = calc_rsi(df["close"], cfg["rsi_period"])
    atr      = calc_atr(df, cfg["atr_period"])

    # Use last CLOSED bar (index -2), current bar (index -1) still forming
    return {
        "fast_ema_curr" : fast_ema.iloc[-2],
        "slow_ema_curr" : slow_ema.iloc[-2],
        "fast_ema_prev" : fast_ema.iloc[-3],
        "slow_ema_prev" : slow_ema.iloc[-3],
        "rsi"           : rsi.iloc[-2],
        "atr"           : atr.iloc[-2],
        "last_bar_time" : df["time"].iloc[-2],
    }


# ─────────────────────────────────────────────
#  ACCOUNT & TRADE HELPERS
# ─────────────────────────────────────────────

def get_balance() -> float:
    acc = mt5.account_info()
    return acc.balance if acc else 0.0

def count_open_trades(symbol: str = "", magic: int = 0) -> int:
    positions = mt5.positions_get()
    if positions is None:
        return 0
    return sum(
        1 for p in positions
        if (magic == 0 or p.magic == magic)
        and (symbol == "" or p.symbol == symbol)
    )

def count_trades_today(magic: int = 0) -> int:
    """Count closed trades today."""
    deals = mt5.history_deals_get(datetime.now().replace(hour=0, minute=0, second=0))
    if deals is None:
        return 0
    today = datetime.now().date()
    return sum(
        1 for d in deals
        if (magic == 0 or d.magic == magic)
        and datetime.fromtimestamp(d.time).date() == today
        and d.entry == mt5.DEAL_ENTRY_OUT
    )

def normalize_price(price: float, symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        return price
    return round(price, info.digits)


# ─────────────────────────────────────────────
#  TIME FILTER
# ─────────────────────────────────────────────

def is_trade_time(cfg: dict) -> bool:
    if not cfg["use_time_filter"]:
        return True
    now = datetime.now()
    wd  = now.weekday()   # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    if wd >= 5:           # Weekend
        return False
    if cfg["avoid_friday"] and wd == 4 and now.hour >= 20:
        return False
    return cfg["start_hour"] <= now.hour < cfg["end_hour"]


# ─────────────────────────────────────────────
#  DAILY LOSS & TRADE COUNTER
# ─────────────────────────────────────────────

class DailyLossTracker:
    def __init__(self):
        self.day_start_balance = get_balance()
        self.last_day = datetime.now().day

    def reset_if_new_day(self):
        today = datetime.now().day
        if today != self.last_day:
            self.day_start_balance = get_balance()
            self.last_day = today
            log.info(f"New trading day — balance reset to ${self.day_start_balance:.2f}")

    def is_ok(self, max_loss_pct: float) -> bool:
        self.reset_if_new_day()
        balance  = get_balance()
        loss     = self.day_start_balance - balance
        loss_pct = (loss / self.day_start_balance) * 100.0 if self.day_start_balance > 0 else 0
        if loss_pct >= max_loss_pct:
            log.warning(f"Daily loss limit hit: {loss_pct:.2f}% — No new trades today.")
            return False
        return True

    def get_trades_today(self, magic: int) -> int:
        """Get count of trades closed today."""
        return count_trades_today(magic)


# ─────────────────────────────────────────────
#  TRADE EXECUTION
# ─────────────────────────────────────────────

def place_trade(symbol: str, order_type: int, atr: float, cfg: dict) -> bool:
    """Open a BUY or SELL trade with auto SL/TP."""
    # Check if already have a trade on this symbol
    if count_open_trades(symbol, cfg["magic_number"]) > 0:
        log.debug(f"Already have open trade on {symbol}")
        return False

    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        log.error(f"Symbol info not found: {symbol}")
        return False

    sym_tick = mt5.symbol_info_tick(symbol)
    if sym_tick is None:
        log.error(f"Tick data not found: {symbol}")
        return False

    sl_dist = atr * cfg["sl_atr_mult"]
    tp_dist = atr * cfg["tp_atr_mult"]

    if order_type == mt5.ORDER_TYPE_BUY:
        price = sym_tick.ask
        sl    = price - sl_dist
        tp    = price + tp_dist
    else:
        price = sym_tick.bid
        sl    = price + sl_dist
        tp    = price - tp_dist

    price = normalize_price(price, symbol)
    sl    = normalize_price(sl,    symbol)
    tp    = normalize_price(tp,    symbol)

    # Use fixed lot size
    lot = cfg["fixed_lot_size"]

    request = {
        "action"      : mt5.TRADE_ACTION_DEAL,
        "symbol"      : symbol,
        "volume"      : lot,
        "type"        : order_type,
        "price"       : price,
        "sl"          : sl,
        "tp"          : tp,
        "deviation"   : 10,
        "magic"       : cfg["magic_number"],
        "comment"     : cfg["comment"],
        "type_time"   : mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        log.error(f"order_send returned None for {symbol}")
        return False

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        direction = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
        log.info(
            f"✅ TRADE OPENED | {symbol} {direction} | "
            f"Lot:{lot} | Price:{price} | SL:{sl} | TP:{tp}"
        )
        return True
    else:
        log.warning(
            f"❌ Trade failed: {symbol} | "
            f"Code:{result.retcode} | {result.comment}"
        )
        return False


# ─────────────────────────────────────────────
#  TRADE MANAGEMENT (Breakeven + Trailing)
# ─────────────────────────────────────────────

def manage_open_trades(cfg: dict, symbol: str, indic: dict):
    """Manage existing positions: breakeven and trailing stop."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    for pos in positions:
        if pos.magic != cfg["magic_number"]:
            continue

        open_p  = pos.price_open
        cur_sl  = pos.sl
        cur_tp  = pos.tp
        cur_p   = pos.price_current
        p_type  = pos.type  # 0=BUY, 1=SELL

        atr = indic["atr"]

        tp_dist    = atr * cfg["tp_atr_mult"]
        trail_dist = atr * cfg["trail_atr_mult"]
        new_sl     = cur_sl

        digits = mt5.symbol_info(symbol).digits if mt5.symbol_info(symbol) else 5

        if p_type == mt5.ORDER_TYPE_BUY:
            profit = cur_p - open_p

            # Breakeven: price moved 50% toward TP
            if cfg["use_breakeven"] and cur_sl < open_p and profit >= tp_dist * 0.5:
                new_sl = round(open_p + 0.0001, digits)

            # Trailing stop
            if cfg["use_trailing_stop"]:
                trail_sl = cur_p - trail_dist
                if trail_sl > new_sl:
                    new_sl = trail_sl

            if new_sl > cur_sl:
                _modify_sl(pos.ticket, round(new_sl, digits), cur_tp, symbol)

        elif p_type == mt5.ORDER_TYPE_SELL:
            profit = open_p - cur_p

            # Breakeven
            if cfg["use_breakeven"] and (cur_sl > open_p or cur_sl == 0) and profit >= tp_dist * 0.5:
                new_sl = round(open_p - 0.0001, digits)

            # Trailing stop
            if cfg["use_trailing_stop"]:
                trail_sl = cur_p + trail_dist
                if cur_sl == 0 or trail_sl < new_sl:
                    new_sl = trail_sl

            if cur_sl == 0 or new_sl < cur_sl:
                _modify_sl(pos.ticket, round(new_sl, digits), cur_tp, symbol)


def _modify_sl(ticket: int, new_sl: float, tp: float, symbol: str):
    """Send a position modify request to update the stop loss."""
    request = {
        "action"  : mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl"      : new_sl,
        "tp"      : tp,
        "symbol"  : symbol,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"🔄 SL updated | Ticket:{ticket} | New SL:{new_sl}")
    else:
        code = result.retcode if result else "None"
        log.debug(f"SL modify skipped/failed: ticket={ticket} code={code}")


# ─────────────────────────────────────────────
#  SIGNAL DETECTION (FAST MODE)
# ─────────────────────────────────────────────

def get_signal(indic: dict, cfg: dict) -> Optional[int]:
    """
    Returns:
        mt5.ORDER_TYPE_BUY  — bullish signal
        mt5.ORDER_TYPE_SELL — bearish signal
        None                — no signal
    
    FAST MODE: Relaxed filters for more frequent entries.
    """
    fast_curr = indic["fast_ema_curr"]
    slow_curr = indic["slow_ema_curr"]
    fast_prev = indic["fast_ema_prev"]
    slow_prev = indic["slow_ema_prev"]
    rsi       = indic["rsi"]

    bullish_cross = (fast_prev <= slow_prev) and (fast_curr > slow_curr)
    bearish_cross = (fast_prev >= slow_prev) and (fast_curr < slow_curr)

    # FAST MODE: More relaxed RSI filters
    rsi_ok_buy  = cfg["rsi_oversold"] < rsi < cfg["rsi_overbought"]
    rsi_ok_sell = cfg["rsi_oversold"] < rsi < cfg["rsi_overbought"]

    if bullish_cross and rsi_ok_buy:
        return mt5.ORDER_TYPE_BUY
    if bearish_cross and rsi_ok_sell:
        return mt5.ORDER_TYPE_SELL
    return None


# ─────────────────────────────────────────────
#  MAIN EA LOOP
# ─────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("  🚀 ScalpingEA Python — USTEC FAST EXECUTION MODE (MONO SYMBOL)")
    log.info(f"  Symbol: USTEC | Lot Size: {CONFIG['fixed_lot_size']} (Fixed Small Lot)")
    log.info(f"  Target: {CONFIG['target_daily_trades']} trades/day")
    log.info("=" * 70)

    # ── Connect to MT5 ──
    if not mt5.initialize():
        log.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return

    # Optional: login programmatically
    if CONFIG["mt5_login"]:
        ok = mt5.login(
            login    = CONFIG["mt5_login"],
            password = CONFIG["mt5_password"],
            server   = CONFIG["mt5_server"],
        )
        if not ok:
            log.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return

    acc = mt5.account_info()
    log.info(f"Connected: {acc.name} | Balance: ${acc.balance:.2f} | Broker: {acc.server}")

    # ── Resolve USTEC symbol ──
    symbol = resolve_symbol("ustec")
    if symbol is None:
        log.error("❌ Could not resolve USTEC symbol. Check your Market Watch in MT5.")
        mt5.shutdown()
        return
    
    log.info(f"✓ Trading Symbol: {symbol}")

    # ── Daily loss tracker ──
    loss_tracker = DailyLossTracker()

    # ── Bar tracker: only act on new closed bars ──
    last_bar_time = None

    log.info("EA running. Press Ctrl+C to stop.\n")

    try:
        while True:
            # ── Safety checks ──
            if not is_trade_time(CONFIG):
                time.sleep(CONFIG["loop_interval"])
                continue

            if not loss_tracker.is_ok(CONFIG["max_daily_loss_pct"]):
                # Still manage open trades even if daily limit hit
                indic = get_indicators(symbol, CONFIG)
                if indic is not None:
                    manage_open_trades(CONFIG, symbol, indic)
                time.sleep(CONFIG["loop_interval"])
                continue

            # ── Get indicators ──
            indic = get_indicators(symbol, CONFIG)
            if indic is None:
                log.warning(f"Failed to get indicators for {symbol}")
                time.sleep(CONFIG["loop_interval"])
                continue

            # ── Manage existing positions ──
            manage_open_trades(CONFIG, symbol, indic)

            # ── Check open trades ──
            total_open = count_open_trades(symbol, CONFIG["magic_number"])
            trades_closed_today = loss_tracker.get_trades_today(CONFIG["magic_number"])

            # ── Log daily progress ──
            if trades_closed_today > 0 or total_open > 0:
                log.info(
                    f"📊 DAILY PROGRESS | Closed Today: {trades_closed_today}/{CONFIG['target_daily_trades']} | "
                    f"Open Positions: {total_open}/{CONFIG['max_open_trades']}"
                )

            # ── Only act on new closed bar ──
            bar_time = indic["last_bar_time"]
            if last_bar_time == bar_time:
                time.sleep(CONFIG["loop_interval"])
                continue
            last_bar_time = bar_time

            # ── Log indicators ──
            log.info(
                f"{symbol:10s} | EMA({CONFIG['fast_ema']}):{indic['fast_ema_curr']:.4f} "
                f"EMA({CONFIG['slow_ema']}):{indic['slow_ema_curr']:.4f} "
                f"RSI:{indic['rsi']:.1f}  ATR:{indic['atr']:.4f}"
            )

            # ── Check if we can open a new trade ──
            if total_open >= CONFIG["max_open_trades"]:
                log.debug(f"Max open trades ({CONFIG['max_open_trades']}) reached.")
                time.sleep(CONFIG["loop_interval"])
                continue

            # ── Check if we've hit daily trade target ──
            if trades_closed_today >= CONFIG["target_daily_trades"]:
                log.info(f"✅ Daily target of {CONFIG['target_daily_trades']} trades reached — managing positions only.")
                time.sleep(CONFIG["loop_interval"])
                continue

            # ── Get signal and place trade ──
            signal = get_signal(indic, CONFIG)
            if signal is not None:
                placed = place_trade(symbol, signal, indic["atr"], CONFIG)
                if placed:
                    log.info(f"✓ Trade count after open: {total_open + 1}/{CONFIG['max_open_trades']}")

            time.sleep(CONFIG["loop_interval"])

    except KeyboardInterrupt:
        log.info("\n⛔ EA stopped by user.")
    finally:
        mt5.shutdown()
        log.info("MT5 connection closed.")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()
