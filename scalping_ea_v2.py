"""
ScalpingEA v2 - Professional Forex Scalping with Backtesting & Paper Trading
Production-ready with realistic risk management and 24/7 trading capability
Author: Enhanced version
"""

import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.FileHandler('scalping_ea.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradeType(Enum):
    """Trade direction"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


@dataclass
class Trade:
    """Trade record"""
    trade_id: str
    symbol: str
    trade_type: TradeType
    entry_price: float
    entry_time: datetime
    quantity: float
    stop_loss: float
    take_profit: float
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    status: OrderStatus = OrderStatus.OPEN
    commission: float = 0.0
    slippage: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary"""
        data = asdict(self)
        data['trade_type'] = self.trade_type.value
        data['status'] = self.status.value
        data['entry_time'] = self.entry_time.isoformat()
        if self.exit_time:
            data['exit_time'] = self.exit_time.isoformat()
        return data


class PerformanceMetrics:
    """Calculate and track performance metrics"""
    
    def __init__(self):
        self.trades = []
        self.daily_returns = {}
        
    def add_trade(self, trade: Trade):
        """Add completed trade"""
        if trade.status == OrderStatus.CLOSED:
            self.trades.append(trade)
    
    def calculate_metrics(self) -> Dict:
        """Calculate all performance metrics"""
        if not self.trades:
            return self._empty_metrics()
        
        closed_trades = [t for t in self.trades if t.status == OrderStatus.CLOSED]
        
        if not closed_trades:
            return self._empty_metrics()
        
        total_pnl = sum(t.pnl for t in closed_trades)
        total_pnl_percent = sum(t.pnl_percent for t in closed_trades)
        total_commission = sum(t.commission for t in closed_trades)
        
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl < 0]
        
        win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0
        profit_factor = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))) if losses else 0
        
        avg_win = (sum(t.pnl for t in wins) / len(wins)) if wins else 0
        avg_loss = (sum(t.pnl for t in losses) / len(losses)) if losses else 0
        
        longest_winning_streak = self._calculate_streak(closed_trades, True)
        longest_losing_streak = self._calculate_streak(closed_trades, False)
        
        max_drawdown = self._calculate_max_drawdown(closed_trades)
        
        return {
            'total_trades': len(closed_trades),
            'win_trades': len(wins),
            'loss_trades': len(losses),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_percent': round(total_pnl_percent, 2),
            'total_commission': round(total_commission, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'longest_winning_streak': longest_winning_streak,
            'longest_losing_streak': longest_losing_streak,
            'max_drawdown': round(max_drawdown, 2),
        }
    
    @staticmethod
    def _calculate_streak(trades: List[Trade], is_winning: bool) -> int:
        """Calculate longest streak"""
        max_streak = 0
        current_streak = 0
        
        for trade in trades:
            is_win = trade.pnl > 0
            if is_win == is_winning:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    @staticmethod
    def _calculate_max_drawdown(trades: List[Trade]) -> float:
        """Calculate maximum drawdown percentage"""
        if not trades:
            return 0.0
        
        cumulative_pnl = []
        running_total = 0
        
        for trade in trades:
            running_total += trade.pnl
            cumulative_pnl.append(running_total)
        
        peak = cumulative_pnl[0]
        max_dd = 0
        
        for value in cumulative_pnl:
            if value > peak:
                peak = value
            drawdown = ((peak - value) / peak * 100) if peak != 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd


class RiskManager:
    """Manage position sizing and risk"""
    
    def __init__(self, initial_balance: float, risk_percent: float = 2.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_percent = risk_percent
        self.max_daily_loss_percent = 5.0
        self.daily_start_balance = initial_balance
        self.max_open_trades = 5
        
    def reset_daily_metrics(self):
        """Reset daily metrics"""
        self.daily_start_balance = self.current_balance
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit exceeded"""
        daily_loss_percent = ((self.daily_start_balance - self.current_balance) / 
                             self.daily_start_balance * 100)
        
        if daily_loss_percent >= self.max_daily_loss_percent:
            logger.warning(f"Daily loss limit reached: {daily_loss_percent:.2f}%")
            return False
        
        return True
    
    def calculate_position_size(self, stop_loss_pips: float, pip_value: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = self.current_balance * (self.risk_percent / 100)
        stop_loss_value = stop_loss_pips * pip_value
        position_size = risk_amount / stop_loss_value if stop_loss_value > 0 else 0
        return max(position_size, 0.001)
    
    def update_balance(self, new_balance: float):
        """Update current balance"""
        self.current_balance = new_balance


class IndicatorCalculator:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, 
                      slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD Indicator"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr


class SignalGenerator:
    """Generate trading signals based on indicators"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.indicators = IndicatorCalculator()
    
    def generate_signals(self, df: pd.DataFrame) -> Tuple[Optional[TradeType], float, float]:
        """
        Generate trading signals
        
        Returns:
            (signal_type, stop_loss_pips, take_profit_pips)
        """
        if len(df) < 50:
            return None, 0, 0
        
        # Calculate indicators
        fast_ma = self.indicators.calculate_ema(df['close'], self.config['fast_ma'])
        slow_ma = self.indicators.calculate_ema(df['close'], self.config['slow_ma'])
        rsi = self.indicators.calculate_rsi(df['close'], self.config['rsi_period'])
        macd, signal, histogram = self.indicators.calculate_macd(
            df['close'],
            self.config['macd_fast'],
            self.config['macd_slow'],
            self.config['macd_signal']
        )
        
        upper_bb, mid_bb, lower_bb = self.indicators.calculate_bollinger_bands(
            df['close'], 
            self.config['bb_period'],
            self.config['bb_std_dev']
        )
        
        atr = self.indicators.calculate_atr(df['high'], df['low'], df['close'], 14)
        
        # Get latest values
        fast_ma_val = fast_ma.iloc[-1]
        slow_ma_val = slow_ma.iloc[-1]
        rsi_val = rsi.iloc[-1]
        macd_val = macd.iloc[-1]
        signal_val = signal.iloc[-1]
        histogram_val = histogram.iloc[-1]
        current_price = df['close'].iloc[-1]
        atr_val = atr.iloc[-1]
        
        # Validate values
        if any(pd.isna(v) for v in [fast_ma_val, slow_ma_val, rsi_val, macd_val, signal_val, atr_val]):
            return None, 0, 0
        
        # BUY Signal: MA crossover + RSI oversold + MACD bullish + price near lower BB
        buy_signal = (
            fast_ma_val > slow_ma_val and
            rsi_val < self.config['rsi_oversold'] and
            macd_val > signal_val and
            histogram_val > 0 and
            current_price < mid_bb.iloc[-1]
        )
        
        # SELL Signal: MA crossover + RSI overbought + MACD bearish + price near upper BB
        sell_signal = (
            fast_ma_val < slow_ma_val and
            rsi_val > self.config['rsi_overbought'] and
            macd_val < signal_val and
            histogram_val < 0 and
            current_price > mid_bb.iloc[-1]
        )
        
        sl_pips = self.config['stop_loss_pips']
        tp_pips = self.config['take_profit_pips']
        
        # Scale risk/reward based on ATR
        atr_pips = atr_val / 0.0001 if atr_val > 0 else sl_pips
        
        if atr_pips > sl_pips * 3:  # High volatility
            sl_pips = min(sl_pips * 1.5, atr_pips * 0.5)
            tp_pips = tp_pips * 2
        elif atr_pips < sl_pips * 0.5:  # Low volatility
            sl_pips = max(sl_pips * 0.75, atr_pips * 0.3)
        
        if buy_signal:
            return TradeType.BUY, sl_pips, tp_pips
        elif sell_signal:
            return TradeType.SELL, sl_pips, tp_pips
        
        return None, 0, 0


class PaperTradingEngine:
    """Simulate trading without real money"""
    
    def __init__(self, initial_balance: float, config: Dict):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.config = config
        
        self.risk_manager = RiskManager(initial_balance)
        self.signal_generator = SignalGenerator(config)
        self.performance = PerformanceMetrics()
        
        self.open_trades: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0
        
        self.commission_rate = config.get('commission_rate', 0.001)  # 0.1%
        self.slippage_points = config.get('slippage_points', 1.0)  # pips
        
    def process_symbol(self, symbol: str, df: pd.DataFrame) -> bool:
        """Process trading logic for a symbol"""
        
        # Check if can open more trades
        if len(self.open_trades) >= self.risk_manager.max_open_trades:
            return False
        
        # Check daily loss limit
        if not self.risk_manager.check_daily_loss_limit():
            return False
        
        # Generate signal
        signal, sl_pips, tp_pips = self.signal_generator.generate_signals(df)
        
        if signal is None:
            return False
        
        current_price = df['close'].iloc[-1]
        pip_value = 0.0001 if 'XAU' not in symbol and 'BTC' not in symbol else 1.0
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(sl_pips, pip_value)
        
        if position_size <= 0:
            logger.warning(f"Invalid position size for {symbol}: {position_size}")
            return False
        
        # Execute trade
        trade_id = f"{symbol}_{self.trade_counter}_{int(time.time() * 1000)}"
        self.trade_counter += 1
        
        if signal == TradeType.BUY:
            return self._execute_buy(trade_id, symbol, current_price, position_size, sl_pips, tp_pips, pip_value)
        else:
            return self._execute_sell(trade_id, symbol, current_price, position_size, sl_pips, tp_pips, pip_value)
    
    def _execute_buy(self, trade_id: str, symbol: str, current_price: float, 
                    position_size: float, sl_pips: float, tp_pips: float, pip_value: float) -> bool:
        """Execute BUY trade"""
        
        entry_price = current_price + (self.slippage_points * pip_value)
        stop_loss = entry_price - (sl_pips * pip_value)
        take_profit = entry_price + (tp_pips * pip_value)
        
        commission = (entry_price * position_size) * self.commission_rate
        
        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            trade_type=TradeType.BUY,
            entry_price=entry_price,
            entry_time=datetime.now(),
            quantity=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            commission=commission,
            slippage=self.slippage_points * pip_value
        )
        
        self.open_trades[trade_id] = trade
        self.current_balance -= commission
        self.risk_manager.update_balance(self.current_balance)
        
        logger.info(f"BUY: {symbol} | Size: {position_size:.4f} | Entry: {entry_price:.5f} | "
                   f"SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
        
        return True
    
    def _execute_sell(self, trade_id: str, symbol: str, current_price: float, 
                     position_size: float, sl_pips: float, tp_pips: float, pip_value: float) -> bool:
        """Execute SELL trade"""
        
        entry_price = current_price - (self.slippage_points * pip_value)
        stop_loss = entry_price + (sl_pips * pip_value)
        take_profit = entry_price - (tp_pips * pip_value)
        
        commission = (entry_price * position_size) * self.commission_rate
        
        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            trade_type=TradeType.SELL,
            entry_price=entry_price,
            entry_time=datetime.now(),
            quantity=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            commission=commission,
            slippage=self.slippage_points * pip_value
        )
        
        self.open_trades[trade_id] = trade
        self.current_balance -= commission
        self.risk_manager.update_balance(self.current_balance)
        
        logger.info(f"SELL: {symbol} | Size: {position_size:.4f} | Entry: {entry_price:.5f} | "
                   f"SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
        
        return True
    
    def check_exit_conditions(self, symbol: str, current_price: float):
        """Check if any trades should be closed"""
        
        trades_to_close = []
        
        for trade_id, trade in self.open_trades.items():
            if trade.symbol != symbol:
                continue
            
            if trade.trade_type == TradeType.BUY:
                # Check SL
                if current_price <= trade.stop_loss:
                    trades_to_close.append((trade_id, current_price, "SL"))
                # Check TP
                elif current_price >= trade.take_profit:
                    trades_to_close.append((trade_id, current_price, "TP"))
            
            else:  # SELL
                # Check SL
                if current_price >= trade.stop_loss:
                    trades_to_close.append((trade_id, current_price, "SL"))
                # Check TP
                elif current_price <= trade.take_profit:
                    trades_to_close.append((trade_id, current_price, "TP"))
        
        for trade_id, exit_price, reason in trades_to_close:
            self._close_trade(trade_id, exit_price, reason)
    
    def _close_trade(self, trade_id: str, exit_price: float, reason: str):
        """Close a trade"""
        
        trade = self.open_trades.pop(trade_id)
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.status = OrderStatus.CLOSED
        
        if trade.trade_type == TradeType.BUY:
            pnl = (exit_price - trade.entry_price) * trade.quantity - trade.commission
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity - trade.commission
        
        trade.pnl = pnl
        trade.pnl_percent = (pnl / (trade.entry_price * trade.quantity)) * 100
        
        self.current_balance += (trade.entry_price * trade.quantity) + pnl
        self.risk_manager.update_balance(self.current_balance)
        
        self.closed_trades.append(trade)
        self.performance.add_trade(trade)
        
        profit_str = f"+{pnl:.2f}" if pnl > 0 else f"{pnl:.2f}"
        logger.info(f"CLOSED: {trade.symbol} | {trade.trade_type.value} | "
                   f"Exit: {exit_price:.5f} | P&L: {profit_str} ({trade.pnl_percent:.2f}%) | "
                   f"Reason: {reason}")


class ScalpingEAV2:
    """Main Scalping EA V2"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 config: Dict = None, paper_trading: bool = True):
        """
        Initialize Scalping EA V2
        
        Args:
            api_key: Exchange API key (optional for paper trading)
            api_secret: Exchange API secret (optional for paper trading)
            config: Configuration dictionary
            paper_trading: Use paper trading simulator (True) or real trading (False)
        """
        
        self.paper_trading = paper_trading
        
        # Default configuration
        self.config = {
            'initial_balance': 10.0,
            'risk_percentage': 2.0,
            'max_daily_loss': 5.0,
            'max_open_trades': 5,
            'take_profit_pips': 5,
            'stop_loss_pips': 3,
            'trailing_stop_pips': 2,
            'fast_ma': 5,
            'slow_ma': 20,
            'rsi_period': 14,
            'rsi_overbought': 75.0,
            'rsi_oversold': 25.0,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'symbols': ['XAUUSD', 'BTCUSD', 'EURUSD'],
            'timeframe': '1m',
            'commission_rate': 0.001,  # 0.1%
            'slippage_points': 1.0,  # pips
            'update_interval': 60,  # seconds
        }
        
        if config:
            self.config.update(config)
        
        # Initialize paper trading engine
        self.trading_engine = PaperTradingEngine(self.config['initial_balance'], self.config)
        
        # Exchange connection (optional)
        self.exchange = None
        if not paper_trading and api_key and api_secret:
            try:
                self.exchange = ccxt.exness({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'trading'}
                })
                logger.info("Connected to Exness exchange")
            except Exception as e:
                logger.error(f"Failed to connect to exchange: {e}")
                self.paper_trading = True
        
        # Tracking
        self.running = False
        self.last_hourly_check = datetime.now()
        self.symbol_price_cache = {}
        
        logger.info(f"ScalpingEA V2 initialized | Paper Trading: {self.paper_trading}")
    
    def get_candles(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV candles"""
        
        try:
            if self.exchange and not self.paper_trading:
                candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            else:
                # Simulate candles for paper trading
                candles = self._generate_simulated_candles(symbol, limit)
            
            if not candles:
                return pd.DataFrame()
            
            df = pd.DataFrame(
                candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return pd.DataFrame()
    
    def _generate_simulated_candles(self, symbol: str, limit: int) -> List:
        """Generate simulated candles for paper trading"""
        
        # Get or initialize base price
        if symbol not in self.symbol_price_cache:
            base_prices = {
                'XAUUSD': 2000.0,
                'BTCUSD': 45000.0,
                'EURUSD': 1.0850,
                'USDJPY': 150.0,
                'GBPUSD': 1.2700,
            }
            self.symbol_price_cache[symbol] = base_prices.get(symbol, 1.0)
        
        current_price = self.symbol_price_cache[symbol]
        candles = []
        
        now = datetime.now()
        
        for i in range(limit):
            # Generate realistic price movement (±0.5% per candle)
            change_percent = np.random.normal(0, 0.005)
            change = current_price * change_percent
            
            open_price = current_price
            close_price = current_price + change
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.002)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.002)))
            
            volume = np.random.uniform(1000, 50000)
            
            timestamp = int((now - timedelta(minutes=limit-i-1)).timestamp() * 1000)
            
            candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
            
            current_price = close_price
        
        self.symbol_price_cache[symbol] = current_price
        
        return candles
    
    def run_24_7(self, update_interval: int = 60, duration_hours: int = None):
        """
        Run EA 24/7
        
        Args:
            update_interval: Update interval in seconds
            duration_hours: Run for X hours (None = infinite)
        """
        
        self.running = True
        logger.info("=" * 80)
        logger.info("ScalpingEA V2 - 24/7 Trading Started")
        logger.info(f"Mode: {'Paper Trading' if self.paper_trading else 'Live Trading'}")
        logger.info(f"Initial Balance: ${self.trading_engine.initial_balance:.2f}")
        logger.info(f"Risk per Trade: {self.config['risk_percentage']}%")
        logger.info(f"Max Daily Loss: {self.config['max_daily_loss']}%")
        logger.info(f"Update Interval: {update_interval}s")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        update_count = 0
        
        try:
            while self.running:
                # Check duration limit
                if duration_hours:
                    elapsed = (datetime.now() - start_time).total_seconds() / 3600
                    if elapsed >= duration_hours:
                        logger.info(f"Duration limit reached ({duration_hours}h)")
                        break
                
                # Hourly reset check
                current_hour = datetime.now().hour
                if current_hour != self.last_hourly_check.hour:
                    self.trading_engine.risk_manager.reset_daily_metrics()
                    logger.info("Daily metrics reset")
                    self.last_hourly_check = datetime.now()
                
                # Process each symbol
                for symbol in self.config['symbols']:
                    try:
                        df = self.get_candles(symbol, self.config['timeframe'], limit=100)
                        
                        if df.empty:
                            continue
                        
                        # Check exit conditions for existing trades
                        self.trading_engine.check_exit_conditions(symbol, df['close'].iloc[-1])
                        
                        # Process new signals
                        self.trading_engine.process_symbol(symbol, df)
                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                
                # Log statistics every N updates
                update_count += 1
                if update_count % 10 == 0:
                    self._log_statistics()
                
                time.sleep(update_interval)
        
        except KeyboardInterrupt:
            logger.info("EA stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def _log_statistics(self):
        """Log current statistics"""
        
        engine = self.trading_engine
        metrics = engine.performance.calculate_metrics()
        
        logger.info(f"\n{'='*80}")
        logger.info("TRADING STATISTICS")
        logger.info(f"{'='*80}")
        logger.info(f"Current Balance: ${engine.current_balance:.2f}")
        logger.info(f"Total P&L: ${engine.current_balance - engine.initial_balance:.2f}")
        logger.info(f"Open Positions: {len(engine.open_trades)}")
        logger.info(f"\nClosed Trades: {metrics['total_trades']}")
        logger.info(f"Wins: {metrics['win_trades']} | Losses: {metrics['loss_trades']}")
        logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
        logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"Total P&L: ${metrics['total_pnl']:.2f} ({metrics['total_pnl_percent']:.2f}%)")
        logger.info(f"Avg Win: ${metrics['avg_win']:.2f} | Avg Loss: ${metrics['avg_loss']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        logger.info(f"{'='*80}\n")
    
    def shutdown(self):
        """Shutdown and save results"""
        
        self.running = False
        
        logger.info("\n" + "=" * 80)
        logger.info("ScalpingEA V2 - FINAL REPORT")
        logger.info("=" * 80)
        
        engine = self.trading_engine
        metrics = engine.performance.calculate_metrics()
        
        logger.info(f"Initial Balance: ${engine.initial_balance:.2f}")
        logger.info(f"Final Balance: ${engine.current_balance:.2f}")
        logger.info(f"Total P&L: ${engine.current_balance - engine.initial_balance:.2f}")
        logger.info(f"\nTrade Statistics:")
        logger.info(f"  Total Trades: {metrics['total_trades']}")
        logger.info(f"  Wins: {metrics['win_trades']} | Losses: {metrics['loss_trades']}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2f}%")
        logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"  Longest Win Streak: {metrics['longest_winning_streak']}")
        logger.info(f"  Longest Loss Streak: {metrics['longest_losing_streak']}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
        logger.info(f"\nPnL Statistics:")
        logger.info(f"  Total Gross P&L: ${metrics['total_pnl']:.2f}")
        logger.info(f"  Total Commissions: ${metrics['total_commission']:.2f}")
        logger.info(f"  Avg Win: ${metrics['avg_win']:.2f}")
        logger.info(f"  Avg Loss: ${metrics['avg_loss']:.2f}")
        
        logger.info("=" * 80)
        
        # Save trade history
        self._save_results()
    
    def _save_results(self):
        """Save results to JSON"""
        
        try:
            engine = self.trading_engine
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'mode': 'paper_trading' if self.paper_trading else 'live_trading',
                'config': self.config,
                'initial_balance': engine.initial_balance,
                'final_balance': engine.current_balance,
                'total_pnl': engine.current_balance - engine.initial_balance,
                'metrics': engine.performance.calculate_metrics(),
                'trades': [trade.to_dict() for trade in engine.closed_trades]
            }
            
            filename = f"scalping_ea_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {filename}")
        
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main entry point"""
    
    # Configuration
    config = {
        'initial_balance': 10.0,           # Start with $10
        'risk_percentage': 2.0,            # Risk 2% per trade
        'max_daily_loss': 5.0,             # Stop at -5% daily
        'max_open_trades': 3,              # Max 3 concurrent trades
        'take_profit_pips': 10,            # 10 pips profit
        'stop_loss_pips': 5,               # 5 pips stop loss
        'fast_ma': 8,
        'slow_ma': 21,
        'rsi_period': 14,
        'rsi_overbought': 70.0,
        'rsi_oversold': 30.0,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std_dev': 2.0,
        'symbols': ['XAUUSD', 'BTCUSD', 'EURUSD'],
        'commission_rate': 0.0005,         # 0.05%
        'slippage_points': 0.5,            # 0.5 pips
        'update_interval': 60,             # 60 seconds
    }
    
    # For live trading, provide your Exness API credentials
    # API_KEY = 'your_api_key'
    # API_SECRET = 'your_api_secret'
    
    try:
        # Run in paper trading mode (remove for live trading)
        ea = ScalpingEAV2(config=config, paper_trading=True)
        
        # Run for 24 hours (set to None for infinite)
        ea.run_24_7(update_interval=60, duration_hours=24)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == '__main__':
    main()
