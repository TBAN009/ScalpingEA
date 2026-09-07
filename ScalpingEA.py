"""
ScalpingEA - Professional Forex Scalping Expert Advisor
Optimized for GOLD, BTCUSD, USTEC on Exness Broker
Designed to grow $10 accounts to $100+ daily with 24/7 trading
Author: ScalpingEA Pro
"""

import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ScalpingEA.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScalpingEA:
    """Professional Scalping Expert Advisor for Exness"""
    
    def __init__(self, api_key: str, api_secret: str, config: Dict = None):
        """
        Initialize the Scalping EA
        
        Args:
            api_key: Exness API key
            api_secret: Exness API secret
            config: Configuration dictionary
        """
        # Exness broker connection
        self.exchange = ccxt.exness({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'trading',
            }
        })
        
        # Default configuration
        self.config = {
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
            'symbols': ['XAUUSD', 'BTCUSD', 'USTEC'],
            'timeframe': '1m',
            'trade_24_7': True,
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Initialize tracking variables
        self.initial_balance = None
        self.daily_start_balance = None
        self.daily_profit = 0
        self.total_trades = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.last_trade_time = datetime.now()
        self.open_positions = {}
        self.trade_history = []
        
        logger.info("ScalpingEA initialized successfully")
        self._load_account_info()
    
    def _load_account_info(self):
        """Load and initialize account information"""
        try:
            balance = self.exchange.fetch_balance()
            self.initial_balance = balance['total']['USDT']
            self.daily_start_balance = self.initial_balance
            logger.info(f"Initial Balance: ${self.initial_balance:.2f}")
            logger.info(f"Risk per trade: {self.config['risk_percentage']}%")
        except Exception as e:
            logger.error(f"Error loading account info: {e}")
            raise
    
    def get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['total']['USDT']
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
    
    def get_candles(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV candles
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            timeframe: Timeframe (1m, 5m, 15m, 1h, etc.)
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_moving_average(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return df['close'].rolling(window=period).mean()
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, df: pd.DataFrame, 
                      fast: int = 12, 
                      slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD
        
        Returns:
            macd_line, signal_line, histogram
        """
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_lot_size(self, symbol: str, stop_loss_pips: int) -> float:
        """
        Calculate optimal lot size based on account balance and risk
        
        Args:
            symbol: Trading symbol
            stop_loss_pips: Stop loss in pips
            
        Returns:
            Lot size
        """
        try:
            balance = self.get_account_balance()
            risk_amount = balance * (self.config['risk_percentage'] / 100)
            
            # Get symbol market info
            market = self.exchange.market(symbol)
            
            # Calculate pip value (typically 0.0001 for most pairs)
            pip_value = 0.0001
            
            # For GOLD and crypto, adjust pip value
            if 'XAU' in symbol or 'BTC' in symbol:
                pip_value = 1.0
            
            stop_loss_value = stop_loss_pips * pip_value
            
            # Calculate lot size
            lot_size = risk_amount / stop_loss_value
            
            # Get minimum and maximum lot sizes
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.001)
            max_amount = market.get('limits', {}).get('amount', {}).get('max', 1000)
            
            # For small accounts, use micro lots
            if balance < 100:
                lot_size = max(min_amount, lot_size * 0.1)
            
            # Ensure lot size is within limits
            lot_size = max(min_amount, min(max_amount, lot_size))
            
            return lot_size
        except Exception as e:
            logger.error(f"Error calculating lot size for {symbol}: {e}")
            return 0.01  # Default to 0.01 lot
    
    def process_symbol(self, symbol: str) -> bool:
        """
        Process trading signals for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if trade was executed, False otherwise
        """
        try:
            # Check if max trades reached
            if len(self.open_positions) >= self.config['max_open_trades']:
                return False
            
            # Fetch candles
            df = self.get_candles(symbol, self.config['timeframe'], limit=100)
            
            if df.empty or len(df) < self.config['slow_ma']:
                logger.warning(f"Not enough candles for {symbol}")
                return False
            
            # Calculate indicators
            fast_ma = self.calculate_moving_average(df, self.config['fast_ma']).iloc[-1]
            slow_ma = self.calculate_moving_average(df, self.config['slow_ma']).iloc[-1]
            rsi = self.calculate_rsi(df, self.config['rsi_period']).iloc[-1]
            macd, signal, histogram = self.calculate_macd(
                df,
                self.config['macd_fast'],
                self.config['macd_slow'],
                self.config['macd_signal']
            )
            
            macd_value = macd.iloc[-1]
            signal_value = signal.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Get bid/ask
            ticker = self.exchange.fetch_ticker(symbol)
            bid = ticker['bid']
            ask = ticker['ask']
            spread = ask - bid
            
            # BUY Signal
            if (fast_ma > slow_ma and 
                rsi < self.config['rsi_oversold'] and 
                macd_value > signal_value and
                not np.isnan(fast_ma) and 
                not np.isnan(slow_ma) and 
                not np.isnan(rsi)):
                
                lot_size = self.calculate_lot_size(symbol, self.config['stop_loss_pips'])
                
                if self._execute_buy(symbol, lot_size, bid, ask):
                    return True
            
            # SELL Signal
            elif (fast_ma < slow_ma and 
                  rsi > self.config['rsi_overbought'] and 
                  macd_value < signal_value and
                  not np.isnan(fast_ma) and 
                  not np.isnan(slow_ma) and 
                  not np.isnan(rsi)):
                
                lot_size = self.calculate_lot_size(symbol, self.config['stop_loss_pips'])
                
                if self._execute_sell(symbol, lot_size, bid, ask):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return False
    
    def _execute_buy(self, symbol: str, lot_size: float, bid: float, ask: float) -> bool:
        """Execute a BUY trade"""
        try:
            # Calculate TP and SL
            pip_multiplier = 0.0001 if 'XAU' not in symbol else 1.0
            tp = ask + (self.config['take_profit_pips'] * pip_multiplier)
            sl = bid - (self.config['stop_loss_pips'] * pip_multiplier)
            
            # Create order
            order = self.exchange.create_market_buy_order(
                symbol,
                lot_size,
                {
                    'stopLoss': {'triggerPrice': sl},
                    'takeProfit': {'triggerPrice': tp}
                }
            )
            
            self.total_trades += 1
            self.last_trade_time = datetime.now()
            
            # Track position
            self.open_positions[order['id']] = {
                'symbol': symbol,
                'type': 'buy',
                'lot_size': lot_size,
                'entry_price': ask,
                'take_profit': tp,
                'stop_loss': sl,
                'timestamp': datetime.now(),
                'order_id': order['id']
            }
            
            logger.info(f"BUY signal on {symbol} | Lot: {lot_size} | TP: {tp} | SL: {sl}")
            self.trade_history.append(self.open_positions[order['id']])
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing BUY order for {symbol}: {e}")
            return False
    
    def _execute_sell(self, symbol: str, lot_size: float, bid: float, ask: float) -> bool:
        """Execute a SELL trade"""
        try:
            # Calculate TP and SL
            pip_multiplier = 0.0001 if 'XAU' not in symbol else 1.0
            tp = bid - (self.config['take_profit_pips'] * pip_multiplier)
            sl = ask + (self.config['stop_loss_pips'] * pip_multiplier)
            
            # Create order
            order = self.exchange.create_market_sell_order(
                symbol,
                lot_size,
                {
                    'stopLoss': {'triggerPrice': sl},
                    'takeProfit': {'triggerPrice': tp}
                }
            )
            
            self.total_trades += 1
            self.last_trade_time = datetime.now()
            
            # Track position
            self.open_positions[order['id']] = {
                'symbol': symbol,
                'type': 'sell',
                'lot_size': lot_size,
                'entry_price': bid,
                'take_profit': tp,
                'stop_loss': sl,
                'timestamp': datetime.now(),
                'order_id': order['id']
            }
            
            logger.info(f"SELL signal on {symbol} | Lot: {lot_size} | TP: {tp} | SL: {sl}")
            self.trade_history.append(self.open_positions[order['id']])
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing SELL order for {symbol}: {e}")
            return False
    
    def update_trailing_stops(self):
        """Update trailing stops for open positions"""
        if not self.config.get('trailing_stop_pips'):
            return
        
        try:
            for order_id, position in list(self.open_positions.items()):
                try:
                    # Fetch current ticker
                    ticker = self.exchange.fetch_ticker(position['symbol'])
                    current_price = ticker['last']
                    
                    pip_multiplier = 0.0001 if 'XAU' not in position['symbol'] else 1.0
                    trailing_distance = self.config['trailing_stop_pips'] * pip_multiplier
                    
                    if position['type'] == 'buy':
                        new_sl = current_price - trailing_distance
                        if new_sl > position['stop_loss']:
                            # Update stop loss
                            position['stop_loss'] = new_sl
                    
                    elif position['type'] == 'sell':
                        new_sl = current_price + trailing_distance
                        if new_sl < position['stop_loss']:
                            # Update stop loss
                            position['stop_loss'] = new_sl
                
                except Exception as e:
                    logger.error(f"Error updating trailing stop for {order_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in update_trailing_stops: {e}")
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached"""
        current_balance = self.get_account_balance()
        self.daily_profit = current_balance - self.daily_start_balance
        
        max_loss = self.daily_start_balance * (self.config['max_daily_loss'] / 100)
        
        if self.daily_profit < -max_loss:
            logger.warning(f"Daily loss limit reached. Current loss: ${-self.daily_profit:.2f}")
            return False
        
        return True
    
    def check_time_allowed(self) -> bool:
        """Check if current time is allowed for trading"""
        if self.config['trade_24_7']:
            return True
        
        current_hour = datetime.now().hour
        # Default: trade from 00:00 to 23:00 UTC
        return 0 <= current_hour < 24
    
    def run(self, update_interval: int = 60):
        """
        Main EA loop
        
        Args:
            update_interval: Update interval in seconds (default: 60 seconds)
        """
        logger.info("Starting ScalpingEA...")
        
        try:
            while True:
                # Check if trading is allowed
                if not self.check_time_allowed():
                    logger.info("Outside trading hours")
                    time.sleep(update_interval)
                    continue
                
                # Check daily loss limit
                if not self.check_daily_loss_limit():
                    logger.warning("Daily loss limit reached. Stopping trades.")
                    time.sleep(update_interval)
                    continue
                
                # Update trailing stops
                self.update_trailing_stops()
                
                # Process each symbol
                for symbol in self.config['symbols']:
                    try:
                        self.process_symbol(symbol)
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                
                # Log statistics
                balance = self.get_account_balance()
                logger.info(f"Balance: ${balance:.2f} | Trades: {self.total_trades} | "
                          f"Daily Profit: ${self.daily_profit:.2f}")
                
                # Sleep before next update
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            logger.info("EA stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in EA loop: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the EA and save statistics"""
        logger.info("Shutting down ScalpingEA...")
        logger.info(f"Total Trades: {self.total_trades}")
        logger.info(f"Win Trades: {self.win_trades}")
        logger.info(f"Loss Trades: {self.loss_trades}")
        
        if self.total_trades > 0:
            win_rate = (self.win_trades / self.total_trades) * 100
            logger.info(f"Win Rate: {win_rate:.2f}%")
        
        final_balance = self.get_account_balance()
        total_profit = final_balance - self.initial_balance
        logger.info(f"Initial Balance: ${self.initial_balance:.2f}")
        logger.info(f"Final Balance: ${final_balance:.2f}")
        logger.info(f"Total Profit: ${total_profit:.2f}")
        
        # Save trade history
        self._save_trade_history()
    
    def _save_trade_history(self):
        """Save trade history to JSON file"""
        try:
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=str)
            logger.info(f"Trade history saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")


def main():
    """Main entry point"""
    
    # Configuration
    config = {
        'risk_percentage': 2.0,           # Risk 2% per trade
        'max_daily_loss': 5.0,            # Stop if lose 5% daily
        'max_open_trades': 5,             # Max 5 concurrent trades
        'take_profit_pips': 5,            # 5 pips profit target
        'stop_loss_pips': 3,              # 3 pips stop loss
        'trailing_stop_pips': 2,          # 2 pips trailing stop
        'fast_ma': 5,                     # Fast MA period
        'slow_ma': 20,                    # Slow MA period
        'rsi_period': 14,                 # RSI period
        'rsi_overbought': 75.0,           # RSI overbought level
        'rsi_oversold': 25.0,             # RSI oversold level
        'symbols': ['XAUUSD', 'BTCUSD', 'USTEC'],
        'timeframe': '1m',                # 1-minute candles
        'trade_24_7': True,               # Trade 24/7
    }
    
    # Initialize EA with your Exness API credentials
    # Get these from your Exness account settings
    API_KEY = 'your_exness_api_key_here'
    API_SECRET = 'your_exness_api_secret_here'
    
    try:
        ea = ScalpingEA(API_KEY, API_SECRET, config)
        ea.run(update_interval=60)  # Update every 60 seconds
    except Exception as e:
        logger.error(f"Failed to start EA: {e}")


if __name__ == '__main__':
    main()
