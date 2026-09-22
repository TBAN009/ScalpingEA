"""Offline characterization tests: never connect to MT5 or send live orders.
Run: python -m unittest discover -s tests -v
Dependencies: pandas, numpy. MetaTrader5 is replaced before importing the EA.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd


def load_ea():
    fake = types.ModuleType('MetaTrader5')
    constants = ['TIMEFRAME_M1', 'TIMEFRAME_M5', 'ORDER_TYPE_BUY',
                 'ORDER_TYPE_SELL', 'TRADE_ACTION_DEAL', 'ORDER_TIME_GTC',
                 'ORDER_FILLING_IOC', 'ORDER_FILLING_FOK', 'TRADE_RETCODE_DONE',
                 'TRADE_RETCODE_DONE_PARTIAL', 'TRADE_RETCODE_PLACED',
                 'DEAL_ENTRY_IN', 'DEAL_ENTRY_OUT', 'DEAL_ENTRY_OUT_BY',
                 'SYMBOL_TRADE_EXECUTION_MARKET']
    for value, name in enumerate(constants, 1):
        setattr(fake, name, value)
    for name in ['initialize', 'shutdown', 'account_info', 'symbol_info',
                 'symbol_info_tick', 'symbol_select', 'positions_get',
                 'orders_get', 'order_send', 'order_check', 'last_error',
                 'copy_rates_from', 'copy_rates_from_pos', 'history_deals_get']:
        setattr(fake, name, Mock(name=name))
    path = Path(__file__).resolve().parents[1] / 'scalping_ea_v2.py'
    spec = importlib.util.spec_from_file_location('ea_under_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'MetaTrader5': fake}):
        spec.loader.exec_module(module)
    return module, fake


class CharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.ea, self.mt5 = load_ea()

    def test_import_does_not_connect_or_trade(self):
        self.mt5.initialize.assert_not_called()
        self.mt5.order_send.assert_not_called()

    def test_indicators_preserve_input_and_expected_columns(self):
        source = pd.DataFrame({'close': [100.0 + i for i in range(50)],
                               'high': [101.0 + i for i in range(50)],
                               'low': [99.0 + i for i in range(50)]})
        original = source.copy(deep=True)
        result = self.ea.compute_indicators(source)
        pd.testing.assert_frame_equal(source, original)
        self.assertEqual(len(result), len(source))
        for column in ['ema_fast', 'ema_slow', 'rsi', 'atr']:
            self.assertIn(column, result)
        self.assertGreater(result.iloc[-1].ema_fast, result.iloc[-1].ema_slow)
        self.assertGreater(result.iloc[-1].atr, 0)

    def test_missing_symbol_raises(self):
        self.mt5.symbol_info.return_value = None
        with self.assertRaises(RuntimeError):
            self.ea.get_symbol_info_or_raise('MISSING')

    def test_initialization_failure_raises(self):
        self.mt5.initialize.return_value = False
        self.mt5.last_error.return_value = (-1, 'offline test')
        with self.assertRaises(RuntimeError):
            self.ea.mt5_connect()


if __name__ == '__main__':
    unittest.main()
