"""Offline tests: MT5 is mocked; tests cannot connect or submit live orders.
Run: python -m unittest discover -s tests -v
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

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


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.ea, self.mt5 = load_ea()
        self.symbol = types.SimpleNamespace(
            trade_tick_value_loss=2.0, trade_tick_value=2.0,
            trade_tick_size=0.5, point=0.01, volume_step=0.01,
            volume_min=0.01, volume_max=10.0, visible=True,
            trade_stops_level=10, filling_mode=3, digits=2)
        self.mt5.symbol_info.return_value = self.symbol

    def test_lot_uses_tick_size_not_point(self):
        # $20 risk; 5 price units / 0.5 * $2 = $20 per lot.
        self.assertEqual(self.ea.calculate_lot('X', 5, 1000), 1.0)

    def test_unaffordable_minimum_lot_blocks_trade(self):
        self.assertEqual(self.ea.calculate_lot('X', 500, 1), 0.0)

    def test_missing_symbol_never_falls_back_to_live_volume(self):
        self.mt5.symbol_info.return_value = None
        self.assertEqual(self.ea.calculate_lot('X', 5, 1000), 0.0)

    def test_invalid_risk_inputs_block(self):
        for distance in [0, -1, float('nan'), float('inf')]:
            self.assertEqual(self.ea.calculate_lot('X', distance, 1000), 0.0)

    def test_failed_position_query_blocks_before_indicators(self):
        self.mt5.positions_get.return_value = None
        self.mt5.orders_get.return_value = ()
        self.ea.evaluate_and_maybe_trade('X', 1000)
        self.mt5.order_send.assert_not_called()
        self.mt5.copy_rates_from_pos.assert_not_called()

    def test_closed_bar_retrieval_uses_position_one(self):
        now = int(datetime.now(timezone.utc).timestamp())
        stamp = now // 60 * 60 - 60
        self.mt5.copy_rates_from_pos.return_value = [
            {'time': stamp - 60 * (39-i), 'open': 100.0,
             'high': 101.0, 'low': 99.0, 'close': 100.0} for i in range(40)]
        result = self.ea.get_rates('X', self.mt5.TIMEFRAME_M1, 40)
        self.mt5.copy_rates_from_pos.assert_called_once_with('X', self.mt5.TIMEFRAME_M1, 1, 40)
        self.assertEqual(len(result), 40)

    def test_history_unavailable_blocks(self):
        self.mt5.history_deals_get.return_value = None
        with self.assertRaises(RuntimeError):
            self.ea.latest_loss_time('X')

    def test_loss_includes_entry_commission_and_manual_exit(self):
        opening = types.SimpleNamespace(position_id=1, symbol='X', magic=self.ea.MAGIC,
            entry=self.mt5.DEAL_ENTRY_IN, volume=1, profit=0, commission=-2,
            swap=0, fee=0, time_msc=1000, ticket=1)
        closing = types.SimpleNamespace(position_id=1, symbol='X', magic=0,
            entry=self.mt5.DEAL_ENTRY_OUT, volume=1, profit=1, commission=0,
            swap=0, fee=0, time_msc=2000, ticket=2)
        self.mt5.history_deals_get.side_effect = [(opening, closing), (opening, closing)]
        self.assertEqual(self.ea.latest_loss_time('X'), 2.0)

    def test_uncertain_submission_is_not_retried(self):
        tick = types.SimpleNamespace(ask=100.0, bid=99.9,
                                     time_msc=int(self.ea.time.time() * 1000))
        self.mt5.symbol_info_tick.return_value = tick
        self.mt5.order_check.return_value = types.SimpleNamespace(retcode=0)
        self.mt5.order_send.return_value = None
        self.ea.place_order('X', self.mt5.ORDER_TYPE_BUY, 1, 100, 95, 110)
        self.ea.place_order('X', self.mt5.ORDER_TYPE_BUY, 1, 100, 95, 110)
        self.assertEqual(self.mt5.order_send.call_count, 1)
        self.assertIn('X', self.ea._blocked_symbols)

    def test_failed_preflight_does_not_send(self):
        self.mt5.symbol_info_tick.return_value = types.SimpleNamespace(
            ask=100.0, bid=99.9, time_msc=int(self.ea.time.time() * 1000))
        self.mt5.order_check.return_value = types.SimpleNamespace(retcode=10019)
        self.ea.place_order('X', self.mt5.ORDER_TYPE_BUY, 1, 100, 95, 110)
        self.mt5.order_send.assert_not_called()

    def test_stale_tick_blocks(self):
        self.mt5.symbol_info_tick.return_value = types.SimpleNamespace(
            ask=100.0, bid=99.9, time_msc=1000)
        with self.assertRaises(RuntimeError):
            self.ea.fresh_tick('X')

    def test_recovery_rejects_excessive_spread_without_m5_fetch(self):
        data = pd.DataFrame({'atr': [1.0] * 30, 'ema_fast': [101.0] * 30,
                             'ema_slow': [100.0] * 30, 'rsi': [55.0] * 30})
        self.assertFalse(self.ea.recovery_checks('X', 1, data, 1.0))
        self.mt5.copy_rates_from_pos.assert_not_called()

    def test_cooldown_blocks_otherwise_valid_entry(self):
        now = int(self.ea.time.time())
        stamp = now // 60 * 60 - 60
        frame = pd.DataFrame({
            'time': pd.to_datetime([stamp-120, stamp-60, stamp], unit='s', utc=True),
            'ema_fast': [99.0, 99.0, 101.0], 'ema_slow': [100.0]*3,
            'rsi': [55.0]*3, 'atr': [2.0]*3})
        self.mt5.positions_get.return_value = ()
        self.mt5.orders_get.return_value = ()
        self.mt5.copy_rates_from_pos.return_value = [{'time': stamp}]
        self.mt5.symbol_info_tick.return_value = types.SimpleNamespace(
            ask=100.0, bid=99.9, time_msc=now*1000)
        with patch.object(self.ea, 'get_rates', return_value=frame), \
             patch.object(self.ea, 'compute_indicators', return_value=frame), \
             patch.object(self.ea, 'latest_loss_time', return_value=now-30), \
             patch.object(self.ea, 'recovery_checks') as reassess:
            self.ea.evaluate_and_maybe_trade('X', 1000)
        reassess.assert_not_called()
        self.mt5.order_send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
