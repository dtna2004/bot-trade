import sys
import os
import csv
import json
import time
import math
import traceback
import collections
from datetime import datetime
from dotenv import load_dotenv
import joblib

import pandas as pd
import numpy as np
import ccxt
import torch
import torch.nn as nn

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QTextEdit,
                               QTableWidget, QTableWidgetItem, QGroupBox,
                               QDialogButtonBox, QFormLayout, QDialog, QMenuBar,
                               QMessageBox, QHeaderView, QSpinBox, QDoubleSpinBox,
                               QComboBox, QCheckBox, QListWidget, QListWidgetItem)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QPalette, QColor

# ==============================================================================
# PHẦN 1: CÁC LỚP ĐỊNH NGHĨA MODEL (Không thay đổi)
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, num_classes)
    def forward(self, src):
        src = self.input_embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output

# ==============================================================================
# PHẦN 2: CÁC LỚP LOGIC GIAO DỊCH
# ==============================================================================

class PaperTradingAccount:
    """Mô phỏng giao dịch trong bộ nhớ, không cần API."""
    def __init__(self, initial_balance=1000, leverage=50, fee=0.001, risk_reward=1.5):
        self.initial_balance, self.balance = initial_balance, initial_balance
        self.positions, self.trade_history = {}, []
        self.leverage, self.fee, self.risk_reward = leverage, fee, risk_reward
        self.csv_file = 'trade_history_paper.csv'
        self._check_and_write_header()

    def place_order(self, symbol, side, amount, price, stop_loss, take_profit):
        if self.has_position(symbol): raise ValueError(f"Vị thế cho {symbol} đã tồn tại.")
        
        # Kiểm tra giá trị lệnh tối thiểu (mô phỏng giống sàn thật)
        notional_value = price * amount
        min_notional = 20  # Giả lập 20 USDT như Binance
        
        if notional_value < min_notional:
            raise ValueError(f"Giá trị lệnh {notional_value:.2f} USDT nhỏ hơn mức tối thiểu {min_notional} USDT. Cần tăng vốn hoặc tăng đòn bẩy.")
        
        order_value = price * amount
        margin_required = order_value / self.leverage
        entry_fee = order_value * self.fee
        if margin_required + entry_fee > self.balance:
            raise ValueError(f"Không đủ số dư. Cần {margin_required + entry_fee:.2f}, có {self.balance:.2f} USDT")
        
        self.balance -= (margin_required + entry_fee)
        position = {
            'symbol': symbol, 'side': side, 'amount': amount, 'entry_price': price,
            'margin': margin_required, 'stop_loss': stop_loss, 'take_profit': take_profit,
            'entry_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'direction': side,
            'fee': entry_fee, 'current_price': price, 'unrealized_pnl': -entry_fee,
            'roe': -(entry_fee / margin_required) * 100 if margin_required > 0 else 0
        }
        self.positions[symbol] = position
        return position

    def update_position(self, symbol, current_price):
        if not self.has_position(symbol): return None
        position = self.positions[symbol]
        position['current_price'] = current_price
        price_diff = current_price - position['entry_price']
        if position['direction'] == 'SHORT': price_diff = -price_diff
        pnl = price_diff * position['amount']
        position['unrealized_pnl'] = pnl - position['fee']
        position['roe'] = (position['unrealized_pnl'] / position['margin']) * 100 if position['margin'] > 0 else 0
        hit_sl = (position['direction'] == 'LONG' and current_price <= position['stop_loss']) or \
                 (position['direction'] == 'SHORT' and current_price >= position['stop_loss'])
        hit_tp = (position['direction'] == 'LONG' and current_price >= position['take_profit']) or \
                 (position['direction'] == 'SHORT' and current_price <= position['take_profit'])
        if hit_sl or hit_tp:
            return self.close_position(symbol, current_price, "Stop Loss" if hit_sl else "Take Profit")
        return None

    def close_position(self, symbol, current_price, reason="Manual"):
        if not self.has_position(symbol): return None
        position = self.positions.pop(symbol)
        price_diff = current_price - position['entry_price']
        if position['direction'] == 'SHORT': price_diff = -price_diff
        pnl = price_diff * position['amount']
        exit_fee = (current_price * position['amount']) * self.fee
        realized_pnl = pnl - position['fee'] - exit_fee
        self.balance += (position['margin'] + realized_pnl)
        final_roe = (realized_pnl / position['margin']) * 100 if position['margin'] > 0 else 0
        trade_record = {
            'entry_time': position['entry_time'], 'exit_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': symbol, 'direction': position['direction'], 'entry_price': position['entry_price'],
            'exit_price': current_price, 'amount': position['amount'], 'leverage': self.leverage,
            'sl': position['stop_loss'], 'tp': position['take_profit'], 'pnl': realized_pnl, 'roe': final_roe,
            'fee': position['fee'] + exit_fee, 'reason': reason
        }
        self.trade_history.append(trade_record)
        self._save_trade_to_csv(trade_record)
        return trade_record

    def get_all_positions(self): return list(self.positions.values())
    def has_position(self, symbol): return symbol in self.positions
    def get_balance_info(self):
        used_margin = sum(pos['margin'] for pos in self.positions.values())
        unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        equity = self.balance + used_margin + unrealized_pnl
        return {'equity': equity, 'available_balance': self.balance,
                'used_margin': used_margin, 'unrealized_pnl': unrealized_pnl}

    def _check_and_write_header(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Entry Time', 'Exit Time', 'Symbol', 'Direction', 'Reason', 'Entry Price', 'Exit Price',
                                 'Amount', 'Leverage', 'Stop Loss', 'Take Profit', 'PnL (USDT)', 'ROE (%)', 'Fee (USDT)'])

    def _save_trade_to_csv(self, trade):
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([trade['entry_time'], trade['exit_time'], trade['symbol'], trade['direction'],
                             trade.get('reason', 'N/A'), f"{trade['entry_price']:.4f}", f"{trade['exit_price']:.4f}",
                             f"{trade['amount']:.6f}", f"{trade['leverage']}x", f"{trade['sl']:.4f}", f"{trade['tp']:.4f}",
                             f"{trade['pnl']:.4f}", f"{trade['roe']:.2f}", f"{trade['fee']:.4f}"])

class LiveTradingAccount:
    """Quản lý giao dịch thực tế qua API (cho cả Live và Testnet)."""
    def __init__(self, exchange):
        self.exchange = exchange
        self.csv_file = 'trade_history_live.csv'
        self._check_and_write_header()

    def place_order(self, symbol, side, amount, price, stop_loss, take_profit):
        if self.has_position(symbol): raise ValueError(f"Vị thế cho {symbol} đã tồn tại.")
        
        try:
            # Lấy thông tin thị trường
            market_info = self.exchange.market(symbol)
            # Chuyển đổi từ LONG/SHORT sang buy/sell
            order_side = 'buy' if side == 'LONG' else 'sell'
            sl_tp_side = 'sell' if order_side == 'buy' else 'buy'
            
            # Kiểm tra giá trị lệnh
            notional_value = amount * price
            min_notional = float(market_info.get('limits', {}).get('cost', {}).get('min', 20))
            
            if notional_value < min_notional:
                # Tính lại số lượng để đạt notional tối thiểu
                adjusted_amount = min_notional / price
                if step_size > 0:
                    precision = int(-math.log10(step_size))
                    # Làm tròn lên đến độ chính xác của step_size
                    adjusted_amount = math.ceil(adjusted_amount / step_size) * step_size
                    # Kiểm tra nếu giá trị lệnh vượt quá quá nhiều so với min_notional
                    potential_value = adjusted_amount * price
                    if potential_value > min_notional * 1.1:  # Nếu vượt quá 10%
                        # Thử giảm xuống 1 step_size
                        reduced_amount = adjusted_amount - step_size
                        reduced_value = reduced_amount * price
                        if reduced_value >= min_notional:
                            adjusted_amount = reduced_amount
                
                print(f"Điều chỉnh số lượng từ {amount} lên {adjusted_amount} để đạt giá trị lệnh tối thiểu {min_notional} USDT")
                amount = adjusted_amount
            
            # Điều chỉnh số lượng theo độ chính xác của sàn
            min_amount = float(market_info['limits']['amount']['min'])
            step_size = float(market_info.get('precision', {}).get('amount', 0))
            
            # Kiểm tra và điều chỉnh các giá trị bất thường
            if step_size >= 1:
                print(f"⚠️ Phát hiện step_size bất thường ({step_size}) cho {symbol}. Sẽ sử dụng giá trị mặc định 0.00000001.")
                step_size = 0.00000001
                
            if min_amount <= 0:
                print(f"⚠️ Phát hiện min_amount bất thường ({min_amount}) cho {symbol}. Sẽ sử dụng giá trị mặc định 0.00001.")
                min_amount = 0.00001
            
            # Đảm bảo số lượng đủ lớn và phù hợp với step_size
            if step_size > 0:
                precision = int(-math.log10(step_size))
                # Sử dụng floor thay vì ceil để tránh mua quá nhiều
                amount = max(min_amount, math.floor(amount * 10**precision) / 10**precision)
            else:
                amount = max(min_amount, amount)
            
            # Định dạng số lượng và giá theo độ chính xác của sàn
            amount_str = self.exchange.amount_to_precision(symbol, amount)
            sl_price_str = self.exchange.price_to_precision(symbol, stop_loss)
            tp_price_str = self.exchange.price_to_precision(symbol, take_profit)
            
            # Kiểm tra số lượng tối thiểu
            if float(amount_str) < min_amount:
                raise ValueError(f"Số lượng {amount_str} nhỏ hơn mức tối thiểu {min_amount}. Cần tăng vốn hoặc giảm đòn bẩy.")
            
            # Kiểm tra lại giá trị lệnh sau khi điều chỉnh
            final_notional = float(amount_str) * price
            if final_notional < min_notional:
                raise ValueError(f"Giá trị lệnh {final_notional:.2f} USDT vẫn nhỏ hơn mức tối thiểu {min_notional} USDT sau khi điều chỉnh. Cần tăng vốn hoặc tăng đòn bẩy.")
            
            # Đặt lệnh vào
            entry_params = {}
            entry_order = self.exchange.create_order(symbol, 'market', order_side, float(amount_str), params=entry_params)
            
            try:
                # Đợi 1 giây để đảm bảo lệnh vào đã được thực hiện
                time.sleep(1)
                
                # Đặt stop loss - Phải thêm stopPrice cho lệnh stop_market
                sl_params = {
                    'reduceOnly': True,
                    'stopPrice': float(sl_price_str),
                    'closePosition': True  # Sử dụng closePosition thay vì chỉ định số lượng
                }
                sl_order = self.exchange.create_order(
                    symbol, 'stop_market', sl_tp_side, None, None, params=sl_params
                )
                
                # Đặt take profit - Xử lý lỗi PERCENT_PRICE filter
                # Kiểm tra giới hạn giá của thị trường
                current_price = self.exchange.fetch_ticker(symbol)['last']
                price_limits = market_info.get('limits', {}).get('price', {})
                min_price, max_price = price_limits.get('min', 0), price_limits.get('max', float('inf'))
                
                # Điều chỉnh take profit nếu vượt quá giới hạn
                percent_limit = 0.1  # Giả sử giới hạn 10% từ giá hiện tại
                max_tp_diff = current_price * percent_limit
                
                # Điều chỉnh take profit để không vượt quá giới hạn
                if side == 'LONG':  # LONG
                    adjusted_tp = min(take_profit, current_price + max_tp_diff)
                else:  # SHORT
                    adjusted_tp = max(take_profit, current_price - max_tp_diff)
                
                adjusted_tp_str = self.exchange.price_to_precision(symbol, adjusted_tp)
                
                tp_params = {
                    'reduceOnly': True,
                    'closePosition': True  # Sử dụng closePosition thay vì chỉ định số lượng
                }
                tp_order = self.exchange.create_order(
                    symbol, 'take_profit_market', sl_tp_side, None, None, 
                    params={**tp_params, 'stopPrice': float(adjusted_tp_str)}
                )
                
                # Thêm thông tin SL/TP vào kết quả
                entry_order['sl_order'] = sl_order
                entry_order['tp_order'] = tp_order
                
                return entry_order
                
            except Exception as order_error:
                # Nếu có lỗi khi đặt SL/TP, cố gắng đóng vị thế vừa mở
                try:
                    self.exchange.cancel_all_orders(symbol)
                    self.exchange.create_order(
                        symbol, 'market', sl_tp_side, float(amount_str), 
                        params={'reduceOnly': True}
                    )
                except Exception as close_error:
                    raise ValueError(f"Lỗi khi đặt SL/TP và không thể đóng vị thế: {order_error}. Lỗi đóng: {close_error}")
                
                raise ValueError(f"Đã mở vị thế nhưng không thể đặt SL/TP, đã tự động đóng: {order_error}")
                
        except ccxt.InsufficientFunds:
            raise ValueError(f"Không đủ tiền để vào lệnh {symbol}. Hãy nạp thêm tiền hoặc giảm số lượng.")
        except ccxt.InvalidOrder as e:
            if "Invalid leverage" in str(e):
                raise ValueError(f"Đòn bẩy không hợp lệ cho {symbol}. Vui lòng điều chỉnh lại trong cài đặt.")
            if "minimum amount precision" in str(e):
                raise ValueError(f"Số lượng không hợp lệ cho {symbol}. Cần tăng vốn hoặc giảm đòn bẩy.")
            if "PERCENT_PRICE filter limit" in str(e):
                raise ValueError(f"Giá vào lệnh vượt quá giới hạn của sàn. Vui lòng thử lại sau.")
            if "requires a stopPrice" in str(e):
                raise ValueError(f"Lỗi đặt SL/TP: {str(e)}. Vui lòng thử lại với cài đặt khác.")
            raise ValueError(f"Lệnh không hợp lệ: {str(e)}")
        except ccxt.ExchangeError as e:
            raise ConnectionError(f"Lỗi từ sàn giao dịch: {str(e)}")
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Lỗi mạng khi kết nối đến sàn: {str(e)}")
        except ccxt.BaseError as e:
            raise ConnectionError(f"Lỗi API khi đặt lệnh {symbol}: {str(e)}")

    def close_position(self, symbol, current_price=None, reason="Manual"):
        try:
            # Lấy thông tin vị thế
            positions = self.exchange.fetch_positions([symbol])
            position = next((p for p in positions if p.get('symbol') == symbol and float(p.get('contracts', 0)) > 0), None)
            
            if not position:
                return None
                
            # Xác định thông tin vị thế
            side = 'sell' if position['side'] == 'long' else 'buy'
            amount = float(position['contracts'])
            entry_price = float(position['entryPrice'])
            
            try:
                # Hủy tất cả lệnh đang mở của symbol
                self.exchange.cancel_all_orders(symbol)
                
                # Đặt lệnh đóng vị thế
                close_params = {'reduceOnly': True}
                
                # Thử sử dụng closePosition trước
                try:
                    close_order = self.exchange.create_order(
                        symbol, 'market', side, None, None, 
                        params={'closePosition': True}
                    )
                except Exception as e:
                    # Nếu không thành công, sử dụng phương pháp truyền thống
                    close_order = self.exchange.create_order(
                        symbol, 'market', side, amount, 
                        params=close_params
                    )
                
                # Lấy thông tin giá đóng
                try:
                    execution_price = float(close_order.get('price', 0))
                    if execution_price == 0:
                        # Nếu không có giá trong lệnh, lấy giá thị trường hiện tại
                        execution_price = float(self.exchange.fetch_ticker(symbol)['last'])
                except Exception:
                    # Nếu không lấy được giá, sử dụng giá current_price nếu có
                    execution_price = current_price if current_price else entry_price
                
                # Tính PnL
                if position['side'] == 'long':
                    pnl_percent = (execution_price - entry_price) / entry_price * 100 * float(position['leverage'])
                else:
                    pnl_percent = (entry_price - execution_price) / entry_price * 100 * float(position['leverage'])
                
                pnl_usdt = float(position.get('unrealizedPnl', 0))
                
                # Tạo bản ghi giao dịch
                trade_record = {
                    'entry_time': datetime.fromtimestamp(position['timestamp'] / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                    'exit_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'symbol': symbol,
                    'direction': position['side'].upper(),
                    'entry_price': entry_price,
                    'exit_price': execution_price,
                    'amount': amount,
                    'leverage': float(position['leverage']),
                    'pnl': pnl_usdt,
                    'roe': pnl_percent,
                    'reason': reason
                }
                
                self._save_trade_to_csv(trade_record)
                return trade_record
                
            except ccxt.OrderNotFound as e:
                raise ConnectionError(f"Lỗi: Không tìm thấy lệnh khi đóng vị thế {symbol}: {str(e)}")
            except ccxt.ExchangeError as e:
                if "ReduceOnly Order is rejected" in str(e):
                    raise ConnectionError(f"Lỗi: Vị thế có thể đã được đóng trước đó - {str(e)}")
                raise ConnectionError(f"Lỗi từ sàn khi đóng vị thế {symbol}: {str(e)}")
            except ccxt.NetworkError as e:
                raise ConnectionError(f"Lỗi mạng khi đóng vị thế {symbol}: {str(e)}")
            except Exception as e:
                raise ConnectionError(f"Lỗi không xác định khi đóng vị thế {symbol}: {str(e)}")
                
        except ccxt.BaseError as e:
            raise ConnectionError(f"Lỗi API khi đóng vị thế {symbol}: {str(e)}")
        except Exception as e:
            raise ConnectionError(f"Lỗi tổng thể khi đóng vị thế {symbol}: {str(e)}")

    def get_all_positions(self):
        try:
            all_pos = self.exchange.fetch_positions()
            open_pos = [p for p in all_pos if p.get('contracts') and float(p.get('contracts')) > 0]
            return [{
                'symbol': p['symbol'], 'direction': p['side'].upper(), 'margin': float(p.get('initialMargin', 0)),
                'entry_price': float(p['entryPrice']), 'current_price': float(p.get('markPrice', 0)),
                'stop_loss': 0, 'take_profit': 0, 'unrealized_pnl': float(p['unrealizedPnl']),
                'roe': float(p.get('percentage', 0))
            } for p in open_pos]
        except ccxt.BaseError as e:
            raise ConnectionError(f"Lỗi API khi lấy vị thế: {e}")

    def has_position(self, symbol):
        try:
            # Nếu đang gọi has_position quá nhiều lần, sử dụng cache nội bộ
            if hasattr(self, '_positions_cache') and hasattr(self, '_last_position_check'):
                # Nếu đã kiểm tra trong vòng 5 giây gần đây, sử dụng cache
                if time.time() - self._last_position_check < 5:
                    # Kiểm tra trong cache
                    for p in self._positions_cache:
                        if p.get('symbol') == symbol and float(p.get('contracts', 0)) > 0:
                            return True
                    return False
            
            # Cập nhật cache vị thế
            try:
                # Trước tiên, kiểm tra xem đã có _all_positions_cache hay chưa
                # Nếu đã gọi get_all_positions gần đây, sử dụng dữ liệu đó
                if hasattr(self, '_all_positions_cache') and hasattr(self, '_last_all_positions_check'):
                    if time.time() - self._last_all_positions_check < 10:
                        for p in self._all_positions_cache:
                            if p.get('symbol') == symbol:
                                return True
                        return False
                
                # Thử lấy vị thế cho symbol cụ thể
                positions = self.exchange.fetch_positions([symbol])
                self._positions_cache = positions
                self._last_position_check = time.time()
            except Exception as e:
                # Nếu có lỗi, thử lấy tất cả vị thế
                try:
                    all_positions = self.exchange.fetch_positions()
                    self._positions_cache = all_positions
                    self._last_position_check = time.time()
                    # Lưu lại để sử dụng cho các lần gọi sau
                    self._all_positions_cache = all_positions
                    self._last_all_positions_check = time.time()
                except Exception as all_e:
                    print(f"Lỗi khi lấy vị thế: {e}, lỗi khi lấy tất cả vị thế: {all_e}")
                    # Nếu vẫn có lỗi và có cache cũ, sử dụng cache cũ
                    if hasattr(self, '_positions_cache'):
                        pass  # Sử dụng cache cũ
                    else:
                        # Không có cache, trả về False để an toàn
                        return False
                
            # Kiểm tra vị thế
            for p in self._positions_cache:
                if p.get('symbol') == symbol and float(p.get('contracts', 0)) > 0:
                    return True
            return False
        except Exception as e:
            # Ghi log lỗi nhưng vẫn trả về False để tránh crash bot
            print(f"Lỗi kiểm tra vị thế cho {symbol}: {e}")
            return False

    def get_balance_info(self):
        try:
            balance = self.exchange.fetch_balance()
            info = balance.get('info', {})
            return {'equity': float(info.get('totalWalletBalance', 0)),
                    'available_balance': float(info.get('availableBalance', 0)),
                    'used_margin': float(info.get('totalInitialMargin', 0)),
                    'unrealized_pnl': float(info.get('totalUnrealizedProfit', 0))}
        except ccxt.BaseError as e:
            raise ConnectionError(f"Lỗi API khi lấy số dư: {e}")

    def _check_and_write_header(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Entry Time', 'Exit Time', 'Symbol', 'Direction', 'Reason', 'Entry Price', 'Exit Price',
                                 'Amount', 'Leverage', 'PnL (USDT)', 'ROE (%)'])

    def _save_trade_to_csv(self, trade):
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([trade['entry_time'], trade['exit_time'], trade['symbol'], trade['direction'],
                             trade.get('reason', 'N/A'), f"{trade['entry_price']:.4f}", f"{trade['exit_price']:.4f}",
                             f"{trade['amount']:.6f}", f"{trade['leverage']}x", f"{trade['pnl']:.4f}", f"{trade['roe']:.2f}"])

# ==============================================================================
# PHẦN 3: LỚP WORKER
# ==============================================================================
class BotWorker(QThread):
    update_signal = Signal(dict); error_signal = Signal(str)
    status_signal = Signal(dict); trade_closed_signal = Signal(dict)
    
    def __init__(self, api_keys, settings):
        super().__init__()
        self.api_keys, self.settings = api_keys, settings
        self.is_running = True
        self.candle_limit = self.settings.get('candle_limit', 100)
        self.last_data = {}
        self.close_requests = collections.deque()
        self.exchange, self.trading_account, self.model, self.scaler = None, None, None, None
        
        # Thêm khóa để tránh race condition khi đặt lệnh
        self.order_locks = {}
        
        try:
            # Kiểm tra nếu _initialize_objects trả về False
            if not self._initialize_objects():
                self.error_signal.emit("Không thể khởi tạo các đối tượng cần thiết. Bot sẽ dừng.")
                self.is_running = False
        except Exception as e:
            self.error_signal.emit(f"Lỗi khởi tạo đối tượng:\n{traceback.format_exc()}")
            self.is_running = False

    def _initialize_objects(self):
        try:
            mode = self.settings.get('trading_mode', 'PAPER')
            if not mode:
                self.error_signal.emit("Lỗi: Không tìm thấy chế độ giao dịch trong cài đặt.")
                return False
                
            # Kiểm tra danh sách biểu tượng
            if not self.settings.get('active_symbols'):
                self.error_signal.emit("Lỗi: Không có biểu tượng nào được chọn.")
                return False
                
            if mode == 'PAPER':
                self.status_signal.emit({'text': "Khởi tạo tài khoản Paper Trading..."})
                self.trading_account = PaperTradingAccount(
                    initial_balance=self.settings.get('total_capital', 1000), 
                    leverage=self.settings.get('leverage', 50),
                    fee=self.settings.get('fee', 0.0005), 
                    risk_reward=self.settings.get('risk_reward', 1.5))
                self.exchange = ccxt.binance({'options': {'defaultType': 'future'}})
                self.status_signal.emit({'text': "Tài khoản Paper Trading đã sẵn sàng."})
            else:
                is_testnet = (mode == 'TESTNET')
                api_key = self.api_keys['test_key'] if is_testnet else self.api_keys['live_key']
                api_secret = self.api_keys['test_secret'] if is_testnet else self.api_keys['live_secret']
                
                if not api_key or not api_secret:
                    error_msg = f"API key/secret cho {mode} không có trong .env"
                    self.error_signal.emit(error_msg)
                    return False
                    
                self.status_signal.emit({'text': f"Khởi tạo kết nối sàn giao dịch {mode}..."})
                try:
                    self.exchange = ccxt.binance({
                        'apiKey': api_key, 'secret': api_secret,
                        'options': {'defaultType': 'future'},
                        'enableRateLimit': True, 'timeout': 30000,
                    })
                    if is_testnet: 
                        self.exchange.set_sandbox_mode(True)
                        self.status_signal.emit({'text': "Đã bật chế độ Sandbox (Testnet)."})
                    
                    # Kiểm tra kết nối
                    self.status_signal.emit({'text': "Kiểm tra kết nối API..."})
                    self.exchange.fetch_balance()
                    self.status_signal.emit({'text': "Kết nối API thành công."})
                    
                    self.trading_account = LiveTradingAccount(self.exchange)
                    self.status_signal.emit({'text': f"Tài khoản {mode} đã sẵn sàng."})
                except ccxt.AuthenticationError:
                    error_msg = f"Lỗi xác thực API. Vui lòng kiểm tra API key và secret."
                    self.error_signal.emit(error_msg)
                    return False
                except ccxt.NetworkError:
                    error_msg = f"Lỗi mạng khi kết nối đến sàn. Vui lòng kiểm tra kết nối Internet."
                    self.error_signal.emit(error_msg)
                    return False
                except Exception as e:
                    error_msg = f"Lỗi khởi tạo kết nối sàn: {str(e)}"
                    self.error_signal.emit(error_msg)
                    return False
            
            return True
        except Exception as e:
            error_msg = f"Lỗi không xác định khi khởi tạo đối tượng: {str(e)}"
            self.error_signal.emit(error_msg)
            return False

    def _configure_exchange(self):
        try:
            # Kiểm tra chế độ giao dịch
            if self.settings['trading_mode'] == 'PAPER': 
                return True
                
            # Kiểm tra nếu cài đặt yêu cầu bỏ qua cấu hình sàn
            if self.settings.get('skip_exchange_config', False):
                self.status_signal.emit({'text': "⚠️ Đang bỏ qua cấu hình sàn theo yêu cầu."})
                return True
            
            self.status_signal.emit({'text': f"Đang tải thông tin thị trường từ {self.settings['trading_mode']}..."})
            
            try:
                # Đặt timeout ngắn hơn để tránh treo khi API không phản hồi
                self.exchange.timeout = 10000  # 10 giây
                self.exchange.load_markets()
                self.status_signal.emit({'text': "Tải thị trường thành công."})
            except ccxt.ExchangeNotAvailable as e:
                self.error_signal.emit(f"Lỗi: Sàn giao dịch không khả dụng: {str(e)}")
                self.status_signal.emit({'text': "⚠️ Sàn giao dịch không khả dụng. Đang tiếp tục với chế độ hạn chế..."})
                return True  # Tiếp tục mặc dù có lỗi
            except ccxt.NetworkError as e:
                self.error_signal.emit(f"Lỗi mạng khi tải thông tin thị trường: {str(e)}")
                self.status_signal.emit({'text': "⚠️ Lỗi mạng. Đang tiếp tục với chế độ hạn chế..."})
                return True  # Tiếp tục mặc dù có lỗi
            
            # Giới hạn đòn bẩy dựa vào chế độ giao dịch
            is_testnet = (self.settings['trading_mode'] == 'TESTNET')
            requested_leverage = self.settings['leverage']
            
            # Giới hạn đòn bẩy tối đa cho Testnet là 50x
            if is_testnet and requested_leverage > 50:
                self.status_signal.emit({'text': f"⚠️ Đòn bẩy {requested_leverage}x quá cao cho Testnet. Giảm xuống 50x."})
                leverage = 50
            else:
                leverage = requested_leverage
            
            # Cấu hình cho từng biểu tượng
            configured_symbols = []
            for symbol in self.settings.get('active_symbols', []):
                try:
                    self.status_signal.emit({'text': f"--- Cấu hình cho {symbol} ---"})
                    
                    # Thử thiết lập chế độ ký quỹ
                    try:
                        self.exchange.set_margin_mode('ISOLATED', symbol)
                        self.status_signal.emit({'text': f"[{symbol}] Đặt chế độ ISOLATED OK."})
                    except ccxt.ExchangeError as e:
                        if "no need to change margin type" in str(e).lower() or "margin type not modified" in str(e).lower():
                            self.status_signal.emit({'text': f"[{symbol}] Chế độ ký quỹ đã là ISOLATED."})
                        else:
                            self.error_signal.emit(f"Lỗi cài đặt chế độ ký quỹ cho {symbol}: {str(e)}")
                            self.status_signal.emit({'text': f"⚠️ Bỏ qua cấu hình chế độ ký quỹ cho {symbol}."})
                    except Exception as e:
                        self.error_signal.emit(f"Lỗi không xác định khi cài đặt chế độ ký quỹ cho {symbol}: {str(e)}")
                        self.status_signal.emit({'text': f"⚠️ Bỏ qua cấu hình chế độ ký quỹ cho {symbol}."})
                    
                    # Thử thiết lập đòn bẩy
                    try:
                        self.exchange.set_leverage(leverage, symbol)
                        self.status_signal.emit({'text': f"[{symbol}] Đặt đòn bẩy {leverage}x OK."})
                        configured_symbols.append(symbol)
                    except ccxt.ExchangeError as e:
                        if "no need to change leverage" in str(e).lower() or "leverage not modified" in str(e).lower():
                            self.status_signal.emit({'text': f"[{symbol}] Đòn bẩy đã là {leverage}x."})
                            configured_symbols.append(symbol)
                        elif "is not valid" in str(e).lower() and leverage > 20:
                            try:
                                self.status_signal.emit({'text': f"⚠️ Đòn bẩy {leverage}x không hợp lệ cho {symbol}. Thử với 20x..."})
                                self.exchange.set_leverage(20, symbol)
                                self.status_signal.emit({'text': f"[{symbol}] Đặt đòn bẩy 20x OK."})
                                configured_symbols.append(symbol)
                            except Exception as inner_e:
                                self.error_signal.emit(f"Không thể thiết lập đòn bẩy cho {symbol}: {inner_e}")
                                self.status_signal.emit({'text': f"⚠️ Bỏ qua cấu hình đòn bẩy cho {symbol}."})
                        else:
                            self.error_signal.emit(f"Lỗi cài đặt đòn bẩy cho {symbol}: {str(e)}")
                            self.status_signal.emit({'text': f"⚠️ Bỏ qua cấu hình đòn bẩy cho {symbol}."})
                    except ccxt.ExchangeNotAvailable as e:
                        self.error_signal.emit(f"Sàn giao dịch không khả dụng khi cài đặt đòn bẩy cho {symbol}: {str(e)}")
                        self.status_signal.emit({'text': f"⚠️ Sàn giao dịch không khả dụng. Bỏ qua cấu hình..."})
                        break  # Thoát khỏi vòng lặp khi sàn không khả dụng
                    except ccxt.NetworkError as e:
                        self.error_signal.emit(f"Lỗi mạng khi cài đặt đòn bẩy cho {symbol}: {str(e)}")
                        self.status_signal.emit({'text': f"⚠️ Lỗi mạng. Bỏ qua cấu hình..."})
                        break  # Thoát khỏi vòng lặp khi có lỗi mạng
                    except Exception as e:
                        self.error_signal.emit(f"Lỗi không xác định khi cài đặt đòn bẩy cho {symbol}: {str(e)}")
                        self.status_signal.emit({'text': f"⚠️ Bỏ qua cấu hình đòn bẩy cho {symbol}."})
                except Exception as e:
                    self.error_signal.emit(f"Lỗi toàn bộ quá trình cấu hình cho {symbol}: {str(e)}")
            
            # Kiểm tra xem có biểu tượng nào được cấu hình thành công không
            if configured_symbols:
                self.status_signal.emit({'text': f"✅ Cấu hình sàn hoàn tất cho {len(configured_symbols)}/{len(self.settings.get('active_symbols', []))} biểu tượng."})
                return True
            else:
                # Nếu không có biểu tượng nào được cấu hình thành công, đặt cờ bỏ qua cấu hình
                self.settings['skip_exchange_config'] = True
                self.status_signal.emit({'text': "⚠️ Không thể cấu hình cho bất kỳ biểu tượng nào. Sẽ tiếp tục với chế độ hạn chế."})
                return True  # Vẫn tiếp tục để có thể tải dữ liệu
                
        except Exception as e:
            self.error_signal.emit(f"Lỗi cấu hình sàn:\n{traceback.format_exc()}")
            self.status_signal.emit({'text': "⚠️ Lỗi cấu hình sàn. Bạn có muốn tiếp tục với chế độ hạn chế?"})
            
            # Tự động chuyển sang chế độ bỏ qua cấu hình
            self.settings['skip_exchange_config'] = True
            return True  # Vẫn trả về True để bot tiếp tục chạy

    def run(self):
        if not self.is_running: 
            self.error_signal.emit("Bot không thể khởi động do lỗi trong quá trình cài đặt.")
            return
            
        self.status_signal.emit({'text': f"🚀 Bot khởi động ở chế độ {self.settings['trading_mode']}..."})
        
        # Kiểm tra model và scaler
        if not self.load_model_and_scaler(): 
            self.error_signal.emit("Không thể tải model và scaler. Bot sẽ dừng.")
            self.stop()
            return
            
        # Kiểm tra cấu hình sàn - Bây giờ luôn trả về True để bot không dừng
        self._configure_exchange()
            
        self.status_signal.emit({'text': "--- Tải dữ liệu ban đầu ---"})
        
        # Biến kiểm tra lỗi tải dữ liệu
        symbols_loaded = []
        
        for symbol in self.settings.get('active_symbols', []):
            if not self.is_running: return
            try:
                if self.fetch_ohlcv_data(symbol):
                    symbols_loaded.append(symbol)
                else:
                    self.error_signal.emit(f"Không thể tải dữ liệu cho {symbol}.")
            except Exception as e:
                self.error_signal.emit(f"Lỗi tải dữ liệu ban đầu cho {symbol}: {e}")
                # Tiếp tục với các symbols khác
        
        if not symbols_loaded:
            self.error_signal.emit("Không thể tải dữ liệu cho bất kỳ biểu tượng nào. Bot sẽ dừng.")
            self.stop(); return
        
        # Kiểm tra tính hợp lệ của dữ liệu đã tải
        valid_symbols = []
        for symbol in symbols_loaded:
            if symbol in self.last_data and not self.last_data[symbol].empty:
                # Kiểm tra dữ liệu có đủ cột và hàng
                df = self.last_data[symbol]
                if len(df) > 20 and all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                    valid_symbols.append(symbol)
                else:
                    self.error_signal.emit(f"Dữ liệu không hợp lệ cho {symbol}. Thiếu cột hoặc không đủ dữ liệu.")
            else:
                self.error_signal.emit(f"Không tìm thấy dữ liệu cho {symbol} hoặc dữ liệu rỗng.")
        
        if not valid_symbols:
            self.error_signal.emit("Không có biểu tượng nào có dữ liệu hợp lệ. Bot sẽ dừng.")
            self.stop(); return
                
        self.status_signal.emit({'text': f"--- Bắt đầu vòng lặp chính với {len(valid_symbols)} biểu tượng ---"})
        self.status_signal.emit({'text': f"--- Các biểu tượng đang giao dịch: {', '.join(valid_symbols)} ---"})
        
        # Cập nhật danh sách các biểu tượng khả dụng
        active_symbols = valid_symbols
        last_update_time = time.time()
        loop_count = 0
        
        while self.is_running:
            try:
                loop_count += 1
                if loop_count % 10 == 0:
                    self.status_signal.emit({'text': f"Bot đang hoạt động... (vòng lặp thứ {loop_count})"})
                    
                    # Mỗi 100 vòng lặp, kiểm tra kết nối API để đảm bảo vẫn hoạt động
                    if loop_count % 100 == 0 and self.settings['trading_mode'] != 'PAPER':
                        try:
                            self.exchange.fetch_balance()
                            self.status_signal.emit({'text': f"✅ Kết nối API OK. Vòng lặp: {loop_count}"})
                            
                            # Cập nhật vị thế vào settings
                            self.update_settings_with_positions()
                        except Exception as e:
                            self.error_signal.emit(f"⚠️ Kiểm tra kết nối API thất bại: {e}. Thử lại sau...")
                
                current_time = time.time()
                if current_time - last_update_time >= self.settings.get('update_interval', 60):
                    self.status_signal.emit({'text': "Đang cập nhật dữ liệu mới..."})
                    success_count = 0
                    for symbol in active_symbols:
                        if symbol in self.last_data:
                            try:
                                if self.fetch_ohlcv_data(symbol):
                                    success_count += 1
                            except Exception as e:
                                self.error_signal.emit(f"Lỗi cập nhật dữ liệu cho {symbol}: {e}")
                    
                    if success_count > 0:
                        self.status_signal.emit({'text': f"✅ Cập nhật dữ liệu thành công cho {success_count}/{len(active_symbols)} biểu tượng"})
                    else:
                        self.error_signal.emit("⚠️ Không thể cập nhật dữ liệu cho bất kỳ biểu tượng nào")
                        
                    last_update_time = current_time
                
                self.process_close_requests()
                
                if isinstance(self.trading_account, PaperTradingAccount):
                    for symbol in list(self.trading_account.positions.keys()):
                        if symbol in self.last_data:
                            try:
                                closed_trade = self.trading_account.update_position(symbol, self.last_data[symbol]['close'].iloc[-1])
                                if closed_trade: self.trade_closed_signal.emit(closed_trade)
                            except Exception as e:
                                self.error_signal.emit(f"Lỗi cập nhật vị thế cho {symbol}: {e}")
                
                for symbol in active_symbols:
                    if symbol in self.last_data:
                        try:
                            # Kiểm tra lại một lần nữa để đảm bảo không có vị thế hiện tại cho biểu tượng này
                            has_position = False
                            
                            # Kiểm tra trong trading_account
                            if self.trading_account.has_position(symbol):
                                has_position = True
                                
                            # Nếu đang có vị thế, bỏ qua phân tích
                            if has_position:
                                continue
                                
                            # Tiếp tục phân tích nếu không có vị thế
                            analysis = self.analyze_market(self.last_data[symbol])
                            current_price = self.last_data[symbol]['close'].iloc[-1]
                            self.status_signal.emit({'symbol': symbol, 'analysis': analysis, 'price': current_price})
                            
                            if analysis['decision'] != 'NEUTRAL':
                                # Kiểm tra lại một lần nữa trước khi đặt lệnh
                                if self.trading_account.has_position(symbol):
                                    self.status_signal.emit({'text': f"⚠️ Đã có vị thế cho {symbol}, bỏ qua tín hiệu {analysis['decision']}."})
                                    continue
                                
                                # Kiểm tra xem symbol này có đang được đặt lệnh không
                                if symbol in self.order_locks and self.order_locks[symbol]:
                                    self.status_signal.emit({'text': f"⚠️ Đang xử lý lệnh khác cho {symbol}, bỏ qua tín hiệu này."})
                                    continue
                                
                                try:
                                    # Đặt khóa cho symbol này
                                    self.order_locks[symbol] = True
                                    
                                    # Lấy thông tin thị trường để xác định giá trị lệnh tối thiểu
                                    try:
                                        market_info = self.exchange.market(symbol)
                                        min_notional = float(market_info.get('limits', {}).get('cost', {}).get('min', 20))
                                    except Exception as e:
                                        self.error_signal.emit(f"Cảnh báo: Không thể lấy thông tin thị trường cho {symbol}: {e}")
                                        min_notional = 20  # Giá trị mặc định nếu không lấy được thông tin
                                    
                                    capital = self.settings['capital']
                                    leverage = self.settings['leverage']
                                    
                                    # Tính số lượng dựa trên vốn và đòn bẩy
                                    notional_value = capital * leverage  # Đây là giá trị lệnh dự kiến với đòn bẩy
                                    
                                    # Thông báo giá trị lệnh dự kiến
                                    self.status_signal.emit({'text': f"Giá trị lệnh dự kiến cho {symbol}: {notional_value:.2f} USDT (Vốn: {capital} USDT, Đòn bẩy: {leverage}x)"})
                                    
                                    # Kiểm tra giá trị lệnh tối thiểu
                                    if notional_value < min_notional:
                                        required_capital = min_notional / leverage
                                        self.status_signal.emit({'text': f"⚠️ Giá trị lệnh {notional_value:.2f} USDT < {min_notional} USDT. Cần tăng vốn lên tối thiểu {required_capital:.2f} USDT với đòn bẩy {leverage}x."})
                                        
                                        # Tự động điều chỉnh vốn
                                        capital = required_capital
                                        notional_value = capital * leverage
                                        self.status_signal.emit({'text': f"⚠️ Tự động tăng vốn từ {self.settings['capital']} lên {capital:.2f} USDT để đạt giá trị lệnh tối thiểu {min_notional} USDT"})
                                    
                                    # Tính số lượng dựa trên giá trị lệnh và giá hiện tại
                                    amount = notional_value / current_price
                                    
                                    # Hiển thị thông tin chi tiết
                                    self.status_signal.emit({'text': f"Tính toán: {capital} USDT × {leverage}x = {notional_value:.2f} USDT ÷ {current_price:.2f} = {amount:.6f} {symbol.split('/')[0]}"})
                                    
                                    # Lấy thông tin thị trường để điều chỉnh số lượng
                                    try:
                                        min_amount = float(market_info['limits']['amount']['min'])
                                        step_size = float(market_info.get('precision', {}).get('amount', 0))
                                        
                                        # Hiển thị thông tin về step_size và min_amount
                                        self.status_signal.emit({'text': f"Thông tin thị trường: Step size = {step_size}, Min amount = {min_amount}"})
                                        
                                        original_amount = amount  # Lưu giá trị ban đầu để so sánh
                                        
                                        # Đảm bảo số lượng đủ lớn và phù hợp với step_size
                                        if step_size > 0:
                                            # Sử dụng hàm format_amount để làm tròn số lượng
                                            amount = max(min_amount, self.format_amount(amount, step_size))
                                            precision = int(-math.log10(step_size))
                                            self.status_signal.emit({'text': f"Sau khi làm tròn xuống: {amount:.6f} (precision={precision})"})
                                        else:
                                            amount = max(min_amount, amount)
                                            self.status_signal.emit({'text': f"Không có step size, giữ nguyên: {amount:.6f}"})
                                            
                                        # Kiểm tra lại giá trị lệnh sau khi điều chỉnh precision
                                        order_value = amount * current_price
                                        self.status_signal.emit({'text': f"Giá trị lệnh sau làm tròn: {order_value:.2f} USDT (yêu cầu tối thiểu: {min_notional} USDT)"})
                                        
                                        if order_value < min_notional:
                                            # Sử dụng hàm check_notional để điều chỉnh số lượng
                                            adjusted_amount, adjusted_value = self.check_notional(
                                                amount, current_price, min_notional, step_size, min_amount
                                            )
                                            
                                            self.status_signal.emit({'text': f"⚠️ Đã điều chỉnh số lượng từ {original_amount:.6f} lên {adjusted_amount:.6f} để đạt giá trị lệnh tối thiểu {min_notional} USDT (Thực tế: {adjusted_value:.2f} USDT)"})
                                            amount = adjusted_amount
                                    except Exception as e:
                                        self.error_signal.emit(f"Cảnh báo: Không thể điều chỉnh precision cho {symbol}: {e}")
                                    
                                    # Lấy quyết định giao dịch
                                    trading_side = analysis['decision']  # LONG hoặc SHORT
                                    
                                    # Tính stop loss và take profit
                                    if trading_side == 'LONG':
                                        stop_loss = current_price - (current_price * self.settings['risk_percent'])
                                        take_profit = current_price + (current_price * self.settings['risk_percent'] * self.settings['risk_reward'])
                                    else:  # SHORT
                                        stop_loss = current_price + (current_price * self.settings['risk_percent'])
                                        take_profit = current_price - (current_price * self.settings['risk_percent'] * self.settings['risk_reward'])
                                    
                                    # Đặt lệnh
                                    try:
                                        trade = self.trading_account.place_order(
                                            symbol=symbol,
                                            side=trading_side,
                                            amount=amount,
                                            price=current_price,
                                            stop_loss=stop_loss,
                                            take_profit=take_profit
                                        )
                                        self.status_signal.emit({'text': f"✅ Đã vào lệnh {trading_side} cho {symbol} với số lượng {amount:.6f} @ {current_price:.4f}"})
                                    except (ValueError, ConnectionError) as e:
                                        self.error_signal.emit(f"Lỗi vào lệnh {symbol}: {e}")
                                finally:
                                    # Giải phóng khóa sau 10 giây
                                    def release_lock(symbol):
                                        self.order_locks[symbol] = False
                                    
                                    # Giải phóng khóa ngay lập tức trong trường hợp Paper Trading
                                    if isinstance(self.trading_account, PaperTradingAccount):
                                        self.order_locks[symbol] = False
                                    else:
                                        # Đối với Live/Testnet, đợi 10 giây để đảm bảo API đã xử lý
                                        QTimer.singleShot(10000, lambda: release_lock(symbol))
                        except Exception as e:
                            self.error_signal.emit(f"Lỗi phân tích {symbol}: {e}")
                
                try:
                    self.update_signal.emit({
                        'positions': self.trading_account.get_all_positions(),
                        'balance': self.trading_account.get_balance_info()})
                except Exception as e:
                    self.error_signal.emit(f"Lỗi cập nhật giao diện: {e}")
                
                self.msleep(int(self.settings.get('analysis_interval', 5) * 1000))
            except ccxt.NetworkError as e:
                self.error_signal.emit(f"Lỗi mạng: {e}. Thử lại sau 10s...")
                self.msleep(10000)
            except Exception as e:
                self.error_signal.emit(f"Lỗi nghiêm trọng trong worker:\n{traceback.format_exc()}")
                # Không dừng bot khi gặp lỗi, chỉ ghi nhận và tiếp tục
                self.msleep(5000)

    def fetch_ohlcv_data(self, symbol):
        try:
            if not self.exchange:
                self.error_signal.emit("Lỗi: Đối tượng exchange chưa được khởi tạo.")
                return False
                
            self.status_signal.emit({'text': f"⏳ Tải dữ liệu {symbol}..."})
            
            # Thử tải dữ liệu từ API
            try:
                # Đặt timeout ngắn hơn để tránh treo
                self.exchange.timeout = 10000  # 10 giây
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', limit=self.candle_limit)
                
                # Nếu không có dữ liệu trả về
                if not ohlcv or len(ohlcv) < 10:  # Yêu cầu ít nhất 10 nến
                    if symbol in self.last_data and not self.last_data[symbol].empty:
                        self.status_signal.emit({'text': f"⚠️ Không đủ dữ liệu cho {symbol}. Sử dụng dữ liệu đã lưu."})
                        return True
                    else:
                        self.error_signal.emit(f"Không nhận được đủ dữ liệu cho {symbol}.")
                        return False
                
                # Chuyển đổi dữ liệu thành DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Nếu có dữ liệu trước đó, kết hợp với dữ liệu mới
                if symbol in self.last_data and not self.last_data[symbol].empty:
                    # Lấy dữ liệu mới nhất
                    new_data = df[~df.index.isin(self.last_data[symbol].index)]
                    if not new_data.empty:
                        combined_df = pd.concat([self.last_data[symbol], new_data])
                        self.last_data[symbol] = combined_df.tail(self.candle_limit + 50)
                        self.status_signal.emit({'text': f"✅ Cập nhật {len(new_data)} nến mới cho {symbol}."})
                    else:
                        self.status_signal.emit({'text': f"ℹ️ Không có dữ liệu mới cho {symbol}."})
                else:
                    # Lưu dữ liệu mới
                    self.last_data[symbol] = df.tail(self.candle_limit + 50)
                    self.status_signal.emit({'text': f"✅ Tải xong {len(df)} nến cho {symbol}."})
                
                # Lưu dữ liệu vào cache
                self._save_data_to_cache(symbol, self.last_data[symbol])
                
                return True
                
            except ccxt.ExchangeNotAvailable as e:
                self.error_signal.emit(f"Sàn giao dịch không khả dụng khi tải dữ liệu {symbol}: {str(e)}")
                
                # Thử tải dữ liệu từ cache
                cached_data = self._load_data_from_cache(symbol)
                if cached_data is not None:
                    self.last_data[symbol] = cached_data
                    self.status_signal.emit({'text': f"⚠️ Sử dụng dữ liệu cache cho {symbol} (sàn không khả dụng)."})
                    return True
                    
                return False
                
            except ccxt.NetworkError as e:
                self.error_signal.emit(f"Lỗi mạng khi tải dữ liệu {symbol}: {str(e)}")
                
                # Thử tải dữ liệu từ cache
                cached_data = self._load_data_from_cache(symbol)
                if cached_data is not None:
                    self.last_data[symbol] = cached_data
                    self.status_signal.emit({'text': f"⚠️ Sử dụng dữ liệu cache cho {symbol} (lỗi mạng)."})
                    return True
                    
                return False
                
        except Exception as e:
            self.error_signal.emit(f"Lỗi tải dữ liệu {symbol}: {e}")
            return False
            
    def _save_data_to_cache(self, symbol, df):
        """Lưu dữ liệu vào cache để sử dụng khi API không khả dụng."""
        try:
            if df is None or df.empty:
                return
                
            # Tạo thư mục cache nếu chưa tồn tại
            os.makedirs('data_cache', exist_ok=True)
            
            # Lưu DataFrame
            cache_file = os.path.join('data_cache', f"{symbol.replace('/', '_')}_cache.csv")
            df.to_csv(cache_file)
        except Exception as e:
            self.error_signal.emit(f"Lỗi khi lưu cache cho {symbol}: {e}")
            
    def _load_data_from_cache(self, symbol):
        """Tải dữ liệu từ cache khi API không khả dụng."""
        try:
            cache_file = os.path.join('data_cache', f"{symbol.replace('/', '_')}_cache.csv")
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            return None
        except Exception as e:
            self.error_signal.emit(f"Lỗi khi tải cache cho {symbol}: {e}")
            return None

    def process_close_requests(self):
        while self.close_requests:
            symbol_to_close = self.close_requests.popleft()
            try:
                if symbol_to_close == "__ALL__":
                    for pos in self.trading_account.get_all_positions():
                        trade_record = self.trading_account.close_position(pos['symbol'], reason="Manual Close All")
                        if trade_record: self.trade_closed_signal.emit(trade_record)
                else:
                    trade_record = self.trading_account.close_position(symbol_to_close, reason="Manual Close")
                    if trade_record: self.trade_closed_signal.emit(trade_record)
            except (ConnectionError, ValueError) as e:
                self.error_signal.emit(f"Lỗi đóng lệnh: {e}")

    def load_model_and_scaler(self):
        try:
            config_path = os.path.join("trained_model", "model_config.json")
            if not os.path.exists(config_path): raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")
            with open(config_path, 'r') as f: config = json.load(f)
            if config.get('num_classes') != 2:
                self.error_signal.emit("Lỗi: Model phải được huấn luyện cho 2 lớp (Tăng/Giảm)."); return False
            self.model = TransformerModel(
                input_dim=config['input_dim'], d_model=config['d_model'],
                nhead=config['nhead'], num_layers=config['num_layers'],
                num_classes=config['num_classes'], dropout=config['dropout']
            ).to(torch.device("cpu"))
            model_path = os.path.join("trained_model", "transformer_btc_trader_statedict.pth")
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            self.model.eval()
            scaler_path = os.path.join("trained_model", "scaler.gz")
            self.scaler = joblib.load(scaler_path)
            self.status_signal.emit({'text': "✅ Model và Scaler đã được tải thành công."})
            return True
        except Exception as e:
            self.error_signal.emit(f"Lỗi khi tải model: {e}"); return False
            
    def format_amount(self, amount, step_size):
        """
        Định dạng số lượng theo đúng độ chính xác dựa vào step_size.
        """
        if step_size == 0:
            return amount
            
        # Nếu step_size ≥ 1, đây có thể là lỗi thông tin thị trường
        if step_size >= 1:
            self.status_signal.emit({
                'text': f"⚠️ Phát hiện step_size bất thường ({step_size}) trong format_amount. Sẽ sử dụng precision=8 thay thế."
            })
            return math.floor(amount * 100000000) / 100000000
            
        precision = int(-math.log10(step_size))
        return math.floor(amount * 10**precision) / 10**precision
        
    def check_notional(self, amount, price, min_notional, step_size, min_amount):
        """
        Kiểm tra và điều chỉnh số lượng để đạt giá trị notional tối thiểu.
        Trả về số lượng đã điều chỉnh và giá trị lệnh mới.
        """
        order_value = amount * price
        
        # Thêm thông báo chi tiết
        self.status_signal.emit({
            'text': f"Chi tiết: min_notional={min_notional}, min_amount={min_amount}, step_size={step_size}, hiện tại: {amount:.6f} ({order_value:.2f} USDT)"
        })
        
        if order_value >= min_notional:
            return amount, order_value
            
        # Tính số lượng tối thiểu để đạt min_notional
        min_notional_amount = min_notional / price
        
        # Đảm bảo đúng độ chính xác
        if step_size > 0:
            precision = int(-math.log10(step_size))
            
            # Nếu step_size ≥ 1, đây có thể là lỗi thông tin thị trường
            if step_size >= 1:
                self.status_signal.emit({
                    'text': f"⚠️ Phát hiện step_size bất thường ({step_size}). Sẽ sử dụng precision=8 thay thế."
                })
                precision = 8
                step_size = 0.00000001
            
            min_notional_amount = math.ceil(min_notional_amount / step_size) * step_size
            
            # Kiểm tra nếu giá trị lệnh vượt quá quá nhiều so với min_notional
            potential_value = min_notional_amount * price
            if potential_value > min_notional * 1.05:  # Nếu vượt quá 5%
                # Thử giảm xuống 1 step_size
                reduced_amount = min_notional_amount - step_size
                reduced_value = reduced_amount * price
                if reduced_value >= min_notional:
                    min_notional_amount = reduced_amount
        
        # Đảm bảo không nhỏ hơn min_amount
        if min_amount > 0:
            min_notional_amount = max(min_amount, min_notional_amount)
        else:
            # Nếu min_amount = 0, đây có thể là lỗi thông tin thị trường
            self.status_signal.emit({
                'text': f"⚠️ Phát hiện min_amount bất thường (0). Sẽ sử dụng giá trị mặc định."
            })
        
        # Tính lại giá trị lệnh
        final_value = min_notional_amount * price
        
        return min_notional_amount, final_value

    def get_feature_columns(self): return ['close', 'volume', 'returns', 'rsi', 'upper_band', 'lower_band']
    def calculate_indicators(self, df):
        df_copy = df.copy()
        df_copy['returns'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
        delta = df_copy['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        df_copy['rsi'] = 100 - (100 / (1 + (gain / loss)))
        df_copy['sma_20'] = df_copy['close'].rolling(window=20).mean()
        std_20 = df_copy['close'].rolling(window=20).std()
        df_copy['upper_band'] = df_copy['sma_20'] + (std_20 * 2)
        df_copy['lower_band'] = df_copy['sma_20'] - (std_20 * 2)
        return df_copy
    def preprocess_data(self, df):
        if self.scaler is None or self.model is None: return None
        df_with_indicators = self.calculate_indicators(df)
        features = df_with_indicators[self.get_feature_columns()].copy().ffill().bfill().fillna(0)
        scaled_data = self.scaler.transform(features)
        return torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)
    def analyze_market(self, df):
        input_tensor = self.preprocess_data(df)
        if input_tensor is None: return {'long': 0, 'short': 0, 'diff': 0, 'decision': 'NEUTRAL'}
        
        # Lấy ngưỡng tín hiệu từ cài đặt
        threshold = self.settings.get('signal_threshold', 10.0)
        
        with torch.no_grad():
            probabilities = torch.softmax(self.model(input_tensor), dim=1)[0]
        
        prob_short, prob_long = probabilities[0].item() * 100, probabilities[1].item() * 100
        prob_diff = prob_long - prob_short
        decision = 'NEUTRAL'
        
        if prob_diff > threshold: 
            decision = 'LONG'
        elif prob_diff < -threshold: 
            decision = 'SHORT'
            
        return {
            'long': prob_long, 
            'short': prob_short, 
            'diff': prob_diff, 
            'decision': decision,
            'threshold': threshold
        }
    def stop(self): self.is_running = False
    def handle_close_request(self, symbol: str): self.close_requests.append(symbol)
    def handle_close_all_request(self): self.close_requests.append("__ALL__")

    def update_settings_with_positions(self):
        """Cập nhật trạng thái vị thế hiện tại vào settings để lưu lại."""
        try:
            # Lấy danh sách các vị thế hiện tại
            positions = self.trading_account.get_all_positions()
            
            # Lưu danh sách symbol của các vị thế
            position_symbols = [pos['symbol'] for pos in positions]
            
            # Cập nhật vào settings
            self.settings['active_positions'] = position_symbols
            
            # Ghi log
            self.status_signal.emit({'text': f"Đã cập nhật {len(position_symbols)} vị thế vào settings."})
            
            return True
        except Exception as e:
            self.error_signal.emit(f"Lỗi cập nhật vị thế vào settings: {e}")
            return False

# ==============================================================================
# PHẦN 4: GIAO DIỆN
# ==============================================================================
class MainWindow(QMainWindow):
    request_close_position = Signal(str); request_close_all_positions = Signal()

    def __init__(self):
        super().__init__()
        load_dotenv()
        self.api_keys = {
            'live_key': os.getenv('BINANCE_API_KEY'), 'live_secret': os.getenv('BINANCE_API_SECRET'),
            'test_key': os.getenv('BINANCE_TESTNET_API_KEY'), 'test_secret': os.getenv('BINANCE_TESTNET_API_SECRET')
        }
        self.available_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
        self.settings_file = "bot_settings.json"; self.settings = {}
        self.load_settings()
        self.bot_worker = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Trading Bot"); self.setMinimumSize(1200, 800)
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Thanh menu
        menubar = self.menuBar()
        settings_menu = menubar.addMenu('Cài đặt')
        settings_menu.addAction('Cài đặt Bot & Vốn', self.show_settings_dialog)
        
        # Hiển thị thông tin cấu hình hiện tại
        config_group = QGroupBox("Cấu hình hiện tại")
        config_layout = QVBoxLayout(config_group)
        
        self.config_label = QLabel()
        self.update_config_display()
        config_layout.addWidget(self.config_label)
        
        main_layout.addWidget(config_group)
        
        # Layout nội dung chính
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        # Panel bên trái
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        content_layout.addWidget(left_panel, 2)
        
        # Khu vực trạng thái và phân tích
        status_group = QGroupBox("Trạng thái & Phân tích")
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        status_group.setLayout(QVBoxLayout())
        status_group.layout().addWidget(self.status_text)
        left_layout.addWidget(status_group)
        
        # Khu vực vị thế hiện tại
        position_group = QGroupBox("Vị thế hiện tại")
        position_layout = QVBoxLayout()
        position_group.setLayout(position_layout)
        
        self.position_table = QTableWidget()
        self.position_table.setColumnCount(10)
        self.position_table.setHorizontalHeaderLabels(["Symbol", "Hướng", "Ký quỹ", "Giá vào", "Giá HT", "SL", "TP", "PnL", "ROE (%)", ""])
        self.position_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        position_layout.addWidget(self.position_table)
        
        pnl_layout = QHBoxLayout()
        self.total_pnl_label = QLabel("Tổng PnL: 0.00 USDT")
        pnl_layout.addStretch()
        pnl_layout.addWidget(self.total_pnl_label)
        position_layout.addLayout(pnl_layout)
        
        left_layout.addWidget(position_group)
        
        # Panel bên phải
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        content_layout.addWidget(right_panel, 1)
        
        # Điều khiển
        control_group = QGroupBox("Điều khiển")
        control_layout = QVBoxLayout()
        control_group.setLayout(control_layout)
        
        self.start_button = QPushButton("Bắt đầu")
        self.start_button.clicked.connect(self.start_bot)
        
        self.stop_button = QPushButton("Dừng")
        self.stop_button.clicked.connect(self.stop_bot)
        self.stop_button.setEnabled(False)
        
        close_all_button = QPushButton("Đóng tất cả")
        close_all_button.clicked.connect(self.confirm_close_all_positions)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(close_all_button)
        
        right_layout.addWidget(control_group)
        
        # Tài khoản
        balance_group = QGroupBox("Tài khoản")
        self.balance_text = QTextEdit()
        self.balance_text.setReadOnly(True)
        self.balance_text.setMaximumHeight(120)
        balance_group.setLayout(QVBoxLayout())
        balance_group.layout().addWidget(self.balance_text)
        
        right_layout.addWidget(balance_group)
        right_layout.addStretch()

    def update_config_display(self):
        mode = self.settings.get('trading_mode', 'PAPER')
        mode_display = {'PAPER': 'Mô phỏng', 'TESTNET': 'Testnet', 'LIVE': 'THẬT'}
        
        active_symbols = self.settings.get('active_symbols', [])
        symbols_text = ", ".join(active_symbols) if active_symbols else "Chưa chọn"
        
        config_text = (
            f"<b>Chế độ:</b> {mode_display.get(mode, mode)} | "
            f"<b>Vốn/lệnh:</b> {self.settings.get('capital', 10)} USDT | "
            f"<b>Đòn bẩy:</b> {self.settings.get('leverage', 20)}x | "
            f"<b>Ngưỡng tín hiệu:</b> {self.settings.get('signal_threshold', 10.0)}% | "
            f"<b>R:R:</b> {self.settings.get('risk_reward', 1.5)} | "
            f"<b>Cặp:</b> {symbols_text}"
        )
        
        self.config_label.setText(config_text)
        
        # Cập nhật tiêu đề cửa sổ
        mode_map = {'PAPER': '[PAPER]', 'TESTNET': '[TESTNET]', 'LIVE': '[LIVE]'}
        self.setWindowTitle(f"{mode_map.get(mode, '[?]')} Trading Bot")

    def on_symbol_changed(self):
        # Phương thức này không cần thiết nữa vì chúng ta xử lý cặp giao dịch trong dialog cài đặt
        pass

    def show_settings_dialog(self):
        dialog = SettingsDialog(self, self.settings)
        if dialog.exec():
            self.settings.update(dialog.get_settings())
            self.save_settings()
            self.load_settings()
            self.update_config_display()
            self.update_status({'text': "Cài đặt đã được cập nhật."})

    def start_bot(self):
        # Kiểm tra cài đặt trước khi khởi động bot
        if not self.settings.get('active_symbols'):
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng chọn ít nhất một cặp giao dịch.")
            return
            
        # Hiển thị thông tin về chế độ giao dịch
        mode = self.settings.get('trading_mode', 'PAPER')
        if mode == 'LIVE':
            result = QMessageBox.warning(
                self, 
                "Cảnh báo - Giao dịch THẬT", 
                "Bạn sắp bắt đầu bot ở chế độ LIVE với tiền thật!\n\nXác nhận tiếp tục?", 
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if result != QMessageBox.StandardButton.Yes:
                return
                
        # Hiển thị thông tin trạng thái
        self.status_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Đang khởi động bot ở chế độ {mode}...")
        self.status_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Các cặp giao dịch: {', '.join(self.settings['active_symbols'])}")
        
        # Tạo và cấu hình worker
        self.bot_worker = BotWorker(self.api_keys, self.settings)
        self.bot_worker.update_signal.connect(self.update_ui_data)
        self.bot_worker.error_signal.connect(self.show_error)
        self.bot_worker.status_signal.connect(self.update_status)
        self.bot_worker.trade_closed_signal.connect(self.handle_trade_closed)
        self.request_close_position.connect(self.bot_worker.handle_close_request)
        self.request_close_all_positions.connect(self.bot_worker.handle_close_all_request)
        
        # Bắt đầu worker
        self.bot_worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_bot(self):
        if self.bot_worker:
            self.bot_worker.stop(); self.bot_worker.wait(5000); self.bot_worker = None
        self.start_button.setEnabled(True); self.stop_button.setEnabled(False)
        self.update_status({'text': "Bot đã dừng."}); self.position_table.setRowCount(0)
        self.total_pnl_label.setText("Tổng PnL: 0.00 USDT")
    
    def confirm_close_all_positions(self):
        if QMessageBox.question(self, 'Xác nhận', 'Đóng tất cả vị thế?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes and self.bot_worker:
            self.request_close_all_positions.emit()

    def confirm_close_position(self, row):
        symbol = self.position_table.item(row, 0).text()
        if QMessageBox.question(self, 'Xác nhận', f'Đóng vị thế {symbol}?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes and self.bot_worker:
            self.request_close_position.emit(symbol)

    def update_ui_data(self, data):
        positions = data.get('positions', [])
        self.position_table.setRowCount(len(positions)); total_pnl = 0
        for i, pos in enumerate(positions):
            pnl, roe = pos.get('unrealized_pnl', 0), pos.get('roe', 0)
            total_pnl += pnl
            items = [
                pos.get('symbol', ''), pos.get('direction', ''), f"{pos.get('margin', 0):.2f}",
                f"{pos.get('entry_price', 0):.4f}", f"{pos.get('current_price', 0):.4f}",
                f"{pos.get('stop_loss', 0):.4f}", f"{pos.get('take_profit', 0):.4f}",
                f"{pnl:+.2f}", f"{roe:+.2f}%"
            ]
            for j, item_text in enumerate(items):
                item = QTableWidgetItem(item_text)
                if j in [7, 8]: item.setForeground(QColor('#2ecc71' if (pnl if j==7 else roe) >= 0 else '#e74c3c'))
                self.position_table.setItem(i, j, item)
            close_button = QPushButton("X"); close_button.setFixedSize(25, 25)
            close_button.clicked.connect(lambda ch, r=i: self.confirm_close_position(r))
            self.position_table.setCellWidget(i, 9, close_button)
        self.total_pnl_label.setText(f"Tổng PnL: {total_pnl:+.2f} USDT")
        self.total_pnl_label.setStyleSheet(f"font-weight: bold; color: {'#2ecc71' if total_pnl >= 0 else '#e74c3c'};")
        balance_info = data.get('balance', {})
        self.balance_text.setText(f"Vốn (Equity): {balance_info.get('equity', 0):.2f}\n"
                                  f"Khả dụng: {balance_info.get('available_balance', 0):.2f}\n"
                                  f"Ký quỹ: {balance_info.get('used_margin', 0):.2f}\n"
                                  f"Lãi/Lỗ: {balance_info.get('unrealized_pnl', 0):.2f}")
    
    def show_error(self, error_msg):
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_error = f"<font color='#e74c3c'><b>[{timestamp}] ❌ LỖI:</b> {error_msg}</font>"
        self.status_text.append(formatted_error)
        
        # Nếu là lỗi nghiêm trọng, hiển thị hộp thoại
        if "Bot sẽ dừng" in error_msg or "Không thể khởi động" in error_msg:
            QMessageBox.critical(self, "Lỗi Bot", error_msg)
        
        # Tự động cuộn xuống cuối
        self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())
    
    def update_status(self, data):
        timestamp = f"[{datetime.now().strftime('%H:%M:%S')}]"
        
        if 'analysis' in data:
            symbol, price, analysis = data['symbol'], data['price'], data['analysis']
            decision = analysis['decision']
            
            # Định dạng dựa trên quyết định
            if decision == 'LONG':
                decision_text = f"<font color='#2ecc71'><b>QUYẾT ĐỊNH: LONG</b></font>"
            elif decision == 'SHORT':
                decision_text = f"<font color='#e74c3c'><b>QUYẾT ĐỊNH: SHORT</b></font>"
            else:
                decision_text = f"<font color='#95a5a6'>QUYẾT ĐỊNH: ĐỨNG NGOÀI</font>"
            
            # Định dạng tỉ lệ Long/Short
            long_ratio = analysis.get('long', 0)
            short_ratio = analysis.get('short', 0)
            ratio_text = f"(Long: <font color='#2ecc71'>{long_ratio:.2f}%</font>, Short: <font color='#e74c3c'>{short_ratio:.2f}%</font>)"
                
            # Hiển thị phân tích với định dạng đẹp hơn
            status_html = (
                f"{timestamp} <b>{symbol}</b> @ ${price:,.4f} | "
                f"Chênh lệch: <b>{analysis['diff']:+.2f}%</b> {ratio_text} (Ngưỡng: {analysis['threshold']:.1f}%) "
                f"-> {decision_text}"
            )
            self.status_text.append(status_html)
            
        elif 'text' in data:
            text = data['text']
            
            # Định dạng màu sắc dựa trên loại thông báo
            if "✅" in text:
                status_html = f"{timestamp} <font color='#2ecc71'>{text}</font>"
            elif "⚠️" in text:
                status_html = f"{timestamp} <font color='#f39c12'>{text}</font>"
            elif "⏳" in text:
                status_html = f"{timestamp} <font color='#3498db'>{text}</font>"
            elif "--- " in text and " ---" in text:
                status_html = f"{timestamp} <b><font color='#9b59b6'>{text}</font></b>"
            elif "Giá trị lệnh" in text:
                # Định dạng thông tin về giá trị lệnh để nổi bật
                status_html = f"{timestamp} <b><font color='#3498db'>{text}</font></b>"
            elif "Tính toán" in text:
                # Định dạng thông tin về tính toán số lượng để nổi bật
                status_html = f"{timestamp} <font color='#2980b9'>{text}</font>"
            else:
                status_html = f"{timestamp} {text}"
                
            self.status_text.append(status_html)
            
        # Tự động cuộn xuống cuối
        self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())

    def handle_trade_closed(self, trade):
        pnl = trade.get('pnl', 0)
        color = "#2ecc71" if pnl >= 0 else "#e74c3c"
        self.status_text.append(f"<br>--- GIAO DỊCH ĐÓNG ({self.settings['trading_mode']}) ---<br>"
                                f"<b>Symbol:</b> {trade['symbol']} ({trade['direction']}) | <b>Lý do:</b> {trade.get('reason', 'N/A')}<br>"
                                f"<b>Lãi/Lỗ:</b> <font color='{color}'>{pnl:+.4f} USDT</font> | "
                                f"<b>ROE:</b> <font color='{color}'>{trade.get('roe', 0):+.2f}%</font><br>"
                                f"-----------------------")
        if pnl > 0: QApplication.beep()

    def closeEvent(self, event): self.stop_bot(); event.accept()

    def load_settings(self):
        default_settings = {
            'trading_mode': 'PAPER', 'total_capital': 1000, 'capital': 10, 'leverage': 20,
            'risk_reward': 1.5, 'fee': 0.0005, 'active_symbols': ['BTC/USDT'], 'signal_threshold': 10.0,
            'risk_percent': 0.01, 'update_interval': 60, 'analysis_interval': 5, 'candle_limit': 100
        }
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f: self.settings = {**default_settings, **json.load(f)}
            else: self.settings = default_settings
            self.save_settings()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi Cài Đặt", f"Không thể tải/lưu cài đặt: {e}")

    def save_settings(self):
        try:
            with open(self.settings_file, 'w') as f: json.dump(self.settings, f, indent=4)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi Cài Đặt", f"Không thể lưu cài đặt: {e}")

class SettingsDialog(QDialog):
    def __init__(self, parent, current_settings):
        super().__init__(parent); self.setWindowTitle("Cài đặt Bot"); self.setModal(True)
        self.settings, self.api_keys = current_settings.copy(), parent.api_keys
        layout, form_layout = QVBoxLayout(self), QFormLayout()
        
        # Chế độ giao dịch
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Paper Trading", "Testnet (Binance)", "Live Trading (Binance)"])
        mode_map = {"Paper Trading": "PAPER", "Testnet (Binance)": "TESTNET", "Live Trading (Binance)": "LIVE"}
        self.mode_rev_map = {v: k for k, v in mode_map.items()}
        self.mode_combo.setCurrentText(self.mode_rev_map.get(self.settings.get('trading_mode', 'PAPER')))

        # Inputs
        self.total_capital_input = QDoubleSpinBox()
        self.capital_input = QDoubleSpinBox()
        self.leverage_input = QSpinBox()
        self.risk_percent_input = QDoubleSpinBox()
        self.reward_input = QDoubleSpinBox()
        self.fee_input = QDoubleSpinBox()
        self.signal_threshold_input = QDoubleSpinBox()
        self.update_interval_input = QSpinBox()
        self.analysis_interval_input = QSpinBox()

        # Cấu hình các input
        self.total_capital_input.setRange(10, 1e6); self.total_capital_input.setValue(self.settings.get('total_capital', 1000)); self.total_capital_input.setSuffix(" USDT")
        self.capital_input.setRange(1, 1e5); self.capital_input.setValue(self.settings.get('capital', 10)); self.capital_input.setSuffix(" USDT")
        self.leverage_input.setRange(1, 125); self.leverage_input.setValue(self.settings.get('leverage', 20)); self.leverage_input.setSuffix("x")
        self.risk_percent_input.setRange(0.1, 10); self.risk_percent_input.setValue(self.settings.get('risk_percent', 0.01) * 100); self.risk_percent_input.setSuffix(" %"); self.risk_percent_input.setDecimals(2)
        self.reward_input.setRange(0.1, 10); self.reward_input.setValue(self.settings.get('risk_reward', 1.5)); self.reward_input.setSingleStep(0.1)
        self.fee_input.setRange(0, 1); self.fee_input.setValue(self.settings.get('fee', 0.0005) * 100); self.fee_input.setSuffix(" %"); self.fee_input.setDecimals(4)
        self.signal_threshold_input.setRange(0.1, 50); self.signal_threshold_input.setValue(self.settings.get('signal_threshold', 10.0)); self.signal_threshold_input.setSuffix(" %"); self.signal_threshold_input.setSingleStep(0.5)
        self.update_interval_input.setRange(5, 3600); self.update_interval_input.setValue(self.settings.get('update_interval', 60)); self.update_interval_input.setSuffix(" giây")
        self.analysis_interval_input.setRange(1, 60); self.analysis_interval_input.setValue(self.settings.get('analysis_interval', 5)); self.analysis_interval_input.setSuffix(" giây")

        # Cặp giao dịch
        symbols_group = QGroupBox("Thêm cặp giao dịch mới")
        symbols_layout = QVBoxLayout(symbols_group)
        
        # Input thêm cặp giao dịch mới
        self.new_symbol_input = QComboBox()
        # Thêm một số cặp giao dịch phổ biến
        popular_symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", 
            "ADA/USDT", "DOGE/USDT", "MATIC/USDT", "DOT/USDT", "AVAX/USDT",
            "LINK/USDT", "LTC/USDT", "UNI/USDT", "ATOM/USDT", "ETC/USDT",
            "SHIB/USDT", "TRX/USDT", "NEAR/USDT", "BCH/USDT", "FIL/USDT"
        ]
        self.new_symbol_input.addItems(popular_symbols)
        self.new_symbol_input.setEditable(True)
        self.new_symbol_input.setCurrentText("")
        
        add_symbol_layout = QHBoxLayout()
        add_symbol_layout.addWidget(self.new_symbol_input)
        add_symbol_btn = QPushButton("Thêm")
        add_symbol_btn.clicked.connect(self.add_new_symbol)
        add_symbol_layout.addWidget(add_symbol_btn)
        symbols_layout.addLayout(add_symbol_layout)
        
        # Danh sách cặp giao dịch hiện tại
        self.symbols_list = QListWidget()
        for symbol in self.settings.get('active_symbols', []):
            item = QListWidgetItem(symbol)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.symbols_list.addItem(item)
        
        # Nút xóa cặp giao dịch
        remove_btn = QPushButton("Xóa cặp đã chọn")
        remove_btn.clicked.connect(self.remove_selected_symbols)
        
        symbols_layout.addWidget(self.symbols_list)
        symbols_layout.addWidget(remove_btn)
        
        # Thêm các widget vào form
        form_layout.addRow("Chế độ giao dịch:", self.mode_combo)
        form_layout.addRow(QLabel("--- Quản lý vốn & Rủi ro ---"))
        form_layout.addRow("Tổng vốn (Paper):", self.total_capital_input)
        form_layout.addRow("Ký quỹ/lệnh:", self.capital_input)
        form_layout.addRow("Đòn bẩy:", self.leverage_input)
        form_layout.addRow("Rủi ro/lệnh (SL):", self.risk_percent_input)
        form_layout.addRow("Tỉ lệ R:R:", self.reward_input)
        form_layout.addRow(QLabel("--- Cấu hình Bot ---"))
        form_layout.addRow("Ngưỡng tín hiệu:", self.signal_threshold_input)
        form_layout.addRow("Tần suất cập nhật dữ liệu:", self.update_interval_input)
        form_layout.addRow("Tần suất phân tích:", self.analysis_interval_input)
        form_layout.addRow("Phí GD (Paper):", self.fee_input)
        
        # Thêm form và group vào layout chính
        layout.addLayout(form_layout)
        layout.addWidget(symbols_group)

        self.info_label = QLabel(); layout.addWidget(self.info_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        self.on_mode_changed(self.mode_combo.currentText())
        
    def add_new_symbol(self):
        new_symbol = self.new_symbol_input.currentText().strip().upper()
        if not new_symbol:
            return
            
        # Kiểm tra nếu cặp giao dịch đã tồn tại
        for i in range(self.symbols_list.count()):
            if self.symbols_list.item(i).text() == new_symbol:
                return  # Đã tồn tại, không thêm nữa
                
        # Thêm cặp giao dịch mới
        item = QListWidgetItem(new_symbol)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        self.symbols_list.addItem(item)
        self.new_symbol_input.setCurrentText("")
        
    def remove_selected_symbols(self):
        for i in range(self.symbols_list.count() - 1, -1, -1):
            if self.symbols_list.item(i).checkState() == Qt.CheckState.Checked:
                self.symbols_list.takeItem(i)

    def on_mode_changed(self, mode_text):
        mode_map = {"Paper Trading": "PAPER", "Testnet (Binance)": "TESTNET", "Live Trading (Binance)": "LIVE"}
        mode = mode_map[mode_text]
        is_paper = (mode == 'PAPER')
        
        # Điều chỉnh giới hạn đòn bẩy dựa trên chế độ
        if mode == 'TESTNET':
            self.leverage_input.setRange(1, 50)
            # Nếu giá trị hiện tại lớn hơn 50, điều chỉnh xuống 50
            if self.leverage_input.value() > 50:
                self.leverage_input.setValue(50)
            self.leverage_input.setToolTip("Đòn bẩy tối đa cho Testnet là 50x")
        elif mode == 'LIVE':
            self.leverage_input.setRange(1, 125)
            self.leverage_input.setToolTip("Đòn bẩy tối đa cho Live là 125x (tùy theo cặp giao dịch)")
        else: # PAPER
            self.leverage_input.setRange(1, 125)
            self.leverage_input.setToolTip("Đòn bẩy tối đa cho Paper Trading là 125x")
        
        self.total_capital_input.setEnabled(is_paper); self.fee_input.setEnabled(is_paper)
        if is_paper:
            self.info_label.setText("Paper Trading: Mô phỏng giao dịch trong bộ nhớ."); self.info_label.setStyleSheet("color: #4db6ac;")
        else:
            is_testnet = (mode == 'TESTNET')
            self.info_label.setText(f"⚠️ Đang kết nối {mode}..."); self.info_label.setStyleSheet("color: #f39c12;")
            QApplication.processEvents()
            key = self.api_keys['test_key'] if is_testnet else self.api_keys['live_key']
            secret = self.api_keys['test_secret'] if is_testnet else self.api_keys['live_secret']
            if not key or not secret:
                self.info_label.setText(f"❌ Lỗi: Không tìm thấy API cho {mode} trong .env"); self.info_label.setStyleSheet("color: #e74c3c;")
                return
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                exchange = ccxt.binance({'apiKey': key, 'secret': secret, 'options': {'defaultType': 'future'}})
                if is_testnet: exchange.set_sandbox_mode(True)
                balance = exchange.fetch_balance()
                usdt_balance = balance['USDT']['total']
                warning = "\nCẢNH BÁO: GIAO DỊCH BẰNG TIỀN THẬT!" if not is_testnet else ""
                self.info_label.setText(f"✅ Kết nối OK. Số dư Futures: {usdt_balance:.2f} USDT.{warning}"); self.info_label.setStyleSheet("color: #2ecc71;")
            except Exception as e:
                self.info_label.setText(f"❌ Lỗi kết nối: {e}"); self.info_label.setStyleSheet("color: #e74c3c;")
            finally: QApplication.restoreOverrideCursor()

    def get_settings(self):
        mode_map = {"Paper Trading": "PAPER", "Testnet (Binance)": "TESTNET", "Live Trading (Binance)": "LIVE"}
        
        # Lấy danh sách cặp giao dịch
        active_symbols = []
        for i in range(self.symbols_list.count()):
            active_symbols.append(self.symbols_list.item(i).text())
        
        return {
            'trading_mode': mode_map[self.mode_combo.currentText()],
            'total_capital': self.total_capital_input.value(),
            'capital': self.capital_input.value(), 
            'leverage': self.leverage_input.value(),
            'risk_reward': self.reward_input.value(), 
            'fee': self.fee_input.value() / 100,
            'signal_threshold': self.signal_threshold_input.value(),
            'risk_percent': self.risk_percent_input.value() / 100,
            'update_interval': self.update_interval_input.value(),
            'analysis_interval': self.analysis_interval_input.value(),
            'active_symbols': active_symbols
        }

if __name__ == '__main__':
    app = QApplication(sys.argv)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53)); palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25)); palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white); palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white); palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white); palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218)); palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette); app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())