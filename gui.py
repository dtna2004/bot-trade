import sys
import os
import csv
import json
import time
import math
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
                               QDialogButtonBox, QFormLayout, QDialog, QMenuBar, QCheckBox,
                               QMessageBox, QHeaderView, QSpinBox, QDoubleSpinBox)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QPalette, QColor

# ==============================================================================
# PHẦN 1: CÁC LỚP ĐỊNH NGHĨA MODEL
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
# PHẦN 2: LỚP LOGIC GIAO DỊCH (PAPER TRADING)
# ==============================================================================

class PaperTradingAccount:
    def __init__(self, initial_balance=1000, leverage=100, fee=0.001, risk_reward=1.5):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        
        self.leverage = leverage
        self.fee = fee
        self.risk_reward = risk_reward

    def place_order(self, symbol, side, amount, price, stop_loss, take_profit):
        if symbol in self.positions:
            raise ValueError(f"Đã có vị thế cho {symbol}.")

        order_value = price * amount
        margin_required = order_value / self.leverage
        entry_fee = order_value * self.fee

        if margin_required + entry_fee > self.balance:
            raise ValueError(f"Không đủ số dư. Cần {margin_required + entry_fee:.2f}, có {self.balance:.2f} USDT")

        self.balance -= (margin_required + entry_fee)

        position = {
            'symbol': symbol, 'side': side, 'amount': amount,
            'entry_price': price, 'margin': margin_required,
            'stop_loss': stop_loss, 'take_profit': take_profit,
            'entry_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'direction': 'LONG' if side == 'BUY' else 'SHORT',
            'fee': entry_fee, 'current_price': price,
            'unrealized_pnl': -entry_fee,
            'roe': -(entry_fee / margin_required) * 100 if margin_required > 0 else 0
        }
        self.positions[symbol] = position
        return position

    def update_position(self, symbol, current_price):
        if symbol not in self.positions: return None
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
        if symbol not in self.positions: return None
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
            'symbol': position['symbol'], 'direction': position['direction'],
            'entry_price': position['entry_price'], 'exit_price': current_price,
            'amount': position['amount'], 'leverage': self.leverage, 'sl': position['stop_loss'],
            'tp': position['take_profit'], 'pnl': realized_pnl, 'roe': final_roe,
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

    def _save_trade_to_csv(self, trade):
        file_path, file_exists = 'trade_history.csv', os.path.exists('trade_history.csv')
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Entry Time', 'Exit Time', 'Symbol', 'Direction', 'Reason',
                               'Entry Price', 'Exit Price', 'Amount', 'Leverage',
                               'Stop Loss', 'Take Profit', 'PnL (USDT)', 'ROE (%)', 'Fee (USDT)'])
            writer.writerow([trade['entry_time'], trade['exit_time'], trade['symbol'],
                             trade['direction'], trade.get('reason', 'N/A'),
                             f"{trade['entry_price']:.4f}", f"{trade['exit_price']:.4f}",
                             f"{trade['amount']:.6f}", f"{trade['leverage']}x",
                             f"{trade['sl']:.4f}", f"{trade['tp']:.4f}",
                             f"{trade['pnl']:.4f}", f"{trade['roe']:.2f}", f"{trade['fee']:.4f}"])

# ==============================================================================
# PHẦN 3: LỚP WORKER (LUỒNG PHỤ CỦA BOT)
# ==============================================================================

class BotWorker(QThread):
    update_signal = Signal(dict); error_signal = Signal(str)
    status_signal = Signal(dict); trade_closed_signal = Signal(dict)
    
    def __init__(self, api_key, api_secret, settings):
        super().__init__()
        self.api_key, self.api_secret, self.settings = api_key, api_secret, settings
        self.is_running = True
        self.update_interval, self.analysis_interval = 60, 5
        self.candle_limit = 50; self.last_data = {}; self.close_requests = []
        
        if settings['test_mode']:
            self.trading_account = PaperTradingAccount(
                initial_balance=settings['total_capital'], leverage=settings['leverage'],
                fee=settings['fee'], risk_reward=settings['risk_reward'])
        else: self.trading_account = None

        self.model = None; self.scaler = None
        
    def load_model_and_scaler(self):
        try:
            config_path = os.path.join("trained_model", "model_config.json")
            if not os.path.exists(config_path): raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")
            with open(config_path, 'r') as f: config = json.load(f)
            
            if config.get('num_classes') != 2:
                self.error_signal.emit(f"Lỗi: Model được cấu hình cho {config.get('num_classes')} lớp. Vui lòng huấn luyện lại model với 2 lớp (Tăng/Giảm).")
                return False

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
            self.status_signal.emit({'text': "✅ Model 2 lớp và Scaler đã được tải thành công."})
            return True
        except Exception as e:
            self.error_signal.emit(f"Lỗi khi tải model: {str(e)}")
            return False

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
        if input_tensor is None: 
            return {'long': 0, 'short': 0, 'diff': 0, 'decision': 'NEUTRAL'}
            
        with torch.no_grad():
            probabilities = torch.softmax(self.model(input_tensor), dim=1)[0]
        
        prob_short = probabilities[0].item() * 100
        prob_long = probabilities[1].item() * 100
        prob_diff = prob_long - prob_short
        
        decision = 'NEUTRAL'
        signal_threshold = self.settings.get('signal_threshold', 10.0)

        if prob_diff > signal_threshold:
            decision = 'LONG'
        elif prob_diff < -signal_threshold:
            decision = 'SHORT'
            
        return {'long': prob_long, 'short': prob_short, 'diff': prob_diff, 'decision': decision}

    def process_close_requests(self):
        while self.close_requests:
            symbol_to_close = self.close_requests.pop(0)
            if symbol_to_close == "__ALL__":
                for sym in list(self.trading_account.positions.keys()):
                    if sym in self.last_data:
                        current_price = self.last_data[sym]['close'].iloc[-1]
                        trade_record = self.trading_account.close_position(sym, current_price, "Manual Close All")
                        if trade_record: self.trade_closed_signal.emit(trade_record)
            elif self.trading_account.has_position(symbol_to_close) and symbol_to_close in self.last_data:
                current_price = self.last_data[symbol_to_close]['close'].iloc[-1]
                trade_record = self.trading_account.close_position(symbol_to_close, current_price, "Manual Close")
                if trade_record: self.trade_closed_signal.emit(trade_record)
                
    def fetch_ohlcv_data(self, exchange, symbol):
        try:
            self.status_signal.emit({'text': f"⏳ Đang tải dữ liệu cho {symbol}..."})
            ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=self.candle_limit)
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                self.last_data[symbol] = df
                self.status_signal.emit({'text': f"✅ Đã tải xong dữ liệu {symbol}."})
                return True
        except Exception as e:
            self.error_signal.emit(f"Lỗi tải dữ liệu {symbol}: {str(e)}")
        return False

    def run(self):
        self.status_signal.emit({'text': "🚀 Bot đang khởi động..."})
        if not self.load_model_and_scaler():
            self.error_signal.emit("Không thể tải model. Bot sẽ dừng lại.")
            return

        try:
            exchange = ccxt.binance({'options': {'defaultType': 'future'}})
            
            self.status_signal.emit({'text': "--- Tải dữ liệu ban đầu ---"})
            for symbol in self.settings.get('symbols', []):
                if not self.is_running: return
                self.fetch_ohlcv_data(exchange, symbol)
            self.status_signal.emit({'text': "--- Dữ liệu ban đầu đã sẵn sàng. Bắt đầu phân tích. ---"})

            last_update_time = time.time()
            
            while self.is_running:
                current_time = time.time()
                
                if current_time - last_update_time >= self.update_interval:
                    for symbol in self.settings.get('symbols', []):
                        if not self.is_running: return
                        self.fetch_ohlcv_data(exchange, symbol)
                    last_update_time = current_time

                self.process_close_requests()

                if self.trading_account:
                    for symbol in self.settings.get('symbols', []):
                        if symbol not in self.last_data: continue
                        
                        df = self.last_data[symbol]
                        current_price = df['close'].iloc[-1]

                        if self.trading_account.has_position(symbol):
                            closed_trade = self.trading_account.update_position(symbol, current_price)
                            if closed_trade: self.trade_closed_signal.emit(closed_trade)
                        else:
                            analysis = self.analyze_market(df)
                            self.status_signal.emit({'symbol': symbol, 'analysis': analysis, 'price': current_price})

                            decision = analysis['decision']
                            if decision != 'NEUTRAL':
                                capital_per_trade = self.settings['capital']
                                leverage = self.settings['leverage']
                                amount = (capital_per_trade * leverage) / current_price
                                risk_percent, rr_ratio = 0.005, self.settings['risk_reward']

                                try:
                                    if decision == 'LONG':
                                        sl, tp = current_price * (1 - risk_percent), current_price * (1 + risk_percent * rr_ratio)
                                        self.trading_account.place_order(symbol, 'BUY', amount, current_price, sl, tp)
                                        self.status_signal.emit({'text': f"🚀 Mở lệnh LONG {symbol} @ {current_price:.2f}"})
                                    elif decision == 'SHORT':
                                        sl, tp = current_price * (1 + risk_percent), current_price * (1 - risk_percent * rr_ratio)
                                        self.trading_account.place_order(symbol, 'SELL', amount, current_price, sl, tp)
                                        self.status_signal.emit({'text': f"🔻 Mở lệnh SHORT {symbol} @ {current_price:.2f}"})
                                except Exception as e:
                                    self.error_signal.emit(f"Lỗi vào lệnh {symbol}: {e}")
                
                    self.update_signal.emit({
                        'positions': self.trading_account.get_all_positions(),
                        'balance': self.trading_account.get_balance_info()})
                
                self.msleep(int(self.analysis_interval * 1000))
        except Exception as e:
            self.error_signal.emit(f"Lỗi nghiêm trọng trong bot worker: {e}")

    def stop(self): self.is_running = False
    def handle_close_request(self, symbol: str): self.close_requests.append(symbol)
    def handle_close_all_request(self): self.close_requests.append("__ALL__")

# ==============================================================================
# PHẦN 4: GIAO DIỆN (MAIN WINDOW & DIALOG)
# ==============================================================================

class MainWindow(QMainWindow):
    request_close_position = Signal(str); request_close_all_positions = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Bot"); self.setMinimumSize(1200, 800)
        
        load_dotenv()
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        if not (self.api_key and self.api_secret):
            QMessageBox.critical(self, "Lỗi API", "Không tìm thấy API key và secret trong file .env")
            sys.exit()

        self.available_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
        self.settings_file = "bot_settings.json"; self.settings = {}
        self.load_settings()
        self.bot_worker = None
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        menubar = self.menuBar()
        settings_menu = menubar.addMenu('Cài đặt')
        settings_menu.addAction('Cài đặt Bot & Vốn', self.show_settings_dialog)
        
        symbol_group = QGroupBox("Chọn cặp giao dịch")
        symbol_layout = QHBoxLayout(); symbol_group.setLayout(symbol_layout)
        self.symbol_checkboxes = {}
        for symbol in self.available_symbols:
            cb = QCheckBox(symbol)
            if symbol in self.settings.get('active_symbols', []): cb.setChecked(True)
            cb.stateChanged.connect(self.on_symbol_changed)
            symbol_layout.addWidget(cb); self.symbol_checkboxes[symbol] = cb
        main_layout.addWidget(symbol_group)
        
        content_layout = QHBoxLayout(); main_layout.addLayout(content_layout)
        left_panel = QWidget(); left_layout = QVBoxLayout(left_panel)
        content_layout.addWidget(left_panel, 2)

        status_group = QGroupBox("Trạng thái & Phân tích")
        self.status_text = QTextEdit(); self.status_text.setReadOnly(True)
        status_group.setLayout(QVBoxLayout()); status_group.layout().addWidget(self.status_text)
        left_layout.addWidget(status_group)

        position_group = QGroupBox("Vị thế hiện tại")
        position_layout = QVBoxLayout(); position_group.setLayout(position_layout)
        self.position_table = QTableWidget()
        self.position_table.setColumnCount(10)
        self.position_table.setHorizontalHeaderLabels(["Symbol", "Hướng", "Ký quỹ", "Giá vào", "Giá hiện tại", 
                                                      "SL", "TP", "PnL (USDT)", "ROE (%)", "Đóng"])
        self.position_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        position_layout.addWidget(self.position_table)
        pnl_layout = QHBoxLayout(); self.total_pnl_label = QLabel("Tổng PnL: 0.00 USDT")
        pnl_layout.addStretch(); pnl_layout.addWidget(self.total_pnl_label)
        position_layout.addLayout(pnl_layout)
        left_layout.addWidget(position_group)
        
        right_panel = QWidget(); right_layout = QVBoxLayout(right_panel)
        content_layout.addWidget(right_panel, 1)

        control_group = QGroupBox("Điều khiển")
        control_layout = QVBoxLayout(); control_group.setLayout(control_layout)
        self.start_button = QPushButton("Bắt đầu"); self.start_button.clicked.connect(self.start_bot)
        self.stop_button = QPushButton("Dừng"); self.stop_button.clicked.connect(self.stop_bot)
        self.stop_button.setEnabled(False)
        close_all_button = QPushButton("Đóng tất cả lệnh"); close_all_button.clicked.connect(self.confirm_close_all_positions)
        control_layout.addWidget(self.start_button); control_layout.addWidget(self.stop_button); control_layout.addWidget(close_all_button)
        right_layout.addWidget(control_group)

        balance_group = QGroupBox("Tài khoản")
        self.balance_text = QTextEdit(); self.balance_text.setReadOnly(True)
        self.balance_text.setMaximumHeight(120)
        balance_group.setLayout(QVBoxLayout()); balance_group.layout().addWidget(self.balance_text)
        right_layout.addWidget(balance_group); right_layout.addStretch()

    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f: self.settings = json.load(f)
            else:
                self.settings = {
                    'test_mode': True, 'total_capital': 1000, 'capital': 10, 
                    'leverage': 50, 'risk_reward': 1.5, 'fee': 0.0005, 
                    'active_symbols': ['BTC/USDT'],
                    'signal_threshold': 10.0
                }
                self.save_settings()
            
            # Đảm bảo 'signal_threshold' luôn tồn tại trong cài đặt
            if 'signal_threshold' not in self.settings:
                self.settings['signal_threshold'] = 10.0
                self.save_settings()

            self.setWindowTitle(f"Trading Bot - {'PAPER MODE' if self.settings.get('test_mode', True) else 'LIVE MODE'}")
        except Exception as e: self.update_status({'text': f"Lỗi load cài đặt: {e}"})

    def save_settings(self):
        try:
            with open(self.settings_file, 'w') as f: json.dump(self.settings, f, indent=4)
        except Exception as e: self.update_status({'text': f"Lỗi lưu cài đặt: {e}"})

    def on_symbol_changed(self):
        self.settings['active_symbols'] = [s for s, cb in self.symbol_checkboxes.items() if cb.isChecked()]
        self.save_settings()

    def show_settings_dialog(self):
        dialog = CapitalDialog(self, self.settings)
        if dialog.exec():
            self.settings.update(dialog.get_settings())
            self.save_settings(); self.load_settings()
            self.update_status({'text': "Cài đặt đã được cập nhật."})

    def start_bot(self):
        if not self.settings.get('active_symbols'):
            return QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng chọn ít nhất một cặp giao dịch.")
        worker_settings = self.settings.copy()
        worker_settings['symbols'] = self.settings.get('active_symbols', [])
        
        self.bot_worker = BotWorker(self.api_key, self.api_secret, worker_settings)
        self.bot_worker.update_signal.connect(self.update_ui_data)
        self.bot_worker.error_signal.connect(self.show_error)
        self.bot_worker.status_signal.connect(self.update_status)
        self.bot_worker.trade_closed_signal.connect(self.handle_trade_closed)
        
        self.request_close_position.connect(self.bot_worker.handle_close_request)
        self.request_close_all_positions.connect(self.bot_worker.handle_close_all_request)
        
        self.bot_worker.start()
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)

    def stop_bot(self):
        if self.bot_worker:
            self.bot_worker.stop(); self.bot_worker.wait(5000); self.bot_worker = None
        self.start_button.setEnabled(True); self.stop_button.setEnabled(False)
        self.update_status({'text': "Bot đã dừng."}); self.position_table.setRowCount(0)
        self.total_pnl_label.setText("Tổng PnL: 0.00 USDT")
    
    def confirm_close_all_positions(self):
        if QMessageBox.question(self, 'Xác nhận', 'Bạn có chắc muốn đóng tất cả các vị thế hiện tại?',
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes and self.bot_worker:
            self.request_close_all_positions.emit()

    def confirm_close_position(self, row):
        symbol = self.position_table.item(row, 0).text()
        if QMessageBox.question(self, 'Xác nhận', f'Bạn có chắc muốn đóng vị thế {symbol}?',
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes and self.bot_worker:
            self.request_close_position.emit(symbol)

    def update_ui_data(self, data):
        positions = data.get('positions', [])
        self.position_table.setRowCount(len(positions)); total_pnl = 0
        for i, pos in enumerate(positions):
            pnl, roe = pos.get('unrealized_pnl', 0), pos.get('roe', 0)
            total_pnl += pnl
            
            self.position_table.setItem(i, 0, QTableWidgetItem(pos.get('symbol', '')))
            self.position_table.setItem(i, 1, QTableWidgetItem(pos.get('direction', '')))
            self.position_table.setItem(i, 2, QTableWidgetItem(f"{pos.get('margin', 0):.2f}"))
            self.position_table.setItem(i, 3, QTableWidgetItem(f"{pos.get('entry_price', 0):.4f}"))
            self.position_table.setItem(i, 4, QTableWidgetItem(f"{pos.get('current_price', 0):.4f}"))
            self.position_table.setItem(i, 5, QTableWidgetItem(f"{pos.get('stop_loss', 0):.4f}"))
            self.position_table.setItem(i, 6, QTableWidgetItem(f"{pos.get('take_profit', 0):.4f}"))
            
            pnl_item = QTableWidgetItem(f"{pnl:+.2f}"); pnl_item.setForeground(QColor('#2ecc71' if pnl >= 0 else '#e74c3c'))
            self.position_table.setItem(i, 7, pnl_item)
            roe_item = QTableWidgetItem(f"{roe:+.2f}%"); roe_item.setForeground(QColor('#2ecc71' if roe >= 0 else '#e74c3c'))
            self.position_table.setItem(i, 8, roe_item)

            close_button = QPushButton("Đóng"); close_button.clicked.connect(lambda ch, r=i: self.confirm_close_position(r))
            self.position_table.setCellWidget(i, 9, close_button)
        
        self.total_pnl_label.setText(f"Tổng PnL: {total_pnl:+.2f} USDT")
        self.total_pnl_label.setStyleSheet(f"font-weight: bold; color: {'#2ecc71' if total_pnl >= 0 else '#e74c3c'};")

        balance_info = data.get('balance', {})
        self.balance_text.setText(f"Tổng Vốn (Equity): {balance_info.get('equity', 0):.2f} USDT\n"
                                  f"Số dư khả dụng: {balance_info.get('available_balance', 0):.2f} USDT\n"
                                  f"Ký quỹ đã dùng: {balance_info.get('used_margin', 0):.2f} USDT\n"
                                  f"Lãi/Lỗ chưa thực hiện: {balance_info.get('unrealized_pnl', 0):.2f} USDT")
    
    def show_error(self, error_msg): self.update_status({'text': f"❌ LỖI: {error_msg}"})
    
    def update_status(self, data):
        timestamp = f"[{datetime.now().strftime('%H:%M:%S')}]"
        
        if 'analysis' in data:
            symbol, price, analysis = data['symbol'], data['price'], data['analysis']
            prob_diff, decision = analysis['diff'], analysis['decision']
            
            # Xác định màu sắc và nội dung quyết định
            if decision == 'LONG':
                decision_text = f"<font color='#2ecc71'><b>QUYẾT ĐỊNH: LONG</b></font>"
            elif decision == 'SHORT':
                decision_text = f"<font color='#e74c3c'><b>QUYẾT ĐỊNH: SHORT</b></font>"
            else:
                decision_text = f"<font color='#95a5a6'>QUYẾT ĐỊNH: ĐỨNG NGOÀI</font>"
            
            # Định dạng HTML để hiển thị đẹp hơn
            status_html = (
                f"{timestamp} <b>{symbol}</b> @ ${price:,.2f} | "
                f"Chênh lệch (L-S): {prob_diff:+.2f}% "
                f"(Yêu cầu: +/-{self.settings.get('signal_threshold', 10.0):.2f}%) "
                f"-> {decision_text}"
            )
            self.status_text.append(status_html)
        
        elif 'text' in data:
            self.status_text.append(f"{timestamp} {data['text']}")
            
        self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())

    def handle_trade_closed(self, trade):
        pnl, color = trade.get('pnl', 0), "lime" if trade.get('pnl', 0) >= 0 else "red"
        self.status_text.append(f"<br>--- GIAO DỊCH ĐÓNG ---<br><b>Symbol:</b> {trade['symbol']} ({trade['direction']})<br>"
                                f"<b>Lý do:</b> {trade.get('reason', 'N/A')}<br>"
                                f"<b>Lãi/Lỗ:</b> <font color='{color}'>{pnl:+.4f} USDT</font><br>"
                                f"<b>ROE:</b> <font color='{color}'>{trade.get('roe', 0):+.2f}%</font><br>"
                                f"-----------------------")
        if pnl > 0: QApplication.beep()

    def closeEvent(self, event): self.stop_bot(); event.accept()

class CapitalDialog(QDialog):
    def __init__(self, parent, current_settings):
        super().__init__(parent); self.setWindowTitle("Cài đặt Bot"); self.setModal(True)
        self.settings = current_settings; self.exchange = None

        layout, form_layout = QVBoxLayout(self), QFormLayout()
        self.test_mode_cb = QCheckBox("Chế độ Paper Trading")
        self.total_capital_input = QDoubleSpinBox()
        self.capital_input = QDoubleSpinBox()
        self.leverage_input = QSpinBox()
        self.reward_input = QDoubleSpinBox()
        self.fee_input = QDoubleSpinBox()
        self.signal_threshold_input = QDoubleSpinBox()

        self.test_mode_cb.setChecked(self.settings.get('test_mode', True))
        self.total_capital_input.setRange(10, 1_000_000); self.total_capital_input.setValue(self.settings.get('total_capital', 1000)); self.total_capital_input.setSuffix(" USDT")
        self.capital_input.setRange(1, 100_000); self.capital_input.setValue(self.settings.get('capital', 10)); self.capital_input.setSuffix(" USDT")
        self.leverage_input.setRange(1, 125); self.leverage_input.setValue(self.settings.get('leverage', 50)); self.leverage_input.setSuffix("x")
        self.reward_input.setRange(0.1, 10); self.reward_input.setValue(self.settings.get('risk_reward', 1.5)); self.reward_input.setSingleStep(0.1)
        self.fee_input.setRange(0, 1); self.fee_input.setValue(self.settings.get('fee', 0.0005) * 100); self.fee_input.setSuffix("%")
        self.signal_threshold_input.setRange(1, 40); self.signal_threshold_input.setValue(self.settings.get('signal_threshold', 10.0)); self.signal_threshold_input.setSuffix(" %")

        form_layout.addRow(self.test_mode_cb)
        form_layout.addRow("Tổng vốn Paper Trading:", self.total_capital_input)
        form_layout.addRow("Ký quỹ mỗi lệnh:", self.capital_input)
        form_layout.addRow("Đòn bẩy:", self.leverage_input)
        form_layout.addRow("Tỉ lệ R:R (1:X):", self.reward_input)
        form_layout.addRow("Ngưỡng tín hiệu (% chênh lệch):", self.signal_threshold_input)
        form_layout.addRow("Phí giao dịch:", self.fee_input)
        layout.addLayout(form_layout)

        self.info_label = QLabel(); layout.addWidget(self.info_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.test_mode_cb.stateChanged.connect(self.on_mode_changed)
        self.on_mode_changed()

    def on_mode_changed(self):
        is_test_mode = self.test_mode_cb.isChecked()
        self.total_capital_input.setEnabled(is_test_mode)
        if is_test_mode:
            self.info_label.setText("Paper Trading mô phỏng giao dịch với dữ liệu thật."); self.info_label.setStyleSheet("color: #4db6ac;")
        else:
            self.info_label.setText("⚠️ Chuyển sang LIVE MODE. Đang kết nối..."); self.info_label.setStyleSheet("color: #f39c12;")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor); QApplication.processEvents()
            try:
                if self.exchange is None:
                    self.exchange = ccxt.binance({'apiKey': self.parent().api_key, 'secret': self.parent().api_secret, 'options': {'defaultType': 'future'}})
                balance = self.exchange.fetch_balance()
                usdt_balance = balance.get('USDT', {}).get('total', 0)
                self.info_label.setText(f"✅ Kết nối thành công. Số dư Futures: {usdt_balance:.2f} USDT.<br>CẢNH BÁO: Bot sẽ giao dịch bằng tiền thật!")
                self.info_label.setStyleSheet("color: #2ecc71;")
            except Exception as e:
                self.info_label.setText(f"❌ Lỗi kết nối: {e}"); self.info_label.setStyleSheet("color: #e74c3c;")
                self.test_mode_cb.setChecked(True)
            finally: QApplication.restoreOverrideCursor()

    def get_settings(self):
        return {'test_mode': self.test_mode_cb.isChecked(), 'total_capital': self.total_capital_input.value(),
                'capital': self.capital_input.value(), 'leverage': self.leverage_input.value(),
                'risk_reward': self.reward_input.value(), 'fee': self.fee_input.value() / 100,
                'signal_threshold': self.signal_threshold_input.value()}

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
