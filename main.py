import ccxt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import time
import logging
import os
from getpass import getpass
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# ==============================================================================
# PHẦN 1: ĐỊNH NGHĨA LẠI MÔ HÌNH TRANSFORMER
# (PyTorch cần điều này để tải lại trọng số)
# ==============================================================================

# ==============================================================================
# PHẦN 1: ĐỊNH NGHĨA LẠI MÔ HÌNH TRANSFORMER
# (PyTorch cần điều này để tải lại trọng số)
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # SỬA LỖI TẠI ĐÂY: Quay lại định nghĩa cũ để khớp với file đã lưu
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x có shape (batch, seq_len, features)
        # self.pe có shape (seq_len, 1, features)
        # Ta cần làm cho chúng tương thích
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model) # Lớp này đã được sửa
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, num_classes)
        
    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        src = self.input_embedding(src) # -> (batch_size, seq_len, d_model)
        src = self.pos_encoder(src) # -> (batch_size, seq_len, d_model)
        output = self.transformer_encoder(src) # -> (batch_size, seq_len, d_model)
        output = self.decoder(output[:, -1, :]) # Lấy output của token cuối cùng -> (batch_size, num_classes)
        return output

# ==============================================================================
# PHẦN 2: LỚP BOT GIAO DỊCH
# ==============================================================================

class TradingBot:
    def __init__(self, model_path, symbol='BTC/USDT', timeframe='15m', leverage=100, rr_ratio=(5, 4)):
        # --- Cấu hình cơ bản ---
        self.symbol = symbol
        self.timeframe = timeframe
        self.leverage = leverage
        self.reward_ratio, self.risk_ratio = rr_ratio # Reward:Risk
        self.seq_length = 48 # Phải giống với lúc train
        self.features_to_use = ['close', 'volume', 'returns', 'rsi', 'upper_band', 'lower_band']
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Trạng thái và dữ liệu ---
        self.exchange = None
        self.model = None
        self.scaler = self.fit_scaler() # Fit scaler từ dữ liệu lịch sử
        self.api_key = None
        self.api_secret = None
        self.capital_per_trade = 0.0
        self.open_position = None # Chỉ cho phép 1 vị thế mở tại một thời điểm
        self.trade_history = []
        
        # --- Thêm biến trạng thái mới ---
        self.start_time = datetime.now()
        self.status = "Khởi động"
        self.next_update = datetime.now()
        
        self.setup_logging()

    def setup_logging(self):
        """Cấu hình hệ thống ghi log"""
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler("trading_bot.log"),
                                logging.StreamHandler()
                            ])

    def fit_scaler(self):
        """Tạo và fit scaler từ dữ liệu lịch sử để đảm bảo tính nhất quán"""
        try:
            logging.info("Đang tạo scaler từ dữ liệu lịch sử...")
            df = pd.read_csv('btcusdt_1h.csv')
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['std_20'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['sma_20'] + (df['std_20'] * 2)
            df['lower_band'] = df['sma_20'] - (df['std_20'] * 2)
            df.dropna(inplace=True)
            
            scaler = MinMaxScaler()
            scaler.fit(df[self.features_to_use])
            return scaler
        except Exception as e:
            logging.error(f"Không thể tạo scaler từ file. Lỗi: {e}")
            return None

    def load_pytorch_model(self):
        """Tải mô hình Transformer đã huấn luyện"""
        try:
            input_dim = len(self.features_to_use)
            d_model = 64
            nhead = 4
            num_layers = 3
            num_classes = 3
            dropout = 0.2
            
            self.model = TransformerModel(input_dim, d_model, nhead, num_layers, num_classes, dropout).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            logging.info("Tải mô hình Transformer thành công.")
        except Exception as e:
            logging.error(f"Lỗi khi tải mô hình: {e}")
            self.model = None

    def setup_connection(self):
        """Nhập API key và kết nối tới Binance"""
        try:
            self.api_key = os.getenv("BINANCE_API_KEY")
            self.api_secret = os.getenv("BINANCE_API_SECRET")
            self.capital_per_trade = float(os.getenv("TRADING_CAPITAL", "10.0"))

            if not self.api_key or not self.api_secret:
                logging.error("Không tìm thấy API key hoặc secret trong biến môi trường")
                return

            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'options': {'defaultType': 'future'},
            })
            # self.exchange.set_sandbox_mode(True) # Bật chế độ testnet nếu cần
            self.exchange.load_markets()
            logging.info("Kết nối tới Binance Futures thành công.")
            self.exchange.set_leverage(self.leverage, self.symbol)
            logging.info(f"Đã đặt đòn bẩy là {self.leverage}x cho {self.symbol}")
        except Exception as e:
            logging.error(f"Lỗi kết nối Binance: {e}")
            self.exchange = None
            
    def get_latest_data(self):
        """Lấy dữ liệu nến mới nhất từ Binance"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=self.seq_length + 50) # Lấy dư để tính indicator
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logging.error(f"Lỗi khi lấy dữ liệu nến: {e}")
            return None

    def preprocess_data(self, df):
        """Tiền xử lý dữ liệu mới để đưa vào model"""
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma_20'] + (df['std_20'] * 2)
        df['lower_band'] = df['sma_20'] - (df['std_20'] * 2)
        df.dropna(inplace=True)
        
        scaled_features = self.scaler.transform(df[self.features_to_use])
        
        # Lấy chuỗi cuối cùng
        last_sequence = scaled_features[-self.seq_length:]
        return torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

    def get_prediction(self, input_tensor):
        """Nhận dự đoán từ model"""
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            return predicted_class.item(), confidence.item()

    def place_trade(self, direction, entry_price):
        """Đặt lệnh giao dịch với SL/TP"""
        # Xác định mức SL/TP dựa trên R:R
        # Với đòn bẩy cao, SL phải rất chặt chẽ
        sl_percentage = (1 / self.leverage) * (self.risk_ratio / 5) * 0.9 # VD: 0.72%
        tp_percentage = sl_percentage * (self.reward_ratio / self.risk_ratio)

        if direction == 'BUY':
            side = 'buy'
            sl_price = entry_price * (1 - sl_percentage)
            tp_price = entry_price * (1 + tp_percentage)
        else: # SELL
            side = 'sell'
            sl_price = entry_price * (1 + sl_percentage)
            tp_price = entry_price * (1 - tp_percentage)

        # Tính toán số lượng
        amount = (self.capital_per_trade * self.leverage) / entry_price

        try:
            logging.info(f"Đang đặt lệnh {side.upper()} {amount:.4f} {self.symbol} tại giá {entry_price}")
            # 1. Đặt lệnh Market để vào vị thế
            market_order = self.exchange.create_order(self.symbol, 'market', side, amount)
            
            # 2. Đặt lệnh SL và TP
            sl_order = self.exchange.create_order(self.symbol, 'stop_market', 'sell' if side == 'buy' else 'buy', amount, params={'stopPrice': round(sl_price, 2)})
            tp_order = self.exchange.create_order(self.symbol, 'take_profit_market', 'sell' if side == 'buy' else 'buy', amount, params={'stopPrice': round(tp_price, 2)})
            
            self.open_position = {
                "symbol": self.symbol,
                "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "direction": direction,
                "entry_price": entry_price,
                "amount": amount,
                "sl": round(sl_price, 2),
                "tp": round(tp_price, 2),
                "market_order_id": market_order['id'],
                "sl_order_id": sl_order['id'],
                "tp_order_id": tp_order['id'],
            }
            logging.info("Đặt lệnh thành công!")

        except Exception as e:
            logging.error(f"Lỗi khi đặt lệnh: {e}")

    def check_position_status(self):
        """Kiểm tra xem vị thế mở đã bị đóng hay chưa"""
        if not self.open_position:
            return

        try:
            # Kiểm tra xem còn vị thế nào trên sàn không
            positions = self.exchange.fetch_positions([self.symbol])
            current_position = next((p for p in positions if p['info']['symbol'] == self.symbol.replace('/', '') and float(p['info']['positionAmt']) != 0), None)

            if not current_position:
                # Vị thế đã bị đóng, tìm lịch sử giao dịch để xác định PNL
                logging.info("Vị thế đã bị đóng. Đang cập nhật lịch sử.")
                
                # Hủy các lệnh SL/TP còn lại (nếu có)
                try:
                    self.exchange.cancel_order(self.open_position['sl_order_id'], self.symbol)
                    self.exchange.cancel_order(self.open_position['tp_order_id'], self.symbol)
                except Exception as e:
                    logging.warning(f"Không thể hủy lệnh SL/TP (có thể đã được thực thi): {e}")

                # Lấy lịch sử giao dịch gần nhất
                trades = self.exchange.fetch_my_trades(self.symbol, limit=5)
                # Tìm trade đóng vị thế
                exit_trade = sorted([t for t in trades if t['order'] != self.open_position['market_order_id']], key=lambda x: x['timestamp'], reverse=True)[0]
                
                exit_price = exit_trade['price']
                pnl = (exit_price - self.open_position['entry_price']) * self.open_position['amount']
                if self.open_position['direction'] == 'SELL':
                    pnl = -pnl
                
                # Cập nhật trade history
                closed_trade = self.open_position.copy()
                closed_trade['exit_time'] = datetime.fromtimestamp(exit_trade['timestamp']/1000).strftime("%Y-%m-%d %H:%M:%S")
                closed_trade['exit_price'] = exit_price
                closed_trade['pnl'] = pnl
                self.trade_history.append(closed_trade)
                
                self.open_position = None # Reset vị thế

        except Exception as e:
            logging.error(f"Lỗi khi kiểm tra trạng thái vị thế: {e}")

    def display_dashboard(self):
        """Hiển thị giao diện dòng lệnh đã được nâng cấp"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # --- Phần 0: Header và Thời gian ---
        current_time = datetime.now()
        runtime = current_time - self.start_time
        hours, remainder = divmod(runtime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Tính thời gian đến lần cập nhật tiếp theo
        time_until_next = max(0, (self.next_update - current_time).seconds)
        next_minutes, next_seconds = divmod(time_until_next, 60)
        
        print("=========================== DASHBOARD BOT GIAO DỊCH ===========================")
        print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Symbol: {self.symbol} | Timeframe: {self.timeframe} | Leverage: {self.leverage}x")
        print(f"Thời gian chạy: {hours:02d}:{minutes:02d}:{seconds:02d} | Cập nhật tiếp theo trong: {next_minutes:02d}:{next_seconds:02d}")
        print(f"Trạng thái: {self.status}")
        print("-" * 75)

        # --- Phần 1: Thống kê tài khoản Futures ---
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']
            
            total_balance = float(usdt_balance.get('total', 0))
            free_balance = float(usdt_balance.get('free', 0))
            used_balance = float(usdt_balance.get('used', 0))
            
            # Lấy PNL chưa thực hiện từ fetch_positions
            positions = self.exchange.fetch_positions([self.symbol])
            unrealized_pnl = sum(float(p['unrealizedPnl']) for p in positions if 'unrealizedPnl' in p)

            account_stats = [
                ["Tổng số dư (Total Balance)", f"{total_balance:.2f} USDT"],
                ["Số dư khả dụng (Available)", f"{free_balance:.2f} USDT"],
                ["Đã sử dụng (Margin)", f"{used_balance:.2f} USDT"],
                ["PNL chưa thực hiện (Unrealized PNL)", f"{unrealized_pnl:+.2f} USDT"] # Dấu + để hiển thị cả số dương
            ]
            print("--- THÔNG TIN TÀI KHOẢN FUTURES ---")
            print(tabulate(account_stats, headers=["Chỉ số", "Giá trị"], tablefmt="fancy_grid"))
            
        except Exception as e:
            logging.warning(f"Không thể lấy thông tin tài khoản: {e}")
            print("--- THÔNG TIN TÀI KHOẢN FUTURES ---\nKhông thể lấy dữ liệu.")

        # --- Phần 2: Thống kê giao dịch của Bot ---
        total_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        bot_stats = [
            ["Vốn mỗi lệnh", f"{self.capital_per_trade:.2f} USDT"],
            ["Tổng PNL đã thực hiện", f"{total_pnl:+.4f} USDT"],
            ["Tổng số lệnh đã đóng", f"{total_trades}"],
            ["Tỷ lệ thắng (Win Rate)", f"{win_rate:.2f}%"]
        ]
        print("\n--- THỐNG KÊ GIAO DỊCH CỦA BOT ---")
        print(tabulate(bot_stats, headers=["Chỉ số", "Giá trị"], tablefmt="fancy_grid"))
        
        # --- Phần 3: Vị thế đang mở ---
        print("\n--- VỊ THẾ ĐANG MỞ ---")
        if self.open_position:
            try:
                current_price = self.exchange.fetch_ticker(self.symbol)['last']
                pnl = (current_price - self.open_position['entry_price']) * self.open_position['amount']
                if self.open_position['direction'] == 'SELL': pnl = -pnl
                
                display_data = self.open_position.copy()
                # Xóa các ID để giao diện gọn gàng hơn
                for key in ['market_order_id', 'sl_order_id', 'tp_order_id']:
                    display_data.pop(key, None)
                
                display_data['pnl (tạm tính)'] = f"{pnl:+.4f}"
                print(tabulate([display_data], headers="keys", tablefmt="fancy_grid"))
            except Exception as e:
                logging.warning(f"Không thể lấy PNL tạm tính: {e}")
                print("Không thể lấy dữ liệu vị thế.")
        else:
            print("Không có vị thế nào đang mở.")

        # --- Phần 4: Lịch sử giao dịch ---
        print("\n--- LỊCH SỬ GIAO DỊCH (5 lệnh gần nhất) ---")
        if self.trade_history:
            recent_history = self.trade_history[-5:][::-1] # Đảo ngược để lệnh mới nhất lên đầu
            display_history = []
            for t in recent_history:
                clean_t = t.copy()
                # Làm tròn PNL và xóa các ID
                clean_t['pnl'] = f"{clean_t.get('pnl', 0):+.4f}"
                for key in ['market_order_id', 'sl_order_id', 'tp_order_id', 'amount']:
                    clean_t.pop(key, None)
                display_history.append(clean_t)
                
            print(tabulate(display_history, headers="keys", tablefmt="fancy_grid"))
        else:
            print("Chưa có lịch sử giao dịch.")
            
        print("\n" + "="*75)

    def run(self):
        """Vòng lặp chính của bot"""
        self.load_pytorch_model()
        if not self.model or not self.scaler:
            logging.error("Không thể khởi động bot do thiếu model hoặc scaler.")
            return

        self.setup_connection()
        if not self.exchange:
            return

        while True:
            try:
                # 1. Kiểm tra vị thế cũ
                self.status = "Đang kiểm tra vị thế..."
                self.check_position_status()

                # 2. Nếu không có vị thế mở, tìm tín hiệu mới
                if not self.open_position:
                    self.status = "Đang tìm tín hiệu giao dịch mới..."
                    logging.info("Đang tìm tín hiệu giao dịch mới...")
                    
                    # Lấy và xử lý dữ liệu
                    self.status = "Đang lấy dữ liệu mới..."
                    df = self.get_latest_data()
                    if df is not None and not df.empty:
                        self.status = "Đang xử lý dữ liệu..."
                        input_tensor = self.preprocess_data(df)
                        
                        # Nhận dự đoán
                        self.status = "Đang phân tích thị trường..."
                        pred_class, confidence = self.get_prediction(input_tensor)
                        logging.info(f"Model dự đoán: Lớp={pred_class}, Độ tin cậy={confidence:.2f}")

                        # Ra quyết định
                        if confidence > 0.6: # Chỉ vào lệnh nếu độ tin cậy > 60%
                            current_price = df['close'].iloc[-1]
                            if pred_class == 2: # Lớp 2: Tăng
                                self.status = "Đang đặt lệnh MUA..."
                                logging.info("Tín hiệu MUA mạnh. Chuẩn bị vào lệnh.")
                                self.place_trade('BUY', current_price)
                            elif pred_class == 0: # Lớp 0: Giảm
                                self.status = "Đang đặt lệnh BÁN..."
                                logging.info("Tín hiệu BÁN mạnh. Chuẩn bị vào lệnh.")
                                self.place_trade('SELL', current_price)
                
                # 3. Chờ đến cây nến tiếp theo với cập nhật real-time
                wait_time = int(self.timeframe.replace('m', '')) * 60
                self.next_update = datetime.now() + timedelta(seconds=wait_time)
                self.status = "Đang chờ cập nhật tiếp theo..."
                
                # Vòng lặp cập nhật real-time
                update_interval = 1  # Cập nhật mỗi giây
                elapsed_time = 0
                while elapsed_time < wait_time:
                    try:
                        # Cập nhật thông tin vị thế và giá nếu có
                        if self.open_position:
                            self.check_position_status()
                        # Hiển thị dashboard
                        self.display_dashboard()
                        time.sleep(update_interval)
                        elapsed_time += update_interval
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except Exception as e:
                        logging.error(f"Lỗi khi cập nhật real-time: {e}")
                        time.sleep(update_interval)
                        elapsed_time += update_interval

            except KeyboardInterrupt:
                self.status = "Đã dừng bởi người dùng"
                logging.info("Bot đã dừng bởi người dùng.")
                break
            except Exception as e:
                self.status = f"Lỗi: {str(e)}"
                logging.error(f"Lỗi trong vòng lặp chính: {e}")
                time.sleep(60) # Chờ 1 phút trước khi thử lại

# ==============================================================================
# PHẦN 3: KHỞI CHẠY BOT
# ==============================================================================

if __name__ == "__main__":
    bot = TradingBot(
        model_path="transformer_btc_trader_statedict.pth",
        symbol='BTC/USDT',
        timeframe='15m',
        leverage=100,
        rr_ratio=(5, 4) # Tỷ lệ Reward:Risk là 5:4 (TP = 1.25 * SL)
    )
    bot.run()