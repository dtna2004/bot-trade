# Trading Bot

Ứng dụng giao dịch tự động sử dụng mô hình Transformer để dự đoán giá tiền điện tử.

## Cài đặt

1. Cài đặt các thư viện cần thiết:
   ```
   pip install -r requirements.txt
   ```

2. Cấu hình API key trong file `.env`:
   ```
   # Binance API keys (Live Trading)
   BINANCE_API_KEY=your_binance_api_key_here
   BINANCE_API_SECRET=your_binance_api_secret_here

   # Binance Testnet API keys
   BINANCE_TESTNET_API_KEY=your_testnet_api_key_here
   BINANCE_TESTNET_API_SECRET=your_testnet_api_secret_here
   ```

## Chạy ứng dụng

```
python gui.py
```

## Các chế độ giao dịch

1. **Paper Trading**: Mô phỏng giao dịch trong bộ nhớ, không cần API.
2. **Testnet**: Giao dịch trên môi trường thử nghiệm của Binance.
3. **Live Trading**: Giao dịch thật với tiền thật trên Binance.

## Lưu ý quan trọng về cài đặt vốn

Khi giao dịch trên Binance (Live hoặc Testnet), cần lưu ý:

- **Giá trị lệnh tối thiểu**: Binance yêu cầu giá trị lệnh (Notional Value = Số lượng × Giá) phải từ 20 USDT trở lên.
- **Cài đặt vốn**: Nếu sử dụng đòn bẩy 100x, vốn mỗi lệnh nên tối thiểu 0.2 USDT (0.2 × 100 = 20 USDT).
- **Tự động điều chỉnh**: Bot sẽ tự động tăng vốn nếu giá trị lệnh nhỏ hơn 20 USDT.

Ví dụ:
- Với ETH giá ~2700 USDT, vốn 1 USDT và đòn bẩy 100x sẽ mua được ~0.037 ETH (giá trị ~100 USDT)
- Với BTC giá ~60000 USDT, vốn 1 USDT và đòn bẩy 100x sẽ mua được ~0.0017 BTC (giá trị ~100 USDT)

## Khắc phục lỗi

### Lỗi phiên bản scikit-learn

Nếu gặp lỗi về phiên bản scikit-learn khi tải scaler:

```
InconsistentVersionWarning: Trying to unpickle estimator MinMaxScaler from version X when using version Y
```

Hãy chạy file `fix_scaler.py` để tạo lại scaler với phiên bản hiện tại:

```
python fix_scaler.py
```

### Lỗi API key

Nếu gặp lỗi "API key/secret không có trong .env" khi sử dụng chế độ Testnet hoặc Live:

1. Kiểm tra file `.env` đã được tạo và cấu hình đúng chưa
2. Hoặc chuyển sang chế độ Paper Trading trong cài đặt

### Lỗi giá trị lệnh quá nhỏ

Nếu gặp lỗi "Order's notional must be no smaller than 20":

1. Chạy công cụ sửa lỗi tự động:
   ```
   python fix_notional.py
   ```
   Công cụ này sẽ kiểm tra và điều chỉnh cài đặt vốn để đảm bảo giá trị lệnh đạt tối thiểu 20 USDT.

2. Hoặc điều chỉnh thủ công:
   - Tăng vốn mỗi lệnh trong cài đặt (tối thiểu 20 USDT với đòn bẩy 1x)
   - Hoặc tăng đòn bẩy và giữ nguyên vốn (đảm bảo Vốn × Đòn bẩy ≥ 20 USDT)

### Lỗi kết nối mạng

Nếu gặp lỗi kết nối mạng khi tải dữ liệu:

1. Kiểm tra kết nối internet
2. Bot sẽ tự động sử dụng dữ liệu cache nếu có
3. Đảm bảo đã chọn các cặp giao dịch hợp lệ

## Liên hệ hỗ trợ

Email: support@tradingbot.com 