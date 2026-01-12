import pandas as pd
import os

# --- 1. CẤU HÌNH ĐƯỜNG DẪN  ---
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
input_dir = os.path.join(project_root, 'data', 'raw')
output_dir = os.path.join(project_root, 'data', 'processed')

os.makedirs(output_dir, exist_ok=True)

print(f"Đang đọc dữ liệu từ: {input_dir}")


def process_data():
    try:
        # --- BƯỚC 1: ĐỌC DỮ LIỆU ---

        # 1. Đọc 30 mã cổ phiếu
        stocks_path = os.path.join(input_dir, 'vn30_stocks_prices.csv')
        # index_col=0: Cột đầu tiên là ngày
        # parse_dates=True: Tự hiểu định dạng ngày tháng
        df_stocks = pd.read_csv(stocks_path, index_col=0, parse_dates=True)
        print(f"Đã đọc VN30 Stocks: {df_stocks.shape}")

        # 2. Đọc VNINDEX
        index_path = os.path.join(input_dir, 'vnindex_manual.csv')
        df_index = pd.read_csv(index_path, index_col=0, parse_dates=True)
        print(f"Đã đọc VNINDEX: {df_index.shape}")

        # --- BƯỚC 2: GỘP DỮ LIỆU ---

        # Gộp 2 bảng lại theo ngày (chỉ lấy ngày nào cả 2 đều có dữ liệu)
        df_daily = df_stocks.join(df_index, how='inner')

        # Làm sạch: Fill giá cũ nếu thiếu, bỏ dòng rỗng
        df_daily = df_daily.ffill()
        df_daily.dropna(inplace=True)

        print(f"   -> Kích thước sau khi gộp: {df_daily.shape}")

        # --- BƯỚC 3: LƯU CÁC FILE CẦN THIẾT ---

        # FILE A: Giá hàng ngày (Dùng cho Dashboard vẽ biểu đồ)
        path_daily = os.path.join(output_dir, 'vn30_cleaned.csv')
        df_daily.to_csv(path_daily)
        print(f"Đã lưu file GIÁ HÀNG NGÀY: {path_daily}")

        # FILE B: Lợi suất tháng (Dùng cho CAPM, Portfolio Analysis)
        # Tính giá cuối tháng
        df_monthly_prices = df_daily.resample('ME').last()
        # Tính % thay đổi
        df_monthly_returns = df_monthly_prices.pct_change().dropna()

        path_monthly = os.path.join(output_dir, 'vn30_monthly_returns.csv')
        df_monthly_returns.to_csv(path_monthly)
        print(f"Đã lưu file LỢI SUẤT THÁNG: {path_monthly}")

        print("XỬ LÝ XONG TOÀN BỘ!")
        print(df_daily.tail())

    except Exception as e:
        print(f"LỖI: {e}")
        print("💡 Gợi ý: Kiểm tra xem File 1 đã chạy thành công chưa?")


if __name__ == "__main__":
    process_data()