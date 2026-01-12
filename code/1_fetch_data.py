import pandas as pd
import yfinance as yf
import requests
import os
import json
from datetime import datetime

# --- 1. CẤU HÌNH ---
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
output_dir = os.path.join(project_root, 'data', 'raw')
os.makedirs(output_dir, exist_ok=True)

API_ENDPOINT = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
START_DATE_CAFEF = "01/01/2020"
END_DATE_CAFEF = datetime.now().strftime("%d/%m/%Y")


def get_vnindex_smart_detect():
    print("Đang tải VNINDEX từ CafeF (Auto-Detect)...")

    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})

    params = {
        'Symbol': 'VNINDEX',
        'StartDate': START_DATE_CAFEF,
        'EndDate': END_DATE_CAFEF,
        'PageIndex': 1,
        'PageSize': 10000
    }

    try:
        response = session.get(API_ENDPOINT, params=params, timeout=30)
        data = response.json()

        if not data or 'Data' not in data or 'Data' not in data['Data']:
            print(f"   API không trả về dữ liệu.")
            return None

        df = pd.DataFrame(data['Data']['Data'])
        if df.empty: return None

        # --- LOGIC DÒ TÌM THÔNG MINH ---
        print(f"   Các cột tìm thấy: {list(df.columns)}")

        # 1. Tìm cột NGÀY: Quét xem cột nào chứa chữ 'date', 'ngay', 'time'
        date_col = None
        # Chuyển hết tên cột về chữ thường để so sánh cho dễ
        col_map = {col.lower(): col for col in df.columns}

        # Các từ khóa nhận diện cột ngày
        date_keywords = ['datadate', 'tradingdate', 'date', 'ngay', 'time']

        for keyword in date_keywords:
            # Tìm xem có cột nào CHỨA từ khóa này không
            match = next((col for col in col_map.keys() if keyword in col), None)
            if match:
                date_col = col_map[match]  # Lấy lại tên gốc (có hoa thường)
                print(f"   -> phát hiện cột Ngày là: '{date_col}' (khớp từ khóa '{keyword}')")
                break

        if not date_col:
            print("   Không tìm thấy cột Ngày!")
            return None

        # 2. Tìm cột GIÁ ĐÓNG CỬA: Quét chữ 'close', 'dongcua', 'adj'
        price_col = None
        # Ưu tiên giá điều chỉnh (adj) trước, rồi đến đóng cửa (close/dongcua)
        price_keywords = ['adjclose', 'giadieuchinh', 'close', 'dongcua', 'price', 'gia']

        for keyword in price_keywords:
            match = next((col for col in col_map.keys() if keyword in col), None)
            if match:
                price_col = col_map[match]
                print(f"   -> phát hiện cột Giá là: '{price_col}' (khớp từ khóa '{keyword}')")
                break

        if not price_col:
            print("   Không tìm thấy cột Giá!")
            return None

        # --- XỬ LÝ DỮ LIỆU ---
        # CafeF thường trả về dd/mm/yyyy
        try:
            df['Date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y')
        except:
            # Nếu format lỗi, thử để Pandas tự đoán
            df['Date'] = pd.to_datetime(df[date_col], dayfirst=True)

        df['Close'] = df[price_col]

        # Làm sạch
        df = df[['Date', 'Close']].sort_values('Date').dropna()
        df = df.set_index('Date')
        df.columns = ['^VNINDEX']

        save_path = os.path.join(output_dir, 'vnindex_manual.csv')
        df.to_csv(save_path)
        print(f"   Đã lưu file VNINDEX: {save_path}")
        return df

    except Exception as e:
        print(f"    Lỗi: {e}")
        return None


def main():
    print("\n--- BẮT ĐẦU TẢI DỮ LIỆU ---")

    # 1. 30 MÃ YAHOO
    print(f"Đang tải 30 mã VN30 (Yahoo)...")
    vn30_tickers = [
        'ACB.VN', 'BCM.VN', 'BID.VN', 'CTG.VN', 'DGC.VN', 'FPT.VN',
        'GAS.VN', 'GVR.VN', 'HDB.VN', 'HPG.VN', 'LPB.VN', 'MBB.VN',
        'MSN.VN', 'MWG.VN', 'PLX.VN', 'SAB.VN', 'SHB.VN', 'SSB.VN',
        'SSI.VN', 'STB.VN', 'TCB.VN', 'TPB.VN', 'VCB.VN', 'VHM.VN',
        'VIB.VN', 'VIC.VN', 'VJC.VN', 'VNM.VN', 'VPB.VN', 'VRE.VN'
    ]
    try:
        yf_end = datetime.now().strftime('%Y-%m-%d')
        df_stocks = yf.download(vn30_tickers, start="2020-01-01", end=yf_end, progress=False, auto_adjust=True)
        if 'Close' in df_stocks.columns: df_stocks = df_stocks['Close']
        stock_path = os.path.join(output_dir, 'vn30_stocks_prices.csv')
        df_stocks.to_csv(stock_path)
        print(f"   Xong 30 mã cổ phiếu.")
    except Exception as e:
        print(f"   Lỗi tải cổ phiếu: {e}")

    # 2. VNINDEX CAFEF (Smart)
    get_vnindex_smart_detect()

    print("HOÀN TẤT!")


if __name__ == "__main__":
    main()