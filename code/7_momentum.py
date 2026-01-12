import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CÀI ĐẶT ĐƯỜNG DẪN ---
# 1. Lấy đường dẫn thư mục gốc
script_dir = os.path.dirname(os.path.realpath(__file__))
if os.path.exists(os.path.join(script_dir, 'data')):
    project_root = script_dir
else:
    project_root = os.path.dirname(script_dir)

# 2. Đường dẫn input
processed_data_dir = os.path.join(project_root, 'data', 'processed')
tables_input_dir = os.path.join(project_root, 'results', 'tables')

input_file_returns = os.path.join(processed_data_dir, 'vn30_monthly_returns.csv')
input_file_betas = os.path.join(tables_input_dir, 'capm_beta_results.csv')

# 3. Đường dẫn output
charts_output_dir = os.path.join(project_root, 'results', 'charts')
tables_output_dir = os.path.join(project_root, 'results', 'tables')

os.makedirs(charts_output_dir, exist_ok=True)
os.makedirs(tables_output_dir, exist_ok=True)


# --- HÀM 1: TÁI TẠO 2 DANH MỤC BETA (DEFENSIVE / AGGRESSIVE) ---
def get_beta_portfolios(df_returns, df_betas, num_stocks=5):
    """
    Tái tạo lợi suất của Defensive (Beta thấp) và Aggressive (Beta cao)
    """
    print("...Dang tai tao 2 danh muc Beta...")

    # Xử lý tên cột Beta
    beta_col = 'Beta (B)'
    for c in df_betas.columns:
        if 'beta' in c.lower():
            beta_col = c
            break

    # Lọc P-value (nếu có cột P-Value)
    if 'P-Value' in df_betas.columns:
        df_betas = df_betas[df_betas['P-Value'] < 0.1]

    df_betas_sorted = df_betas.sort_values(by=beta_col)

    # Lấy Top Beta thấp > 0 (Defensive)
    low_beta_tickers = df_betas_sorted[df_betas_sorted[beta_col] > 0].head(num_stocks).index
    # Lấy Top Beta cao (Aggressive)
    high_beta_tickers = df_betas_sorted.tail(num_stocks).index

    # Tính lợi suất trung bình
    portfolio_defensive = df_returns[low_beta_tickers].mean(axis=1)
    portfolio_aggressive = df_returns[high_beta_tickers].mean(axis=1)

    return portfolio_defensive, portfolio_aggressive


# --- HÀM 2: TÍNH MOMENTUM STRATEGY ---
def get_momentum_portfolio(df_returns, lookback_period=6, num_stocks=5):
    """
    Chiến lược Momentum: Mua Top 5 mã tăng mạnh nhất trong lookback_period tháng qua.
    """
    print(f"...Dang chay Backtest Momentum (Lookback={lookback_period}, Top={num_stocks})...")

    portfolio_momentum_returns = pd.Series(index=df_returns.index, dtype=float)
    stock_returns = df_returns.drop(columns=['^VNINDEX'], errors='ignore')

    # Bắt đầu vòng lặp backtest
    for i in range(lookback_period, len(stock_returns)):
        current_month = stock_returns.index[i]

        # 1. Tính hiệu suất quá khứ (Cumulative Return)
        # Lấy dữ liệu từ t-lookback đến t-1
        past_data = stock_returns.iloc[i - lookback_period: i]
        momentum_score = (1 + past_data).prod() - 1

        # 2. Chọn Top N mã thắng (Winners)
        winners = momentum_score.nlargest(num_stocks).index

        # 3. Tính lợi suất tháng hiện tại của danh mục Winners
        # (Giả định mua Equal Weight)
        current_return = stock_returns.loc[current_month, winners].mean()

        portfolio_momentum_returns.iloc[i] = current_return

    return portfolio_momentum_returns.dropna()


# --- HÀM CHÍNH ---
if __name__ == "__main__":
    print("--- Bat dau BUOC 7: Phan tich Momentum (Standardized) ---")

    try:
        # 1. Load Data
        df_returns = pd.read_csv(input_file_returns, index_col='Date', parse_dates=True)
        df_betas = pd.read_csv(input_file_betas, index_col=0)  # Index là Ticker

        # 2. Tái tạo danh mục Beta
        ret_defensive, ret_aggressive = get_beta_portfolios(df_returns, df_betas)

        # 3. Chạy chiến lược Momentum
        # Cấu hình: Lookback 6 tháng, Top 5 mã (Chuẩn phổ biến)
        ret_momentum = get_momentum_portfolio(df_returns, lookback_period=6, num_stocks=5)

        # 4. Gộp tất cả vào 1 DataFrame & ĐẶT TÊN CHUẨN
        df_final = pd.DataFrame({
            'Defensive': ret_defensive,
            'Aggressive': ret_aggressive,
            'Momentum Strategy': ret_momentum,
            'Market (VN30)': df_returns['^VNINDEX']
        }).dropna()

        print(f"So lieu so sanh san sang: {len(df_final)} thang.")

        # --- XUẤT DỮ LIỆU 1: BIỂU ĐỒ TĂNG TRƯỞNG (CHO DASHBOARD VẼ) ---
        cum_growth = (1 + df_final).cumprod()
        chart_path = os.path.join(tables_output_dir, 'bonus_momentum_chart_data.csv')
        cum_growth.to_csv(chart_path)
        print(f"-> Da xuat data bieu do: {chart_path}")

        # --- XUẤT DỮ LIỆU 2: BẢNG SO SÁNH HIỆU QUẢ ---
        # Tính chỉ số
        ann_ret = df_final.mean() * 12
        ann_vol = df_final.std() * (12 ** 0.5)
        sharpe = ann_ret / (ann_vol + 1e-9)
        max_dd = (cum_growth / cum_growth.cummax() - 1).min()

        df_metrics = pd.DataFrame({
            'Lợi suất/Năm': ann_ret,
            'Rủi ro (Std)': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd
        })

        # Đổi tên dòng (Index) sang tiếng Việt để hiện lên Dashboard
        rename_map = {
            'Defensive': 'Defensive (Phòng thủ)',
            'Aggressive': 'Aggressive (Tăng trưởng)',
            'Momentum Strategy': 'Chiến lược Momentum',
            'Market (VN30)': 'Thị trường (VN30)'
        }
        df_metrics.rename(index=rename_map, inplace=True)

        metrics_path = os.path.join(tables_output_dir, 'bonus_momentum_hieu_qua_so_sanh.csv')
        df_metrics.to_csv(metrics_path)
        print(f"-> Da xuat data bang hieu qua: {metrics_path}")

        # --- VẼ BIỂU ĐỒ TĨNH (PNG) ---
        plt.figure(figsize=(12, 6))
        plt.plot(cum_growth['Momentum Strategy'], label='Momentum', color='green', linewidth=2)
        plt.plot(cum_growth['Market (VN30)'], label='VN30', color='black', linestyle='--')
        plt.title('Backtest: Momentum Strategy vs VN30')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(charts_output_dir, 'bonus_momentum_so_sanh_tang_truong.png'))

        print("\n--- HOAN THANH! ---")

    except Exception as e:
        print(f"Loi: {e}")