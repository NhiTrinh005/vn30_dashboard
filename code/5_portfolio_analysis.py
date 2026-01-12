import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CÀI ĐẶT ĐƯỜNG DẪN ---
current_dir = os.path.dirname(os.path.realpath(__file__))
if os.path.exists(os.path.join(current_dir, 'data')):
    project_root = current_dir
else:
    project_root = os.path.dirname(current_dir)

processed_data_dir = os.path.join(project_root, 'data', 'processed')
tables_input_dir = os.path.join(project_root, 'results', 'tables')

input_file_returns = os.path.join(processed_data_dir, 'vn30_monthly_returns.csv')
input_file_betas = os.path.join(tables_input_dir, 'capm_beta_results.csv')

charts_output_dir = os.path.join(project_root, 'results', 'charts')
tables_output_dir = os.path.join(project_root, 'results', 'tables')

# Tạo thư mục nếu chưa có
os.makedirs(charts_output_dir, exist_ok=True)
os.makedirs(tables_output_dir, exist_ok=True)


# --- HÀM 1: XÂY DỰNG DANH MỤC (SỬA TÊN TẠI ĐÂY) ---
def construct_portfolios(df_returns, df_betas, num_stocks=5):
    """
    Phân loại và xây dựng 2 danh mục (Aggressive, Defensive)
    """
    print(f"...Dang phan loai va xay dung 2 danh muc (Top {num_stocks} ma)...")

    # Sắp xếp Beta tăng dần
    df_betas_sorted = df_betas.sort_values(by='Beta (B)')

    # Lấy Top Beta thấp (Dương) -> Defensive
    low_beta_tickers = df_betas_sorted[df_betas_sorted['Beta (B)'] > 0].head(num_stocks).index
    # Lấy Top Beta cao -> Aggressive
    high_beta_tickers = df_betas_sorted.tail(num_stocks).index

    print(f"Danh muc DEFENSIVE (Phong thu): {list(low_beta_tickers)}")
    print(f"Danh muc AGGRESSIVE (Tang truong): {list(high_beta_tickers)}")

    # 1. Lưu file Phân loại danh mục
    low_beta_df = pd.DataFrame({'Ticker': low_beta_tickers, 'LoaiDanhMuc': 'Defensive (Phòng thủ)'})
    high_beta_df = pd.DataFrame({'Ticker': high_beta_tickers, 'LoaiDanhMuc': 'Aggressive (Tăng trưởng)'})

    composition_df = pd.concat([low_beta_df, high_beta_df]).set_index('Ticker')

    output_path = os.path.join(tables_output_dir, 'danh_muc_phan_loai.csv')
    composition_df.to_csv(output_path)
    print(f"Da luu bang Phan loai danh muc vao: {output_path}")

    # 2. Tính lợi suất của danh mục
    portfolio_low_beta = df_returns[low_beta_tickers].mean(axis=1)
    portfolio_high_beta = df_returns[high_beta_tickers].mean(axis=1)

    portfolios_df = pd.DataFrame({
        'Defensive': portfolio_low_beta,
        'Aggressive': portfolio_high_beta,
        'Market': df_returns['^VNINDEX']
    })

    return portfolios_df


# --- HÀM 2: TÍNH TOÁN CHỈ SỐ ---
def calculate_performance_metrics(df_returns):
    print("...Dang tinh toan chi so hieu qua (Sharpe, Drawdown)...")

    annualized_returns = df_returns.mean() * 12
    annualized_volatility = df_returns.std() * np.sqrt(12)
    sharpe_ratio = annualized_returns / (annualized_volatility + 1e-9)

    cumulative_returns = (1 + df_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    metrics = pd.DataFrame({
        'LoiSuatTB_Nam': annualized_returns,
        'RuiRo_Nam (Std)': annualized_volatility,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown
    })

    output_path = os.path.join(tables_output_dir, 'danh_muc_hieu_qua.csv')
    metrics.to_csv(output_path, float_format='%.4f')
    print(f"Da luu bang chi so hieu qua vao: {output_path}")
    return metrics


# --- HÀM 3: VẼ BIỂU ĐỒ TĂNG TRƯỞNG ---
def plot_portfolio_growth(df_returns):
    print("...Dang ve Bieu do Tang truong Danh muc...")

    cumulative_returns = (1 + df_returns).cumprod()

    plt.figure(figsize=(16, 10))

    plt.plot(cumulative_returns['Defensive'], label='Defensive (Phòng thủ)', color='#28a745', linewidth=2)
    plt.plot(cumulative_returns['Aggressive'], label='Aggressive (Tăng trưởng)', color='#dc3545', linewidth=2)
    plt.plot(cumulative_returns['Market'], label='Market (VN30)', color='#1a2236', linestyle='--', linewidth=2.5)

    plt.title('So sánh Tăng trưởng Tài sản (Wealth Index)', fontsize=18)
    plt.xlabel('Thời gian', fontsize=12)
    plt.ylabel('Giá trị tài sản (Từ 1 VND ban đầu)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)

    output_path = os.path.join(charts_output_dir, 'danh_muc_bieu_do_tang_truong.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- HÀM 4: VẼ BIỂU ĐỒ DRAWDOWN  ---
def plot_drawdown_comparison(df_returns):
    print("...Dang ve Bieu do Sut giam (Drawdown)...")

    cumulative_returns = (1 + df_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max

    plt.figure(figsize=(16, 8))

    plt.plot(drawdown['Defensive'], label='Defensive', color='#28a745', linewidth=1.5)
    plt.plot(drawdown['Aggressive'], label='Aggressive', color='#dc3545', linewidth=1.5)
    plt.plot(drawdown['Market'], label='Market', color='#1a2236', linestyle='--', linewidth=2)

    plt.title('Biểu đồ Sụt giảm tối đa (Maximum Drawdown)', fontsize=18)
    plt.xlabel('Thời gian', fontsize=12)
    plt.ylabel('Mức sụt giảm (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)

    output_path = os.path.join(charts_output_dir, 'danh_muc_bieu_do_drawdown.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- HÀM 5: VẼ BIỂU ĐỒ SO SÁNH HIỆU QUẢ ---
def plot_performance_barchart(metrics_data):
    print("...Dang ve Bieu do cot So sanh Hieu qua...")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

    # Màu sắc: Xanh (Defensive), Đỏ (Aggressive), Đen (Market)
    colors = ['#28a745', '#dc3545', '#1a2236']

    metrics_data['LoiSuatTB_Nam'].plot(kind='bar', ax=axes[0], color=colors)
    axes[0].set_title('Lợi suất Trung bình Năm', fontsize=14)
    axes[0].tick_params(axis='x', rotation=0)

    metrics_data['RuiRo_Nam (Std)'].plot(kind='bar', ax=axes[1], color=colors)
    axes[1].set_title('Rủi ro (Độ lệch chuẩn)', fontsize=14)
    axes[1].tick_params(axis='x', rotation=0)

    metrics_data['Sharpe_Ratio'].plot(kind='bar', ax=axes[2], color=colors)
    axes[2].set_title('Chỉ số Sharpe Ratio', fontsize=14)
    axes[2].tick_params(axis='x', rotation=0)

    fig.suptitle('Tổng hợp Hiệu quả Đầu tư', fontsize=18, y=1.03)
    plt.tight_layout()

    output_path = os.path.join(charts_output_dir, 'danh_muc_so_sanh_hieu_qua.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- HÀM CHÍNH ---
if __name__ == "__main__":
    print("--- Bat dau BUOC 5: Phan tich Danh muc (Updated Names) ---")

    try:
        df_monthly_returns = pd.read_csv(input_file_returns, index_col='Date', parse_dates=True)
        # Sử dụng hàm dò file CAPM thông minh (nếu cần) hoặc đọc trực tiếp
        # giả sử tên file đầu vào là chuẩn
        df_capm_results = pd.read_csv(input_file_betas, index_col='Ticker')

        # XỬ LÝ TÊN CỘT BETA
        beta_col = 'Beta (B)'  # Mặc định
        for col in df_capm_results.columns:
            if 'beta' in col.lower():
                beta_col = col
                break

        # Đổi tên cột Beta về chuẩn để sort
        df_capm_results.rename(columns={beta_col: 'Beta (B)'}, inplace=True)

        print(f"Doc thanh cong du lieu. Bat dau xay dung danh muc...")

        # Lọc Beta có ý nghĩa
        if 'P-Value' in df_capm_results.columns:
            significant_betas = df_capm_results[df_capm_results['P-Value'] < 0.1]
        elif 'P-Value (Beta)' in df_capm_results.columns:
            significant_betas = df_capm_results[df_capm_results['P-Value (Beta)'] < 0.1]
        else:
            significant_betas = df_capm_results  # Lấy hết nếu không có cột P-Value

        N_STOCKS = min(5, len(significant_betas) // 2)

        if N_STOCKS < 1:
            print("LOI: Khong du ma co phi co Beta y nghia.")
        else:
            # 1. Xây dựng danh mục
            df_portfolios = construct_portfolios(df_monthly_returns, significant_betas, num_stocks=N_STOCKS)

            # 2. Tính chỉ số
            df_metrics = calculate_performance_metrics(df_portfolios)

            # 3. Vẽ biểu đồ
            plot_portfolio_growth(df_portfolios)
            plot_drawdown_comparison(df_portfolios)
            plot_performance_barchart(df_metrics)

            print("\n--- Hoan thanh! ---")
            print(f"File CSV moi da co ten cot chuan: {tables_output_dir}")

    except Exception as e:
        print(f"Loi: {e}")