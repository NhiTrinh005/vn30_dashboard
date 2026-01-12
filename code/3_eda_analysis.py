import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CÀI ĐẶT ĐƯỜNG DẪN ---
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
processed_data_dir = os.path.join(project_root, 'data', 'processed')
input_file = os.path.join(processed_data_dir, 'vn30_monthly_returns.csv')

# (Tạo thư mục results/charts và results/tables)
charts_output_dir = os.path.join(project_root, 'results', 'charts')
tables_output_dir = os.path.join(project_root, 'results', 'tables')
os.makedirs(charts_output_dir, exist_ok=True)
os.makedirs(tables_output_dir, exist_ok=True)


# --- HÀM 1: KIỂM TRA DỮ LIỆU ---
def check_data(df_returns):
    """Kiểm tra nhanh dữ liệu và in ra thông tin."""
    print("...Dang kiem tra du lieu...")

    # Mục "xem kích cỡ" và "kiểu dữ liệu"
    print("\n--- Thong tin Du lieu (df.info()) ---")
    df_returns.info()
    print(f"\nKich co du lieu: {df_returns.shape}")

    # Mục "Thống kê nhanh số lượng giá trị thiếu"
    missing_values = df_returns.isnull().sum()
    print("\n--- Kiem tra Gia tri thieu (Missing Values) ---")
    if missing_values.sum() == 0:
        print("=> Du lieu da SẠCH (khong co gia tri thieu).")
    else:
        print(missing_values[missing_values > 0])

    print("\nKiem tra du lieu hoan tat.")


# --- HÀM 2: THỐNG KÊ MÔ TẢ ---
def calculate_descriptive_stats(df_returns):
    """Tính toán các chỉ số thống kê mô tả quan trọng."""
    print("\n...Dang tinh Thong ke mo ta...")

    annualized_returns = df_returns.mean() * 12
    annualized_volatility = df_returns.std() * np.sqrt(12)
    sharpe_ratio = annualized_returns / (annualized_volatility + 1e-9)

    summary_stats = pd.DataFrame({
        'LoiSuatTB_Nam': annualized_returns,
        'RuiRo_Nam (Std)': annualized_volatility,
        'Sharpe_Ratio': sharpe_ratio,
        'LoiSuatTB_Thang': df_returns.mean(),
        'Min_LoiSuat_Thang': df_returns.min(),
        'Max_LoiSuat_Thang': df_returns.max()
    })

    summary_stats = summary_stats.sort_values(by='Sharpe_Ratio', ascending=False)

    output_path = os.path.join(tables_output_dir, 'eda_thong_ke_mo_ta.csv')
    summary_stats.to_csv(output_path, float_format='%.4f')

    print(f"Da luu bang Thong ke mo ta vao: {output_path}")
    return summary_stats


# --- HÀM 3: VẼ HEATMAP ---
def plot_correlation_heatmap(df_returns):
    """Vẽ và lưu biểu đồ heatmap tương quan."""
    print("\n...Dang ve Heatmap tuong quan...")

    corr_matrix = df_returns.corr()
    plt.figure(figsize=(24, 20))
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt='.1f',
        annot_kws={'size': 8}
    )
    heatmap.set_title('Ban do nhiet Tuong quan Loi suat thang (VN30 & VNINDEX)',
                      fontdict={'fontsize': 20},
                      pad=20)

    output_path = os.path.join(charts_output_dir, 'eda_heatmap_tuong_quan.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Da luu Heatmap vao: {output_path}")
    plt.close()


# --- HÀM 4: VẼ BIỂU ĐỒ TĂNG TRƯỞNG ---
def plot_cumulative_returns(df_returns):
    """Vẽ biểu đồ tăng trưởng (cumulative returns) của 1 đồng đầu tư."""
    print("\n...Dang ve Bieu do Tang truong...")

    cumulative_returns = (1 + df_returns).cumprod()
    plt.figure(figsize=(16, 10))

    stocks_to_plot = cumulative_returns.drop(columns=['^VNINDEX'])
    plt.plot(stocks_to_plot, linewidth=1.0, alpha=0.5)

    if '^VNINDEX' in cumulative_returns.columns:
        plt.plot(cumulative_returns['^VNINDEX'],
                 label='^VNINDEX',
                 color='black',
                 linestyle='--',
                 linewidth=2.5)

    plt.title('Tang truong cua 1 dong dau tu (2020-2025)', fontsize=18)
    plt.xlabel('Ngay', fontsize=12)
    plt.ylabel('Gia tri (1 dong ban dau)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    output_path = os.path.join(charts_output_dir, 'eda_bieu_do_tang_truong.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Da luu Bieu do Tang truong vao: {output_path}")
    plt.close()


# --- HÀM 5: VẼ BOX PLOT ---
def plot_return_boxplots(df_returns):
    """Vẽ Box plot để so sánh phân phối lợi suất (rủi ro) của các mã."""
    print("\n...Dang ve Box Plot so sanh loi suat...")

    plt.figure(figsize=(25, 10))
    sns.boxplot(data=df_returns)
    plt.title('So sanh Phan phoi Loi suat thang (2020-2025)', fontsize=18)
    plt.ylabel('Loi suat thang', fontsize=12)
    plt.xlabel('Ma Chung Khoan', fontsize=12)
    plt.xticks(rotation=90)

    output_path = os.path.join(charts_output_dir, 'eda_boxplot_loi_suat.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Da luu Box Plot vao: {output_path}")
    plt.close()


# --- HÀM 6: VẼ HISTOGRAM PHÂN PHỐI ---
def plot_return_distributions(df_returns):
    """Vẽ biểu đồ phân phối (histogram/density) cho các mã tiêu biểu."""
    print("\n...Dang ve Bieu do Phan phoi (Histogram)...")

    plt.figure(figsize=(16, 8))

    sns.histplot(df_returns['FPT.VN'], kde=True, label='FPT.VN', color='blue', stat='density')
    sns.histplot(df_returns['VIC.VN'], kde=True, label='VIC.VN', color='red', stat='density')
    sns.histplot(df_returns['^VNINDEX'], kde=True, label='^VNINDEX', color='black', stat='density')

    plt.title('Bieu do phan phoi loi suat thang (Histogram & Density)', fontsize=18)
    plt.xlabel('Loi suat thang', fontsize=12)
    plt.ylabel('Mat do (Density)', fontsize=12)
    plt.legend()

    output_path = os.path.join(charts_output_dir, 'eda_histogram_phan_phoi.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Da luu Histogram Phan phoi vao: {output_path}")
    plt.close()


# --- HÀM CHÍNH ĐỂ CHẠY ---
if __name__ == "__main__":
    print("--- Bat dau BUOC 3: Phan tich EDA ---")

    try:
        # 1. Đọc file "vàng" (lợi suất tháng)
        df_monthly_returns = pd.read_csv(input_file, index_col='Date', parse_dates=True)
        print(
            f"Doc thanh cong file loi suat voi {df_monthly_returns.shape[0]} thang va {df_monthly_returns.shape[1]} cot.")

        # 2. Chạy hàm Kiểm tra dữ liệu
        check_data(df_monthly_returns)

        # 3. Chạy hàm tính Thống kê mô tả
        calculate_descriptive_stats(df_monthly_returns)

        # 4. Chạy hàm vẽ Heatmap
        plot_correlation_heatmap(df_monthly_returns)

        # 5. Chạy hàm vẽ Biểu đồ Tăng trưởng
        plot_cumulative_returns(df_monthly_returns)

        # 6. Chạy hàm vẽ Box Plot
        plot_return_boxplots(df_monthly_returns)

        # 7. Chạy hàm vẽ Histogram
        plot_return_distributions(df_monthly_returns)

        print("\n--- Hoan thanh BUOC 3 ---")
        print(f"Vui long kiem tra 2 thu muc ket qua moi:")
        print(f"Bang (CSV): {tables_output_dir}")
        print(f"Bieu do (PNG): {charts_output_dir}")

    except FileNotFoundError:
        print(f"LOI: Khong tim thay file: {input_file}")
        print("Vui long chay lai file '2_processing.py' truoc.")
    except Exception as e:
        print(f"Da co loi xay ra: {e}")