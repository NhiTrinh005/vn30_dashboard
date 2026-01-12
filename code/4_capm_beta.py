import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CÀI ĐẶT ĐƯỜNG DẪN ---
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
processed_data_dir = os.path.join(project_root, 'data', 'processed')
input_file = os.path.join(processed_data_dir, 'vn30_monthly_returns.csv')

charts_output_dir = os.path.join(project_root, 'results', 'charts')
tables_output_dir = os.path.join(project_root, 'results', 'tables')


# --- HÀM 1: CHẠY CAPM ---
def calculate_capm_betas(df_returns):
    """
    Chạy mô hình CAPM (Hồi quy) cho từng mã cổ phiếu so với VNINDEX.
    """
    print("...Dang chay mo hinh CAPM cho 30 ma co phieu...")

    market_returns = df_returns['^VNINDEX']
    stock_returns = df_returns.drop(columns=['^VNINDEX'])

    capm_results = []

    for ticker in stock_returns.columns:
        try:
            stock_y = stock_returns[ticker]

            pair_df = pd.concat([stock_y, market_returns], axis=1).dropna()

            if len(pair_df) < 12:
                print(f"Canh bao: Ma {ticker} co qua it du lieu ({len(pair_df)} thang), bo qua...")
                continue

            y_var = pair_df[ticker]
            x_var = sm.add_constant(pair_df['^VNINDEX'])

            model = sm.OLS(y_var, x_var).fit()

            alpha = model.params['const']
            beta = model.params['^VNINDEX']
            r_squared = model.rsquared
            p_value_beta = model.pvalues['^VNINDEX']

            capm_results.append({
                'Ticker': ticker,
                'Alpha (hang thang)': alpha,
                'Beta (B)': beta,
                'R-Squared (R2)': r_squared,
                'P-Value (Beta)': p_value_beta,
                'SoThangQuanSat': len(pair_df)
            })

        except Exception as inner_e:
            # Nếu 1 mã bị lỗi, chỉ in ra và tiếp tục vòng lặp
            print(f"LOI: Khong the tinh Beta cho ma {ticker}. Loi: {inner_e}")
            continue

    results_df = pd.DataFrame(capm_results).set_index('Ticker')
    results_df = results_df.sort_values(by='Beta (B)', ascending=False)

    output_path = os.path.join(tables_output_dir, 'capm_beta_results.csv')
    results_df.to_csv(output_path, float_format='%.4f')

    print(f"Da luu bang ket qua Alpha/Beta/R2 vao: {output_path}")
    return results_df


# --- HÀM 2: VẼ BIỂU ĐỒ HỒI QUY ---
def plot_capm_regression(df_returns, ticker, capm_results):
    """
    Vẽ biểu đồ Scatter plot (phân tán) và đường hồi quy CAPM.
    """
    print(f"...Dang ve bieu do hoi quy cho {ticker}...")

    beta = capm_results.loc[ticker]['Beta (B)']
    alpha = capm_results.loc[ticker]['Alpha (hang thang)']
    df_pair = pd.concat([df_returns[ticker], df_returns['^VNINDEX']], axis=1).dropna()

    plt.figure(figsize=(12, 7))
    sns.regplot(
        x='^VNINDEX',
        y=ticker,
        data=df_pair,
        line_kws={'color': 'red', 'linestyle': '--'},
        scatter_kws={'alpha': 0.5}
    )

    plt.title(f'Mo hinh CAPM: {ticker} vs. VNINDEX', fontsize=16)
    plt.xlabel('Loi suat VNINDEX (Market)', fontsize=12)
    plt.ylabel(f'Loi suat {ticker} (Stock)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.text(
        0.05, 0.95,
        f'Alpha = {alpha:.4f}\nBeta = {beta:.4f}',
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    )

    output_path = os.path.join(charts_output_dir, f'capm_regression_plot_{ticker}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- HÀM 3: VẼ PHÂN PHỐI BETA ---
def plot_beta_distribution(capm_results):
    """Vẽ histogram/density plot của các hệ số Beta."""
    print("...Dang ve bieu do phan phoi Beta...")

    plt.figure(figsize=(12, 7))
    sns.histplot(capm_results['Beta (B)'], kde=True, bins=10)

    beta_mean = capm_results['Beta (B)'].mean()
    plt.axvline(beta_mean, color='red', linestyle='--', label=f'Beta Trung Binh = {beta_mean:.2f}')
    plt.axvline(1.0, color='black', linestyle='-', label='Beta = 1.0 (Thi Truong)')

    plt.title('Bieu do Phan phoi He so Beta cua VN30', fontsize=16)
    plt.xlabel('He so Beta (B)', fontsize=12)
    plt.ylabel('So luong ma (Count)', fontsize=12)
    plt.legend()

    output_path = os.path.join(charts_output_dir, 'capm_beta_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- HÀM 4: VẼ XẾP HẠNG BETA ---
def plot_beta_ranking_chart(capm_results):
    """Vẽ biểu đồ thanh (bar chart) xếp hạng Beta từ cao đến thấp."""
    print("...Dang ve bieu do xep hang Beta...")

    df_to_plot = capm_results.reset_index()

    plt.figure(figsize=(10, 15))

    sns.barplot(
        x='Beta (B)',
        y='Ticker',
        data=df_to_plot,
        palette='vlag'
    )

    plt.title('Xep hang He so Beta (Rui ro He thong) cua VN30', fontsize=16)
    plt.xlabel('He so Beta (B)', fontsize=12)
    plt.ylabel('Ma Chung Khoan', fontsize=12)

    output_path = os.path.join(charts_output_dir, 'capm_beta_ranking.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- HÀM CHÍNH ĐỂ CHẠY ---
if __name__ == "__main__":
    print("--- Bat dau BUOC 4: Mo hinh CAPM & Tinh Beta ---")

    try:
        df_monthly_returns = pd.read_csv(input_file, index_col='Date', parse_dates=True)
        print(
            f"Doc thanh cong file loi suat voi {df_monthly_returns.shape[0]} thang va {df_monthly_returns.shape[1]} cot.")

        df_capm_results = calculate_capm_betas(df_monthly_returns)

        if not df_capm_results.empty:
            if 'FPT.VN' in df_capm_results.index:
                plot_capm_regression(df_monthly_returns, 'FPT.VN', df_capm_results)
            if 'VIC.VN' in df_capm_results.index:
                plot_capm_regression(df_monthly_returns, 'VIC.VN', df_capm_results)

            plot_beta_distribution(df_capm_results)
            plot_beta_ranking_chart(df_capm_results)

            print("\n--- Hoan thanh BUOC 4 ---")
            print(f"Vui long kiem tra bang ket qua ('capm_beta_results.csv') trong thu muc: {tables_output_dir}")
            print(f"Va cac bieu do moi trong thu muc: {charts_output_dir}")

        else:
            print("LOI: Khong tinh toan duoc Beta cho bat ky ma nao. Kiem tra lai du lieu.")

    except FileNotFoundError:
        print(f"LOI: Khong tim thay file: {input_file}")
        print("Vui long chay lai file '2_processing.py' truoc.")
    except Exception as e:
        print(f"Da co loi xay ra: {e}")