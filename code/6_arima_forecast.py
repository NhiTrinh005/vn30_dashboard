import pandas as pd
import numpy as np
import pmdarima as pm  # Thu vien cho auto_arima
from statsmodels.tsa.stattools import adfuller  # Kiem dinh dung
import matplotlib.pyplot as plt
import os
# Thu vien de danh gia loi
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- CÀI ĐẶT ĐƯỜNG DẪN ---
current_dir = os.path.dirname(os.path.realpath(__file__))
if os.path.exists(os.path.join(current_dir, 'data')):
    project_root = current_dir
else:
    project_root = os.path.dirname(current_dir)

processed_data_dir = os.path.join(project_root, 'data', 'processed')
input_file = os.path.join(processed_data_dir, 'vn30_monthly_returns.csv')

charts_output_dir = os.path.join(project_root, 'results', 'charts')
tables_output_dir = os.path.join(project_root, 'results', 'tables')

# Tạo thư mục nếu chưa có
os.makedirs(charts_output_dir, exist_ok=True)
os.makedirs(tables_output_dir, exist_ok=True)

# --- THAM SỐ CHỌN MÃ ---
TICKER_TO_FORECAST = 'FPT.VN'  # Mã tiêu biểu
N_TEST_PERIODS = 12  # Giữ lại 12 tháng cuối để kiểm tra (Test)
N_FORECAST_PERIODS = 12  # Dự báo 12 tháng tương lai


# --- HÀM 1: KIỂM ĐỊNH DỪNG ---
def check_stationarity(series, ticker_name):
    print(f"\n...[BUOC 6.1] Dang chay Kiem dinh dung (ADF) cho {ticker_name}...")
    result = adfuller(series.dropna())
    p_value = result[1]
    print(f"P-value cua kiem dinh ADF: {p_value:.4f}")
    if p_value < 0.05:
        print(f"=> Ket luan: Chuoi loi suat cua {ticker_name} la DUNG (stationary).")
    else:
        print(f"=> Ket luan: Chuoi loi suat cua {ticker_name} la KHONG DUNG.")


# --- HÀM 2: CHỌN MÔ HÌNH ---
def find_best_arima_model(series):
    print(f"\n...[BUOC 6.2] Dang tim mo hinh ARIMA tot nhat...")
    auto_model = pm.auto_arima(
        series, start_p=1, start_q=1, test='adf', max_p=3, max_q=3,
        m=12, d=None, seasonal=True, start_P=0, D=None,
        trace=False, error_action='ignore', suppress_warnings=True, stepwise=True
    )
    print("\n--- KET QUA CHON MO HINH ---")
    print(auto_model.summary())

    summary_path = os.path.join(tables_output_dir, 'arima_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(auto_model.summary().as_text())
    print(f"Da luu bang tom tat mo hinh vao: {summary_path}")
    return auto_model


# --- HÀM 3: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ---
def evaluate_model_on_test_set(model, series, n_test, ticker_name):
    print(f"\n...[BUOC 6.3] Dang danh gia mo hinh tren {n_test} thang Test...")

    # 1. Chia dữ liệu
    train_data = series[:-n_test]
    test_data = series[-n_test:]

    # 2. Huấn luyện lại trên tập Train
    model.fit(train_data)

    # 3. Dự báo tập Test
    test_forecast = model.predict(n_periods=n_test)

    # 4. Tính toán lỗi
    mae = mean_absolute_error(test_data, test_forecast)
    rmse = np.sqrt(mean_squared_error(test_data, test_forecast))

    print("\n--- KET QUA DANH GIA (TREN TAP TEST) ---")
    print(f"Loi tuyet doi trung binh (MAE): {mae:.4f}")
    print(f"Loi binh phuong trung binh (RMSE): {rmse:.4f}")

    # LƯU DATA ĐÁNH GIÁ RA CSV CHO DASHBOARD
    eval_df = pd.DataFrame({
        'Actual': test_data.values,
        'Forecast': test_forecast.values if hasattr(test_forecast, 'values') else test_forecast
    }, index=test_data.index)

    eval_csv_path = os.path.join(tables_output_dir, f'arima_evaluation_{ticker_name}.csv')
    eval_df.to_csv(eval_csv_path)
    print(f"-> [Dashboard] Da luu data danh gia vao: {eval_csv_path}")

    # 5. Vẽ biểu đồ tĩnh
    plt.figure(figsize=(16, 8))
    plt.plot(train_data, label='Du lieu Huan luyen (Train)')
    plt.plot(test_data, label='Du lieu Thuc te (Test)', color='orange', linestyle='--')
    plt.plot(test_data.index, test_forecast, label='Du bao (Forecast)', color='red', linestyle=':')
    plt.title(f'Danh gia Mo hinh ARIMA: {ticker_name} (So sanh Train/Test)', fontsize=18)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(charts_output_dir, f'arima_evaluation_plot_{ticker_name}.png'), dpi=300,
                bbox_inches='tight')
    plt.close()


# --- HÀM 4: PHÂN TÍCH PHẦN DƯ ---
def plot_residual_diagnostics(model, ticker_name):

    print(f"\n...[BUOC 6.4] Dang ve Bieu do Chan doan Phan du (Residuals)...")

    residuals = model.resid()
    # Nếu residuals là index time series thì giữ nguyên, không thì tạo DataFrame đơn giản
    resid_df = pd.DataFrame(residuals, columns=['Residuals'])

    resid_csv_path = os.path.join(tables_output_dir, f'arima_residuals_{ticker_name}.csv')
    resid_df.to_csv(resid_csv_path)
    print(f"-> [Dashboard] Da luu data phan du vao: {resid_csv_path}")

    # Vẽ biểu đồ tĩnh (PNG)
    fig = model.plot_diagnostics(figsize=(16, 10))
    fig.suptitle(f'Bieu do Chan doan Phan du (Residuals) - {ticker_name}', fontsize=18, y=1.03)
    plt.savefig(os.path.join(charts_output_dir, f'arima_residuals_plot_{ticker_name}.png'), dpi=300,
                bbox_inches='tight')
    plt.close()


# --- HÀM 5: DỰ BÁO TƯƠNG LAI ---
def forecast_future(model, series, ticker_name, n_periods):
    print(f"\n...[BUOC 6.5] Huan luyen tren 100% du lieu & Du bao {n_periods} thang toi...")

    # 1. Huấn luyện Full data
    model.fit(series)

    # 2. Dự báo
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True, alpha=0.05)

    # 3. Lưu Bảng dự báo
    forecast_df = pd.DataFrame({
        'DuBaoLoiSuat': forecast,
        'KhoangTinCay_Thap (95%)': conf_int[:, 0],
        'KhoangTinCay_Cao (95%)': conf_int[:, 1]
    })
    # Tạo index ngày tháng tương lai
    last_date = series.index[-1]
    forecast_index = pd.date_range(start=last_date, periods=n_periods + 1, freq='M')[1:]  # Pandas < 2.2
    # Nếu lỗi freq='M' (deprecated), dùng freq='ME'

    forecast_df.index = forecast_index

    output_path = os.path.join(tables_output_dir, f'arima_forecast_{ticker_name}.csv')
    forecast_df.to_csv(output_path, float_format='%.4f')
    print(f"Da luu bang Du bao vao: {output_path}")

    # Vẽ biểu đồ tĩnh
    plt.figure(figsize=(16, 8))
    plt.plot(series, label='Lich su')
    plt.plot(forecast_df.index, forecast_df['DuBaoLoiSuat'], label='Du bao', color='red', linestyle='--')
    plt.fill_between(forecast_df.index, forecast_df['KhoangTinCay_Thap (95%)'], forecast_df['KhoangTinCay_Cao (95%)'],
                     color='red', alpha=0.1)
    plt.title(f'Du bao 12 thang toi - {ticker_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(charts_output_dir, f'arima_forecast_plot_{ticker_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()


# --- HÀM CHÍNH ---
if __name__ == "__main__":
    print("--- Bat dau BUOC 6: Du bao ARIMA (Updated for Dashboard) ---")

    try:
        df_monthly_returns = pd.read_csv(input_file, index_col='Date', parse_dates=True)
        series_to_forecast = df_monthly_returns[TICKER_TO_FORECAST].dropna()

        if series_to_forecast.empty:
            print(f"LOI: Khong co du lieu cho {TICKER_TO_FORECAST}")
        else:
            # 1. Kiểm định
            check_stationarity(series_to_forecast, TICKER_TO_FORECAST)
            # 2. Tìm mô hình
            best_model = find_best_arima_model(series_to_forecast)
            # 3. Đánh giá
            evaluate_model_on_test_set(best_model, series_to_forecast, N_TEST_PERIODS, TICKER_TO_FORECAST)
            # 4. Phần dư
            # Lưu ý: fit lại để lấy resid chuẩn nhất
            best_model.fit(series_to_forecast)
            plot_residual_diagnostics(best_model, TICKER_TO_FORECAST)
            # 5. Dự báo tương lai
            forecast_future(best_model, series_to_forecast, TICKER_TO_FORECAST, N_FORECAST_PERIODS)

            print("\n--- Hoan thanh! ---")
            print("Da xuat day du file CSV cho Dashboard Interactive.")

    except Exception as e:
        print(f"Loi: {e}")