import subprocess
import os
import sys

# --- CẤU HÌNH ---
# Lấy đường dẫn của thư mục chứa file này (thư mục 'code')
current_dir = os.path.dirname(os.path.realpath(__file__))

# Danh sách các file cần chạy theo thứ tự
scripts = [
    "1_fetch_data.py",
    "2_clean_data.py",
    "3_eda_analysis.py",
    "4_capm_beta.py",
    "5_portfolio_analysis.py",
    "6_arima_forecast.py",
    "7_momentum.py"
]


def run_script(script_name):
    """Hàm chạy một file Python con"""
    script_path = os.path.join(current_dir, script_name)
    print(f"\n{'=' * 60}")
    print(f"🚀 ĐANG CHẠY: {script_name}")
    print(f"{'=' * 60}")

    try:
        # Gọi lệnh python để chạy file con
        # check=True để nếu file con lỗi thì pipeline dừng lại luôn
        subprocess.run([sys.executable, script_path], check=True)
        print(f"✅ HOÀN THÀNH: {script_name}")
    except subprocess.CalledProcessError:
        print(f"❌ LỖI: {script_name} gặp sự cố. Pipeline dừng lại.")
        sys.exit(1)  # Thoát chương trình


def main():
    print("🎬 BẮT ĐẦU QUY TRÌNH TỰ ĐỘNG (PIPELINE)...")

    for script in scripts:
        run_script(script)

    print(f"\n{'=' * 60}")
    print("🎉🎉🎉 TẤT CẢ ĐÃ HOÀN TẤT! 🎉🎉🎉")
    print("Bây giờ bạn có thể chạy Dashboard bằng lệnh:")
    print("streamlit run code/dashboard.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()