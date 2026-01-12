# FINANCIAL DASHBOARD PROJECT: VN30 STOCK ANALYSIS

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

> **Project Description:** Hệ thống End-to-End tự động thu thập dữ liệu tài chính, phân tích rủi ro (CAPM), dự báo xu hướng (ARIMA), Backtest chiến lược (Momentum) và xuất báo cáo khuyến nghị đầu tư tự động.
---
## Thông tin Tác giả

 **Developer**: **Trinh Hue Nhi** 

---
## Tính năng Chính (Key Features)

Dự án cung cấp một bộ công cụ toàn diện cho phân tích đầu tư:

- 📊 **Dashboard Tương tác:** Theo dõi biến động giá, so sánh hiệu quả đầu tư real-time.
- 🤖 **AI Assistant:** Sử dụng OpenAI GPT để hỗ trợ diễn giải kết quả và trả lời câu hỏi người dùng dựa trên output của mô hình định lượng.
- 🛡️ **Phân tích Rủi ro (CAPM):** Tự động tính Beta, phân loại cổ phiếu Tấn công/Phòng thủ.
- 🔮 **Dự báo Giá (ARIMA):** Mô hình Auto-ARIMA dự báo xu hướng giá 12 tháng tới.
- 🚀 **Chiến lược Momentum:** Backtest chiến lược đầu tư theo đà tăng trưởng.
- 📑 **Báo cáo Tự động:** Xuất báo cáo phân tích chi tiết dưới dạng HTML/PDF chỉ với 1 click.

---
## Công nghệ Sử dụng

Dự án sử dụng các thư viện Python chuyên dụng cho Tài chính & Khoa học dữ liệu:

- **Streamlit**: Xây dựng giao diện Web App tương tác.
- **Yfinance & Requests**: Thu thập dữ liệu từ Yahoo Finance và Crawler dữ liệu VNINDEX từ CafeF.
- **Pandas & Numpy**: Xử lý, làm sạch và tính toán dữ liệu tài chính.
- **Plotly**: Vẽ biểu đồ tương tác (Interactive Charts).
- **Statsmodels & Pmdarima**: Xây dựng mô hình định lượng (CAPM, ARIMA).
- **OpenAI API**: Tích hợp trí tuệ nhân tạo để viết nhận xét tự động.

---
## Cấu trúc Thư mục Dự án
Dự án được tổ chức khoa học, tách biệt giữa Mã nguồn (Code), Dữ liệu (Data) và Kết quả (Results):
```text
vn30_dashboard/
├── code/                       # Source Code (Pipeline xử lý)
│   ├── 1_fetch_data.py         # Tải data tự động (Yahoo + CafeF Crawler)
│   ├── 2_clean_data.py         # Làm sạch, ghép nối & Tính lợi suất
│   ├── 3_eda_analysis.py       # Phân tích khám phá (EDA) & Thống kê
│   ├── 4_capm_beta.py          # Mô hình CAPM & Phân loại Rủi ro Beta
│   ├── 5_portfolio_analysis.py # Tối ưu hóa Danh mục (Efficient Frontier)
│   ├── 6_arima_forecast.py     # Dự báo chuỗi thời gian (ARIMA)
│   ├── 7_momentum.py           # Chiến lược Momentum Strategy (Backtest)
│   ├── run_pipeline.py         # Script chạy tự động toàn bộ hệ thống (One-click)
│   └── dashboard.py            # Giao diện Web tương tác (Streamlit App)
├── data/                       # Kho dữ liệu
│   ├── raw/                    # Dữ liệu thô tải từ API
│   └── processed/              # Dữ liệu đã xử lý (Returns, Prices)
├── results/                    # Kết quả đầu ra (Outputs)
│   ├── charts/                 # Biểu đồ PNG (Tự động lưu từ code)
│   └── tables/                 # Bảng số liệu CSV
├── requirements.txt            # Danh sách thư viện cần thiết
└── README.md                   # Tài liệu hướng dẫn sử dụng
```
---
## Cài đặt & Hướng dẫn Sử dụng
**1. Yêu cầu cài đặt**
Dự án sử dụng **Python 3.12**. Để đảm bảo mã nguồn chạy ổn định, vui lòng cài đặt các thư viện cần thiết bằng lệnh sau trong Terminal:
```bash
pip install -r requirements.txt
```
*(Nếu cài thủ công, các thư viện chính bao gồm: `pandas`, `numpy`, `matplotlib`, `seaborn`, `yfinance`, `statsmodels`, `pmdarima`, `scikit-learn`, `streamlit`, `plotly`,`openai`,`requests`, `openpyxl`.*

**2. Cập nhật Dữ liệu (Pipeline)**
- Dự án tích hợp script tự động hóa. Để cập nhật dữ liệu và chạy lại toàn bộ mô hình (từ bước 1 đến bước 7), bạn chỉ cần chạy một lệnh duy nhất:
```bash
python code/run_pipeline.py
```
(Hệ thống sẽ tự động tải data mới -> làm sạch -> chạy CAPM -> ARIMA -> Momentum -> Lưu kết quả)

**3. Khởi động Dashboard Tương tác**
Dự án bao gồm một Dashboard trực quan được xây dựng bằng **Streamlit**, cho phép xem kết quả và biểu đồ tương tác. Cách chạy Dashboard:
3.1. Mở Terminal tại thư mục gốc của dự án.
3.2.  Chạy lệnh sau:
```bash
streamlit run code/dashboard.py
```
Trình duyệt web sẽ tự động mở ra giao diện Dashboard. Truy cập tại địa chỉ: http://localhost:8501.

---
## ❗Giả định, Hạn chế và Lưu ý
**Dự án được xây dựng với mục đích học thuật, không cấu thành lời khuyên tài chính hoặc khuyến nghị đầu tư. Do đó tồn tại một số giả định đơn giản hóa:**
- Phí giao dịch & Trượt giá (Slippage): Backtest giả định mua bán tại giá đóng cửa, chưa tính phí giao dịch (0.1-0.2%) và tác động của lệnh lớn lên thị trường.
- Cổ tức: Dữ liệu giá chưa điều chỉnh đầy đủ cổ tức tiền mặt trong một số trường hợp.
- Giả định ARIMA: Mô hình giả định dữ liệu quá khứ lặp lại trong tương lai, có thể không chính xác khi có tin tức vĩ mô đột biến (Thiên nga đen).
- Thanh khoản: Chiến lược chưa lọc các mã có thanh khoản thấp, có thể khó giải ngân trong thực tế.

**OpenAI API Key:**
- Để sử dụng tính năng Chat với AI và Nhận xét thông minh trong báo cáo, bạn cần nhập API Key của mình vào ô nhập liệu trên Dashboard (Tab Utilities).
- Nếu không có Key, hệ thống vẫn hoạt động bình thường ở chế độ "Offline" (sử dụng nhận xét mẫu).

**Dữ liệu VNINDEX:**
Hệ thống sử dụng cơ chế "Smart Detect" để crawl dữ liệu từ CafeF. Nếu cấu trúc web CafeF thay đổi, có thể cần cập nhật lại file 1_fetch_data.py.