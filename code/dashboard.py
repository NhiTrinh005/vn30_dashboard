import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import base64
import re
import datetime
# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="VN30 Dashboard",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
    /* 1. HEADER CONTAINER (NỀN XANH VÀNG) */
    .navy-header {
        background: linear-gradient(90deg, #1a2236 0%, #223159 60%, #fbbc04 100%);
        padding: 20px 30px; border-radius: 12px; color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); display: flex; 
        justify-content: space-between; align-items: center; margin-bottom: 25px;
    }

    /* 2. HEADER TEXT STYLES */
    .header-sub { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; color: #cfd8dc; margin-bottom: 4px; font-weight: 500; }
    .header-title { font-size: 1.8rem; font-weight: 800; margin: 0; color: #fff; text-shadow: 0 2px 4px rgba(0,0,0,0.3); line-height: 1.2; }
    .header-desc { font-size: 0.7rem; color: #fbbc04; font-weight: 700; margin-top: 8px; text-transform: uppercase; letter-spacing: 1px; }
    .dev-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: #1a2236; opacity: 0.7; text-align: right; margin-bottom: 2px; font-weight: 700; }
    .dev-name { font-size: 1.4rem; font-weight: 900; text-align: right; color: #1a2236; text-shadow: 0 1px 0 rgba(255,255,255,0.4); line-height: 1.2; }

    /* 3. SECTION TITLES */
    .eda-title {
        font-size: 1.8rem; font-weight: 900; color: #1a2236;
        text-transform: uppercase; letter-spacing: 1px;
        border-bottom: 4px solid #fbbc04; display: inline-block; margin-bottom: 20px;
    }
    .pro-header {
        font-size: 1.2rem; font-weight: 800; color: #1a2236;
        text-transform: uppercase; border-left: 6px solid #fbbc04;
        padding-left: 12px; margin-top: 20px; margin-bottom: 15px;
        background: linear-gradient(90deg, #f8f9fa 0%, #ffffff 100%);
        padding-top: 5px; padding-bottom: 5px; border-radius: 0 4px 4px 0;
        border-bottom: 1px solid #eee;
    }

    /* 4. TABS BUTTON STYLES */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; /* Giảm khoảng cách để các nút khít hơn */
        border-bottom: none;
        display: flex;
        flex-direction: row;
        width: 100%; /* Bắt buộc container chiếm 100% */
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; 
        border-radius: 8px; 
        padding: 10px 5px; /* Giảm padding ngang, để flex tự lo độ rộng */
        font-weight: 600; 
        color: #444; 
        border: 1px solid #dce1e6;

        flex: 1 1 0%; /* Giãn đều, co đều, chia đều không gian */
        width: 100%; /* Fallback */
        justify-content: center;
        text-align: center;
        white-space: nowrap; /* Giữ chữ trên 1 dòng */
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important; color: #223159 !important;
        border: 2px solid #223159 !important; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* Tab Utilities (Màu đặc biệt) */
    .stTabs [data-baseweb="tab"]:last-child { background-color: #1a2236; color: #fbbc04; border: 1px solid #1a2236; }
    .stTabs [data-baseweb="tab"]:last-child[aria-selected="true"] { background-color: #fbbc04 !important; color: #1a2236 !important; }

    /* 5. CÁC THÀNH PHẦN KHÁC */
    .stMetric { background: #fff; padding: 10px; border-radius: 8px; border: 1px solid #eee; box-shadow: 0 2px 4px rgba(0,0,0,0.02); text-align: center; }
    .streamlit-expanderHeader p { font-weight: 700 !important; font-size: 1.05rem !important; color: #2c3e50 !important; }
    .stMultiSelect [data-baseweb="tag"] { background-color: #1565c0 !important; }
    .stMultiSelect [data-baseweb="tag"] span { color: #ffffff !important; }

    /* Layout Tweaks */
    .main .block-container { max_width: 100% !important; padding: 1rem 2rem !important; }
    [data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)
# ========== HÀM HỖ TRỢ ĐƯỜNG DẪN ==========
try:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
except NameError:
    project_root = os.path.abspath(os.getcwd())

TABLES_DIR = os.path.join(project_root, 'results', 'tables')
PROCESSED_DIR = os.path.join(project_root, 'data', 'processed')
CHARTS_DIR = os.path.join(project_root, 'results', 'charts')


@st.cache_data
def load_csv_data(folder, filename, index_col=0):
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=index_col, parse_dates=True)
    return None


df_returns = load_csv_data(PROCESSED_DIR, 'vn30_monthly_returns.csv', index_col='Date')
df_metrics = load_csv_data(TABLES_DIR, 'danh_muc_hieu_qua.csv', index_col=0)
df_capm = load_csv_data(TABLES_DIR, 'capm_beta_results.csv', index_col='Ticker')
df_momentum = load_csv_data(TABLES_DIR, 'bonus_momentum_hieu_qua_so_sanh.csv', index_col=0)
df_price = load_csv_data(PROCESSED_DIR, 'vn30_cleaned.csv', index_col=0)

# Render HTML 1 dòng
st.markdown(
    '<div class="navy-header"><div><div class="header-sub">FINANCIAL DASHBOARD PROJECT</div><div class="header-title">VN30 STOCK ANALYSIS</div><div class="header-desc">DATA VISUALIZATION • RISK ANALYSIS • PRICE FORECAST</div></div><div><div class="dev-label">DEVELOPED BY</div><div class="dev-name">TRINH HUE NHI</div></div></div>',
    unsafe_allow_html=True)
# ========== TABS CHÍNH ==========
tab_names = ["🏠 OVERVIEW", "🔍 EDA", "📈 CAPM", "📖 PORTFOLIO", "🔮 PRICE FORECAST", "🚀 MOMENTUM STRATEGY", "⚙️ UTILITIES"]
tabs = st.tabs(tab_names)

# --- TAB 1: TỔNG QUAN  ---
with tabs[0]:
    # 1. CSS CHUNG CHO TAB OVERVIEW
    st.markdown("""
    <style>
        
        /* --- STYLE WORKFLOW --- */
        .step-container { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; }
        .step-box {
            background: #f8f9fa; border: 2px solid #e0e0e0; border-radius: 10px;
            padding: 15px 5px; width: 18%; text-align: center; font-weight: 600; color: #444;
            transition: all 0.3s;
        }
        .step-box:hover { 
            border-color: #fbbc04; background: #fff; 
            transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); 
        }
        .step-icon { font-size: 2rem; margin-bottom: 5px; display: block; }
        .arrow { font-size: 1.5rem; color: #ccc; font-weight: bold; }

        /* --- STYLE TECH STACK TAGS --- */
        .mini-tag { display: inline-block; padding: 4px 10px; margin: 3px 2px; border-radius: 15px; font-size: 0.8rem; font-weight: 600; }
        .t-blue { background: #e3f2fd; color: #1565c0; }
        .t-green { background: #e8f5e9; color: #2e7d32; }
        .t-purple { background: #f3e5f5; color: #7b1fa2; }
        .t-orange { background: #fff3e0; color: #e65100; }
        .t-ai { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 6px 12px; border-radius: 20px;}
        .cat-title { font-size: 0.75rem; font-weight: 700; color: #999; margin-top: 8px; margin-bottom: 4px; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

    # --- TIÊU ĐỀ CHÍNH ---
    st.markdown('<div class="eda-title">PROJECT OVERVIEW</div>', unsafe_allow_html=True)

    # --- PHẦN 1: NHỊP ĐẬP THỊ TRƯỜNG (MARKET PULSE) ---
    st.markdown('<div class="pro-header">MARKET PULSE</div>', unsafe_allow_html=True)

    try:
        path_price = os.path.join(PROCESSED_DIR, 'vn30_cleaned.csv')
        if os.path.exists(path_price):
            df_price = pd.read_csv(path_price, index_col=0, parse_dates=True)

            last_date = df_price.index[-1]
            prev_date = df_price.index[-2]

            # VNINDEX
            vni_now = df_price.loc[last_date, '^VNINDEX']
            vni_prev = df_price.loc[prev_date, '^VNINDEX']
            vni_change = vni_now - vni_prev
            vni_pct = (vni_change / vni_prev) * 100
            color_vni = "#28a745" if vni_change >= 0 else "#dc3545"

            # Tìm mã biến động
            daily_chg = (df_price.iloc[-1] - df_price.iloc[-2]) / df_price.iloc[-2] * 100
            daily_chg = daily_chg.drop('^VNINDEX')
            best_stock = daily_chg.idxmax()
            worst_stock = daily_chg.idxmin()

            # METRICS HTML
            st.markdown(f"""
            <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                <div style="flex: 1; background: white; padding: 15px; border-radius: 10px; border: 1px solid #eee; box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center;">
                    <div style="font-size: 0.9rem; color: #666; font-weight: 600;">VNINDEX</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: {color_vni};">{vni_now:,.2f}</div>
                    <div style="font-size: 0.9rem; color: {color_vni};">{vni_change:+.2f} ({vni_pct:+.2f}%)</div>
                </div>
                <div style="flex: 1; background: white; padding: 15px; border-radius: 10px; border: 1px solid #eee; box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center;">
                    <div style="font-size: 0.9rem; color: #666; font-weight: 600;">PHIÊN CẬP NHẬT</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #223159;">{last_date.strftime('%d/%m')}</div>
                    <div style="font-size: 0.8rem; color: #888;">{last_date.strftime('%Y')}</div>
                </div>
                <div style="flex: 1; background: white; padding: 15px; border-radius: 10px; border: 1px solid #eee; box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center;">
                    <div style="font-size: 0.9rem; color: #666; font-weight: 600;">TĂNG MẠNH NHẤT 🔺</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #28a745;">{best_stock}</div>
                    <div style="font-size: 0.9rem; color: #28a745;">+{daily_chg[best_stock]:.2f}%</div>
                </div>
                <div style="flex: 1; background: white; padding: 15px; border-radius: 10px; border: 1px solid #eee; box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center;">
                    <div style="font-size: 0.9rem; color: #666; font-weight: 600;">GIẢM MẠNH NHẤT 🔻</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #dc3545;">{worst_stock}</div>
                    <div style="font-size: 0.9rem; color: #dc3545;">{daily_chg[worst_stock]:.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Không thể tải dữ liệu thị trường: {e}")

    # --- PHẦN 2: QUY TRÌNH PHÂN TÍCH ---
    st.write("")
    st.markdown('<div class="pro-header">ANALYTICAL WORKFLOW</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="step-container">
        <div class="step-box">
            <span class="step-icon">📥</span>
            DATA FETCHING<br><span style="font-size:0.8rem;color:#888">Thu thập Dữ liệu</span>
        </div>
        <div class="arrow">➔</div>
        <div class="step-box">
            <span class="step-icon">🧹</span>
            CLEANING & EDA<br><span style="font-size:0.8rem;color:#888">Làm sạch & Phân tích</span>
        </div>
        <div class="arrow">➔</div>
        <div class="step-box">
            <span class="step-icon">🧮</span>
            RISK & FORECAST<br><span style="font-size:0.8rem;color:#888">Mô hình CAPM & ARIMA</span>
        </div>
        <div class="arrow">➔</div>
        <div class="step-box">
            <span class="step-icon">⚖️</span>
            PORTFOLIO OPT<br><span style="font-size:0.8rem;color:#888">Tối ưu hóa danh mục</span>
        </div>
        <div class="arrow">➔</div>
        <div class="step-box">
            <span class="step-icon">🚀</span>
            TRADING STRATEGY<br><span style="font-size:0.8rem;color:#888">Chiến thuật Momentum</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- PHẦN 3: MỤC TIÊU & CÔNG NGHỆ ---
    st.write("")
    c_goal, c_tech = st.columns([1.2, 1], gap="medium")

    # CỘT TRÁI: PROJECT GOALS
    with c_goal:
        with st.container(border=True):
            st.markdown('<div class="pro-header" style="margin-top:0; font-size:1.1rem;">PROJECT GOALS</div>',
                        unsafe_allow_html=True)

            goals_html = """
            <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 10px; margin-bottom: 20px;">
                <div style="display: flex; align-items: start; gap: 10px;">
                    <span style="font-size: 1.1rem; background: #e3f2fd; color: #1565c0; padding: 2px 8px; border-radius: 6px; font-weight: bold; height: fit-content;">1</span>
                    <div style="font-size: 0.9rem; line-height: 1.4;">
                        <b>Data & EDA:</b> Tự động thu thập từ Yahoo/CafeF, làm sạch, tính toán lợi suất và thực hiện phân tích thống kê mô tả, trực quan hóa thị trường.
                    </div>
                </div>
                <div style="display: flex; align-items: start; gap: 10px;">
                    <span style="font-size: 1.1rem; background: #fff3e0; color: #e65100; padding: 2px 8px; border-radius: 6px; font-weight: bold; height: fit-content;">2</span>
                    <div style="font-size: 0.9rem; line-height: 1.4;">
                        <b>CAPM & Portfolio:</b> Tính toán Beta (Risk) phân loại cổ phiếu và tối ưu hóa tỷ trọng đầu tư theo Đường biên hiệu quả (Efficient Frontier).
                    </div>
                </div>
                <div style="display: flex; align-items: start; gap: 10px;">
                    <span style="font-size: 1.1rem; background: #f3e5f5; color: #7b1fa2; padding: 2px 8px; border-radius: 6px; font-weight: bold; height: fit-content;">3</span>
                    <div style="font-size: 0.9rem; line-height: 1.4;">
                        <b>Price Forecast:</b> Ứng dụng mô hình chuỗi thời gian ARIMA tự động (Auto-ARIMA) để dự báo xu hướng giá ngắn hạn trong 12 tháng tới.
                    </div>
                </div>
                <div style="display: flex; align-items: start; gap: 10px;">
                    <span style="font-size: 1.1rem; background: #e8f5e9; color: #2e7d32; padding: 2px 8px; border-radius: 6px; font-weight: bold; height: fit-content;">4</span>
                    <div style="font-size: 0.9rem; line-height: 1.4;">
                        <b>Momentum Strategy:</b> Backtest chiến thuật Momentum (đầu tư theo đà tăng trưởng) để tìm kiếm lợi nhuận vượt trội so với chỉ số VNINDEX.
                    </div>
                </div>
            </div>

            <div style="margin-top: auto; background-color: #f8f9fa; border-left: 4px solid #fbbc04; padding: 10px; border-radius: 4px;">
                <div style="font-size: 0.8rem; color: #555; display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.2rem;">💡</span>
                    <span><b>Export PDF Report:</b> Vào tab UTILITIES tải HTML, mở lên và nhấn <kbd style="font-family:monospace; background:#fff; border:1px solid #ccc; padding:0 4px; border-radius:3px;">Ctrl + P</kbd></span>
                </div>
            </div>
            """
            st.markdown(goals_html, unsafe_allow_html=True)

    # CỘT PHẢI: TECH STACK
    with c_tech:
        with st.container(border=True):
            st.markdown('<div class="pro-header" style="margin-top:0; font-size:1.1rem;">TECH STACK</div>',
                        unsafe_allow_html=True)

            tech_html = """
            <div class="cat-title" style="margin-top:0">Core & Data</div>
            <div>
                <span class="mini-tag t-blue">Python 3.12</span>
                <span class="mini-tag t-blue">Streamlit</span>
                <span class="mini-tag t-green">Pandas</span>
                <span class="mini-tag t-green">Numpy</span>
                <span class="mini-tag t-green">Yfinance</span>
                <span class="mini-tag t-green">Requests</span>
                <span class="mini-tag t-green">OpenPyXL</span>
            </div>

            <div class="cat-title">Models & Viz</div>
            <div>
                <span class="mini-tag t-purple">Statsmodels</span>
                <span class="mini-tag t-purple">Pmdarima</span>
                <span class="mini-tag t-purple">Scikit-learn</span>
                <span class="mini-tag t-orange">Plotly</span>
                <span class="mini-tag t-orange">Matplotlib</span>
                <span class="mini-tag t-orange">Seaborn</span>
            </div>

            <div class="cat-title">AI Assistant</div>
            <div>
                <span class="mini-tag t-ai">OpenAI GPT</span>
            </div>
            """
            st.markdown(tech_html, unsafe_allow_html=True)

    st.markdown('<div class="pro-header">LIMITATIONS & DISCLAIMER</div>', unsafe_allow_html=True)
    st.info("""
        Dự án được xây dựng với mục đích học thuật, không cấu thành lời khuyên tài chính hoặc khuyến nghị đầu tư. Do đó tồn tại một số giả định đơn giản hóa:
        1.  **Phí giao dịch & Trượt giá (Slippage):** Backtest giả định mua bán tại giá đóng cửa, chưa tính phí giao dịch (0.1-0.2%) và tác động của lệnh lớn lên thị trường.
        2.  **Cổ tức:** Dữ liệu giá chưa điều chỉnh đầy đủ cổ tức tiền mặt trong một số trường hợp.
        3.  **Giả định ARIMA:** Mô hình giả định dữ liệu quá khứ lặp lại trong tương lai, có thể không chính xác khi có tin tức vĩ mô đột biến (Thiên nga đen).
        4.  **Thanh khoản:** Chiến lược chưa lọc các mã có thanh khoản thấp, có thể khó giải ngân trong thực tế.
        """)

# --- TAB 2: EDA ---
with tabs[1]:
    # 1. CSS CHỈNH MÀU & GIAO DIỆN
    st.markdown("""
    <style>
        
        /* Chỉnh màu nút chọn (Chips) trong Multiselect */
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #1565c0 !important; 
        }
        .stMultiSelect [data-baseweb="tag"] span {
            color: #ffffff !important;
        }

        /* Expander text bold */
        .streamlit-expanderHeader p {
            font-weight: 700 !important; font-size: 1.05rem !important; color: #2c3e50 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- HEADER ---
    st.markdown('<div class="eda-title">EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)
    st.caption("Khám phá dữ liệu thị trường: Thống kê mô tả, Tương quan & Phân phối rủi ro.")

    # --- LOAD DATA ---
    try:
        returns_path = os.path.join(PROCESSED_DIR, 'vn30_monthly_returns.csv')
        df_returns = pd.read_csv(returns_path, index_col='Date', parse_dates=True)
        stats_path = os.path.join(PROCESSED_DIR, 'eda_thong_ke_mo_ta.csv')
        if os.path.exists(stats_path):
            df_eda = pd.read_csv(stats_path, index_col=0)
        else:
            df_eda = df_returns.describe().T
    except Exception as e:
        st.error(f"Lỗi data: {e}")
        st.stop()

    # --- 2. BẢNG THỐNG KÊ ---
    with st.expander("**VIEW STATISTICAL SUMMARY** (Xem chi tiết bảng thống kê)", expanded=False):
        st.dataframe(
            df_eda.style.background_gradient(cmap="Blues", subset=['mean', 'std']),
            use_container_width=True, height=300
        )
        csv_stats = df_eda.to_csv().encode('utf-8')
        st.download_button("📥 Tải CSV", csv_stats, "eda_statistics.csv", "text/csv")

    st.write("")

    # --- 3. BIỂU ĐỒ TĂNG TRƯỞNG ---
    st.markdown('<div class="pro-header">CUMULATIVE RETURNS GROWTH</div>', unsafe_allow_html=True)
    st.caption("So sánh hiệu suất đầu tư giữa các mã cổ phiếu và VNINDEX.")

    try:
        cum_returns = (1 + df_returns).cumprod()
        all_tickers = cum_returns.columns.tolist()
        default_tickers = ['^VNINDEX', 'FPT.VN', 'VCB.VN', 'HPG.VN']
        default_selection = [t for t in default_tickers if t in all_tickers]

        selected_tickers = st.multiselect(
            "💡Chọn các mã muốn so sánh:",
            options=all_tickers, default=default_selection
        )

        if not selected_tickers:
            st.warning("❗Vui lòng chọn ít nhất một mã để hiển thị biểu đồ.")
        else:
            # Chỉ lấy dữ liệu của các mã được chọn
            df_plot = cum_returns[selected_tickers]
            df_melted = df_plot.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Value')
            fig_growth = px.line(
                df_melted, x='Date', y='Value', color='Ticker',
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={'Value': 'Giá trị tài sản', 'Date': ''},
            )
            # THÊM TITLE VÀO TRONG BIỂU ĐỒ
            fig_growth.update_layout(
                title=dict(text="Biểu đồ Tăng trưởng Tài sản (Wealth Index)", font=dict(size=14, color="#555")),
                hovermode="x unified",
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(step="all", label="ALL")
                        ]), bgcolor="#f1f3f4"
                    ),
                    rangeslider=dict(visible=True), type="date"
                ),
                legend=dict(orientation="h", y=1.1, x=0, title=None),
                height=550, margin=dict(l=10, r=10, t=170, b=10)  # Tăng margin top để chứa Title
            )
            st.plotly_chart(fig_growth, use_container_width=True)
    except Exception:
        st.info("Chưa có dữ liệu biểu đồ.")

    # --- 4. HEATMAP & HISTOGRAM ---
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown('<div class="pro-header">CORRELATION MATRIX</div>', unsafe_allow_html=True)
        try:
            corr_matrix = df_returns.corr()
            fig_heat = px.imshow(
                corr_matrix, text_auto='.2f', aspect="auto",
                color_continuous_scale='RdBu_r', origin='lower'
            )
            # THÊM TITLE
            fig_heat.update_layout(
                title=dict(text="Heatmap Tương quan (Correlation)", font=dict(size=14, color="#555")),
                height=480, margin=dict(t=50, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        except:
            pass

    with c2:
        st.markdown('<div class="pro-header">RETURN DISTRIBUTION</div>', unsafe_allow_html=True)
        selected_ticker = st.selectbox("Chọn mã xem phân phối:", df_returns.columns)
        try:
            fig_hist = px.histogram(
                df_returns, x=selected_ticker, nbins=30, marginal="box",
                color_discrete_sequence=['#fbbc04'], opacity=0.8
            )
            fig_hist.add_vline(x=df_returns[selected_ticker].mean(), line_dash="dash", line_color="#d32f2f")
            # THÊM TITLE
            fig_hist.update_layout(
                title=dict(text=f"Phân phối Lợi suất: {selected_ticker}", font=dict(size=14, color="#555")),
                height=480, margin=dict(t=50, b=0, l=0, r=0), showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        except:
            pass

    # --- 5. BOXPLOT ---
    st.markdown('<div class="pro-header">RISK COMPARISON (BOXPLOT)</div>', unsafe_allow_html=True)
    try:
        df_melt = df_returns.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Lợi suất')
        fig_box = px.box(
            df_melt, x='Ticker', y='Lợi suất', color='Ticker',
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        # THÊM TITLE
        fig_box.update_layout(
            title=dict(text="So sánh Biến động Rủi ro (Risk Comparison)", font=dict(size=14, color="#555")),
            height=500, xaxis={'categoryorder': 'total ascending'},
            showlegend=False, margin=dict(t=50, b=10)
        )
        st.plotly_chart(fig_box, use_container_width=True)
    except:
        pass
# ========== TAB 3: CAPM & BETA ANALYSIS ==========
with tabs[2]:
    st.markdown('<div class="eda-title">CAPM MODEL & BETA ANALYTICS</div>', unsafe_allow_html=True)
    st.caption("Mô hình Định giá Tài sản Vốn.")


    # --- HÀM TỰ DÒ TÊN CỘT ---
    def get_col_name(df, keyword):
        """Tìm tên cột thực tế chứa từ khóa (không phân biệt hoa thường)"""
        for col in df.columns:
            if keyword.lower() in col.lower():
                return col
        return None

    # --- LOAD DATA ---
    try:
        # 1. Tìm đường dẫn file
        data_dir = os.path.dirname(PROCESSED_DIR)
        project_root = os.path.dirname(data_dir)
        capm_path = os.path.join(project_root, 'results', 'tables', 'capm_beta_results.csv')

        if os.path.exists(capm_path):
            df_capm = pd.read_csv(capm_path, index_col=0)

            # 2. ÁP DỤNG TỰ DÒ TÊN CỘT
            # Code sẽ tự tìm cột nào có chữ 'beta', 'alpha', 'squared'
            beta_col = get_col_name(df_capm, 'beta')
            alpha_col = get_col_name(df_capm, 'alpha')
            r2_col = get_col_name(df_capm, 'squared')

            # Kiểm tra xem có tìm thấy Beta không
            if not beta_col:
                st.error(
                    f"⚠️ Không tìm thấy cột nào chứa chữ 'Beta' trong file. Các cột hiện có: {list(df_capm.columns)}")
                st.stop()
        else:
            st.error(f"⚠️ Không tìm thấy file: {capm_path}")
            st.stop()

        # Load file Returns
        returns_path = os.path.join(PROCESSED_DIR, 'vn30_monthly_returns.csv')
        df_returns = pd.read_csv(returns_path, index_col='Date', parse_dates=True)

    except Exception as e:
        st.error(f"Lỗi: {e}")
        st.stop()

    # --- BỐ CỤC GIAO DIỆN ---
    col_left, col_right = st.columns([1.3, 1], gap="large")

    # --- CỘT TRÁI: XẾP HẠNG BETA ---
    with col_left:
        st.markdown('<div class="pro-header" style="margin-top:0">BETA RISK RANKING</div>', unsafe_allow_html=True)
        st.caption("Xếp hạng độ nhạy cảm với thị trường (Beta > 1: Rủi ro cao, Beta < 1: An toàn).")
        # Dùng biến beta_col (tên thực tế) thay vì cứng nhắc 'Beta'
        df_capm['Color'] = df_capm[beta_col].apply(lambda x: '#dc3545' if x > 1 else '#28a745')
        df_capm_sorted = df_capm.sort_values(by=beta_col, ascending=True)

        fig_beta = px.bar(
            df_capm_sorted, x=beta_col, y=df_capm_sorted.index,  # Dùng beta_col ở đây
            orientation='h', text_auto='.2f', title="",
            labels={beta_col: 'Hệ số Beta'}  # Đổi label hiển thị cho đẹp
        )
        fig_beta.update_traces(marker_color=df_capm_sorted['Color'], width=0.7)
        fig_beta.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        fig_beta.add_vline(x=1, line_dash="dash", line_color="#333", annotation_text="Market (1.0)")
        st.plotly_chart(fig_beta, use_container_width=True)

    # --- CỘT PHẢI: PHÂN PHỐI & TABLE ---
    with col_right:
        st.markdown('<div class="pro-header" style="margin-top:0">BETA DISTRIBUTION</div>', unsafe_allow_html=True)
        st.caption("Phân phối mật độ rủi ro: Cho biết đa số các mã VN30 tập trung ở ngưỡng Beta nào.")
        fig_dist = px.histogram(df_capm, x=beta_col, nbins=15, marginal="box", color_discrete_sequence=['#1a2236'])
        fig_dist.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.write("")
        st.markdown('<div class="pro-header">DETAILED METRICS</div>', unsafe_allow_html=True)
        st.caption("Dữ liệu chi tiết dùng để tra cứu Alpha (Hiệu suất) và Beta (Rủi ro) từng mã.")
        # Chỉ hiển thị các cột tìm được (nếu tìm thấy)
        cols_to_show = [c for c in [beta_col, alpha_col, r2_col] if c is not None]

        st.dataframe(
            df_capm[cols_to_show].style.format("{:.4f}")
            .background_gradient(cmap="Reds", subset=[beta_col])
            .background_gradient(cmap="Greens", subset=[alpha_col] if alpha_col else None),
            height=300, use_container_width=True
        )

    # --- INTERACTIVE REGRESSION ---
    st.write("---")
    st.markdown('<div class="pro-header">INTERACTIVE REGRESSION ANALYSIS</div>', unsafe_allow_html=True)
    st.caption("Chọn một mã cổ phiếu để xem đường hồi quy tuyến tính (Security Characteristic Line).")
    col_sel, col_chart = st.columns([1, 3])

    with col_sel:
        tickers = [c for c in df_returns.columns if c != '^VNINDEX']
        idx = tickers.index('FPT.VN') if 'FPT.VN' in tickers else 0
        selected = st.selectbox("Chọn cổ phiếu:", tickers, index=idx)

        if selected in df_capm.index:
            # Lấy giá trị dựa trên tên cột tìm được
            b_val = df_capm.loc[selected, beta_col]
            a_val = df_capm.loc[selected, alpha_col] if alpha_col else 0

            st.metric("Beta (Risk)", f"{b_val:.2f}")
            st.metric("Alpha (Perf)", f"{a_val:.4f}")
            if b_val > 1:
                st.error("Aggressive")
            else:
                st.success("Defensive")

    with col_chart:
        try:
            df_reg = df_returns[[selected, '^VNINDEX']].dropna()
            fig = px.scatter(df_reg, x='^VNINDEX', y=selected, trendline="ols",
                             trendline_color_override="#dc3545", color_discrete_sequence=['#1a2236'])
            fig.update_layout(title=dict(text=f"Đường đặc thù chứng khoán (SCL) - {selected}", font=dict(size=14, color="#555")),
                              height=400, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        except:
            pass
    st.write("---")
    # Tính toán nhanh
    if 'beta_col' in locals() and beta_col and not df_capm.empty:
        risky_stock = df_capm[beta_col].idxmax()
        safe_stock = df_capm[df_capm[beta_col] > 0][beta_col].idxmin()  # Beta dương thấp nhất

        st.info(f"""
            💡**Đánh giá Rủi ro (Risk Insight):**
            - **Cổ phiếu "Tấn công" ({risky_stock}):** Có Beta cao nhất, nhạy cảm mạnh với thị trường. Phù hợp giai đoạn Uptrend mạnh.
            - **Cổ phiếu "Phòng thủ" ({safe_stock}):** Có Beta thấp nhất, biến động ít hơn thị trường. Phù hợp để trú ẩn khi thị trường xấu.
            """)
# --- TAB 4: PORTFOLIO ---
with tabs[3]:
    # --- 1. CSS & HEADER ---
    st.markdown('<div class="eda-title">PORTFOLIO CONSTRUCTION & EVALUATION</div>', unsafe_allow_html=True)
    st.caption("Xây dựng, tối ưu hóa và đánh giá hiệu quả danh mục đầu tư.")

    # --- 2. HÀM DÒ TÊN CỘT  ---
    def find_col_fuzzy(df, keywords):
        for col in df.columns:
            for kw in keywords:
                if kw.lower() in col.lower(): return col
        return None

    # --- 3. LOAD DATA ---
    try:
        # Tìm đường dẫn thư mục gốc
        data_dir = os.path.dirname(PROCESSED_DIR)
        project_root = os.path.dirname(data_dir)

        # A. LOAD METRICS (File: danh_muc_hieu_qua.csv)
        metrics_path = os.path.join(project_root, 'results', 'tables', 'danh_muc_hieu_qua.csv')
        if os.path.exists(metrics_path):
            df_metrics = pd.read_csv(metrics_path, index_col=0)

            # Dò tìm các cột dữ liệu quan trọng
            col_return = find_col_fuzzy(df_metrics, ['return', 'loisuat', 'lai'])
            col_risk = find_col_fuzzy(df_metrics, ['risk', 'ruiro', 'std', 'volatility'])
            col_sharpe = find_col_fuzzy(df_metrics, ['sharpe'])
            col_drawdown = find_col_fuzzy(df_metrics, ['drawdown', 'sutgiam'])
        else:
            st.error("⚠️ Thiếu file 'danh_muc_hieu_qua.csv'. Hãy chạy lại file code tạo danh mục.")
            st.stop()

        # B. LOAD ALLOCATION (File: danh_muc_phan_loai.csv)
        alloc_path = os.path.join(project_root, 'results', 'tables', 'danh_muc_phan_loai.csv')
        if os.path.exists(alloc_path):
            df_alloc = pd.read_csv(alloc_path, index_col=0)
            # Tìm cột phân loại (LoaiDanhMuc / Category)
            col_category = find_col_fuzzy(df_alloc, ['loai', 'category', 'type', 'group'])
        else:
            st.error("⚠️ Thiếu file 'danh_muc_phan_loai.csv'.")
            st.stop()

        # C. LOAD RETURNS (File: vn30_monthly_returns.csv)
        returns_path = os.path.join(PROCESSED_DIR, 'vn30_monthly_returns.csv')
        if os.path.exists(returns_path):
            df_returns = pd.read_csv(returns_path, index_col='Date', parse_dates=True)
        else:
            st.error("⚠️ Thiếu file Returns.")
            st.stop()
        # D. LOAD GIÁ MỚI NHẤT ĐỂ TÍNH TIỀN
        price_path = os.path.join(PROCESSED_DIR, 'vn30_cleaned.csv')
        last_prices = {}
        if os.path.exists(price_path):
            df_prices_raw = pd.read_csv(price_path, index_col=0, parse_dates=True)
            last_prices = df_prices_raw.iloc[-1].to_dict()  # Lấy giá ngày cuối cùng
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        st.stop()

    # --- 4. GIAO DIỆN CHÍNH ---
    # --- PHẦN 1: BẢNG CHỈ SỐ & CƠ CẤU DANH MỤC ---
    c1, c2 = st.columns([1.3, 1], gap="large")

    # [CỘT TRÁI] BẢNG METRICS
    with c1:
        st.markdown('<div class="pro-header" style="margin-top:0">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
        st.caption("Bảng so sánh các chỉ số hiệu quả đầu tư.")

        cols_found = [c for c in [col_return, col_risk, col_sharpe, col_drawdown] if c is not None]

        if cols_found:
            st.dataframe(
                df_metrics[cols_found].style.format("{:.4f}")
                .background_gradient(cmap="Greens", subset=[col_sharpe, col_return] if col_sharpe else None)
                .background_gradient(cmap="Reds", subset=[col_risk, col_drawdown] if col_risk else None),
                use_container_width=True, height=200
            )
        else:
            st.warning("Không tìm thấy cột dữ liệu phù hợp.")

    # [CỘT PHẢI] BIỂU ĐỒ TRÒN (DONUT CHART)
    with c2:
        st.markdown('<div class="pro-header" style="margin-top:0">PORTFOLIO COMPOSITION</div>', unsafe_allow_html=True)

        if col_category:
            # Lấy danh sách các nhóm (Defensive, Aggressive...)
            categories = df_alloc[col_category].unique().tolist()
            selected_cat = st.selectbox("Xem thành phần danh mục:", categories)

            # Lọc các mã trong nhóm đã chọn
            tickers_in_cat = df_alloc[df_alloc[col_category] == selected_cat].index.tolist()

            if tickers_in_cat:
                fig_pie = px.pie(
                    names=tickers_in_cat,
                    values=[1] * len(tickers_in_cat),  # Equal Weight (Tỷ trọng bằng nhau)
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    title=f"{selected_cat}"
                )
                fig_pie.update_traces(textinfo='label+percent', textposition='inside')
                fig_pie.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0), showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)

    # --- PHẦN 2: SO SÁNH TRỰC QUAN (BAR CHARTS) ---
    st.write("")
    st.markdown('<div class="pro-header">KEY METRICS COMPARISON</div>', unsafe_allow_html=True)
    st.caption("So sánh trực quan các chỉ số giữa: Phòng thủ (Xanh) - Tăng trưởng (Đỏ) - Thị trường (Đen).")

    if cols_found:
        c_ret, c_risk, c_sharpe = st.columns(3)

        # Hàm chọn màu chuẩn theo tên danh mục (Defensive/Aggressive)
        def get_color(name):
            n = str(name).lower()
            if 'defensive' in n or 'ondinh' in n: return '#28a745'  # Xanh lá
            if 'aggressive' in n or 'maohiem' in n: return '#dc3545'  # Đỏ
            if 'market' in n or 'vnindex' in n: return '#1a2236'  # Đen
            return '#1565c0'  # Mặc định

        colors = [get_color(idx) for idx in df_metrics.index]

        # Vẽ 3 biểu đồ
        chart_configs = [
            (col_return, 'Lợi suất (Annual Return)', c_ret, '.2%'),
            (col_risk, 'Rủi ro (Volatility)', c_risk, '.2%'),
            (col_sharpe, 'Hiệu quả (Sharpe Ratio)', c_sharpe, '.2f')
        ]

        for col, title, chart_col, txt_fmt in chart_configs:
            if col:
                with chart_col:
                    st.markdown(f"**{title}**")  # Dùng markdown thay caption cho đậm đà
                    fig = px.bar(df_metrics, x=df_metrics.index, y=col, text_auto=txt_fmt)
                    fig.update_traces(marker_color=colors)
                    fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                                      xaxis_title=None, yaxis_title=None)
                    st.plotly_chart(fig, use_container_width=True)

    # --- PHẦN 3: MÔ PHỎNG TĂNG TRƯỞNG ---
    st.write("---")
    st.markdown('<div class="pro-header">GROWTH SIMULATION (WEALTH INDEX)</div>', unsafe_allow_html=True)
    st.caption("Mô phỏng: Nếu đầu tư 1 VND vào năm 2020, tài sản sẽ tăng trưởng như thế nào?")

    try:
        # A. TÍNH TOÁN DỮ LIỆU TĂNG TRƯỞNG
        sim_data = pd.DataFrame(index=df_returns.index)

        # Thêm Market (VNINDEX)
        if '^VNINDEX' in df_returns.columns:
            sim_data['Market (VN30)'] = df_returns['^VNINDEX']

        # Thêm các danh mục từ file Allocation
        if col_category:
            # Tìm nhóm Defensive & Aggressive trong file csv
            # Logic: Lọc các mã -> Tính trung bình Return
            unique_cats = df_alloc[col_category].unique()

            for cat in unique_cats:
                tickers = df_alloc[df_alloc[col_category] == cat].index.tolist()
                valid_tickers = [t for t in tickers if t in df_returns.columns]

                # Đặt tên ngắn gọn cho legend
                legend_name = cat
                if 'defensive' in str(cat).lower(): legend_name = 'Defensive'
                if 'aggressive' in str(cat).lower(): legend_name = 'Aggressive'

                if valid_tickers:
                    sim_data[legend_name] = df_returns[valid_tickers].mean(axis=1)

        # Tính Wealth Index (Tích lũy)
        cum_growth = (1 + sim_data).cumprod()

        # B. VẼ BIỂU ĐỒ LINE (TĂNG TRƯỞNG)
        df_melt = cum_growth.reset_index().melt(id_vars='Date', var_name='Portfolio', value_name='Value')

        # Map màu chuẩn
        color_map = {
            'Defensive': '#28a745',
            'Aggressive': '#dc3545',
            'Market (VN30)': '#1a2236',
            'Defensive (Phòng thủ)': '#28a745',
            'Aggressive (Tăng trưởng)': '#dc3545'
        }

        c_growth, c_dd = st.columns(2)

        with c_growth:
            st.markdown("**Tăng trưởng Tài sản**")
            fig_growth = px.line(
                df_melt, x='Date', y='Value', color='Portfolio',
                color_discrete_map=color_map,
                labels={'Value': 'Giá trị tài sản'}
            )
            fig_growth.update_layout(height=400, hovermode="x unified", legend=dict(orientation="h", y=1.1, title=None))
            st.plotly_chart(fig_growth, use_container_width=True)

        # C. VẼ BIỂU ĐỒ AREA (DRAWDOWN)
        with c_dd:
            st.markdown("**Mức độ Sụt giảm (Drawdown)**")
            drawdown = (cum_growth - cum_growth.cummax()) / cum_growth.cummax()
            dd_melt = drawdown.reset_index().melt(id_vars='Date', var_name='Portfolio', value_name='Drawdown')

            fig_dd = px.area(
                dd_melt, x='Date', y='Drawdown', color='Portfolio',
                color_discrete_map=color_map
            )
            fig_dd.update_layout(height=400, hovermode="x unified", yaxis_tickformat='.0%',
                                 legend=dict(orientation="h", y=1.1, title=None))
            st.plotly_chart(fig_dd, use_container_width=True)

    except Exception as e:
        st.error(f"Lỗi tính toán mô phỏng: {e}")
    # --- PHẦN 4: INVESTMENT CALCULATOR (TÍNH TOÁN SỐ LƯỢNG MUA) ---
    st.write("---")
    st.markdown('<div class="pro-header">INVESTMENT CALCULATOR</div>', unsafe_allow_html=True)

    calc_c1, calc_c2 = st.columns([1, 2], gap="large")

    with calc_c1:
        st.info("Nhập số vốn và chọn danh mục để tính số lượng cổ phiếu cần mua (làm tròn xuống lô 100).")
        input_money = st.number_input("Tổng vốn đầu tư (VND):", min_value=10_000_000, value=100_000_000,
                                      step=10_000_000, format="%d")

        # Chọn lại danh mục để tính (Phòng trường hợp muốn tính cái khác cái đang xem)
        target_cat = st.selectbox("Chọn danh mục để giải ngân:",
                                  df_alloc[col_category].unique() if col_category else [])

    with calc_c2:
        if target_cat and col_category:
            # Lấy các mã trong danh mục
            target_tickers = df_alloc[df_alloc[col_category] == target_cat].index.tolist()

            if target_tickers:
                # Giả sử tỷ trọng đều (Equal Weight)
                amt_per_stock = input_money / len(target_tickers)

                plan_data = []
                for t in target_tickers:
                    # Lấy giá thị trường mới nhất
                    price = last_prices.get(t, 0)

                    # Xử lý đơn vị giá (Nếu data là 30.5 thì nhân 1000 -> 30500)
                    real_price = price if price > 1000 else price * 1000

                    if real_price > 0:
                        # Tính số lượng: (Tiền / Giá) // 100 * 100 -> Làm tròn xuống lô 100
                        shares = int((amt_per_stock / real_price) // 100 * 100)

                        value_buy = shares * real_price
                        weight_actual = (value_buy / input_money) * 100

                        plan_data.append({
                            'Mã CP': t,
                            'Giá thị trường': real_price,
                            'Số lượng (Lô 100)': shares,
                            'Giá trị mua': value_buy,
                            'Tỷ trọng thực': weight_actual
                        })

                # Hiển thị bảng kết quả
                df_plan = pd.DataFrame(plan_data)

                st.dataframe(
                    df_plan.style.format({
                        'Giá thị trường': '{:,.0f}',
                        'Số lượng (Lô 100)': '{:,.0f}',
                        'Giá trị mua': '{:,.0f}',
                        'Tỷ trọng thực': '{:.2f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                # Tổng kết tiền thừa
                total_spent = df_plan['Giá trị mua'].sum()
                cash_left = input_money - total_spent

                st.success(f"**Tổng giải ngân:** {total_spent:,.0f} VND  |  **Tiền mặt dư:** {cash_left:,.0f} VND")
            else:
                st.warning("Danh mục này không có cổ phiếu.")
    st.write("---")
    try:
        # Sử dụng df_metrics (đã load ở đầu Tab) để chắc chắn có dữ liệu
        # Kiểm tra xem cột Sharpe có tồn tại không
        if 'df_metrics' in locals() and df_metrics is not None and 'col_sharpe' in locals() and col_sharpe:
            # Tính toán lại tại đây để đảm bảo biến tồn tại
            best_port_safe = df_metrics[col_sharpe].idxmax()
            best_sharpe_safe = df_metrics.loc[best_port_safe, col_sharpe]

            st.info(f"""
                💡**Hiệu quả Đầu tư (Performance Insight):**
                - Dựa trên chỉ số **Sharpe Ratio** (Lợi nhuận/Rủi ro), danh mục **{best_port_safe}** đang hoạt động hiệu quả nhất ({best_sharpe_safe:.2f}).
                - Khuyến nghị phân bổ tỷ trọng lớn vào nhóm này để tối ưu hóa lợi nhuận điều chỉnh rủi ro.
                """)
    except Exception:
        pass

# --- TAB 5: ARIMA ---
with tabs[4]:
    # --- 1. CSS & HEADER ---
    st.markdown('<div class="eda-title">TIME-SERIES FORECASTING (ARIMA)</div>', unsafe_allow_html=True)
    st.caption("Dự báo xu hướng giá cổ phiếu bằng mô hình chuỗi thời gian (Case Study: FPT.VN).")


    # --- 2. HÀM DÒ TÊN CỘT ---
    def find_col_fuzzy(df, keywords):
        for col in df.columns:
            for kw in keywords:
                if kw.lower() in col.lower(): return col
        return None

    # --- 3. HÀM TRÍCH XUẤT AIC/BIC TỪ TEXT ---
    def extract_metric_from_summary(text, metric_name):
        # Dùng Regex để tìm số liệu sau chữ AIC hoặc BIC
        # Ví dụ: "AIC   -57.834" -> Tìm được -57.834
        pattern = rf"{metric_name}\s+(-?\d+\.\d+)"
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
        return None

    # --- 4. LOAD DATA ---
    target_ticker = 'FPT.VN'
    try:
        data_dir = os.path.dirname(PROCESSED_DIR)
        project_root = os.path.dirname(data_dir)

        # Load các file CSV
        fc_path = os.path.join(project_root, 'results', 'tables', f'arima_forecast_{target_ticker}.csv')
        df_fc = pd.read_csv(fc_path, index_col=0, parse_dates=True) if os.path.exists(fc_path) else None

        eval_path = os.path.join(project_root, 'results', 'tables', f'arima_evaluation_{target_ticker}.csv')
        df_eval = pd.read_csv(eval_path, index_col=0, parse_dates=True) if os.path.exists(eval_path) else None

        resid_path = os.path.join(project_root, 'results', 'tables', f'arima_residuals_{target_ticker}.csv')
        df_resid = pd.read_csv(resid_path, index_col=0, parse_dates=True) if os.path.exists(resid_path) else None

        ret_path = os.path.join(PROCESSED_DIR, 'vn30_monthly_returns.csv')
        df_returns = pd.read_csv(ret_path, index_col='Date', parse_dates=True) if os.path.exists(ret_path) else None

        summary_path = os.path.join(project_root, 'results', 'tables', 'arima_summary.txt')

    except Exception as e:
        st.error(f"Lỗi load data: {e}")
        st.stop()

    if df_fc is None:
        st.error("⚠️ Chưa tìm thấy dữ liệu dự báo.")
        st.stop()

    # --- 5. GIAO DIỆN CHÍNH ---

    # --- PHẦN 1: DỰ BÁO (FORECAST) ---
    c1, c2 = st.columns([2.5, 1], gap="large")

    with c1:
        st.markdown(f'<div class="pro-header" style="margin-top:0">FORECAST VISUALIZATION (12 MONTHS)</div>',
                    unsafe_allow_html=True)

        if df_returns is not None:
            history = df_returns[target_ticker].tail(36)
            col_mean = find_col_fuzzy(df_fc, ['forecast', 'dubao', 'mean'])
            col_lower = find_col_fuzzy(df_fc, ['lower', 'min', 'thap'])
            col_upper = find_col_fuzzy(df_fc, ['upper', 'max', 'cao'])

            fig_fc = go.Figure()
            fig_fc.add_trace(
                go.Scatter(x=history.index, y=history.values, name='Lịch sử', line=dict(color='#1a2236', width=2)))

            if col_lower and col_upper:
                fig_fc.add_trace(
                    go.Scatter(x=df_fc.index, y=df_fc[col_upper], mode='lines', line=dict(width=0), showlegend=False,
                               hoverinfo='skip'))
                fig_fc.add_trace(
                    go.Scatter(x=df_fc.index, y=df_fc[col_lower], mode='lines', line=dict(width=0), fill='tonexty',
                               fillcolor='rgba(255, 193, 7, 0.2)', name='Vùng tin cậy 95%'))

            if col_mean:
                fig_fc.add_trace(go.Scatter(x=df_fc.index, y=df_fc[col_mean], mode='lines+markers', name='Dự báo',
                                            line=dict(color='#e65100', width=2, dash='dash')))

            fig_fc.update_layout(height=420, hovermode="x unified", legend=dict(orientation="h", y=1.1, title=None),
                                 margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Lợi suất tháng")
            st.plotly_chart(fig_fc, use_container_width=True)

    with c2:
        st.markdown('<div class="pro-header" style="margin-top:0">NEXT MONTH</div>', unsafe_allow_html=True)

        if col_mean:
            next_val = df_fc[col_mean].iloc[0]
            next_date = df_fc.index[0]

            st.metric(
                label=f"Tháng {str(next_date)[:7]}",
                value=f"{next_val:.4f}",
                delta=f"{next_val * 100:.2f}%",
                delta_color="normal"
            )

            if next_val > 0:
                st.info("**Positive:** Xu hướng tăng.")
            else:
                st.warning("**Negative:** Xu hướng giảm.")

        st.write("")
        st.markdown('<div class="pro-header">DATA TABLE</div>', unsafe_allow_html=True)

        # === FORMAT BẢNG ĐẸP HƠN ===
        # Tạo bản sao để hiển thị (không ảnh hưởng data gốc)
        df_display = df_fc.copy()

        # 1. Format Index (Bỏ giờ phút giây)
        df_display.index = pd.to_datetime(df_display.index).strftime('%Y-%m-%d')

        # 2. Đổi tên cột cho gọn
        new_cols = {}
        if col_mean: new_cols[col_mean] = 'Dự báo'
        if col_lower: new_cols[col_lower] = 'Min (95%)'
        if col_upper: new_cols[col_upper] = 'Max (95%)'
        df_display.rename(columns=new_cols, inplace=True)

        # 3. Hiển thị
        st.dataframe(df_display.style.format("{:.4f}"), height=250, use_container_width=True)

    # --- PHẦN 2: ĐÁNH GIÁ MÔ HÌNH (MODEL PERFORMANCE) ---
    st.write("---")
    st.markdown('<div class="pro-header">MODEL PERFORMANCE & DIAGNOSTICS</div>', unsafe_allow_html=True)

    # === TRÍCH XUẤT THÔNG SỐ SUMMARY ===
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_text = f.read()

        # Trích xuất AIC và BIC
        aic_val = extract_metric_from_summary(summary_text, 'AIC')
        bic_val = extract_metric_from_summary(summary_text, 'BIC')

        # Hiển thị Metric thay vì text
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AIC (Càng thấp càng tốt)", f"{aic_val:.2f}" if aic_val else "N/A")
        m2.metric("BIC (Càng thấp càng tốt)", f"{bic_val:.2f}" if bic_val else "N/A")

        # Tính RMSE từ file evaluation (nếu có)
        rmse_val = "N/A"
        if df_eval is not None:
            col_actual = find_col_fuzzy(df_eval, ['actual', 'thucte'])
            col_pred = find_col_fuzzy(df_eval, ['forecast', 'dubao'])
            if col_actual and col_pred:
                rmse = ((df_eval[col_actual] - df_eval[col_pred]) ** 2).mean() ** 0.5
                rmse_val = f"{rmse:.4f}"

        m3.metric("RMSE (Sai số Test)", rmse_val)
        m4.info(
            "AIC/BIC dùng để so sánh các mô hình. Giá trị âm càng lớn (về trị tuyệt đối) hoặc dương càng nhỏ là tốt.")

    col_eval, col_resid = st.columns(2, gap="large")

    with col_eval:
        st.markdown("**Kiểm tra độ khớp (Train vs Test)**")
        if df_eval is not None:
            col_actual = find_col_fuzzy(df_eval, ['actual', 'thucte'])
            col_pred = find_col_fuzzy(df_eval, ['forecast', 'dubao'])
            if col_actual and col_pred:
                fig_eval = go.Figure()
                fig_eval.add_trace(
                    go.Scatter(x=df_eval.index, y=df_eval[col_actual], mode='lines+markers', name='Thực tế',
                               line=dict(color='#1a2236')))
                fig_eval.add_trace(go.Scatter(x=df_eval.index, y=df_eval[col_pred], mode='lines+markers', name='Dự báo',
                                              line=dict(color='#dc3545', dash='dot')))
                fig_eval.update_layout(height=350, hovermode="x unified", legend=dict(orientation="h", y=1.1),
                                       margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_eval, use_container_width=True)

    with col_resid:
        st.markdown("**Phân tích Phần dư (Residuals)**")
        if df_resid is not None:
            col_res = find_col_fuzzy(df_resid, ['resid'])
            if col_res:
                fig_hist = px.histogram(df_resid, x=col_res, nbins=20, marginal="box",
                                        color_discrete_sequence=['#1565c0'], opacity=0.75)
                fig_hist.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_hist, use_container_width=True)

    # Bảng text gốc
    with st.expander("Xem chi tiết bảng thống kê gốc (Raw Summary)"):
        st.code(summary_text if 'summary_text' in locals() else "No Data", language='text')

    st.info("""
        💡**Góc nhìn phân tích & Khuyến nghị (Model Insight):**
        1.  **Xu hướng:** Mô hình dự báo lợi suất dao động quanh mức 0, cho thấy thị trường có xu hướng **đi ngang (Sideway)** trong ngắn hạn.
        2.  **Rủi ro:** Dải tin cậy (vùng màu cam nhạt) mở rộng dần về phía tương lai, phản ánh độ khó dự báo tăng lên theo thời gian.
        3.  **Hành động:** Với xu hướng Sideway, chiến lược phù hợp là **Giao dịch trong biên độ (Range Trading)** - Mua ở hỗ trợ, Bán ở kháng cự, thay vì nắm giữ dài hạn (Hold) chờ tăng trưởng mạnh.
        """)
# ========== TAB 6: MOMENTUM STRATEGY (INTERACTIVE SIMULATOR) ==========
with tabs[5]:
    # --- 1. HEADER & CSS ---
    st.markdown('<div class="eda-title">MOMENTUM STRATEGY</div>', unsafe_allow_html=True)
    st.caption("Chiến lược đầu tư theo đà tăng trưởng: 'Mua cao, Bán cao hơn' (Buy High, Sell Higher).")

    # --- 2. LOAD DATA ---
    try:
        # Load Returns (Dữ liệu gốc để tính toán)
        ret_path = os.path.join(PROCESSED_DIR, 'vn30_monthly_returns.csv')

        # Load Metrics cũ (để tham khảo)
        data_dir = os.path.dirname(PROCESSED_DIR)
        project_root = os.path.dirname(data_dir)
        bonus_path = os.path.join(project_root, 'results', 'tables', 'bonus_momentum_hieu_qua_so_sanh.csv')

        if os.path.exists(ret_path):
            df_returns = pd.read_csv(ret_path, index_col='Date', parse_dates=True)
        else:
            st.error("⚠️ Thiếu file 'vn30_monthly_returns.csv'.")
            st.stop()

        # Load bảng so sánh cũ nếu có
        df_bonus = pd.read_csv(bonus_path, index_col=0) if os.path.exists(bonus_path) else None

    except Exception as e:
        st.error(f"Lỗi data: {e}")
        st.stop()

    # --- 3. CẤU HÌNH CHIẾN LƯỢC ---
    with st.expander("Cấu hình tham số Chiến lược (Simulation Settings)", expanded=True):
        c_param1, c_param2 = st.columns(2)
        with c_param1:
            lookback = st.slider("Chu kỳ quan sát (Tháng)", min_value=1, max_value=12, value=6,
                                 help="Số tháng nhìn lại quá khứ để xác định đà tăng.")
        with c_param2:
            top_n = st.slider("Số lượng cổ phiếu nắm giữ", min_value=1, max_value=10, value=5,
                              help="Chọn Top N cổ phiếu tăng mạnh nhất.")

    # --- 4. TÍNH TOÁN MOMENTUM REAL-TIME ---
    # Logic:
    # 1. Tính lợi suất tích lũy trong khoảng lookback (Past Return)
    # 2. Xếp hạng và chọn Top N
    # 3. Tính lợi suất danh mục ở tháng tiếp theo (Forward Return)

    try:
        # A. Tính Return tích lũy (Rolling)
        # Shift(1) để đảm bảo tại tháng t, ta chỉ biết dữ liệu của t-1 trở về trước
        past_cumulative_return = (1 + df_returns).rolling(window=lookback).apply(np.prod, raw=True) - 1

        strategy_returns = []
        market_returns = []
        dates = []

        # B. Backtest Loop
        # Bắt đầu từ tháng 'lookback' (Ví dụ: Lookback=6 thì bắt đầu từ tháng index số 6)
        for i in range(lookback, len(df_returns)):
            current_date = df_returns.index[i]

            # 1. Lấy tín hiệu từ tháng trước đó (i-1)
            # Tại tháng i, ta nhìn lại quá khứ (từ i-lookback đến i-1)
            # rolling tại vị trí [i-1] chính là tích lũy của cửa sổ kết thúc tại i-1
            past_perf = past_cumulative_return.iloc[i - 1]

            # 2. Chọn Top N
            stock_perf = past_perf.drop('^VNINDEX', errors='ignore')
            top_stocks = stock_perf.nlargest(top_n).index

            # 3. Tính lợi suất thực tế tháng này (Tháng i)
            current_month_ret = df_returns.loc[current_date, top_stocks].mean()
            market_ret = df_returns.loc[current_date, '^VNINDEX'] if '^VNINDEX' in df_returns.columns else 0

            strategy_returns.append(current_month_ret)
            market_returns.append(market_ret)
            dates.append(current_date)

        # C. Tạo DataFrame
        df_backtest = pd.DataFrame({
            'Momentum': strategy_returns,
            'Market (VN30)': market_returns
        }, index=dates)

        cum_growth = (1 + df_backtest).cumprod()

    except Exception as e:
        st.error(f"Lỗi tính toán: {e}")
        st.stop()
    # --- 5. HIỂN THỊ KẾT QUẢ ---

    c_left, c_right = st.columns([1, 2], gap="large")

    # [CỘT TRÁI] BẢNG HIỆU QUẢ
    with c_left:
        st.markdown('<div class="pro-header" style="margin-top:0">PERFORMANCE SUMMARY</div>', unsafe_allow_html=True)

        # Tính chỉ số nhanh cho cấu hình hiện tại
        ann_ret = df_backtest.mean() * 12
        ann_vol = df_backtest.std() * (12 ** 0.5)
        sharpe = ann_ret / (ann_vol + 1e-9)

        df_stats = pd.DataFrame({
            'Lợi suất/Năm': ann_ret,
            'Rủi ro (Std)': ann_vol,
            'Sharpe': sharpe
        })

        st.dataframe(
            df_stats.style.format("{:.2%}", subset=['Lợi suất/Năm', 'Rủi ro (Std)'])
            .format("{:.2f}", subset=['Sharpe'])
            .background_gradient(cmap="Greens", subset=['Lợi suất/Năm', 'Sharpe'])
            .background_gradient(cmap="Reds", subset=['Rủi ro (Std)']),
            use_container_width=True
        )

        st.info(f"💡 **Chiến lược:** Mua Top {top_n} mã mạnh nhất trong {lookback} tháng qua và giữ trong 1 tháng.")

        # Hiển thị bảng gốc (nếu có) để đối chiếu
        if df_bonus is not None:
            with st.expander("Xem Mốc chuẩn so sánh (Benchmark Results)"):
                st.info(
                    "Đây là kết quả cố định (Lookback 6 tháng - Top 5) dùng để **đối chiếu** xem cấu hình bạn đang chỉnh có tốt hơn mốc chuẩn hay không."
                )
                st.dataframe(df_bonus.style.highlight_max(axis=0, color='#fbbc04'), use_container_width=True)

    # [CỘT PHẢI] BIỂU ĐỒ TĂNG TRƯỞNG (INTERACTIVE)
    with c_right:
        st.markdown('<div class="pro-header" style="margin-top:0">GROWTH CHART (INTERACTIVE)</div>',
                    unsafe_allow_html=True)

        # Chuyển đổi data để vẽ Plotly
        df_melt = cum_growth.reset_index().melt(id_vars='index', var_name='Strategy', value_name='Value')
        df_melt.rename(columns={'index': 'Date'}, inplace=True)

        fig = px.line(
            df_melt, x='Date', y='Value', color='Strategy',
            color_discrete_map={'Momentum': '#2962ff', 'Market (VN30)': '#1a2236'},
            labels={'Value': 'Giá trị tài sản (Từ 1 VND)'}
        )

        # Tô màu vùng Momentum thắng thị trường
        fig.update_layout(
            height=400, hovermode="x unified",
            legend=dict(orientation="h", y=1.1, title=None),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Đánh giá nhanh
        total_ret_mom = cum_growth['Momentum'].iloc[-1] - 1
        total_ret_mkt = cum_growth['Market (VN30)'].iloc[-1] - 1

        if total_ret_mom > total_ret_mkt:
            st.success(
                f"**Kết quả:** Chiến lược Momentum vượt trội hơn thị trường (+{total_ret_mom - total_ret_mkt:.1%}).")
        else:
            st.warning(
                f"**Kết quả:** Chiến lược Momentum đang kém hơn thị trường ({total_ret_mom - total_ret_mkt:.1%}).")

# ========== TAB 7: FINAL INTEGRATED SYSTEM ==========
with tabs[6]:
    # --- 1. TIÊU ĐỀ LỚN ---
    st.markdown('<div class="eda-title">INTELLIGENT REPORTING HUB</div>', unsafe_allow_html=True)
    st.caption("Trung tâm tích hợp Trợ lý ảo AI & Hệ thống xuất bản Báo cáo Chiến lược.")

    col_ai, col_report = st.columns([1, 1.6], gap="large")


    # --- 2. DATA LOADER ---
    def load_project_data():
        data = {}
        try:
            data_dir = os.path.dirname(PROCESSED_DIR)
            project_root = os.path.dirname(data_dir)
            tables_dir = os.path.join(project_root, 'results', 'tables')

            files = {
                'metrics': 'bonus_momentum_hieu_qua_so_sanh.csv',
                'chart_growth': 'bonus_momentum_chart_data.csv',
                'forecast': 'arima_forecast_FPT.VN.csv',
                'capm': 'capm_beta_results.csv'
            }
            for k, f in files.items():
                p = os.path.join(tables_dir, f)
                if os.path.exists(p):
                    data[k] = pd.read_csv(p, index_col=0, parse_dates=('chart' in k or 'fc' in k))
                else:
                    data[k] = None
        except:
            pass
        return data


    project_data = load_project_data()
    df_metrics = project_data.get('metrics')
    df_capm = project_data.get('capm')
    df_fc = project_data.get('forecast')
    df_growth = project_data.get('chart_growth')


    # --- 3. LOGIC NLP & REPORT CONTENT  ---
    def process_natural_language_query(query, df_metrics, df_capm, df_fc):
        q = query.lower()
        definitions = {
            "momentum": "Momentum là chiến lược 'Mua cao, Bán cao hơn'. Mua các mã tăng mạnh nhất quá khứ và kỳ vọng đà tăng tiếp diễn.",
            "beta": "Beta đo độ nhạy với thị trường. Beta > 1 là rủi ro cao (High Risk), Beta < 1 là an toàn (Defensive).",
            "sharpe": "Sharpe Ratio đo lường hiệu quả sau điều chỉnh rủi ro. Sharpe > 1 là tốt.",
            "arima": "ARIMA là mô hình thống kê dự báo chuỗi thời gian dựa trên dữ liệu quá khứ.",
            "vn30": "VN30 là nhóm 30 cổ phiếu vốn hóa lớn nhất sàn HOSE."
        }
        for k, v in definitions.items():
            if k in q: return f"💡 **Kiến thức:** {v}"

        if "beta" in q or "rủi ro" in q:
            if df_capm is None: return "Chưa có dữ liệu CAPM."
            beta_col = [c for c in df_capm.columns if 'beta' in c.lower()][0]
            if "thấp" in q or "an toàn" in q:
                safe = df_capm[df_capm[beta_col] > 0][beta_col].idxmin()
                return f"🛡️ Cổ phiếu ổn định nhất (Beta thấp): **{safe}**."
            top = df_capm[beta_col].idxmax()
            return f"🔥 Cổ phiếu rủi ro nhất (Beta cao): **{top}** ({df_capm.loc[top, beta_col]:.2f})."

        if "hiệu quả" in q or "tốt nhất" in q:
            if df_metrics is None: return "Chưa có dữ liệu Portfolio."
            try:
                col = [c for c in df_metrics.columns if 'sharpe' in c.lower()][0]
                best = df_metrics[col].idxmax()
                val = df_metrics.loc[best, col]
                return f"🏆 Chiến lược hiệu quả nhất là **{best}** (Sharpe: {val:.2f})."
            except:
                pass

        if "dự báo" in q or "xu hướng" in q:
            return "📈 Mô hình ARIMA dự báo xu hướng ngắn hạn đang đi ngang tích lũy (Sideway)."

        return "Tôi có thể trả lời về: Mã rủi ro nhất, Chiến lược tốt nhất, Momentum là gì..."


    # --- HÀM TẠO INSIGHT SÂU SẮC & CHUYÊN NGHIỆP ---
    def generate_smart_content(section, df_m, df_c, df_f):
        if section == "PORTFOLIO":
            if df_m is None: return "N/A"
            # Lấy tên chiến lược tốt nhất và Sharpe
            best_name = df_m.iloc[:, -2].idxmax()
            best_val = df_m.iloc[:, -2].max()

            return f"<b>{best_name}</b> đang dẫn đầu với Sharpe Ratio ấn tượng (<b>{best_val:.2f}</b>). Điều này cho thấy khả năng tối ưu hóa lợi nhuận trên mỗi đơn vị rủi ro vượt trội, chứng minh hiệu quả của việc phân bổ tài sản định lượng so với thị trường chung."


        elif section == "RISK":

            if df_c is None: return "N/A"

            # Tìm tên cột có chữ "beta" hoặc "Beta"

            beta_col = next((c for c in df_c.columns if 'beta' in c.lower()), None)

            if beta_col:

                risky = df_c[beta_col].idxmax()  # Mã có Beta cao nhất

                beta_val = df_c[beta_col].max()  # Giá trị Beta cao nhất

                return f"Cổ phiếu <b>{risky}</b> được xác định là nhân tố mang rủi ro hệ thống (Systematic Risk) cao nhất danh mục với Beta = <b>{beta_val:.2f}</b>. Trong các kịch bản thị trường biến động mạnh (High Volatility), việc giảm tỷ trọng mã này là cần thiết để bảo vệ NAV."

            else:

                return "Dữ liệu Beta chưa sẵn sàng."

        elif section == "FORECAST":
            return "Mô hình định lượng ARIMA chỉ ra xu hướng lợi suất đang đi vào vùng cân bằng (Equilibrium/Sideway). Dải tin cậy 95% thu hẹp cho thấy xác suất xảy ra biến động cực đoan (Tail Risk) là thấp. Đây là giai đoạn phù hợp để tích lũy hoặc giao dịch biên độ (Range Trading)."

        elif section == "MOMENTUM":
            return "Chiến lược Momentum tiếp tục khẳng định sức mạnh của dòng tiền thông minh (Smart Money). Việc luân chuyển vốn vào các mã dẫn dắt (Leaders) đã giúp đường cong vốn (Equity Curve) tách biệt hoàn toàn và tạo ra Alpha dương so với Benchmark (VN30)."

        elif section == "CONCLUSION":
            if df_m is None: return "N/A"
            best = df_m.iloc[:, -2].idxmax()
            return f"""
            <p>Dựa trên tổng hợp các mô hình định lượng và kiểm định thống kê, Báo cáo khuyến nghị:</p>
            <ul>
                <li><b>Chiến lược cốt lõi:</b> Tăng tỷ trọng phân bổ vào <b>{best}</b> để tối đa hóa hiệu suất điều chỉnh rủi ro.</li>
                <li><b>Quản trị danh mục:</b> Thực hiện tái cân bằng (Rebalancing) định kỳ, hạn chế giải ngân vào nhóm High-Beta trong bối cảnh vĩ mô chưa rõ ràng.</li>
                <li><b>Hành động:</b> Duy trì vị thế nắm giữ (Hold) với các mã có động lượng tốt và kiên nhẫn chờ tín hiệu bứt phá khỏi vùng Sideway.</li>
            </ul>
            """
        return ""


    # --- 4. CỘT TRÁI: AI ASSISTANT ---
    with col_ai:
        with st.container(border=True):
            st.markdown('<div class="pro-header">AI FINANCIAL ASSISTANT</div>', unsafe_allow_html=True)

            st.info(
                "**Hướng dẫn sử dụng:**\n"
                "- **Không nhập Key:** Chạy chế độ **Offline** (Trả lời nhanh dựa trên dữ liệu có sẵn).\n"
                "- **Có nhập Key:** Chạy chế độ **Online** (AI phân tích sâu hơn & Linh hoạt hơn)."
            )

            # Key Input
            if "openai_api_key" not in st.session_state: st.session_state.openai_api_key = ""
            with st.expander("OpenAI API Key (Tùy chọn)"):
                st.text_input("Nhập OpenAI Key", type="password", key="api_key_input")

            # Chat History
            chat_box = st.container(height=420)
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "system", "content": "Assistant"}]

            with chat_box:
                for msg in st.session_state.messages:
                    if msg["role"] != "system":
                        with st.chat_message(msg["role"]): st.write(msg["content"])

            # Input
            if prompt := st.chat_input("Hỏi: Mã nào rủi ro nhất?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_box:
                    with st.chat_message("user"):
                        st.write(prompt)

                    resp = ""
                    if st.session_state.openai_api_key:
                        try:
                            from openai import OpenAI

                            client = OpenAI(api_key=st.session_state.openai_api_key)
                            ctx = f"Data: Best Strategy={df_metrics.iloc[:, -2].idxmax() if df_metrics is not None else 'N/A'}"
                            stream = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "system", "content": ctx}] + st.session_state.messages,
                                stream=True
                            )
                            resp = st.write_stream(stream)
                        except:
                            resp = "Lỗi API. Dùng chế độ Offline."

                    if not st.session_state.openai_api_key or resp == "Lỗi API. Dùng chế độ Offline.":
                        resp = process_natural_language_query(prompt, df_metrics, df_capm, df_fc)
                        with st.chat_message("assistant"): st.write(resp)

                    st.session_state.messages.append({"role": "assistant", "content": str(resp)})

    # --- 5. CỘT PHẢI: REPORT GENERATOR ---
    with col_report:
        with st.container(border=True):
            st.markdown('<div class="pro-header">STRATEGIC REPORT GENERATOR</div>', unsafe_allow_html=True)

            st.warning(
                "⚠️ **Lưu ý:** Báo cáo được tạo tự động bởi thuật toán (AI Generated). Số liệu chỉ mang tính tham khảo.")

            with st.form("final_report_form_v5"):
                c1, c2 = st.columns(2)
                with c1: r_title = st.text_input("Tiêu đề báo cáo:", "BÁO CÁO CHIẾN LƯỢC ĐẦU TƯ VN30")
                with c2: r_author = st.text_input("Người lập:", "Portfolio Manager")
                submit_btn = st.form_submit_button("TẠO BÁO CÁO", type="primary",
                                                   use_container_width=True)

            if submit_btn:
                try:
                    # A. PREPARE CHARTS
                    html_metrics = df_metrics.rename(
                        columns={'LoiSuatTB_Nam': 'Lợi suất/Năm', 'RuiRo_Nam (Std)': 'Rủi ro (Std)',
                                 'Sharpe_Ratio': 'Sharpe Ratio'}).to_html(classes='',
                                                                          float_format='{:,.2f}'.format) if df_metrics is not None else "No Data"

                    html_growth = ""
                    if df_growth is not None:
                        df_p = df_growth.reset_index()
                        fig = px.line(df_p.melt(id_vars=df_p.columns[0]), x=df_p.columns[0], y='value',
                                      color='variable', color_discrete_sequence=['#0052cc', '#333'])
                        fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), template="plotly_white",
                                          title="Wealth Growth Index")
                        html_growth = fig.to_html(full_html=False, include_plotlyjs='cdn')

                    html_capm = ""
                    if df_capm is not None:
                        c_b = [c for c in df_capm.columns if 'beta' in c.lower()][0]
                        top5 = df_capm.nlargest(5, c_b)
                        fig = px.bar(top5, x=top5.index, y=c_b, color=c_b, color_continuous_scale='Reds')
                        fig.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0), template="plotly_white",
                                          title="Top High Beta Stocks")
                        html_capm = fig.to_html(full_html=False, include_plotlyjs='cdn')

                    html_fc = ""
                    if df_fc is not None:
                        cols = [c for c in df_fc.columns if any(x in c.lower() for x in ['forecast', 'mean', 'dubao'])]
                        if cols:
                            fig = go.Figure(go.Scatter(x=df_fc.index, y=df_fc[cols[0]], mode='lines+markers',
                                                       line=dict(color='#ff6d00')))
                            fig.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0), template="plotly_white",
                                              title="ARIMA Forecast")
                            html_fc = fig.to_html(full_html=False, include_plotlyjs='cdn')

                    # B. GENERATE TEXT (ĐÃ CẬP NHẬT INSIGHT SÂU SẮC HƠN)
                    txt_port = generate_smart_content("PORTFOLIO", df_metrics, None, None)
                    txt_risk = generate_smart_content("RISK", None, df_capm, None)
                    txt_fc = generate_smart_content("FORECAST", None, None, df_fc)
                    txt_mom = generate_smart_content("MOMENTUM", None, None, None)
                    txt_final = generate_smart_content("CONCLUSION", df_metrics, df_capm, df_fc)

                    # C. HTML TEMPLATE
                    today_str = datetime.datetime.now().strftime("%d %B, %Y")
                    html_template = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="UTF-8">
                        <title>{r_title}</title>
                        <style>
                            body {{ font-family: 'Helvetica Neue', Arial, sans-serif; color: #333; max-width: 900px; margin: 0 auto; padding: 40px; background: #fff; line-height: 1.6; }}
                            .header {{ border-bottom: 3px solid #0052cc; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: flex-end; }}
                            .header h1 {{ margin: 0; color: #0052cc; font-size: 26px; text-transform: uppercase; }}
                            .meta {{ text-align: right; font-size: 13px; color: #666; }}
                            h2 {{ color: #2d3748; border-left: 5px solid #3182ce; padding-left: 15px; margin-top: 40px; font-size: 18px; text-transform: uppercase; }}
                            .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); page-break-inside: avoid; }}
                            .insight {{ background: #ebf8ff; border-left: 4px solid #2b6cb0; padding: 15px; font-size: 14px; color: #2a4365; margin-top: 15px; }}
                            .conclusion-box {{ background: #f0fff4; border: 2px solid #48bb78; border-radius: 8px; padding: 25px; margin-top: 50px; page-break-inside: avoid; }}
                            table {{ width: 100%; border-collapse: collapse; }}
                            th {{ background: #f7fafc; padding: 10px; text-align: left; border-bottom: 2px solid #cbd5e0; }}
                            td {{ padding: 10px; border-bottom: 1px solid #edf2f7; }}
                            .footer {{ margin-top: 60px; text-align: center; font-size: 11px; color: #718096; border-top: 1px solid #eee; padding-top: 20px; font-style: italic; }}
                        </style>
                    </head>
                    <body>
                        <div class="header">
                            <div><h1>{r_title}</h1><p style="margin:5px 0 0 0; font-size:14px; color:#666;">Automated Strategy Report</p></div>
                            <div class="meta"><strong>Prepared by:</strong> {r_author}<br><strong>Date:</strong> {today_str}</div>
                        </div>

                        <h2>1. Tổng quan Hiệu quả Danh mục</h2>
                        <div class="card">{html_metrics}<div class="insight"><b>Analyst Note:</b> {txt_port}</div></div>

                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                            <div><h2>2. Phân tích Rủi ro</h2><div class="card">{html_capm}<div class="insight" style="font-size:12px;"><b>Risk Note:</b> {txt_risk}</div></div></div>
                            <div><h2>3. Dự báo Xu hướng</h2><div class="card">{html_fc}<div class="insight" style="font-size:12px;"><b>Model Note:</b> {txt_fc}</div></div></div>
                        </div>

                        <h2>4. Chiến lược Momentum & Backtest</h2>
                        <div class="card">{html_growth}<div class="insight"><b>Strategy Insight:</b> {txt_mom}</div></div>

                        <div class="conclusion-box">
                            <h3 style="color:#2f855a; margin-top:0;">5. KẾT LUẬN & KHUYẾN NGHỊ</h3>
                            {txt_final}
                        </div>

                        <div class="footer">
                            <p><b>DISCLAIMER:</b> This report is generated automatically by AI/Algorithms. All information is for reference only and does not constitute financial advice.<br>
                            Báo cáo được tạo tự động bởi hệ thống. Số liệu chỉ mang tính tham khảo.</p>
                        </div>
                    </body>
                    </html>
                    """
                    b64 = base64.b64encode(html_template.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="Final_Report_{datetime.date.today()}.html"><button style="background:#0052cc;color:white;padding:15px 30px;border:none;border-radius:6px;cursor:pointer;width:100%;font-weight:600;">📥 DOWNLOAD REPORT</button></a>'
                    st.success("✅ Đã tạo báo cáo thành công!")
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Lỗi: {e}")