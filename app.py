import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# --- 版面設定 ---
st.set_page_config(layout="wide", page_title="全方位操盤助手 (終極版)")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("📈 全方位操盤助手 (技術+基本雙刀流)")

# --- 股票資料庫 ---
stock_categories = {
    "🇺🇸 美股科技巨頭": {
        "NVDA": "NVDA - 輝達 (AI霸主)",
        "AAPL": "AAPL - 蘋果",
        "MSFT": "MSFT - 微軟",
        "GOOG": "GOOG - Google",
        "TSLA": "TSLA - 特斯拉",
        "AMD":  "AMD - 超微",
        "AVGO": "AVGO - 博通"
    },
    "🇹🇼 台積電與半導體供應鏈": {
        "2330.TW": "2330 - 台積電 (晶圓代工)",
        "2454.TW": "2454 - 聯發科 (IC設計)",
        "2303.TW": "2303 - 聯電",
        "3711.TW": "3711 - 日月光投控 (封測)",
        "3443.TW": "3443 - 創意 (IP矽智財)",
        "3661.TW": "3661 - 世芯-KY"
    },
    "💾 記憶體族群": {
        "MU":      "MU - 美光 (美股)",
        "2337.TW": "2337 - 旺宏 (Flash)",
        "2344.TW": "2344 - 華邦電",
        "2408.TW": "2408 - 南亞科 (DRAM)",
        "6770.TW": "6770 - 力積電"
    },
    "⚡ 電源供應器廠": {
        "2308.TW": "2308 - 台達電 (龍頭)",
        "2301.TW": "2301 - 光寶科",
        "6409.TW": "6409 - 旭隼 (UPS股王)"
    },
    "🚢 航運三雄": {
        "2603.TW": "2603 - 長榮",
        "2609.TW": "2609 - 陽明",
        "2615.TW": "2615 - 萬海"
    },
    "🤖 AI 伺服器組裝": {
        "2382.TW": "2382 - 廣達",
        "3231.TW": "3231 - 緯創",
        "2317.TW": "2317 - 鴻海",
        "2356.TW": "2356 - 英業達"
    }
}

# --- 側邊欄 ---
col1, col2, col3 = st.columns([1.2, 1, 1]) 

with col1:
    selected_category = st.selectbox("1️⃣ 選擇板塊/群組", list(stock_categories.keys()))
    current_stock_list = stock_categories[selected_category]
    selected_stock = st.selectbox("2️⃣ 選擇股票", options=list(current_stock_list.keys()), format_func=lambda x: current_stock_list[x])

with col2:
    lookback_years = st.slider("回顧歷史年數:", 1, 5, 1)

with col3:
    strategy_mode = st.radio("選擇操作風格", ("短線衝浪 (MA5 + MA10)", "波段趨勢 (MA20 + MA60)"))

stock_name = current_stock_list[selected_stock]
start_date = pd.to_datetime(TODAY) - pd.DateOffset(years=lookback_years)
start_date_str = start_date.strftime("%Y-%m-%d")

# --- 新增：基本面資料抓取函數 ---
@st.cache_data
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except:
        return None

# --- 原本的技術面資料抓取函數 ---
@st.cache_data
def load_data(ticker, start):
    data = yf.download(ticker, start, TODAY)
    data.reset_index(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA60'] = data['Close'].rolling(window=60).mean()
    
    # KD
    data['9_High'] = data['High'].rolling(9).max()
    data['9_Low'] = data['Low'].rolling(9).min()
    data['RSV'] = (data['Close'] - data['9_Low']) / (data['9_High'] - data['9_Low']) * 100
    data['RSV'] = data['RSV'].fillna(50)
    k_values, d_values = [50], [50]
    rsv_list = data['RSV'].tolist()
    for i in range(1, len(rsv_list)):
        k = (2/3) * k_values[-1] + (1/3) * rsv_list[i]
        d = (2/3) * d_values[-1] + (1/3) * k
        k_values.append(k)
        d_values.append(d)
    data['K'], data['D'] = k_values, d_values
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['DIF'] = exp1 - exp2
    data['DEM'] = data['DIF'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['DIF'] - data['DEM']
    
    return data

data_load_state = st.text("正在分析大數據...")
data = load_data(selected_stock, start_date_str)
info = get_stock_info(selected_stock) # 抓取基本面
data_load_state.empty()

# --- 顯示基本面資訊 (放在側邊欄) ---
with st.sidebar:
    st.header(f"🏢 {selected_stock} 基本面")
    if info:
        # 容錯處理：有些股票可能沒這些資料
        pe = info.get('trailingPE', 'N/A')
        eps = info.get('trailingEps', 'N/A')
        mkt_cap = info.get('marketCap', 0) / 100000000 # 換算成億
        sector = info.get('sector', '未知')
        
        st.markdown(f"**產業**: {sector}")
        
        # 顯示指標
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("本益比 (P/E)", f"{pe}")
        with col_s2:
            st.metric("EPS", f"{eps}")
        
        st.metric("總市值 (億)", f"{mkt_cap:,.0f}")
        
        # 簡單的基本面評價邏輯
        st.divider()
        st.write("🔍 **體質快篩**:")
        if isinstance(pe, (int, float)) and pe > 40:
            st.warning("⚠️ 本益比過高 (股價偏貴)")
        elif isinstance(pe, (int, float)) and pe < 15 and pe > 0:
            st.success("✅ 本益比便宜 (價值股)")
            
        if isinstance(eps, (int, float)) and eps < 0:
            st.error("❌ 公司目前虧損中")
        else:
            st.success("✅ 公司獲利中")
    else:
        st.write("查無基本面資料")

# --- 參數設定 ---
if strategy_mode == "短線衝浪 (MA5 + MA10)":
    ma_fast_col, ma_slow_col = 'MA5', 'MA10'
    ma_fast_label, ma_slow_label = "MA5 (攻擊線)", "MA10 (操盤線)"
    line_color_fast, line_color_slow = '#00FFFF', '#FF00FF'
else:
    ma_fast_col, ma_slow_col = 'MA20', 'MA60'
    ma_fast_label, ma_slow_label = "MA20 (月線)", "MA60 (季線)"
    line_color_fast, line_color_slow = '#FFD700', '#FF8C00'

# --- 訊號判讀區 ---
last_row = data.iloc[-1]
prev_row = data.iloc[-2]

curr_fast, curr_slow = last_row[ma_fast_col], last_row[ma_slow_col]
prev_fast, prev_slow = prev_row[ma_fast_col], prev_row[ma_slow_col]

kd_msg = "KD中性"
if last_row['K'] > 80: kd_msg = "KD超買(過熱)"
elif last_row['K'] < 20: kd_msg = "KD超賣(過冷)"

signal_status = "無動作"
signal_color = "gray"
signal_msg = f"KD數值: K={last_row['K']:.1f}, D={last_row['D']:.1f} ({kd_msg})"

if prev_fast < prev_slow and curr_fast > curr_slow:
    signal_status = "🚀 黃金交叉 (買進)"
    signal_color = "green"
    signal_msg += f"\nMA趨勢轉強！"
elif prev_fast > prev_slow and curr_fast < curr_slow:
    signal_status = "📉 死亡交叉 (賣出)"
    signal_color = "red"
    signal_msg += f"\nMA趨勢轉弱！"
else:
    if curr_fast > curr_slow:
        signal_status = "📈 持股續抱 (多頭)"
        signal_color = "green"
    else:
        signal_status = "🐻 空手觀望 (空頭)"
        signal_color = "blue"

st.divider()

# 顯示戰情
st.subheader(f"📢 {stock_name} - 綜合分析")
if signal_color == "green": st.success(f"### {signal_status}\n{signal_msg}")
elif signal_color == "red": st.error(f"### {signal_status}\n{signal_msg}")
else: st.info(f"### {signal_status}\n{signal_msg}")

change = last_row['Close'] - prev_row['Close']
pct_change = (change / prev_row['Close']) * 100
st.metric(label=f"最新收盤價 ({last_row['Date'].strftime('%Y-%m-%d')})", 
          value=f"{last_row['Close']:.2f}", 
          delta=f"{change:.2f} ({pct_change:.2f}%)")

# --- 進階圖表區 ---
with st.container(border=True):
    st.markdown(f"### 📊 專業技術線圖 (MA + Vol + KD + MACD)")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.02, 
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        subplot_titles=("股價 & 均線", "成交量", "KD 指標", "MACD 指標"))

    fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data[ma_fast_col], name=ma_fast_label, line=dict(color=line_color_fast, width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data[ma_slow_col], name=ma_slow_label, line=dict(color=line_color_slow, width=1.5)), row=1, col=1)

    condition = data[ma_fast_col] > data[ma_slow_col]
    buy_signals = data.loc[(condition == True) & (condition.shift(1) == False)]
    sell_signals = data.loc[(condition == False) & (condition.shift(1) == True)]
    fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Low']*0.98, mode='markers', name='MA買訊', marker=dict(symbol='triangle-up', size=10, color='#00FF00')), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['High']*1.02, mode='markers', name='MA賣訊', marker=dict(symbol='triangle-down', size=10, color='#FF0000')), row=1, col=1)

    colors_vol = ['#ef5350' if row['Open'] - row['Close'] < 0 else '#26a69a' for index, row in data.iterrows()]
    fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name="成交量", marker_color=colors_vol), row=2, col=1)

    fig.add_trace(go.Scatter(x=data['Date'], y=data['K'], name="K值", line=dict(color='orange', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['D'], name="D值", line=dict(color='purple', width=1)), row=3, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="gray", row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="gray", row=3, col=1)

    colors_macd = ['#ef5350' if val >= 0 else '#26a69a' for val in data['MACD_Hist']]
    fig.add_trace(go.Bar(x=data['Date'], y=data['MACD_Hist'], name="MACD柱狀", marker_color=colors_macd), row=4, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['DIF'], name="DIF (快)", line=dict(color='#2962FF', width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['DEM'], name="DEM (慢)", line=dict(color='#FF6D00', width=1)), row=4, col=1)

    dt_all = pd.date_range(start=data['Date'].iloc[0], end=data['Date'].iloc[-1])
    dt_obs = [d.strftime("%Y-%m-%d") for d in data['Date']]
    dt_breaks = [d.strftime("%Y-%m-%d") for d in dt_all if d.strftime("%Y-%m-%d") not in dt_obs]
    
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, dragmode='pan', hovermode='x unified', margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
    
    st.plotly_chart(fig, width='stretch')