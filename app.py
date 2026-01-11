import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

# --- 版面設定 ---
st.set_page_config(layout="wide", page_title="全方位操盤助手 (戰情版)")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("📈 全方位操盤助手 (戰情中心)")

# --- 1. 股票分類資料庫 (巢狀字典) ---
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

# --- 側邊欄與選單邏輯 ---
col1, col2, col3 = st.columns([1.2, 1, 1]) 

with col1:
    # 第一層：選擇族群
    selected_category = st.selectbox("1️⃣ 選擇板塊/群組", list(stock_categories.keys()))
    
    # 根據選到的族群，抓出對應的股票清單
    current_stock_list = stock_categories[selected_category]
    
    # 第二層：選擇股票
    selected_stock = st.selectbox(
        "2️⃣ 選擇股票", 
        options=list(current_stock_list.keys()), 
        format_func=lambda x: current_stock_list[x]
    )

with col2:
    lookback_years = st.slider("回顧歷史年數:", 1, 5, 1)

with col3:
    strategy_mode = st.radio("選擇操作風格", ("短線衝浪 (MA5 + MA10)", "波段趨勢 (MA20 + MA60)"))

# 取得股票中文名稱 (用於標題)
stock_name = current_stock_list[selected_stock]

# 計算日期
start_date = pd.to_datetime(TODAY) - pd.DateOffset(years=lookback_years)
start_date_str = start_date.strftime("%Y-%m-%d")

@st.cache_data
def load_data(ticker, start):
    # 下載資料
    data = yf.download(ticker, start, TODAY)
    data.reset_index(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # 計算均線
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA60'] = data['Close'].rolling(window=60).mean()
    return data

data_load_state = st.text("正在抓取最新數據...")
data = load_data(selected_stock, start_date_str)
data_load_state.empty()

# --- 參數設定 ---
if strategy_mode == "短線衝浪 (MA5 + MA10)":
    ma_fast_col, ma_slow_col = 'MA5', 'MA10'
    ma_fast_label, ma_slow_label = "MA5 (攻擊線)", "MA10 (操盤線)"
    line_color_fast, line_color_slow = '#00FFFF', '#FF00FF'
else:
    ma_fast_col, ma_slow_col = 'MA20', 'MA60'
    ma_fast_label, ma_slow_label = "MA20 (月線)", "MA60 (季線)"
    line_color_fast, line_color_slow = '#FFD700', '#FF8C00'

# --- 核心：即時訊號判讀邏輯 ---
last_row = data.iloc[-1]   # 今天
prev_row = data.iloc[-2]   # 昨天

curr_fast = last_row[ma_fast_col]
curr_slow = last_row[ma_slow_col]
prev_fast = prev_row[ma_fast_col]
prev_slow = prev_row[ma_slow_col]

signal_status = "無動作"
signal_color = "gray"
signal_msg = "趨勢延續中..."

if prev_fast < prev_slow and curr_fast > curr_slow:
    signal_status = "🚀 黃金交叉 (買進)"
    signal_color = "green"
    signal_msg = f"注意！{ma_fast_label} 剛剛向上穿過 {ma_slow_label}，趨勢轉強！"
elif prev_fast > prev_slow and curr_fast < curr_slow:
    signal_status = "📉 死亡交叉 (賣出)"
    signal_color = "red"
    signal_msg = f"警告！{ma_fast_label} 剛剛向下跌破 {ma_slow_label}，建議獲利了結。"
else:
    if curr_fast > curr_slow:
        signal_status = "📈 持股續抱 (多頭)"
        signal_color = "green"
        signal_msg = f"目前趨勢向上，{ma_fast_label} 在 {ma_slow_label} 之上。"
    else:
        signal_status = "🐻 空手觀望 (空頭)"
        signal_color = "blue"
        signal_msg = f"目前趨勢向下，不建議進場。"

st.divider()

# --- 戰情中心顯示區 ---
st.subheader(f"📢 {stock_name} - 目前訊號狀態")

if signal_color == "green":
    st.success(f"### {signal_status}\n{signal_msg}")
elif signal_color == "red":
    st.error(f"### {signal_status}\n{signal_msg}")
else:
    st.info(f"### {signal_status}\n{signal_msg}")

# 顯示最新數據
change = last_row['Close'] - prev_row['Close']
pct_change = (change / prev_row['Close']) * 100
st.metric(label=f"最新收盤價 ({last_row['Date'].strftime('%Y-%m-%d')})", 
          value=f"{last_row['Close']:.2f}", 
          delta=f"{change:.2f} ({pct_change:.2f}%)")

# --- 圖表區 ---
with st.container(border=True):
    st.markdown(f"### 📊 技術分析圖表")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data[ma_fast_col], name=ma_fast_label, line=dict(color=line_color_fast, width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data[ma_slow_col], name=ma_slow_label, line=dict(color=line_color_slow, width=1.5)), row=1, col=1)

    # 標記歷史買賣點
    condition = data[ma_fast_col] > data[ma_slow_col]
    buy_signals = data.loc[(condition == True) & (condition.shift(1) == False)]
    sell_signals = data.loc[(condition == False) & (condition.shift(1) == True)]

    fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Low']*0.98, mode='markers', name='買訊', marker=dict(symbol='triangle-up', size=12, color='#00FF00', line=dict(width=1, color='black'))), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['High']*1.02, mode='markers', name='賣訊', marker=dict(symbol='triangle-down', size=12, color='#FF0000', line=dict(width=1, color='black'))), row=1, col=1)

    colors = ['#ef5350' if row['Open'] - row['Close'] < 0 else '#26a69a' for index, row in data.iterrows()]
    fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name="成交量", marker_color=colors), row=2, col=1)

    dt_all = pd.date_range(start=data['Date'].iloc[0], end=data['Date'].iloc[-1])
    dt_obs = [d.strftime("%Y-%m-%d") for d in data['Date']]
    dt_breaks = [d.strftime("%Y-%m-%d") for d in dt_all if d.strftime("%Y-%m-%d") not in dt_obs]
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, dragmode='pan', hovermode='x unified')
    fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
    
    # 這裡照你的要求修改為 width='stretch'
    st.plotly_chart(fig, width='stretch')