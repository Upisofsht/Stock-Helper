import streamlit as st
from datetime import date, timedelta
from FinMind.data import DataLoader
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# --- 版面設定 ---
st.set_page_config(layout="wide", page_title="台股全方位操盤助手")

st.title("🇹🇼 台股全方位操盤助手")

if st.button('🔄 刷新最新股價'):
    st.cache_data.clear() # 清除快取，強制重抓
    st.rerun() # 重新執行程式

# --- 1. 台股分類資料庫 ---
stock_categories = {
    "🔥 台積電概念股": {
        "2330": "2330 - 台積電 (護國神山)",
        "2454": "2454 - 聯發科 (IC設計)",
        "3711": "3711 - 日月光投控 (封測)",
        "3443": "3443 - 創意 (IP矽智財)",
        "3661": "3661 - 世芯-KY"
    },
    "🚢 航運三雄": {
        "2603": "2603 - 長榮",
        "2609": "2609 - 陽明",
        "2615": "2615 - 萬海"
    },
    "🤖 AI 伺服器 & 代工": {
        "2382": "2382 - 廣達",
        "3231": "3231 - 緯創",
        "2317": "2317 - 鴻海",
        "2356": "2356 - 英業達",
        "6669": "6669 - 緯穎"
    },
    "⚡ 重電與綠能": {
        "1513": "1513 - 中興電",
        "1519": "1519 - 華城",
        "1503": "1503 - 士電"
    },
    "💾 記憶體": {
        "2337": "2337 - 旺宏",
        "2344": "2344 - 華邦電",
        "2408": "2408 - 南亞科"
    },
     "🏦 金融權值": {
        "2881": "2881 - 富邦金",
        "2882": "2882 - 國泰金",
        "2886": "2886 - 兆豐金"
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

# --- 核心：FinMind 資料抓取函數 (修復版) ---
@st.cache_data
def load_data_finmind(ticker, years):
    dl = DataLoader()
    
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=years*365)).strftime("%Y-%m-%d")
    
    # 1. 股價
    df_price = dl.taiwan_stock_daily(stock_id=ticker, start_date=start_date, end_date=end_date)
    df_price = df_price.rename(columns={
        'date': 'Date', 'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'
    })
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_price[cols] = df_price[cols].astype(float)
    df_price['Date'] = pd.to_datetime(df_price['Date'])

    # 2. 三大法人 (s + buy_sell 計算)
    df_chips = dl.taiwan_stock_institutional_investors(stock_id=ticker, start_date=start_date, end_date=end_date)
    
    if not df_chips.empty:
        df_chips['buy_sell'] = df_chips['buy'] - df_chips['sell'] # 手動計算買賣超
        df_chips = df_chips.pivot(index='date', columns='name', values='buy_sell')
        df_chips.reset_index(inplace=True)
        df_chips.rename(columns={'date': 'Date'}, inplace=True)
        df_chips['Date'] = pd.to_datetime(df_chips['Date'])
        
        df = pd.merge(df_price, df_chips, on='Date', how='left')
        
        if 'Foreign_Investor' not in df.columns: df['Foreign_Investor'] = 0
        if 'Investment_Trust' not in df.columns: df['Investment_Trust'] = 0
        df[['Foreign_Investor', 'Investment_Trust']] = df[['Foreign_Investor', 'Investment_Trust']].fillna(0)
    else:
        df = df_price
        df['Foreign_Investor'] = 0
        df['Investment_Trust'] = 0

    # 3. 本益比 (per_pbr)
    df_per = dl.taiwan_stock_per_pbr(stock_id=ticker, start_date=start_date, end_date=end_date)
    if not df_per.empty:
        df_per = df_per[['date', 'PER', 'dividend_yield']]
        df_per.rename(columns={'date': 'Date'}, inplace=True)
        df_per['Date'] = pd.to_datetime(df_per['Date'])
        df = pd.merge(df, df_per, on='Date', how='left')
    
    # --- 技術指標計算 ---
    # MA
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # KD
    df['9_High'] = df['High'].rolling(9).max()
    df['9_Low'] = df['Low'].rolling(9).min()
    df['RSV'] = (df['Close'] - df['9_Low']) / (df['9_High'] - df['9_Low']) * 100
    df['RSV'] = df['RSV'].fillna(50)
    k_values, d_values = [50], [50]
    rsv_list = df['RSV'].tolist()
    for i in range(1, len(rsv_list)):
        k = (2/3) * k_values[-1] + (1/3) * rsv_list[i]
        d = (2/3) * d_values[-1] + (1/3) * k
        k_values.append(k)
        d_values.append(d)
    df['K'], df['D'] = k_values, d_values
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEM'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['DIF'] - df['DEM']
    
    return df

data_load_state = st.text("FinMind 正在連線證交所抓取資料...")
data = load_data_finmind(selected_stock, lookback_years)
data_load_state.empty()

# --- 基本面看板 ---
last_row = data.iloc[-1]
with st.sidebar:
    st.header(f"🏢 {selected_stock} 營運體質")
    
    per = last_row.get('PER', 0)
    yield_rate = last_row.get('dividend_yield', 0)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if per > 0: st.metric("本益比 (P/E)", f"{per:.1f}")
        else: st.metric("本益比 (P/E)", "⚠️ 虧損中")
            
    with col_s2:
        if yield_rate > 0: st.metric("殖利率 (%)", f"{yield_rate:.1f}")
        else: st.metric("殖利率 (%)", "無配息")

    st.divider()
    st.write("🔍 **籌碼快篩** (今日):")
    
    foreign_buy = last_row['Foreign_Investor']
    trust_buy = last_row['Investment_Trust']
    
    if foreign_buy > 0: st.success(f"💰 外資買: {int(foreign_buy/1000):,} 張")
    elif foreign_buy < 0: st.error(f"💸 外資賣: {int(abs(foreign_buy)/1000):,} 張")
        
    if trust_buy > 0: st.success(f"🏦 投信買: {int(trust_buy/1000):,} 張")
    elif trust_buy < 0: st.warning(f"📉 投信賣: {int(abs(trust_buy)/1000):,} 張")

# --- 訊號綜合判讀 logic (更新!) ---
if strategy_mode == "短線衝浪 (MA5 + MA10)":
    ma_fast_col, ma_slow_col = 'MA5', 'MA10'
    ma_fast_label, ma_slow_label = "MA5 (攻擊線)", "MA10 (操盤線)"
    line_color_fast, line_color_slow = '#00FFFF', '#FF00FF'
else:
    ma_fast_col, ma_slow_col = 'MA20', 'MA60'
    ma_fast_label, ma_slow_label = "MA20 (月線)", "MA60 (季線)"
    line_color_fast, line_color_slow = '#FFD700', '#FF8C00'

prev_row = data.iloc[-2]
curr_fast, curr_slow = last_row[ma_fast_col], last_row[ma_slow_col]
prev_fast, prev_slow = prev_row[ma_fast_col], prev_row[ma_slow_col]

# 1. MA 訊號
ma_status = "持平"
if prev_fast < prev_slow and curr_fast > curr_slow:
    ma_status = "Gold" # 黃金交叉
elif prev_fast > prev_slow and curr_fast < curr_slow:
    ma_status = "Death" # 死亡交叉
else:
    ma_status = "Bull" if curr_fast > curr_slow else "Bear"

# 2. KD 訊號
k_curr = last_row['K']
kd_msg = "KD中性"
if k_curr > 80: kd_msg = "⚠️ KD超買 (過熱)"
elif k_curr < 20: kd_msg = "💎 KD超賣 (地板)"

# 3. MACD 訊號
macd_hist = last_row['MACD_Hist']
macd_msg = "MACD翻紅 (多)" if macd_hist > 0 else "MACD翻綠 (空)"

# 4. 綜合文字生成
signal_title = "無動作"
signal_color = "gray"
signal_body = f"📊 技術指標狀態:\n- {kd_msg}\n- {macd_msg}"

if ma_status == "Gold":
    signal_title = "🚀 黃金交叉 (買進)"
    signal_color = "green"
    signal_body = f"注意！{ma_fast_label} 穿過 {ma_slow_label}，且 {macd_msg}。\n" + signal_body
elif ma_status == "Death":
    signal_title = "📉 死亡交叉 (賣出)"
    signal_color = "red"
    signal_body = f"警告！{ma_fast_label} 跌破 {ma_slow_label}，趨勢轉弱。\n" + signal_body
elif ma_status == "Bull":
    signal_title = "📈 持股續抱 (多頭)"
    signal_color = "green"
    signal_body = f"目前均線多頭排列。\n" + signal_body
else:
    signal_title = "🐻 空手觀望 (空頭)"
    signal_color = "blue"
    signal_body = f"目前均線空頭排列，不建議進場。\n" + signal_body

st.divider()

# --- 戰情中心 ---
st.subheader(f"📢 {stock_name} - 綜合分析")
if signal_color == "green": st.success(f"### {signal_title}\n{signal_body}")
elif signal_color == "red": st.error(f"### {signal_title}\n{signal_body}")
else: st.info(f"### {signal_title}\n{signal_body}")

change = last_row['Close'] - prev_row['Close']
pct_change = (change / prev_row['Close']) * 100
st.metric(label=f"最新收盤價 ({last_row['Date'].strftime('%Y-%m-%d')})", 
          value=f"{last_row['Close']:.2f}", 
          delta=f"{change:.2f} ({pct_change:.2f}%)")

# --- 圖表區 ---
with st.container(border=True):
    st.markdown(f"### 📊 台股專業線圖 (含三大法人籌碼)")
    
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.02, 
                        row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
                        subplot_titles=("股價 & 均線", "成交量", "法人籌碼 (外資/投信)", "KD 指標", "MACD 指標"))

    # 1. K線圖
    fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="K線", increasing_line_color='#ef5350',decreasing_line_color='#26a69a'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data[ma_fast_col], name=ma_fast_label, line=dict(color=line_color_fast, width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data[ma_slow_col], name=ma_slow_label, line=dict(color=line_color_slow, width=1.5)), row=1, col=1)

    condition = data[ma_fast_col] > data[ma_slow_col]
    buy_signals = data.loc[(condition == True) & (condition.shift(1) == False)]
    sell_signals = data.loc[(condition == False) & (condition.shift(1) == True)]
    fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Low']*0.98, mode='markers', name='MA買訊', marker=dict(symbol='triangle-up', size=10, color='#00FF00')), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['High']*1.02, mode='markers', name='MA賣訊', marker=dict(symbol='triangle-down', size=10, color='#FF0000')), row=1, col=1)

    # 2. 成交量
    colors_vol = ['#ef5350' if row['Open'] - row['Close'] < 0 else '#26a69a' for index, row in data.iterrows()]
    fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name="成交量", marker_color=colors_vol), row=2, col=1)

    # 3. 法人籌碼
    fig.add_trace(go.Bar(x=data['Date'], y=data['Foreign_Investor'], name="外資買賣超", marker_color='#2962FF'), row=3, col=1)
    fig.add_trace(go.Bar(x=data['Date'], y=data['Investment_Trust'], name="投信買賣超", marker_color='#FF6D00'), row=3, col=1)

    # 4. KD 指標
    fig.add_trace(go.Scatter(x=data['Date'], y=data['K'], name="K值", line=dict(color='orange', width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['D'], name="D值", line=dict(color='purple', width=1)), row=4, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="gray", row=4, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="gray", row=4, col=1)

    # 5. MACD 指標
    colors_macd = ['#ef5350' if val >= 0 else '#26a69a' for val in data['MACD_Hist']]
    fig.add_trace(go.Bar(x=data['Date'], y=data['MACD_Hist'], name="MACD柱狀", marker_color=colors_macd), row=5, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['DIF'], name="DIF", line=dict(color='#2962FF', width=1)), row=5, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['DEM'], name="DEM", line=dict(color='#FF6D00', width=1)), row=5, col=1)

    dt_all = pd.date_range(start=data['Date'].iloc[0], end=data['Date'].iloc[-1])
    dt_obs = [d.strftime("%Y-%m-%d") for d in data['Date']]
    dt_breaks = [d.strftime("%Y-%m-%d") for d in dt_all if d.strftime("%Y-%m-%d") not in dt_obs]
    
    fig.update_layout(height=1000, xaxis_rangeslider_visible=False, dragmode='pan', hovermode='x unified', margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
    st.plotly_chart(fig, width='stretch')