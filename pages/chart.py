import streamlit as st
from datetime import date, timedelta
from FinMind.data import DataLoader
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from config import stock_categories, FINMIND_API_TOKEN

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="板塊指數線圖", initial_sidebar_state="collapsed")
st.title("📈 板塊指數技術線圖")

# 返回首頁按鈕
if st.button("⬅️ 返回首頁", key="back_home"):
    st.switch_page("app.py")

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 圖表設定")
    
    selected_sector = st.selectbox("選擇板塊", list(stock_categories.keys()))
    lookback_days = st.slider("顯示天數", 30, 365, 90, step=30)
    
    st.divider()
    st.subheader("📊 技術指標")
    show_ma = st.checkbox("顯示均線", value=True)
    show_volume = st.checkbox("顯示成交量", value=True)
    show_macd = st.checkbox("顯示 MACD", value=True)
    show_kd = st.checkbox("顯示 KD", value=True)
    
    st.divider()
    st.subheader("🎨 均線設定")
    ma_fast = st.selectbox("快線", [5, 10, 20], index=0)
    ma_mid = st.selectbox("中線", [10, 20, 60], index=1)
    ma_slow = st.selectbox("慢線", [60, 120, 240], index=0)

# --- 工具函數 ---
def extract_stock_info(stock_dict):
    """從 '2330-台積電' 格式提取代號和純名稱"""
    clean_dict = {}
    for code, full_name in stock_dict.items():
        name = full_name.split('-', 1)[1] if '-' in full_name else full_name
        clean_dict[code] = name
    return clean_dict

@st.cache_data(ttl=3600, show_spinner=False)
def load_sector_index_data(sector_stocks, days):
    """載入板塊數據並計算指數"""
    dl = DataLoader()
    dl.login_by_token(api_token=FINMIND_API_TOKEN)
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=days+120)).strftime("%Y-%m-%d")
    
    all_stocks_data = []
    
    for stock_id, stock_name in sector_stocks.items():
        try:
            # 股價資料
            df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)
            if df.empty:
                continue
            
            df = df.rename(columns={
                'date': 'Date', 'open': 'Open', 'max': 'High', 
                'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'
            })
            df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            df['Date'] = pd.to_datetime(df['Date'])
            
            all_stocks_data.append({
                'id': stock_id,
                'name': stock_name,
                'data': df
            })
            
        except Exception as e:
            continue
    
    if not all_stocks_data:
        return pd.DataFrame()
    
    # 計算板塊指數（等權重平均）
    # 找出共同日期
    common_dates = all_stocks_data[0]['data']['Date']
    for stock_info in all_stocks_data[1:]:
        common_dates = pd.merge(
            pd.DataFrame({'Date': common_dates}),
            stock_info['data'][['Date']],
            on='Date'
        )['Date']
    
    index_data = []
    
    for date_val in common_dates:
        daily_open = []
        daily_high = []
        daily_low = []
        daily_close = []
        daily_volume = []
        
        for stock_info in all_stocks_data:
            day_data = stock_info['data'][stock_info['data']['Date'] == date_val]
            if not day_data.empty:
                daily_open.append(day_data['Open'].iloc[0])
                daily_high.append(day_data['High'].iloc[0])
                daily_low.append(day_data['Low'].iloc[0])
                daily_close.append(day_data['Close'].iloc[0])
                daily_volume.append(day_data['Volume'].iloc[0])
        
        if daily_close:
            index_data.append({
                'Date': date_val,
                'Open': np.mean(daily_open),
                'High': np.mean(daily_high),
                'Low': np.mean(daily_low),
                'Close': np.mean(daily_close),
                'Volume': np.sum(daily_volume)
            })
    
    index_df = pd.DataFrame(index_data)
    
    if index_df.empty:
        return pd.DataFrame()
    
    # 計算技術指標
    # 均線
    index_df[f'MA{ma_fast}'] = index_df['Close'].rolling(ma_fast).mean()
    index_df[f'MA{ma_mid}'] = index_df['Close'].rolling(ma_mid).mean()
    index_df[f'MA{ma_slow}'] = index_df['Close'].rolling(ma_slow).mean()
    
    # KD
    index_df['9_High'] = index_df['High'].rolling(9).max()
    index_df['9_Low'] = index_df['Low'].rolling(9).min()
    index_df['RSV'] = (index_df['Close'] - index_df['9_Low']) / (index_df['9_High'] - index_df['9_Low']) * 100
    index_df['RSV'] = index_df['RSV'].fillna(50)
    
    k, d = 50, 50
    k_list, d_list = [], []
    for rsv in index_df['RSV']:
        k = k * 2/3 + rsv * 1/3
        d = d * 2/3 + k * 1/3
        k_list.append(k)
        d_list.append(d)
    index_df['K'], index_df['D'] = k_list, d_list
    
    # MACD
    exp1 = index_df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = index_df['Close'].ewm(span=26, adjust=False).mean()
    index_df['DIF'] = exp1 - exp2
    index_df['DEM'] = index_df['DIF'].ewm(span=9, adjust=False).mean()
    index_df['MACD_Hist'] = index_df['DIF'] - index_df['DEM']
    
    return index_df.tail(days)

# --- 載入資料 ---
with st.spinner(f'📊 正在載入 {selected_sector} 板塊指數...'):
    sector_stocks = extract_stock_info(stock_categories[selected_sector])
    index_data = load_sector_index_data(sector_stocks, lookback_days)
    
    if index_data.empty:
        st.error("❌ 無法載入板塊數據，請稍後再試")
        st.stop()

# --- 顯示關鍵指標 ---
st.header(f"{selected_sector} 板塊指數概況")

last = index_data.iloc[-1]
prev = index_data.iloc[-2] if len(index_data) > 1 else last
first = index_data.iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)

change = last['Close'] - prev['Close']
pct_change = (change / prev['Close']) * 100

period_return = (last['Close'] - first['Close']) / first['Close'] * 100

with col1:
    st.metric("最新指數", f"{last['Close']:.2f}", 
             f"{change:.2f} ({pct_change:.2f}%)")

with col2:
    st.metric(f"{lookback_days}日報酬", f"{period_return:.2f}%",
             delta_color="normal" if period_return > 0 else "inverse")

with col3:
    ma_trend = "多頭" if last['Close'] > last[f'MA{ma_slow}'] else "空頭"
    st.metric("趨勢", ma_trend)

with col4:
    kd_status = "超賣" if last['K'] < 20 else "超買" if last['K'] > 80 else "中性"
    st.metric("KD 狀態", kd_status, f"K={last['K']:.0f}")

with col5:
    macd_status = "多頭" if last['MACD_Hist'] > 0 else "空頭"
    st.metric("MACD", macd_status, f"{last['MACD_Hist']:.2f}")

# --- 繪製圖表 ---
st.divider()

# 計算子圖數量
subplot_count = 1  # 主圖
if show_volume:
    subplot_count += 1
if show_macd:
    subplot_count += 1
if show_kd:
    subplot_count += 1

# 設定行高
row_heights = [0.5]  # 主圖
if show_volume:
    row_heights.append(0.15)
if show_macd:
    row_heights.append(0.175)
if show_kd:
    row_heights.append(0.175)

# 創建子圖
subplot_titles = ["板塊指數 K 線圖"]
if show_volume:
    subplot_titles.append("成交量")
if show_macd:
    subplot_titles.append("MACD")
if show_kd:
    subplot_titles.append("KD")

fig = make_subplots(
    rows=subplot_count, 
    cols=1, 
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=row_heights,
    subplot_titles=subplot_titles
)

current_row = 1

# 1. 主圖 - K線和均線
fig.add_trace(go.Candlestick(
    x=index_data['Date'],
    open=index_data['Open'],
    high=index_data['High'],
    low=index_data['Low'],
    close=index_data['Close'],
    increasing_line_color='#ef5350',
    decreasing_line_color='#26a69a',
    name="指數"
), row=current_row, col=1)

if show_ma:
    fig.add_trace(go.Scatter(
        x=index_data['Date'], 
        y=index_data[f'MA{ma_fast}'],
        line=dict(color='orange', width=1.5),
        name=f"MA{ma_fast}"
    ), row=current_row, col=1)
    
    fig.add_trace(go.Scatter(
        x=index_data['Date'], 
        y=index_data[f'MA{ma_mid}'],
        line=dict(color='purple', width=1.5),
        name=f"MA{ma_mid}"
    ), row=current_row, col=1)
    
    fig.add_trace(go.Scatter(
        x=index_data['Date'], 
        y=index_data[f'MA{ma_slow}'],
        line=dict(color='blue', width=1.5),
        name=f"MA{ma_slow}"
    ), row=current_row, col=1)

current_row += 1

# 2. 成交量
if show_volume:
    colors_vol = ['#ef5350' if o < c else '#26a69a' 
                  for o, c in zip(index_data['Open'], index_data['Close'])]
    fig.add_trace(go.Bar(
        x=index_data['Date'], 
        y=index_data['Volume'],
        marker_color=colors_vol,
        name="量",
        showlegend=False
    ), row=current_row, col=1)
    current_row += 1

# 3. MACD
if show_macd:
    fig.add_trace(go.Scatter(
        x=index_data['Date'], 
        y=index_data['DIF'],
        line=dict(color='blue', width=1),
        name="DIF"
    ), row=current_row, col=1)
    
    fig.add_trace(go.Scatter(
        x=index_data['Date'], 
        y=index_data['DEM'],
        line=dict(color='orange', width=1),
        name="DEM"
    ), row=current_row, col=1)
    
    fig.add_trace(go.Bar(
        x=index_data['Date'], 
        y=index_data['MACD_Hist'],
        marker_color=['red' if v > 0 else 'green' for v in index_data['MACD_Hist']],
        name="MACD",
        showlegend=False
    ), row=current_row, col=1)
    current_row += 1

# 4. KD
if show_kd:
    fig.add_trace(go.Scatter(
        x=index_data['Date'], 
        y=index_data['K'],
        line=dict(color='orange', width=1),
        name="K"
    ), row=current_row, col=1)
    
    fig.add_trace(go.Scatter(
        x=index_data['Date'], 
        y=index_data['D'],
        line=dict(color='purple', width=1),
        name="D"
    ), row=current_row, col=1)
    
    # 添加超買超賣線
    fig.add_hline(y=80, line_dash="dash", line_color="red", 
                 opacity=0.5, row=current_row, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", 
                 opacity=0.5, row=current_row, col=1)

# 更新佈局
fig.update_layout(
    height=800,
    margin=dict(l=10, r=10, t=50, b=10),
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# 移除週末空白
dt_all = pd.date_range(start=index_data['Date'].iloc[0], end=index_data['Date'].iloc[-1])
dt_breaks = [d.strftime("%Y-%m-%d") for d in dt_all 
            if d.strftime("%Y-%m-%d") not in index_data['Date'].dt.strftime("%Y-%m-%d").tolist()]
fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

st.plotly_chart(fig, width='stretch', config={'displayModeBar': True})

# --- 技術分析總結 ---
st.divider()
st.header("📊 技術分析總結")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.subheader("📈 趨勢分析")
    
    if last['Close'] > last[f'MA{ma_slow}']:
        st.success(f"✅ 多頭趨勢 (價格 > MA{ma_slow})")
    else:
        st.error(f"❌ 空頭趨勢 (價格 < MA{ma_slow})")
    
    # 均線排列
    if last[f'MA{ma_fast}'] > last[f'MA{ma_mid}'] > last[f'MA{ma_slow}']:
        st.success("✅ 多頭排列 (快>中>慢)")
    elif last[f'MA{ma_fast}'] < last[f'MA{ma_mid}'] < last[f'MA{ma_slow}']:
        st.error("❌ 空頭排列 (快<中<慢)")
    else:
        st.warning("⚠️ 均線糾結")

with col_b:
    st.subheader("🎯 KD 指標")
    
    k_val = last['K']
    d_val = last['D']
    
    if k_val < 20:
        st.success(f"💎 超賣區 (K={k_val:.0f})")
        st.write("建議：可考慮逢低布局")
    elif k_val > 80:
        st.error(f"🔥 超買區 (K={k_val:.0f})")
        st.write("建議：注意獲利了結")
    else:
        st.info(f"📊 中性區 (K={k_val:.0f})")
    
    # KD 交叉
    if len(index_data) > 1:
        prev_k = index_data.iloc[-2]['K']
        prev_d = index_data.iloc[-2]['D']
        
        if prev_k < prev_d and k_val > d_val:
            st.success("🚀 黃金交叉 (K上穿D)")
        elif prev_k > prev_d and k_val < d_val:
            st.error("💀 死亡交叉 (K下穿D)")

with col_c:
    st.subheader("⚡ MACD 動能")
    
    macd_val = last['MACD_Hist']
    
    if macd_val > 0:
        st.success(f"📈 多頭動能 ({macd_val:.2f})")
    else:
        st.error(f"📉 空頭動能 ({macd_val:.2f})")
    
    # MACD 柱狀圖翻轉
    if len(index_data) > 1:
        prev_macd = index_data.iloc[-2]['MACD_Hist']
        
        if prev_macd <= 0 and macd_val > 0:
            st.success("🚀 翻多訊號")
        elif prev_macd >= 0 and macd_val < 0:
            st.error("💀 翻空訊號")
        elif macd_val > prev_macd:
            st.info("📊 動能增強")
        else:
            st.warning("⚠️ 動能減弱")

# --- 綜合建議 ---
st.divider()
st.header("💡 綜合操作建議")

# 計算綜合評分
score = 0
reasons = []

# 1. 趨勢 (40分)
if last['Close'] > last[f'MA{ma_slow}']:
    score += 40
    reasons.append("✅ 多頭趨勢")
elif last['Close'] > last[f'MA{ma_slow}'] * 0.97:
    score += 20
    reasons.append("⚠️ 接近趨勢線")
else:
    reasons.append("❌ 空頭趨勢")

# 2. KD (30分)
if k_val < 30:
    score += 30
    reasons.append(f"✅ KD 低檔 ({k_val:.0f})")
elif k_val < 50:
    score += 20
    reasons.append(f"📊 KD 中性偏低 ({k_val:.0f})")
elif k_val < 80:
    score += 10
    reasons.append(f"⚠️ KD 中性偏高 ({k_val:.0f})")
else:
    reasons.append(f"❌ KD 過熱 ({k_val:.0f})")

# 3. MACD (30分)
if macd_val > 0:
    score += 30
    reasons.append("✅ MACD 多頭")
elif macd_val > index_data.iloc[-2]['MACD_Hist']:
    score += 15
    reasons.append("📊 MACD 收斂中")
else:
    reasons.append("❌ MACD 空頭")

col_rec1, col_rec2 = st.columns([1, 2])

with col_rec1:
    if score >= 80:
        st.success(f"### 🚀 強力買進\n板塊評分: {score}/100")
    elif score >= 60:
        st.info(f"### 📊 可逢低布局\n板塊評分: {score}/100")
    elif score >= 40:
        st.warning(f"### ⚠️ 觀望為主\n板塊評分: {score}/100")
    else:
        st.error(f"### 🛑 建議減碼\n板塊評分: {score}/100")

with col_rec2:
    st.write("**評分依據：**")
    for reason in reasons:
        st.write(reason)
    
    st.write(f"\n**{lookback_days}日報酬率**: {period_return:.2f}%")

st.divider()
st.caption(f"⚠️ 技術分析基於 {selected_sector} 板塊等權重指數，僅供參考。投資有風險，請審慎評估。")
