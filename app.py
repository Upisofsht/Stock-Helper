import streamlit as st
from datetime import date, timedelta
from FinMind.data import DataLoader
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

# --- 1. 頁面設定 (設定為寬版，但手機會自動適應) ---
st.set_page_config(layout="wide", page_title="台股戰情室", initial_sidebar_state="collapsed")
# initial_sidebar_state="collapsed" -> 手機預設把選單收起來，讓畫面更大

st.title("🇹🇼 台股戰情室")

# --- 2. 側邊欄：所有「控制項」都藏在這裡 ---
with st.sidebar:
    st.header("⚙️ 設定面板")
    
    # 資料庫
    stock_categories = {
        "🔥 台積電概念": {"2330": "2330-台積電", "2454": "2454-聯發科", "3711": "3711-日月光", "3661": "3661-世芯"},
        "🚢 航運三雄": {"2603": "2603-長榮", "2609": "2609-陽明", "2615": "2615-萬海"},
        "🤖 AI 伺服器": {"2382": "2382-廣達", "3231": "3231-緯創", "2317": "2317-鴻海", "6669": "6669-緯穎"},
        "⚡ 重電綠能": {"1513": "1513-中興電", "1519": "1519-華城", "1503": "1503-士電"},
        "💾 記憶體": {"2337": "2337-旺宏", "2344": "2344-華邦電", "2408": "2408-南亞科"},
        "🏦 金融權值": {"2881": "2881-富邦金", "2882": "2882-國泰金", "2886": "2886-兆豐金"}
    }
    
    # 選單邏輯
    sel_cat = st.selectbox("板塊", list(stock_categories.keys()))
    sel_stock_list = stock_categories[sel_cat]
    sel_stock = st.selectbox("股票", options=list(sel_stock_list.keys()), format_func=lambda x: sel_stock_list[x])
    stock_name = sel_stock_list[sel_stock]

    st.divider()
    
    # 參數設定
    lookback_years = st.slider("K線長度(年)", 0.5, 3.0, 1.0, step=0.5) # 縮短預設長度加快繪圖
    strategy_mode = st.radio("策略", ("短線 (MA5/10)", "波段 (MA20/60)"))

# --- 3. 核心數據抓取 (速度優化版) ---
# ttl=43200 (12小時)，代表你早上抓過一次，下午再開都不用重新連線，秒開
@st.cache_data(ttl=43200, show_spinner=False)
def load_data_finmind(ticker, years):
    dl = DataLoader()
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=years*365)).strftime("%Y-%m-%d")
    
    # 股價
    df = dl.taiwan_stock_daily(stock_id=ticker, start_date=start_date, end_date=end_date)
    df = df.rename(columns={'date': 'Date', 'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'})
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df['Date'] = pd.to_datetime(df['Date'])

    # 籌碼
    chips = dl.taiwan_stock_institutional_investors(stock_id=ticker, start_date=start_date, end_date=end_date)
    if not chips.empty:
        chips['bs'] = chips['buy'] - chips['sell']
        chips = chips.pivot(index='date', columns='name', values='bs').reset_index().rename(columns={'date': 'Date'})
        chips['Date'] = pd.to_datetime(chips['Date'])
        df = pd.merge(df, chips, on='Date', how='left')
        for col in ['Foreign_Investor', 'Investment_Trust']:
            if col not in df.columns: df[col] = 0
            else: df[col] = df[col].fillna(0)
    else:
        df['Foreign_Investor'] = 0
        df['Investment_Trust'] = 0

    # 本益比
    per = dl.taiwan_stock_per_pbr(stock_id=ticker, start_date=start_date, end_date=end_date)
    if not per.empty:
        per = per[['date', 'PER', 'dividend_yield']].rename(columns={'date': 'Date'})
        per['Date'] = pd.to_datetime(per['Date'])
        df = pd.merge(df, per, on='Date', how='left')

    # 技術指標一次算完 (這樣切換策略不用重抓)
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # KD & MACD
    df['9_High'] = df['High'].rolling(9).max()
    df['9_Low'] = df['Low'].rolling(9).min()
    df['RSV'] = (df['Close'] - df['9_Low']) / (df['9_High'] - df['9_Low']) * 100
    df['RSV'] = df['RSV'].fillna(50)
    
    # 快速計算 KD (Vectorized approach optimization is hard for recursive, sticking to loop but simplified)
    k, d = 50, 50
    k_list, d_list = [], []
    for rsv in df['RSV']:
        k = k * 2/3 + rsv * 1/3
        d = d * 2/3 + k * 1/3
        k_list.append(k)
        d_list.append(d)
    df['K'], df['D'] = k_list, d_list

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEM'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['DIF'] - df['DEM']

    return df

# 執行資料載入
with st.spinner('🚀 正在連線交易所...'):
    data = load_data_finmind(sel_stock, lookback_years)

# --- 4. 戰情中心 (手機最上面先看這個) ---
last = data.iloc[-1]
prev = data.iloc[-2]
chg = last['Close'] - prev['Close']
pct = chg / prev['Close'] * 100

# 用 columns 排列重點資訊，節省垂直空間
c1, c2 = st.columns([1.5, 2.5])
with c1:
    st.metric("最新價", f"{last['Close']:.1f}", f"{chg:.1f} ({pct:.1f}%)")
with c2:
    # 策略判讀
    if strategy_mode == "短線 (MA5/10)":
        fast, slow = 'MA5', 'MA10'
        fast_n, slow_n = 'MA5', 'MA10'
    else:
        fast, slow = 'MA20', 'MA60'
        fast_n, slow_n = 'MA20', 'MA60'

    curr_f, curr_s = last[fast], last[slow]
    prev_f, prev_s = prev[fast], prev[slow]
    
    status_text = ""
    if prev_f < prev_s and curr_f > curr_s:
        st.success(f"🚀 **黃金交叉**\n({fast_n} 穿過 {slow_n})")
    elif prev_f > prev_s and curr_f < curr_s:
        st.error(f"📉 **死亡交叉**\n({fast_n} 跌破 {slow_n})")
    elif curr_f > curr_s:
        st.success(f"📈 **多頭續抱**\n(均線向上)")
    else:
        st.info(f"🐻 **空頭觀望**\n(均線向下)")

# --- 5. 分頁切換 (Tabs) - 這是手機版面乾淨的關鍵 ---
tab1, tab2 = st.tabs(["📊 技術線圖", "🏢 體質與籌碼"])

with tab1:
    # 圖表優化：邊距縮小，隱藏不必要的工具列
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        subplot_titles=("價量均線", "成交量", "法人籌碼", "KD & MACD"))
    
    # 繪圖邏輯簡化版
    fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], 
                                 increasing_line_color='#ef5350', decreasing_line_color='#26a69a', name="K"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data[fast], line=dict(color='orange', width=1), name=fast_n), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data[slow], line=dict(color='purple', width=1), name=slow_n), row=1, col=1)
    
    # 買賣點 (只畫最近的，減少運算)
    buy = data.loc[(data[fast] > data[slow]) & (data[fast].shift(1) <= data[slow].shift(1))]
    sell = data.loc[(data[fast] < data[slow]) & (data[fast].shift(1) >= data[slow].shift(1))]
    if not buy.empty: fig.add_trace(go.Scatter(x=buy['Date'], y=buy['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', size=10, color='red'), name='買'), row=1, col=1)
    if not sell.empty: fig.add_trace(go.Scatter(x=sell['Date'], y=sell['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', size=10, color='green'), name='賣'), row=1, col=1)

    # 副圖們
    colors_vol = ['#ef5350' if o < c else '#26a69a' for o, c in zip(data['Open'], data['Close'])]
    fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], marker_color=colors_vol, name="量"), row=2, col=1)
    
    fig.add_trace(go.Bar(x=data['Date'], y=data['Foreign_Investor'], marker_color='blue', name="外資"), row=3, col=1)
    fig.add_trace(go.Bar(x=data['Date'], y=data['Investment_Trust'], marker_color='orange', name="投信"), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=data['Date'], y=data['K'], line=dict(color='orange', width=1), name="K"), row=4, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['D'], line=dict(color='purple', width=1), name="D"), row=4, col=1)
    fig.add_trace(go.Bar(x=data['Date'], y=data['MACD_Hist'], marker_color=['red' if v>0 else 'green' for v in data['MACD_Hist']], name="MACD"), row=4, col=1)

    # 手機版面配置：隱藏 RangeSlider，調整邊距
    fig.update_layout(height=800, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False, showlegend=False)
    
    # 去除假日空隙
    dt_all = pd.date_range(start=data['Date'].iloc[0], end=data['Date'].iloc[-1])
    dt_breaks = [d.strftime("%Y-%m-%d") for d in dt_all if d.strftime("%Y-%m-%d") not in data['Date'].dt.strftime("%Y-%m-%d").tolist()]
    fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False}) 
    # config={'displayModeBar': False} -> 這是關鍵！手機上把那些煩人的縮放按鈕藏起來

with tab2:
    st.subheader("📊 籌碼與基本面")
    
    col_a, col_b = st.columns(2)
    per = last.get('PER', 0)
    dy = last.get('dividend_yield', 0)
    
    with col_a:
        if per > 0: st.metric("本益比", f"{per:.1f}")
        else: st.metric("本益比", "虧損/無")
    with col_b:
        st.metric("殖利率", f"{dy:.1f}%")
        
    st.divider()
    
    col_c, col_d = st.columns(2)
    fi = last['Foreign_Investor']
    it = last['Investment_Trust']
    
    with col_c:
        st.metric("外資今日", f"{int(fi/1000)}張", delta_color="normal" if fi==0 else "inverse")
    with col_d:
        st.metric("投信今日", f"{int(it/1000)}張", delta_color="normal" if it==0 else "inverse")
        
    st.info("💡 提示：外資適合看波段，投信適合看短線爆發。")