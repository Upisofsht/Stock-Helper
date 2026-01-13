import streamlit as st
from datetime import date, timedelta
from FinMind.data import DataLoader
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from config import stock_categories, FINMIND_API_TOKEN

# --- 1. 頁面設定 ---
st.set_page_config(layout="wide", page_title="台股自用戰略看盤程式", initial_sidebar_state="expanded")
st.title("🇹🇼 台股自用戰略看盤程式 (進階策略回測版)")

# --- 快速導航按鈕 ---
st.markdown("**📌 快速導航：**")
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

def safe_switch_page(page_path):
    """安全切換頁面，失敗時顯示提示"""
    try:
        st.switch_page(page_path)
    except Exception as e:
        st.error(f"無法切換到 {page_path}，請使用左側邊欄的頁面導航")

with nav_col1:
    if st.button("🎯 板塊資金雷達", key="nav_sector", use_container_width=True):
        safe_switch_page("pages/sector.py")
with nav_col2:
    if st.button("🔄 板塊輪動分析", key="nav_rotation", use_container_width=True):
        safe_switch_page("pages/rotation.py")
with nav_col3:
    if st.button("🤖 AI 選股", key="nav_ai", use_container_width=True):
        safe_switch_page("pages/ai_picker.py")
with nav_col4:
    if st.button("📈 板塊線圖", key="nav_chart", use_container_width=True):
        safe_switch_page("pages/chart.py")

st.divider()

# --- 2. 側邊欄設定 ---
with st.sidebar:
    st.header("📌 頁面導航")
    nav_sidebar_col1, nav_sidebar_col2 = st.columns(2)
    with nav_sidebar_col1:
        if st.button("🎯 資金雷達", key="sb_nav_sector", use_container_width=True):
            safe_switch_page("pages/sector.py")
        if st.button("🔄 輪動分析", key="sb_nav_rotation", use_container_width=True):
            safe_switch_page("pages/rotation.py")
    with nav_sidebar_col2:
        if st.button("🤖 AI選股", key="sb_nav_ai", use_container_width=True):
            safe_switch_page("pages/ai_picker.py")
        if st.button("📈 板塊線圖", key="sb_nav_chart", use_container_width=True):
            safe_switch_page("pages/chart.py")
    
    st.divider()
    st.header("⚙️ 設定面板")
    
    sel_cat = st.selectbox("板塊", list(stock_categories.keys()))
    sel_stock_list = stock_categories[sel_cat]
    sel_stock = st.selectbox("股票", options=list(sel_stock_list.keys()), 
                            format_func=lambda x: sel_stock_list[x])
    stock_name = sel_stock_list[sel_stock]
    
    st.divider()
    lookback_years = st.slider("回測期間(年)", 0.5, 3.0, 2.0, step=0.5)
    strategy_mode = st.radio("策略模式", ("短線 (MA5/10)", "波段 (MA20/60)"))
    
    st.divider()
    st.subheader("🛡️ 風控參數")
    stop_loss_pct = st.slider("停損(%)", 3, 20, 8, step=1)
    take_profit_pct = st.slider("停利(%)", 5, 50, 15, step=5)
    
    st.divider()
    st.subheader("📊 策略權重")
    weight_trend = st.slider("趨勢權重", 0, 100, 60, step=5)
    weight_kd = st.slider("KD權重", 0, 100, 15, step=5)
    weight_macd = st.slider("MACD權重", 0, 100, 25, step=5)

# --- 3. 資料抓取 ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_data_finmind(ticker, years):
    try:
        dl = DataLoader()
        dl.login_by_token(api_token=FINMIND_API_TOKEN)
        end_date = date.today().strftime("%Y-%m-%d")
        start_date = (date.today() - timedelta(days=int(years*365 + 100))).strftime("%Y-%m-%d") # 多抓100天供均線運算
        
        # 股價
        df = dl.taiwan_stock_daily(stock_id=ticker, start_date=start_date, end_date=end_date)
        if df.empty: return pd.DataFrame()

        df = df.rename(columns={'date': 'Date', 'open': 'Open', 'max': 'High', 
                               'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'})
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 籌碼 (加入例外處理，以免 FinMind 沒回傳籌碼導致崩潰)
        try:
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
                df['Foreign_Investor'] = 0; df['Investment_Trust'] = 0
        except:
            df['Foreign_Investor'] = 0; df['Investment_Trust'] = 0
        
        # 計算技術指標
        df = calculate_indicators(df)
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame()

def calculate_indicators(df):
    """計算所有技術指標"""
    # 均線
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['MA120'] = df['Close'].rolling(120).mean()  # 趨勢濾網用
    
    # KD 指標
    df['9_High'] = df['High'].rolling(9).max()
    df['9_Low'] = df['Low'].rolling(9).min()
    df['RSV'] = (df['Close'] - df['9_Low']) / (df['9_High'] - df['9_Low']) * 100
    df['RSV'] = df['RSV'].fillna(50)
    
    k, d = 50, 50
    k_list, d_list = [], []
    for rsv in df['RSV']:
        k = k * 2/3 + rsv * 1/3
        d = d * 2/3 + k * 1/3
        k_list.append(k)
        d_list.append(d)
    df['K'], df['D'] = k_list, d_list
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEM'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['DIF'] - df['DEM']
    
    return df

def calculate_strategy_score(row, prev_row, mode, w_trend, w_kd, w_macd):
    """策略評分制：0-100分"""
    score = 0
    details = []
    
    # 選擇均線參數
    if mode == "短線 (MA5/10)":
        fast, slow, trend = 'MA5', 'MA10', 'MA60'
    else:
        fast, slow, trend = 'MA20', 'MA60', 'MA120'
    
    # 1️⃣ 趨勢濾網 (Trend Filter)
    trend_score = 0
    if pd.notna(row[trend]) and pd.notna(row['Close']):
        if row['Close'] > row[trend]:
            trend_score = 100
            details.append(f"✅ 多頭趨勢 (價>{trend})")
        elif row['Close'] > row[trend] * 0.97:  # 接近趨勢線
            trend_score = 50
            details.append(f"⚠️ 接近趨勢線")
        else:
            trend_score = 0
            details.append(f"❌ 空頭趨勢 (價<{trend})")
    
    # 2️⃣ KD 進場時機
    kd_score = 0
    k_val = row['K']
    if k_val < 20:
        kd_score = 100
        details.append(f"💎 KD超賣 ({k_val:.0f})")
    elif k_val < 30:
        kd_score = 80
        details.append(f"🟢 KD偏低 ({k_val:.0f})")
    elif k_val < 50:
        kd_score = 60
        details.append(f"🟡 KD中性偏低 ({k_val:.0f})")
    elif k_val < 80:
        kd_score = 40
        details.append(f"🟠 KD中性偏高 ({k_val:.0f})")
    else:
        kd_score = 0
        details.append(f"🔴 KD過熱 ({k_val:.0f})")
    
    # 3️⃣ MACD 動能
    macd_score = 0
    macd_val = row['MACD_Hist']
    prev_macd = prev_row['MACD_Hist'] if prev_row is not None else 0
    
    if macd_val > 0 and prev_macd <= 0:  # 柱狀圖翻正
        macd_score = 100
        details.append("🚀 MACD翻多")
    elif macd_val > 0:
        macd_score = 70
        details.append("📈 MACD偏多")
    elif macd_val > prev_macd:  # 負值但收斂
        macd_score = 50
        details.append("🔄 MACD收斂中")
    else:
        macd_score = 20
        details.append("📉 MACD偏空")
    
    # 加權計算總分
    total_weight = w_trend + w_kd + w_macd
    if total_weight > 0:
        score = (trend_score * w_trend + kd_score * w_kd + macd_score * w_macd) / total_weight
    
    return score, details

def run_backtest(df, mode, stop_loss, take_profit, w_trend, w_kd, w_macd):
    """執行回測"""
    if mode == "短線 (MA5/10)":
        fast, slow = 'MA5', 'MA10'
    else:
        fast, slow = 'MA20', 'MA60'
    
    trades = []
    position = None
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 計算當前評分
        score, details = calculate_strategy_score(row, prev_row, mode, w_trend, w_kd, w_macd)
        
        # 如果沒有持倉
        if position is None:
            # 買入條件：評分 >= 70 且 MA黃金交叉
            if score >= 70 and prev_row[fast] < prev_row[slow] and row[fast] > row[slow]:
                position = {
                    'entry_date': row['Date'],
                    'entry_price': row['Close'],
                    'entry_score': score,
                    'stop_loss': row['Close'] * (1 - stop_loss/100),
                    'take_profit': row['Close'] * (1 + take_profit/100)
                }
        
        # 如果有持倉，檢查出場條件
        elif position is not None:
            exit_reason = None
            exit_price = row['Close']
            
            # 1. 停損 (使用 Low 來判斷是否觸價)
            if row['Low'] <= position['stop_loss']:
                exit_reason = '停損'
                exit_price = position['stop_loss'] # 假設停損價成交
            
            # 2. 停利 (使用 High 來判斷是否觸價)
            elif row['High'] >= position['take_profit']:
                exit_reason = '停利'
                exit_price = position['take_profit'] # 假設停利價成交
            
            # 3. 趨勢反轉：評分掉到30以下
            elif score < 30:
                exit_reason = '趨勢轉弱'
            
            # 4. MA死亡交叉
            elif prev_row[fast] > prev_row[slow] and row[fast] < row[slow]:
                exit_reason = 'MA死叉'
            
            # 記錄交易
            if exit_reason:
                pnl = (exit_price - position['entry_price']) / position['entry_price'] * 100
                trades.append({
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry_price'],
                    'exit_date': row['Date'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl,
                    'reason': exit_reason,
                    'hold_days': (row['Date'] - position['entry_date']).days
                })
                position = None
    
    # 如果還有持倉，以最後價格結算
    if position is not None:
        last_row = df.iloc[-1]
        pnl = (last_row['Close'] - position['entry_price']) / position['entry_price'] * 100
        trades.append({
            'entry_date': position['entry_date'],
            'entry_price': position['entry_price'],
            'exit_date': last_row['Date'],
            'exit_price': last_row['Close'],
            'pnl_pct': pnl,
            'reason': '持有中',
            'hold_days': (last_row['Date'] - position['entry_date']).days
        })
    
    return trades

# --- 主程式區塊 ---
with st.spinner('🚀 正在連線交易所...'):
    data = load_data_finmind(sel_stock, lookback_years)

if data.empty:
    st.error(f"❌ 無法取得 {sel_stock_list[sel_stock]} 的資料，可能是代號錯誤或交易所連線中斷。")
else:
    # --- 4. 當前訊號分析 ---
    last = data.iloc[-1]
    prev = data.iloc[-2]
    chg = last['Close'] - prev['Close']
    pct = chg / prev['Close'] * 100

    # 計算當前評分
    current_score, score_details = calculate_strategy_score(
        last, prev, strategy_mode, 
        weight_trend, weight_kd, weight_macd
    )

    # 顯示當前狀態
    c1, c2, c3 = st.columns([1.2, 1.5, 1.3])

    with c1:
        st.metric("最新價", f"{last['Close']:.1f}", f"{chg:.1f} ({pct:.1f}%)")

    with c2:
        # 評分顯示
        if current_score >= 70:
            st.success(f"### 🎯 策略評分: {current_score:.0f}/100")
            st.write("**強力買進訊號**")
        elif current_score >= 50:
            st.info(f"### 🎯 策略評分: {current_score:.0f}/100")
            st.write("**觀望，可等更佳時機**")
        else:
            st.warning(f"### 🎯 策略評分: {current_score:.0f}/100")
            st.write("**建議觀望或減碼**")

    with c3:
        st.write("**評分細節:**")
        for detail in score_details:
            st.write(detail)

    # --- 5. 執行回測 ---
    with st.spinner('📊 執行回測分析...'):
        trades = run_backtest(data, strategy_mode, stop_loss_pct, take_profit_pct,
                             weight_trend, weight_kd, weight_macd)

    # 計算績效
    if trades:
        trades_df = pd.DataFrame(trades)
        win_trades = trades_df[trades_df['pnl_pct'] > 0]
        loss_trades = trades_df[trades_df['pnl_pct'] <= 0]
        
        total_return = trades_df['pnl_pct'].sum()
        win_rate = len(win_trades) / len(trades_df) * 100
        avg_win = win_trades['pnl_pct'].mean() if len(win_trades) > 0 else 0
        avg_loss = loss_trades['pnl_pct'].mean() if len(loss_trades) > 0 else 0
        
        # 處理 loss 為 0 的除法錯誤
        if len(loss_trades) > 0 and loss_trades['pnl_pct'].sum() != 0:
            profit_factor = abs(win_trades['pnl_pct'].sum() / loss_trades['pnl_pct'].sum())
        else:
            profit_factor = float('inf')
        
        # 績效儀表板
        st.divider()
        st.subheader("📈 回測績效總覽")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("交易次數", f"{len(trades_df)}筆")
        col2.metric("累積報酬 (單利)", f"{total_return:.1f}%", 
                    delta_color="normal" if total_return > 0 else "inverse")
        col3.metric("勝率", f"{win_rate:.1f}%")
        col4.metric("平均獲利", f"{avg_win:.1f}%")
        col5.metric("平均虧損", f"{avg_loss:.1f}%")

    # --- 6. 分頁內容 ---
    tab1, tab2, tab3 = st.tabs(["📊 技術線圖", "💰 回測明細", "🏢 體質與籌碼"])

    with tab1:
        # 繪製圖表
        if strategy_mode == "短線 (MA5/10)":
            fast, slow = 'MA5', 'MA10'
        else:
            fast, slow = 'MA20', 'MA60'
        
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03,
                            row_heights=[0.5, 0.15, 0.15, 0.2],
                            subplot_titles=("價量與交易訊號", "成交量", "法人籌碼", "KD & MACD"))
        
        # K線
        fig.add_trace(go.Candlestick(
            x=data['Date'], open=data['Open'], high=data['High'], 
            low=data['Low'], close=data['Close'],
            increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
            name="K線"), row=1, col=1)
        
        # 均線
        fig.add_trace(go.Scatter(x=data['Date'], y=data[fast], 
                                line=dict(color='orange', width=1.5), name=fast), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data[slow], 
                                line=dict(color='purple', width=1.5), name=slow), row=1, col=1)
        
        # 標註交易點
        if trades:
            for trade in trades:
                # 買點
                fig.add_trace(go.Scatter(
                    x=[trade['entry_date']], y=[trade['entry_price']*0.98],
                    mode='markers+text',
                    marker=dict(symbol='triangle-up', size=12, color='red'),
                    text=['買'], textposition='bottom center',
                    showlegend=False), row=1, col=1)
                
                # 賣點
                color = 'green' if trade['pnl_pct'] > 0 else 'black'
                fig.add_trace(go.Scatter(
                    x=[trade['exit_date']], y=[trade['exit_price']*1.02],
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', size=12, color=color),
                    text=[f"{trade['pnl_pct']:.1f}%"], textposition='top center',
                    showlegend=False), row=1, col=1)
        
        # 成交量
        colors_vol = ['#ef5350' if o < c else '#26a69a' for o, c in zip(data['Open'], data['Close'])]
        fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], 
                            marker_color=colors_vol, name="量"), row=2, col=1)
        
        # 籌碼
        fig.add_trace(go.Bar(x=data['Date'], y=data['Foreign_Investor'], 
                            marker_color='blue', name="外資"), row=3, col=1)
        fig.add_trace(go.Bar(x=data['Date'], y=data['Investment_Trust'], 
                            marker_color='orange', name="投信"), row=3, col=1)
        
        # KD & MACD
        fig.add_trace(go.Scatter(x=data['Date'], y=data['K'], 
                                line=dict(color='orange', width=1), name="K"), row=4, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['D'], 
                                line=dict(color='purple', width=1), name="D"), row=4, col=1)
        fig.add_trace(go.Bar(x=data['Date'], y=data['MACD_Hist'],
                            marker_color=['red' if v>0 else 'green' for v in data['MACD_Hist']],
                            name="MACD"), row=4, col=1)
        
        fig.update_layout(height=900, margin=dict(l=10, r=10, t=30, b=10),
                         xaxis_rangeslider_visible=False, showlegend=False)
        
        # 移除週末空白
        dt_all = pd.date_range(start=data['Date'].iloc[0], end=data['Date'].iloc[-1])
        dt_breaks = [d.strftime("%Y-%m-%d") for d in dt_all 
                    if d.strftime("%Y-%m-%d") not in data['Date'].dt.strftime("%Y-%m-%d").tolist()]
        fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
        
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

    with tab2:
        if trades:
            st.subheader("💰 歷史交易明細")
            
            # 轉換為DataFrame並格式化
            display_df = trades_df.copy()
            display_df['entry_date'] = display_df['entry_date'].dt.strftime('%Y-%m-%d')
            display_df['exit_date'] = display_df['exit_date'].dt.strftime('%Y-%m-%d')
            display_df['entry_price'] = display_df['entry_price'].round(2)
            display_df['exit_price'] = display_df['exit_price'].round(2)
            display_df['pnl_pct'] = display_df['pnl_pct'].round(2)
            
            display_df.columns = ['進場日期', '進場價', '出場日期', '出場價', '報酬率(%)', '出場原因', '持有天數']
            
            st.dataframe(display_df, width='stretch')
            
            # 策略分析
            st.divider()
            st.subheader("📊 策略分析")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**出場原因分布:**")
                reason_count = trades_df['reason'].value_counts()
                for reason, count in reason_count.items():
                    st.write(f"• {reason}: {count}筆")
            
            with col_b:
                st.write("**持倉天數分析:**")
                st.write(f"• 平均: {trades_df['hold_days'].mean():.1f}天")
                st.write(f"• 最長: {trades_df['hold_days'].max()}天")
                st.write(f"• 最短: {trades_df['hold_days'].min()}天")
            
            if profit_factor != float('inf'):
                st.success(f"💎 **獲利因子 (Profit Factor): {profit_factor:.2f}** (賺賠比，> 1.5 為優質策略)")
        else:
            st.info("📭 在此期間內沒有產生交易訊號")

    with tab3:
        st.subheader("📊 籌碼與基本面")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("外資今日", f"{int(last['Foreign_Investor']/1000)}張",
                     delta_color="normal" if last['Foreign_Investor']==0 else "inverse")
        
        with col_b:
            st.metric("投信今日", f"{int(last['Investment_Trust']/1000)}張",
                     delta_color="normal" if last['Investment_Trust']==0 else "inverse")
        
        st.info("💡 策略說明：\n\n"
               "• **趨勢濾網**: 確保在主要趨勢方向交易 (MA60/MA120)\n"
               "• **KD時機**: 專注低檔轉折，避免追高\n"
               "• **風控機制**: 嚴格執行停損停利 (Stop Loss / Take Profit)")

st.divider()