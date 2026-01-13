"""
AI 智能選股系統 - 優化版

性能優化策略：
1. 資料抓取與運算分離：使用獨立快取函數 load_single_stock_data()
2. 進階快取策略：
   - 單股資料快取 2 小時 (TTL=7200s)
   - 計算結果快取，避免重複運算
   - 參數變動時只重新計算，不重新抓取資料
3. 批次處理：使用進度條顯示載入進度
4. 手動控制：提供「重新整理資料」按鈕清除快取
"""

import streamlit as st
from datetime import date, timedelta
from FinMind.data import DataLoader
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from config import stock_categories, FINMIND_API_TOKEN
import time

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="AI 智能選股", initial_sidebar_state="collapsed")
st.title("🤖 AI 智能選股 - 板塊 × 個股雙重評分")

# 返回首頁按鈕
if st.button("⬅️ 返回首頁", key="back_home"):
    st.switch_page("app.py")

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 選股設定")
    
    analysis_days = st.slider("分析週期(天)", 10, 60, 20, step=5)
    
    st.divider()
    st.subheader("📊 板塊評分權重")
    sector_weight_momentum = st.slider("板塊動能", 0, 100, 40, step=5, key="sw_momentum")
    sector_weight_capital = st.slider("板塊資金", 0, 100, 40, step=5, key="sw_capital")
    sector_weight_sync = st.slider("板塊同步", 0, 100, 20, step=5, key="sw_sync")
    
    st.divider()
    st.subheader("🎯 個股評分權重")
    stock_weight_trend = st.slider("趨勢濾網", 0, 100, 40, step=5, key="stw_trend")
    stock_weight_kd = st.slider("KD 時機", 0, 100, 30, step=5, key="stw_kd")
    stock_weight_macd = st.slider("MACD 動能", 0, 100, 30, step=5, key="stw_macd")
    
    st.divider()
    st.subheader("🔍 篩選條件")
    min_sector_score = st.slider("最低板塊評分", 0, 100, 60, step=5)
    min_stock_score = st.slider("最低個股評分", 0, 100, 70, step=5)
    max_recommendations = st.slider("推薦數量", 3, 20, 10, step=1)
    
    st.divider()
    if st.button("🔄 重新整理資料", type="primary"):
        st.cache_data.clear()
        st.rerun()

# --- 進階快取：單一股票資料載入 ---
@st.cache_data(ttl=7200, show_spinner=False)  # 2小時快取
def load_single_stock_data(stock_id, stock_name, sector_name, start_date, end_date, days):
    """載入單一股票的完整數據"""
    try:
        dl = DataLoader()
        dl.login_by_token(api_token=FINMIND_API_TOKEN)
        
        # 股價資料
        df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)
        if df.empty:
            return None
        
        df = df.rename(columns={
            'date': 'Date', 'open': 'Open', 'max': 'High', 
            'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'
        })
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 籌碼資料
        try:
            chips = dl.taiwan_stock_institutional_investors(
                stock_id=stock_id, start_date=start_date, end_date=end_date
            )
            if not chips.empty:
                chips['bs'] = chips['buy'] - chips['sell']
                chips = chips.pivot(index='date', columns='name', values='bs').reset_index()
                chips = chips.rename(columns={'date': 'Date'})
                chips['Date'] = pd.to_datetime(chips['Date'])
                df = pd.merge(df, chips, on='Date', how='left')
                for col in ['Foreign_Investor', 'Investment_Trust']:
                    if col in df.columns:
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = 0
            else:
                df['Foreign_Investor'] = 0
                df['Investment_Trust'] = 0
        except:
            df['Foreign_Investor'] = 0
            df['Investment_Trust'] = 0
        
        # 計算技術指標
        df = calculate_indicators(df)
        df = df.tail(days).reset_index(drop=True)
        
        return {
            'stock_id': stock_id,
            'stock_name': stock_name,
            'sector': sector_name,
            'data': df
        }
        
    except Exception as e:
        return None

def calculate_indicators(df):
    """計算所有技術指標（獨立函數，便於維護）"""
    # 均線
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
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

@st.cache_data(ttl=7200, show_spinner=False)
def load_all_stocks_parallel(days):
    """批次載入所有股票資料（使用快取）"""
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=days+120)).strftime("%Y-%m-%d")
    
    all_stocks_info = []
    
    # 建立股票清單
    for sector_name, stocks_dict in stock_categories.items():
        for stock_id, stock_name_full in stocks_dict.items():
            stock_name = stock_name_full.split('-', 1)[1] if '-' in stock_name_full else stock_name_full
            all_stocks_info.append((stock_id, stock_name, sector_name))
    
    return all_stocks_info, start_date, end_date

def calculate_stock_score(df, w_trend, w_kd, w_macd):
    """計算個股評分"""
    if len(df) < 2:
        return 0, []
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    details = []
    
    # 1. 趨勢濾網 (MA60)
    trend_score = 0
    if pd.notna(last['MA60']) and pd.notna(last['Close']):
        if last['Close'] > last['MA60']:
            trend_score = 100
            details.append("✅ 多頭趨勢")
        elif last['Close'] > last['MA60'] * 0.97:
            trend_score = 50
            details.append("⚠️ 接近趨勢線")
        else:
            trend_score = 0
            details.append("❌ 空頭趨勢")
    
    # 2. KD 進場時機
    kd_score = 0
    k_val = last['K']
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
    
    # 3. MACD 動能
    macd_score = 0
    macd_val = last['MACD_Hist']
    prev_macd = prev['MACD_Hist']
    
    if macd_val > 0 and prev_macd <= 0:
        macd_score = 100
        details.append("🚀 MACD翻多")
    elif macd_val > 0:
        macd_score = 70
        details.append("📈 MACD偏多")
    elif macd_val > prev_macd:
        macd_score = 50
        details.append("🔄 MACD收斂")
    else:
        macd_score = 20
        details.append("📉 MACD偏空")
    
    # 加權計算
    total_weight = w_trend + w_kd + w_macd
    if total_weight > 0:
        score = (trend_score * w_trend + kd_score * w_kd + macd_score * w_macd) / total_weight
    else:
        score = 50
    
    return score, details

def calculate_sector_score(stocks_data, w_momentum, w_capital, w_sync):
    """計算板塊評分"""
    if not stocks_data:
        return 0, []
    
    returns = []
    momentums = []
    foreign_sum = 0
    trust_sum = 0
    
    for stock_info in stocks_data:
        df = stock_info['data']
        if len(df) < 2:
            continue
        
        # 報酬率
        ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
        returns.append(ret)
        
        # 動能
        last = df.iloc[-1]
        momentum = (
            (last['Close'] > last['MA5']) * 25 +
            (last['Close'] > last['MA20']) * 35 +
            (last['Close'] > last['MA60']) * 40
        )
        momentums.append(momentum)
        
        # 資金
        foreign_sum += df['Foreign_Investor'].sum()
        trust_sum += df['Investment_Trust'].sum()
    
    # 動能分數
    momentum_score = np.mean(momentums) if momentums else 0
    
    # 資金分數 (簡化正規化到 0-100)
    capital_score = min(100, max(0, (foreign_sum / 1000 + 50)))
    
    # 同步分數
    if len(returns) > 1:
        sync_score = (1 - (np.std(returns) / (np.mean(np.abs(returns)) + 1))) * 100
        sync_score = max(0, min(100, sync_score))
    else:
        sync_score = 50
    
    # 綜合評分
    total_weight = w_momentum + w_capital + w_sync
    if total_weight > 0:
        final_score = (
            momentum_score * w_momentum +
            capital_score * w_capital +
            sync_score * w_sync
        ) / total_weight
    else:
        final_score = 50
    
    details = [
        f"動能: {momentum_score:.0f}分",
        f"資金: {capital_score:.0f}分",
        f"同步: {sync_score:.0f}分"
    ]
    
    return final_score, details

# --- 載入資料（使用進度條）---
st.info("💡 提示：資料已快取 2 小時，若需要最新資料請點擊側邊欄的「重新整理資料」按鈕")

# 獲取股票清單
all_stocks_info, start_date, end_date = load_all_stocks_parallel(analysis_days)

# 建立進度條
progress_bar = st.progress(0)
status_text = st.empty()
total_stocks = len(all_stocks_info)

# 批次載入資料
all_stocks_data = []
failed_count = 0

for idx, (stock_id, stock_name, sector_name) in enumerate(all_stocks_info):
    # 更新進度
    progress = (idx + 1) / total_stocks
    progress_bar.progress(progress)
    status_text.text(f"載入中... {idx+1}/{total_stocks} - {stock_name} ({stock_id})")
    
    # 載入資料（快取會自動處理）
    stock_data = load_single_stock_data(
        stock_id, stock_name, sector_name, 
        start_date, end_date, analysis_days
    )
    
    if stock_data is not None:
        all_stocks_data.append(stock_data)
    else:
        failed_count += 1

# 清除進度條
progress_bar.empty()
status_text.empty()

if not all_stocks_data:
    st.error("❌ 無法載入資料，請稍後再試")
    st.stop()

if failed_count > 0:
    st.warning(f"⚠️ {failed_count} 檔股票載入失敗，已跳過")

# 計算評分（快取計算結果）
@st.cache_data(show_spinner=False)
def calculate_all_recommendations(stocks_data, sw_momentum, sw_capital, sw_sync, stw_trend, stw_kd, stw_macd):
    """計算所有推薦結果（快取）"""
    recommendations = []
    sector_cache = {}
    
    for stock_info in stocks_data:
        sector_name = stock_info['sector']
        
        # 計算板塊評分（快取）
        if sector_name not in sector_cache:
            sector_stocks = [s for s in stocks_data if s['sector'] == sector_name]
            sector_score, sector_details = calculate_sector_score(
                sector_stocks,
                sw_momentum,
                sw_capital,
                sw_sync
            )
            sector_cache[sector_name] = (sector_score, sector_details)
        else:
            sector_score, sector_details = sector_cache[sector_name]
        
        # 計算個股評分
        stock_score, stock_details = calculate_stock_score(
            stock_info['data'],
            stw_trend,
            stw_kd,
            stw_macd
        )
        
        # 綜合評分
        final_score = (sector_score * 0.4 + stock_score * 0.6)
        
        # 計算額外指標
        df = stock_info['data']
        if len(df) > 1:
            last = df.iloc[-1]
            price_change = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
            foreign_net = df['Foreign_Investor'].sum() / 1000
            trust_net = df['Investment_Trust'].sum() / 1000
            
            recommendations.append({
                'stock_id': stock_info['stock_id'],
                'stock_name': stock_info['stock_name'],
                'sector': sector_name,
                'sector_score': sector_score,
                'stock_score': stock_score,
                'final_score': final_score,
                'price': last['Close'],
                'price_change': price_change,
                'k_value': last['K'],
                'macd': last['MACD_Hist'],
                'foreign_net': foreign_net,
                'trust_net': trust_net,
                'stock_details': stock_details,
                'sector_details': sector_details
            })
    
    return pd.DataFrame(recommendations)

# 執行計算（使用快取）
with st.spinner('📊 計算評分中...'):
    recommendations_df = calculate_all_recommendations(
        all_stocks_data,
        sector_weight_momentum,
        sector_weight_capital,
        sector_weight_sync,
        stock_weight_trend,
        stock_weight_kd,
        stock_weight_macd
    )

# 篩選並排序
filtered_df = recommendations_df[
    (recommendations_df['sector_score'] >= min_sector_score) &
    (recommendations_df['stock_score'] >= min_stock_score)
].sort_values('final_score', ascending=False).head(max_recommendations)

# --- 顯示結果 ---
st.header(f"🎯 AI 推薦結果 (共 {len(filtered_df)} 檔)")

if len(filtered_df) == 0:
    st.warning("😢 沒有符合條件的推薦標的，請降低篩選條件")
else:
    # 關鍵指標
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        top_pick = filtered_df.iloc[0]
        st.metric("首選標的", f"{top_pick['stock_name']} ({top_pick['stock_id']})", 
                 f"{top_pick['final_score']:.0f}分")
    
    with col2:
        avg_sector = filtered_df['sector_score'].mean()
        st.metric("平均板塊分", f"{avg_sector:.0f}", 
                 "強勢" if avg_sector >= 70 else "中性")
    
    with col3:
        avg_stock = filtered_df['stock_score'].mean()
        st.metric("平均個股分", f"{avg_stock:.0f}",
                 "強勢" if avg_stock >= 70 else "中性")
    
    with col4:
        sectors_count = filtered_df['sector'].nunique()
        st.metric("涵蓋板塊", f"{sectors_count} 個")
    
    # 分頁顯示
    tab1, tab2, tab3 = st.tabs(["📋 推薦清單", "📊 評分分析", "💡 操作建議"])
    
    with tab1:
        st.subheader("📋 AI 推薦標的清單")
        
        # 顯示表格
        display_df = filtered_df[[
            'stock_id', 'stock_name', 'sector', 'final_score', 
            'sector_score', 'stock_score', 'price', 'price_change',
            'k_value', 'foreign_net', 'trust_net'
        ]].copy()
        
        display_df.columns = [
            '代號', '股票', '板塊', '綜合評分', 
            '板塊分', '個股分', '股價', '漲跌%',
            'KD值', '外資(千張)', '投信(千張)'
        ]
        
        display_df['綜合評分'] = display_df['綜合評分'].round(0)
        display_df['板塊分'] = display_df['板塊分'].round(0)
        display_df['個股分'] = display_df['個股分'].round(0)
        display_df['股價'] = display_df['股價'].round(2)
        display_df['漲跌%'] = display_df['漲跌%'].round(2)
        display_df['KD值'] = display_df['KD值'].round(0)
        display_df['外資(千張)'] = display_df['外資(千張)'].round(0)
        display_df['投信(千張)'] = display_df['投信(千張)'].round(0)
        
        st.dataframe(display_df, width='stretch', hide_index=True)
    
    with tab2:
        st.subheader("📊 評分分布分析")
        
        # 評分散布圖
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=filtered_df['sector_score'],
            y=filtered_df['stock_score'],
            mode='markers+text',
            marker=dict(
                size=filtered_df['final_score'] / 5,
                color=filtered_df['final_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="綜合評分")
            ),
            text=filtered_df['stock_name'],
            textposition='top center'
        ))
        
        fig_scatter.update_layout(
            title="個股評分 vs 板塊評分分布圖",
            xaxis_title="板塊評分",
            yaxis_title="個股評分",
            height=500
        )
        
        st.plotly_chart(fig_scatter, width='stretch')
        
        # 板塊分布
        st.divider()
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write("**板塊分布**")
            sector_count = filtered_df['sector'].value_counts()
            for sector, count in sector_count.items():
                st.write(f"• {sector}: {count}檔")
        
        with col_b:
            st.write("**評分區間分布**")
            high_score = len(filtered_df[filtered_df['final_score'] >= 80])
            mid_score = len(filtered_df[(filtered_df['final_score'] >= 70) & (filtered_df['final_score'] < 80)])
            low_score = len(filtered_df[filtered_df['final_score'] < 70])
            st.write(f"• 80分以上: {high_score}檔")
            st.write(f"• 70-79分: {mid_score}檔")
            st.write(f"• 70分以下: {low_score}檔")
    
    with tab3:
        st.subheader("💡 Top 5 詳細操作建議")
        
        top5 = filtered_df.head(5)
        
        for idx, row in top5.iterrows():
            with st.expander(f"#{idx+1} {row['stock_name']} ({row['stock_id']}) - 綜合評分: {row['final_score']:.0f}"):
                col1, col2, col3 = st.columns(3)
                
                col1.metric("股價", f"{row['price']:.2f}")
                col2.metric(f"{analysis_days}日漲跌", f"{row['price_change']:.2f}%")
                col3.metric("KD值", f"{row['k_value']:.0f}")
                
                st.write(f"**所屬板塊**: {row['sector']} (評分: {row['sector_score']:.0f})")
                
                st.write("**個股評分細節**:")
                for detail in row['stock_details']:
                    st.write(f"• {detail}")
                
                st.write("**板塊評分細節**:")
                for detail in row['sector_details']:
                    st.write(f"• {detail}")
                
                st.write(f"**籌碼**: 外資 {row['foreign_net']:.0f}千張 | 投信 {row['trust_net']:.0f}千張")
                
                # 操作建議
                if row['final_score'] >= 80:
                    st.success("""
                    **🎯 強力買進訊號**
                    - 建議進場價: 當前價
                    - 停損: -8%
                    - 停利: +15%
                    - 策略: 短線或波段持有
                    """)
                elif row['final_score'] >= 70:
                    st.info("""
                    **📊 可逢低布局**
                    - 建議進場價: 回檔 2-3% 再進
                    - 停損: -10%
                    - 停利: +12%
                    - 策略: 波段持有
                    """)
                else:
                    st.warning("""
                    **⚠️ 觀望為主**
                    - 建議等待更佳時機
                    - 持續追蹤評分變化
                    """)

st.divider()
st.caption("⚠️ AI 選股基於量化評分，不保證獲利。投資有風險，請審慎評估。")
