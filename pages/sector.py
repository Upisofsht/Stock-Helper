"""
板塊資金雷達頁面
將此檔案儲存為: pages/1_🎯_板塊資金雷達.py
"""

import streamlit as st
from datetime import date, timedelta
from FinMind.data import DataLoader
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from config import stock_categories, FINMIND_API_TOKEN

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="板塊資金雷達", initial_sidebar_state="collapsed")
st.title("🎯 板塊資金雷達 - 追蹤熱錢流向")

# 返回首頁按鈕
if st.button("⬅️ 返回首頁", key="back_home"):
    st.switch_page("app.py")

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 板塊設定")
    selected_sector = st.selectbox("選擇板塊", list(stock_categories.keys()))
    analysis_days = st.slider("分析天數", 5, 60, 20, step=5)
    
    st.divider()
    st.subheader("📊 顯示選項")
    show_chips = st.checkbox("顯示籌碼熱力圖", value=True)
    show_momentum = st.checkbox("顯示動能排行", value=True)
    show_correlation = st.checkbox("顯示板塊同步性", value=True)

# --- 資料載入函數 ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_sector_data(sector_stocks, days):
    """載入整個板塊的資料"""
    dl = DataLoader()
    dl.login_by_token(api_token=FINMIND_API_TOKEN)
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=days+120)).strftime("%Y-%m-%d")
    
    sector_data = {}
    
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
                    
                    for col in ['Foreign_Investor', 'Investment_Trust', 'Dealer']:
                        if col not in df.columns:
                            df[col] = 0
                        else:
                            df[col] = df[col].fillna(0)
                else:
                    df['Foreign_Investor'] = 0
                    df['Investment_Trust'] = 0
                    df['Dealer'] = 0
            except:
                df['Foreign_Investor'] = 0
                df['Investment_Trust'] = 0
                df['Dealer'] = 0
            
            # 計算技術指標
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA60'] = df['Close'].rolling(60).mean()
            df['Returns'] = df['Close'].pct_change() * 100
            
            # 計算動能分數
            df['Momentum_Score'] = (
                (df['Close'] > df['MA5']).astype(int) * 25 +
                (df['Close'] > df['MA20']).astype(int) * 35 +
                (df['Close'] > df['MA60']).astype(int) * 40
            )
            
            sector_data[stock_id] = {
                'name': stock_name,
                'data': df
            }
            
        except Exception as e:
            st.warning(f"⚠️ {stock_name} 資料載入失敗: {str(e)}")
            continue
    
    return sector_data

def analyze_sector_momentum(sector_data, days):
    """分析板塊整體動能"""
    momentum_summary = []
    
    for stock_id, info in sector_data.items():
        df = info['data'].tail(days)
        
        if len(df) < days:
            continue
        
        last = df.iloc[-1]
        first = df.iloc[0]
        
        # 計算各項指標
        price_change = (last['Close'] - first['Close']) / first['Close'] * 100
        avg_volume = df['Volume'].mean()
        recent_volume = df.tail(5)['Volume'].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # 籌碼分析
        foreign_net = df['Foreign_Investor'].sum() / 1000
        trust_net = df['Investment_Trust'].sum() / 1000
        
        momentum_summary.append({
            'stock_id': stock_id,
            'name': info['name'],
            'price_change': price_change,
            'current_price': last['Close'],
            'momentum_score': last['Momentum_Score'],
            'volume_ratio': volume_ratio,
            'foreign_net': foreign_net,
            'trust_net': trust_net,
            'ma5': last['MA5'],
            'ma20': last['MA20'],
            'ma60': last['MA60']
        })
    
    return pd.DataFrame(momentum_summary)

def calculate_sector_capital_flow(sector_data, days):
    """計算板塊資金流向"""
    all_dates = None
    foreign_flow = {}
    trust_flow = {}
    dealer_flow = {}
    
    for stock_id, info in sector_data.items():
        df = info['data'].tail(days)
        
        if all_dates is None:
            all_dates = df['Date'].values
        
        foreign_flow[info['name']] = df['Foreign_Investor'].values
        trust_flow[info['name']] = df['Investment_Trust'].values
        dealer_flow[info['name']] = df.get('Dealer', pd.Series([0]*len(df))).values
    
    # 轉換為DataFrame
    foreign_df = pd.DataFrame(foreign_flow, index=all_dates)
    trust_df = pd.DataFrame(trust_flow, index=all_dates)
    dealer_df = pd.DataFrame(dealer_flow, index=all_dates)
    
    # 計算每日總和
    sector_flow = pd.DataFrame({
        'Date': all_dates,
        'Foreign': foreign_df.sum(axis=1),
        'Trust': trust_df.sum(axis=1),
        'Dealer': dealer_df.sum(axis=1)
    })
    
    return sector_flow, foreign_df, trust_df

def calculate_sector_correlation(sector_data, days):
    """計算板塊內股票相關性（同步性）"""
    returns_dict = {}
    
    for stock_id, info in sector_data.items():
        df = info['data'].tail(days)
        if len(df) > 0:
            returns_dict[info['name']] = df['Returns'].values
    
    returns_df = pd.DataFrame(returns_dict)
    correlation_matrix = returns_df.corr()
    
    # 計算平均相關性（板塊同步性指標）
    avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
    
    return correlation_matrix, avg_correlation

# --- 載入資料 ---
with st.spinner('🔄 正在分析板塊資金流向...'):
    sector_stocks = stock_categories[selected_sector]
    sector_data = load_sector_data(sector_stocks, analysis_days)
    
    if not sector_data:
        st.error("❌ 無法載入板塊資料，請稍後再試")
        st.stop()

# --- 板塊概況儀表板 ---
st.header(f"{selected_sector} 板塊總覽")

momentum_df = analyze_sector_momentum(sector_data, analysis_days)
sector_flow, foreign_detail, trust_detail = calculate_sector_capital_flow(sector_data, analysis_days)

# 關鍵指標
col1, col2, col3, col4, col5 = st.columns(5)

avg_return = momentum_df['price_change'].mean()
strong_stocks = len(momentum_df[momentum_df['momentum_score'] >= 70])
total_foreign = sector_flow['Foreign'].sum() / 1000
total_trust = sector_flow['Trust'].sum() / 1000
avg_volume_ratio = momentum_df['volume_ratio'].mean()

with col1:
    st.metric("平均漲跌幅", f"{avg_return:.2f}%", 
             delta_color="normal" if avg_return > 0 else "inverse")

with col2:
    st.metric("強勢股數量", f"{strong_stocks}/{len(momentum_df)}")

with col3:
    color = "normal" if total_foreign > 0 else "inverse"
    st.metric("外資淨買(張)", f"{total_foreign:.0f}", delta_color=color)

with col4:
    color = "normal" if total_trust > 0 else "inverse"
    st.metric("投信淨買(張)", f"{total_trust:.0f}", delta_color=color)

with col5:
    st.metric("量能比", f"{avg_volume_ratio:.2f}x")

# --- 板塊資金流向圖 ---
st.divider()
st.subheader("💰 板塊資金流向趨勢")

fig_flow = go.Figure()

fig_flow.add_trace(go.Bar(
    x=sector_flow['Date'],
    y=sector_flow['Foreign'] / 1000,
    name='外資',
    marker_color='#3b82f6'
))

fig_flow.add_trace(go.Bar(
    x=sector_flow['Date'],
    y=sector_flow['Trust'] / 1000,
    name='投信',
    marker_color='#f59e0b'
))

fig_flow.add_trace(go.Bar(
    x=sector_flow['Date'],
    y=sector_flow['Dealer'] / 1000,
    name='自營商',
    marker_color='#8b5cf6'
))

fig_flow.update_layout(
    barmode='group',
    height=400,
    xaxis_title="日期",
    yaxis_title="淨買賣 (千張)",
    hovermode='x unified',
    margin=dict(l=10, r=10, t=10, b=10)
)

st.plotly_chart(fig_flow, width='stretch')

# 資金流向解讀
recent_foreign = sector_flow.tail(5)['Foreign'].sum() / 1000
recent_trust = sector_flow.tail(5)['Trust'].sum() / 1000

col_a, col_b = st.columns(2)

with col_a:
    if recent_foreign > 1000:
        st.success(f"✅ 外資近5日大幅買超 {recent_foreign:.0f}千張，板塊資金強勢流入")
    elif recent_foreign > 0:
        st.info(f"📊 外資近5日買超 {recent_foreign:.0f}千張，持續看好")
    else:
        st.warning(f"⚠️ 外資近5日賣超 {abs(recent_foreign):.0f}千張，資金流出")

with col_b:
    if recent_trust > 500:
        st.success(f"✅ 投信近5日大幅買超 {recent_trust:.0f}千張，可能有主力拉抬")
    elif recent_trust > 0:
        st.info(f"📊 投信近5日買超 {recent_trust:.0f}千張")
    else:
        st.warning(f"⚠️ 投信近5日賣超 {abs(recent_trust):.0f}千張")

# --- 分頁內容 ---
tab1, tab2, tab3 = st.tabs(["🎯 個股動能排行", "🔥 籌碼熱力圖", "🔗 板塊同步性"])

with tab1:
    st.subheader("📊 個股動能與資金排行")
    
    # 排序選項
    sort_col = st.selectbox("排序依據", 
                           ["動能分數", "漲跌幅", "外資買超", "投信買超", "量能比"],
                           key="sort_momentum")
    
    sort_map = {
        "動能分數": "momentum_score",
        "漲跌幅": "price_change",
        "外資買超": "foreign_net",
        "投信買超": "trust_net",
        "量能比": "volume_ratio"
    }
    
    sorted_df = momentum_df.sort_values(sort_map[sort_col], ascending=False)
    
    # 顯示表格
    display_df = sorted_df[[
        'stock_id', 'name', 'current_price', 'price_change', 'momentum_score',
        'volume_ratio', 'foreign_net', 'trust_net'
    ]].copy()
    
    display_df.columns = ['代號', '股票', '股價', '漲跌%', '動能分數', 
                         '量能比', '外資(千張)', '投信(千張)']
    
    display_df['股價'] = display_df['股價'].round(2)
    display_df['漲跌%'] = display_df['漲跌%'].round(2)
    display_df['動能分數'] = display_df['動能分數'].round(0)
    display_df['量能比'] = display_df['量能比'].round(2)
    display_df['外資(千張)'] = display_df['外資(千張)'].round(0)
    display_df['投信(千張)'] = display_df['投信(千張)'].round(0)
    
    st.dataframe(display_df, width='stretch', hide_index=True)
    
    # 動能分布圖
    st.divider()
    fig_momentum = go.Figure()
    
    colors = ['#22c55e' if x >= 70 else '#f59e0b' if x >= 40 else '#ef4444' 
             for x in sorted_df['momentum_score']]
    
    fig_momentum.add_trace(go.Bar(
        x=sorted_df['name'],
        y=sorted_df['momentum_score'],
        marker_color=colors,
        text=sorted_df['momentum_score'].round(0),
        textposition='outside'
    ))
    
    fig_momentum.update_layout(
        title="個股動能分數分布",
        xaxis_title="股票",
        yaxis_title="動能分數",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_momentum, width='stretch')

with tab2:
    if show_chips:
        st.subheader("🔥 外資 vs 投信籌碼熱力圖")
        
        col_heat1, col_heat2 = st.columns(2)
        
        with col_heat1:
            st.write("**外資買賣分布**")
            
            # 外資熱力圖
            fig_foreign = go.Figure(data=go.Heatmap(
                z=foreign_detail.T.values,
                x=pd.to_datetime(foreign_detail.index).strftime('%m/%d'),
                y=foreign_detail.columns,
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(foreign_detail.T.values / 1000, 0),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="張數")
            ))
            
            fig_foreign.update_layout(
                height=400,
                xaxis_title="日期",
                yaxis_title="股票",
                margin=dict(l=10, r=10, t=10, b=10)
            )
            
            st.plotly_chart(fig_foreign, width='stretch')
        
        with col_heat2:
            st.write("**投信買賣分布**")
            
            # 投信熱力圖
            fig_trust = go.Figure(data=go.Heatmap(
                z=trust_detail.T.values,
                x=pd.to_datetime(trust_detail.index).strftime('%m/%d'),
                y=trust_detail.columns,
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(trust_detail.T.values / 1000, 0),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="張數")
            ))
            
            fig_trust.update_layout(
                height=400,
                xaxis_title="日期",
                yaxis_title="股票",
                margin=dict(l=10, r=10, t=10, b=10)
            )
            
            st.plotly_chart(fig_trust, width='stretch')
        
        # 解讀
        st.info("""
        💡 **如何解讀熱力圖：**
        - 🟢 綠色 = 買超，顏色越深買越多
        - 🔴 紅色 = 賣超，顏色越深賣越多
        - ⚪ 白色 = 中性，無明顯買賣
        - 📊 觀察「成片綠色」= 板塊性資金流入
        - 📊 觀察「集中綠色」= 主力鎖定特定個股
        """)

with tab3:
    if show_correlation:
        st.subheader("🔗 板塊同步性分析")
        
        corr_matrix, avg_corr = calculate_sector_correlation(sector_data, analysis_days)
        
        # 同步性指標
        col_sync1, col_sync2, col_sync3 = st.columns(3)
        
        with col_sync1:
            st.metric("板塊同步性", f"{avg_corr:.2f}")
        
        with col_sync2:
            if avg_corr > 0.7:
                st.success("✅ 高度同步")
                sync_msg = "板塊內個股走勢一致，資金集中"
            elif avg_corr > 0.4:
                st.info("📊 中度同步")
                sync_msg = "板塊內有明顯領漲股"
            else:
                st.warning("⚠️ 低度同步")
                sync_msg = "板塊內各股分歧，缺乏主軸"
        
        with col_sync3:
            st.write(f"**{sync_msg}**")
        
        # 相關性熱力圖
        st.divider()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="相關係數")
        ))
        
        fig_corr.update_layout(
            title="個股報酬率相關性矩陣",
            height=500,
            xaxis_title="",
            yaxis_title="",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        st.plotly_chart(fig_corr, width='stretch')
        
        st.info("""
        💡 **同步性的意義：**
        - **高同步性 (>0.7)**：板塊有明確題材，資金集中流入/流出
        - **中同步性 (0.4-0.7)**：有領漲股帶動，但個股表現有差異
        - **低同步性 (<0.4)**：個股各自表現，沒有板塊輪動效應
        
        ⚠️ **實戰應用：**
        - 同步性↑ + 外資買超 = 強勢板塊，可追蹤領漲股
        - 同步性↓ + 資金分散 = 選股不選市，看個股基本面
        """)

# --- 板塊策略建議 ---
st.divider()
st.header("🎯 板塊投資策略建議")

# 綜合評分邏輯
score = 0
reasons = []

# 1. 資金面 (40分)
if recent_foreign > 1000 and recent_trust > 500:
    score += 40
    reasons.append("✅ 外資+投信雙買超，資金面極強")
elif recent_foreign > 0 and recent_trust > 0:
    score += 30
    reasons.append("✅ 三大法人同步買超")
elif recent_foreign > 0 or recent_trust > 0:
    score += 20
    reasons.append("⚠️ 資金面偏多但不一致")
else:
    reasons.append("❌ 資金面轉弱，法人賣超")

# 2. 技術面 (30分)
if strong_stocks >= len(momentum_df) * 0.7:
    score += 30
    reasons.append(f"✅ 強勢股佔比高 ({strong_stocks}/{len(momentum_df)})")
elif strong_stocks >= len(momentum_df) * 0.4:
    score += 20
    reasons.append(f"📊 部分個股技術面強勢")
else:
    reasons.append("❌ 多數個股技術面轉弱")

# 3. 板塊同步性 (30分)
if avg_corr > 0.7:
    score += 30
    reasons.append(f"✅ 板塊高度同步 ({avg_corr:.2f})，有輪動效應")
elif avg_corr > 0.4:
    score += 20
    reasons.append(f"📊 板塊中度同步，有領漲股")
else:
    reasons.append(f"⚠️ 板塊同步性低，個股分歧")

# 顯示建議
col_rec1, col_rec2 = st.columns([1, 2])

with col_rec1:
    if score >= 80:
        st.success(f"### 🚀 強力買進\n板塊評分: {score}/100")
    elif score >= 60:
        st.info(f"### 📊 可逢低布局\n板塊評分: {score}/100")
    elif score >= 40:
        st.warning(f"### ⚠️ 觀望為主\n板塊評分: {score}/100")
    else:
        st.error(f"### 🛑 建議迴避\n板塊評分: {score}/100")

with col_rec2:
    st.write("**評分依據：**")
    for reason in reasons:
        st.write(reason)

# 推薦個股
if score >= 60:
    st.divider()
    st.subheader("💎 板塊內推薦標的 (動能分數 Top 3)")
    
    top3 = momentum_df.nlargest(3, 'momentum_score')
    
    for idx, row in top3.iterrows():
        with st.expander(f"**{row['name']} ({row['stock_id']})** - 動能分數: {row['momentum_score']:.0f}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("股價", f"{row['current_price']:.2f}")
            col2.metric(f"{analysis_days}日漲跌", f"{row['price_change']:.2f}%")
            col3.metric("量能比", f"{row['volume_ratio']:.2f}x")
            
            st.write(f"外資淨買: {row['foreign_net']:.0f}千張 | 投信淨買: {row['trust_net']:.0f}千張")

st.divider()
st.caption("⚠️ 板塊分析僅供參考，實際交易請搭配個股戰情室進行精準進場。")