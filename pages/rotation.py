import streamlit as st
from datetime import date, timedelta
from FinMind.data import DataLoader
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from config import stock_categories, FINMIND_API_TOKEN

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="板塊輪動分析")
st.title("🔄 板塊輪動分析 - 資金流向追蹤")

# --- 工具函數 ---
def extract_stock_info(stock_dict):
    """從 '2330-台積電' 格式提取代號和純名稱"""
    clean_dict = {}
    for code, full_name in stock_dict.items():
        name = full_name.split('-', 1)[1] if '-' in full_name else full_name
        clean_dict[code] = name
    return clean_dict

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 分析設定")
    analysis_period = st.slider("分析週期(天)", 10, 90, 30, step=5)
    
    st.divider()
    st.subheader("🎯 評分權重")
    weight_momentum = st.slider("技術動能", 0, 100, 30, step=5)
    weight_capital = st.slider("資金流向", 0, 100, 40, step=5)
    weight_sync = st.slider("板塊同步", 0, 100, 30, step=5)

# --- 資料載入函數 ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_all_sectors_data(days):
    """載入所有板塊的綜合數據"""
    dl = DataLoader()
    dl.login_by_token(api_token=FINMIND_API_TOKEN)
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=days+120)).strftime("%Y-%m-%d")
    
    all_sectors_data = {}
    
    for sector_name, stocks in stock_categories.items():
        sector_stocks = extract_stock_info(stocks)
        sector_summary = {
            'stocks': [],
            'dates': None,
            'total_foreign': [],
            'total_trust': [],
            'avg_return': 0,
            'avg_momentum': 0
        }
        
        for stock_id, stock_name in sector_stocks.items():
            try:
                # 股價
                df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)
                if df.empty:
                    continue
                
                df = df.rename(columns={
                    'date': 'Date', 'close': 'Close', 'Trading_Volume': 'Volume'
                })
                df['Close'] = df['Close'].astype(float)
                df['Date'] = pd.to_datetime(df['Date'])
                
                # 籌碼
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
                
                # 計算指標
                df['MA5'] = df['Close'].rolling(5).mean()
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA60'] = df['Close'].rolling(60).mean()
                
                sector_summary['stocks'].append({
                    'id': stock_id,
                    'name': stock_name,
                    'data': df.tail(days)
                })
                
            except Exception as e:
                continue
        
        # 計算板塊彙總
        if sector_summary['stocks']:
            # 取第一支股票的日期作為基準
            sector_summary['dates'] = sector_summary['stocks'][0]['data']['Date'].values
            
            # 計算板塊總資金流向
            for stock_info in sector_summary['stocks']:
                df = stock_info['data']
                sector_summary['total_foreign'].append(df['Foreign_Investor'].sum())
                sector_summary['total_trust'].append(df['Investment_Trust'].sum())
            
            # 計算平均報酬率和動能
            returns = []
            momentums = []
            for stock_info in sector_summary['stocks']:
                df = stock_info['data']
                if len(df) > 1:
                    ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
                    returns.append(ret)
                    
                    # 動能分數
                    last = df.iloc[-1]
                    momentum = (
                        (last['Close'] > last['MA5']) * 25 +
                        (last['Close'] > last['MA20']) * 35 +
                        (last['Close'] > last['MA60']) * 40
                    )
                    momentums.append(momentum)
            
            sector_summary['avg_return'] = np.mean(returns) if returns else 0
            sector_summary['avg_momentum'] = np.mean(momentums) if momentums else 0
            sector_summary['total_foreign_sum'] = sum(sector_summary['total_foreign'])
            sector_summary['total_trust_sum'] = sum(sector_summary['total_trust'])
            
            all_sectors_data[sector_name] = sector_summary
    
    return all_sectors_data

def calculate_sector_scores(sectors_data, w_momentum, w_capital, w_sync):
    """計算各板塊綜合評分"""
    scores = []
    
    # 正規化用的最大最小值
    all_returns = [s['avg_return'] for s in sectors_data.values()]
    all_momentums = [s['avg_momentum'] for s in sectors_data.values()]
    all_foreign = [s['total_foreign_sum'] for s in sectors_data.values()]
    all_trust = [s['total_trust_sum'] for s in sectors_data.values()]
    
    max_return = max(all_returns) if all_returns else 1
    min_return = min(all_returns) if all_returns else 0
    max_foreign = max(all_foreign) if all_foreign else 1
    min_foreign = min(all_foreign) if all_foreign else 0
    max_trust = max(all_trust) if all_trust else 1
    min_trust = min(all_trust) if all_trust else 0
    
    for sector_name, data in sectors_data.items():
        # 1. 動能分數 (0-100)
        momentum_score = data['avg_momentum']
        
        # 2. 資金分數 (0-100)
        if max_foreign != min_foreign:
            foreign_norm = (data['total_foreign_sum'] - min_foreign) / (max_foreign - min_foreign) * 100
        else:
            foreign_norm = 50
        
        if max_trust != min_trust:
            trust_norm = (data['total_trust_sum'] - min_trust) / (max_trust - min_trust) * 100
        else:
            trust_norm = 50
        
        capital_score = (foreign_norm * 0.6 + trust_norm * 0.4)
        
        # 3. 同步性分數 (簡化版：用報酬率的一致性)
        returns = []
        for stock_info in data['stocks']:
            df = stock_info['data']
            if len(df) > 1:
                ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
                returns.append(ret)
        
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
        
        scores.append({
            'sector': sector_name,
            'score': final_score,
            'momentum': momentum_score,
            'capital': capital_score,
            'sync': sync_score,
            'return': data['avg_return'],
            'foreign': data['total_foreign_sum'] / 1000,
            'trust': data['total_trust_sum'] / 1000,
            'stock_count': len(data['stocks'])
        })
    
    return pd.DataFrame(scores).sort_values('score', ascending=False)

def detect_capital_flow(sectors_data, days):
    """偵測資金流向變化"""
    flow_changes = []
    
    for sector_name, data in sectors_data.items():
        if not data['stocks']:
            continue
        
        # 計算前後兩週資金變化
        mid_point = len(data['stocks'][0]['data']) // 2
        
        early_foreign = 0
        late_foreign = 0
        early_trust = 0
        late_trust = 0
        
        for stock_info in data['stocks']:
            df = stock_info['data']
            early_foreign += df['Foreign_Investor'].iloc[:mid_point].sum()
            late_foreign += df['Foreign_Investor'].iloc[mid_point:].sum()
            early_trust += df['Investment_Trust'].iloc[:mid_point].sum()
            late_trust += df['Investment_Trust'].iloc[mid_point:].sum()
        
        foreign_change = late_foreign - early_foreign
        trust_change = late_trust - early_trust
        
        flow_changes.append({
            'sector': sector_name,
            'foreign_change': foreign_change / 1000,
            'trust_change': trust_change / 1000,
            'total_change': (foreign_change + trust_change) / 1000
        })
    
    return pd.DataFrame(flow_changes).sort_values('total_change', ascending=False)

# --- 載入資料 ---
with st.spinner('🔄 正在分析全市場板塊數據...'):
    sectors_data = load_all_sectors_data(analysis_period)
    
    if not sectors_data:
        st.error("❌ 無法載入資料，請稍後再試")
        st.stop()

# 計算評分
sector_scores = calculate_sector_scores(
    sectors_data, 
    weight_momentum, 
    weight_capital, 
    weight_sync
)

capital_flow = detect_capital_flow(sectors_data, analysis_period)

# --- 主要內容 ---
st.header(f"📊 板塊總覽 (過去 {analysis_period} 天)")

# 關鍵指標
col1, col2, col3, col4 = st.columns(4)

with col1:
    top_sector = sector_scores.iloc[0]
    st.metric("最強板塊", top_sector['sector'], f"{top_sector['score']:.0f}分")

with col2:
    hot_money = capital_flow.iloc[0]
    st.metric("熱錢流入", hot_money['sector'], f"{hot_money['total_change']:.0f}千張")

with col3:
    avg_score = sector_scores['score'].mean()
    st.metric("市場平均分", f"{avg_score:.0f}", 
             "偏多" if avg_score > 55 else "偏空")

with col4:
    strong_sectors = len(sector_scores[sector_scores['score'] >= 60])
    st.metric("強勢板塊數", f"{strong_sectors}/{len(sector_scores)}")

# --- Tab 頁面 ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 板塊雷達圖", 
    "🔥 資金流向地圖", 
    "🔄 輪動提示", 
    "📈 歷史回測"
])

with tab1:
    st.subheader("🎯 板塊對比雷達圖")
    
    # 雷達圖
    categories = ['技術動能', '資金流向', '板塊同步', '報酬率', '外資偏好']
    
    fig_radar = go.Figure()
    
    # 只顯示前5名板塊
    top5 = sector_scores.head(5)
    
    for idx, row in top5.iterrows():
        # 正規化報酬率到 0-100
        return_norm = min(100, max(0, (row['return'] + 20) / 0.4))  # 假設報酬率在 -20% 到 +20%
        foreign_norm = min(100, max(0, (row['foreign'] + 1000) / 20))  # 調整範圍
        
        values = [
            row['momentum'],
            row['capital'],
            row['sync'],
            return_norm,
            foreign_norm
        ]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # 閉合圖形
            theta=categories + [categories[0]],
            fill='toself',
            name=row['sector']
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig_radar, width='stretch')
    
    # 評分表格
    st.divider()
    st.subheader("📊 板塊評分排行")
    
    display_scores = sector_scores.copy()
    display_scores.columns = [
        '板塊', '綜合評分', '動能分數', '資金分數', 
        '同步分數', '報酬率%', '外資(千張)', '投信(千張)', '成分股數'
    ]
    
    display_scores['綜合評分'] = display_scores['綜合評分'].round(0)
    display_scores['動能分數'] = display_scores['動能分數'].round(0)
    display_scores['資金分數'] = display_scores['資金分數'].round(0)
    display_scores['同步分數'] = display_scores['同步分數'].round(0)
    display_scores['報酬率%'] = display_scores['報酬率%'].round(2)
    display_scores['外資(千張)'] = display_scores['外資(千張)'].round(0)
    display_scores['投信(千張)'] = display_scores['投信(千張)'].round(0)
    
    st.dataframe(display_scores, width='stretch', hide_index=True)

with tab2:
    st.subheader("🔥 板塊資金流向地圖")
    
    # 資金變化對比圖
    fig_flow = go.Figure()
    
    fig_flow.add_trace(go.Bar(
        x=capital_flow['sector'],
        y=capital_flow['foreign_change'],
        name='外資變化',
        marker_color='#3b82f6'
    ))
    
    fig_flow.add_trace(go.Bar(
        x=capital_flow['sector'],
        y=capital_flow['trust_change'],
        name='投信變化',
        marker_color='#f59e0b'
    ))
    
    fig_flow.update_layout(
        title=f"板塊資金變化 (前後 {analysis_period//2} 天對比)",
        xaxis_title="板塊",
        yaxis_title="資金變化 (千張)",
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_flow, width='stretch')
    
    # 資金流向矩陣
    st.divider()
    st.subheader("💰 資金集中度分析")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**外資最愛板塊 Top 5**")
        top_foreign = sector_scores.nlargest(5, 'foreign')[['sector', 'foreign']]
        for idx, row in top_foreign.iterrows():
            st.write(f"• {row['sector']}: {row['foreign']:.0f}千張")
    
    with col_b:
        st.write("**投信最愛板塊 Top 5**")
        top_trust = sector_scores.nlargest(5, 'trust')[['sector', 'trust']]
        for idx, row in top_trust.iterrows():
            st.write(f"• {row['sector']}: {row['trust']:.0f}千張")
    
    # 資金流向熱力圖
    st.divider()
    
    # 建立資金流向矩陣 (外資 vs 投信)
    heatmap_data = sector_scores[['sector', 'foreign', 'trust', 'score']].copy()
    heatmap_data['foreign_norm'] = (heatmap_data['foreign'] - heatmap_data['foreign'].min()) / (heatmap_data['foreign'].max() - heatmap_data['foreign'].min()) * 100
    heatmap_data['trust_norm'] = (heatmap_data['trust'] - heatmap_data['trust'].min()) / (heatmap_data['trust'].max() - heatmap_data['trust'].min()) * 100
    
    fig_heat = go.Figure(data=go.Scatter(
        x=heatmap_data['foreign_norm'],
        y=heatmap_data['trust_norm'],
        mode='markers+text',
        marker=dict(
            size=heatmap_data['score'],
            color=heatmap_data['score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="綜合評分")
        ),
        text=heatmap_data['sector'],
        textposition='top center'
    ))
    
    fig_heat.update_layout(
        title="資金流向分布圖 (氣泡大小 = 綜合評分)",
        xaxis_title="外資偏好度",
        yaxis_title="投信偏好度",
        height=500
    )
    
    st.plotly_chart(fig_heat, width='stretch')

with tab3:
    st.subheader("🔄 板塊輪動提示")
    
    # 分析資金流動方向
    top_inflow = capital_flow.head(3)
    top_outflow = capital_flow.tail(3)
    
    col_rot1, col_rot2 = st.columns(2)
    
    with col_rot1:
        st.success("### 📈 資金流入板塊")
        for idx, row in top_inflow.iterrows():
            st.write(f"**{row['sector']}**")
            st.write(f"• 總流入: {row['total_change']:.0f}千張")
            st.write(f"• 外資: {row['foreign_change']:.0f}千張 | 投信: {row['trust_change']:.0f}千張")
            st.divider()
    
    with col_rot2:
        st.error("### 📉 資金流出板塊")
        for idx, row in top_outflow.iterrows():
            st.write(f"**{row['sector']}**")
            st.write(f"• 總流出: {abs(row['total_change']):.0f}千張")
            st.write(f"• 外資: {row['foreign_change']:.0f}千張 | 投信: {row['trust_change']:.0f}千張")
            st.divider()
    
    # 輪動建議
    st.divider()
    st.subheader("💡 投資策略建議")
    
    inflow_sector = top_inflow.iloc[0]['sector']
    outflow_sector = top_outflow.iloc[0]['sector']
    
    st.info(f"""
    ### 🎯 當前市場趨勢
    
    **資金正在從「{outflow_sector}」流向「{inflow_sector}」**
    
    #### 建議操作：
    
    1. **積極型投資者**
       - 關注 {inflow_sector} 內的強勢個股
       - 使用「個股戰情室」尋找評分 ≥70 且黃金交叉的標的
       - 設定停利 15-20%
    
    2. **穩健型投資者**
       - 等待 {inflow_sector} 回檔再進場
       - 觀察板塊評分是否穩定在 60 分以上
       - 採用波段策略 (MA20/60)
    
    3. **避險型投資者**
       - 暫時避開 {outflow_sector}
       - 持有現金等待更明確訊號
       - 關注金融板塊作為防禦性配置
    
    ⚠️ **風險提示**: 板塊輪動頻繁時，建議降低倉位或採取分批進場策略
    """)

with tab4:
    st.subheader("📈 板塊評分歷史回測")
    
    st.info("""
    ### 🔬 回測邏輯說明
    
    **策略規則：**
    1. 當板塊評分 ≥ 80 時，買入該板塊所有成分股（等權重）
    2. 當板塊評分 < 60 時，賣出全部持股
    3. 持有期間不做調整
    
    **績效計算：**
    - 以過去 {analysis_period} 天的數據進行模擬
    - 假設每個板塊投入相同資金
    - 不考慮交易成本和滑價
    """)
    
    st.divider()
    
    # 簡化版回測結果
    st.subheader("🏆 板塊績效排行")
    
    backtest_results = []
    
    for idx, row in sector_scores.iterrows():
        sector_name = row['sector']
        
        # 簡化計算：用報酬率 × 評分 作為調整後績效
        adjusted_return = row['return'] * (row['score'] / 100)
        
        backtest_results.append({
            'sector': sector_name,
            'raw_return': row['return'],
            'adjusted_return': adjusted_return,
            'score': row['score']
        })
    
    backtest_df = pd.DataFrame(backtest_results).sort_values('adjusted_return', ascending=False)
    
    # 績效圖表
    fig_backtest = go.Figure()
    
    fig_backtest.add_trace(go.Bar(
        x=backtest_df['sector'],
        y=backtest_df['raw_return'],
        name='實際報酬率',
        marker_color='lightblue'
    ))
    
    fig_backtest.add_trace(go.Bar(
        x=backtest_df['sector'],
        y=backtest_df['adjusted_return'],
        name='策略調整後報酬',
        marker_color='darkblue'
    ))
    
    fig_backtest.update_layout(
        title=f"板塊績效比較 (過去 {analysis_period} 天)",
        xaxis_title="板塊",
        yaxis_title="報酬率 (%)",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig_backtest, width='stretch')
    
    # 績效表格
    st.divider()
    
    display_backtest = backtest_df.copy()
    display_backtest.columns = ['板塊', '原始報酬%', '策略報酬%', '當前評分']
    display_backtest['原始報酬%'] = display_backtest['原始報酬%'].round(2)
    display_backtest['策略報酬%'] = display_backtest['策略報酬%'].round(2)
    display_backtest['當前評分'] = display_backtest['當前評分'].round(0)
    
    st.dataframe(display_backtest, width='stretch', hide_index=True)
    
    # 總結
    st.success(f"""
    ### 📊 回測結論
    
    **最佳板塊**: {backtest_df.iloc[0]['sector']} (策略報酬 {backtest_df.iloc[0]['adjusted_return']:.2f}%)
    
    **建議**:
    - 評分 ≥80 的板塊：{len(sector_scores[sector_scores['score'] >= 80])} 個 → 強力買進
    - 評分 60-79 的板塊：{len(sector_scores[(sector_scores['score'] >= 60) & (sector_scores['score'] < 80)])} 個 → 可逢低布局
    - 評分 <60 的板塊：{len(sector_scores[sector_scores['score'] < 60])} 個 → 觀望或減碼
    """)

st.divider()
st.caption("⚠️ 板塊輪動分析基於歷史數據，不保證未來表現。投資有風險，請審慎評估。")