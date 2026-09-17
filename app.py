# ============================================
# 港股AI智能分析系統 v2.1
# Streamlit Cloud 部署優化版
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 頁面配置（必須在最前面）
# ============================================
st.set_page_config(
    page_title="港股AI智能分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS樣式
# ============================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .signal-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .signal-strong-buy { background: linear-gradient(135deg, #00b09b, #96c93d); }
    .signal-buy { background: linear-gradient(135deg, #56ab2f, #a8e063); }
    .signal-hold { background: linear-gradient(135deg, #f2994a, #f2c94c); }
    .signal-sell { background: linear-gradient(135deg, #eb3349, #f45c43); }
    .signal-strong-sell { background: linear-gradient(135deg, #c31432, #240b36); }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: bold;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# 延遲導入（避免啟動時報錯）
# ============================================
@st.cache_resource(show_spinner=False)
def load_libraries():
    """延遲載入重量級套件"""
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return yf, go, make_subplots


# ============================================
# 數據獲取（帶重試機制）
# ============================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(symbol, period):
    """獲取股票數據 - 帶快取和重試"""
    try:
        yf, _, _ = load_libraries()
        
        # 使用 Ticker 並加入重試
        for attempt in range(3):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, auto_adjust=True)
                
                if not df.empty:
                    break
            except Exception as e:
                if attempt == 2:
                    raise e
                continue
        
        if df.empty:
            return None
        
        # 計算技術指標
        df = calculate_indicators(df)
        return df
    
    except Exception as e:
        st.error(f"❌ 數據獲取失敗: {str(e)}")
        return None


def calculate_indicators(df):
    """計算技術指標"""
    df = df.copy()
    
    # 移動平均線
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # RSI（加入除零保護）
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)  # 填補缺失值
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 布林帶
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Mid'] - 2 * bb_std
    
    # ATR
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # 波動率
    df['Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    
    # 成交量均線
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    
    # 填補缺失值（不使用 dropna，避免數據過少）
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df


# ============================================
# AI 預測模型
# ============================================
def ai_predict(df, days=30):
    """AI股價預測 - 線性回歸 + 指數平滑"""
    recent = df['Close'].tail(90).values
    
    if len(recent) < 20:
        # 數據不足時使用簡單預測
        last = recent[-1]
        avg_change = np.diff(recent).mean() / last if len(recent) > 1 else 0
        return [max(last * (1 + avg_change) ** i, 0.01) for i in range(1, days + 1)]
    
    # 線性回歸
    x = np.arange(len(recent))
    x_mean = x.mean()
    y_mean = recent.mean()
    numerator = ((x - x_mean) * (recent - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    slope = numerator / denominator if denominator != 0 else 0
    intercept = y_mean - slope * x_mean
    
    linear = intercept + slope * (len(recent) + np.arange(1, days + 1))
    
    # 指數平滑
    alpha = 0.3
    smoothed = [recent[0]]
    for i in range(1, len(recent)):
        smoothed.append(alpha * recent[i] + (1 - alpha) * smoothed[-1])
    last_smoothed = smoothed[-1]
    
    # 動量
    momentum = (recent[-1] - recent[-20]) / recent[-20] if len(recent) >= 20 else 0
    
    # 組合預測
    predictions = []
    for i in range(days):
        weight = 1 - (i / days) * 0.6
        pred = linear[i] * weight + last_smoothed * (1 + momentum) * (1 - weight)
        predictions.append(max(pred, 0.01))
    
    return predictions


# ============================================
# 交易信號生成
# ============================================
def generate_signal(df, predictions):
    """綜合交易信號"""
    price = df['Close'].iloc[-1]
    pred_price = predictions[-1]
    expected_return = (pred_price - price) / price * 100
    
    rsi = df['RSI'].iloc[-1]
    ma5 = df['MA5'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_sig = df['MACD_Signal'].iloc[-1]
    
    bb_range = df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1]
    bb_pos = (price - df['BB_Lower'].iloc[-1]) / bb_range if bb_range > 0 else 0.5
    
    signals = []
    score = 0
    
    # RSI信號
    if rsi < 30:
        signals.append(("RSI超賣", "強力買入", 30)); score += 30
    elif rsi < 40:
        signals.append(("RSI偏低", "買入", 15)); score += 15
    elif rsi > 70:
        signals.append(("RSI超買", "強力賣出", -30)); score -= 30
    elif rsi > 60:
        signals.append(("RSI偏高", "減持", -15)); score -= 15
    else:
        signals.append(("RSI正常", "中性", 0))
    
    # 均線信號
    if ma5 > ma20:
        signals.append(("多頭排列", "看漲", 20)); score += 20
    else:
        signals.append(("空頭排列", "看跌", -20)); score -= 20
    
    # MACD信號
    if macd > macd_sig:
        signals.append(("MACD金叉", "看漲", 15)); score += 15
    else:
        signals.append(("MACD死叉", "看跌", -15)); score -= 15
    
    # 布林帶信號
    if bb_pos < 0.1:
        signals.append(("觸及下軌", "強力買入", 25)); score += 25
    elif bb_pos > 0.9:
        signals.append(("觸及上軌", "強力賣出", -25)); score -= 25
    else:
        signals.append(("布林帶中位", "中性", 0))
    
    # AI預測信號
    if expected_return > 10:
        signals.append((f"AI預測+{expected_return:.1f}%", "強力買入", 25)); score += 25
    elif expected_return > 3:
        signals.append((f"AI預測+{expected_return:.1f}%", "買入", 15)); score += 15
    elif expected_return < -10:
        signals.append((f"AI預測{expected_return:.1f}%", "強力賣出", -25)); score -= 25
    elif expected_return < -3:
        signals.append((f"AI預測{expected_return:.1f}%", "賣出", -15)); score -= 15
    else:
        signals.append((f"AI預測{expected_return:+.1f}%", "持有", 0))
    
    # 決策
    if score >= 50:
        action, css, emoji = "強烈買入", "signal-strong-buy", "🔥"
    elif score >= 20:
        action, css, emoji = "買入", "signal-buy", "📈"
    elif score >= -20:
        action, css, emoji = "持有觀望", "signal-hold", "✋"
    elif score >= -50:
        action, css, emoji = "賣出", "signal-sell", "📉"
    else:
        action, css, emoji = "強烈賣出", "signal-strong-sell", "⚠️"
    
    return {
        'action': action, 'score': score, 'signals': signals,
        'css': css, 'emoji': emoji,
        'expected_return': expected_return, 'predicted_price': pred_price
    }


# ============================================
# 回測引擎
# ============================================
def run_backtest(df, strategy_type, capital=100000):
    """執行策略回測"""
    df = df.copy()
    
    # 生成交易信號
    if strategy_type == 'sma':
        df['Buy'] = (df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1))
        df['Sell'] = (df['MA5'] < df['MA20']) & (df['MA5'].shift(1) >= df['MA20'].shift(1))
        name = "SMA交叉(5/20)"
    elif strategy_type == 'rsi':
        df['Buy'] = (df['RSI'] < 30) & (df['RSI'].shift(1) >= 30)
        df['Sell'] = (df['RSI'] > 70) & (df['RSI'].shift(1) <= 70)
        name = "RSI(30/70)"
    elif strategy_type == 'macd':
        df['Buy'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        df['Sell'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        name = "MACD金叉死叉"
    else:
        df['Buy'] = (df['Close'] <= df['BB_Lower']) & (df['Close'].shift(1) > df['BB_Lower'].shift(1))
        df['Sell'] = (df['Close'] >= df['BB_Upper']) & (df['Close'].shift(1) < df['BB_Upper'].shift(1))
        name = "布林帶軌道"
    
    # 執行交易
    cash = capital
    shares = 0
    buy_price = 0
    wins = losses = 0
    equity = []
    
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        
        if df['Buy'].iloc[i] and shares == 0:
            shares = int(cash * 0.95 / price)
            if shares > 0:
                cash -= shares * price
                buy_price = price
        
        elif df['Sell'].iloc[i] and shares > 0:
            cash += shares * price
            if price > buy_price:
                wins += 1
            else:
                losses += 1
            shares = 0
        
        equity.append(cash + shares * price)
    
    # 平倉
    if shares > 0:
        final_price = df['Close'].iloc[-1]
        cash += shares * final_price
        equity[-1] = cash
        if final_price > buy_price:
            wins += 1
        else:
            losses += 1
    
    # 績效計算
    total_return = (cash - capital) / capital * 100
    eq_series = pd.Series(equity)
    returns = eq_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    rolling_max = eq_series.expanding().max()
    max_dd = ((eq_series - rolling_max) / rolling_max * 100).min()
    
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'name': name, 'return': total_return, 'sharpe': sharpe,
        'max_dd': max_dd, 'trades': total_trades,
        'wins': wins, 'losses': losses, 'win_rate': win_rate,
        'equity': equity, 'final_capital': cash
    }


# ============================================
# 主程式
# ============================================
def main():
    _, go, make_subplots = load_libraries()
    
    # 標題
    st.markdown('<div class="main-title">📊 港股AI智能分析系統</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 側邊欄
    with st.sidebar:
        st.markdown("## ⚙️ 分析設定")
        
        stock_map = {
            "0700.HK": "騰訊控股",
            "9988.HK": "阿里巴巴",
            "0005.HK": "匯豐控股",
            "1299.HK": "友邦保險",
            "1810.HK": "小米集團",
            "3690.HK": "美團",
            "AAPL": "蘋果公司",
            "TSLA": "特斯拉",
            "NVDA": "英偉達",
        }
        
        symbol = st.selectbox(
            "選擇股票",
            list(stock_map.keys()),
            format_func=lambda x: f"{stock_map[x]} ({x})",
            key="sb_stock"
        )
        
        period = st.selectbox(
            "數據期間",
            ["6mo", "1y", "2y", "3y"],
            index=1,
            key="sb_period"
        )
        
        pred_days = st.slider(
            "AI預測天數",
            min_value=7, max_value=90, value=30, step=1,
            key="sb_days"
        )
        
        capital = st.number_input(
            "初始資金 (HKD)",
            min_value=10000, max_value=10000000, value=100000, step=10000,
            key="sb_capital"
        )
        
        st.markdown("---")
        run_btn = st.button("🚀 開始分析", type="primary", use_container_width=True, key="sb_run")
        
        st.markdown("---")
        st.caption("⚠️ 本系統僅供學習研究，不構成投資建議")
    
    # 主內容
    if run_btn:
        progress = st.progress(0, text="📊 正在獲取數據...")
        
        df = fetch_data(symbol, period)
        progress.progress(30, text="📈 計算技術指標...")
        
        if df is None or len(df) < 30:
            st.error("❌ 無法獲取足夠數據，請稍後重試或更換股票")
            progress.empty()
            return
        
        predictions = ai_predict(df, pred_days)
        progress.progress(50, text="🤖 AI 預測分析...")
        
        signal = generate_signal(df, predictions)
        progress.progress(70, text="💰 策略回測中...")
        
        backtest_results = [
            run_backtest(df, 'sma', capital),
            run_backtest(df, 'rsi', capital),
            run_backtest(df, 'macd', capital),
            run_backtest(df, 'bollinger', capital),
        ]
        
        progress.progress(100, text="✅ 完成！")
        progress.empty()
        
        # ==========================================
        # 市場概況
        # ==========================================
        st.markdown("### 📊 市場概況")
        
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = (current_price - prev_price) / prev_price * 100
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("當前價格", f"${current_price:.2f}", f"{change:+.2f}%")
        c2.metric("RSI(14)", f"{df['RSI'].iloc[-1]:.1f}")
        c3.metric("波動率", f"{df['Volatility'].iloc[-1]*100:.1f}%")
        c4.metric("ATR(14)", f"${df['ATR'].iloc[-1]:.2f}")
        vol_ratio = df['Volume'].iloc[-1] / df['Volume_MA'].iloc[-1] if df['Volume_MA'].iloc[-1] > 0 else 1
        c5.metric("成交量比", f"{vol_ratio:.2f}x")
        
        # ==========================================
        # AI交易建議
        # ==========================================
        st.markdown("---")
        st.markdown("### 🎯 AI 交易建議")
        
        st.markdown(f"""
        <div class="signal-card {signal['css']}">
            <h1 style="font-size: 4rem; margin: 0;">{signal['emoji']}</h1>
            <h2 style="margin: 0.5rem 0;">{signal['action']}</h2>
            <p style="font-size: 1.2rem; margin-top: 1rem;">
                信心指數: <b>{signal['score']}</b> / 100 &nbsp;|&nbsp;
                預期回報: <b>{signal['expected_return']:+.2f}%</b> &nbsp;|&nbsp;
                目標價: <b>${signal['predicted_price']:.2f}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔍 詳細信號分析", expanded=True):
            cols = st.columns(2)
            for i, (name, desc, score) in enumerate(signal['signals']):
                with cols[i % 2]:
                    if score > 0:
                        st.success(f"🟢 **{name}** → {desc} ({score:+d})")
                    elif score < 0:
                        st.error(f"🔴 **{name}** → {desc} ({score:+d})")
                    else:
                        st.info(f"⚪ **{name}** → {desc} (0)")
        
        # ==========================================
        # 圖表
        # ==========================================
        st.markdown("---")
        st.markdown("### 📈 技術分析圖表")
        
        tab1, tab2, tab3 = st.tabs(["📊 K線圖", "📉 RSI & MACD", "🤖 AI預測"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='K線'
            ))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA5'],
                                     name='MA5', line=dict(color='#FFA500', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'],
                                     name='MA20', line=dict(color='#00BFFF', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='布林上軌',
                                     line=dict(color='red', dash='dash', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='布林下軌',
                                     line=dict(color='green', dash='dash', width=1)))
            fig.update_layout(
                template="plotly_dark", height=550,
                xaxis_rangeslider_visible=False,
                title=f"{stock_map[symbol]} 股價走勢"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("RSI", "MACD"),
                                vertical_spacing=0.1)
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'],
                                     name='RSI', line=dict(color='purple')), row=1, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],
                                     name='MACD', line=dict(color='#00BFFF')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'],
                                     name='Signal', line=dict(color='orange')), row=2, col=1)
            fig.update_layout(template="plotly_dark", height=550)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = go.Figure()
            hist_df = df.tail(90)
            fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'],
                                     name='歷史股價',
                                     line=dict(color='#00BFFF', width=2)))
            
            future_dates = pd.date_range(
                start=df.index[-1] + timedelta(days=1),
                periods=pred_days, freq='D'
            )
            fig.add_trace(go.Scatter(x=future_dates, y=predictions,
                                     name='AI預測',
                                     line=dict(color='#FF6B6B', dash='dash', width=2)))
            
            std = df['Close'].pct_change().std() * current_price
            upper = [p + 1.96 * std for p in predictions]
            lower = [max(p - 1.96 * std, 0) for p in predictions]
            
            fig.add_trace(go.Scatter(x=future_dates, y=upper, fill=None,
                                     mode='lines', line_color='rgba(0,0,0,0)',
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=future_dates, y=lower, fill='tonexty',
                                     mode='lines',
                                     line_color='rgba(255,107,107,0.2)',
                                     name='95%置信區間'))
            
            fig.update_layout(template="plotly_dark", height=550,
                              title=f"未來 {pred_days} 天 AI 股價預測")
            st.plotly_chart(fig, use_container_width=True)
        
        # ==========================================
        # 回測分析
        # ==========================================
        st.markdown("---")
        st.markdown("### 💰 多策略回測分析")
        
        best = max(backtest_results, key=lambda x: x['return'])
        
        st.markdown("#### 🏆 最佳策略")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("策略名稱", best['name'])
        c2.metric("總回報率", f"{best['return']:.2f}%", delta=f"{best['return']:.2f}%")
        c3.metric("夏普比率", f"{best['sharpe']:.3f}")
        c4.metric("勝率", f"{best['win_rate']:.1f}%")
        
        st.markdown("#### 📋 策略績效比較")
        comparison = pd.DataFrame([{
            '策略': r['name'],
            '總回報率 (%)': f"{r['return']:.2f}%",
            '夏普比率': f"{r['sharpe']:.3f}",
            '最大回撤 (%)': f"{r['max_dd']:.2f}%",
            '交易次數': r['trades'],
            '勝率 (%)': f"{r['win_rate']:.1f}%",
            '最終資金': f"${r['final_capital']:,.0f}"
        } for r in backtest_results])
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        
        st.markdown("#### 📈 資金曲線對比")
        fig_eq = go.Figure()
        colors = ['#00BFFF', '#00FF7F', '#FF6B6B', '#FFA500']
        for i, r in enumerate(backtest_results):
            fig_eq.add_trace(go.Scatter(
                y=r['equity'], name=r['name'],
                line=dict(color=colors[i], width=2)
            ))
        fig_eq.add_hline(y=capital, line_dash="dash", line_color="gray",
                         annotation_text="初始資金")
        fig_eq.update_layout(
            template="plotly_dark", height=450,
            xaxis_title="交易天數", yaxis_title="資金 (HKD)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_eq, use_container_width=True)
        
        with st.expander("📊 最佳策略詳細分析", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                **📈 績效指標**
                - 策略名稱: **{best['name']}**
                - 總回報率: **{best['return']:.2f}%**
                - 夏普比率: **{best['sharpe']:.3f}**
                - 最大回撤: **{best['max_dd']:.2f}%**
                - 卡瑪比率: **{best['return']/abs(best['max_dd']) if best['max_dd'] != 0 else 0:.2f}**
                """)
            with c2:
                st.markdown(f"""
                **📊 交易統計**
                - 總交易次數: **{best['trades']}**
                - 盈利次數: **{best['wins']}**
                - 虧損次數: **{best['losses']}**
                - 勝率: **{best['win_rate']:.1f}%**
                - 最終資金: **${best['final_capital']:,.0f}**
                """)
        
        st.markdown("#### 💡 策略建議")
        tips = {
            'SMA': "📌 **SMA交叉策略** 適合趨勢明顯的市場，牛市中表現較好。建議搭配止損使用。",
            'RSI': "📌 **RSI策略** 適合震盪市場，在超買超賣區域操作勝率較高。",
            'MACD': "📌 **MACD策略** 適合捕捉趨勢轉折點，建議配合成交量確認信號。",
            '布林帶': "📌 **布林帶策略** 適合高波動市場，價格觸及軌道邊緣時反轉概率大。"
        }
        for key, tip in tips.items():
            if key in best['name']:
                st.info(tip)
                break
        
        st.warning("⚠️ **風險提示**：過去績效不代表未來表現，投資有風險，請謹慎決策。")
        
        # ==========================================
        # 數據下載
        # ==========================================
        st.markdown("---")
        st.markdown("### 📥 數據導出")
        
        c1, c2 = st.columns(2)
        with c1:
            csv_data = df[['Open', 'High', 'Low', 'Close', 'Volume',
                          'RSI', 'MACD', 'ATR']].round(3).to_csv()
            st.download_button(
                "📊 下載歷史數據 (CSV)",
                csv_data,
                f"{symbol}_data_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True,
                key="dl_csv"
            )
        
        with c2:
            report = f"""港股AI分析報告
================
股票: {stock_map[symbol]} ({symbol})
分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
當前價格: ${current_price:.2f}
AI建議: {signal['action']} (信心指數: {signal['score']})
目標價: ${signal['predicted_price']:.2f}
預期回報: {signal['expected_return']:+.2f}%

最佳策略: {best['name']}
總回報率: {best['return']:.2f}%
夏普比率: {best['sharpe']:.3f}
最大回撤: {best['max_dd']:.2f}%
勝率: {best['win_rate']:.1f}%
"""
            st.download_button(
                "📄 下載分析報告 (TXT)",
                report,
                f"{symbol}_report_{datetime.now().strftime('%Y%m%d')}.txt",
                "text/plain",
                use_container_width=True,
                key="dl_report"
            )
        
        with st.expander("📋 查看原始數據 (最近30天)"):
            st.dataframe(
                df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']]
                .tail(30).round(2),
                use_container_width=True
            )
    
    else:
        st.info("👈 請在左側選擇股票和分析參數，然後點擊「🚀 開始分析」按鈕")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            ### 🤖 AI預測
            - 線性回歸 + 指數平滑
            - 95%置信區間
            - 7-90天靈活預測
            """)
        with c2:
            st.markdown("""
            ### 💰 回測引擎
            - SMA交叉策略
            - RSI超買超賣
            - MACD金叉死叉
            - 布林帶軌道
            """)
        with c3:
            st.markdown("""
            ### 🎯 智能信號
            - 多指標綜合評分
            - 0-100信心指數
            - 買入/賣出建議
            """)


if __name__ == "__main__":
    main()