import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# 1. PAGE CONFIG
st.set_page_config(page_title="VoltRisk Analytics", page_icon="⚡", layout="wide")

# 2. PRO STYLING
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    section[data-testid="stSidebar"] { background-color: #161B22 !important; border-right: 1px solid #30363D; }
    .signal-box { padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #30363D; margin-bottom: 20px; }
    .beginner-card { background-color: #161B22; padding: 15px; border-radius: 10px; border-left: 5px solid #00FBFF; margin-bottom: 10px; min-height: 150px; }
    .pro-badge { background-color: #FFD700; color: black; padding: 2px 8px; border-radius: 5px; font-weight: bold; font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)

# 3. SIDEBAR & AUTH SIMULATION
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #00FFAA;'>⚡ VOLTRISK</h1>", unsafe_allow_html=True)
    
    # Subscription Simulation (In production, this is linked to Stripe/Auth)
    st.markdown("---")
    user_status = st.toggle("Simulate Pro Status", value=False)
    status_label = "<span class='pro-badge'>PRO</span>" if user_status else "FREE"
    st.markdown(f"Account Status: {status_label}", unsafe_allow_html=True)
    
    st.markdown("---")
    ticker = st.text_input("Asset Ticker", value="NVDA").upper()
    investment = st.number_input("Capital Allocation ($)", min_value=10.0, value=1000.0)
    
    # Gated Features
    max_sims = 10000 if user_status else 500
    iterations = st.slider("Number of Simulations", 100, max_sims, 500, step=100)
    if not user_status:
        st.caption("🚀 Upgrade to Pro for 10,000 simulations")
        
    time_horizon = st.slider("Days to Forecast", 1, 730, 252)
    apply_crash = st.checkbox("Simulate '2020 Covid crash'")
    
    start_sim = st.button("RUN SIMULATION", use_container_width=True)
    
    st.markdown("---")
    st.markdown("**⚙️ Engine Notes**")
    st.markdown("<div style='font-size:12px; color:gray;'>VoltRisk uses GBM math based on 3-year history. Past performance != future results.</div>", unsafe_allow_html=True)

# 4. MAIN DASHBOARD
st.title("⚡ :blue[Volt]Risk Analytics")

if start_sim:
    with st.spinner('Accessing Financial Data...'):
        # Data Fetching
        data = yf.download(ticker, start=(datetime.now() - timedelta(days=1095)), auto_adjust=False)
        spy_data = yf.download("SPY", start=(datetime.now() - timedelta(days=1095)), auto_adjust=False)
        
        if data.empty:
            st.error("Invalid Ticker.")
        else:
            # Flattening MultiIndex
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            if isinstance(spy_data.columns, pd.MultiIndex): spy_data.columns = spy_data.columns.get_level_values(0)

            # Engine
            def run_mc(df, inv, n):
                rets = df['Adj Close'].pct_change().dropna()
                mu, sigma, last = rets.mean(), rets.std(), df['Adj Close'].iloc[-1]
                daily = np.random.normal(mu, sigma, (time_horizon, n))
                paths = np.zeros_like(daily); paths[0] = last * (1 + daily[0])
                for t in range(1, time_horizon): paths[t] = paths[t-1] * (1 + daily[t])
                return (paths / last) * inv

            asset_paths = run_mc(data, investment, iterations)
            final_vals = asset_paths[-1]
            win_prob = (np.sum(final_vals > investment) / iterations) * 100
            tp_95, sl_5, mean_outcome = np.percentile(final_vals, 95), np.percentile(final_vals, 5), np.mean(final_vals)
            
            cum_max = np.maximum.accumulate(asset_paths, axis=0)
            drawdowns = (asset_paths - cum_max) / cum_max
            avg_max_dd = np.mean(np.min(drawdowns, axis=0)) * 100

            # 5. GAUGE & SIGNAL
            st.divider()
            c1, c2 = st.columns([1, 1])
            with c1:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=win_prob, number={'suffix': "%"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1b5e20"},
                           'steps': [{'range': [0, 40], 'color': "#FF4B4B"},
                                     {'range': [40, 70], 'color': "#FFD700"},
                                     {'range': [70, 100], 'color': "#00FFAA"}]}))
                fig_g.update_layout(height=280, margin=dict(t=50, b=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_g, use_container_width=True)
            with c2:
                st.markdown("<br>", unsafe_allow_html=True)
                if win_prob > 60:
                    st.markdown(f"<div class='signal-box' style='border-color:#00FFAA;'><h2>BUY SIGNAL</h2><p>Probability favors appreciation.</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='signal-box'><h2>WAIT</h2><p>Risk-adjusted probability is neutral/weak.</p></div>", unsafe_allow_html=True)

            # 6. METRICS
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CURRENT PRICE", f"${data['Adj Close'].iloc[-1]:,.2f}")
            m2.metric("MEAN OUTCOME", f"${mean_outcome:,.2f}")
            m3.metric("MAX DRAWDOWN", f"{avg_max_dd:.1f}%")
            m4.metric("STOP LOSS (5%)", f"${sl_5:,.2f}")

            # 7. CHART
            st.subheader("🔍 Market Benchmark & Volatility Bands")
            fig = go.Figure()
            days = list(range(time_horizon))
            
            for i in range(min(50, iterations)):
                fig.add_trace(go.Scatter(x=days, y=asset_paths[:, i], line=dict(color='rgba(0, 251, 255, 0.05)', width=1), hoverinfo='none', showlegend=False))
            
            # Gated Benchmark
            if user_status:
                spy_paths = run_mc(spy_data, investment, 1000)
                fig.add_trace(go.Scatter(x=days, y=np.mean(spy_paths, axis=1), name="S&P 500", line=dict(color='white', dash='dot')))
            
            fig.add_trace(go.Scatter(x=days, y=np.mean(asset_paths, axis=1), name="Mean Projection", line=dict(color='#FFD700', width=4)))
            
            low, high = np.percentile(asset_paths, 5, axis=1), np.percentile(asset_paths, 95, axis=1)
            fig.add_trace(go.Scatter(x=days+days[::-1], y=list(high)+list(low)[::-1], fill='toself', fillcolor='rgba(0, 255, 170, 0.1)', line_color='rgba(0,0,0,0)', name='95% Confidence'))
            
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500)
            st.plotly_chart(fig, use_container_width=True)

            # 8. STRATEGIC INTERPRETATION
            st.divider()
            st.subheader("📊 Strategic Analysis Guide")
            b1, b2, b3 = st.columns(3)
            with b1: st.markdown("<div class='beginner-card'><b>I. Probabilistic Success</b><br>Mathematical frequency of the asset closing above initial capital. Institutional benchmarks require >60% for 'High-Conviction'.</div>", unsafe_allow_html=True)
            with b2: st.markdown("<div class='beginner-card'><b>II. Volatility Endurance</b><br>Quantifies 'Intra-period Risk'. If the Max Drawdown exceeds your tolerance, position size should be reduced.</div>", unsafe_allow_html=True)
            with b3: st.markdown("<div class='beginner-card'><b>III. Relative Alpha</b><br>Reveals if the asset outperforms the S&P 500 (Pro Only). Essential for verifying if the extra risk is 'worth it'.</div>", unsafe_allow_html=True)

            if not user_status:
                st.info("🔒 S&P 500 Benchmark and Downloadable Reports are available for Pro users.")
            else:
                report = io.BytesIO()
                pd.DataFrame({"Metric": ["Win Prob", "Mean", "Max DD"], "Value": [win_prob, mean_outcome, avg_max_dd]}).to_excel(report)
                st.download_button("📩 DOWNLOAD PRO REPORT", data=report, file_name="VoltRisk_Pro.xlsx")