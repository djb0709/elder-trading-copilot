import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rag import (
    load_or_build_vector_store,
    retrieve, build_prompt, generate_response,
)

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Elder Trading Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# RAG Initialization (cached so it only loads once)
# ============================================================

@st.cache_resource(show_spinner="Loading RAG index...")
def init_rag():
    return load_or_build_vector_store()

vector_store = init_rag()

# ============================================================
# Indicator Functions
# ============================================================

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ============================================================
# Backtest Engine
# ============================================================

def run_backtest(data, win_short, win_long, rsi_lower, rsi_upper, breakout_window):
    data = data.copy()
    data["EMA_short"] = ema(data["Close"], win_short)
    data["EMA_long"] = ema(data["Close"], win_long)
    data["MACD"], data["MACD_signal"] = macd(data["Close"])
    data["RSI"] = rsi(data["Close"])
    data["Breakout_high"] = data["High"].rolling(window=breakout_window).max().shift(1)
    data["Breakout_low"] = data["Low"].rolling(window=breakout_window).min().shift(1)

    capital = 10000
    cash = capital
    position = 0
    shares = 0
    entry_price = 0
    equity_curve = []
    buy_signals = []
    sell_signals = []

    for i in range(len(data)):
        price = data["Close"].iloc[i]
        trend_up = data["EMA_short"].iloc[i] > data["EMA_long"].iloc[i]
        trend_down = data["EMA_short"].iloc[i] < data["EMA_long"].iloc[i]
        macd_up = data["MACD"].iloc[i] > data["MACD_signal"].iloc[i]
        macd_down = data["MACD"].iloc[i] < data["MACD_signal"].iloc[i]
        rsi_low = data["RSI"].iloc[i] < rsi_lower
        rsi_high = data["RSI"].iloc[i] > rsi_upper
        breakout_high = data["Breakout_high"].iloc[i]
        breakout_low = data["Breakout_low"].iloc[i]

        if position == 0 and trend_up and macd_up and price > breakout_high:
            position = 1
            shares = int(cash // price)
            entry_price = price
            cash -= shares * price
            buy_signals.append(i)
        elif position == 0 and trend_down and macd_down and price < breakout_low:
            position = -1
            shares = int(cash // price)
            entry_price = price
            cash += shares * price
            sell_signals.append(i)
        elif position == 1 and rsi_high:
            cash += shares * price
            position = 0
            shares = 0
        elif position == -1 and rsi_low:
            cash -= shares * price
            position = 0
            shares = 0

        if position == 1:
            equity = cash + shares * price
        elif position == -1:
            equity = cash - shares * price
        else:
            equity = cash
        equity_curve.append(equity)

    data["Equity"] = equity_curve

    total_return = (equity_curve[-1] - capital) / capital * 100
    trade_count = len(buy_signals) + len(sell_signals)
    daily_returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0
    equity_series = pd.Series(equity_curve)
    peak = equity_series.cummax()
    drawdown = (peak - equity_series) / peak
    max_drawdown = drawdown.max() * 100

    if max_drawdown < 15:
        risk_level = "Low"
    elif max_drawdown < 30:
        risk_level = "Medium"
    else:
        risk_level = "High"

    metrics = {
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_drawdown, 2),
        "trade_count": trade_count,
        "risk_level": risk_level,
    }
    return data, buy_signals, sell_signals, metrics

# ============================================================
# Charts
# ============================================================

def build_charts(data, buy_signals, sell_signals, rsi_lower, rsi_upper):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.03,
        subplot_titles=("Price + Signals", "RSI", "Equity Curve"),
    )

    fig.add_trace(go.Scatter(
        x=data.index, y=data["Close"], name="Close",
        line=dict(color="#636EFA"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["EMA_short"], name="EMA Short",
        line=dict(color="#00CC96", dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["EMA_long"], name="EMA Long",
        line=dict(color="#FFA15A", dash="dot"),
    ), row=1, col=1)

    if buy_signals:
        fig.add_trace(go.Scatter(
            x=data.index[buy_signals], y=data["Close"].iloc[buy_signals],
            mode="markers", name="Long Entry",
            marker=dict(symbol="triangle-up", size=10, color="#00CC96"),
        ), row=1, col=1)
    if sell_signals:
        fig.add_trace(go.Scatter(
            x=data.index[sell_signals], y=data["Close"].iloc[sell_signals],
            mode="markers", name="Short Entry",
            marker=dict(symbol="triangle-down", size=10, color="#EF553B"),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data.index, y=data["RSI"], name="RSI",
        line=dict(color="#AB63FA"),
    ), row=2, col=1)
    fig.add_hline(y=rsi_lower, line_dash="dash", line_color="green",
                  annotation_text=f"Oversold ({rsi_lower})", row=2, col=1)
    fig.add_hline(y=rsi_upper, line_dash="dash", line_color="red",
                  annotation_text=f"Overbought ({rsi_upper})", row=2, col=1)
    fig.add_hrect(y0=0, y1=rsi_lower, fillcolor="green", opacity=0.05, row=2, col=1)
    fig.add_hrect(y0=rsi_upper, y1=100, fillcolor="red", opacity=0.05, row=2, col=1)

    buy_hold = 10000 * data["Close"] / data["Close"].iloc[0]
    fig.add_trace(go.Scatter(
        x=data.index, y=buy_hold, name="Buy & Hold",
        line=dict(color="gray", dash="dash"),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["Equity"], name="Strategy",
        line=dict(color="#19D3F3"),
        fill="tozeroy", fillcolor="rgba(25,211,243,0.1)",
    ), row=3, col=1)

    fig.update_layout(
        height=700, template="plotly_white", showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio ($)", row=3, col=1)
    return fig

# ============================================================
# Sidebar — only strategy controls
# ============================================================

with st.sidebar:
    st.markdown("### Stock Selection")
    ticker = st.selectbox("Ticker", ["SPY", "AAPL", "TSLA", "MSFT", "QQQ", "AMZN", "GOOG"])
    col_s, col_e = st.columns(2)
    start_date = col_s.date_input("Start", pd.to_datetime("2020-01-01"))
    end_date = col_e.date_input("End", pd.to_datetime("2025-12-31"))

    st.markdown("### Strategy Parameters")
    win_short = st.slider("EMA Short", 5, 50, 15)
    win_long = st.slider("EMA Long", 50, 300, 200)
    rsi_lower = st.slider("RSI Lower (Oversold)", 20, 60, 40)
    rsi_upper = st.slider("RSI Upper (Overbought)", 40, 95, 75)
    breakout_window = st.slider("Breakout Window (Screen 3)", 3, 20, 5)

# ============================================================
# Load data & run backtest
# ============================================================

@st.cache_data(show_spinner="Downloading stock data...")
def load_stock(tk, start, end):
    stock = yf.download(tk, start=start, end=end)
    if isinstance(stock.columns, pd.MultiIndex):
        stock = stock.xs(tk, axis=1, level="Ticker")
    return stock

stock = load_stock(ticker, start_date, end_date)

if stock.empty:
    st.error("No data found. Check ticker or date range.")
else:
    data, buy_signals, sell_signals, metrics = run_backtest(
        stock, win_short, win_long, rsi_lower, rsi_upper, breakout_window
    )
    st.session_state["backtest_results"] = {
        "data": data,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "metrics": metrics,
        "ticker": ticker,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "params": {
            "win_short": win_short,
            "win_long": win_long,
            "rsi_lower": rsi_lower,
            "rsi_upper": rsi_upper,
            "breakout_window": breakout_window,
        },
    }

    m = metrics

    # ========================================================
    # Metrics row
    # ========================================================
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Return", f"{m['total_return']}%")
    c2.metric("Sharpe Ratio", f"{m['sharpe']}")
    c3.metric("Max Drawdown", f"-{m['max_drawdown']}%")
    c4.metric("Trade Count", f"{m['trade_count']}")
    c5.metric("Risk Level", m["risk_level"])

    # ========================================================
    # Left: Dashboard (60%) | Right: Chatbot (40%)
    # ========================================================
    col_dash, col_chat = st.columns([60, 40])

    # --- Dashboard ---
    with col_dash:
        fig = build_charts(data, buy_signals, sell_signals, rsi_lower, rsi_upper)
        st.plotly_chart(fig, use_container_width=True)

    # --- Chatbot ---
    with col_chat:
        # Context bar
        st.markdown(
            f"<div style='background:#f0f2f6; padding:8px 12px; border-radius:6px; "
            f"font-size:0.8rem; color:#555; margin-bottom:8px;'>"
            f"<strong>{ticker}</strong> | {start_date} ~ {end_date} | "
            f"EMA {win_short}/{win_long} | RSI {rsi_lower}-{rsi_upper} | "
            f"Breakout {breakout_window}<br>"
            f"Return <strong>{m['total_return']}%</strong> | "
            f"Sharpe <strong>{m['sharpe']}</strong> | "
            f"MaxDD <strong>-{m['max_drawdown']}%</strong> | "
            f"Risk: <strong>{m['risk_level']}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Model selector (inside chatbot area)
        model_choice = st.selectbox(
            "Model",
            ["Llama 3.3 (Groq)", "GPT-4o-mini (OpenAI)"],
            label_visibility="collapsed",
        )

        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        chat_container = st.container(height=380)
        with chat_container:
            if not st.session_state["chat_history"]:
                st.markdown(
                    "<div style='text-align:center; color:#aaa; padding:2rem;'>"
                    "Ask me about Elder's trading strategy, "
                    "your current setup, or backtest results."
                    "</div>",
                    unsafe_allow_html=True,
                )
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        # Quick ask buttons
        quick_query = None
        qcol1, qcol2, qcol3 = st.columns(3)
        if qcol1.button("Explain Setup", use_container_width=True):
            quick_query = "Explain my current strategy setup and what each parameter means."
        if qcol2.button("Explain Perf", use_container_width=True):
            quick_query = "Analyze my current backtest results and explain the trade-offs."
        if qcol3.button("Risk?", use_container_width=True):
            quick_query = "What are the risks of my current strategy configuration?"

        # Chat input
        user_input = st.chat_input("Ask about Elder's strategy...")
        query = quick_query or user_input

        if query:
            st.session_state["chat_history"].append(
                {"role": "user", "content": query}
            )

            dashboard_ctx = st.session_state.get("backtest_results", None)

            with st.spinner("Thinking..."):
                retrieved = retrieve(vector_store, query, k=5)
                prompt = build_prompt(query, retrieved, dashboard_ctx)
                response = generate_response(prompt, model_choice)

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": response}
            )
            st.rerun()
