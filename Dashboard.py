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
    page_icon="data/elder_jpa.png",
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
# CSS for slide-in chatbot panel
# ============================================================

st.markdown("""
<style>
/* Hide default Streamlit page nav */
[data-testid="stSidebarNav"] { display: none !important; }

/* ── Chatbot panel wrapper ── */
.chat-panel {
    background: #fff;
    border: 1px solid #e8eaef;
    border-radius: 12px;
    padding: 1rem 1rem 0.6rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.chat-panel-title {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 0.7rem;
    padding-bottom: 0.55rem;
    border-bottom: 2px solid #eef0f4;
}
.chat-panel-title .cp-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: linear-gradient(135deg, #667eea, #764ba2);
}
.chat-panel-title span {
    font-weight: 600; font-size: 0.95rem; color: #333;
}
.chat-context {
    background: linear-gradient(135deg, #f8f9fb, #f0f2f6);
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 0.78rem;
    color: #555;
    margin-bottom: 10px;
    line-height: 1.55;
    border: 1px solid #eaecf0;
}

/* ── Quick-ask buttons — compact row ── */
[class*="quick-ask-wrap"] [data-testid="stHorizontalBlock"] {
    gap: 0.5rem !important;
    justify-content: flex-start !important;
}
[class*="quick-ask-wrap"] [data-testid="stColumn"] {
    width: auto !important;
    flex: 0 0 auto !important;
}

/* ── Close button — scoped by container key ── */
div[data-testid="stElementContainer"]:has(button[key="close_copilot"]) button,
div[data-testid="stVerticalBlock"] > div:has(> div > button[key="close_copilot"]) button {
    background: #f0f2f6 !important;
    border: none !important;
    border-radius: 50% !important;
    width: 30px !important; height: 30px !important; min-width: 30px !important;
    padding: 0 !important;
    color: #888 !important;
    cursor: pointer !important;
    transition: all 0.15s;
}
/* Fallback: target via unique container key */
[class*="close-copilot-wrap"] button {
    background: #f0f2f6 !important;
    border: none !important;
    border-radius: 50% !important;
    width: 30px !important; height: 30px !important; min-width: 30px !important;
    padding: 0 !important;
    color: #888 !important;
    cursor: pointer !important;
    transition: all 0.15s;
}
[class*="close-copilot-wrap"] button:hover {
    background: #667eea !important;
}
[class*="close-copilot-wrap"] button:hover p {
    color: #fff !important;
}
[class*="close-copilot-wrap"] button p {
    color: #888 !important;
    font-size: 0.95rem !important;
    line-height: 1 !important;
}

/* ── Open trigger pill ── */
.open-marker { display: none; }
div[data-testid="stColumn"]:has(.open-marker) {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
}
div[data-testid="stColumn"]:has(.open-marker) button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 0.45rem 1rem !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    box-shadow: 0 2px 8px rgba(102,126,234,0.3) !important;
    cursor: pointer !important;
    transition: box-shadow 0.2s, transform 0.15s;
}
div[data-testid="stColumn"]:has(.open-marker) button:hover {
    box-shadow: 0 4px 14px rgba(102,126,234,0.45) !important;
    transform: translateY(-1px);
}
div[data-testid="stColumn"]:has(.open-marker) button p {
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

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

def run_backtest(data, win_short, win_long, rsi_lower, rsi_upper, breakout_window, capital=10000, position_pct=100):
    data = data.copy()
    data["EMA_short"] = ema(data["Close"], win_short)
    data["EMA_long"] = ema(data["Close"], win_long)
    data["MACD"], data["MACD_signal"] = macd(data["Close"])
    data["RSI"] = rsi(data["Close"])
    data["Breakout_high"] = data["High"].rolling(window=breakout_window).max().shift(1)
    data["Breakout_low"] = data["Low"].rolling(window=breakout_window).min().shift(1)

    cash = capital
    position = 0
    shares = 0
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
            allocate = cash * position_pct / 100
            shares = int(allocate // price)
            cash -= shares * price
            buy_signals.append(i)
        elif position == 0 and trend_down and macd_down and price < breakout_low:
            position = -1
            allocate = cash * position_pct / 100
            shares = int(allocate // price)
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

def build_charts(data, buy_signals, sell_signals, rsi_lower, rsi_upper, capital=10000):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        vertical_spacing=0.04,
        subplot_titles=("Price + Signals", "MACD (Screen 2)", "RSI", "Equity Curve"),
    )

    # Row 1: Price + Signals
    fig.add_trace(go.Scatter(
        x=data.index, y=data["Close"], name="Close",
        line=dict(color="#636EFA"),
        legend="legend1",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["EMA_short"], name="EMA Short",
        line=dict(color="#00CC96", dash="dot"),
        legend="legend1",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["EMA_long"], name="EMA Long",
        line=dict(color="#FFA15A", dash="dot"),
        legend="legend1",
    ), row=1, col=1)

    if buy_signals:
        fig.add_trace(go.Scatter(
            x=data.index[buy_signals], y=data["Close"].iloc[buy_signals],
            mode="markers", name="Long Entry",
            marker=dict(symbol="triangle-up", size=10, color="#00CC96"),
            legend="legend1",
        ), row=1, col=1)
    if sell_signals:
        fig.add_trace(go.Scatter(
            x=data.index[sell_signals], y=data["Close"].iloc[sell_signals],
            mode="markers", name="Short Entry",
            marker=dict(symbol="triangle-down", size=10, color="#EF553B"),
            legend="legend1",
        ), row=1, col=1)

    # Row 2: MACD
    macd_hist = data["MACD"] - data["MACD_signal"]
    colors = ["#26A69A" if v >= 0 else "#EF5350" for v in macd_hist]
    fig.add_trace(go.Bar(
        x=data.index, y=macd_hist, name="Histogram",
        marker_color=colors, showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["MACD"], name="MACD",
        line=dict(color="#2196F3", width=1.5),
        legend="legend2",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["MACD_signal"], name="Signal",
        line=dict(color="#FF9800", width=1.5),
        legend="legend2",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#999", line_width=0.5, row=2, col=1)

    # Row 3: RSI
    fig.add_trace(go.Scatter(
        x=data.index, y=data["RSI"], name="RSI",
        line=dict(color="#AB63FA"),
        legend="legend3",
    ), row=3, col=1)
    fig.add_hline(y=rsi_lower, line_dash="dash", line_color="green",
                  annotation_text=f"Oversold ({rsi_lower})", row=3, col=1)
    fig.add_hline(y=rsi_upper, line_dash="dash", line_color="red",
                  annotation_text=f"Overbought ({rsi_upper})", row=3, col=1)
    fig.add_hrect(y0=0, y1=rsi_lower, fillcolor="green", opacity=0.05, row=3, col=1)
    fig.add_hrect(y0=rsi_upper, y1=100, fillcolor="red", opacity=0.05, row=3, col=1)

    # Row 4: Equity Curve
    buy_hold = capital * data["Close"] / data["Close"].iloc[0]
    fig.add_trace(go.Scatter(
        x=data.index, y=buy_hold, name="Buy & Hold",
        line=dict(color="gray", dash="dash"),
        legend="legend4",
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["Equity"], name="Strategy",
        line=dict(color="#19D3F3"),
        fill="tozeroy", fillcolor="rgba(25,211,243,0.1)",
        legend="legend4",
    ), row=4, col=1)

    # Each subplot gets its own legend positioned at its top-right
    fig.update_layout(
        height=850, template="plotly_white",
        margin=dict(l=0, r=0, t=60, b=0),
        legend1=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0, font=dict(size=10)),
        legend2=dict(orientation="h", yanchor="bottom", y=0.605, xanchor="left", x=0, font=dict(size=10)),
        legend3=dict(orientation="h", yanchor="bottom", y=0.39, xanchor="left", x=0, font=dict(size=10)),
        legend4=dict(orientation="h", yanchor="bottom", y=0.175, xanchor="left", x=0, font=dict(size=10)),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="Portfolio ($)", row=4, col=1)
    return fig

# ============================================================
# Sidebar — strategy controls
# ============================================================

with st.sidebar:
    from components import render_sidebar_header
    render_sidebar_header()
    st.markdown("### Stock Selection")
    ticker = st.selectbox("Ticker", ["SPY", "AAPL", "TSLA", "MSFT", "QQQ", "AMZN", "GOOG"])
    col_s, col_e = st.columns(2)
    start_date = col_s.date_input("Start", pd.to_datetime("2020-01-01"))
    end_date = col_e.date_input("End", pd.to_datetime("2025-12-31"))

    st.markdown("### Portfolio")
    capital = st.number_input(
        "Starting Capital ($)", min_value=1000, max_value=1000000,
        value=10000, step=1000,
        help="The initial amount of money to invest.",
    )
    position_pct = st.slider(
        "Position Size (%)", 10, 100, 100, step=10,
        help="Percentage of available cash to allocate per trade. "
             "100% = full capital (aggressive), 50% = half capital (conservative). "
             "Lower values reduce risk but also reduce potential returns.",
    )

    st.markdown("### Strategy Parameters")
    win_short = st.slider(
        "EMA Short", 5, 50, 15,
        help="Short-term Exponential Moving Average window (days). "
             "Used as the tactical trend filter in Screen 2. "
             "A shorter value reacts faster to price changes.",
    )
    win_long = st.slider(
        "EMA Long", 50, 300, 200,
        help="Long-term Exponential Moving Average window (days). "
             "Defines the strategic trend direction in Screen 1. "
             "When price is above this EMA, the market is in an uptrend.",
    )
    rsi_lower = st.slider(
        "RSI Lower (Oversold)", 20, 60, 40,
        help="RSI level below which the market is considered oversold. "
             "Short positions are closed when RSI drops below this level. "
             "Elder suggests looking for buying opportunities in oversold zones.",
    )
    rsi_upper = st.slider(
        "RSI Upper (Overbought)", 40, 95, 75,
        help="RSI level above which the market is considered overbought. "
             "Long positions are closed when RSI rises above this level. "
             "Elder warns that overbought readings signal potential reversals.",
    )
    breakout_window = st.slider(
        "Breakout Window (Screen 3)", 3, 20, 5,
        help="Number of days to look back for price breakouts (Screen 3 entry trigger). "
             "A long entry fires when price exceeds the N-day high; "
             "a short entry fires when price drops below the N-day low.",
    )

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
        stock, win_short, win_long, rsi_lower, rsi_upper, breakout_window, capital, position_pct
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
            "capital": capital,
            "position_pct": position_pct,
        },
    }

    m = metrics

    # ========================================================
    # Copilot toggle
    # ========================================================
    if "copilot_open" not in st.session_state:
        st.session_state["copilot_open"] = True

    def toggle_copilot():
        st.session_state["copilot_open"] = not st.session_state["copilot_open"]

    copilot_open = st.session_state["copilot_open"]

    # ========================================================
    # Metrics row (always 6 columns — 6th is copilot trigger)
    # ========================================================
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 0.6])
    c1.metric(
        "Total Return", f"{m['total_return']}%",
        help="Percentage gain or loss from starting capital. "
             "Calculated as (final equity - starting capital) / starting capital.",
    )
    c2.metric(
        "Sharpe Ratio", f"{m['sharpe']}",
        help="Risk-adjusted return. Higher is better. "
             "Sharpe = (mean daily return / std of daily returns) x sqrt(252). "
             "Above 1.0 is good, above 2.0 is excellent.",
    )
    c3.metric(
        "Max Drawdown", f"-{m['max_drawdown']}%",
        help="Largest peak-to-trough decline in portfolio value. "
             "A 33% drawdown means the portfolio lost a third of its peak value at some point.",
    )
    c4.metric(
        "Trade Count", f"{m['trade_count']}",
        help="Total number of trade entries (long + short). "
             "More trades mean more opportunities but also more transaction exposure.",
    )
    c5.metric(
        "Risk Level", m["risk_level"],
        help="Based on Max Drawdown: Low (< 15%), Medium (15-30%), High (>= 30%). "
             "Elder emphasizes strict risk control to preserve trading capital.",
    )
    with c6:
        if not copilot_open:
            st.markdown('<div class="open-marker"></div>', unsafe_allow_html=True)
            st.button("\u2039 Copilot", key="open_copilot", on_click=toggle_copilot)

    # ========================================================
    # Layout
    # ========================================================
    if copilot_open:
        col_dash, col_chat = st.columns([60, 40])
    else:
        col_dash = st.container()
        col_chat = None

    with col_dash:
        fig = build_charts(data, buy_signals, sell_signals, rsi_lower, rsi_upper, capital)
        st.plotly_chart(fig, use_container_width=True)

    if col_chat is not None:
        with col_chat:
            # Header: title + close button
            hdr_title, hdr_close = st.columns([6, 1])
            with hdr_title:
                st.markdown(
                    '<div class="chat-panel-title">'
                    '<span class="cp-dot"></span>'
                    '<span>Elder Copilot</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with hdr_close:
                close_wrap = st.container(key="close-copilot-wrap")
                with close_wrap:
                    st.button("\u203A", key="close_copilot", on_click=toggle_copilot)

            # Clear chat button
            if st.button("Clear Chat", key="clear_chat", type="secondary"):
                st.session_state["chat_history"] = []
                st.rerun()

            # Context bar
            st.markdown(
                f'<div class="chat-context">'
                f'<strong>{ticker}</strong> &nbsp;|&nbsp; {start_date} ~ {end_date} &nbsp;|&nbsp; '
                f'EMA {win_short}/{win_long} &nbsp;|&nbsp; RSI {rsi_lower}-{rsi_upper} &nbsp;|&nbsp; '
                f'Breakout {breakout_window}<br>'
                f'Return <strong>{m["total_return"]}%</strong> &nbsp;|&nbsp; '
                f'Sharpe <strong>{m["sharpe"]}</strong> &nbsp;|&nbsp; '
                f'MaxDD <strong>-{m["max_drawdown"]}%</strong> &nbsp;|&nbsp; '
                f'Risk: <strong>{m["risk_level"]}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Model selector
            model_choice = st.selectbox(
                "Model",
                ["Llama 3.3 (Groq)", "GPT-4o-mini (OpenAI)"],
                label_visibility="collapsed",
            )

            # Chat history
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

            chat_container = st.container(height=550)
            with chat_container:
                if not st.session_state["chat_history"]:
                    _, img_col, _ = st.columns([2, 1, 2])
                    with img_col:
                        st.image("data/anime_edit.png", use_container_width=True)
                    st.markdown(
                        '<div style="text-align:center; padding:0 0.5rem 0.5rem;">'
                        '<p style="color:#888; font-size:0.95rem; margin-bottom:1rem;">'
                        "Ask me anything about your strategy, performance, or Elder's teachings."
                        "</p>"
                        '<div style="display:flex; flex-wrap:wrap; gap:0.5rem; '
                        'justify-content:center;">'
                        '<span style="background:#f0f2f6; padding:0.35rem 0.8rem; '
                        'border-radius:16px; font-size:0.78rem; color:#555;">'
                        "How is Sharpe Ratio calculated?</span>"
                        '<span style="background:#f0f2f6; padding:0.35rem 0.8rem; '
                        'border-radius:16px; font-size:0.78rem; color:#555;">'
                        "What is the Triple Screen system?</span>"
                        '<span style="background:#f0f2f6; padding:0.35rem 0.8rem; '
                        'border-radius:16px; font-size:0.78rem; color:#555;">'
                        "Analyze my current results</span>"
                        "</div></div>",
                        unsafe_allow_html=True,
                    )
                for msg in st.session_state["chat_history"]:
                    avatar = "data/anime_edit.png" if msg["role"] == "assistant" else None
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.write(msg["content"])

            # Quick ask buttons
            quick_query = None
            quick_wrap = st.container(key="quick-ask-wrap")
            with quick_wrap:
                qcol1, qcol2, qcol3 = st.columns(3)
                if qcol1.button("Explain Setup"):
                    quick_query = "Explain my current strategy setup and what each parameter means."
                if qcol2.button("Explain Perf"):
                    quick_query = "Analyze my current backtest results and explain the trade-offs."
                if qcol3.button("Risk?"):
                    quick_query = "What are the risks of my current strategy configuration?"

            # Chat input
            user_input = st.chat_input("Ask about Elder's strategy...")
            query = quick_query or user_input

            if query:
                st.session_state["chat_history"].append(
                    {"role": "user", "content": query}
                )

                dashboard_ctx = st.session_state.get("backtest_results", None)

                try:
                    with st.spinner("Thinking..."):
                        retrieved = retrieve(vector_store, query, k=5)
                        prompt = build_prompt(query, retrieved, dashboard_ctx)
                        response = generate_response(prompt, model_choice)

                    model_short = "Llama 3.3" if "Groq" in model_choice else "GPT-4o-mini"
                    meta = f"\n\n---\n*{model_short} | {len(retrieved)} chunks retrieved*"
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": response + meta}
                    )
                    st.rerun()
                except Exception:
                    st.error("Failed to generate response. Please check your API keys in .env file.")
                    # Remove the pending user message so it doesn't linger without a reply
                    if st.session_state["chat_history"] and st.session_state["chat_history"][-1]["role"] == "user":
                        st.session_state["chat_history"].pop()
