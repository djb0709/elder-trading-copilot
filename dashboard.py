import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rag import (
    load_or_build_vector_store,
    retrieve, build_prompt, generate_response,
    build_vector_store_for_model, EMBEDDING_MODELS,
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

@st.cache_resource(show_spinner="Building index...")
def init_rag_with_model(model_key):
    return build_vector_store_for_model(model_key)

# ============================================================
# CSS for slide-in chatbot panel
# ============================================================

st.markdown("""
<style>
/* Hide default Streamlit page nav */
[data-testid="stSidebarNav"] { display: none !important; }

/* Sidebar divider compact */
[data-testid="stSidebar"] hr {
    margin-top: 10px !important;
    margin-bottom: 10px !important;
}


/* Copilot slide-in animation */
@keyframes copilotSlideIn {
    from { opacity: 0; transform: translateX(30px); }
    to   { opacity: 1; transform: translateX(0); }
}
.st-key-copilot-panel {
    animation: copilotSlideIn 0.5s ease !important;
    will-change: transform, opacity;
}

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
/* Welcome buttons centered */
[class*="welcome-ask-wrap"] [data-testid="stHorizontalBlock"] {
    gap: 0.5rem !important;
    justify-content: center !important;
}
[class*="welcome-ask-wrap"] [data-testid="stColumn"] {
    width: auto !important;
    flex: 0 0 auto !important;
}

/* Action bar: flex row, pill-style buttons */
.st-key-action-bar {
    flex-flow: row nowrap !important;
    justify-content: flex-start !important;
    align-items: center !important;
    gap: 0.4rem !important;
}
.st-key-action-bar > div {
    width: auto !important;
    flex: 0 0 auto !important;
}
.st-key-action-bar button {
    border-radius: 20px !important;
    padding: 0.25rem 0.85rem !important;
    font-size: 0.82rem !important;
    height: 34px !important;
}
.st-key-action-bar > div:last-child {
    margin-left: auto !important;
}
/* Clear button red */
.st-key-clear_chat button {
    background: #fee2e2 !important;
    color: #dc2626 !important;
    border: 1px solid #fca5a5 !important;
}
.st-key-clear_chat button:hover {
    background: #dc2626 !important;
    color: #fff !important;
}
.st-key-clear_chat button p {
    color: inherit !important;
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
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    max-width: 32px !important;
    min-height: 32px !important;
    max-height: 32px !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #888 !important;
    cursor: pointer !important;
    transition: all 0.15s;
    line-height: 1 !important;
}
[class*="close-copilot-wrap"] button:hover {
    background: #667eea !important;
}
[class*="close-copilot-wrap"] button:hover p {
    color: #fff !important;
}
.st-key-close_copilot {
    align-self: flex-end !important;
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
    entry_price = 0
    entry_idx = 0
    equity_curve = []
    buy_signals = []
    sell_signals = []
    trade_log = []

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
            entry_price = price
            entry_idx = i
            cash -= shares * price
            buy_signals.append(i)
        elif position == 0 and trend_down and macd_down and price < breakout_low:
            position = -1
            allocate = cash * position_pct / 100
            shares = int(allocate // price)
            entry_price = price
            entry_idx = i
            cash += shares * price
            sell_signals.append(i)
        elif position == 1 and rsi_high:
            pnl = (price - entry_price) * shares
            trade_log.append({
                "Type": "Long",
                "Entry Date": str(data.index[entry_idx].date()),
                "Entry Price": round(entry_price, 2),
                "Exit Date": str(data.index[i].date()),
                "Exit Price": round(price, 2),
                "Shares": shares,
                "P&L ($)": round(pnl, 2),
                "Return (%)": round(pnl / (entry_price * shares) * 100, 2) if shares else 0,
                "Days Held": i - entry_idx,
            })
            cash += shares * price
            position = 0
            shares = 0
        elif position == -1 and rsi_low:
            pnl = (entry_price - price) * shares
            trade_log.append({
                "Type": "Short",
                "Entry Date": str(data.index[entry_idx].date()),
                "Entry Price": round(entry_price, 2),
                "Exit Date": str(data.index[i].date()),
                "Exit Price": round(price, 2),
                "Shares": shares,
                "P&L ($)": round(pnl, 2),
                "Return (%)": round(pnl / (entry_price * shares) * 100, 2) if shares else 0,
                "Days Held": i - entry_idx,
            })
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
    equity_series = pd.Series(equity_curve, index=data.index)
    peak = equity_series.cummax()
    drawdown_series = (peak - equity_series) / peak * 100
    data["Drawdown"] = drawdown_series.values

    total_return = (equity_curve[-1] - capital) / capital * 100
    trade_count = len(buy_signals) + len(sell_signals)
    daily_returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0
    max_drawdown = drawdown_series.max()

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
    return data, buy_signals, sell_signals, metrics, trade_log

# ============================================================
# Charts
# ============================================================

def build_charts(data, buy_signals, sell_signals, rsi_lower, rsi_upper, capital=10000,
                  visible_panels=None):
    if visible_panels is None:
        visible_panels = ["Price + Signals", "MACD", "RSI", "Equity Curve", "Drawdown"]

    panel_map = {
        "Price + Signals": "Price + Signals",
        "MACD": "MACD (Screen 2)",
        "RSI": "RSI",
        "Equity Curve": "Equity Curve",
        "Drawdown": "Drawdown",
    }
    active = [p for p in panel_map if p in visible_panels]
    if not active:
        active = ["Price + Signals"]
    n_rows = len(active)
    titles = tuple(panel_map[p] for p in active)

    # Give Price row more height
    heights = []
    for p in active:
        heights.append(0.4 if p == "Price + Signals" else 0.2)
    total = sum(heights)
    heights = [h / total for h in heights]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.04,
        subplot_titles=titles,
    )

    # Map panel name to row number
    row_of = {p: i + 1 for i, p in enumerate(active)}

    # Use a single legend since row numbers are dynamic
    legend_idx = 1

    if "Price + Signals" in row_of:
        r = row_of["Price + Signals"]
        lg = f"legend{legend_idx}"; legend_idx += 1
        fig.add_trace(go.Scatter(
            x=data.index, y=data["Close"], name="Close",
            line=dict(color="#636EFA"), legend=lg,
        ), row=r, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data["EMA_short"], name="EMA Short",
            line=dict(color="#00CC96", dash="dot"), legend=lg,
        ), row=r, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data["EMA_long"], name="EMA Long",
            line=dict(color="#FFA15A", dash="dot"), legend=lg,
        ), row=r, col=1)
        if buy_signals:
            fig.add_trace(go.Scatter(
                x=data.index[buy_signals], y=data["Close"].iloc[buy_signals],
                mode="markers", name="Long Entry",
                marker=dict(symbol="triangle-up", size=10, color="#00CC96"), legend=lg,
            ), row=r, col=1)
        if sell_signals:
            fig.add_trace(go.Scatter(
                x=data.index[sell_signals], y=data["Close"].iloc[sell_signals],
                mode="markers", name="Short Entry",
                marker=dict(symbol="triangle-down", size=10, color="#EF553B"), legend=lg,
            ), row=r, col=1)
        fig.update_yaxes(title_text="Price ($)", row=r, col=1)

    if "MACD" in row_of:
        r = row_of["MACD"]
        lg = f"legend{legend_idx}"; legend_idx += 1
        macd_hist = data["MACD"] - data["MACD_signal"]
        colors = ["#26A69A" if v >= 0 else "#EF5350" for v in macd_hist]
        fig.add_trace(go.Bar(
            x=data.index, y=macd_hist, name="Histogram",
            marker_color=colors, showlegend=False,
        ), row=r, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data["MACD"], name="MACD",
            line=dict(color="#2196F3", width=1.5), legend=lg,
        ), row=r, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data["MACD_signal"], name="Signal",
            line=dict(color="#FF9800", width=1.5), legend=lg,
        ), row=r, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#999", line_width=0.5, row=r, col=1)
        fig.update_yaxes(title_text="MACD", row=r, col=1)

    if "RSI" in row_of:
        r = row_of["RSI"]
        lg = f"legend{legend_idx}"; legend_idx += 1
        fig.add_trace(go.Scatter(
            x=data.index, y=data["RSI"], name="RSI",
            line=dict(color="#AB63FA"), legend=lg,
        ), row=r, col=1)
        fig.add_hline(y=rsi_lower, line_dash="dash", line_color="green",
                      annotation_text=f"Oversold ({rsi_lower})", row=r, col=1)
        fig.add_hline(y=rsi_upper, line_dash="dash", line_color="red",
                      annotation_text=f"Overbought ({rsi_upper})", row=r, col=1)
        fig.add_hrect(y0=0, y1=rsi_lower, fillcolor="green", opacity=0.05, row=r, col=1)
        fig.add_hrect(y0=rsi_upper, y1=100, fillcolor="red", opacity=0.05, row=r, col=1)
        fig.update_yaxes(title_text="RSI", row=r, col=1)

    if "Equity Curve" in row_of:
        r = row_of["Equity Curve"]
        lg = f"legend{legend_idx}"; legend_idx += 1
        buy_hold = capital * data["Close"] / data["Close"].iloc[0]
        fig.add_trace(go.Scatter(
            x=data.index, y=buy_hold, name="Buy & Hold",
            line=dict(color="gray", dash="dash"), legend=lg,
        ), row=r, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data["Equity"], name="Strategy",
            line=dict(color="#19D3F3"),
            fill="tozeroy", fillcolor="rgba(25,211,243,0.1)", legend=lg,
        ), row=r, col=1)
        fig.update_yaxes(title_text="Portfolio ($)", row=r, col=1)

    if "Drawdown" in row_of:
        r = row_of["Drawdown"]
        lg = f"legend{legend_idx}"; legend_idx += 1
        fig.add_trace(go.Scatter(
            x=data.index, y=-data["Drawdown"], name="Drawdown",
            line=dict(color="#EF5350", width=1),
            fill="tozeroy", fillcolor="rgba(239,83,80,0.15)", legend=lg,
        ), row=r, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#999", line_width=0.5, row=r, col=1)
        fig.update_yaxes(title_text="DD (%)", row=r, col=1)

    # Dynamic height based on panel count
    chart_height = max(400, n_rows * 200)
    fig.update_layout(
        height=chart_height, template="plotly_white",
        margin=dict(l=0, r=0, t=60, b=0),
    )

    # Position legends dynamically based on subplot domains
    for i in range(1, legend_idx):
        y_domain = fig.layout[f"yaxis{'' if i == 1 else i}"].domain
        fig.update_layout(**{
            f"legend{i}": dict(
                orientation="h", yanchor="top", y=y_domain[1],
                xanchor="left", x=0.05, font=dict(size=10),
            )
        })

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

    st.markdown("### Chart Panels")
    optional_panels = ["MACD", "RSI", "Equity Curve", "Drawdown"]
    extra_panels = st.multiselect(
        "Additional panels",
        optional_panels,
        default=optional_panels,
        label_visibility="collapsed",
    )
    visible_panels = ["Price + Signals"] + extra_panels


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
    data, buy_signals, sell_signals, metrics, trade_log = run_backtest(
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
    # Celebration animations (only on improvement/deterioration)
    # ========================================================
    anim_on = st.session_state.get("anim_on", True)

    prev = st.session_state.get("prev_metrics")
    if prev is not None:
        prev_ret = prev["total_return"]
        cur_ret = m["total_return"]
        prev_sharpe = prev["sharpe"]
        cur_sharpe = m["sharpe"]

        if anim_on:
            if cur_ret > prev_ret and cur_sharpe > prev_sharpe:
                st.balloons()
                st.toast(
                    f"Sharpe {prev_sharpe} -> {cur_sharpe}, "
                    f"Return {prev_ret}% -> {cur_ret}%",
                    icon="🎉",
                )
            elif cur_ret < prev_ret and cur_sharpe < prev_sharpe:
                st.snow()
                st.toast(
                    f"Sharpe {prev_sharpe} -> {cur_sharpe}, "
                    f"Return {prev_ret}% -> {cur_ret}%",
                    icon="❄️",
                )
    st.session_state["prev_metrics"] = m

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
        fig = build_charts(data, buy_signals, sell_signals, rsi_lower, rsi_upper, capital, visible_panels)
        st.plotly_chart(fig, use_container_width=True)

        # Trade log table
        if trade_log:
            with st.expander(f"Trade Log ({len(trade_log)} closed trades)", expanded=False):
                trade_df = pd.DataFrame(trade_log)
                def color_pnl(v):
                    if isinstance(v, (int, float)) and v > 0:
                        return "color: #16a34a"
                    elif isinstance(v, (int, float)) and v < 0:
                        return "color: #dc2626"
                    return ""
                styled = trade_df.style.map(color_pnl, subset=["P&L ($)", "Return (%)"])
                st.dataframe(styled, use_container_width=True, hide_index=True)
                # Summary stats
                wins = sum(1 for t in trade_log if t["P&L ($)"] > 0)
                losses = sum(1 for t in trade_log if t["P&L ($)"] <= 0)
                total_pnl = sum(t["P&L ($)"] for t in trade_log)
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Wins", wins)
                s2.metric("Losses", losses)
                s3.metric("Win Rate", f"{wins / len(trade_log) * 100:.0f}%")
                s4.metric("Total P&L", f"${total_pnl:,.0f}")

    @st.fragment
    def copilot_chat():
        """Chatbot fragment — reruns only this section, not the full page."""
        panel = st.container(key="copilot-panel")
        with panel:
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
                    if st.button("\u203A", key="close_copilot"):
                        toggle_copilot()
                        st.rerun(scope="app")

            # Context bar
            ctx = st.session_state.get("backtest_results", {})
            ctx_m = ctx.get("metrics", m)
            ctx_p = ctx.get("params", {})
            st.markdown(
                f'<div class="chat-context">'
                f'<strong>{ctx.get("ticker", ticker)}</strong> &nbsp;|&nbsp; '
                f'{ctx.get("start_date", start_date)} ~ {ctx.get("end_date", end_date)} &nbsp;|&nbsp; '
                f'EMA {ctx_p.get("win_short", win_short)}/{ctx_p.get("win_long", win_long)} &nbsp;|&nbsp; '
                f'RSI {ctx_p.get("rsi_lower", rsi_lower)}-{ctx_p.get("rsi_upper", rsi_upper)} &nbsp;|&nbsp; '
                f'Breakout {ctx_p.get("breakout_window", breakout_window)}<br>'
                f'Return <strong>{ctx_m.get("total_return", m["total_return"])}%</strong> &nbsp;|&nbsp; '
                f'Sharpe <strong>{ctx_m.get("sharpe", m["sharpe"])}</strong> &nbsp;|&nbsp; '
                f'MaxDD <strong>-{ctx_m.get("max_drawdown", m["max_drawdown"])}%</strong> &nbsp;|&nbsp; '
                f'Risk: <strong>{ctx_m.get("risk_level", m["risk_level"])}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Embedding & LLM selectors
            emb_col, llm_col = st.columns(2)
            with emb_col:
                emb_choice = st.selectbox(
                    "Embedding Model",
                    list(EMBEDDING_MODELS.keys()),
                    index=0,
                    help="Converts text to vectors for similarity search.",
                )
            with llm_col:
                model_choice = st.selectbox(
                    "LLM Model",
                    ["Llama 3.3 70B (Groq)", "Llama 3.1 8B (Groq)", "Mixtral 8x7B (Groq)", "GPT-4o-mini (OpenAI)"],
                    help="Generates the final answer from retrieved context.",
                )

            # Load vector store based on embedding choice
            if emb_choice == "bge-small-en-v1.5":
                vector_store = init_rag()
            else:
                vector_store = init_rag_with_model(emb_choice)

            # Chat history
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

            welcome_query = None
            chat_container = st.container(height=550)
            with chat_container:
                welcome_placeholder = st.empty()
                if not st.session_state.get("chat_history"):
                    with welcome_placeholder.container():
                        st.markdown(
                            '<div style="height:120px;"></div>',
                            unsafe_allow_html=True,
                        )
                        _, img_col, _ = st.columns([2, 1, 2])
                        with img_col:
                            st.image("data/anime_edit.png", use_container_width=True)
                        st.markdown(
                            '<p style="text-align:center; color:#888; font-size:0.95rem; '
                            'margin-bottom:0.5rem;">'
                            "Ask me anything about your strategy, performance, or Elder's teachings."
                            "</p>",
                            unsafe_allow_html=True,
                        )
                        welcome_wrap = st.container(key="welcome-ask-wrap")
                        with welcome_wrap:
                            wc1, wc2, wc3 = st.columns(3)
                            if wc1.button("Sharpe Ratio?", key="w1"):
                                welcome_query = "How is the Sharpe Ratio calculated in our system?"
                            if wc2.button("Triple Screen?", key="w2"):
                                welcome_query = "What is Elder's Triple Screen Trading System and how does it work?"
                            if wc3.button("My Results", key="w3"):
                                welcome_query = "Analyze my current backtest results and explain the trade-offs."
                for msg in st.session_state["chat_history"]:
                    avatar = "data/anime_edit.png" if msg["role"] == "assistant" else None
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.write(msg["content"])

            # Quick ask buttons (pill style) + Clear
            generating = st.session_state.get("generating", False)
            quick_query = None
            action_bar = st.container(key="action-bar")
            with action_bar:
                if st.button("Explain Setup", key="q1", disabled=generating):
                    quick_query = "Explain my current strategy setup and what each parameter means."
                if st.button("Explain Perf", key="q2", disabled=generating):
                    quick_query = "Analyze my current backtest results and explain the trade-offs."
                if st.button("Risk?", key="q3", disabled=generating):
                    quick_query = "What are the risks of my current strategy configuration?"
                if st.button("Clear", key="clear_chat", disabled=generating):
                    st.session_state["chat_history"] = []
                    st.rerun()

            # Chat input
            user_input = st.chat_input("Ask about Elder's strategy...", disabled=generating)
            query = welcome_query or quick_query or user_input

            # Step 1: New query → save it, set generating, rerun to disable UI
            if query and not generating:
                try:
                    welcome_placeholder.empty()
                except Exception:
                    pass
                st.session_state["chat_history"].append(
                    {"role": "user", "content": query}
                )
                st.session_state["pending_query"] = query
                st.session_state["generating"] = True
                st.rerun()

            # Step 2: After rerun with generating=True, process the pending query
            pending = st.session_state.pop("pending_query", None)
            if pending and generating:
                dashboard_ctx = st.session_state.get("backtest_results", None)

                try:
                    with st.status("Running RAG pipeline...", expanded=True) as status:
                        status.update(label="Embedding query...", state="running")
                        retrieved = retrieve(vector_store, pending, k=5)
                        status.update(label=f"Retrieved {len(retrieved)} chunks", state="running")
                        prompt = build_prompt(pending, retrieved, dashboard_ctx)
                        status.update(label=f"Generating response ({model_choice})...", state="running")
                        response = generate_response(prompt, model_choice)
                        status.update(label="Done", state="complete", expanded=False)

                    model_short = model_choice.split(" (")[0]
                    meta = f"\n\n---\n*{model_short} | {len(retrieved)} chunks retrieved*"
                    full_response = response + meta

                    # Stream response inside chat container
                    import time as _time
                    def _stream_words():
                        for word in full_response.split(" "):
                            yield word + " "
                            _time.sleep(0.03)

                    with chat_container:
                        with st.chat_message("assistant", avatar="data/anime_edit.png"):
                            st.write_stream(_stream_words())

                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": full_response}
                    )
                    st.session_state["generating"] = False
                    st.rerun()
                except Exception as e:
                    st.session_state["generating"] = False
                    err_msg = str(e)
                    if "rate_limit" in err_msg or "429" in err_msg:
                        st.error("Rate limit reached. Try switching to the other model or wait a few minutes.")
                    else:
                        st.error("Failed to generate response. Please check your API keys in .env file.")
                    if st.session_state["chat_history"] and st.session_state["chat_history"][-1]["role"] == "user":
                        st.session_state["chat_history"].pop()

    if col_chat is not None:
        with col_chat:
            copilot_chat()
