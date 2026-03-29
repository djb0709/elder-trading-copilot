# Dashboard Design Plan

---

## 一、布局结构

```
┌─────────────────────────────────────────────────────────────────┐
│  Sidebar (左侧)                                                  │
│  ┌───────────────────────┐                                      │
│  │ Ticker: [SPY    ▼]    │                                      │
│  │ Date Range:            │                                      │
│  │  Start: [2020-01-01]  │                                      │
│  │  End:   [2025-12-31]  │                                      │
│  │                        │                                      │
│  │ --- Strategy Params ---│                                      │
│  │ EMA Short:  [15  ◄►]  │                                      │
│  │ EMA Long:   [200 ◄►]  │                                      │
│  │ RSI Lower:  [50  ◄►]  │                                      │
│  │ RSI Upper:  [53  ◄►]  │                                      │
│  │ Breakout:   [5   ◄►]  │                                      │
│  │                        │                                      │
│  │ [▶ Run Backtest]       │                                      │
│  │                        │                                      │
│  │ --- Model Select ---   │                                      │
│  │ ○ Llama 3 (Groq)      │                                      │
│  │ ○ GPT-4o-mini          │                                      │
│  └───────────────────────┘                                      │
├─────────────────────────────────────────────────────────────────┤
│  Main Area (右侧)                                                │
│                                                                  │
│  ┌─── Tab 1: Dashboard ──────────────────────────────────────┐  │
│  │                                                            │  │
│  │  [Total Return] [Sharpe] [Max Drawdown] [Trades] [Risk]   │  │
│  │   +12.3%        1.24     -8.7%          23       Medium   │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │  Chart 1: Price + EMA + Buy/Sell Signals             │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │  Chart 2: RSI + Overbought/Oversold Lines            │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │  Chart 3: Equity Curve                               │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─── Tab 2: Chatbot ────────────────────────────────────────┐  │
│  │                                                            │  │
│  │  [Current: SPY | EMA 15/200 | RSI 50-53 | Return +12.3%] │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │  Chat history (st.chat_message)                      │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  │  Quick Ask: [Explain Setup] [Explain Performance] [Risk?] │  │
│  │                                                            │  │
│  │  [Type your question...                        ] [Send]   │  │
│  │                                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、Sidebar 控件

| 控件 | 组件 | 默认值 | 范围 |
|---|---|---|---|
| Ticker | `st.selectbox` | SPY | SPY, AAPL, TSLA, MSFT, QQQ + 自定义输入 |
| Start Date | `st.date_input` | 2020-01-01 | — |
| End Date | `st.date_input` | 2025-12-31 | — |
| EMA Short | `st.slider` | 15 | 5 - 50 |
| EMA Long | `st.slider` | 200 | 50 - 300 |
| RSI Lower | `st.slider` | 50 | 20 - 60 |
| RSI Upper | `st.slider` | 53 | 40 - 80 |
| Breakout Window | `st.slider` | 5 | 3 - 20 |
| Run Backtest | `st.button` | — | — |
| Model Select | `st.radio` | Llama 3 | Llama 3 (Groq) / GPT-4o-mini |

---

## 三、Triple Screen 交易逻辑（含 Screen 3 Breakout）

### 指标计算

```python
# Screen 1: 趋势
data["EMA_short"] = ema(data["Close"], win_short)
data["EMA_long"]  = ema(data["Close"], win_long)

# Screen 2: 动量
data["MACD"], data["MACD_signal"] = macd(data["Close"])
data["RSI"] = rsi(data["Close"])

# Screen 3: 入场突破
data["Breakout_high"] = data["High"].rolling(window=breakout_window).max()
data["Breakout_low"]  = data["Low"].rolling(window=breakout_window).min()
```

### 入场/出场规则

**做多入场（三个条件全部满足）：**
1. Screen 1 — EMA_short > EMA_long（趋势向上）
2. Screen 2 — MACD > MACD_signal（动量向上）
3. Screen 3 — Close > Breakout_high（突破最近 N 根 K 线高点）

**做多出场：** RSI > RSI_upper（超买离场）

**做空入场（三个条件全部满足）：**
1. Screen 1 — EMA_short < EMA_long（趋势向下）
2. Screen 2 — MACD < MACD_signal（动量向下）
3. Screen 3 — Close < Breakout_low（跌破最近 N 根 K 线低点）

**做空出场：** RSI < RSI_lower（超卖离场）

---

## 四、指标卡片（Metrics Row）

用 `st.columns(5)` + `st.metric` 横排展示：

| 指标 | 计算方式 |
|---|---|
| Total Return | `(final_equity - initial_capital) / initial_capital * 100` |
| Sharpe Ratio | `mean(daily_returns) / std(daily_returns) * sqrt(252)` |
| Max Drawdown | `max((peak - trough) / peak)` 基于 equity curve |
| Trade Count | 总入场次数（多 + 空） |
| Risk Level | 基于 Max Drawdown：< 5% Low (绿) / 5-15% Medium (黄) / > 15% High (红) |

---

## 五、三图结构

用 Plotly `make_subplots(rows=3, cols=1, shared_xaxes=True)` 实现 x 轴联动。

### Chart 1: Price + Signals（主图，占比 50%）

- 收盘价折线（或 candlestick）
- EMA Short 线（蓝色）
- EMA Long 线（橙色）
- Breakout High 线（虚线灰色，可选展示）
- 做多入场标记（绿色三角 ▲）
- 做空入场标记（红色三角 ▼）

### Chart 2: RSI（占比 25%）

- RSI 曲线
- RSI Lower 水平线（虚线）
- RSI Upper 水平线（虚线）
- 超买/超卖区域填充色（浅红 / 浅绿）

### Chart 3: Equity Curve（占比 25%）

- Equity 折线
- 初始资金水平基准线（灰色虚线 $10,000）
- 盈利区域绿色填充 / 亏损区域红色填充

---

## 六、Chatbot Tab

### Context Summary Bar

顶部一行灰色背景展示当前 dashboard 状态：
```
SPY | 2020-01-01 ~ 2025-12-31 | EMA 15/200 | RSI 50-53 | Breakout 5 | Return +12.3% | Sharpe 1.24
```

用户一眼知道 bot 在基于什么回答。

### Chat 界面

- `st.chat_message` 展示历史对话
- `st.chat_input` 输入框

### Quick Ask 按钮

三个 `st.button` 横排，点击自动发送预设问题：
- **Explain Setup** → "Explain my current strategy setup and what each parameter means"
- **Explain Performance** → "Analyze my current backtest results and explain the trade-offs"
- **Risk?** → "What are the risks of my current strategy configuration?"

---

## 七、Streamlit 组件总结

| UI 元素 | 组件 |
|---|---|
| Ticker 输入 | `st.sidebar.selectbox` |
| 日期范围 | `st.sidebar.date_input` |
| 参数滑块 | `st.sidebar.slider` |
| 运行按钮 | `st.sidebar.button` |
| 模型切换 | `st.sidebar.radio` |
| Tab 切换 | `st.tabs(["Dashboard", "Chatbot"])` |
| 指标卡片 | `st.columns(5)` + `st.metric` |
| 三图联动 | `st.plotly_chart` + `make_subplots` |
| 聊天界面 | `st.chat_message` + `st.chat_input` |
| Quick Ask | `st.columns(3)` + `st.button` |
| Context Bar | `st.caption` 或 `st.info` |
