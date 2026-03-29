# Elder Trading Copilot — System Methodology

How every indicator, signal, and metric is calculated in our backtest engine.

---

## 1. Indicators

### EMA (Exponential Moving Average)

**Formula:**

$$EMA_t = \alpha \times Price_t + (1 - \alpha) \times EMA_{t-1}$$

where $\alpha = \frac{2}{span + 1}$

**Example (EMA Short = 15):**
- $\alpha = 2 / (15 + 1) = 0.125$
- If yesterday's EMA = 400 and today's Close = 410:
- $EMA_{today} = 0.125 \times 410 + 0.875 \times 400 = 401.25$

**In our system:**
- **EMA Short** (default 15): Tactical trend — reacts quickly to price changes
- **EMA Long** (default 200): Strategic trend — slow-moving, filters out noise

---

### MACD (Moving Average Convergence Divergence)

**Formula:**

$$MACD_{line} = EMA(12) - EMA(26)$$
$$Signal_{line} = EMA(9)\ of\ MACD_{line}$$

**Example:**
- If EMA(12) = 405 and EMA(26) = 400 → MACD line = +5 (bullish)
- If Signal line = +3 → MACD > Signal → bullish momentum confirmed

**Note:** MACD parameters (12, 26, 9) are fixed and not user-adjustable.

---

### RSI (Relative Strength Index)

**Step-by-step:**

1. Calculate daily price change: $\Delta_t = Close_t - Close_{t-1}$
2. Separate gains and losses:
   - $Gain_t = \Delta_t$ if $\Delta_t > 0$, else $0$
   - $Loss_t = |\Delta_t|$ if $\Delta_t < 0$, else $0$
3. Average over 14 days (rolling window):
   - $AvgGain = mean(Gain_{t-13}\ ...\ Gain_t)$
   - $AvgLoss = mean(Loss_{t-13}\ ...\ Loss_t)$
4. Relative Strength:
   $$RS = \frac{AvgGain}{AvgLoss}$$
5. RSI:
   $$RSI = 100 - \frac{100}{1 + RS}$$

**Example:**
- Over last 14 days: AvgGain = 1.5, AvgLoss = 0.5
- RS = 1.5 / 0.5 = 3
- RSI = 100 - 100 / (1 + 3) = 100 - 25 = **75** (overbought at default threshold)

**In our system:**
- RSI period is fixed at 14 days
- RSI Lower (default 40): Below this → oversold → close short positions
- RSI Upper (default 75): Above this → overbought → close long positions

---

## 2. Trading Logic (Elder Triple Screen)

### Entry Conditions

**Long Entry** — ALL three screens must agree:

| Screen | Condition | Meaning |
|--------|-----------|---------|
| Screen 1 (Trend) | EMA_short > EMA_long | Market is in an uptrend |
| Screen 2 (Momentum) | MACD > Signal line | Bullish momentum is active |
| Screen 3 (Entry) | Close > N-day highest high | Price breaks out above recent range |

**Short Entry** — ALL three screens must agree:

| Screen | Condition | Meaning |
|--------|-----------|---------|
| Screen 1 (Trend) | EMA_short < EMA_long | Market is in a downtrend |
| Screen 2 (Momentum) | MACD < Signal line | Bearish momentum is active |
| Screen 3 (Entry) | Close < N-day lowest low | Price breaks down below recent range |

### Exit Conditions

| Position | Exit When | Meaning |
|----------|-----------|---------|
| Long | RSI > RSI Upper | Overbought — take profit |
| Short | RSI < RSI Lower | Oversold — take profit |

### Position Sizing

- **Full capital allocation**: `shares = floor(cash / price)`
- Starting capital: **$10,000**
- No fractional shares, no margin, no commissions, no slippage

### Example Trade

Suppose EMA Short=15, EMA Long=200, Breakout=5, RSI Upper=75:

1. Day 100: EMA(15) = 405 > EMA(200) = 390 → **Screen 1: uptrend** ✓
2. Day 100: MACD = +2.5 > Signal = +1.8 → **Screen 2: bullish** ✓
3. Day 100: Close = 412 > 5-day high = 410 → **Screen 3: breakout** ✓
4. → **Enter Long**: buy floor(10000 / 412) = 24 shares @ $412
5. Day 115: RSI rises to 78 > 75 → **Overbought exit**
6. → **Close Long**: sell 24 shares @ current price

---

## 3. Performance Metrics

### Total Return

$$Total\ Return = \frac{Final\ Equity - 10000}{10000} \times 100\%$$

**Example:** Final equity = $16,565 → Total Return = (16565 - 10000) / 10000 × 100 = **65.65%**

---

### Sharpe Ratio

$$Sharpe = \frac{\bar{r}}{\sigma_r} \times \sqrt{252}$$

where:
- $\bar{r}$ = mean of daily returns
- $\sigma_r$ = standard deviation of daily returns
- $\sqrt{252}$ = annualization factor (252 trading days/year)
- Risk-free rate $R_f$ = 0 (assumed)

**Full formula with risk-free rate:**

$$Sharpe = \frac{R_p - R_f}{\sigma_p}$$

Our simplified version is equivalent when $R_f = 0$:
- $R_p = \bar{r} \times 252$ (annualized return)
- $\sigma_p = \sigma_r \times \sqrt{252}$ (annualized volatility)
- $\frac{\bar{r} \times 252}{\sigma_r \times \sqrt{252}} = \frac{\bar{r}}{\sigma_r} \times \sqrt{252}$

**Example:**
- Daily returns: mean = 0.0004, std = 0.015
- Sharpe = (0.0004 / 0.015) × √252 = 0.0267 × 15.87 = **0.42**

**Interpretation:**
- Sharpe > 1.0: Good risk-adjusted return
- Sharpe > 2.0: Excellent
- Sharpe < 0: Losing money

---

### Max Drawdown

$$Max\ Drawdown = \max_t \left( \frac{Peak_t - Equity_t}{Peak_t} \right) \times 100\%$$

where $Peak_t = \max(Equity_1, Equity_2, ..., Equity_t)$ is the running maximum.

**Example:**
- Equity rises to $15,000 (peak), then drops to $12,000
- Drawdown = (15000 - 12000) / 15000 = 20%
- If this is the largest drop → Max Drawdown = **20%**

**Interpretation:** The worst peak-to-trough decline. A 33% max drawdown means at some point the portfolio lost a third of its peak value.

---

### Trade Count

$$Trade\ Count = Number\ of\ Long\ Entries + Number\ of\ Short\ Entries$$

Each entry signal counts as one trade, regardless of whether it was profitable.

---

### Risk Level

| Max Drawdown | Risk Level |
|-------------|------------|
| < 15% | Low |
| 15% – 30% | Medium |
| >= 30% | High |

---

## 4. Parameter Reference

| Parameter | Default | Range | Role |
|-----------|---------|-------|------|
| EMA Short | 15 | 5–50 | Tactical trend (Screen 1) |
| EMA Long | 200 | 50–300 | Strategic trend (Screen 1) |
| RSI Lower | 40 | 20–60 | Oversold exit threshold |
| RSI Upper | 75 | 40–95 | Overbought exit threshold |
| Breakout Window | 5 | 3–20 | N-day high/low for entry (Screen 3) |
| MACD Fast | 12 | Fixed | Fast EMA for MACD |
| MACD Slow | 26 | Fixed | Slow EMA for MACD |
| MACD Signal | 9 | Fixed | Signal line smoothing |
| RSI Period | 14 | Fixed | RSI lookback |
| Starting Capital | $10,000 | Fixed | Initial portfolio |
