# CSYE 7380 Final Project — Elder Trading Copilot

> 一个结合 Trading Strategy Dashboard 和 RAG-based Chatbot 的 context-aware 智能交易辅助系统

---

## 一、项目目标

构建一个围绕 Alexander Elder 交易理念的智能交易辅助系统，直接实现提升版（context-aware）。

系统包含两个核心部分：

1. **Trading Strategy Dashboard** — 展示基于 Elder 策略的交易逻辑、技术指标、买卖信号和回测结果
2. **Elder Strategy Chatbot** — 解释 Elder 交易框架、指标含义、风险控制逻辑，并结合 dashboard 当前状态进行上下文感知的解释

核心定位：**"会展示策略、会解释策略、会解释结果、并具有风险意识的交易系统"**

---

## 二、技术栈

| 层 | 选择 | 原因 |
|---|---|---|
| UI | Streamlit | Python 全栈，适合 data app / dashboard / chatbot，适合课程 demo |
| Data | yfinance | 免费获取历史股票数据 |
| Indicators | pandas 手写 MA/RSI/MACD | 简单可控，不依赖额外库 |
| Chunking | QA 不分块 + RecursiveCharacterTextSplitter（文本） | QA 整条是一个知识单元；文本需要切块 |
| Embedding | BAAI/bge-small-en-v1.5 | 比 paraphrase-MiniLM 效果更好，体积相近，MTEB 排行榜靠前 |
| Vector Store | FAISS (IndexFlatL2) | 暴力搜索，~900 条数据毫秒级返回，最准确 |
| Retrieval | Top-k=5 similarity search | 基础方案，效果不好可加 MMR / Reranker / threshold |
| Generation 1 | Groq Llama 3.3 (70B) | 免费、推理快、效果好 |
| Generation 2 | OpenAI GPT-4o-mini | 便宜、质量高，与 Llama 对比有说服力 |
| Framework | LangChain | 统一 RAG pipeline 管理 |

### 数据流

```
用户提问
    ↓
bge-small-en-v1.5 生成 query embedding
    ↓
FAISS IndexFlatL2 检索 top-5 最相似 chunks
    ↓
拼接：Dashboard Context + Retrieved Chunks + User Question
    ↓
Groq Llama 3.3 / GPT-4o-mini 生成回答
    ↓
显示在 Chatbot 界面
```

---

## 三、系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
│  ┌────────────────────────────┬───────────────────────────────┐  │
│  │  Dashboard (60%)           │  Chatbot (40%)                │  │
│  │                            │                               │  │
│  │  [Metrics: Return/Sharpe/  │  [Context Summary Bar]        │  │
│  │   Drawdown/Trades/Risk]    │  [Model Selector]             │  │
│  │                            │  [Chat History]               │  │
│  │  Chart 1: Price + Signals  │  [Quick Ask Buttons]          │  │
│  │  Chart 2: RSI              │  [Chat Input]                 │  │
│  │  Chart 3: Equity + B&H     │                               │  │
│  └────────────┬───────────────┴───────────────┬───────────────┘  │
│               │                               │                  │
└───────────────┼───────────────────────────────┼──────────────────┘
                │                               │
                ▼                               ▼
      ┌───────────────────┐           ┌──────────────────────┐
      │ Backtest Engine   │           │   RAG Pipeline       │
      │                   │           │                      │
      │ - yfinance data   │           │ 1. Load from FAISS   │
      │ - EMA / RSI / MACD│           │ 2. Embed user query  │
      │ - Breakout (Scr3) │           │ 3. Top-5 retrieval   │
      │ - Full-position   │           │ 4. Build prompt      │
      │ - Metrics calc    │           │                      │
      └─────────┬─────────┘           └──────────┬───────────┘
                │                                │
                ▼                                ▼
      ┌───────────────────┐           ┌──────────────────────┐
      │ Context Builder   │──────────▶│ Generation Layer     │
      │                   │           │                      │
      │ - ticker / dates  │           │ - Groq Llama 3.3 70B │
      │ - strategy params │           │ - OpenAI GPT-4o-mini │
      │ - backtest metrics│           │                      │
      │ - risk level      │           │                      │
      └───────────────────┘           └──────────────────────┘
```

---

## 四、模块说明

### 模块 1：Trading Strategy Dashboard

**功能：**
- Sidebar：ticker 选择、日期范围、策略参数滑块（EMA Short/Long、RSI Lower/Upper、Breakout Window）
- 指标卡片：Total Return / Sharpe Ratio / Max Drawdown / Trade Count / Risk Level
- 三图联动（Plotly, shared x-axis）：Price+Signals / RSI / Equity Curve + Buy&Hold 基准线
- 参数调整实时更新（无需按钮，股票数据 cached）
- 模型选择在 chatbot 区域内

### 模块 2：Backtest Engine

**Triple Screen 交易逻辑（含 Screen 3 Breakout）：**
- Screen 1（趋势）：EMA_short > EMA_long
- Screen 2（动量）：MACD > MACD_signal
- Screen 3（入场）：Close > 前 N 日最高价（Breakout Window）
- 出场：RSI 超买/超卖

**全仓交易：** `shares = cash // price`，真实反映策略收益

### 模块 3：RAG Pipeline

**分块策略（混合）：**
- QA 数据（877 条）：不分块，每条 Q+A 拼接为一个 chunk
- Raw 文本（视频讲解）：RecursiveCharacterTextSplitter (chunk_size=400, overlap=50)

**Embedding：** Q+A 整条 embed（用户问问题命中 Q 部分，描述内容命中 A 部分）

**索引：** FAISS IndexFlatL2，预构建存盘，启动秒加载

**检索：** top-k=5

### 模块 4：Generation Layer（双模型对比）

| 模式 | 模型 | 特点 |
|---|---|---|
| Mode A | Llama 3.3 (70B) via Groq | 免费、快速 |
| Mode B | GPT-4o-mini via OpenAI | 高质量、便宜 |

### 模块 5：Context Builder（核心亮点）

把 dashboard 当前状态注入 chatbot prompt，实现 context-aware 解释。

**Structured Prompt Template:**
```
## Current Dashboard State
- Ticker: {ticker}
- Date Range: {start} to {end}
- Strategy: Elder Triple Screen
- Parameters: EMA={ema_short}/{ema_long}, RSI={rsi_lower}-{rsi_upper}, Breakout={bw}

## Backtest Results
- Total Return: {return_pct}%
- Sharpe Ratio: {sharpe}
- Max Drawdown: -{max_dd}%
- Trade Count: {trades}
- Risk Level: {risk_level}

## Retrieved Knowledge (from Elder's books and teachings)
{rag_context}

## User Question
{question}
```

---

## 五、功能清单

### Dashboard
- [x] ticker/date 输入
- [x] strategy params 设置（含 Breakout Window）
- [x] 三图联动：Price+Signals / RSI / Equity Curve
- [x] Buy & Hold 基准线对比
- [x] backtest metrics 展示（5 指标卡片）
- [x] Risk Level 展示
- [x] 参数实时更新（无需按钮）

### Chatbot
- [x] RAG pipeline（加载、embed、索引、检索）
- [x] 预构建 FAISS 索引（秒启动）
- [x] Context Builder（dashboard 状态注入）
- [x] 双模型支持（Groq / OpenAI）
- [x] 聊天历史
- [x] Quick Ask 按钮（Explain Setup / Explain Perf / Risk?）
- [x] 独立 RAG Debug 页面（Pipeline 可视化）

---

## 六、RAG 知识库数据集

### 最终合并文件：`knowledge_base.csv` — 877 条 QA pairs

| 来源 | 数量 | 说明 |
|---|---|---|
| Base Knowledge Dataset | 635 | Elder 策略、风险、心理、时机、适应性 |
| The New Trading for a Living（手工提取） | 50 | MA、RSI、MACD、Triple Screen、风险管理原文 |
| Study Guide for Come Into My Trading Room（自动提取） | 100 | 100 道 QA 完整配对 |

**补充文本：** `elder_new_high_new_low.txt` — Elder 视频讲解（~5,400 字符，切块后 ~15 段）

### 数据处理流程

```
Raw Source (PDF / 视频字幕)
    ↓
提取 → QA pairs / 纯文本
    ↓
合并 → knowledge_base.csv + .txt
    ↓
python build_index.py（跑一次）
    ↓
faiss_index/（index.faiss + index.pkl）
    ↓
App 启动秒加载
```

---

## 七、RAG 优化方向（如果效果不好）

| 问题 | 优化方案 |
|---|---|
| 检索结果不相关 | 加 similarity score threshold；Q 加权 embed；换 bge-base |
| 返回内容重复 | 使用 MMR 检索（兼顾相关性和多样性） |
| Personal Life 干扰 | metadata filter 按 Label 过滤 |
| Context 太长 | 减 top-k 到 3；加 cross-encoder Reranker 二次筛选 |
| Dashboard 问题命中差 | prompt 指示优先用 Dashboard Context；dashboard 类问题跳过 RAG |
