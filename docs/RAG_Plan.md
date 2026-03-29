# RAG Pipeline Plan

---

## 一、数据准备

### 数据源

| 来源 | 格式 | 数量 |
|---|---|---|
| knowledge_base.csv | QA pairs (Question, Answer, Label, Source) | 877 条 |
| elder_new_high_new_low.txt | 纯文本（视频讲解） | 1 篇 (~5,400 字符) |

### 分块策略（混合）

**QA 数据：不分块**
- 每条 Q+A 拼接为一个完整 chunk：`"{Question} {Answer}"`
- 理由：每条 QA 本身就是一个完整知识单元，切开会破坏语义

**Raw 文本：RecursiveCharacterTextSplitter**
- chunk_size = 400
- chunk_overlap = 50
- 理由：视频文本是连续口语，需要切成小块保证检索粒度

---

## 二、Embedding

### 模型选择：`BAAI/bge-small-en-v1.5`

| 对比 | bge-small-en-v1.5 | all-MiniLM-L6-v2 | bge-base-en-v1.5 |
|---|---|---|---|
| 维度 | 384 | 384 | 768 |
| 大小 | ~130MB | ~80MB | ~440MB |
| MTEB 排名 | 高 | 中 | 更高 |
| 速度 | 快 | 快 | 较慢 |

选择理由：效果比 MiniLM 更好，体积可接受，速度够快

### Embed 内容：Q+A 整条

- 将 Question 和 Answer 拼接后整体 embed
- 用户问问题时 → Question 部分命中
- 用户描述内容时 → Answer 部分命中
- 两侧都能检索到，覆盖更全

---

## 三、向量索引

### FAISS IndexFlatL2（暴力搜索）

- 877 条 QA + 约 15 条文本 chunk ≈ 900 条
- 900 条用暴力搜索毫秒级返回，不需要近似索引
- 最准确，无信息损失

---

## 四、检索策略

### 基础方案：Top-k = 5

- 返回与用户问题最相似的 5 条 chunk
- 将 5 条 chunk 拼接为 context 注入 prompt

### Prompt 结构

```
## Current Dashboard State
- Ticker: {ticker}
- Date Range: {start} to {end}
- Parameters: EMA={ema_short}/{ema_long}, RSI={rsi_lower}-{rsi_upper}, Breakout={bw}

## Backtest Results
- Total Return: {return_pct}%
- Sharpe Ratio: {sharpe}
- Max Drawdown: {max_dd}%
- Trade Count: {trades}
- Risk Level: {risk_level}

## Retrieved Knowledge (from Elder's books and teachings)
{rag_context}

## User Question
{question}

Answer based on the retrieved knowledge and current dashboard state.
If the question is about the current setup or performance, use the dashboard data.
If the question is about Elder's strategy or concepts, use the retrieved knowledge.
Cite Elder's specific insights when relevant.
```

---

## 五、Generation

### 双模型

| 模式 | 模型 | API |
|---|---|---|
| Mode A | Llama 3 (8B) | Groq（免费、快） |
| Mode B | GPT-4o-mini | OpenAI（高质量、便宜） |

用户可在 sidebar 切换

---

## 六、完整数据流

```
用户提问
    ↓
bge-small-en-v1.5 生成 query embedding
    ↓
FAISS IndexFlatL2 检索 top-5 最相似 chunks
    ↓
拼接：Dashboard Context + Retrieved Chunks + User Question
    ↓
送入 Llama 3 或 GPT-4o-mini 生成回答
    ↓
显示在 Chatbot 界面
```

---

## 七、如果效果不好的优化方向

### 问题 1：检索结果不相关

**症状：** 返回的 top-5 和用户问题关系不大

**优化方案：**
- 加 similarity score threshold（如 > 0.7 才返回），过滤低质量结果
- 尝试 Question 加权：embed 时把 Question 放前面重复一次，增大问题部分的权重
- 切换 embedding 模型到 `bge-base-en-v1.5`（更大更准）

### 问题 2：返回内容重复

**症状：** top-5 里有 3 条都在说同一个概念（如多条都讲 RSI）

**优化方案：**
- 使用 MMR（Maximum Marginal Relevance）检索代替纯 top-k
- `FAISS.max_marginal_relevance_search(query, k=5, fetch_k=20, lambda_mult=0.5)`
- MMR 在相关性和多样性之间平衡，避免重复

### 问题 3：QA 数据量大但检索命中太泛

**症状：** 635 条 Base Knowledge 中的 Personal Life 类内容经常混入结果

**优化方案：**
- 给 chunk 加 metadata filter（按 Label 过滤）
- 或从 knowledge_base.csv 中移除/降权 Personal Life 类条目（107 条）

### 问题 4：Context 太长导致回答质量下降

**症状：** 5 条 chunk 拼起来太长，模型抓不住重点

**优化方案：**
- 减少 top-k 到 3
- 加 Reranker：用 `cross-encoder/ms-marco-MiniLM-L-6-v2` 对 top-20 重排，只取 top-3
- Reranker 比 embedding 更精准但更慢，适合二次筛选

### 问题 5：用户问 dashboard 相关问题但 RAG 返回无关理论

**症状：** 用户问 "Why is my drawdown so high?" 但 RAG 返回 Elder 个人经历

**优化方案：**
- 在 prompt 中明确指示：dashboard 相关问题优先用 Dashboard Context 回答
- 或对 dashboard 类问题跳过 RAG 检索，直接用 Context Builder + LLM

---

## 八、实现顺序

1. 加载 knowledge_base.csv + elder_new_high_new_low.txt
2. 分块（QA 不切，文本 RecursiveCharacterTextSplitter）
3. bge-small-en-v1.5 生成 embeddings
4. 存入 FAISS
5. 接入 Streamlit chatbot UI
6. 接入 Groq / OpenAI generation
7. 测试 + 优化
