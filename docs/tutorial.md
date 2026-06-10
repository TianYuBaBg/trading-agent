# 交易智能体 — 使用教程

## 目录

1. [准备工作](#1-准备工作)
2. [Hyperliquid API 钱包](#2-hyperliquid-api-钱包)
3. [配置 DeepSeek](#3-配置-deepseek)
4. [启动智能体](#4-启动智能体)
5. [自动运行](#5-自动运行)
6. [AI 决策流程](#6-ai-决策流程)
7. [教训记忆系统](#7-教训记忆系统)
8. [安全机制](#8-安全机制)
9. [文件说明](#9-文件说明)

---

## 1. 准备工作

### 安装依赖

```bash
pip install openai hyperliquid-python-sdk fastembed numpy eth-account
```

### 注册账号

你需要两个东西：

| 用途 | 注册地址 |
|---|---|
| Hyperliquid 交易所 | [app.hyperliquid.xyz](https://app.hyperliquid.xyz) |
| DeepSeek AI（决策大脑） | [platform.deepseek.com](https://platform.deepseek.com) |

---

## 2. Hyperliquid API 钱包

### 什么是 API 钱包？

API 钱包是**受限钱包**——它只能交易，不能提现。即使私钥泄露，攻击者也转不走你的钱。

### 创建步骤

1. 打开 [app.hyperliquid.xyz](https://app.hyperliquid.xyz) → **API** 页面
2. 点击 **Create API Wallet**
3. 记下两样东西：

```
API 私钥（api_key）      → 用于签名交易  ← 保存好
主钱包地址（main_address） → 标识账户身份  ← 从页面顶部复制
```

API 钱包地址创建时也有显示，但代码里用不上。

### 填写配置

编辑 `trading_agent/data/config.json`：

```json
{
    "main_address": "0x你的主钱包地址",
    "api_key": "0x你的API钱包私钥"
}
```

> `data/config.json` 已在 `.gitignore` 中，不会提交到 GitHub。

### 存入资金

往你的**主钱包**充 USDC（建议至少 $30）：

```
交易所（币安/欧易等）
  → 提 USDC
  → 网络选择: Arbitrum One
  → 地址: 你的主钱包地址
```

Hyperliquid 每笔最小交易 $10，建议留点余量。

---

## 3. 配置 DeepSeek

1. 打开 [platform.deepseek.com](https://platform.deepseek.com) 注册
2. 充值（首次注册送几块钱额度，够跑很久）
3. 在 API Keys 页面创建一个 key

编辑 `trading_agent/config.py` 第 10 行：

```python
DEEPSEEK_API_KEY = "sk-你的DeepSeek Key"
```

---

## 4. 启动智能体

### 单次运行（测试用）

```bash
cd 项目目录
python -c "from trading_agent import agent; result = agent.run_trade_cycle(); print(result['reply'])"
```

AI 会执行一次完整交易周期：收集数据 → 分析市场 → 决定是否交易。

### 调试模式

想看 AI 每一轮的详细思考过程：

```bash
python trading_agent/debug_react.py
```

---

## 5. 自动运行

### 使用 Claude Code 持久化监控

在 Claude Code 中执行：

```
/loop 5m 运行一次交易智能体周期并汇报
```

这会每 5 分钟自动执行一次 `run_trade_cycle()`，并汇报 AI 决策和账户变化。重启 Claude Code 后会自动恢复。

### 取消自动运行

```
CronDelete id=任务ID
```

### 各文件说明

| 文件 | 作用 |
|---|---|
| `agent.py` | ReAct 循环大脑：收集数据 → DeepSeek 决策 → 执行 |
| `hl_connector.py` | Hyperliquid 连接器：行情查询 + 交易执行 |
| `memory.py` | 向量记忆系统：存储和搜索交易教训 |
| `config.py` | 配置：API Key、杠杆限制、风控参数 |

---

## 6. AI 决策流程

### ReAct 多轮循环

每轮交易周期，AI 按以下流程运行：

```
第 1 轮：收集当前状态（余额 + 持仓 + 行情 + 历史教训）
         → 发给 DeepSeek 分析
         → AI 返回需要调用的工具（查 K 线、查深度等）
         ↓
第 2 轮：工具结果返回给 AI
         → AI 根据新数据做最终决策（开仓/平仓/等待）
         ↓
第 3 轮（如果需要）：继续分析或执行
```

### 可用工具

| 工具 | 作用 |
|---|---|
| `get_market_overview` | 查看所有币种行情（前 20 个活跃币） |
| `get_candles(coin, interval, hours)` | 查看 K 线趋势（1m/5m/15m/1h/4h/1d） |
| `get_orderbook(coin)` | 查看买卖盘深度 |
| `get_positions` | 查看持仓和盈亏 |
| `get_balance` | 查看账户余额 |
| `open_trade(coin, side, size, leverage)` | 开仓（最小 $10） |
| `close_trade(coin)` | 平仓 |
| `recall_lessons(context)` | 回想历史教训 |
| `add_lesson(lesson, tags, outcome)` | 记录新教训 |

### 防重复保护

同一轮中，AI 对同一币种重复调用 `open_trade` 或 `close_trade` 会被自动跳过，防止意外翻倍仓位。

---

## 7. 教训记忆系统

### 工作原理

```
复盘/平仓时：
  AI 分析 → 提取教训 → 中文向量模型（BAAI/bge-small-zh-v1.5）
  → 转成向量 → 存入 data/lessons.json

交易时：
  当前市场情况 → 转成向量 → 搜索最相似的历史教训
  → 把相关教训放进 DeepSeek 提示词
  → AI 决策前先看到过去的经验
```

### 查看教训

```python
from trading_agent import memory
print(memory.get_all_lessons())
```

---

## 8. 安全机制

| 机制 | 说明 |
|---|---|
| API 钱包模式 | 私钥只能交易，不能提现 |
| 私钥本地存储 | 存在 `data/config.json`，不入 git |
| 单笔仓位限制 | 默认不超过总资金 80%（可调） |
| 最高杠杆 | 3 倍 |
| 每日止损 | 日亏损达 -5% 停止自动交易（可调） |
| 重复开仓保护 | 同一轮内对同一币种只执行一次开/平仓 |

### 风险参数配置

编辑 `trading_agent/config.py`：

```python
MAX_POSITION_SIZE_PCT = 80    # 单笔最大仓位比例（%）
MAX_LEVERAGE = 3              # 最高杠杆倍数
DAILY_LOSS_LIMIT_PCT = -5     # 每日亏损止损线（%）
```

> **风险提示**：永续合约交易有高风险，本软件仅供学习参考，不构成投资建议。建议从小资金开始测试。

---

## 9. 文件说明

```
项目目录/
├── trading_agent/
│   ├── config.py           # 配置：DeepSeek Key、杠杆、仓位限制
│   ├── hl_connector.py     # Hyperliquid 连接器：K线、订单簿、交易执行
│   ├── memory.py           # 教训记忆：向量存储 + 语义搜索
│   ├── agent.py            # ReAct 循环：DeepSeek 决策大脑
│   ├── main.py             # 入口（旧版，建议直接调 agent.py）
│   ├── auto_runner.py      # 循环运行脚本（旧版，已被 cron 替代）
│   ├── debug_react.py      # ReAct 调试工具：看 AI 每轮思考
│   ├── SETUP.md            # 快速配置指南
│   └── data/
│       ├── config.json     # 钱包 API key（不提交 git）
│       ├── trades.json     # 本地交易记录（备胎，主数据来自链上）
│       └── lessons.json    # 教训向量库
├── docs/
│   └── tutorial.md         # 本教程
├── README.md
└── .gitignore
```

### 各模块关系

```
你（手动触发 / cron 自动调度）
  ↓
agent.py.run_trade_cycle()    ← ReAct 多轮循环
  ├── 收集数据 → hl_connector（行情 + 持仓 + 余额）
  ├── 回想教训 → memory（向量搜索）
  ├── DeepSeek 分析 → 返回 JSON 决策
  ├── 循环调工具 → hl_connector（查K线、查深度、开平仓）
  └── 保存教训 → memory（新教训入库）
       ↓
结果：AI 回复 + 交易执行（或不交易）
```
