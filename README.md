# Trading Agent — Hyperliquid 交易智能体

一个基于 ReAct 模式自主交易的 AI 智能体，连接 Hyperliquid 永续合约交易所，支持自动交易 + 复盘 + 教训记忆。

## 功能

- **ReAct 交易大脑** — AI 自己思考市场情况，决定开仓、平仓还是观望
- **自动运行** — 每 5 分钟检查一次市场，无人值守
- **向量记忆** — 交易教训自动存入，下次遇到类似行情会自动回想
- **复盘分析** — 自动统计盈亏、胜率，提取教训
- **创建钱包** — 首次使用自动生成 Hyperliquid 钱包

## 快速开始

```bash
# 启动
python -m trading_agent.main

# 首次启动 → 输入 Y 创建钱包 → 拿到地址
# 往地址转 USDC → 重新启动

# 自动模式
python -m trading_agent.main --auto
```

> 详细使用教程见 [docs/tutorial.md](docs/tutorial.md)

## 项目结构

```
trading_agent/
├── config.py          # API key、安全限制
├── hl_connector.py   # Hyperliquid 链上连接
├── memory.py         # 教训记忆 + 向量搜索
├── agent.py          # ReAct 循环大脑
└── main.py           # 入口（手动/自动）
```

## 安全限制

| 限制 | 默认值 |
|---|---|
| 单笔最大仓位 | 总资金 10% |
| 最高杠杆 | 3 倍 |
| 每日止损 | -5% 自动停止 |

可在 `config.py` 中调整。

## 技术栈

- Python 3.10+
- Anthropic DeepSeek API（决策 + 分析）
- BAAI/bge-small-zh-v1.5（中文向量模型）
- hyperliquid-python-sdk（链上交易）
