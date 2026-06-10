# 交易智能体使用教程

## 你需要准备

### 1. Hyperliquid 账户

1. 打开 [app.hyperliquid.xyz](https://app.hyperliquid.xyz) 创建钱包
2. 往主钱包充入 USDC（建议至少 $30，因为每笔最小 $10）

### 2. 创建 API 钱包（安全模式）

API 钱包是**受限**的——它只能交易，不能提现。就算私钥泄露，资金也转不走。

1. Hyperliquid 网站 → **API** 页面
2. 点击 "Create API Wallet"
3. 系统会给你三样东西：
   - **API 私钥**（`api_key`）→ 用于签名交易 ← **记下来**
   - **API 钱包地址** → 不需要
   - **主钱包地址**（`main_address`）→ 从页面顶部复制 ← **记下来**

### 3. DeepSeek API Key

1. 打开 [platform.deepseek.com](https://platform.deepseek.com) 注册
2. 充值（首次注册送几块钱额度，够跑很久）
3. 在 API Keys 页面创建一个 key，复制下来

## 配置步骤

### 1. 安装依赖

```bash
pip install openai hyperliquid-python-sdk fastembed numpy eth-account
```

### 2. 填写配置

编辑 `trading_agent/data/config.json`：

```json
{
    "main_address": "0x你的主钱包地址",
    "api_key": "0x你的API钱包私钥"
}
```

编辑 `trading_agent/config.py`，修改第 10 行的 DeepSeek Key：

```python
DEEPSEEK_API_KEY = "sk-你的DeepSeek API Key"
```

### 3. 启动

```bash
cd 项目目录
python -c "from trading_agent import agent; print(agent.run_trade_cycle())"
```

## 工作原理

| 组件 | 作用 |
|---|---|
| `agent.py` | AI 大脑，每轮收集数据 → DeepSeek 决策 → 执行交易 |
| `hl_connector.py` | Hyperliquid 连接器，封装所有交易所操作 |
| `memory.py` | 向量记忆系统，记住交易教训 |
| `config.py` | 配置（API Key、风险限制） |

AI 每轮会：
1. 查看余额、持仓、市场行情
2. 回想历史相关教训
3. 用 DeepSeek 分析并决定开/平/等
4. 执行交易并保存新教训

## 自动运行

启动持久化监控（每 5 分钟跑一次）：

```bash
# 在 Claude Code 中运行
/loop 5m 运行一次交易智能体周期并汇报
```

## 风险提示

- 默认单笔最大仓位 80%，最高杠杆 3 倍
- 亏损超过 5% 会自动停止交易
- 建议从小资金开始测试
