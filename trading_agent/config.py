"""
配置管理

使用方式：
1. 复制 config.example.json 为 data/config.json
2. 填入你的钱包私钥和 DeepSeek API key

也可通过环境变量配置：
  DEEPSEEK_API_KEY 覆盖 API key
"""

import os
import json

# DeepSeek - 环境变量 > data/config.json > 报错
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    # fallback: 从 data/config.json 读取
    _cfg_path = os.path.join(os.path.dirname(__file__), "data", "config.json")
    if os.path.exists(_cfg_path):
        try:
            with open(_cfg_path, encoding="utf-8") as _f:
                _cfg = json.load(_f)
            DEEPSEEK_API_KEY = _cfg.get("deepseek_api_key", "")
        except Exception:
            pass
if not DEEPSEEK_API_KEY:
    print("  [配置] 警告: DEEPSEEK_API_KEY 未设置，请在环境变量中配置")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# ========== 安全限制（防止 AI 乱来）==========
# 单笔最大仓位：钱包总资金的百分比
MAX_POSITION_SIZE_PCT = 80  # 80%（激进模式）

# 最大杠杆
MAX_LEVERAGE = 3  # 最高 3 倍

# 每日最大亏损：超过这个比例就停止自动交易
DAILY_LOSS_LIMIT_PCT = -5  # -5%，日亏 5% 就停

# 只允许交易的币种（空 = 不限）
ALLOWED_COINS = []  # 例如 ["BTC", "ETH"]

# 自动交易间隔（秒）
AUTO_TRADE_INTERVAL = 300  # 5 分钟
