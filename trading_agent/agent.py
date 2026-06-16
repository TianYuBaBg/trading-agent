"""
交易智能体 — ReAct 循环大脑

工作方式：
  每轮循环：
    1. 收集市场数据（行情、持仓、余额）
    2. 回想历史教训（向量搜索）
    3. 发给 DeepSeek 做决策
    4. 执行交易或等待
    5. 记录教训
"""

import json
import os
import datetime
from openai import OpenAI, AuthenticationError
from trading_agent import config
from trading_agent import hl_connector
from trading_agent import memory


# ========== DeepSeek 客户端（懒加载）==========

_client = None


def _get_client():
    global _client
    if _client is None:
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置，请通过环境变量或 data/config.json 配置")
        _client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    return _client


# ========== 工具函数描述 ==========

_TOOL_DESCRIPTIONS = {
    "get_market_overview": "查看所有币种的最新行情（价格、成交量）",
    "get_orderbook": "查看某个币的买卖盘深度，参数：coin（币种，如 BTC），depth（深度档位，默认5）",
    "get_candles": "查看某个币的K线趋势，参数：coin（币种），interval（间隔：1m/5m/15m/1h/4h/1d，默认1h），hours（往回看几小时，默认24）",
    "get_positions": "查看当前持仓及盈亏",
    "get_balance": "查看账户总资产和可用余额",
    "open_trade": f"开仓交易，参数：coin（币种），side（long/short），size（USD金额≥{config.MIN_ORDER_USD}），leverage（杠杆倍数，默认1）",
    "close_trade": "平仓，参数：coin（币种），size（平仓数量，可选，不传则全平）",
    "recall_lessons": "回想过去类似情况学到的教训，参数：context（当前市场情况描述），top_k（返回几条，默认3）",
    "add_lesson": "记录一条新教训，参数：lesson（教训内容），tags（标签列表），outcome（结果）",
    "transfer_spot_to_perp": "把 USDC 从 Spot 转到永续合约账户（仅直连私钥模式有效），参数：amount（金额，0=全部转入）",
    "set_stop_loss": "设置止损触发单（价格触发后自动市价平仓），参数：coin（币种），stop_price（触发价格），size（数量，可选，默认全仓）。开仓后务必立即设置止损来保护仓位",
    "set_take_profit": "设置止盈触发单（价格到达目标后自动市价平仓），参数：coin（币种），target_price（目标价格），size（数量，可选，默认全仓）",
    "get_open_orders": "查看当前所有挂单（含已设置的止损止盈单），参数：coin（可选，指定币种）",
    "cancel_order": "取消挂单，参数：coin（币种），oid（订单ID）",
    "get_funding_rates": "查看资金费率排行（含年化APY、每千刀时收益），发现套利机会。正费率=做空收钱，负费率=做多收钱。参数：top_n（返回条数，默认20）",
    "get_funding_history": "查看某个币的资金费率历史趋势，判断费率稳定性（持续高/低还是脉冲）。参数：coin（币种），hours（往回看几小时，默认48）",
}


# ========== 构建系统提示词 ==========

def _build_system_prompt():
    return f"""你是 Hyperliquid 永续合约交易员，核心目标是积极发现机会并盈利。

## 交易规则
1. 所有持仓累计名义价值（持仓数量 x 价格，杠杆不影响名义价值）不超过总资金的 {config.MAX_POSITION_SIZE_PCT}%
2. 最高杠杆 {config.MAX_LEVERAGE} 倍
3. 每日亏损超过 {abs(config.DAILY_LOSS_LIMIT_PCT)}% 时停止交易
4. 每次开仓前先回想相关教训
5. 平仓后复盘并记录教训
6. 开仓前计算可开金额：wallet_value * {config.MAX_POSITION_SIZE_PCT}% - 已有持仓名义价值（持仓数量 x 价格），选择在此范围内的金额。注意杠杆不影响名义价值，只影响保证金占用。开仓需要账户有足够可用资金（available = wallet_value - 已用保证金），账户总资金（含 Spot）都可作为保证金
7. **核心目标：盈利。优先做资金费率套利（低风险稳定收入），其次做趋势交易。开仓后立即用 set_stop_loss 设置止损保护仓位，利润达标后用 set_take_profit 锁定利润**
8. **资金利用率优先：费率套利风险低（止损 Tight），尽量用满 {config.MAX_POSITION_SIZE_PCT}% 仓位上限。闲置资金 = 浪费收益，有高费率机会就果断加仓。费率套利仓位统一用 3x 杠杆，不要用 1x——3x 提高资金效率且风险可控**
9. 不要只会等待，没有明确风险信号就该尝试入场
10. **同时关注非加密永续合约**：SPX（标普500指数永续）和 PAXG（黄金代币）是 Hyperliquid 上的非加密资产，定期查看其K线趋势和订单簿深度，与传统 crypto 品种对照分析，发现趋势机会时同样可以交易
	11. **资金费率套利（低风险优先）**：用 get_funding_rates 查看费率排行。正费率=做空收钱，负费率=做多收钱。费率换算：funding_pct × 24 × 365 = APY。极端费率（funding_pct > 0.05% 即 APY > 438%）是高胜率信号。用 get_funding_history 看费率是否稳定——持续高费率比脉冲更可靠。**即使已有持仓，也值得平掉低潜力仓位换仓到高费率币种**——资金费是持续流入，方向判断正确时双重收益。优先选成交量大的币种（volume > 1000万U）避免流动性风险。
		**硬约束：禁止逆费率方向开仓。** 费率为负时必须做多（不可做空），费率为正时必须做空（不可做多）。趋势分析和K线形态仅用于选择入场时机，不能用于推翻费率方向。如果当前持仓方向与费率方向相反，必须立即平仓纠正。
	12. **让利润奔跑，不设微利止盈**：费率套利的利润主要来自资金费率持续流入，不要赚 0.3% 就跑。用 set_take_profit 在合理目标价位挂止盈单（如支撑/阻力位、前高/前低），让仓位有空间走出趋势。平仓理由应该是费率回归正常、趋势反转或找到更好机会，而不是为了锁定微利
	13. **分散风险**：当可用资金 > $100 时，开至少 2 个不同币种，单币名义价值不超过总资金的 300%。不要把所有资金压在一个币上，至少找 2 个有高费率机会的币同时持仓。如果当前只有一个币有仓位但市场还有别的机会，加仓新币而不是继续堆旧币
	14. 每天最多开仓 3 次，达到上限后只能管理现有持仓（平仓/调SL/TP），不能开新仓
	15. **远离被投票下架的币**：TON 正在进行下架投票（Validator vote to delist TON），不要开 TON 新仓位，已有 TON 仓位应立即平仓

## 可用工具
{json.dumps(_TOOL_DESCRIPTIONS, ensure_ascii=False, indent=2)}

## 输出格式
每次返回 JSON：
{{
    "thought": "你的分析和决策思考过程",
    "tool_calls": [
        {{"name": "工具名", "args": 参数对象或{{}}}}
    ],
    "reply": "你要回复的话，没有就说空",
    "need_review": false,
    "lesson_to_save": ""  // 也可传对象 {{"text": "教训", "tags": ["btc"], "outcome": "亏损2%"}}
}}

流程说明：
16. 第一轮：分析当前状态，如果数据不够就调工具获取更多信息
17. 工具结果会返回给你，根据新数据做最终决策
18. tool_calls 为空数组 = 最终决策完成（不交易或已执行）
19. 如果发现交易机会，果断用 open_trade 开仓；如果持仓方向错了，close_trade 平仓
	20. **开仓后务必立即用 set_stop_loss 设置止损**，利润达标时用 set_take_profit 止盈"""


# ========== 每日开仓计数 ==========
_DAILY_TRADE_FILE = os.path.join(config.DATA_DIR, "daily_trade_count.json")


def _get_daily_open_count() -> int:
    """读取当日已开仓次数，跨日自动重置"""
    today = datetime.date.today().isoformat()
    try:
        if os.path.exists(_DAILY_TRADE_FILE):
            with open(_DAILY_TRADE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("date") == today:
                return int(data.get("count", 0))
    except Exception:
        pass
    return 0


def _increment_daily_open_count():
    """当日开仓计数 +1"""
    today = datetime.date.today().isoformat()
    count = _get_daily_open_count() + 1
    try:
        with open(_DAILY_TRADE_FILE, "w", encoding="utf-8") as f:
            json.dump({"date": today, "count": count}, f, ensure_ascii=False)
    except OSError:
        pass  # 不影响交易流程


# ========== 防重复执行 ==========
# 记录当前周期已执行过的操作，防止 AI 同一轮重复开仓
_executed_ops: set[str] = set()

# 工具结果排序优先级：交易操作优先，避免截断丢失关键信息
_KEY_OPS = {"open_trade", "close_trade", "set_stop_loss", "set_take_profit", "transfer_spot_to_perp"}


def _reset_executed_ops():
    """每轮交易循环开始时调用"""
    global _executed_ops
    _executed_ops = set()


def _truncate_results(items: list, max_len: int = 8000) -> str:
    """在完整 item 边界截断 JSON 列表，避免切在字符串中间；单条超限时至少保留第一条"""
    # 先检查完整长度：不超限时保持原始调用顺序返回，避免 AI 看到错序的结果
    payload = json.dumps(items, ensure_ascii=False)
    if len(payload) <= max_len:
        return payload
    # 超限时按 KEY_OPS 优先级排序，确保交易操作结果不被截断
    ordered = [item for _, item in sorted(enumerate(items), key=lambda r: (0 if str(r[1].get("tool") or "") in _KEY_OPS else 1, r[0]))]
    safe = []
    size = 2  # 预留 "[]" 括号的空间
    for item in ordered:
        s = json.dumps(item, ensure_ascii=False)
        needed = len(s) + (1 if safe else 0)  # 逗号分隔符仅 1 字节
        if size + needed > max_len:
            break
        safe.append(s)
        size += needed
    if not safe and ordered:
        safe.append(json.dumps(ordered[0], ensure_ascii=False))
    return "[" + ",".join(safe) + "]"


# ========== 执行工具调用 ==========

def _run_tool(name: str, args: dict, _balance=None, _positions=None, _funding_rates=None):
    """执行一个工具调用并返回原始 Python 对象（非 JSON 字符串）
    _balance/_positions/_funding_rates: 调用方已查询的数据，用于避免重复 RPC"""

    if not isinstance(args, dict):
        return {"error": f"工具 {name} 参数格式错误：需要 JSON 对象，收到 {type(args).__name__}"}

    if name == "get_market_overview":
        return hl_connector.get_market_overview()
    elif name == "get_orderbook":
        coin = str(args.get("coin") or "").strip().upper()
        if not coin:
            return {"error": "获取订单簿失败：缺少必填参数 coin"}
        try:
            depth = int(float(args.get("depth") or 5))
        except (TypeError, ValueError):
            return {"error": f"get_orderbook depth 参数无效：'{args.get('depth')}'"}
        return hl_connector.get_orderbook(coin, depth)
    elif name == "get_candles":
        coin = str(args.get("coin") or "").strip().upper()
        if not coin:
            return {"error": "获取K线失败：缺少必填参数 coin"}
        interval = str(args.get("interval") or "1h").strip().lower()
        if interval not in ("1m", "5m", "15m", "1h", "4h", "1d"):
            return {"error": f"get_candles interval 参数无效：'{interval}'，有效值：1m/5m/15m/1h/4h/1d"}
        try:
            hours = int(float(args.get("hours") or 24))
        except (TypeError, ValueError):
            return {"error": f"get_candles hours 参数无效：'{args.get('hours')}'"}
        return hl_connector.get_candles(coin, interval, hours)
    elif name == "get_positions":
        return hl_connector.get_positions()
    elif name == "get_balance":
        return hl_connector.get_balance()
    elif name == "open_trade":
        # 检查必填参数（coin 统一转大写，避免大小写不一致导致去重/白名单/API 调用出问题）
        coin = str(args.get("coin") or "").strip().upper()
        side = str(args.get("side") or "").strip().lower()
        size = args.get("size")
        if not coin:
            return {"error": "开仓失败：缺少必填参数 coin"}
        if not side:
            return {"error": "开仓失败：缺少必填参数 side"}
        if size is None:
            return {"error": "开仓失败：缺少必填参数 size"}
        if side not in ("long", "short"):
            return {"error": f"开仓失败：side 必须为 long 或 short，收到: {side}"}
        # 检查币种白名单
        if config.ALLOWED_COINS and coin not in config.ALLOWED_COINS:
            return {"error": f"{coin} 不在允许交易列表(ALLOWED_COINS)中"}
        key = f"open_trade:{coin}"
        if key in _executed_ops:
            return {"warning": f"已在当前周期对 {coin} 开过仓，操作已跳过"}
        try:
            size_val = float(size)
            if size_val <= 0:
                return {"error": f"开仓失败：size 必须大于 0，收到 {size}"}
        except (TypeError, ValueError):
            return {"error": f"开仓失败：size 参数格式无效（{size}）"}
        try:
            leverage_val = int(float(args.get("leverage") or 1))
            if leverage_val <= 0 or leverage_val > config.MAX_LEVERAGE:
                return {"error": f"开仓失败：杠杆 {leverage_val}x 超出允许范围 [1, {config.MAX_LEVERAGE}]"}
        except (TypeError, ValueError):
            return {"error": f"开仓失败：leverage 参数格式无效（{args.get('leverage')}）"}
        # 资金费率方向硬检查：禁止逆费率方向开仓
        _dir_check = None
        if isinstance(_funding_rates, list):
            for fr in _funding_rates:
                if fr.get("coin") == coin:
                    _dir_check = fr
                    break
        if _dir_check is None:
            _dir_check = hl_connector.get_funding_rate_for_coin(coin)
        if isinstance(_dir_check, dict):
            req_dir = _dir_check.get("direction")  # funding>0→"short", funding<0→"long"
            if req_dir and req_dir != side:
                return {"error": f"开仓失败：{coin} 资金费率 {_dir_check.get('funding_pct')}%，{req_dir == 'long' and '费率负(空付多)' or '费率正(多付空)'}，必须做{req_dir}，不可做{side}"}
        result = hl_connector.open_trade(coin, side, size_val, leverage_val, _balance=_balance, _positions=_positions)
        if isinstance(result, dict) and result.get("status") == "已开仓":
            _executed_ops.add(key)
            # 自动设置默认止盈（±3%），防 AI 忘记设 TP
            try:
                entry_px = result.get("trade", {}).get("entry_price")
                if not entry_px or not isinstance(entry_px, (int, float)) or entry_px <= 0:
                    mids = hl_connector._get_mids()
                    entry_px = float(mids.get(coin, 0)) if isinstance(mids, dict) else 0
                if entry_px and isinstance(entry_px, (int, float)) and entry_px > 0:
                    tp_target = entry_px * 0.97 if side == "short" else entry_px * 1.03
                    print(f"  [Agent] 自动止盈: {coin} {side} 入场${entry_px:.4f} → TP${tp_target:.4f}")
                    hl_connector.reset_user_state_cache()  # 刷新持仓缓存，否则 set_take_profit 读不到新仓位
                    tp_result = hl_connector.set_take_profit(coin, tp_target)
                    if isinstance(tp_result, dict) and tp_result.get("status") == "止盈已设置":
                        result["_auto_tp"] = f"自动止盈已设置于 ${tp_target:.4f}"
                        print(f"  [Agent] 自动止盈: [OK] {coin} TP${tp_target:.4f}")
                    else:
                        print(f"  [Agent] 自动止盈: [FAIL] {coin} → {str(tp_result)[:100]}")
                else:
                    print(f"  [Agent] 自动止盈: [SKIP] {coin} 无法获取入场价")
            except Exception as e:
                print(f"  [Agent] 自动止盈: [ERR] {coin} → {e}")
        return result
    elif name == "close_trade":
        coin = str(args.get("coin") or "").strip().upper()
        if not coin:
            return {"error": "平仓失败：缺少必填参数 coin"}
        key = f"close_trade:{coin}"
        if key in _executed_ops:
            return {"warning": f"已在当前周期对 {coin} 平过仓，操作已跳过"}
        close_size = args.get("size")
        if close_size is not None:
            try:
                close_size = abs(float(close_size))
                if close_size == 0:
                    close_size = None  # 0 等价于全平，统一用 None 处理
            except (TypeError, ValueError):
                return {"error": f"平仓失败：size 参数格式无效（{args.get('size')}）"}
        result = hl_connector.close_trade(coin, size=close_size, _positions=_positions)
        if isinstance(result, dict) and result.get("status") == "已平仓":
            if not (result.get("remaining_size") or 0):  # 仅全平时标记为已执行，部分平保留重试
                _executed_ops.add(key)
        return result
    elif name == "recall_lessons":
        try:
            top_k = int(float(args.get("top_k") or 3))
        except (TypeError, ValueError):
            return {"error": f"recall_lessons top_k 参数无效：'{args.get('top_k')}'"}
        return memory.recall(args.get("context") or "", top_k)
    elif name == "add_lesson":
        lesson = args.get("lesson")
        if not lesson or not isinstance(lesson, str):
            return {"error": "记录教训失败：缺少必填参数 lesson 或类型错误（需要字符串）"}
        tags = args.get("tags")
        if tags is not None and not isinstance(tags, (list, str)):
            return {"error": "记录教训失败：tags 参数类型错误（需要列表或逗号分隔字符串）"}
        # tags 交给 memory.add_lesson 统一规范化（str→list / 去空格 / 小写 / 去重）
        result = memory.add_lesson(lesson, tags, args.get("outcome") or "")
        if result is None:
            return {"warning": "教训未保存：内容过短（少于5字符）"}
        if isinstance(result, dict) and result.get("_duplicate"):
            lesson_clean = {k: v for k, v in result.items() if k not in ("embedding", "_duplicate")}
            return {"status": "已存在（重复）", "lesson": lesson_clean}
        # 去掉 embedding 向量（384 维 float 列表 ~2.3KB），避免浪费 AI 上下文 token
        lesson_clean = {k: v for k, v in result.items() if k != "embedding"} if isinstance(result, dict) else result
        return {"status": "已保存", "lesson": lesson_clean}
    elif name == "transfer_spot_to_perp":
        if hl_connector.get_wallet_mode() == "api":
            # 保存教训：API 模式无需转账，账户总资金（含 Spot）都可直接作为永续保证金使用
            memory.add_lesson(
                "API 钱包模式下转账功能不可用但无需处理，账户总资金（含 Spot）都可直接用作永续保证金，查 available 余额即可开仓",
                ["api", "perp"],
            )
            return {"error": "转账失败：API 钱包模式不支持转账，但无需转账，账户总资金（含 Spot）都可直接作为永续保证金"}
        if "transfer_spot_to_perp" in _executed_ops:
            return {"warning": "已在当前周期执行过转账，操作已跳过"}
        try:
            amount = float(args.get("amount") or 0)
        except (TypeError, ValueError):
            return {"error": f"转账失败：amount 参数格式无效（{args.get('amount')}）"}
        if amount <= 0:
            # amount <= 0 表示全额转入（与 main.py 手动模式保持一致）
            if isinstance(_balance, dict):
                amount = _balance.get("spot_value") or 0
            if amount <= 0:
                return {"error": "转账失败：Spot 账户无可用 USDC"}
        result = hl_connector.transfer_spot_to_perp(amount)
        if isinstance(result, dict) and result.get("status") == "ok":
            _executed_ops.add("transfer_spot_to_perp")
        return result
    elif name == "set_stop_loss":
        coin = str(args.get("coin") or "").strip().upper()
        if not coin:
            return {"error": "设置止损失败：缺少必填参数 coin"}
        stop_price = args.get("stop_price")
        if stop_price is None:
            return {"error": "设置止损失败：缺少必填参数 stop_price"}
        try:
            stop_price = float(stop_price)
        except (TypeError, ValueError):
            return {"error": f"设置止损失败：stop_price 参数无效（{stop_price}）"}
        if stop_price <= 0:
            return {"error": f"设置止损失败：stop_price 必须大于 0（{stop_price}）"}
        size = args.get("size")
        if size is not None:
            try:
                size = abs(float(size))
            except (TypeError, ValueError):
                return {"error": f"设置止损失败：size 参数无效（{size}）"}
        return hl_connector.set_stop_loss(coin, stop_price, size)
    elif name == "set_take_profit":
        coin = str(args.get("coin") or "").strip().upper()
        if not coin:
            return {"error": "设置止盈失败：缺少必填参数 coin"}
        target_price = args.get("target_price")
        if target_price is None:
            return {"error": "设置止盈失败：缺少必填参数 target_price"}
        try:
            target_price = float(target_price)
        except (TypeError, ValueError):
            return {"error": f"设置止盈失败：target_price 参数无效（{target_price}）"}
        if target_price <= 0:
            return {"error": f"设置止盈失败：target_price 必须大于 0（{target_price}）"}
        size = args.get("size")
        if size is not None:
            try:
                size = abs(float(size))
            except (TypeError, ValueError):
                return {"error": f"设置止盈失败：size 参数无效（{size}）"}
        return hl_connector.set_take_profit(coin, target_price, size)
    elif name == "get_open_orders":
        coin = str(args.get("coin") or "").strip().upper() or None
        return hl_connector.get_open_orders(coin)
    elif name == "cancel_order":
        coin = str(args.get("coin") or "").strip().upper()
        if not coin:
            return {"error": "撤单失败：缺少必填参数 coin"}
        oid = args.get("oid")
        if oid is None:
            return {"error": "撤单失败：缺少必填参数 oid"}
        try:
            oid = int(float(oid))
        except (TypeError, ValueError):
            return {"error": f"撤单失败：oid 参数无效（{oid}）"}
        return hl_connector.cancel_order(coin, oid)
    elif name == "get_funding_rates":
        try:
            top_n = int(float(args.get("top_n") or 20))
        except (TypeError, ValueError):
            return {"error": f"get_funding_rates top_n 参数无效：'{args.get('top_n')}'"}
        return hl_connector.get_funding_rates(top_n)
    elif name == "get_funding_history":
        coin = str(args.get("coin") or "").strip().upper()
        if not coin:
            return {"error": "获取资金费率历史失败：缺少必填参数 coin"}
        try:
            hours = int(float(args.get("hours") or 48))
        except (TypeError, ValueError):
            return {"error": f"get_funding_history hours 参数无效：'{args.get('hours')}'"}
        return hl_connector.get_funding_history(coin, hours)
    return {"error": f"未知工具: {name}"}


# ========== 状态文本格式化 ==========

def _format_state(wallet_mode, balance, positions, overview, extra_assets):
    return f"""钱包模式：{wallet_mode}
当前账户：{json.dumps(balance, ensure_ascii=False)}
当前持仓：{json.dumps(positions, ensure_ascii=False)}
市场概况：{json.dumps(overview[:20], ensure_ascii=False)}
非加密资产：{json.dumps(extra_assets, ensure_ascii=False) if extra_assets else "无"}"""


# ========== JSON 提取工具 ==========

def _extract_json(text: str) -> dict:
    """从 AI 回复中提取 JSON 对象，解析失败时返回 error dict"""
    if not text:
        return {"error": "AI 返回空响应", "_raw": ""}
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1]
        clean = clean.rsplit("```", 1)[0].strip()
    # 先尝试直接解析（能正确处理 } 出现在字符串值中的情况）
    if clean:
        try:
            result = json.loads(clean)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    # 降级：手动找 {} 包裹（兼容尾部有额外文本的情况）
    brace_start = clean.find("{")
    brace_end = clean.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        clean = clean[brace_start:brace_end + 1]
    try:
        result = json.loads(clean)
        if not isinstance(result, dict):
            return {"error": f"JSON 解析结果不是对象：{type(result).__name__}", "_raw": text[:200]}
        return result
    except json.JSONDecodeError as e:
        return {"error": f"JSON 解析失败: {e}", "_raw": text[:200]}


# ========== 单次交易决策循环（ReAct） ==========

def run_trade_cycle():
    """
    执行一次完整 ReAct 交易循环
    1. 收集当前状态 + 历史教训
    2. DeepSeek 思考 -> 调工具 -> 结果回送 -> 再思考 -> 最终决策
    3. 保存教训、返回结果
    """
    _reset_executed_ops()
    # ---- 检查 API Key（在 RPC 调用前快速失败，避免浪费）----
    if not config.DEEPSEEK_API_KEY:
        msg = "DEEPSEEK_API_KEY 未配置，跳过交易周期"
        print(f"  [Agent] {msg}")
        return {"error": msg, "reply": msg, "state": "", "tool_results": [], "thought": msg, "need_review": False, "lesson_to_save": ""}

    # ---- 第 1 步：收集数据 ----
    print(f"\n{'='*40}")
    print(f"交易循环开始：{datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*40}")

    hl_connector.reset_user_state_cache()
    hl_connector.reset_mids_cache()
    balance = hl_connector.get_balance()
    if isinstance(balance, str):
        print(f"  [Agent] 获取余额失败，跳过交易周期（{balance}）")
        return {"error": f"获取余额失败：{balance}", "reply": f"获取余额失败：{balance}", "state": "", "tool_results": [], "thought": "", "need_review": False, "lesson_to_save": ""}

    wallet_value = balance.get("wallet_value") or 0
    if wallet_value <= 0:
        msg = "账户余额为 0，跳过交易周期，等待充值 USDC"
        print(f"  [Agent] {msg}")
        return {"error": msg, "reply": msg, "state": json.dumps(balance, ensure_ascii=False),
                "tool_results": [], "thought": msg, "need_review": False, "lesson_to_save": ""}
    # 低于最小开仓所需金额时提前跳过，避免浪费 RPC 和 AI 调用
    min_value = config.MIN_ORDER_USD / (config.MAX_POSITION_SIZE_PCT / 100)
    if wallet_value < min_value:
        msg = f"余额 ${wallet_value:.2f} 低于开仓所需最低 ${min_value:.2f}（${config.MIN_ORDER_USD} / {config.MAX_POSITION_SIZE_PCT}%），跳过交易周期"
        print(f"  [Agent] {msg}")
        return {"error": msg, "reply": msg, "state": json.dumps(balance, ensure_ascii=False),
                "tool_results": [], "thought": msg, "need_review": False, "lesson_to_save": ""}

    # 每日开仓上限检查
    _daily_open = _get_daily_open_count()
    if _daily_open >= config.MAX_DAILY_OPEN_TRADES:
        msg = f"今日已开仓 {_daily_open} 次，达到上限 {config.MAX_DAILY_OPEN_TRADES} 次，跳过开仓（可继续管理现有持仓）"
        print(f"  [Agent] {msg}")
        # 注意：这不是错误，只是跳过开仓，仍然可以管理持仓/设置SL/TP
        # 这里继续执行，但用标记阻止 AI 开新仓

    positions = hl_connector.get_positions()
    if isinstance(positions, str):
        print(f"  [Agent] 获取持仓失败，跳过交易周期（{positions}）")
        return {"error": f"获取持仓失败：{positions}", "reply": f"获取持仓失败：{positions}", "state": "", "tool_results": [], "thought": "", "need_review": False, "lesson_to_save": ""}

    overview = hl_connector.get_market_overview()
    if isinstance(overview, str):
        print(f"  [Agent] 获取行情失败，跳过交易周期（{overview}）")
        return {"error": f"获取行情失败：{overview}", "reply": f"获取行情失败：{overview}", "state": "", "tool_results": [], "thought": "", "need_review": False, "lesson_to_save": ""}
    # 风控检查：DAILY_LOSS_LIMIT_PCT < 0 表示启用风控（-5 表示日亏 5% 停止）
    # 执行至此 wallet_value >= min_value（前面已对 <= 0 和 < min_value 提前返回）
    if config.DAILY_LOSS_LIMIT_PCT < 0:
        # 按自然日（凌晨到凌晨）查询成交记录，避免滚动 24h 窗口把昨晚的亏损算到今天
        _today_midnight = int(datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        recent_trades = hl_connector.get_trade_history(days=1, since_ms=_today_midnight)
        if not isinstance(recent_trades, list):
            print(f"  [风控] 获取今日成交记录失败（{str(recent_trades)[:80]}），跳过风控检查")
        elif recent_trades:
            today_pnl = sum(t["pnl"] for t in recent_trades if t.get("pnl") is not None)
            # 用今日盈亏前的余额（当前余额 - 已实现盈亏）作为止损基准，
            # 避免亏损缩小分母后止损线同步降低，导致止损比预期更激进
            pre_trade_base = wallet_value - today_pnl  # today_pnl 为负时 pre_trade_base > wallet_value
            loss_limit = max(wallet_value, pre_trade_base) * abs(config.DAILY_LOSS_LIMIT_PCT) / 100
            if today_pnl < -loss_limit:
                msg = f"今日亏损 ${today_pnl:.2f} 超过限制 ${-loss_limit:.2f}（{config.DAILY_LOSS_LIMIT_PCT}%），停止交易"
                print(f"  [风控] {msg}")
                return {
                    "error": msg,
                    "thought": msg,
                    "reply": msg,
                    "need_review": False,
                    "tool_results": [],
                    "state": f"余额：{json.dumps(balance, ensure_ascii=False)}\n今日盈亏：{today_pnl:.2f}",
                    "lesson_to_save": "",
                }

    wallet_mode = hl_connector.get_wallet_mode() or "未配置"
    _extra_assets = {}
    if isinstance(overview, list):
        for _c in overview:
            coin_name = _c.get("coin")
            if coin_name in ("SPX", "PAXG"):
                _extra_assets[coin_name] = _c
                if len(_extra_assets) == 2:
                    break
        # 后续只用 top 30，裁剪 overview 减少 JSON 序列化数据量
        overview = overview[:30]

    # 资金费率数据（用于套利分析 + 开仓方向硬检查）
    _funding_rates = hl_connector.get_funding_rates(50)
    funding_text = json.dumps(_funding_rates, ensure_ascii=False) if isinstance(_funding_rates, list) else "无"

    state_text = _format_state(wallet_mode, balance, positions, overview, _extra_assets)
    state_text += f"\n资金费率排行(套利机会)：{funding_text}"
    state_text += f"\n今日已开仓 {_daily_open}/{config.MAX_DAILY_OPEN_TRADES} 次"

    # ---- 构造自然语言召回上下文（向量搜索用，不用 JSON）----
    recall_context_parts = []
    if isinstance(balance, dict):
        recall_context_parts.append(f"余额 ${balance.get('wallet_value') or 0}，可用 ${balance.get('available') or 0}，模式 {wallet_mode}")
    if isinstance(positions, list) and positions:
        pos_desc = "; ".join(
            f"{p.get('coin') or '?'} {p.get('direction') or '?'} {abs(p.get('size') or 0)}张 盈亏${p.get('pnl') or 0}"
            for p in positions
        )
        recall_context_parts.append(f"持仓: {pos_desc}")
    if isinstance(overview, list):
        top_coins = overview[:5]
        mkt_desc = "; ".join(f"{c.get('coin') or '?'} ${c.get('mid') or 0}" for c in top_coins)
        recall_context_parts.append(f"行情: {mkt_desc}")
        if _extra_assets:
            _ea_desc = "; ".join(f"{c} ${d.get('mid') or 0}" for c, d in _extra_assets.items())
            recall_context_parts.append(f"非加密: {_ea_desc}")
    if isinstance(_funding_rates, list) and _funding_rates:
        _fr_desc = "; ".join(f"{r['coin']} {r['funding_pct']}%/APY{r['funding_apy']}%→{r['direction']}" for r in _funding_rates[:5])
        recall_context_parts.append(f"资金费率: {_fr_desc}")
    recall_context = "，".join(recall_context_parts) or state_text

    # ---- 第 2 步：回想教训 ----
    relevant_lessons = memory.recall(recall_context)
    lessons_text = json.dumps(relevant_lessons, ensure_ascii=False, indent=2) if relevant_lessons else "暂无相关教训"

    # ---- 第 3 步：ReAct 多轮循环 ----
    print("  [Agent] 思考中...")

    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": f"## 当前状态\n{state_text}\n## 相关历史教训\n{lessons_text}\n## 分析市场并决定下一步操作"},
    ]

    max_rounds = 3
    all_results = []
    final_thought = ""
    final_reply = ""
    need_review = False
    lesson_to_save = ""

    for _round in range(max_rounds):
        try:
            resp = _get_client().chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            if not resp.choices:
                raise ValueError("AI 返回空响应（choices 为空）")
            decision = _extract_json(resp.choices[0].message.content)
            if "error" in decision:
                raise ValueError(decision["error"])
        except AuthenticationError as e:
            # 认证错误重试也无效，直接返回
            return {
                "error": f"API 认证失败: {e}",
                "reply": f"API 认证失败: {e}",
                "state": state_text,
                "tool_results": all_results,
                "thought": "",
                "need_review": False,
                "lesson_to_save": "",
            }
        except Exception as e:
            if _round >= max_rounds - 1:
                return {
                    "error": f"AI 决策失败: {e}",
                    "reply": f"AI 决策失败: {e}",  # 确保 reply 字段存在，避免调用方 KeyError
                    "state": state_text,
                    "tool_results": all_results,
                    "thought": "",
                    "need_review": False,
                    "lesson_to_save": "",
                }
            messages.append({"role": "user", "content": f"调用出错: {e}，请重新尝试"})
            continue

        final_thought = decision.get("thought") or ""
        final_reply = decision.get("reply") or ""
        # 重置每轮变量，避免前一轮的教训/复盘标记在后续轮次被误用
        need_review = bool(decision.get("need_review") or False)
        lesson_to_save = decision.get("lesson_to_save") or ""

        tool_calls = decision.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            tool_calls = []
        if not tool_calls:
            break  # AI 做出最终决策

        # 执行工具（每轮独立执行，结果不回累积）
        had_errors = []
        had_warnings = []
        results_before = len(all_results)
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name") or ""
            args = tc.get("args") or {}
            print(f"  [Agent] 调工具: {name}({args})")
            try:
                result = _run_tool(name, args, _balance=balance, _positions=positions, _funding_rates=_funding_rates)
            except Exception as e:
                result = {"error": f"工具 {name} 执行异常: {e}"}
            # 交易或转账成功后刷新 balance/positions，避免后续仓位检查用旧数据
            need_refresh = False
            if isinstance(result, dict):
                if result.get("status") in ("已开仓", "已平仓", "部分平仓"):
                    need_refresh = True
                    if name == "open_trade" and result.get("status") == "已开仓":
                        _increment_daily_open_count()
                elif name == "transfer_spot_to_perp" and result.get("status") == "ok":
                    need_refresh = True
            elif name == "open_trade" and isinstance(result, str):
                # open_trade 失败（如部分成交拒绝）也可能在交易所修改了仓位，刷新缓存保一致
                need_refresh = True
            if need_refresh:
                hl_connector.reset_user_state_cache()
                new_balance = hl_connector.get_balance()
                new_positions = hl_connector.get_positions()
                if isinstance(new_balance, dict):
                    balance = new_balance
                if isinstance(new_positions, list):
                    positions = new_positions
                # 同步更新 state_text，确保返回的 state 字段反映最新余额/持仓
                state_text = _format_state(wallet_mode, balance, positions, overview, _extra_assets)
            print(f"  [Agent] 结果: {str(result)[:80]}...")
            all_results.append({"tool": name, "result": result})
            # 分离错误与警告，避免去重提示干扰 AI 对真正失败的判断
            if isinstance(result, str):
                had_errors.append(f"{name}: {result}")
            elif isinstance(result, dict):
                if result.get("error"):
                    had_errors.append(f"{name}: {result['error']}")
                elif result.get("warning"):
                    had_warnings.append(f"{name}: {result['warning']}")

        # 把所有工具结果送回 AI 继续分析
        messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})
        # 只传本轮工具结果（非累积），避免多轮累积导致 8000 字截断切掉新结果
        round_results = all_results[results_before:]
        all_results_str = _truncate_results(round_results)
        feedback = f"## 工具返回结果\n{all_results_str}\n"
        if had_errors:
            feedback += "\n⚠️ 以下工具调用失败，请仔细检查，不要假设操作已成功：\n" + "\n".join(had_errors) + "\n"
        if had_warnings:
            feedback += "\nℹ️ 以下操作已执行（被去重跳过），无需重复操作：\n" + "\n".join(had_warnings) + "\n"
        feedback += "\n## 根据以上数据做最终决策。开仓用 open_trade，平仓用 close_trade，不要忘记设置止损（set_stop_loss）保护仓位。注意资金费率套利机会，可平旧仓换新仓。不交易则 tool_calls 留空。"
        messages.append({"role": "user", "content": feedback})

    # ---- 第 4 步：保存教训 ----
    if lesson_to_save:
        if isinstance(lesson_to_save, dict):
            text = (lesson_to_save.get("text") or lesson_to_save.get("lesson") or "").strip()
            if text:
                memory.add_lesson(text, lesson_to_save.get("tags"), lesson_to_save.get("outcome") or "")
            else:
                print(f"  [Agent] 教训保存跳过：AI 返回了 lesson_to_save 但 text 为空，raw={str(lesson_to_save)[:100]}")
        elif isinstance(lesson_to_save, str):
            if not memory.add_lesson(lesson_to_save):
                print(f"  [Agent] 教训保存跳过：内容过短（少于5字符）")
        else:
            print(f"  [Agent] 教训保存跳过：lesson_to_save 类型异常（{type(lesson_to_save).__name__}），raw={str(lesson_to_save)[:100]}")

    return {
        "thought": final_thought,
        "reply": final_reply,
        "need_review": need_review,
        "tool_results": all_results,
        "state": state_text,
        "lesson_to_save": lesson_to_save,
        "balance": balance,
        "positions": positions,
    }


# ========== 复盘循环 ==========

def run_review():
    """
    复盘：分析近期交易，提取教训，更新策略
    """
    print(f"\n{'='*40}")
    print("开始复盘")
    print(f"{'='*40}")

    trades = hl_connector.get_trade_history(days=7)
    if isinstance(trades, str):
        print(f"  获取交易记录失败: {trades}")
        return trades
    if not isinstance(trades, list) or not trades:
        print("  没有近期交易")
        return

    # 过滤开仓成交（closedPnl=None），只保留已平仓成交用于计算胜率
    closed = [t for t in trades if t.get("pnl") is not None]
    total_pnl = sum(t["pnl"] for t in closed)
    win_trades = [t for t in closed if t["pnl"] > 0]
    loss_trades = [t for t in closed if t["pnl"] < 0]

    review_data = {
        "total_trades": len(closed),
        "total_pnl": total_pnl,
        "win_count": len(win_trades),
        "loss_count": len(loss_trades),
        "win_rate": round(len(win_trades) / max(len(closed), 1) * 100, 1),
    }

    print(f"  交易次数: {review_data['total_trades']}")
    print(f"  总盈亏: {review_data['total_pnl']:.2f} USD")
    print(f"  胜率: {review_data['win_rate']}%")

    try:
        response = _get_client().chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是交易复盘分析师，分析以下近期交易数据，找出问题并提出改进建议。"},
                {"role": "user", "content": json.dumps(trades, ensure_ascii=False, indent=2)},
            ],
            temperature=0.7,
        )
        if not response.choices:
            print("  复盘 AI 分析：返回空响应，跳过")
            return review_data
        analysis = response.choices[0].message.content
        if not analysis:
            print("  复盘 AI 分析：返回内容为空，跳过")
            return review_data
        print(f"\n  AI 分析：\n  {analysis}")

        lesson_response = _get_client().chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "从以下交易复盘分析中提取 1-2 条最重要的教训，以 JSON 格式返回：{\"lessons\": [{\"text\": \"教训内容\", \"tags\": [\"标签1\", \"标签2\"], \"outcome\": \"亏损/盈利金额\"}]}"},
                {"role": "user", "content": analysis},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        if not lesson_response.choices:
            print("  复盘教训提取：AI 返回空响应，跳过")
            return review_data
        lesson_data = _extract_json(lesson_response.choices[0].message.content)
        if "error" in lesson_data:
            print(f"  复盘教训提取 JSON 解析失败: {lesson_data['error']}")
            return review_data
        lessons_list = lesson_data.get("lessons") or []
        if isinstance(lessons_list, list):
            for l in lessons_list:
                text = (l.get("text") or "")
                if text:
                    memory.add_lesson(text, l.get("tags"), l.get("outcome") or "")

    except Exception as e:
        print(f"  复盘 AI 分析失败: {e}")

    return review_data
