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
import datetime
from functools import lru_cache
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
            raise RuntimeError("DEEPSEEK_API_KEY 未设置，请通过环境变量配置")
        _client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    return _client


# ========== 工具函数描述 ==========

_TOOL_DESCRIPTIONS = {
    "get_market_overview": "查看所有币种的最新行情（价格、成交量）",
    "get_orderbook": "查看某个币的买卖盘深度，参数：coin（币种，如 BTC），depth（深度档位，默认5）",
    "get_candles": "查看某个币的K线趋势，参数：coin（币种），interval（间隔：1m/5m/15m/1h/4h/1d，默认1h），hours（往回看几小时，默认24）",
    "get_positions": "查看当前持仓及盈亏",
    "get_balance": "查看账户总资产和可用余额",
    "open_trade": "开仓交易，参数：coin（币种），side（long/short），size（USD金额≥10），leverage（杠杆倍数，默认1）",
    "close_trade": "平仓，参数：coin（币种），size（平仓数量，可选，不传则全平）",
    "recall_lessons": "回想过去类似情况学到的教训，参数：context（当前市场情况描述），top_k（返回几条，默认3）",
    "add_lesson": "记录一条新教训，参数：lesson（教训内容），tags（标签列表），outcome（结果）",
    "transfer_spot_to_perp": "把 USDC 从 Spot 转到永续合约账户（仅直连私钥模式有效），参数：amount（金额，0=全部转入）",
}


# ========== 构建系统提示词 ==========

@lru_cache(maxsize=1)
def _build_system_prompt():
    return f"""你是 Hyperliquid 永续合约交易员，核心目标是积极发现机会并盈利。

## 交易规则
1. 所有持仓累计名义价值（持仓数量 x 价格，杠杆不影响名义价值）不超过总资金的 {config.MAX_POSITION_SIZE_PCT}%
2. 最高杠杆 {config.MAX_LEVERAGE} 倍
3. 每日亏损超过 {abs(config.DAILY_LOSS_LIMIT_PCT)}% 时停止交易
4. 每次开仓前先回想相关教训
5. 平仓后复盘并记录教训
6. 开仓前计算可开金额：wallet_value * {config.MAX_POSITION_SIZE_PCT}% - 已有持仓名义价值（持仓数量 x 价格），选择在此范围内的金额。注意杠杆不影响名义价值，只影响保证金占用
7. **核心目标：盈利。积极寻找交易机会，小仓位试单、发现趋势加仓**
8. 小账户可用高杠杆（3x）提高资金利用率，每笔赚 2-5% 就很好
9. 不要只会等待，没有明确风险信号就该尝试入场

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
1. 第一轮：分析当前状态，如果数据不够就调工具获取更多信息
2. 工具结果会返回给你，根据新数据做最终决策
3. tool_calls 为空数组 = 最终决策完成（不交易或已执行）
4. 如果发现交易机会，果断用 open_trade 开仓；如果持仓方向错了，close_trade 平仓"""


# ========== 防重复执行 ==========
# 记录当前周期已执行过的操作，防止 AI 同一轮重复开仓
_executed_ops: set = set()

# 工具结果排序优先级：交易操作优先，避免截断丢失关键信息
_KEY_OPS = {"open_trade", "close_trade", "transfer_spot_to_perp", "add_lesson"}


def _reset_executed_ops():
    """每轮交易循环开始时调用"""
    global _executed_ops
    _executed_ops = set()


def _truncate_results(items: list, max_len: int = 8000) -> str:
    """在完整 item 边界截断 JSON 列表，避免切在字符串中间；单条超限时至少保留第一条"""
    ordered = sorted(items, key=lambda r: (0 if r["tool"] in _KEY_OPS else 1, r["tool"]))
    payload = json.dumps(ordered, ensure_ascii=False)
    if len(payload) <= max_len:
        return payload
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

def _run_tool(name: str, args: dict, _balance=None, _positions=None, _wallet_mode=None):
    """执行一个工具调用并返回原始 Python 对象（非 JSON 字符串）
    _balance/_positions/_wallet_mode: 调用方已查询的数据，用于避免重复 RPC"""

    if not isinstance(args, dict):
        return {"error": f"工具 {name} 参数格式错误：需要 JSON 对象，收到 {type(args).__name__}"}

    if name == "get_market_overview":
        return hl_connector.get_market_overview()
    elif name == "get_orderbook":
        return hl_connector.get_orderbook(str(args.get("coin") or "BTC").strip().upper(), int(args.get("depth") or 5))
    elif name == "get_candles":
        return hl_connector.get_candles(str(args.get("coin") or "BTC").strip().upper(), str(args.get("interval") or "1h").strip().lower(), int(args.get("hours") or 24))
    elif name == "get_positions":
        return hl_connector.get_positions()
    elif name == "get_balance":
        return hl_connector.get_balance()
    elif name == "open_trade":
        # 检查必填参数（coin 统一转大写，避免大小写不一致导致去重/白名单/API 调用出问题）
        coin = str(args.get("coin") or "").strip().upper()
        side = str(args.get("side") or "").strip().lower()
        size = args.get("size")
        if not coin or not side or not size:
            return {"error": "开仓失败：缺少必填参数 coin/side/size"}
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
        except (TypeError, ValueError):
            return {"error": f"开仓失败：size 参数格式无效（{size}）"}
        try:
            leverage_val = int(args.get("leverage") or 1)
        except (TypeError, ValueError):
            return {"error": f"开仓失败：leverage 参数格式无效（{args.get('leverage')}）"}
        result = hl_connector.open_trade(coin, side, size_val, leverage_val, _balance=_balance, _positions=_positions)
        if isinstance(result, dict) and result.get("status") == "已开仓":
            _executed_ops.add(key)
        return result
    elif name == "close_trade":
        coin = str(args.get("coin") or "").strip().upper()
        if not coin:
            return {"error": "平仓失败：缺少必填参数 coin"}
        key = f"close_trade:{coin}"
        if key in _executed_ops:
            return {"warning": f"已在当前周期对 {coin} 平过仓，操作已跳过"}
        size_arg = args.get("size")
        if size_arg is not None:
            try:
                size_val = abs(float(size_arg))
                if size_val <= 0:
                    return {"error": "平仓失败：size 必须为正数"}
                size_arg = size_val
            except (TypeError, ValueError):
                return {"error": f"平仓失败：size 参数格式无效（{size_arg}）"}
        result = hl_connector.close_trade(coin, size=size_arg, _positions=_positions)
        if isinstance(result, dict) and result.get("status") == "已平仓":
            if not (result.get("remaining_size") or 0):  # 仅全平时标记为已执行，部分平保留重试
                _executed_ops.add(key)
        return result
    elif name == "recall_lessons":
        return memory.recall(args.get("context") or "", int(args.get("top_k") or 3))
    elif name == "add_lesson":
        lesson = args.get("lesson")
        if not lesson or not isinstance(lesson, str):
            return {"error": "记录教训失败：缺少必填参数 lesson 或类型错误（需要字符串）"}
        # tags 交给 memory.add_lesson 统一规范化（str→list / 去空格 / 小写 / 去重）
        result = memory.add_lesson(lesson, args.get("tags"), args.get("outcome") or "")
        if result is None:
            return {"warning": "教训未保存：内容过短（少于5字符）"}
        # 去掉 embedding 向量（384 维 float 列表 ~2.3KB），避免浪费 AI 上下文 token
        lesson_clean = {k: v for k, v in result.items() if k != "embedding"} if isinstance(result, dict) else result
        return {"status": "已保存", "lesson": lesson_clean}
    elif name == "transfer_spot_to_perp":
        if (_wallet_mode or hl_connector.get_wallet_mode()) == "api":
            return {"error": "转账失败：API 钱包模式不支持转账到永续合约，如需此功能请使用直连私钥模式"}
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
        return hl_connector.transfer_spot_to_perp(amount)
    return {"error": f"未知工具: {name}"}


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
    # ---- 第 1 步：收集数据 ----
    print(f"\n{'='*40}")
    print(f"交易循环开始：{datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*40}")

    hl_connector.reset_user_state_cache()
    balance = hl_connector.get_balance()
    if isinstance(balance, str):
        print(f"  [Agent] 获取余额失败，跳过交易周期（{balance}）")
        return {"error": f"获取余额失败：{balance}", "reply": f"获取余额失败：{balance}", "state": "", "tool_results": [], "thought": "", "need_review": False, "lesson_to_save": ""}

    positions = hl_connector.get_positions()
    if isinstance(positions, str):
        print(f"  [Agent] 获取持仓失败，跳过交易周期（{positions}）")
        return {"error": f"获取持仓失败：{positions}", "reply": f"获取持仓失败：{positions}", "state": "", "tool_results": [], "thought": "", "need_review": False, "lesson_to_save": ""}

    overview = hl_connector.get_market_overview()
    if isinstance(overview, str):
        print(f"  [Agent] 获取行情失败，跳过交易周期（{overview}）")
        return {"error": f"获取行情失败：{overview}", "reply": f"获取行情失败：{overview}", "state": "", "tool_results": [], "thought": "", "need_review": False, "lesson_to_save": ""}
    # 风控检查：DAILY_LOSS_LIMIT_PCT < 0 表示启用风控（-5 表示日亏 5% 停止）
    if config.DAILY_LOSS_LIMIT_PCT < 0:
        wallet_value = balance.get("wallet_value") or 0
        if wallet_value > 0:
            recent_trades = hl_connector.get_trade_history(days=1)
            if not isinstance(recent_trades, list):
                print(f"  [风控] 获取今日成交记录失败（{str(recent_trades)[:80]}），跳过风控检查")
            elif recent_trades:
                today_pnl = sum(t.get("pnl") or 0 for t in recent_trades)
                loss_limit = wallet_value * abs(config.DAILY_LOSS_LIMIT_PCT) / 100
                if today_pnl < -loss_limit:
                    msg = f"今日亏损 ${today_pnl:.2f} 超过限制 ${-loss_limit:.2f}（{config.DAILY_LOSS_LIMIT_PCT}%），停止交易"
                    print(f"  [风控] {msg}")
                    return {
                        "error": msg,
                        "thought": msg,
                        "reply": msg,
                        "need_review": False,  # 避免 auto 模式每周期重复调 DeepSeek
                        "tool_results": [],
                        "state": f"余额：{json.dumps(balance, ensure_ascii=False)}\n今日盈亏：{today_pnl:.2f}",
                        "lesson_to_save": "",
                    }

    wallet_mode = hl_connector.get_wallet_mode() or "未配置"
    state_text = f"""
钱包模式：{wallet_mode}
当前账户：{json.dumps(balance, ensure_ascii=False)}
当前持仓：{json.dumps(positions, ensure_ascii=False)}
市场概况：{json.dumps(overview[:20], ensure_ascii=False)}
"""

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
        if "need_review" in decision:
            need_review = decision["need_review"] or False
        if "lesson_to_save" in decision:
            lesson_to_save = decision["lesson_to_save"] or ""

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
                result = _run_tool(name, args, _balance=balance, _positions=positions, _wallet_mode=wallet_mode)
            except Exception as e:
                result = {"error": f"工具 {name} 执行异常: {e}"}
            # 交易或转账成功后刷新 balance/positions，避免后续仓位检查用旧数据
            need_refresh = False
            if isinstance(result, dict):
                if result.get("status") in ("已开仓", "已平仓"):
                    need_refresh = True
                elif name == "transfer_spot_to_perp" and result.get("status") == "ok":
                    need_refresh = True
            if need_refresh:
                hl_connector.reset_user_state_cache()
                new_balance = hl_connector.get_balance()
                new_positions = hl_connector.get_positions()
                if isinstance(new_balance, dict):
                    balance = new_balance
                if isinstance(new_positions, list):
                    positions = new_positions
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
        feedback += "\n## 根据以上数据做最终决策。开仓用 open_trade，平仓用 close_trade，不交易则 tool_calls 留空。"
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
            memory.add_lesson(lesson_to_save)

    return {
        "thought": final_thought,
        "reply": final_reply,
        "need_review": need_review,
        "tool_results": all_results,
        "state": state_text,
        "lesson_to_save": lesson_to_save,
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

    # 过滤开仓成交（closedPnl=0），只保留已平仓成交用于计算胜率
    closed = [t for t in trades if t.get("pnl") is not None and t.get("pnl") != 0]
    total_pnl = sum(t.get("pnl") or 0 for t in closed)
    win_trades = [t for t in closed if (t.get("pnl") or 0) > 0]
    loss_trades = [t for t in closed if (t.get("pnl") or 0) < 0]

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
        print(f"\n  AI 分析：\n  {analysis}")

        lesson_response = _get_client().chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "从以下交易复盘分析中提取 1-2 条最重要的教训，以 JSON 格式返回：{\"lessons\": [{\"text\": \"教训内容\", \"tags\": [\"标签1\", \"标签2\"]}]}"},
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
