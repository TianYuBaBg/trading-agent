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
from openai import OpenAI
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
    "get_orderbook": "查看某个币的买卖盘深度，参数：coin（币种，如 BTC）",
    "get_candles": "查看某个币的K线趋势，参数：coin（币种），interval（间隔：1m/5m/15m/1h/4h/1d，默认1h），hours（往回看几小时，默认24）",
    "get_positions": "查看当前持仓及盈亏",
    "get_balance": "查看账户总资产和可用余额",
    "open_trade": "开仓交易，参数：coin（币种），side（long/short），size（USD金额≥10），leverage（杠杆倍数，默认1）",
    "close_trade": "平仓，参数：coin（币种）",
    "recall_lessons": "回想过去类似情况学到的教训，参数：context（当前市场情况描述）",
    "add_lesson": "记录一条新教训，参数：lesson（教训内容），tags（标签列表），outcome（结果）",
}


# ========== 构建系统提示词 ==========

def _build_system_prompt():
    return f"""你是 Hyperliquid 永续合约交易员，核心目标是积极发现机会并盈利。

## 交易规则
1. 所有持仓累计名义价值不超过总资金的 {config.MAX_POSITION_SIZE_PCT}%
2. 最高杠杆 {config.MAX_LEVERAGE} 倍
3. 每日亏损超过 {config.DAILY_LOSS_LIMIT_PCT}% 时停止交易
4. 每次开仓前先回想相关教训
5. 平仓后复盘并记录教训
6. 开仓前计算可开金额：wallet_value * {config.MAX_POSITION_SIZE_PCT}% - 已有持仓名义价值，选择在此范围内的金额
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
    "lesson_to_save": ""
}}

流程说明：
1. 第一轮：分析当前状态，如果数据不够就调工具获取更多信息
2. 工具结果会返回给你，根据新数据做最终决策
3. tool_calls 为空数组 = 最终决策完成（不交易或已执行）
4. 如果发现交易机会，果断用 open_trade 开仓；如果持仓方向错了，close_trade 平仓"""


# ========== 防重复执行 ==========
# 记录当前周期已执行过的操作，防止 AI 同一轮重复开仓
_executed_ops: set = set()


def _reset_executed_ops():
    """每轮交易循环开始时调用"""
    global _executed_ops
    _executed_ops = set()


# ========== 执行工具调用 ==========

def _run_tool(name: str, args: dict):
    """执行一个工具调用并返回原始 Python 对象（非 JSON 字符串）"""

    if name == "get_market_overview":
        return hl_connector.get_market_overview()
    elif name == "get_orderbook":
        return hl_connector.get_orderbook(args.get("coin", "BTC"))
    elif name == "get_candles":
        return hl_connector.get_candles(args.get("coin", "BTC"), (args.get("interval") or "1h").lower(), int(args.get("hours") or 24))
    elif name == "get_positions":
        return hl_connector.get_positions()
    elif name == "get_balance":
        return hl_connector.get_balance()
    elif name == "open_trade":
        # 检查必填参数
        coin = args.get("coin")
        side = (args.get("side") or "").lower()
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
        result = hl_connector.open_trade(coin, side, float(size), int(args.get("leverage") or 1))
        if isinstance(result, dict) and result.get("status") == "已开仓":
            _executed_ops.add(key)
        return result
    elif name == "close_trade":
        coin = args.get("coin")
        if not coin:
            return {"error": "平仓失败：缺少必填参数 coin"}
        key = f"close_trade:{coin}"
        if key in _executed_ops:
            return {"warning": f"已在当前周期对 {coin} 平过仓，操作已跳过"}
        result = hl_connector.close_trade(coin)
        if isinstance(result, dict) and result.get("status") == "已平仓":
            _executed_ops.add(key)
        return result
    elif name == "recall_lessons":
        return memory.recall(args.get("context", ""))
    elif name == "add_lesson":
        lesson = args.get("lesson")
        if not lesson:
            return {"error": "记录教训失败：缺少必填参数 lesson"}
        tags = args.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        return memory.add_lesson(lesson, tags, args.get("outcome", ""))
    elif name == "transfer_spot_to_perp":
        return hl_connector.transfer_spot_to_perp(float(args.get("amount") or 0))
    return {"error": f"未知工具: {name}"}


# ========== JSON 提取工具 ==========

def _extract_json(text: str) -> dict:
    """从 AI 回复中提取 JSON 对象"""
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1]
        clean = clean.rsplit("```", 1)[0].strip()
    brace_start = clean.find("{")
    brace_end = clean.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        clean = clean[brace_start:brace_end + 1]
    return json.loads(clean)


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

    balance = hl_connector.get_balance()
    positions = hl_connector.get_positions()
    overview = hl_connector.get_market_overview()

    # ---- 风控：每日亏损限制 ----
    if config.DAILY_LOSS_LIMIT_PCT < 0 and isinstance(balance, dict):
        wallet_value = balance.get("wallet_value", 0)
        if wallet_value > 0:
            recent_trades = hl_connector.get_trade_history(days=1)
            if isinstance(recent_trades, list) and recent_trades:
                today_pnl = sum(t.get("pnl", 0) for t in recent_trades if t.get("pnl"))
                loss_limit = wallet_value * abs(config.DAILY_LOSS_LIMIT_PCT) / 100
                if today_pnl < -loss_limit:
                    msg = f"今日亏损 ${today_pnl:.2f} 超过限制 ${-loss_limit:.2f}（{config.DAILY_LOSS_LIMIT_PCT}%），停止交易"
                    print(f"  [风控] {msg}")
                    return {
                        "thought": msg,
                        "reply": msg,
                        "need_review": True,
                        "tool_results": [],
                        "state": f"余额：{json.dumps(balance, ensure_ascii=False)}\n今日盈亏：{today_pnl:.2f}",
                    }

    state_text = f"""
当前账户：{json.dumps(balance, ensure_ascii=False)}
当前持仓：{json.dumps(positions, ensure_ascii=False)}
市场概况：{json.dumps(overview[:20] if isinstance(overview, list) else overview, ensure_ascii=False)}
"""

    # ---- 第 2 步：回想教训 ----
    relevant_lessons = memory.recall(state_text)
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
            decision = _extract_json(resp.choices[0].message.content)
        except Exception as e:
            if _round >= max_rounds - 1:
                return {"error": f"AI 决策失败: {e}", "state": state_text}
            messages.append({"role": "user", "content": f"JSON 解析错误: {e}，请重新输出有效的 JSON 对象"})
            continue

        final_thought = decision.get("thought", "")
        final_reply = decision.get("reply", "")
        need_review = decision.get("need_review", False)
        lesson_to_save = decision.get("lesson_to_save", "")

        tool_calls = decision.get("tool_calls", [])
        if not tool_calls:
            break  # AI 做出最终决策

        # 执行工具
        had_errors = []
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            print(f"  [Agent] 调工具: {name}({args})")
            try:
                result = _run_tool(name, args)
            except Exception as e:
                result = {"error": f"工具 {name} 执行异常: {e}"}
            print(f"  [Agent] 结果: {str(result)[:80]}...")
            all_results.append({"tool": name, "result": result})
            # 检测工具是否返回错误
            if isinstance(result, str) and ("失败" in result or "错误" in result):
                had_errors.append(f"{name}: {result[:100]}")
            elif isinstance(result, dict) and result.get("error"):
                had_errors.append(f"{name}: {result['error'][:100]}")

        # 把所有工具结果送回 AI 继续分析
        messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})
        # 关键操作结果放前面，避免截断切掉
        key_ops = {"open_trade", "close_trade", "transfer_spot_to_perp", "add_lesson"}
        ordered = sorted(all_results, key=lambda r: (r["tool"] not in key_ops, 0))
        all_results_str = json.dumps(ordered, ensure_ascii=False)[:8000]
        feedback = f"## 工具返回结果\n{all_results_str}\n"
        if had_errors:
            feedback += "\n⚠️ 注意：以上工具有失败的情况，请仔细检查失败原因，不要假设操作已成功执行。\n"
        feedback += "\n## 根据以上数据做最终决策。开仓用 open_trade，平仓用 close_trade，不交易则 tool_calls 留空。"
        messages.append({"role": "user", "content": feedback})

    # ---- 第 4 步：保存教训 ----
    if lesson_to_save:
        memory.add_lesson(lesson_to_save)

    return {
        "thought": final_thought,
        "reply": final_reply,
        "need_review": need_review,
        "tool_results": all_results,
        "state": state_text,
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
    if not isinstance(trades, list) or not trades:
        print("  没有近期交易")
        return

    total_pnl = sum(t.get("pnl", 0) for t in trades if t.get("pnl"))
    win_trades = [t for t in trades if t.get("pnl", 0) > 0]
    loss_trades = [t for t in trades if t.get("pnl", 0) < 0]

    review_data = {
        "total_trades": len(trades),
        "total_pnl": total_pnl,
        "win_count": len(win_trades),
        "loss_count": len(loss_trades),
        "win_rate": round(len(win_trades) / max(len(trades), 1) * 100, 1),
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
        analysis = response.choices[0].message.content
        print(f"\n  AI 分析：\n  {analysis}")

        lesson_response = _client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "从以下交易复盘分析中提取 1-2 条最重要的教训，以 JSON 格式返回：{\"lessons\": [{\"text\": \"教训内容\", \"tags\": [\"标签1\", \"标签2\"]}]}"},
                {"role": "user", "content": analysis},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        lesson_data = json.loads(lesson_response.choices[0].message.content)
        for l in lesson_data.get("lessons", []):
            text = l.get("text", "")
            if text:
                tags = l.get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                memory.add_lesson(text, tags, l.get("outcome", ""))

    except Exception as e:
        print(f"  复盘 AI 分析失败: {e}")

    return review_data
