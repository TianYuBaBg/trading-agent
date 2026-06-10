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


# ========== DeepSeek 客户端 ==========

_client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


# ========== 工具函数描述 ==========

_TOOL_DESCRIPTIONS = {
    "get_market_overview": "查看所有币种的最新行情（价格、成交量）",
    "get_orderbook": "查看某个币的买卖盘深度，参数：coin（币种，如 BTC）",
    "get_positions": "查看当前持仓及盈亏",
    "get_balance": "查看账户总资产和可用余额",
    "open_trade": "开仓交易，参数：coin（币种），side（long/short），size（USD金额），leverage（杠杆倍数，默认1）",
    "close_trade": "平仓，参数：coin（币种）",
    "recall_lessons": "回想过去类似情况学到的教训，参数：context（当前市场情况描述）",
    "add_lesson": "记录一条新教训，参数：lesson（教训内容），tags（标签列表），outcome（结果）",
}


# ========== 构建系统提示词 ==========

def _build_system_prompt():
    return f"""你是 Hyperliquid 永续合约交易员，目标是通过分析和交易盈利。

## 交易规则
1. 单笔最大仓位不超过总资金的 {config.MAX_POSITION_SIZE_PCT}%
2. 最高杠杆 {config.MAX_LEVERAGE} 倍
3. 每日亏损超过 {config.DAILY_LOSS_LIMIT_PCT}% 时停止交易
4. 每次开仓前先回想相关教训
5. 平仓后复盘并记录教训
6. 宁可错过不可做亏

## 可用工具
{json.dumps(_TOOL_DESCRIPTIONS, ensure_ascii=False, indent=2)}

## 输出格式
每次返回 JSON：
{{
    "thought": "你的分析和决策思考过程",
    "tool_calls": [
        {{"name": "工具名", "args": 参数对象或{{}}}}
    ],
    "reply": "你要回复的话，如果没有要说的话就写空",
    "need_review": false,  // 是否要做复盘（平仓后设为true）
    "lesson_to_save": ""   // 如果有要记住的教训，填在这里
}}

如果要调用工具，tool_calls 里写工具名和参数。
如果决定不交易，tool_calls 设为空数组。
工具返回的结果会作为新的消息发给你。"""


# ========== 执行工具调用 ==========

def _run_tool(name: str, args: dict):
    """执行一个工具调用并返回结果"""
    if name == "get_market_overview":
        return json.dumps(hl_connector.get_market_overview(), ensure_ascii=False, indent=2)
    elif name == "get_orderbook":
        return json.dumps(hl_connector.get_orderbook(args.get("coin", "BTC")), ensure_ascii=False, indent=2)
    elif name == "get_positions":
        return json.dumps(hl_connector.get_positions(), ensure_ascii=False, indent=2)
    elif name == "get_balance":
        return json.dumps(hl_connector.get_balance(), ensure_ascii=False, indent=2)
    elif name == "open_trade":
        return json.dumps(hl_connector.open_trade(
            args["coin"], args["side"], float(args["size"]), int(args.get("leverage", 1))
        ), ensure_ascii=False, indent=2)
    elif name == "close_trade":
        return json.dumps(hl_connector.close_trade(args["coin"]), ensure_ascii=False, indent=2)
    elif name == "recall_lessons":
        lessons = memory.recall(args.get("context", ""))
        return json.dumps(lessons, ensure_ascii=False, indent=2)
    elif name == "add_lesson":
        lesson = memory.add_lesson(args["lesson"], args.get("tags", []), args.get("outcome", ""))
        return json.dumps(lesson, ensure_ascii=False, indent=2)
    return f"未知工具: {name}"


# ========== 单次交易决策循环 ==========

def run_trade_cycle():
    """
    执行一次交易循环
    1. 收集当前状态
    2. 回想相关教训
    3. DeepSeek 决策
    4. 执行
    5. 返回结果
    """
    # ---- 第 1 步：收集数据 ----
    print(f"\n{'='*40}")
    print(f"交易循环开始：{datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*40}")

    balance = hl_connector.get_balance()
    positions = hl_connector.get_positions()
    overview = hl_connector.get_market_overview()

    # 把当前状态转成文本
    state_text = f"""
当前账户：{json.dumps(balance, ensure_ascii=False)}
当前持仓：{json.dumps(positions, ensure_ascii=False)}
市场概况：{json.dumps(overview[:5] if isinstance(overview, list) else overview, ensure_ascii=False)}  # 取前5个币
"""

    # ---- 第 2 步：回想教训 ----
    relevant_lessons = memory.recall(state_text)
    lessons_text = json.dumps(relevant_lessons, ensure_ascii=False, indent=2) if relevant_lessons else "暂无相关教训"

    # ---- 第 3 步：DeepSeek 决策 ----
    print("  [Agent] 思考中...")
    try:
        response = _client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": f"## 当前状态\n{state_text}\n## 相关历史教训\n{lessons_text}\n## 请分析并决定下一步操作"},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        decision_text = response.choices[0].message.content
        # 提取 JSON
        clean = decision_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
            clean = clean.rsplit("```", 1)[0].strip()
        brace_start = clean.find("{")
        brace_end = clean.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            clean = clean[brace_start:brace_end + 1]
        decision = json.loads(clean)

    except Exception as e:
        return {"error": f"AI 决策失败: {e}", "state": state_text}

    # ---- 第 4 步：执行工具调用 ----
    tool_calls = decision.get("tool_calls", [])
    results = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args", {})
        print(f"  [Agent] 调工具: {name}({args})")
        result = _run_tool(name, args)
        print(f"  [Agent] 结果: {str(result)[:100]}...")
        results.append({"tool": name, "result": result})

    # ---- 第 5 步：处理教训和复盘 ----
    lesson_text = decision.get("lesson_to_save", "")
    if lesson_text:
        memory.add_lesson(lesson_text)

    return {
        "thought": decision.get("thought", ""),
        "reply": decision.get("reply", ""),
        "need_review": decision.get("need_review", False),
        "tool_results": results,
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

    # 获取近期交易
    trades = hl_connector.get_trade_history(days=7)
    if not trades:
        print("  没有近期交易")
        return

    # 计算统计
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

    # 让 AI 分析
    try:
        response = _client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是交易复盘分析师，分析以下近期交易数据，找出问题并提出改进建议。"},
                {"role": "user", "content": json.dumps(trades, ensure_ascii=False, indent=2)},
            ],
            temperature=0.3,
        )
        analysis = response.choices[0].message.content
        print(f"\n  AI 分析：\n  {analysis}")

        # 从分析中提取教训
        lesson_response = _client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "从以下交易复盘分析中提取 1-2 条最重要的教训，以 JSON 格式返回：{\"lessons\": [{\"text\": \"教训内容\", \"tags\": [\"标签1\", \"标签2\"]}]}"},
                {"role": "user", "content": analysis},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        lesson_data = json.loads(lesson_response.choices[0].message.content)
        for l in lesson_data.get("lessons", []):
            memory.add_lesson(l["text"], l.get("tags", []))

    except Exception as e:
        print(f"  复盘 AI 分析失败: {e}")

    return review_data
