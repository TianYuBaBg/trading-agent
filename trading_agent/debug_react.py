"""调试 ReAct 循环，看 AI 到底在每轮说了什么"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI
from trading_agent import config, hl_connector, memory
from trading_agent.agent import _build_system_prompt, _run_tool, _TOOL_DESCRIPTIONS, _extract_json, _truncate_results

client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

# 1. 收集状态
balance = hl_connector.get_balance()
positions = hl_connector.get_positions()
overview = hl_connector.get_market_overview()

state_text = f"""当前账户：{json.dumps(balance, ensure_ascii=False)}
当前持仓：{json.dumps(positions, ensure_ascii=False)}
市场概况：{json.dumps(overview[:8] if isinstance(overview, list) else overview, ensure_ascii=False)}"""

lessons_text = json.dumps(memory.recall(state_text), ensure_ascii=False, indent=2)
if not lessons_text.strip() or lessons_text == "[]":
    lessons_text = "暂无相关教训"

# 2. ReAct 循环
messages = [
    {"role": "system", "content": _build_system_prompt()},
    {"role": "user", "content": f"## 当前状态\n{state_text}\n## 相关历史教训\n{lessons_text}\n## 请分析并决定下一步操作"},
]




for round_idx in range(3):
    print(f"\n{'='*40}")
    print(f"  ReAct 第 {round_idx+1} 轮")
    print(f"{'='*40}")

    try:
        resp = client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        if not resp.choices:
            print("  [错误] AI 返回空响应")
            break
        decision = _extract_json(resp.choices[0].message.content)
        if "error" in decision:
            print(f"  [错误] JSON 解析失败: {decision['error']}")
            break
    except Exception as e:
        print(f"  [错误] API 调用异常: {e}")
        break

    thought = decision.get("thought") or ""
    reply = decision.get("reply") or ""
    tool_calls = decision.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        tool_calls = []

    print(f"  思考: {thought[:300]}")
    print(f"  回复: {reply[:200]}")
    print(f"  工具: {[t['name'] for t in tool_calls] if tool_calls else '无（最终决策）'}")

    if not tool_calls:
        print(f"\n  >>> 最终决定 {'(空回复)' if not reply else ''}")
        if reply:
            print(f"  >>> AI 说: {reply}")
        break

    # 执行所有工具，收集结果
    all_results = []
    for tc in tool_calls:
        name = tc.get("name") or ""
        args = tc.get("args") or {}
        print(f"  → 执行: {name}{args}")
        result = _run_tool(name, args)
        print(f"  → 结果: {str(result)[:100]}")
        all_results.append({"tool": name, "result": result})

    # 把全部结果一次性送回给 AI（item 边界安全截断）
    messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})
    results_str = _truncate_results(all_results)
    messages.append({"role": "user", "content": f"## 以下是工具返回结果\n{results_str}\n## 根据这些新数据，分析并得出最终决策。\n## 同时注意：以下工具可能有失败的情况，请检查原因，不要假设操作已成功执行。\n## tool_calls为空 = 最终决策。如需开仓用 open_trade，平仓用 close_trade。"})
