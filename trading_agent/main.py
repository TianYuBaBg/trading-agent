"""
交易智能体 — 入口

两种模式：
1. 手动模式：你发指令，AI 执行交易
2. 自动模式：AI 每隔 N 分钟自动检查市场并交易

使用前：
  1. 在 data/config.json 中配置钱包私钥
  2. 确保账户有资金
"""

import sys
import json
import os
import time
import datetime

from trading_agent import config
from trading_agent import agent
from trading_agent import hl_connector

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "config.json")


def _ensure_config():
    """确保配置文件存在"""
    if not os.path.exists(_CONFIG_PATH):
        example = {
            "wallet_private_key": "",
            "note": "去 Hyperliquid 导出你的钱包私钥填入上面",
        }
        with open(_CONFIG_PATH, "w") as f:
            json.dump(example, f, ensure_ascii=False, indent=2)
        print("请在 data/config.json 中配置 wallet_private_key")
        return False
    with open(_CONFIG_PATH) as f:
        cfg = json.load(f)
    if not cfg.get("wallet_private_key"):
        print("请在 data/config.json 中配置 wallet_private_key")
        return False
    return True


# ========== 手动模式 ==========

def manual_mode():
    """你说话，AI 执行"""
    print("=" * 45)
    print("  交易智能体 — 手动模式")
    print("  输入你的想法，AI 会先看市场再决策")
    print("  输入 '复盘' 运行复盘分析")
    print("  输入 '状态' 看当前账户状况")
    print("  输入 '教训' 看历史教训")
    print("  输入 '自动' 切换到自动模式")
    print("  输入 exit 退出")
    print("=" * 45)

    while True:
        try:
            cmd = input("\n> ").strip()
            if cmd.lower() in ["exit", "quit"]:
                print("再见！")
                break

            if cmd == "状态":
                print(f"余额: {hl_connector.get_balance()}")
                print(f"持仓: {hl_connector.get_positions()}")
                continue

            if cmd == "复盘":
                agent.run_review()
                continue

            if cmd == "教训":
                from trading_agent.memory import get_all_lessons
                for l in get_all_lessons():
                    print(f"  #{l['id']} [{l['date']}] {l['lesson']}")
                continue

            if cmd == "自动":
                print("切换到自动模式")
                auto_mode()
                return

            # 走 AI
            result = agent.run_trade_cycle()
            if result.get("reply"):
                print(f"\n{result['reply']}")
            if result.get("thought"):
                print(f"\n[思考] {result['thought'][:200]}...")

        except KeyboardInterrupt:
            print("\n再见！")
            break


# ========== 自动模式 ==========

def auto_mode():
    """AI 自动检查市场并交易"""
    print("=" * 45)
    print(f"  交易智能体 — 自动模式")
    print(f"  每 {config.AUTO_TRADE_INTERVAL} 秒检查一次市场")
    print("  输入回车回到手动模式")
    print("=" * 45)

    import threading

    running = True
    last_review_time = 0

    def _check_input():
        nonlocal running
        while running:
            cmd = input()
            if cmd.strip() == "":
                running = False
                print("\n切换到手动模式")
            elif cmd == "复盘":
                agent.run_review()
            elif cmd == "状态":
                print(f"余额: {hl_connector.get_balance()}")
                print(f"持仓: {hl_connector.get_positions()}")

    # 开一个线程监听键盘输入
    input_thread = threading.Thread(target=_check_input, daemon=True)
    input_thread.start()

    cycle_count = 0
    while running:
        try:
            cycle_count += 1
            result = agent.run_trade_cycle()

            # 如果需要复盘
            if result.get("need_review"):
                agent.run_review()

            # 如果是第一个周期就显示思考过程
            if cycle_count <= 1 and result.get("thought"):
                print(f"\n[思考] {result['thought']}")
            if result.get("reply"):
                print(f"\n[AI] {result['reply']}")

            # 定期复盘（每 6 小时）
            now = time.time()
            if now - last_review_time > 21600:  # 6 小时
                agent.run_review()
                last_review_time = now

            # 等待下一个周期
            for _ in range(config.AUTO_TRADE_INTERVAL):
                if not running:
                    break
                time.sleep(1)

        except Exception as e:
            print(f"自动交易异常: {e}")
            time.sleep(60)


# ========== 主入口 ==========

def main():
    print(f"{'='*45}")
    print(f"  Hyperliquid 交易智能体")
    print(f"  启动时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*45}")

    if not _ensure_config():
        sys.exit(1)

    # 测试连接
    try:
        balance = hl_connector.get_balance()
        if isinstance(balance, dict) and "wallet_value" in balance:
            print(f"  连接成功！账户余额: {balance['wallet_value']:.2f} USD")
        else:
            print(f"  无法连接 Hyperliquid: {balance}")
            sys.exit(1)
    except Exception as e:
        print(f"  连接失败: {e}")
        print("  请检查网络和钱包配置")
        sys.exit(1)

    # 选择模式
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        auto_mode()
    else:
        manual_mode()


if __name__ == "__main__":
    main()
