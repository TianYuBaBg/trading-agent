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


def create_wallet():
    """生成新的 Hyperliquid 钱包"""
    from eth_account import Account
    import json
    import os

    # 检查是否已有钱包
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            existing = json.load(f)
        if existing.get("wallet_private_key"):
            print("\n⚠️  你已经有一个钱包了！创建新钱包会覆盖旧的。")
            print(f"   当前地址: {existing['wallet_address']}")
            print(f"   如果旧钱包里还有钱，先转走再创建新的。")
            confirm = input("\n确认覆盖？(输入 YES 确认): ").strip()
            if confirm != "YES":
                print("已取消")
                return None

    print("\n正在生成新钱包...")
    account = Account.create()

    address = account.address
    private_key = account.key.hex()

    # 保存到 config.json
    config_data = {
        "wallet_private_key": private_key,
        "wallet_address": address,
        "note": "这是自动生成的钱包，往这个地址转 USDC 或 ETH 就可以开始交易",
    }
    with open(_CONFIG_PATH, "w") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*45}")
    print(f"  🎉 钱包已创建！")
    print(f"  {'='*45}")
    print(f"  地址: {address}")
    print(f"  私钥: {private_key[:20]}...（已保存到 config.json）")
    print(f"  {'='*45}")
    print(f"  ⚠️  重要：往这个地址转 USDC 才能交易")
    print(f"  ⚠️  私钥存在你本地，不要外传")
    print(f"  {'='*45}\n")

    return address


def _ensure_config():
    """确保配置文件存在，不存在则提示创建钱包"""
    if not os.path.exists(_CONFIG_PATH):
        print("检测到首次使用，需要创建一个钱包。")
        print("输入 Y 创建新钱包，输入 N 手动配置已有钱包")
        choice = input("\n是否创建钱包？(Y/n): ").strip().lower()
        if choice in ["", "y", "yes"]:
            create_wallet()
            return True
        else:
            example = {
                "wallet_private_key": "",
                "wallet_address": "",
                "note": "填写你的 Hyperliquid 钱包私钥",
            }
            with open(_CONFIG_PATH, "w") as f:
                json.dump(example, f, ensure_ascii=False, indent=2)
            print("请在 data/config.json 中配置 wallet_private_key")
            return False

    with open(_CONFIG_PATH) as f:
        cfg = json.load(f)
    if not cfg.get("wallet_private_key"):
        print("config.json 中缺少 wallet_private_key")
        print("输入 Y 重新创建钱包，输入 N 手动配置")
        choice = input("\n重新创建钱包？(Y/n): ").strip().lower()
        if choice in ["", "y", "yes"]:
            create_wallet()
            return True
        return False
    return True


# ========== 手动模式 ==========

def manual_mode():
    """你说话，AI 执行"""
    print("=" * 45)
    print("  交易智能体 — 手动模式")
    print("  输入你的想法，AI 会先看市场再决策")
    print("  输入 '创建钱包' 生成新钱包")
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

            if cmd in ["创建钱包", "新建钱包"]:
                create_wallet()
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
