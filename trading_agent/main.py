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

    # 检查是否已有钱包
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH, encoding="utf-8") as f:
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
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
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
    """确保配置文件存在，支持两种钱包模式"""
    if not os.path.exists(_CONFIG_PATH):
        print("首次使用，请选择钱包方式：")
        print("  1) 创建新钱包 — 生成新地址，往里面打钱")
        print("  2) 配置 API 钱包 — 用网站邮箱登录生成的 API 私钥（推荐）")
        choice = input("\n请选择 (1/2): ").strip()

        if choice == "2":
            return _setup_api_wallet()
        else:
            create_wallet()
            return True

    # 已有配置文件，检查是否有效
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    # API 钱包模式：有 main_address 和 api_key 就行
    if cfg.get("main_address") and cfg.get("api_key"):
        return True

    # 直连模式：有 wallet_private_key 就行
    if cfg.get("wallet_private_key"):
        return True

    # 配置无效
    print("config.json 配置不完整。")
    print("  1) 重新创建钱包")
    print("  2) 配置 API 钱包（推荐）")
    choice = input("\n请选择 (1/2): ").strip()
    if choice == "2":
        return _setup_api_wallet()
    else:
        create_wallet()
        return True


def _setup_api_wallet():
    """引导用户配置 API 钱包"""
    print("\n如何获取 API 钱包：")
    print("  1. 打开 https://app.hyperliquid.xyz")
    print("  2. 用邮箱登录")
    print("  3. 点 API → 创建 API 钱包")
    print("  4. 复制主钱包地址 和 API 私钥")
    print()

    addr = input("主钱包地址 (0x...): ").strip()
    key = input("API 私钥 (0x...): ").strip()

    if not addr or not key:
        print("地址和私钥不能为空")
        return False

    config_data = {
        "main_address": addr,
        "api_key": key,
        "note": "API 钱包模式。main_address 是主钱包地址，api_key 是网站生成的 API 私钥",
    }
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ API 钱包配置完成！main_address: {addr[:20]}...")
    return True


# ========== 手动模式 ==========

def manual_mode():
    """你说话，AI 执行"""
    print("=" * 45)
    print("  交易智能体 — 手动模式")
    print("  输入你的想法，AI 会先看市场再决策")
    print("  输入 '创建钱包' 生成新钱包")
    print("  输入 'API 钱包' 配置 API 钱包（网站邮箱登录生成）")
    print("  输入 '转入资金' 把 Spot 的钱转到永续合约账户")
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

            if cmd in ["API 钱包", "api"]:
                _setup_api_wallet()
                continue

            if cmd in ["转入资金", "转入"]:
                try:
                    amt_input = input("转入金额（直接回车 = 全部转入）: ").strip()
                    if amt_input:
                        amount = float(amt_input)
                    else:
                        amount = 0  # 下面用正数判断替代
                except (ValueError, TypeError):
                    print("金额格式无效")
                    continue
                if amount <= 0:
                    bal = hl_connector.get_balance()
                    spot = bal.get("spot_value") if isinstance(bal, dict) else 0
                    if spot <= 0:
                        print("Spot 账户无可用 USDC")
                        continue
                    amount = spot
                result = hl_connector.transfer_spot_to_perp(amount)
                print(f"转入结果: {result}")
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
    last_review_time = time.time()

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

            if result.get("error"):
                print(f"\n[错误] {result['error'][:200]}")

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
