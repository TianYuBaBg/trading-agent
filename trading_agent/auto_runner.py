"""
自动交易循环（后台版）
每 300 秒跑一次交易决策，不需要键盘输入
"""
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_agent import agent, config

INTERVAL = config.AUTO_TRADE_INTERVAL  # 300 秒

print("=" * 45)
print("  自动交易循环已启动")
print(f"  每 {INTERVAL} 秒检查一次市场")
print("=" * 45)

cycle_count = 0
last_review_time = 0

while True:
    try:
        cycle_count += 1
        result = agent.run_trade_cycle()

        if result.get("need_review"):
            print("\n[复盘] 触发复盘...")
            agent.run_review()

        if cycle_count <= 5 and result.get("thought"):
            print(f"\n[思考] {result['thought']}")
        if result.get("reply"):
            print(f"\n[AI] {result['reply']}")

        # 每 6 小时复盘
        now = time.time()
        if now - last_review_time > 21600:
            print("\n[复盘] 6 小时定期复盘...")
            agent.run_review()
            last_review_time = now

        print(f"\n--- 等待 {INTERVAL} 秒进入下一周期 ---")
        time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\n自动交易已停止")
        break
    except Exception as e:
        print(f"\n异常: {e}")
        time.sleep(60)
