"""
服务器部署脚本 — 永续循环版
每 10 分钟运行一次风险监控 + 交易周期，带日志和自动重启

使用：
  python run_server.py                  # 前台运行
  nohup python run_server.py > server.log 2>&1 &    # 后台运行（Linux）
"""

import time
import sys
import os
import datetime
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_agent import risk_manager, agent

INTERVAL = 600  # 10 分钟
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "server.log")


def log(msg: str):
    """带时间戳的日志"""
    ts = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def main():
    log("=" * 45)
    log("服务器监控启动")
    log(f"周期: 每 {INTERVAL} 秒")
    log("=" * 45)

    cycle = 0
    last_review = 0
    fail_count = 0

    while True:
        try:
            cycle += 1
            log(f"=== 第 {cycle} 周期 ===")

            # 1. 运行风险监控（止损 + 状态验证 + 交易周期）
            risk_manager.main()

            # 2. 每 6 小时复盘一次
            now = time.time()
            if now - last_review > 21600:  # 6 小时
                log("触发定期复盘...")
                try:
                    agent.run_review()
                except Exception as e:
                    log(f"复盘失败: {e}")
                last_review = now

            fail_count = 0  # 成功一次重置失败计数

        except KeyboardInterrupt:
            log("用户中断，停止运行")
            break
        except Exception as e:
            fail_count += 1
            log(f"异常: {e}")
            log(traceback.format_exc()[:500])
            if fail_count >= 5:
                log("连续 5 次异常，退出")
                break

        log(f"等待 {INTERVAL} 秒...")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
