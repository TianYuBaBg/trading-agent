"""
止盈检查 + 资金部署脚本

流程：
1. 检查每个仓位的未实现盈亏
2. 盈利 > $1 的仓位直接平仓（不经过 AI，确保执行）
3. 运行正常交易周期：AI 会看到剩余仓位，自行决定是否重新入场或部署闲置资金

使用：python trading_agent/profit_take.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_agent import hl_connector, agent


def main():
    print("=" * 45)
    print("止盈检查 + 资金部署")
    print("=" * 45)

    # 1. 检查持仓
    positions = hl_connector.get_positions()
    if isinstance(positions, str):
        print(f"[止盈] 获取持仓失败: {positions}")
        return

    # 2. 找出盈利 > $1 的仓位，直接平仓
    profitable = [p for p in positions if isinstance(p, dict) and (p.get("pnl") or 0) > 1]

    if profitable:
        for p in profitable:
            coin = p["coin"]
            pnl = p["pnl"]
            print(f"[止盈] {coin} 盈利 ${pnl:.2f} > $1，直接平仓")
            result = hl_connector.close_trade(coin)
            if isinstance(result, dict) and result.get("status") == "已平仓":
                print(f"[止盈] ✓ {coin} 已平仓，盈利 ${pnl:.2f}")
            else:
                print(f"[止盈] ✗ {coin} 平仓失败: {result}")
    else:
        print("[止盈] 无盈利超过 $1 的仓位")

    # 3. 运行交易周期（AI 自行决策补仓、换仓或部署闲置资金）
    #    run_trade_cycle() 内部会检查余额，如果钱不够会自动跳过
    balance = hl_connector.get_balance()
    available = balance.get("available", 0) if isinstance(balance, dict) else 0
    print(f"\n[资金] 可用 ${available:.2f}")

    if available > 10:
        print("[资金] 有闲置资金，运行交易周期...")
        result = agent.run_trade_cycle()
        if isinstance(result, dict):
            reply = (result.get("reply") or "").strip()
            error = result.get("error")
            if error:
                print(f"[资金] {error[:200]}")
            if reply:
                print(f"[AI] {reply[:200]}")
        print("[资金] 交易周期完成")
    else:
        print("[资金] 可用资金不足 $10，跳过交易周期")


if __name__ == "__main__":
    main()
