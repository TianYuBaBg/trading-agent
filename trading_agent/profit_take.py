"""
止盈 + 止损 + 资金部署脚本

策略（全自动，不需要 AI 参与执行）：
  止盈：名义价值 × 0.3%，最低 $0.30 —— 小盈利及时锁定
  止损：名义价值 × -5%，最低 -$5.00 —— 兜底止损，防 AI SL 失效
  说明：硬止损比 AI 设置的 SL（~-4%）宽约 25%，只在 AI SL 未生效时触发

流程：
1. 检查每个仓位：盈利 ≥ 阈值 → 止盈；亏损 ≤ 阈值 → 止损
2. 运行交易周期：AI 看到剩余仓位，自行决策换仓或部署闲置资金

使用：python trading_agent/profit_take.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_agent import hl_connector, agent


def _notional(position: dict) -> float:
    size = abs(position.get("size") or 0)
    entry_px = position.get("entry_px") or 0
    return size * entry_px


def _profit_threshold(position: dict) -> float:
    """止盈阈值：名义价值 × 0.3%，最低 $0.30"""
    return max(_notional(position) * 0.003, 0.30)


def _stop_loss_threshold(position: dict) -> float:
    """止损阈值：名义价值 × -5%，最低 -$5.00"""
    return min(_notional(position) * -0.05, -5.00)


def _close_position(p, reason: str):
    coin = p["coin"]
    pnl = p.get("pnl") or 0
    ntl = _notional(p)
    print(f"[{reason}] {coin} PnL ${pnl:.2f} (名义 ${ntl:.0f})，直接平仓")
    result = hl_connector.close_trade(coin)
    if isinstance(result, dict) and result.get("status") == "已平仓":
        print(f"[{reason}] [OK] {coin} 已平仓")
    else:
        print(f"[{reason}] [FAIL] {coin} 平仓失败: {result}")


def main():
    print("=" * 45)
    print("止盈 + 止损 + 资金部署")
    print("  止盈 0.3% | 止损 -5%（兜底）")
    print("=" * 45)

    # 1. 检查持仓
    positions = hl_connector.get_positions()
    if isinstance(positions, str):
        print(f"[检查] 获取持仓失败: {positions}")
        return

    # 2. 止盈 + 止损检查
    closed_any = False
    for p in positions:
        if not isinstance(p, dict):
            continue
        pnl = p.get("pnl") or 0
        if pnl >= _profit_threshold(p):
            _close_position(p, "止盈")
            closed_any = True
        elif pnl <= _stop_loss_threshold(p):
            _close_position(p, "止损")
            closed_any = True

    if not closed_any:
        print("[检查] 无仓位触发止盈或止损")

    # 3. 运行交易周期（AI 自行决策补仓、换仓或部署闲置资金）
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
