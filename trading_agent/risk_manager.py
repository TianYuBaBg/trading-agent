"""
风险监控 + 资金部署脚本

策略：
  止损：名义价值 × -5%，最低 -$5.00 —— 兜底止损，防 AI SL 失效
  说明：硬止损比 AI 设置的 SL（~-4%）宽约 25%，只在 AI SL 未生效时触发
  止盈：交由 AI 通过 exchange TP 订单管理，脚本不做硬止盈

流程：
1. 检查每个仓位：亏损 ≤ 阈值 → 止损
2. 运行交易周期：AI 自行决策开仓或部署闲置资金

使用：python trading_agent/risk_manager.py
"""

import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_agent import hl_connector, agent


def _notional(position: dict) -> float:
    size = abs(position.get("size") or 0)
    entry_px = position.get("entry_px") or 0
    return size * entry_px


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


def _verify_state(positions: list, balance: dict):
    """验证持仓 vs 挂单一致性，检查成交记录，发现异常自动处理"""
    print("\n[验证] === 状态一致性检查 ===")

    # 1. 获取当前所有挂单
    orders = hl_connector.get_open_orders()
    if isinstance(orders, str):
        print(f"[验证] 获取挂单失败: {orders}")
        return
    if not orders:
        print("[验证] 无挂单")
        return

    pos_coins = {p.get("coin") for p in positions if isinstance(p, dict)}
    order_coins = {o.get("coin") for o in orders if isinstance(o, dict)}

    # 2. 找出孤儿单（有挂单但无持仓的币种）
    orphan_coins = order_coins - pos_coins
    if orphan_coins:
        print(f"[验证] [孤儿单] {', '.join(sorted(orphan_coins))} — 有挂单但无持仓，正在清理...")
        for oc in sorted(orphan_coins):
            result = hl_connector.close_trade(oc)
            print(f"[验证] [孤儿单] {oc} → {str(result)[:80]}")
    else:
        print("[验证] 无孤儿单")

    # 3. 检查 dust 仓位（< 最小可交易数量）
    dust_coins = []
    for p in positions:
        if not isinstance(p, dict):
            continue
        sz = abs(p.get("size") or 0)
        coin = p.get("coin", "?")
        entry = p.get("entry_px") or 0
        ntl = sz * entry
        if sz > 0 and ntl < 0.50:  # 名义价值 < $0.50 视为 dust
            dust_coins.append(f"{coin} {sz:.4f}@${entry:.2f}=${ntl:.2f}")
    if dust_coins:
        print(f"[验证] [Dust] 微小仓位: {'; '.join(dust_coins)}")
    else:
        print("[验证] 无 dust")

    # 4. 检查费率方向一致性：持仓方向 vs 当前费率方向，不一致→自动平仓
    for p in positions:
        if not isinstance(p, dict):
            continue
        coin = p.get("coin")
        side = p.get("direction")
        fr = hl_connector.get_funding_rate_for_coin(coin)
        if not isinstance(fr, dict):
            continue
        req_dir = fr.get("direction")  # funding>0→"short", funding<0→"long"
        if req_dir and req_dir != side:
            print(f"[验证] [费率翻转] {coin} {side} 但费率{fr.get('funding_pct')}%（{req_dir == 'long' and '空付多' or '多付空'}），方向错误，自动平仓")
            result = hl_connector.close_trade(coin)
            status = (result.get("status") if isinstance(result, dict) else str(result)) or "?"
            print(f"[验证] [费率翻转] {coin} 平仓结果: {status}")

    # 5. 检查重复挂单（同币种同方向的多个 trigger 单）
    order_types = Counter()
    for o in orders:
        coin = o.get("coin", "?")
        side = o.get("side", "?")
        trigger = o.get("trigger_px")
        order_types[f"{coin}/{side}@{trigger}"] += 1
    dupes = {k: v for k, v in order_types.items() if v > 2}  # 超过2个相同trigger为可疑
    if dupes:
        for k, v in dupes.items():
            print(f"[验证] [重复单] {k} ×{v}")

    # 5. 检查近期成交记录（看有无意外平仓）
    try:
        trades = hl_connector.get_trade_history(days=1)
        if isinstance(trades, list):
            closed_recently = [t for t in trades if t.get("pnl") is not None]
            recent_pos = {p.get("coin") for p in positions if isinstance(p, dict)}
            for t in closed_recently[-3:]:
                coin = t.get("coin", "?")
                pnl = t.get("pnl", 0)
                if coin not in recent_pos:
                    print(f"[验证] [历史平仓] {t.get('time')} {coin} PnL${pnl:.2f}（已不在持仓中）")
        # 计算今日总盈亏
        today_pnl = sum(t["pnl"] for t in closed_recently if t.get("pnl") is not None)
        wallet_val = balance.get("wallet_value") or 0
        print(f"[验证] 今日已实现 PnL ${today_pnl:.2f} | 账户 ${wallet_val:.2f}")
    except Exception as e:
        print(f"[验证] 获取成交记录失败: {e}")


def main():
    print("=" * 45)
    print("风险监控 + 资金部署")
    print("  止损 -5%（兜底）| 止盈由 AI 管理")
    print("=" * 45)

    # 1. 检查持仓
    positions = hl_connector.get_positions()
    if isinstance(positions, str):
        print(f"[检查] 获取持仓失败: {positions}")
        return

    # 2. 止损检查（不做硬止盈，交给 AI 通过 exchange TP 管理）
    closed_any = False
    for p in positions:
        if not isinstance(p, dict):
            continue
        pnl = p.get("pnl") or 0
        if pnl <= _stop_loss_threshold(p):
            _close_position(p, "止损")
            closed_any = True

    if not closed_any:
        print("[检查] 无仓位触发止损")

    # 3. 状态一致性验证
    balance = hl_connector.get_balance()
    if isinstance(balance, dict):
        _verify_state(positions, balance)
    else:
        print("[验证] 获取余额失败，跳过验证")

    # 4. 运行交易周期（AI 自行决策补仓、换仓或部署闲置资金）
    if isinstance(balance, str):
        print("[资金] 获取余额失败，跳过交易周期")
        return
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
