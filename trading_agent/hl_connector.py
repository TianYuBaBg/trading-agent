"""
Hyperliquid 链上连接器

封装 SDK，提供交易智能体需要的所有工具函数：
- 查行情、查深度
- 查持仓、查余额
- 开仓、平仓、设止损
"""

import datetime
import json
import math
import os
import shutil
import time

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account

from trading_agent import config

# 读取钱包配置
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "config.json")


_wallet_cache = None


def _load_wallet():
    """
    从 data/config.json 读取钱包配置（结果缓存，仅首次读取文件）。
    支持两种模式：
      1. API 钱包模式（推荐）：
         { "main_address": "0x...", "api_key": "0x..." }
         main_address 从 Hyperliquid 网站上看
         api_key 从网站 /API 生成
      2. 直连模式：
         { "wallet_private_key": "0x..." }
    """
    global _wallet_cache
    if _wallet_cache is not None:
        return _wallet_cache
    if not os.path.exists(_CONFIG_PATH):
        _wallet_cache = (None, None, None)
        return _wallet_cache
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    # 优先使用 API 钱包模式
    main_addr = cfg.get("main_address") or ""
    api_key = cfg.get("api_key") or ""
    if main_addr and api_key:
        _wallet_cache = (api_key, main_addr, "api")
        return _wallet_cache

    # 降级到直接私钥模式
    pk = cfg.get("wallet_private_key") or ""
    if pk:
        _wallet_cache = (pk, pk, "direct")
        return _wallet_cache

    _wallet_cache = (None, None, None)
    return _wallet_cache


# ========== 初始化连接 ==========

_info = None
_exchange = None
_wallet_address = None
_sz_decimals_cache = {}
_user_state_cache = None


def _ensure_info():
    global _info
    if _info is None:
        _info = Info(constants.MAINNET_API_URL, skip_ws=True)
    return _info


def _ensure_exchange():
    global _exchange, _wallet_address
    if _exchange is None:
        _wallet_address = _ensure_wallet_address()
        sign_key, addr_or_key, mode = _load_wallet()
        if not sign_key:
            raise Exception("请先在 data/config.json 中配置钱包")

        if mode == "api":
            account = Account.from_key(sign_key)
            _exchange = Exchange(account, constants.MAINNET_API_URL, account_address=_wallet_address)
        else:
            account = Account.from_key(sign_key)
            _exchange = Exchange(account, constants.MAINNET_API_URL)
    return _exchange


def _ensure_wallet_address():
    """获取钱包地址（优先从配置读取，兼容两种模式）"""
    global _wallet_address
    if _wallet_address:
        return _wallet_address
    sign_key, addr_or_key, mode = _load_wallet()
    if mode == "api":
        _wallet_address = addr_or_key
    elif sign_key:
        _wallet_address = Account.from_key(sign_key).address
    return _wallet_address


# ========== user_state 缓存（减少重复 RPC）==========

def reset_user_state_cache():
    """重置 user_state 缓存，确保下次读取获取最新数据"""
    global _user_state_cache
    _user_state_cache = None


def _get_user_state(addr: str):
    """获取 user_state（带 per-cycle 缓存，避免一轮内重复 RPC）"""
    global _user_state_cache
    if _user_state_cache is not None:
        return _user_state_cache
    info = _ensure_info()
    _user_state_cache = info.user_state(addr)
    return _user_state_cache


# ========== 查询工具 ==========

def get_market_overview():
    """
    获取所有币种的最新行情
    返回：[{"coin": "BTC", "mid": 50000, "volume": 1.2e9}, ...]
    """
    info = _ensure_info()
    try:
        mids = info.all_mids()
        # 获取成交量 / 资金费率等上下文数据
        meta_ctxs = info.meta_and_asset_ctxs()
        universe = meta_ctxs[0].get("universe") or []
        ctxs = meta_ctxs[1] if len(meta_ctxs) > 1 else []
        # 建立 coin -> ctx 的映射（按下标匹配）
        coin_to_ctx = {}
        for i, asset in enumerate(universe):
            coin = asset.get("name")
            if coin and i < len(ctxs):
                coin_to_ctx[coin] = ctxs[i]

        overview = []
        for coin, mid in mids.items():
            # 过滤 Spot 内部编号（# 或 @ 开头），只保留标准永续合约币种
            if coin.startswith("#") or coin.startswith("@"):
                continue
            item = {"coin": coin, "mid": float(mid)}
            # 补充成交量和资金费率
            ctx = coin_to_ctx.get(coin)
            if ctx:
                item["volume"] = float(ctx.get("dayNtlVlm") or 0)
                item["funding"] = float(ctx.get("funding") or 0)
            overview.append(item)
        # 按成交量降序排列，流动性高的币种在前
        overview.sort(key=lambda x: x.get("volume") or 0, reverse=True)
        return overview
    except Exception as e:
        return f"获取行情失败：{e}"


def get_orderbook(coin: str, depth: int = 5):
    """
    获取某个币的订单簿深度
    coin: "BTC"
    depth: 显示几档
    返回：{"bids": [[price, size], ...], "asks": [[price, size], ...]}
    """
    info = _ensure_info()
    try:
        book = info.l2_snapshot(coin)
        levels = book["levels"]
        # levels[0] = bids（买盘）, levels[1] = asks（卖盘）
        bids = [[float(b["px"]), float(b["sz"])] for b in levels[0][:depth]]
        asks = [[float(a["px"]), float(a["sz"])] for a in levels[1][:depth]]
        return {"bids": bids, "asks": asks}
    except Exception as e:
        return f"获取深度失败：{e}"


def get_candles(coin: str, interval: str = "1h", hours: int = 24):
    """
    获取某个币的 K 线数据（蜡烛图）
    coin: "BTC"
    interval: "1m"/"5m"/"15m"/"1h"/"4h"/"1d"
    hours: 往回看多少小时，自动限制在 [1, 720] 范围内
    返回：[{"time": "HH:MM", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}, ...]
    """
    info = _ensure_info()
    try:
        # 校验 interval，无效值降级为 1h
        valid_intervals = {"1m", "5m", "15m", "1h", "4h", "1d"}
        if interval not in valid_intervals:
            interval = "1h"
        hours = max(1, min(int(hours), 720))
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - hours * 3600 * 1000
        raw = info.candles_snapshot(coin, interval, start_ms, now_ms)
        if not raw:
            return []
        # 限制最多返回 200 条，防止 1m/5m 线数据量撑爆 prompt
        # 从尾部逆向采样，确保最新蜡烛始终被包含
        step = max(1, len(raw) // 200)
        candles = []
        for i in range(len(raw) - 1, -1, -step):
            c = raw[i]
            candles.append({
                "time": time.strftime("%m-%d %H:%M", time.localtime(c["t"] / 1000)),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            })
            if len(candles) >= 200:
                break
        candles.reverse()  # 恢复时间正序
        return candles
    except Exception as e:
        return f"获取 K 线失败：{e}"


def get_positions():
    """
    获取当前持仓
    返回：[{"coin": "BTC", "size": 0.1, "entry_px": 50000, "pnl": 50, ...}]
    """
    info = _ensure_info()
    addr = _ensure_wallet_address()
    if not addr:
        return "未配置钱包地址"
    try:
        state = _get_user_state(addr)
        positions = []
        for pos in (state.get("assetPositions") or []):
            p = pos.get("position")
            if not p:
                continue
            szi = p.get("szi") or 0
            if float(szi) == 0:
                continue
            lev = p.get("leverage") or 1
            if isinstance(lev, dict):
                lev = lev.get("value") or 1
            positions.append({
                "coin": p.get("coin") or "",
                "size": float(szi),
                "direction": "long" if float(szi) > 0 else "short",
                "entry_px": float(p.get("entryPx", 0) or 0),
                "pnl": float(p.get("unrealizedPnl", 0) or 0),
                "leverage": float(lev or 1),
                "liquidation_px": float(p.get("liquidationPx") or 0),
            })
        return positions
    except Exception as e:
        return f"获取持仓失败：{e}"


def get_balance():
    """
    获取账户余额（Perp + Spot 统一视图）
    返回：{"wallet_value": 1000, "available": 800, "margin_used": 200}
    """
    info = _ensure_info()
    addr = _ensure_wallet_address()
    if not addr:
        return "未配置钱包地址"
    try:
        state = _get_user_state(addr)
        cross = state.get("crossMarginSummary") or {}
        isolated = state.get("isolatedMarginSummary") or {}
        perp_value = float(cross.get("accountValue") or 0) + float(isolated.get("accountValue") or 0)
        margin_used = float(cross.get("totalMarginUsed") or 0) + float(isolated.get("totalMarginUsed") or 0)

        # 也查 Spot 余额
        spot_value = 0.0
        try:
            spot_state = info.spot_user_state(addr)
            for b in (spot_state.get("balances") or []):
                if b.get("coin") == "USDC":
                    spot_value = float(b.get("total") or 0)
                    break
        except Exception:
            pass

        total = perp_value + spot_value
        return {
            "wallet_value": round(total, 2),
            "margin_used": round(margin_used, 2),
            "available": round(total - margin_used, 2),
            "perp_value": round(perp_value, 2),
            "spot_value": round(spot_value, 2),
        }
    except Exception as e:
        return f"获取余额失败：{e}"


# ========== 交易工具 ==========

def _get_fill_price(statuses: list, fallback: float) -> float:
    """从 SDK 响应 statuses 中提取实际成交均价 (avgPx)"""
    if statuses and isinstance(statuses[0], dict):
        fill = statuses[0].get("filled") or {}
        if fill and fill.get("avgPx") is not None:
            return float(fill["avgPx"])
    return fallback


def _get_sz_decimals(coin: str) -> int:
    """获取币种的数量精度（szDecimals），结果缓存（一次 meta() 缓存所有币种）"""
    global _sz_decimals_cache
    if coin in _sz_decimals_cache:
        return _sz_decimals_cache[coin]
    info = _ensure_info()
    try:
        meta = info.meta()
        # 一次遍历缓存所有币种的 szDecimals，避免每个新币种都调一次 meta()
        for asset in (meta.get("universe") or []):
            name = asset.get("name")
            if name:
                szd = asset.get("szDecimals")
                _sz_decimals_cache[name] = szd if szd is not None else 4
        return _sz_decimals_cache.get(coin, 4)
    except Exception:
        return 4


def open_trade(coin: str, side: str, size: float, leverage: int = 1, _balance=None, _positions=None):
    """
    开仓
    coin: "BTC"
    side: "long" 或 "short"
    size: 以 USD 计价的仓位大小（例如 100 = 100U）
    leverage: 杠杆倍数
    _balance/_positions: 调用方已查询过的数据，传入可避免重复 RPC
    返回：订单结果
    """
    ex = _ensure_exchange()
    try:
        coin = coin.upper()
        # 检查杠杆为正数且不超过上限
        if leverage <= 0:
            return f"开仓失败：杠杆必须为正整数，收到 {leverage}"
        if leverage > config.MAX_LEVERAGE:
            return f"开仓失败：杠杆 {leverage}x 超过上限 {config.MAX_LEVERAGE}x"

        # 获取当前市价
        info = _ensure_info()
        mids = info.all_mids()
        current_price = float(mids.get(coin, 0))
        if current_price == 0:
            return f"无法获取 {coin} 价格"

        # 校验 size 为正数
        if size <= 0:
            return f"开仓失败：size 必须为正数，收到 {size}"

        # 按 szDecimals 精度计算币数量
        sz_decimals = _get_sz_decimals(coin)
        coin_size = round(size / current_price, sz_decimals)

        # 确保最小名义价值 >= $10（Hyperliquid 要求）
        min_sz = math.ceil(10.0 / current_price * (10 ** sz_decimals)) / (10 ** sz_decimals)
        if coin_size < min_sz:
            coin_size = min_sz

        # 检查总仓位限制（用实际开仓额，避免 min_sz 膨胀后实际超限）
        try:
            if _balance is not None and isinstance(_balance, dict):
                total_value = _balance.get("wallet_value") or 0
            else:
                bal = get_balance()
                total_value = (bal.get("wallet_value") or 0) if isinstance(bal, dict) else 0
            if _positions is not None and isinstance(_positions, list):
                pos_list = _positions
            else:
                pos_list = get_positions()
            existing_notional = 0
            if isinstance(pos_list, list):
                for p in pos_list:
                    coin_price = mids.get(p["coin"])
                    if coin_price is None or float(coin_price) == 0:
                        coin_price = p.get("entry_px") or 0
                    if float(coin_price) <= 0:
                        continue
                    existing_notional += abs(p["size"]) * float(coin_price)
            actual_new = coin_size * current_price
            total_notional = existing_notional + actual_new
            max_allowed = total_value * (config.MAX_POSITION_SIZE_PCT / 100)
            if total_notional > max_allowed:
                min_add_info = f"，最小开仓额 $10 已将请求的 ${size:.1f} 提升至 ${actual_new:.1f}" if actual_new > size else ""
                return f"开仓失败：已有 ${existing_notional:.1f} + 实际开仓 ${actual_new:.1f} = ${total_notional:.1f} 超过限额 ${max_allowed:.1f}（{config.MAX_POSITION_SIZE_PCT}%）{min_add_info}，最多还能开 ${max(0, max_allowed - existing_notional):.1f}"
        except Exception as e:
            return f"开仓失败：仓位限制检查异常（{e}），为安全起见拒绝开仓"

        # 所有校验通过后再设置杠杆（避免 size 非法时杠杆被误设）
        # 检查当前杠杆是否已设好，避免不必要的链上交易
        # 注意：Hyperliquid 的杠杆是按方向独立设置的（long/short 分别存储）
        is_buy = side == "long"
        _need_leverage_update = True
        if isinstance(_positions, list):
            for _p in _positions:
                if (_p.get("coin") == coin and _p.get("direction") == side
                        and abs(float(_p.get("leverage") or 0) - leverage) < 0.01):
                    _need_leverage_update = False
                    break
        if _need_leverage_update:
            ex.update_leverage(leverage, coin, is_buy)

        # 市价单（IOC = Immediate-or-Cancel，市价吃单）
        result = ex.market_open(coin, is_buy, coin_size, current_price, slippage=0.01)
        if not result:
            return "开仓失败：订单返回空结果"

        # 检查订单状态
        statuses = result.get("response") or {}
        statuses = (statuses.get("data") or {}).get("statuses") or []
        if not statuses or not isinstance(statuses[0], dict):
            return f"开仓失败：订单返回状态异常（statuses 为空或格式错误），raw={str(result)[:200]}"
        first = statuses[0]
        if "error" in first:
            return f"开仓失败：{first['error']}"
        if "resting" in first:
            return f"开仓未成交：订单未立即成交（挂单中）"
        # 提取实际成交均价（avgPx），比 all_mids 更精确
        current_price = _get_fill_price(statuses, current_price)

        trade_record = {
            "coin": coin,
            "side": side,
            "size": coin_size,
            "usd_size": round(coin_size * current_price, 2),
            "entry_price": current_price,
            "leverage": leverage,
        }

        # 保存交易记录
        _save_trade(trade_record)

        return {"status": "已开仓", "trade": trade_record, "raw": str(result)[:200]}
    except Exception as e:
        return f"开仓失败：{e}"


def close_trade(coin: str, size=None, _positions=None):
    """
    平仓
    coin: "BTC"
    size: 平仓数量，None = 全平
    _positions: 调用方已查询过的持仓数据，传入可避免重复 RPC
    返回：平仓结果
    """
    ex = _ensure_exchange()
    info = _ensure_info()
    try:
        coin = coin.upper()
        # 获取当前仓位（优先使用调用方传入的数据，避免重复 RPC）
        if _positions is not None and isinstance(_positions, list):
            positions = _positions
        else:
            positions = get_positions()
        if isinstance(positions, str):
            return f"平仓失败：获取持仓失败（{positions}）"
        pos = None
        if isinstance(positions, list):
            for p in positions:
                if p["coin"] == coin:
                    pos = p
                    break
        if not pos:
            return f"没有 {coin} 的持仓"

        mids = info.all_mids()
        current_price = float(mids.get(coin, 0))
        if current_price == 0:
            return f"无法获取 {coin} 价格"

        close_sz = abs(pos["size"]) if size is None else abs(float(size))
        if close_sz <= 0:
            return "平仓失败：平仓数量必须大于 0"
        # 禁止平仓超过实际持仓数量
        max_sz = abs(pos["size"])
        if close_sz > max_sz:
            close_sz = max_sz
        result = ex.market_close(coin, close_sz, current_price, slippage=0.01)
        if not result:
            return "平仓失败：订单返回空结果"

        # 检查平仓订单状态
        statuses = result.get("response") or {}
        statuses = (statuses.get("data") or {}).get("statuses") or []
        if not statuses or not isinstance(statuses[0], dict):
            return f"平仓失败：订单返回状态异常（statuses 为空或格式错误），raw={str(result)[:200]}"
        first = statuses[0]
        if "error" in first:
            return f"平仓失败：{first['error']}"
        if "resting" in first:
            return f"平仓未成交：订单未立即成交（挂单中）"
        # 提取实际成交价（avgPx），比 all_mids 的中间价更精确
        current_price = _get_fill_price(statuses, current_price)

        # 计算盈亏
        pnl = (current_price - pos["entry_px"]) * close_sz
        if pos["size"] < 0:  # 空单
            pnl = -pnl

        # 更新交易记录
        _update_trade_close(coin, current_price, pnl)

        # 返回剩余数量，方便调用方判断是否为全平（remaining=0 表示已全部平完）
        sz_decimals = _get_sz_decimals(coin)
        remaining = round(max(0, max_sz - close_sz), sz_decimals)

        return {
            "status": "已平仓",
            "coin": coin,
            "exit_price": current_price,
            "pnl": round(pnl, 2),
            "remaining_size": remaining,
            "raw": str(result)[:200],
        }
    except Exception as e:
        return f"平仓失败：{e}"


# ========== 交易记录 ==========

_TRADES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trades.json")


def _load_trades():
    """读取 trades.json，文件损坏时备份并重建"""
    if not os.path.exists(_TRADES_FILE):
        return []
    try:
        with open(_TRADES_FILE, encoding="utf-8") as f:
            trades = json.load(f)
        if isinstance(trades, list):
            return [t for t in trades if isinstance(t, dict)]
    except (json.JSONDecodeError, ValueError):
        pass
    # 文件损坏，备份
    backup = _TRADES_FILE + ".bak"
    try:
        shutil.copy2(_TRADES_FILE, backup)
    except Exception:
        pass
    print(f"  [交易记录] trades.json 损坏，已备份到 {backup}，重新创建")
    with open(_TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    return []


def _save_trade(trade: dict):
    """保存一笔交易到本地"""
    os.makedirs(os.path.dirname(_TRADES_FILE), exist_ok=True)
    trades = _load_trades()
    trade["open_time"] = datetime.datetime.now().isoformat()
    trade["id"] = max(((t.get("id") or 0) for t in trades), default=0) + 1
    trades.append(trade)
    with open(_TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)


def _update_trade_close(coin: str, exit_price: float, pnl: float):
    """平仓后更新交易记录：标记最近一笔该币种的开仓交易为已平仓"""
    if not os.path.exists(_TRADES_FILE):
        return
    trades = _load_trades()
    # 标记该币种最近一笔未平仓记录为已平仓
    now = datetime.datetime.now().isoformat()
    found = False
    for t in reversed(trades):
        if t.get("coin") == coin and not t.get("close_time"):
            t["close_time"] = now
            t["exit_price"] = exit_price
            t["pnl"] = round(pnl, 2)
            t["status"] = "closed"
            found = True
            break  # 只关最近一笔，避免多笔重复的 PnL
    if found:
        with open(_TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(trades, f, ensure_ascii=False, indent=2)


def get_trade_history(days: int = 7):
    """
    从链上获取最近成交记录（使用 Hyperliquid 真实成交数据）
    days: 最近几天
    返回：交易列表，每条包含 coin, side, price, size, pnl, time
    """
    info = _ensure_info()
    addr = _ensure_wallet_address()
    if not addr:
        return []
    try:
        fills = info.user_fills(addr)
        if not isinstance(fills, list):
            return []
        cutoff = int(time.time() * 1000) - days * 86400 * 1000
        recent = []
        for f in fills:
            # 跳过关键字段缺失或无效的成交记录，避免单条坏数据炸掉整个函数
            fill_time = f.get("time")
            if not fill_time:
                continue
            try:
                fill_time = int(fill_time)
            except (TypeError, ValueError):
                continue
            # Hyperliquid 按时间倒序返回，遇到 cutoff 后即可停止
            if fill_time < cutoff:
                break
            px = f.get("px")
            sz = f.get("sz")
            if px is None or sz is None:
                continue
            try:
                price = float(px)
                size = float(sz)
            except (TypeError, ValueError):
                continue
            if price <= 0 or size <= 0:
                continue
            raw_side = f.get("side")
            side_str = "long" if raw_side == "B" else "short" if raw_side == "S" else "unknown"
            recent.append({
                "coin": f.get("coin") or "",
                "side": side_str,
                "price": price,
                "size": size,
                "pnl": float(f.get("closedPnl") or 0),
                "dir": f.get("dir") or "",
                "time": time.strftime("%m-%d %H:%M", time.localtime(fill_time / 1000)),
            })
        return recent
    except Exception as e:
        return f"获取成交记录失败：{e}"


def transfer_spot_to_perp(amount: float):
    """把 USDC 从 Spot 转到永续合约账户"""
    ex = _ensure_exchange()
    try:
        result = ex.usd_class_transfer(amount, to_perp=True)
        if not result:
            return "转账失败：SDK 返回空结果"
        # SDK 可能在返回的 dict 中嵌了错误
        raw_str = str(result)
        if '"status":"err"' in raw_str or '"status": "err"' in raw_str or '"response":"Must deposit"' in raw_str:
            return f"转账失败：API 钱包模式不支持转账（如需转账请用直连私钥模式），raw={raw_str[:150]}"
        return {"status": "ok", "amount": amount, "raw": raw_str[:200]}
    except Exception as e:
        return f"转账失败：{e}"
