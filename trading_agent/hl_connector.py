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
    main_addr = cfg.get("main_address", "")
    api_key = cfg.get("api_key", "")
    if main_addr and api_key:
        _wallet_cache = (api_key, main_addr, "api")
        return _wallet_cache

    # 降级到直接私钥模式
    pk = cfg.get("wallet_private_key", "")
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
        universe = meta_ctxs[0].get("universe", [])
        ctxs = meta_ctxs[1] if len(meta_ctxs) > 1 else []
        # 建立 coin -> ctx 的映射（按下标匹配）
        coin_to_ctx = {}
        for i, asset in enumerate(universe):
            coin = asset["name"]
            if i < len(ctxs):
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
        overview.sort(key=lambda x: x.get("volume", 0), reverse=True)
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
        hours = max(1, min(int(hours), 720))
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - hours * 3600 * 1000
        raw = info.candles_snapshot(coin, interval, start_ms, now_ms)
        if not raw:
            return []
        # 限制最多返回 200 条，防止 1m/5m 线数据量撑爆 prompt
        step = max(1, len(raw) // 200)
        candles = []
        for i, c in enumerate(raw):
            if i % step != 0:
                continue
            candles.append({
                "time": time.strftime("%m-%d %H:%M", time.localtime(c["t"] / 1000)),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            })
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
        state = info.user_state(addr)
        positions = []
        for pos in state.get("assetPositions", []):
            p = pos.get("position")
            if not p:
                continue
            szi = p.get("szi", 0)
            if szi is None or float(szi) == 0:
                continue
            lev = p.get("leverage", 1)
            if isinstance(lev, dict):
                lev = lev.get("value", 1)
            positions.append({
                "coin": p.get("coin", ""),
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
        state = info.user_state(addr)
        cross = state.get("crossMarginSummary", {})
        perp_value = float(cross.get("accountValue", 0))
        margin_used = float(cross.get("totalMarginUsed", 0))

        # 也查 Spot 余额
        spot_value = 0.0
        try:
            spot_state = info.spot_user_state(addr)
            for b in spot_state.get("balances", []):
                if b.get("coin") == "USDC":
                    spot_value = float(b.get("total", 0))
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
        fill = statuses[0].get("filled", {})
        if fill and fill.get("avgPx"):
            return float(fill["avgPx"])
    return fallback


def _get_sz_decimals(coin: str) -> int:
    """获取币种的数量精度（szDecimals），结果缓存"""
    global _sz_decimals_cache
    if coin in _sz_decimals_cache:
        return _sz_decimals_cache[coin]
    info = _ensure_info()
    try:
        meta = info.meta()
        for asset in meta.get("universe", []):
            if asset["name"] == coin:
                _sz_decimals_cache[coin] = asset["szDecimals"]
                return asset["szDecimals"]
        _sz_decimals_cache[coin] = 4
        return 4
    except Exception:
        return 4


def open_trade(coin: str, side: str, size: float, leverage: int = 1):
    """
    开仓
    coin: "BTC"
    side: "long" 或 "short"
    size: 以 USD 计价的仓位大小（例如 100 = 100U）
    leverage: 杠杆倍数
    返回：订单结果
    """
    ex = _ensure_exchange()
    try:
        # 检查杠杆上限
        if leverage > config.MAX_LEVERAGE:
            return f"开仓失败：杠杆 {leverage}x 超过上限 {config.MAX_LEVERAGE}x"
        # 设置杠杆
        ex.update_leverage(leverage, coin, True)

        # 获取当前市价
        info = _ensure_info()
        mids = info.all_mids()
        current_price = float(mids.get(coin, 0))
        if current_price == 0:
            return f"无法获取 {coin} 价格"

        # 按 szDecimals 精度计算币数量
        sz_decimals = _get_sz_decimals(coin)
        coin_size = round(size / current_price, sz_decimals)

        # 确保最小名义价值 >= $10（Hyperliquid 要求，向上取整）
        min_sz = math.ceil(10.0 / current_price * (10 ** sz_decimals)) / (10 ** sz_decimals)
        if coin_size < min_sz:
            coin_size = min_sz

        # 检查总仓位是否超过限制
        try:
            bal = get_balance()
            total_value = bal.get("wallet_value", 0) if isinstance(bal, dict) else 0
            pos_list = get_positions()
            existing_notional = 0
            if isinstance(pos_list, list):
                existing_notional = sum(abs(p["size"]) * float(mids.get(p["coin"], p.get("entry_px", 0))) for p in pos_list)
            new_notional = coin_size * current_price
            total_notional = existing_notional + new_notional
            max_allowed = total_value * (config.MAX_POSITION_SIZE_PCT / 100)
            if total_notional > max_allowed:
                max_add = max(0, max_allowed - existing_notional)
                return f"开仓失败：总仓位 ${total_notional:.1f} 超过限制 ${max_allowed:.1f}（{config.MAX_POSITION_SIZE_PCT}%），最多还能开 ${max_add:.1f}"
        except Exception as e:
            return f"开仓失败：仓位限制检查异常（{e}），为安全起见拒绝开仓"

        is_buy = side == "long"

        # 市价单（IOC = Immediate-or-Cancel，市价吃单）
        result = ex.market_open(coin, is_buy, coin_size, current_price, slippage=0.01)

        # 检查订单状态
        statuses = result.get("response", {}).get("data", {}).get("statuses", [])
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


def close_trade(coin: str, size: float = None):
    """
    平仓
    coin: "BTC"
    size: 平仓数量，None = 全平
    返回：平仓结果
    """
    ex = _ensure_exchange()
    info = _ensure_info()
    try:
        # 获取当前仓位
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

        close_sz = abs(pos["size"]) if size is None else abs(float(size))
        if close_sz <= 0:
            return "平仓失败：平仓数量必须大于 0"
        result = ex.market_close(coin, close_sz, current_price, slippage=0.01)

        # 检查平仓订单状态
        statuses = result.get("response", {}).get("data", {}).get("statuses", [])
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

        return {
            "status": "已平仓",
            "coin": coin,
            "exit_price": current_price,
            "pnl": round(pnl, 2),
            "raw": str(result)[:200],
        }
    except Exception as e:
        return f"平仓失败：{e}"


# ========== 交易记录 ==========

_TRADES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trades.json")


def _save_trade(trade: dict):
    """保存一笔交易到本地"""
    os.makedirs(os.path.dirname(_TRADES_FILE), exist_ok=True)
    trades = []
    if os.path.exists(_TRADES_FILE):
        try:
            with open(_TRADES_FILE, encoding="utf-8") as f:
                trades = json.load(f)
        except (json.JSONDecodeError, ValueError):
            trades = []
    trade["open_time"] = datetime.datetime.now().isoformat()
    trade["id"] = max((t.get("id", 0) for t in trades), default=0) + 1
    trades.append(trade)
    with open(_TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)


def _update_trade_close(coin: str, exit_price: float, pnl: float):
    """平仓后更新交易记录：标记最近一笔该币种的开仓交易为已平仓"""
    if not os.path.exists(_TRADES_FILE):
        return
    try:
        with open(_TRADES_FILE, encoding="utf-8") as f:
            trades = json.load(f)
    except (json.JSONDecodeError, ValueError):
        trades = []
    # 标记该币种最近一笔未平仓记录为已平仓
    now = datetime.datetime.now().isoformat()
    for t in reversed(trades):
        if t.get("coin") == coin and not t.get("close_time"):
            t["close_time"] = now
            t["exit_price"] = exit_price
            t["pnl"] = round(pnl, 2)
            t["status"] = "closed"
            break  # 只关最近一笔，避免多笔重复的 PnL
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
            if f["time"] < cutoff:
                continue
            recent.append({
                "coin": f["coin"],
                "side": "long" if f["side"] == "B" else "short",
                "price": float(f["px"]),
                "size": float(f["sz"]),
                "pnl": float(f.get("closedPnl", 0)),
                "dir": f.get("dir", ""),
                "time": time.strftime("%m-%d %H:%M", time.localtime(f["time"] / 1000)),
            })
        return recent
    except Exception as e:
        return f"获取成交记录失败：{e}"


def transfer_spot_to_perp(amount: float):
    """把 USDC 从 Spot 转到永续合约账户"""
    ex = _ensure_exchange()
    try:
        result = ex.usd_class_transfer(amount, to_perp=True)
        # SDK 可能在返回的 dict 中嵌了错误
        raw_str = str(result)
        if '"status":"err"' in raw_str or '"status": "err"' in raw_str or '"response":"Must deposit"' in raw_str:
            return f"转账失败：API 钱包模式不支持转账（如需转账请用直连私钥模式），raw={raw_str[:150]}"
        return {"status": "ok", "amount": amount, "raw": raw_str[:200]}
    except Exception as e:
        return f"转账失败：{e}"
