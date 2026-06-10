"""
Hyperliquid 链上连接器

封装 SDK，提供交易智能体需要的所有工具函数：
- 查行情、查深度
- 查持仓、查余额
- 开仓、平仓、设止损
"""

import json
import os
from decimal import Decimal

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account

# 读取钱包配置
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "config.json")


def _load_wallet():
    """
    从 data/config.json 读取钱包配置。
    支持两种模式：
      1. API 钱包模式（推荐）：
         { "main_address": "0x...", "api_key": "0x..." }
         main_address 从 Hyperliquid 网站上看
         api_key 从网站 /API 生成
      2. 直连模式：
         { "wallet_private_key": "0x..." }
    """
    if not os.path.exists(_CONFIG_PATH):
        return None, None, None
    with open(_CONFIG_PATH) as f:
        cfg = json.load(f)

    # 优先使用 API 钱包模式
    main_addr = cfg.get("main_address", "")
    api_key = cfg.get("api_key", "")
    if main_addr and api_key:
        return main_addr, api_key, "api"

    # 降级到直接私钥模式
    pk = cfg.get("wallet_private_key", "")
    if pk:
        return pk, pk, "direct"

    return None, None, None


# ========== 初始化连接 ==========

_info = None
_exchange = None
_wallet_address = None


def _ensure_info():
    global _info
    if _info is None:
        _info = Info(constants.MAINNET_API_URL, skip_ws=True)
    return _info


def _ensure_exchange():
    global _exchange, _wallet_address
    if _exchange is None:
        sign_key, addr_or_key, mode = _load_wallet()
        if not sign_key:
            raise Exception("请先在 data/config.json 中配置钱包")

        if mode == "api":
            # API 钱包模式：用主钱包地址查，用 API 私钥签名
            _wallet_address = addr_or_key  # main_address
            account = Account.from_key(sign_key)  # api_key
            _exchange = Exchange(account, constants.MAINNET_API_URL, account_address=_wallet_address)
        else:
            # 直连模式：私钥既当身份又当签名
            account = Account.from_key(sign_key)
            _wallet_address = account.address
            _exchange = Exchange(account, constants.MAINNET_API_URL)
    return _exchange


# ========== 查询工具 ==========

def get_market_overview():
    """
    获取所有币种的最新行情
    返回：[{"coin": "BTC", "mid": 50000, "volume": 1.2e9}, ...]
    """
    info = _ensure_info()
    try:
        mids = info.all_mids()
        # 获取成交量数据
        ctxs = info.asset_ctxs() if hasattr(info, "asset_ctxs") else []
        overview = []
        for coin, mid in mids.items():
            item = {"coin": coin, "mid": float(mid)}
            # 补充成交量（如果有）
            for ctx in ctxs:
                if ctx.get("coin") == coin:
                    item["volume"] = float(ctx.get("volume", 0))
                    item["funding"] = float(ctx.get("funding", 0))
                    break
            overview.append(item)
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
        book = info.l2_book(coin)
        bids = [[float(b[0]), float(b[1])] for b in book["bids"][:depth]]
        asks = [[float(a[0]), float(a[1])] for a in book["asks"][:depth]]
        return {"bids": bids, "asks": asks}
    except Exception as e:
        return f"获取深度失败：{e}"


def get_positions():
    """
    获取当前持仓
    返回：[{"coin": "BTC", "size": 0.1, "entry_px": 50000, "pnl": 50, ...}]
    """
    info = _ensure_info()
    try:
        state = info.user_state(_wallet_address)
        positions = []
        for pos in state.get("assetPositions", []):
            p = pos["position"]
            if float(p.get("szi", 0)) != 0:
                positions.append({
                    "coin": p["coin"],
                    "size": float(p["szi"]),
                    "entry_px": float(p.get("entryPx", 0)),
                    "pnl": float(p.get("unrealizedPnl", 0)),
                    "leverage": float(p.get("leverage", {}).get("value", 1)),
                    "liquidation_px": float(p.get("liquidationPx", 0)),
                })
        return positions
    except Exception as e:
        return f"获取持仓失败：{e}"


def get_balance():
    """
    获取账户余额
    返回：{"wallet_value": 1000, "available": 800, "margin_used": 200}
    """
    info = _ensure_info()
    try:
        state = info.user_state(_wallet_address)
        margin = state.get("marginSummary", {})
        cross = state.get("crossMarginSummary", {})
        return {
            "wallet_value": float(cross.get("accountValue", 0)),
            "margin_used": float(cross.get("totalMarginUsed", 0)),
            "available": float(cross.get("accountValue", 0)) - float(cross.get("totalMarginUsed", 0)),
        }
    except Exception as e:
        return f"获取余额失败：{e}"


# ========== 交易工具 ==========

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
        # 设置杠杆
        ex.update_leverage(leverage, coin, True)

        # 获取当前市价
        info = _ensure_info()
        mids = info.all_mids()
        current_price = float(mids.get(coin, 0))
        if current_price == 0:
            return f"无法获取 {coin} 价格"

        # 计算币数量
        coin_size = round(size / current_price, 4)

        is_buy = side == "long"

        # 市价单（IOC = Immediate-or-Cancel，市价吃单）
        result = ex.market_open(coin, is_buy, coin_size, current_price, slippage=0.01)

        trade_record = {
            "coin": coin,
            "side": side,
            "size": coin_size,
            "usd_size": size,
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
        pos = None
        for p in positions:
            if p["coin"] == coin:
                pos = p
                break
        if not pos:
            return f"没有 {coin} 的持仓"

        mids = info.all_mids()
        current_price = float(mids.get(coin, 0))

        close_sz = abs(pos["size"]) if size is None else size
        result = ex.market_close(coin, close_sz, current_price, slippage=0.01)

        # 计算盈亏
        pnl = (current_price - pos["entry_px"]) * close_sz
        if pos["size"] < 0:  # 空单
            pnl = -pnl

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
    import datetime
    trades = []
    if os.path.exists(_TRADES_FILE):
        with open(_TRADES_FILE) as f:
            trades = json.load(f)
    trade["open_time"] = datetime.datetime.now().isoformat()
    trade["id"] = len(trades) + 1
    trades.append(trade)
    with open(_TRADES_FILE, "w") as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)


def get_trade_history(days: int = 7):
    """
    获取最近交易记录
    days: 最近几天
    返回：交易列表，每条包含 open_time, close_time, coin, side, pnl
    """
    import datetime
    if not os.path.exists(_TRADES_FILE):
        return []
    with open(_TRADES_FILE) as f:
        trades = json.load(f)
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    recent = []
    for t in trades:
        t_time = t.get("open_time", t.get("close_time", ""))
        try:
            t_dt = datetime.datetime.fromisoformat(t_time)
            if t_dt >= cutoff:
                recent.append(t)
        except:
            recent.append(t)
    return recent
