# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Run one trade cycle (ReAct loop) — most common action
python trading_agent/risk_manager.py

# Debug mode — show AI's per-round thinking
python trading_agent/debug_react.py

# Manual interactive mode
python -m trading_agent.main

# Auto mode (runs every 5 min)
python -m trading_agent.main --auto

# Headless auto mode (no keyboard input, pure loop)
python trading_agent/auto_runner.py

# List all saved lessons
python -c "from trading_agent.memory import get_all_lessons; print(get_all_lessons())"

# Review recent trades
python -c "from trading_agent import agent; agent.run_review()"

# Quick state check (balance + positions)
python -c "from trading_agent import hl_connector; print(hl_connector.get_balance()); print(hl_connector.get_positions())"
```

## Architecture

**8 modules, ~3000 lines total:**

| File | Lines | Role |
|---|---|---|
| `agent.py` | ~780 | ReAct loop brain — collects state, calls DeepSeek, dispatches tools |
| `hl_connector.py` | ~1170 | Hyperliquid SDK wrapper — market data, orders, balance, trade journal |
| `memory.py` | ~325 | Vector memory — store/search lessons via fastembed (BAAI/bge-small-zh-v1.5) |
| `main.py` | ~330 | Entry point — manual REPL or auto-mode with background input thread |
| `risk_manager.py` | ~195 | Cron-based risk monitor — hard SL, state verification, funding flip detection, triggers trade cycles |
| `config.py` | ~60 | API keys, position limits, leverage cap, allowed coins |
| `auto_runner.py` | ~55 | Headless auto loop — no input thread, runs until killed |
| `debug_react.py` | ~85 | Debug tool — prints every ReAct round's thought + tool calls |

### Data Flow

```
cron (every 10 min) → risk_manager.py
  ├── hard -5% SL check (disaster backup, wider than AI's SL)
  ├── state verification: orphan orders, dust, duplicates, funding flip detection
  └── if available > $10 → agent.run_trade_cycle()
        ├── collect: balance + positions + market_overview + funding_rates
        ├── recall: memory.recall(state_text) → vector/keyword lesson search
        ├── ReAct loop (max 3 rounds):
        │     └── DeepSeek JSON response with thought + tool_calls
        │         → _run_tool(name, args) dispatches to hl_connector / memory
        │         → results fed back (trade ops sorted first, 8000 char limit)
        │         → errors trigger retry via `had_errors` warnings
        ├── lesson_to_save → memory.add_lesson()
        └── return {thought, reply, need_review, tool_results}
```

### ReAct Loop Structure

1. **Round 0**: State collected + lessons recalled → sent to DeepSeek
2. **Round 1-N**: Tool calls executed sequentially → results fed back for next round
3. **Final**: Empty `tool_calls` = final decision. Errors trigger retry

Key: `response_format={"type": "json_object"}` ensures DeepSeek always returns valid JSON. `_extract_json()` strips markdown fences as safety layer.

### Wallet Modes (trading_agent/data/config.json)

- **API wallet mode** (recommended): `{main_address, api_key}` — can trade Perp, can't withdraw, can't transfer Spot→Perp
- **Direct key mode**: `{wallet_private_key}` — full access, can transfer Spot→Perp
- Priority: API mode first, then direct, then error

### Config Loading Priority

1. Environment variable `DEEPSEEK_API_KEY`
2. `trading_agent/data/config.json` (`deepseek_api_key` + wallet fields)
3. Error if neither found

DeepSeek model defaults to `deepseek-chat` via `https://api.deepseek.com`. Override with `DEEPSEEK_MODEL` and `DEEPSEEK_BASE_URL` env vars.

### Safety Limits (config.py)

| Constant | Default | Enforced in |
|---|---|---|
| `MAX_POSITION_SIZE_PCT` | 500 | `hl_connector.open_trade()` — total notional ≤ wallet × this % |
| `MAX_LEVERAGE` | 3 | `hl_connector.open_trade()` — rejects before update_leverage |
| `DAILY_LOSS_LIMIT_PCT` | -5 | `agent.run_trade_cycle()` — stops auto-trading if hit |
| `ALLOWED_COINS` | [] | `agent._run_tool()` — empty = all coins allowed |

### Tools Available to AI (agent.py _TOOL_DESCRIPTIONS)

| Tool | Parameters |
|---|---|
| `get_market_overview` | — |
| `get_orderbook` | `coin`, `depth` (default 5) |
| `get_candles` | `coin`, `interval` (1m/5m/15m/1h/4h/1d), `hours` (clamped [1, 720]) |
| `get_positions` | — |
| `get_balance` | — |
| `open_trade` | `coin`, `side` (long/short), `size` (USD ≥ 10), `leverage` (1-3) |
| `close_trade` | `coin`, `size` (optional, default = full) |
| `set_stop_loss` | `coin`, `stop_price`, `size` (optional, default = full position) |
| `set_take_profit` | `coin`, `target_price`, `size` (optional, default = full position) |
| `get_open_orders` | `coin` (optional) |
| `cancel_order` | `coin`, `oid` |
| `get_funding_rates` | `top_n` (default 20) |
| `get_funding_history` | `coin`, `hours` (default 48) |
| `recall_lessons` | `context` (market description), `top_k` (default 3) |
| `add_lesson` | `lesson` (text), `tags` (list), `outcome` (string) |
| `transfer_spot_to_perp` | `amount` (0 = all, API mode unsupported) |

### Auto-TP After Open Trade

After successful `open_trade`, agent.py automatically calls `set_take_profit` at ±3% from entry (TP = entry × 1.03 for long, entry × 0.97 for short). Includes `reset_user_state_cache()` before the call to avoid stale position data. Logged as `[Agent] 自动止盈: [OK/FAIL/SKIP]`.

### Funding Rate Direction Enforcement

Code-level check in `_run_tool` before open_trade:
- Fetches funding rate for target coin (from pre-fetched list of 50 or single-coin fallback)
- Rejects trade if `side ≠ direction`: positive funding → must short, negative funding → must long
- Error message includes funding % and violated rule

### Volume Filter & Partial Fill Detection

In `hl_connector.open_trade()`:
- Rejects coins with daily volume < $500K (liquidity risk)
- After IOC market order, checks actual filled size vs requested:
  - `resting` response: extracts `filled.totalSz`, rejects if < 50% of requested
  - Non-resting response: also extracts `filled.totalSz` from statuses[0], rejects if < 50%
- On rejection, returns error string with fill % details, triggers cache refresh in ReAct loop

### Cache Refresh After Trade Ops

In `agent.py` ReAct loop (lines 626-645), after ANY tool call:
- If status is "已开仓" / "已平仓" / "部分平仓" → `reset_user_state_cache()` + re-fetch balance/positions
- If `open_trade` returned error string (partial fill rejection) → same refresh (partial fill modified exchange state)
- Updated `balance`/`positions` variables used for subsequent tools in same round

## Key Patterns

### Return Type Convention

Functions return `dict/list` on success, `str` on error. Callers check `isinstance()` before dict access:

```python
result = hl_connector.get_balance()
if isinstance(result, dict):
    value = result.get("wallet_value", 0)
```

### None-Safe Parameter Extraction (_run_tool)

`args.get("key") or default` handles both missing key AND null value from AI JSON:

```python
int(args.get("leverage") or 1)     # handles null, 0 → 1
args.get("context") or ""          # handles null → ""
```

Without `or`, `dict.get("key", default)` returns `None` when key exists with null.

### Dedup Protection (_executed_ops)

Module-level `set` reset per cycle. Tracks `open_trade:{coin}` / `close_trade:{coin}` on SUCCESS only (so AI can retry failed operations). Prevents same-cycle duplicate trades.

### Position Size Check (open_trade)

Calculates existing total notional (ALL positions sum). Compares against `wallet * MAX_POSITION_SIZE_PCT / 100`. Adds new position notional before checking, so same-coin stacking is correctly limited. Also checks `required_margin = size / leverage ≤ available + 0.01`.

### Error Feedback in ReAct

When tools fail, `had_errors` list with ⚠️ prefix is appended to next AI prompt. Trade ops are sorted FIRST in the 8000-char result string (via `_truncate_results`) to avoid truncation cutting critical data.

### Cache Patterns

| Cache | What | Type |
|---|---|---|
| `_info` | Hyperliquid SDK Info | Global, lazy |
| `_exchange` | Hyperliquid SDK Exchange | Global, lazy |
| `_user_state_cache` | `info.user_state(addr)` | Global, per-cycle, reset on trade ops |
| `_mids_cache` | `info.all_mids()` | Per-cycle, reset at cycle start |
| `_embedding_model` | fastembed TextEmbedding | Global, lazy, caches failure as `False` |
| `_sz_decimals_cache` | Coin → szDecimals | Global, pre-filled by get_market_overview |
| `_wallet_address` | Derived address | Global, lazy |

### Memory Recall Hierarchy

1. **Vector search** — cosine similarity on `bge-small-zh-v1.5` embeddings
2. **Keyword fallback** — tag match + keyword overlap + CJK character overlap
3. **Unembedded supplement** — fills remaining slots from unembedded lessons via keyword scoring

`used_count` is incremented on ALL recall paths.

### File Paths

| File | Path (relative to project root) | Tracked in Git |
|---|---|---|
| Wallet + API key config | `trading_agent/data/config.json` | ignored |
| Lessons (vector + metadata) | `data/lessons.json` | ignored |
| Local trades journal | `data/trades.json` | ignored |
| Scheduled tasks | `.claude/scheduled_tasks.json` | ignored |

Lessons and trades are in `data/` (project root, via `os.path.dirname(os.path.dirname(__file__))`). Wallet config is in `trading_agent/data/` (via one dirname).

### GitHub

Remote: `https://github.com/tianyu0103/Claudos.git`
Branch: `master`
Commits are in Chinese (e.g., "修复代码审查发现的两个问题").
