# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Run one trade cycle (ReAct loop)
python -c "from trading_agent import agent; result = agent.run_trade_cycle(); print(result['reply'])"

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
```

## Architecture

**6 modules, ~700 lines total:**

| File | Lines | Role |
|---|---|---|
| `agent.py` | ~290 | ReAct loop brain — collects state, calls DeepSeek, dispatches tools |
| `hl_connector.py` | ~440 | Hyperliquid SDK wrapper — market data, orders, balance, trade journal |
| `memory.py` | ~250 | Vector memory — store/search lessons via fastembed (BAAI/bge-small-zh-v1.5) |
| `main.py` | ~305 | Entry point — manual REPL or auto-mode with background input thread |
| `config.py` | ~47 | API keys, position limits, leverage cap, allowed coins |
| `auto_runner.py` | ~53 | Headless auto loop — no input thread, runs until killed |
| `debug_react.py` | ~83 | Debug tool — prints every ReAct round's thought + tool calls |

### Data Flow

```
trigger (manual / --auto / auto_runner.py) → agent.run_trade_cycle()
  ├── collect: hl_connector.get_balance() + get_positions() + get_market_overview()
  ├── recall:  memory.recall(state_text) → vector (or keyword) search for relevant lessons
  ├── ReAct loop (max 3 rounds):
  │     └── DeepSeek JSON response with thought + tool_calls
  │         → _run_tool(name, args) dispatches to hl_connector / memory
  │         → results fed back for next round (errors listed as warnings)
  ├── lesson_to_save → memory.add_lesson()
  └── return {thought, reply, need_review, tool_results}
```

### ReAct Loop Structure

1. **Round 0**: State collected + lessons recalled → sent to DeepSeek
2. **Round 1-N**: Tool calls executed → results fed back (sorted: trade ops first, 8000 char limit)
3. **Final**: Empty `tool_calls` = final decision. Errors trigger retry (`had_errors` warning appended)

Key: `response_format={"type": "json_object"}` ensures DeepSeek always returns valid JSON. `_extract_json()` strips markdown fences as a safety layer.

### Wallet Modes (data/config.json)

- **API wallet mode** (recommended): `{main_address, api_key}` — can trade Perp, can't withdraw
- **Direct key mode**: `{wallet_private_key}` — full access, can transfer Spot→Perp
- Priority: API mode first, then direct, then error

### Config Loading Priority

1. Environment variable `DEEPSEEK_API_KEY`
2. `trading_agent/data/config.json` (`deepseek_api_key` + wallet fields)
3. Error if neither found

### Safety Limits (config.py)

| Constant | Default | Enforced in |
|---|---|---|
| `MAX_POSITION_SIZE_PCT` | 80 | `hl_connector.open_trade()` — rejects exceeding total notional |
| `MAX_LEVERAGE` | 3 | `hl_connector.open_trade()` — rejects before update_leverage |
| `DAILY_LOSS_LIMIT_PCT` | -5 | `agent.run_trade_cycle()` — code-enforced, `<= 0` trigger |
| `ALLOWED_COINS` | [] | `agent._run_tool()` — empty = all coins allowed |

### Tools Available to AI

| Tool | Parameters |
|---|---|
| `get_market_overview` | — |
| `get_orderbook` | `coin`, `depth` (default 5) |
| `get_candles` | `coin`, `interval` (1m/5m/15m/1h/4h/1d), `hours` (clamped [1, 720]) |
| `get_positions` | — |
| `get_balance` | — |
| `open_trade` | `coin`, `side` (long/short), `size` (USD >= 10), `leverage` (1-3) |
| `close_trade` | `coin` |
| `recall_lessons` | `context` (market description), `top_k` (default 3) |
| `add_lesson` | `lesson` (text), `tags` (list), `outcome` (string) |

## Key Patterns

### Return Type Convention

Functions return `dict/list` on success, `str` on error. Callers check `isinstance()` before dict access:

```python
result = hl_connector.get_balance()
if isinstance(result, dict):
    value = result.get("wallet_value", 0)
```

### None-Safe Parameter Extraction (`_run_tool`)

`args.get("key") or default` handles both missing key AND `null` value from AI JSON:

```python
int(args.get("leverage") or 1)     # handles null, 0 → 1
int(args.get("hours") or 24)       # handles null, 0 → 24
args.get("context") or ""          # handles null → ""
```

Without `or`, `dict.get("key", default)` returns `None` when key exists with `null`.

### Dedup Protection (`_executed_ops`)

Module-level `set` reset per cycle. Tracks `open_trade:{coin}` / `close_trade:{coin}` on SUCCESS only (so AI can retry failed operations). Prevents same-cycle duplicate trades.

### Position Size Check (open_trade)

Calculates existing total notional (sum of ALL positions, not filtering current coin — fixed bug). Compares against `wallet * MAX_POSITION_SIZE_PCT / 100`. Adds to `existing_notional` before checking, so same-coin stacking is correctly limited.

### Memory Recall Hierarchy

1. **Vector search** — if any lessons have embeddings and model loads: `cosine_similarity` on `bge-small-zh-v1.5` vectors
2. **Keyword fallback** — if vector search unavailable (model failed, no embeddings): tag match + keyword overlap + CJK character overlap
3. **Unembedded supplement** — when vector results < top_k, fills remaining slots from unembedded lessons via keyword scoring

`used_count` is incremented on ALL recall paths (fixed bug — was missing in keyword fallback).

### Error Feedback in ReAct

When tools fail, `had_errors` list + `⚠️` warning is appended to the next AI prompt. Trade ops (`open_trade`, `close_trade`, `transfer_spot_to_perp`, `add_lesson`) are sorted FIRST in the 8000-char result string to avoid truncation cutting critical data.

### Cache Patterns

| Cache | What | Type |
|---|---|---|
| `_info` | Hyperliquid SDK Info | Global, lazy |
| `_exchange` | Hyperliquid SDK Exchange | Global, lazy |
| `_embedding_model` | fastembed TextEmbedding | Global, lazy, caches failure as `False` |
| `_sz_decimals_cache` | Coin → szDecimals | Dict, per-coin |
| `_wallet_address` | Derived address | Global, lazy |

### File Paths

| File | Path (relative to project root) | Git |
|---|---|---|
| Wallet + API key config | `trading_agent/data/config.json` | ignored |
| Lessons (vector + metadata) | `data/lessons.json` | ignored |
| Local trades journal | `data/trades.json` | ignored |

Lessons and trades are in `data/` (project root, via `os.path.dirname(os.path.dirname(__file__))`). Wallet config is in `trading_agent/data/` (via one dirname).
