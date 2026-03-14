from collections import deque
import numpy as np
import csv
from dataclasses import dataclass, field
import utils
import asyncio
import itertools
import time
import traceback
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import db
import math



MAX_LEN = 500
COINT_WINDOW = 200

_BAD_SYMBOL_PATTERNS = (
    'UPUSDT', 'DOWNUSDT', 'BEAR', 'BULL',
    'DAI', 'TUSD', 'USDP', 'FDUSD', 'USDC',
    'EURUSDT', 'GBPUSDT',
    '3L', '3S', '2L', '2S',
    'LEVERAGE',
)

# Canonical close reason mapping (used in pairs_trading and main.py)
CLOSE_REASONS = {
    'z_tp': '\U0001F4B0 Z-Score Take Profit',
    'z_sl': '\U0001F6D1 Z-Score Stop Loss',
    'circuit': '\U0001F534 Circuit Breaker',
    'broken_coint': '\U0001F6A8 Broken Correlation',
    'hardware_sl': '\U0001F6E1\ufe0f Hardware SL',
    'hardware_tp': '\U0001F6E1\ufe0f Hardware TP',
    'manual': '\U0001F464 Manual Close',
    'desync': '\u26a0\ufe0f Leg Desync',
    'beta_drift': '\U0001F4C9 Beta Drift',
    'beta_critical': '\U0001F6A8 Beta Critical',
    'btc_shock': '\U0001F4A5 BTC Market Shock',
    'external': '\u26a1 External Close',
    'orphan_restart': '\U0001F504 Orphan on Restart',
    'stale_symbols': '\u23f3 Stale Symbols',
    'manual_partial': '\U0001F464 Manual Close (1 leg)',
    'audit_fail': '\U0001F9FE Trade Audit Safety Close',
    'time_exit': '\u23f1\ufe0f Time Exit',
}


def _is_tradeable_usdt_symbol(symbol: str) -> bool:
    """Single source of symbol eligibility for warmup/discovery/runtime subscriptions."""
    if not isinstance(symbol, str):
        return False
    s = symbol.strip().upper()
    if not s or not s.endswith('USDT'):
        return False
    if not s.isascii() or '_' in s or not s.isalnum():
        return False
    if any(pattern in s for pattern in _BAD_SYMBOL_PATTERNS):
        return False
    return True


def _repair_mojibake_text(text: str) -> str:
    """Best-effort repair for UTF-8 text decoded as latin1/cp1252."""
    if not isinstance(text, str) or not text:
        return text
    if not any(ch in text for ch in ("\u00f0", "\u00e2", "\u00ce", "\u00c3", "\u00c2")):
        return text

    def _score(s: str) -> int:
        bad = sum(s.count(x) for x in ("\u00f0", "\u00e2", "\u00ce", "\u00c3", "\u00c2", "\u00ef\u00b8", "\u00e2\u2020"))
        return -bad

    candidates = [text]
    for enc in ("latin-1", "cp1252"):
        cur = text
        for _ in range(2):
            try:
                nxt = cur.encode(enc, errors="strict").decode("utf-8", errors="strict")
            except Exception:
                break
            if nxt == cur:
                break
            candidates.append(nxt)
            cur = nxt

    best = max(candidates, key=_score)
    return best


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ('true', '1', 'yes', 'on')


def _canonical_pair_key(symbol1: str, symbol2: str) -> tuple[str, str]:
    a = str(symbol1 or '').strip().upper()
    b = str(symbol2 or '').strip().upper()
    return tuple(sorted((a, b)))


def _extract_pair_entries(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        pairs = raw.get('pairs')
        if isinstance(pairs, list):
            return pairs
    return []


def _parse_pair_text(entry):
    pair_str = ''
    if isinstance(entry, str):
        pair_str = entry.strip().upper()
    elif isinstance(entry, dict):
        pair_str = str(entry.get('pair', '') or '').strip().upper()
        if not pair_str:
            s1 = str(entry.get('symbol1', '') or '').strip().upper()
            s2 = str(entry.get('symbol2', '') or '').strip().upper()
            if s1 and s2:
                pair_str = f"{s1}-{s2}"

    if '-' not in pair_str:
        return None
    s1, s2 = [x.strip().upper() for x in pair_str.split('-', 1)]
    if not s1 or not s2 or s1 == s2:
        return None
    if not _is_tradeable_usdt_symbol(s1) or not _is_tradeable_usdt_symbol(s2):
        return None
    return (s1, s2)

class Data:
    """
    Stores time series data in a deque for each symbol.
    """
    def __init__(self, maxlen=500):
        self.ts = deque(maxlen=maxlen)
        self.open = deque(maxlen=maxlen)
        self.high = deque(maxlen=maxlen)
        self.low = deque(maxlen=maxlen)
        self.close = deque(maxlen=maxlen)

    def add_kline(self, ts, open_p, high_p, low_p, close_p):
        """
        Adds a new kline if it's not a duplicate.
        """
        ts = int(ts)
        if not self.ts or ts > self.ts[-1]:
            self.ts.append(ts)
            self.open.append(float(open_p))
            self.high.append(float(high_p))
            self.low.append(float(low_p))
            self.close.append(float(close_p))
            return True
        return False

@dataclass
class PairInfo:
    """
    Stores information about a cointegrated pair.
    """
    symbol1: str
    symbol2: str
    hedge_ratio: float = 0.0
    half_life: float = 0.0
    last_z_score: float = 0.0
    position_status: int = 0  # 0: no position, 1: long spread, -1: short spread
    qty1: float = 0.0
    qty2: float = 0.0
    entry_price1: float = 0.0
    entry_price2: float = 0.0
    db_id: int = None 
    current_trade_id: int = None 
    is_trading: bool = False
    # TG message tracking
    tg_message_id: int = 0     # TG message ID for reply threading
    open_time: int = 0         # Unix timestamp when trade opened
    entry_z_score: float = 0.0 # Z-score at trade entry
    entry_half_life_bars: float = 0.0  # Half-life frozen at entry for time-exit parity with backtest
    entry_hold_limit_bars: int = 0     # Hold limit frozen at entry for restart-safe time_exit parity
    # Market neutrality
    beta_btc: float = 0.0      # Beta to BTC (should be near 0 for market-neutral)
    last_pvalue: float = 0.0   # Last p-value from cointegration test
    # Signal confirmation (for real-time mode)
    pending_signal: float = None  # Pending Z-score signal awaiting confirmation
    pending_since: float = None   # Time when signal started
    pending_source: str = ''      # 'realtime' | 'candle' (for confirmation fallback logic)
    # Cached quality score (updated on candle-close metrics refresh)
    quality_score: float = 0.0
    quality_updated_at: float = 0.0
    # Entry diagnostics
    entry_expected_hours: float = 0.0
    entry_coint_streak_bars: int = 0
    # Cointegration persistence
    coint_streak_bars: int = 0
    coint_broken_count: int = 0
    # Idle pair management
    discovered_at: float = field(default_factory=time.time)  # When pair was discovered
    # Close tracking - prevents duplicate notifications
    close_handled: bool = False    # True if bot already processed close notification
    last_close_reason: str = ''    # Reason for last close (for debugging)
    # Cooldown after stop-loss to prevent immediate re-entry
    _close_cooldown_until: float = 0.0  # Unix timestamp: skip entry signals until this time
    # Wait-for-candle: after ANY close, block re-entry until next candle closes
    _wait_for_candle: bool = False  # True = pair just closed, wait for next candle before re-entry
    # Persisted/derived candle anchor: do not re-enter while latest closed candle <= this ts (ms).
    reentry_block_candle_ts: int = 0

class PairsManager:
    """
    Manages symbol data, finds cointegrated pairs, and generates signals.
    Supports Multi-Timeframe (MTF) mode: main TF for discovery, entry TF for faster signals.
    """
    def __init__(self, client, loop, all_symbols, timeframe='1h', min_data_points=200, notify_callback=None, config_info=None):
        self.client = client
        self.loop = loop
        self.all_symbols = all_symbols
        self.timeframe = timeframe
        self.min_data_points = min_data_points
        self.notify_callback = notify_callback
        self.config = config_info
        
        self.max_len = int(min_data_points * 2.5)
        
        # Main TF data (for discovery + validation)
        self.all_data: dict[str, Data] = {}
        
        self.active_pairs: dict[frozenset, PairInfo] = {}
        # Pair-local same-candle re-entry guard that survives pair object rotation.
        self._reentry_block_by_pair: dict[frozenset, int] = {}
        # O(1) symbol → list[PairInfo] index (maintained by _register_pair/_unregister_pair)
        self._symbol_to_pairs: dict[str, list[PairInfo]] = {}
        self.leverage_cache = {} # {symbol: leverage_int}
        self._discovery_task = None
        self._last_discovery_time = 0
        self._cleanup_task = None  # Periodic orphaned orders cleanup
        self._last_cleanup_time = 0
        
        # CRITICAL: Lock to prevent race condition when opening trades
        self._trade_lock = asyncio.Lock()
        
        # CPU Pool for heavy computations (bounded workers to avoid weak-laptop thrashing)
        # Windows + Python 3.14 may produce unstable SpawnProcess behavior for heavy churn.
        # Keep default conservative there; allow manual override via env/config.
        cpu_count = os.cpu_count() or 2
        if os.name == 'nt' and sys.version_info >= (3, 14):
            default_workers = 2
        else:
            default_workers = max(1, min(4, cpu_count - 1))
        configured_workers = getattr(self.config, 'discovery_workers', None) if self.config else None
        env_workers = os.getenv('DISCOVERY_WORKERS')
        try:
            if env_workers:
                worker_count = int(env_workers)
            else:
                worker_count = int(configured_workers) if configured_workers else default_workers
        except Exception:
            worker_count = default_workers
        self.executor = ProcessPoolExecutor(max_workers=max(1, worker_count))
        
        # Real-time Z-score monitoring
        self.last_prices: dict[str, float] = {}  # {symbol: last_price} from WebSocket ticker
        self._signal_confirmation_task = None    # Task for checking confirmed signals
        
        # Dynamic markPrice subscription for newly discovered pairs
        self._subscribed_mark_symbols: set = set()  # Symbols with active markPrice subscription
        self._subscribe_mark_callback = None  # Callback to subscribe new symbols (set by main.py)
        
        # NOTE: Initialization is now EXPLICIT - call await pairs_manager.initialize() in main.py
        self._initialized = False
        
        # Algo order tracking: algoId -> {pair_key, symbol, order_type}
        self.algo_orders: dict[str, dict] = {}
        
        # Grace period: prevent broken_coint closures right after init (data may not be warm yet)
        self._init_complete_time = 0  # Set after initialize() finishes
        self._broken_coint_grace_sec = 120  # 2 minutes grace period
        
        # BTC Market Shock Protector: track BTC price over rolling window
        self._btc_price_history: deque = deque(maxlen=300)  # (timestamp, price) — ~5 min at 1s ticks
        self._btc_shock_triggered = False  # Prevent duplicate closures during same shock event
        self._btc_shock_cooldown = 0       # Timestamp until shock protection resets
        
        # Cached exchange position count (updated periodically and inside trade lock)
        self._exchange_position_count = 0
        self._exchange_positions_cache: dict[str, float] = {}  # {symbol: qty}
        
        # Cached unrealized PnL from exchange (updated every 15s from get_position_risk)
        # Source of truth for all PnL decisions — NO manual calculations
        self._exchange_pnl_cache: dict[str, float] = {}  # {symbol: unrealizedProfit}
        
        # Symbol-level cooldown after margin/capital/order-limit failures
        self._symbol_block_until: dict[str, float] = {}
        
        # Background warmup/discovery task (quick startup mode)
        self._warmup_task = None
        # Prevent recursive/parallel reconcile loops.
        self._reconcile_lock = asyncio.Lock()
        # best_pairs v2 refresh controls
        self._best_pairs_refresh_lock = asyncio.Lock()
        self._best_pairs_last_refresh = 0.0
        self._priority_pairs_cache_path = ''
        self._priority_pairs_cache_mtime = None
        self._priority_pairs_cache_entries: list[tuple[str, str]] = []
        self._priority_pairs_cache_keys: set[tuple[str, str]] = set()
        self._pair_blacklist_cache_path = ''
        self._pair_blacklist_cache_mtime = None
        self._pair_blacklist_cache_keys: set[tuple[str, str]] = set()
        # Main-TF progress heartbeat (to prove bot is processing closed candles).
        self._progress_last_log_ts = 0.0
        self._progress_kline_added = 0
        self._progress_analysis_runs = 0
        self._progress_coint_evals = 0
        self._progress_last_symbol = ''
        self._progress_last_kline_open_ts = 0
        self._progress_last_pair = ''
        self._progress_last_pair_ts = 0
        # Discovery sharding cursor (for weak CPU: spread full universe over cycles).
        self._discovery_round_idx = 0
        # Discovery health/watchdog state.
        self._health_task = None
        self._last_pair_found_ts = time.time()
        self._stagnation_last_full_scan_ts = 0.0
        self._diag_last_report_ts = time.time()
        self._diag_discovery_runs = 0
        self._diag_discovery_new_pairs = 0
        self._diag_reject_reason_counts: dict[str, int] = {}
        # Pair-level anti-repeat reject cooldown:
        # {frozenset([s1,s2]): {'reason': str, 'count': int, 'updated_at': ts, 'blocked_until': ts}}
        self._pair_reject_state: dict[frozenset, dict] = {}

    async def initialize(self):
        """
        MUST be called after creation and awaited before any trading.
        Loads state from DB and reconciles with exchange.
        """
        if self._initialized:
            return
        
        print("🔄 Initializing PairsManager...")
        
        # Load from DB first (also runs _reconcile_with_exchange internally)
        await self._load_state_from_db()
        await self._load_reentry_blocks_from_db()
        
        # Update exchange position cache
        await self._refresh_exchange_position_count()
        
        self._initialized = True
        self._init_complete_time = time.time()
        max_pairs = getattr(self.config, 'max_active_pairs', 5) or 5
        print(f"✅ PairsManager initialized. Max active pairs: {max_pairs}")
        
        # Start periodic leg sync loop (BACKUP only - primary sync is via WebSocket)
        self._leg_sync_task = self.loop.create_task(self._periodic_leg_sync_loop())
        print("🔄 Started backup leg sync loop (every 30s, primary via WebSocket)")
        # Do not rebuild best_pairs at startup: keep curated full positive history intact.
        # Runtime refresh remains available (additive) after discovery/TP closes.

    async def _load_reentry_blocks_from_db(self):
        """Restore same-candle guard anchors from DB for restart-safe behavior."""
        try:
            rows = await db.get_all_pairs(include_archived=True)
            restored = 0
            for p in rows:
                if getattr(p, 'position_status', 0) != 0:
                    continue
                sym1 = getattr(p, 'symbol1', '')
                sym2 = getattr(p, 'symbol2', '')
                if not sym1 or not sym2:
                    continue
                ts = int(getattr(p, 'last_close_candle_ts', 0) or 0)
                if ts <= 0:
                    close_time = int(getattr(p, 'close_time', 0) or 0)
                    ts = close_time if close_time > 1_000_000_000_000 else close_time * 1000
                if ts <= 0:
                    continue
                key = frozenset([sym1, sym2])
                prev = int(self._reentry_block_by_pair.get(key, 0) or 0)
                if ts > prev:
                    self._reentry_block_by_pair[key] = ts
                    restored += 1
            if restored:
                print(f"🔒 Restored same-candle re-entry guards: {restored}")
        except Exception as e:
            print(f"⚠️ Could not restore re-entry guards from DB: {e}")

    async def _load_state_from_db(self, run_reconcile: bool = True):
        """
        CRITICAL: Exchange is source of truth.
        1. Fetch actual positions from exchange FIRST
        2. Load from DB only pairs that exist on exchange
        3. Mark non-existent pairs as closed
        """
        print("🔄 Syncing state with exchange (source of truth)...")
        
        try:
            # STEP 1: Get REAL positions from exchange FIRST
            exchange_positions = await self.client.get_position_risk()
            open_on_exchange = {}
            for pos in exchange_positions:
                symbol = pos.get('symbol', '')
                qty = abs(float(pos.get('positionAmt', 0)))
                if qty > 0:
                    open_on_exchange[symbol] = {
                        'qty': qty,
                        'side': 'LONG' if float(pos.get('positionAmt', 0)) > 0 else 'SHORT',
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'unrealized_pnl': float(pos.get('unRealizedProfit', 0))
                    }
            
            print(f"  Exchange has {len(open_on_exchange)} open positions")
            
            # STEP 2: Load pairs from DB and validate against exchange
            pairs = await db.get_all_pairs()
            restored_count = 0
            restored_idle_count = 0
            closed_count = 0
            
            # Detailed logging for debugging
            open_pairs_in_db = [p for p in pairs if p.position_status != 0]
            print(f"  DB has {len(pairs)} total pairs, {len(open_pairs_in_db)} with open positions")
            for p in open_pairs_in_db:
                print(f"    📋 DB open: {p.symbol1}-{p.symbol2} status={p.position_status} db_id={p.id}")
            
            for p in pairs:
                pair_set = frozenset([p.symbol1, p.symbol2])
                
                # DUPLICATE CHECK: Skip if pair already loaded
                if pair_set in self.active_pairs:
                    if p.position_status != 0:
                        print(f"  ⚠️ Skipping duplicate WITH POSITION: {p.symbol1}-{p.symbol2} (db_id={p.id}, already in active_pairs)")
                    continue
                
                if p.position_status != 0:
                    # DB says position is open - verify against exchange
                    s1_open = p.symbol1 in open_on_exchange
                    s2_open = p.symbol2 in open_on_exchange
                    
                    if s1_open and s2_open:
                        # VALID: Both legs exist on exchange - restore
                        close_anchor = int(getattr(p, 'last_close_candle_ts', 0) or 0)
                        if close_anchor <= 0:
                            close_time = int(getattr(p, 'close_time', 0) or 0)
                            close_anchor = close_time if close_time > 1_000_000_000_000 else close_time * 1000
                        info = PairInfo(
                            symbol1=p.symbol1,
                            symbol2=p.symbol2,
                            hedge_ratio=p.hedge_ratio,
                            half_life=p.half_life,
                            entry_half_life_bars=float(getattr(p, 'entry_half_life_bars', 0.0) or 0.0),
                            entry_hold_limit_bars=int(getattr(p, 'entry_hold_limit_bars', 0) or 0),
                            position_status=p.position_status,
                            qty1=p.qty1,
                            qty2=p.qty2,
                            entry_price1=p.entry_price1,
                            entry_price2=p.entry_price2,
                            db_id=p.id,
                            open_time=int(getattr(p, 'open_time', 0) or 0),
                            tg_message_id=getattr(p, 'tg_message_id', 0) or 0,
                            reentry_block_candle_ts=close_anchor
                        )
                        # Restore market neutrality metrics from DB (survive restart)
                        info.beta_btc = getattr(p, 'beta_btc', 0.0) or 0.0
                        info.last_pvalue = getattr(p, 'last_pvalue', 0.0) or 0.0
                        info.entry_z_score = getattr(p, 'entry_z_score', 0.0) or 0.0
                        self._update_quality_score_cache(info)
                        
                        last_trade = await db.get_last_open_trade_for_pair(p.id)
                        if last_trade:
                            info.current_trade_id = last_trade.id
                        await self._ensure_entry_time_exit_state(info, persist=True)
                        
                        self.active_pairs[pair_set] = info
                        self._register_pair(info)
                        restored_count += 1
                        print(f"  ✅ Restored: {p.symbol1}-{p.symbol2} (β:{info.beta_btc:.3f}, p:{info.last_pvalue:.4f})")
                    
                    elif s1_open != s2_open:
                        # ORPHAN: One leg closed externally, need to close remaining leg
                        remaining_sym = p.symbol1 if s1_open else p.symbol2
                        closed_sym = p.symbol2 if s1_open else p.symbol1
                        remaining_pos = open_on_exchange[remaining_sym]
                        remaining_qty = remaining_pos['qty']
                        remaining_side = remaining_pos['side']
                        unrealized_pnl = remaining_pos['unrealized_pnl']
                        
                        tg_msg_id = getattr(p, 'tg_message_id', 0) or 0
                        
                        print(f"  🚨 ORPHAN: {p.symbol1}-{p.symbol2} | {closed_sym} closed externally")
                        print(f"      Remaining: {remaining_sym} ({remaining_side}) PnL: {unrealized_pnl:.2f}")
                        
                        # Get PnL from the already closed leg
                        import time as time_mod
                        now_ms = int(time_mod.time() * 1000)
                        start_ms = now_ms - 86400_000  # Last 24 hours
                        closed_leg_trades = await self.client.get_account_trades(symbol=closed_sym, startTime=start_ms, limit=100)
                        closed_leg_pnl = sum(float(t.get('realizedPnl', 0)) for t in closed_leg_trades)
                        
                        # Close orphan immediately (no wait, no buttons)
                        pnl_emoji = "🔴" if unrealized_pnl < 0 else "🟢"
                        
                        # Notify about orphan detection
                        await self._notify(
                            f"🚨 <b>ORPHAN PAIR DETECTED</b>\n\n"
                            f"Pair: {p.symbol1}-{p.symbol2}\n"
                            f"❌ Closed externally: {closed_sym}\n"
                            f"   └─ PnL: {closed_leg_pnl:+.2f} USDT\n\n"
                            f"⚠️ Closing: {remaining_sym} ({remaining_side})\n"
                            f"   └─ Unrealized PnL: {pnl_emoji} <b>{unrealized_pnl:.2f} USDT</b>",
                            reply_to_msg_id=tg_msg_id
                        )
                        # Close the remaining position
                        try:
                            # Re-verify position still exists before closing
                            verify_positions = await self.client.get_position_risk()
                            position_exists = False
                            for pos in verify_positions:
                                if pos.get('symbol') == remaining_sym and abs(float(pos.get('positionAmt', 0))) > 0:
                                    position_exists = True
                                    remaining_qty = abs(float(pos.get('positionAmt', 0)))
                                    remaining_side = 'LONG' if float(pos.get('positionAmt', 0)) > 0 else 'SHORT'
                                    break
                            
                            if not position_exists:
                                print(f"      → Position {remaining_sym} already closed, skipping")
                            else:
                                await self.client.cancel_open_orders(remaining_sym)
                                close_side = 'SELL' if remaining_side == 'LONG' else 'BUY'
                                await self._close_leg_reduce_only(
                                    symbol=remaining_sym,
                                    side=close_side,
                                    quantity=remaining_qty
                                )
                                print(f"      ✅ Closed orphan {remaining_sym}")
                            
                            import asyncio
                            await asyncio.sleep(1)
                            
                            # Fetch PnL for remaining leg
                            now_ms = int(time_mod.time() * 1000)
                            start_ms = now_ms - 300_000
                            remaining_trades = await self.client.get_account_trades(symbol=remaining_sym, startTime=start_ms, limit=50)
                            remaining_leg_pnl = sum(float(t.get('realizedPnl', 0)) for t in remaining_trades)
                            
                            total_pnl = closed_leg_pnl + remaining_leg_pnl
                            total_emoji = "🟢" if total_pnl >= 0 else "🔴"
                            
                            await self._notify(
                                f"⚡ <b>Orphan Pair Closed</b>\n\n"
                                f"Pair: {p.symbol1}-{p.symbol2}\n\n"
                                f"❌ {closed_sym}: {closed_leg_pnl:+.2f} USDT (closed externally)\n"
                                f"⚡ {remaining_sym}: {remaining_leg_pnl:+.2f} USDT (closed by bot)\n\n"
                                f"💰 <b>Total PnL: {total_emoji} {total_pnl:+.2f} USDT</b>",
                                reply_to_msg_id=tg_msg_id
                            )
                        except Exception as e:
                            print(f"      ⚠️ Failed to close orphan: {e}")
                            await self._notify(f"🚨 ORPHAN CLOSE FAILED: {remaining_sym}\nError: {e}")
                        
                        # Mark pair as closed in DB
                        await db.update_pair({
                            'id': p.id,
                            'position_status': 0,
                            'qty1': 0,
                            'qty2': 0,
                            'entry_price1': 0,
                            'entry_price2': 0
                        })
                        try:
                            open_trade = await db.get_last_open_trade_for_pair(p.id)
                            if open_trade:
                                await db.close_trade_record(
                                    open_trade.id,
                                    status='CLOSED_ORPHAN',
                                    close_reason='orphan_restart',
                                )
                        except Exception as trade_close_err:
                            print(f"  ⚠️ Could not close stale OPEN trade for orphan {p.symbol1}-{p.symbol2}: {trade_close_err}")
                        closed_count += 1
                    
                    else:
                        # Both legs closed on exchange - just mark as closed
                        print(f"  ⚠️ Stale: {p.symbol1}-{p.symbol2} (both closed on exchange)")
                        await db.update_pair({
                            'id': p.id,
                            'position_status': 0,
                            'qty1': 0,
                            'qty2': 0,
                            'entry_price1': 0,
                            'entry_price2': 0
                        })
                        try:
                            open_trade = await db.get_last_open_trade_for_pair(p.id)
                            if open_trade:
                                await db.close_trade_record(
                                    open_trade.id,
                                    status='CLOSED_EXTERNAL',
                                    close_reason='external',
                                )
                        except Exception as trade_close_err:
                            print(f"  ⚠️ Could not close stale OPEN trade for stale pair {p.symbol1}-{p.symbol2}: {trade_close_err}")
                        closed_count += 1
                else:
                    # DB idle pairs must also be restored into runtime state.
                    # Otherwise they remain non-archived in DB, block duplicate checks,
                    # but are never monitored after restart.
                    close_anchor = int(getattr(p, 'last_close_candle_ts', 0) or 0)
                    if close_anchor <= 0:
                        close_time = int(getattr(p, 'close_time', 0) or 0)
                        close_anchor = close_time if close_time > 1_000_000_000_000 else close_time * 1000

                    discovered_at = time.time()
                    if close_anchor > 0:
                        discovered_at = close_anchor / 1000.0
                    elif int(getattr(p, 'open_time', 0) or 0) > 0:
                        discovered_at = int(getattr(p, 'open_time', 0) or 0)

                    info = PairInfo(
                        symbol1=p.symbol1,
                        symbol2=p.symbol2,
                        hedge_ratio=p.hedge_ratio,
                        half_life=p.half_life,
                        entry_half_life_bars=float(getattr(p, 'entry_half_life_bars', 0.0) or 0.0),
                        entry_hold_limit_bars=int(getattr(p, 'entry_hold_limit_bars', 0) or 0),
                        position_status=0,
                        qty1=0.0,
                        qty2=0.0,
                        entry_price1=0.0,
                        entry_price2=0.0,
                        db_id=p.id,
                        open_time=0,
                        tg_message_id=getattr(p, 'tg_message_id', 0) or 0,
                        discovered_at=discovered_at,
                        reentry_block_candle_ts=close_anchor
                    )
                    info.beta_btc = getattr(p, 'beta_btc', 0.0) or 0.0
                    info.last_pvalue = getattr(p, 'last_pvalue', 0.0) or 0.0
                    info.entry_z_score = getattr(p, 'entry_z_score', 0.0) or 0.0
                    self._update_quality_score_cache(info)

                    self.active_pairs[pair_set] = info
                    self._register_pair(info)
                    restored_idle_count += 1
            
            print(
                f"  Restored {restored_count} open pairs, {restored_idle_count} idle pairs, "
                f"marked {closed_count} stale pairs as CLOSED"
            )
            
            # Continue with full reconciliation (orphan handling, unknown positions, etc.)
            # MUST be inside try block - if DB load failed, we must NOT reconcile with empty active_pairs
            if run_reconcile:
                await self._reconcile_with_exchange()
            
        except Exception as e:
            print(f"❌ Error loading state: {e}")
            import traceback
            traceback.print_exc()
            # Do NOT call _reconcile_with_exchange here - it would close all positions as "unknown"
            print("⚠️ SKIPPING reconciliation due to DB load error. Positions on exchange are safe.")

    async def _reconcile_with_exchange(self):
        """
        CRITICAL: Synchronize DB state with actual exchange positions.
        Exchange is the SINGLE SOURCE OF TRUTH.
        """
        try:
            if self._reconcile_lock.locked():
                print("⏭️ Reconcile already in progress, skipping re-entry.")
                return
            await self._reconcile_lock.acquire()
            print("🔄 Reconciling DB with exchange positions...")
            # Get all open positions from exchange
            exchange_positions = await self.client.get_position_risk()
            
            # Build set of symbols with actual open positions on exchange
            open_on_exchange = {}
            for pos in exchange_positions:
                symbol = pos.get('symbol', '')
                qty = abs(float(pos.get('positionAmt', 0)))
                if qty > 0:
                    open_on_exchange[symbol] = {
                        'qty': qty,
                        'side': 'LONG' if float(pos.get('positionAmt', 0)) > 0 else 'SHORT',
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'unrealized_pnl': float(pos.get('unRealizedProfit', 0))
                    }
            
            # Fetch all open orders to check for SL/TP protection
            all_open_orders = await self.client.get_orders()
            orders_by_symbol = {}
            for o in all_open_orders:
                sym = o['symbol']
                if sym not in orders_by_symbol: orders_by_symbol[sym] = []
                orders_by_symbol[sym].append(o)

            # Fetch algo orders ONCE for all pairs (not inside loop!)
            all_algo_orders = []
            try:
                algo_result = await self.client.get_algo_orders()
                if isinstance(algo_result, dict) and 'orders' in algo_result:
                    all_algo_orders = algo_result.get('orders', [])
                elif isinstance(algo_result, list):
                    all_algo_orders = algo_result
                print(f"  Fetched {len(all_algo_orders)} algo orders")
            except Exception as e:
                print(f"  ⚠️ Could not fetch algo orders: {e}")

            print(f"  Exchange has {len(open_on_exchange)} open positions, {len(all_open_orders)} regular orders, {len(all_algo_orders)} algo orders.")
            
            # Warn about positions on exchange that are NOT in our DB
            tracked_symbols = set()
            for pair_info in self.active_pairs.values():
                # Include ALL symbols from active_pairs (even orphans with position_status=0)
                # If pair is in active_pairs, its symbols should not be flagged as unknown
                tracked_symbols.add(pair_info.symbol1)
                tracked_symbols.add(pair_info.symbol2)
            
            unknown_positions = [s for s in open_on_exchange.keys() if s not in tracked_symbols]
            if unknown_positions:
                print(f"  🚨 UNKNOWN POSITIONS on exchange (not tracked by bot): {unknown_positions}")
                
                import asyncio
                
                # SAFETY: If active_pairs is empty but exchange has positions, 
                # attempt emergency recovery from DB before giving up.
                if len(self.active_pairs) == 0 and len(open_on_exchange) > 0:
                    print("⚠️ SAFETY: active_pairs is EMPTY but exchange has positions. Attempting emergency DB recovery...")
                    try:
                        await self._load_state_from_db(run_reconcile=False)
                    except Exception as e:
                        print(f"⚠️ Emergency DB load failed: {e}")
                    
                    if len(self.active_pairs) == 0:
                        warn_msg = (f"⚠️ <b>SAFETY BLOCK</b>: active_pairs is still EMPTY but exchange has "
                                    f"{len(open_on_exchange)} positions.\n\n"
                                    f"Bot will NOT auto-close these unknown positions to prevent data loss.\n"
                                    f"Please check DB integrity or close positions manually.")
                        print(warn_msg)
                        await self._notify(warn_msg)
                        # We don't return here anymore - we let the individual 'unknown' check below run
                        # to see if it can recover them one by one.
                
                # SAFETY: Check DB directly before closing - maybe the pair exists but wasn't loaded
                try:
                    all_db_pairs = await db.get_all_pairs()
                    db_pair_map = {}  # symbol → list of (pair, other_symbol)
                    for p in all_db_pairs:
                        if p.position_status != 0:
                            db_pair_map.setdefault(p.symbol1, []).append((p, p.symbol2))
                            db_pair_map.setdefault(p.symbol2, []).append((p, p.symbol1))
                except Exception as e:
                    print(f"  ⚠️ Could not query DB for safety check: {e}")
                    db_pair_map = {}
                
                recovered_pairs = []
                unknown_to_close = []
                unknown_already_closed = []
                unknown_closed = []
                unknown_failed = []

                for symbol in unknown_positions:
                    pos_data = open_on_exchange[symbol]
                    qty = pos_data['qty']
                    side = pos_data['side']
                    unrealized_pnl = pos_data['unrealized_pnl']
                    recovered_from_db = False
                    
                    print(f"      Unknown: {symbol} ({side} {qty}) PnL: {unrealized_pnl:.2f}")
                    
                    # CHECK DB: Does this symbol belong to a pair with open position?
                    if symbol in db_pair_map:
                        for db_pair, other_sym in db_pair_map[symbol]:
                            other_on_exchange = other_sym in open_on_exchange
                            if other_on_exchange:
                                # BOTH legs are on exchange AND pair exists in DB with position_status != 0
                                # This is NOT an unknown position - RESTORE it!
                                pair_set = frozenset([symbol, other_sym])
                                if pair_set not in self.active_pairs:
                                    close_anchor = int(getattr(db_pair, 'last_close_candle_ts', 0) or 0)
                                    if close_anchor <= 0:
                                        close_time = int(getattr(db_pair, 'close_time', 0) or 0)
                                        close_anchor = close_time if close_time > 1_000_000_000_000 else close_time * 1000
                                    info = PairInfo(
                                        symbol1=db_pair.symbol1,
                                        symbol2=db_pair.symbol2,
                                        hedge_ratio=db_pair.hedge_ratio,
                                        half_life=db_pair.half_life,
                                        entry_half_life_bars=float(getattr(db_pair, 'entry_half_life_bars', 0.0) or 0.0),
                                        entry_hold_limit_bars=int(getattr(db_pair, 'entry_hold_limit_bars', 0) or 0),
                                        position_status=db_pair.position_status,
                                        qty1=db_pair.qty1,
                                        qty2=db_pair.qty2,
                                        entry_price1=db_pair.entry_price1,
                                        entry_price2=db_pair.entry_price2,
                                        db_id=db_pair.id,
                                        open_time=int(getattr(db_pair, 'open_time', 0) or 0),
                                        tg_message_id=getattr(db_pair, 'tg_message_id', 0) or 0,
                                        reentry_block_candle_ts=close_anchor
                                    )
                                    info.beta_btc = getattr(db_pair, 'beta_btc', 0.0) or 0.0
                                    info.last_pvalue = getattr(db_pair, 'last_pvalue', 0.0) or 0.0
                                    info.entry_z_score = getattr(db_pair, 'entry_z_score', 0.0) or 0.0
                                    self._update_quality_score_cache(info)
                                    await self._ensure_entry_time_exit_state(info, persist=True)
                                    self.active_pairs[pair_set] = info
                                    self._register_pair(info)
                                    tracked_symbols.add(db_pair.symbol1)
                                    tracked_symbols.add(db_pair.symbol2)
                                    print(f"      🔄 RECOVERED from DB: {db_pair.symbol1}-{db_pair.symbol2} (was missed during load)")
                                    recovered_pairs.append(f"{db_pair.symbol1}-{db_pair.symbol2}")
                                recovered_from_db = True
                                break
                    if recovered_from_db:
                        continue  # Skip closing - symbol is no longer unknown
                    
                    # Only close if truly unknown (not found in DB either)
                    unknown_to_close.append(f"{symbol} ({side} {qty}) upnl={unrealized_pnl:+.2f}")
                    
                    try:
                        # Re-verify position still exists before closing
                        verify_positions = await self.client.get_position_risk()
                        position_exists = False
                        for p in verify_positions:
                            if p.get('symbol') == symbol and abs(float(p.get('positionAmt', 0))) > 0:
                                position_exists = True
                                qty = abs(float(p.get('positionAmt', 0)))
                                side = 'LONG' if float(p.get('positionAmt', 0)) > 0 else 'SHORT'
                                break
                        
                        if not position_exists:
                            print(f"      → Position {symbol} already closed, skipping")
                            unknown_already_closed.append(symbol)
                        else:
                            await self.client.cancel_open_orders(symbol)
                            
                            close_side = 'SELL' if side == 'LONG' else 'BUY'
                            await self._close_leg_reduce_only(
                                symbol=symbol,
                                side=close_side,
                                quantity=qty
                            )
                            print(f"      ✅ Closed unknown position {symbol}")
                            
                            # Fetch PnL
                            await asyncio.sleep(1)
                            import time as time_mod
                            now_ms = int(time_mod.time() * 1000)
                            start_ms = now_ms - 300_000
                            trades = await self.client.get_account_trades(symbol=symbol, startTime=start_ms, limit=50)
                            pnl = sum(float(t.get('realizedPnl', 0)) for t in trades)
                            unknown_closed.append((symbol, pnl))
                    except Exception as e:
                        print(f"      ⚠️ Failed to close {symbol}: {e}")
                        unknown_failed.append((symbol, str(e)))

                # Send ONE aggregated notification instead of N per-symbol messages.
                if recovered_pairs or unknown_to_close or unknown_already_closed or unknown_closed or unknown_failed:
                    lines = [
                        "🚨 <b>UNKNOWN POSITIONS RECONCILE</b>",
                        f"Detected: {len(unknown_positions)}",
                    ]
                    if recovered_pairs:
                        uniq_recovered = sorted(set(recovered_pairs))
                        lines.append(f"Recovered from DB: {len(uniq_recovered)}")
                        lines.extend([f"  • {p}" for p in uniq_recovered[:10]])
                        if len(uniq_recovered) > 10:
                            lines.append(f"  • ... and {len(uniq_recovered) - 10} more")
                    if unknown_to_close:
                        lines.append(f"Tried to close: {len(unknown_to_close)}")
                        lines.extend([f"  • {x}" for x in unknown_to_close[:10]])
                        if len(unknown_to_close) > 10:
                            lines.append(f"  • ... and {len(unknown_to_close) - 10} more")
                    if unknown_already_closed:
                        lines.append(f"Already closed before action: {len(unknown_already_closed)}")
                    if unknown_closed:
                        lines.append(f"Closed: {len(unknown_closed)}")
                        for sym, pnl in unknown_closed[:10]:
                            lines.append(f"  • {sym}: pnl={pnl:+.2f} USDT")
                        if len(unknown_closed) > 10:
                            lines.append(f"  • ... and {len(unknown_closed) - 10} more")
                    if unknown_failed:
                        lines.append(f"Failed: {len(unknown_failed)}")
                        for sym, err in unknown_failed[:5]:
                            lines.append(f"  • {sym}: {err[:120]}")
                        if len(unknown_failed) > 5:
                            lines.append(f"  • ... and {len(unknown_failed) - 5} more")
                    await self._notify("\n".join(lines))
            
            # Check each pair in DB
            pairs_to_fix = []
            for pair_set, pair_info in list(self.active_pairs.items()):
                s1, s2 = pair_info.symbol1, pair_info.symbol2
                s1_open = s1 in open_on_exchange
                s2_open = s2 in open_on_exchange
                
                if pair_info.position_status != 0:
                    # DB says position is OPEN
                    if not s1_open and not s2_open:
                        # But exchange says BOTH legs are closed!
                        print(f"  ⚠️ MISMATCH: {s1}-{s2} marked OPEN in DB but CLOSED on exchange. Fixing...")
                        pairs_to_fix.append((pair_info, 'close_db'))
                    elif s1_open != s2_open:
                        # One leg open, one closed - orphaned position!
                        remaining_sym = s1 if s1_open else s2
                        remaining_pos = open_on_exchange[remaining_sym]
                        remaining_qty = remaining_pos['qty']
                        remaining_side = remaining_pos['side']
                        unrealized_pnl = remaining_pos['unrealized_pnl']
                        
                        print(f"  🚨 ORPHAN: {s1}-{s2} has mismatched legs! {s1}:{s1_open}, {s2}:{s2_open}")
                        print(f"      Remaining: {remaining_sym} ({remaining_side} {remaining_qty}) PnL: {unrealized_pnl:.2f}")
                        
                        # Decision based on PnL
                        should_close = True
                        if unrealized_pnl < 0:
                            # Losing position - wait 30 seconds, notify user
                            print(f"      → PnL negative, waiting 30 seconds for user decision...")
                            pnl_emoji = "🔴"
                            await self._notify(
                                f"🚨 <b>ORPHAN POSITION DETECTED</b>\n\n"
                                f"Pair: {s1}-{s2}\n"
                                f"Remaining: {remaining_sym} ({remaining_side})\n"
                                f"💵 Unrealized PnL: {pnl_emoji} <b>{unrealized_pnl:.2f} USDT</b>\n\n"
                                f"⏱️ <b>Closing orphan...</b>"
                            )
                        else:
                            # Profitable or breakeven - auto close immediately
                            pnl_emoji = "🟢"
                            print(f"      → PnL >= 0, auto-closing immediately...")
                        
                        if should_close:
                            try:
                                # Re-verify position still exists before closing
                                verify_positions = await self.client.get_position_risk()
                                position_exists = False
                                for p in verify_positions:
                                    if p.get('symbol') == remaining_sym and abs(float(p.get('positionAmt', 0))) > 0:
                                        position_exists = True
                                        remaining_qty = abs(float(p.get('positionAmt', 0)))
                                        remaining_side = 'LONG' if float(p.get('positionAmt', 0)) > 0 else 'SHORT'
                                        break
                                
                                if not position_exists:
                                    print(f"      → Position {remaining_sym} already closed, skipping")
                                    pairs_to_fix.append((pair_info, 'close_db'))
                                    should_close = False
                                else:
                                    # Cancel any remaining orders for this symbol
                                    await self.client.cancel_open_orders(remaining_sym)
                                    
                                    # Close remaining leg with market order
                                    close_side = 'SELL' if remaining_side == 'LONG' else 'BUY'
                                    await self._close_leg_reduce_only(
                                        symbol=remaining_sym,
                                        side=close_side,
                                        quantity=remaining_qty
                                    )
                                    print(f"      ✅ Closed orphan leg {remaining_sym}")
                                
                                # Wait for trade data
                                import asyncio
                                await asyncio.sleep(1)
                                
                                # Fetch PnL from recent trades
                                import time as time_mod
                                now_ms = int(time_mod.time() * 1000)
                                start_ms = now_ms - 300_000  # Last 5 minutes
                                trades1 = await self.client.get_account_trades(symbol=s1, startTime=start_ms, limit=50)
                                trades2 = await self.client.get_account_trades(symbol=s2, startTime=start_ms, limit=50)
                                
                                pnl1 = sum(float(t.get('realizedPnl', 0)) for t in trades1)
                                pnl2 = sum(float(t.get('realizedPnl', 0)) for t in trades2)
                                fee1 = sum(float(t.get('commission', 0)) for t in trades1)
                                fee2 = sum(float(t.get('commission', 0)) for t in trades2)
                                total_pnl = pnl1 + pnl2
                                net_pnl = total_pnl - (fee1 + fee2)
                                pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"
                                
                                # Update DB with PnL
                                pairs_to_fix.append((pair_info, 'close_db_with_pnl', net_pnl, pnl1, pnl2, fee1, fee2))
                                await self._persist_pair_executions(
                                    pair_info, trades1, trades2, phase='ORPHAN_RESTART_CLOSE', trade_id=pair_info.current_trade_id
                                )
                                
                                # Notify with details
                                await self._notify(f"⚡ <b>Orphan Closed (Restart):</b> {s1}-{s2}\n\n"
                                                   f"💵 PnL: {pnl_emoji} <b>{net_pnl:.2f} USDT</b>\n"
                                                   f"   {s1}: {pnl1:+.2f} USDT\n"
                                                   f"   {s2}: {pnl2:+.2f} USDT\n"
                                                   f"💸 Fees: {fee1 + fee2:.4f} USDT")
                            except Exception as e:
                                print(f"      ⚠️ Failed to close orphan: {e}")
                                pairs_to_fix.append((pair_info, 'close_db'))
                                await self._notify(f"🚨 ORPHAN FAILED: {s1}-{s2}\n{remaining_sym} still open!\nError: {e}")
                    else:
                        # Both open - check algo orders for SL/TP protection
                        # Use pre-fetched all_algo_orders (not API call per pair!)
                        s1_orders = [o for o in all_algo_orders if o.get('symbol') == s1]
                        s2_orders = [o for o in all_algo_orders if o.get('symbol') == s2]
                        
                        def _algo_type(order: dict) -> str:
                            return str(order.get('orderType') or order.get('type') or order.get('o') or '').upper()
                        
                        def _is_sl(order: dict) -> bool:
                            t = _algo_type(order)
                            return t in ('STOP', 'STOP_MARKET')
                        
                        def _is_tp(order: dict) -> bool:
                            t = _algo_type(order)
                            return t in ('TAKE_PROFIT', 'TAKE_PROFIT_MARKET')
                        
                        has_sl1 = any(_is_sl(o) for o in s1_orders)
                        has_sl2 = any(_is_sl(o) for o in s2_orders)
                        has_tp1 = any(_is_tp(o) for o in s1_orders)
                        has_tp2 = any(_is_tp(o) for o in s2_orders)
                        
                        missing = []
                        if not has_sl1: missing.append(f"{s1} SL")
                        if not has_sl2: missing.append(f"{s2} SL")
                        if not has_tp1: missing.append(f"{s1} TP")
                        if not has_tp2: missing.append(f"{s2} TP")
                        
                        if missing:
                            print(f"  ⚠️ MISSING PROTECTION for {s1}-{s2}: {', '.join(missing)}")
                            await self._notify(f"⚠️ <b>MISSING PROTECTION:</b> {s1}-{s2}\n"
                                               f"Missing: {', '.join(missing)}\n\n"
                                               f"Bot will attempt to restore SL/TP (max 2 attempts)...")
                            restored = await self._restore_pair_protection(pair_info, max_attempts=2)
                            if restored:
                                await self._notify(f"✅ <b>Protection Restored:</b> {s1}-{s2}")
                            else:
                                await self._notify(
                                    f"🚨 <b>Protection restore FAILED:</b> {s1}-{s2}\n"
                                    f"After 2 attempts bot will close this pair and remove it from active rotation."
                                )
                                close_ok = False
                                try:
                                    if pair_info.position_status != 0:
                                        pair_info.close_handled = True
                                        pair_info.is_trading = True
                                        await self._execute_trade(pair_info, 0, close_reason='stale_symbols')
                                    close_ok = pair_info.position_status == 0
                                except Exception as close_err:
                                    print(f"⚠️ Protection failure close error for {s1}-{s2}: {close_err}")
                                if close_ok:
                                    if pair_info.db_id:
                                        await db.archive_pair(pair_info.db_id, reason='protection_restore_failed')
                                    if pair_set in self.active_pairs:
                                        self._unregister_pair(pair_info)
                                        del self.active_pairs[pair_set]
                                    self._cleanup_unused_subscription(s1)
                                    self._cleanup_unused_subscription(s2)
                                    await self._notify(f"🗑️ <b>Pair Removed:</b> {s1}-{s2} (protection failure)")
                                else:
                                    await self._notify(
                                        f"🚨 <b>PAIR NOT REMOVED</b>: {s1}-{s2}\n"
                                        f"Reason: could not safely confirm full close on exchange."
                                    )
                else:
                    # DB says position is CLOSED
                    if s1_open or s2_open:
                        # But exchange has open position!
                        print(f"  ⚠️ MISMATCH: {s1}-{s2} marked CLOSED in DB but has positions on exchange!")
                        # Update DB to reflect reality
                        if s1_open and s2_open:
                            pairs_to_fix.append((pair_info, 'open_db', open_on_exchange.get(s1), open_on_exchange.get(s2)))
                        else:
                            orphan_symbol = s1 if s1_open else s2
                            orphan_pos = open_on_exchange.get(orphan_symbol)
                            if orphan_pos:
                                owner_pair = self._find_symbol_owner_pair(orphan_symbol, exclude_pair=pair_info)
                                if owner_pair:
                                    print(
                                        f"  ℹ️ Skip orphan close for CLOSED pair {s1}-{s2}: "
                                        f"{orphan_symbol} belongs to active pair {owner_pair}"
                                    )
                                else:
                                    pairs_to_fix.append((pair_info, 'close_orphan_leg_keep_closed', orphan_symbol, orphan_pos))
            
            # Apply fixes
            externally_closed_pairs = []
            for fix in pairs_to_fix:
                pair_info = fix[0]
                action = fix[1]
                
                if action == 'close_db':
                    s1 = pair_info.symbol1
                    s2 = pair_info.symbol2
                    pnl1 = 0.0
                    pnl2 = 0.0
                    total_pnl = 0.0
                    net_pnl = 0.0
                    fee1 = 0.0
                    fee2 = 0.0
                    pnl_loaded = False
                    try:
                        start_ms = self._trade_window_start_ms(pair_info, default_lookback_sec=3600, buffer_sec=180)
                        trades1 = await self.client.get_account_trades(symbol=s1, startTime=start_ms, limit=100)
                        trades2 = await self.client.get_account_trades(symbol=s2, startTime=start_ms, limit=100)
                        pnl1 = sum(float(t.get('realizedPnl', 0)) for t in trades1)
                        pnl2 = sum(float(t.get('realizedPnl', 0)) for t in trades2)
                        fee1 = sum(float(t.get('commission', 0)) for t in trades1)
                        fee2 = sum(float(t.get('commission', 0)) for t in trades2)
                        total_pnl = pnl1 + pnl2
                        net_pnl = total_pnl - (fee1 + fee2)
                        pnl_loaded = (len(trades1) + len(trades2)) > 0
                    except Exception as pnl_err:
                        print(f"  ⚠️ Could not fetch external-close PnL for {s1}-{s2}: {pnl_err}")

                    # Mark as closed in DB
                    pair_info.position_status = 0
                    pair_info.qty1 = 0
                    pair_info.qty2 = 0
                    pair_info.entry_price1 = 0
                    pair_info.entry_price2 = 0
                    pair_info.is_trading = False
                    self.mark_pair_wait_for_next_candle(pair_info, reason='external')
                    
                    if pair_info.db_id:
                        pair_update = {
                            'id': pair_info.db_id,
                            'position_status': 0,
                            'qty1': 0,
                            'qty2': 0,
                            'entry_price1': 0,
                            'entry_price2': 0,
                            'close_time': int(time.time()),
                            'close_reason': 'external'
                        }
                        if pnl_loaded:
                            pair_update.update({
                                'close_pnl': net_pnl,
                                'pnl1': pnl1,
                                'pnl2': pnl2,
                                'fee1': fee1,
                                'fee2': fee2
                            })
                        await db.update_pair(pair_update)
                    
                    # Close any open trade records
                    if pair_info.current_trade_id:
                        await db.close_trade_record(
                            pair_info.current_trade_id,
                            status='CLOSED_EXTERNAL',
                            close_reason='external',
                            pnl=net_pnl if pnl_loaded else None,
                            fee1=fee1 if pnl_loaded else None,
                            fee2=fee2 if pnl_loaded else None,
                        )
                        if pnl_loaded:
                            await self._persist_pair_executions(
                                pair_info, trades1, trades2, phase='EXTERNAL_CLOSE_SYNC', trade_id=pair_info.current_trade_id
                            )
                        pair_info.current_trade_id = None

                    externally_closed_pairs.append({
                        'pair': f"{s1}-{s2}",
                        'pnl': net_pnl if pnl_loaded else None
                    })
                    
                    print(f"  ✅ Fixed: {pair_info.symbol1}-{pair_info.symbol2} marked as CLOSED in DB")
                
                elif action == 'open_db' and len(fix) >= 4:
                    # Mark as open in DB based on exchange data
                    pos1_data = fix[2]
                    pos2_data = fix[3]
                    if pos1_data and pos2_data:
                        pair_info.position_status = 1 if pos1_data['side'] == 'LONG' else -1
                        pair_info.qty1 = pos1_data['qty']
                        pair_info.qty2 = pos2_data['qty']
                        pair_info.entry_price1 = pos1_data['entry_price']
                        pair_info.entry_price2 = pos2_data['entry_price']
                        
                        if pair_info.db_id:
                            await db.update_pair({
                                'id': pair_info.db_id,
                                'position_status': pair_info.position_status,
                                'qty1': pair_info.qty1,
                                'qty2': pair_info.qty2,
                                'entry_price1': pair_info.entry_price1,
                                'entry_price2': pair_info.entry_price2
                            })
                        
                        print(f"  ✅ Fixed: {pair_info.symbol1}-{pair_info.symbol2} marked as OPEN in DB (synced from exchange)")

                elif action == 'close_orphan_leg_keep_closed' and len(fix) >= 4:
                    orphan_symbol = fix[2]
                    orphan_pos = fix[3] or {}
                    orphan_qty = abs(float(orphan_pos.get('qty', 0) or 0))
                    orphan_side = orphan_pos.get('side', 'LONG')
                    if orphan_qty <= 0:
                        continue
                    owner_pair = self._find_symbol_owner_pair(orphan_symbol, exclude_pair=pair_info)
                    if owner_pair:
                        print(
                            f"  ℹ️ Skip orphan close action for CLOSED pair {pair_info.symbol1}-{pair_info.symbol2}: "
                            f"{orphan_symbol} belongs to active pair {owner_pair}"
                        )
                        continue

                    close_side = 'SELL' if orphan_side == 'LONG' else 'BUY'
                    pnl_loaded = False
                    orphan_pnl = 0.0
                    orphan_fee = 0.0
                    net_pnl = 0.0

                    try:
                        await self.client.cancel_open_orders(orphan_symbol)
                    except Exception:
                        pass

                    await self._close_leg_reduce_only(
                        symbol=orphan_symbol,
                        side=close_side,
                        quantity=orphan_qty
                    )

                    try:
                        start_ms = self._trade_window_start_ms(pair_info, default_lookback_sec=3600, buffer_sec=180)
                        orphan_trades = await self._fetch_account_trades_window(orphan_symbol, start_ms, max_records=300)
                        orphan_pnl = sum(float(t.get('realizedPnl', 0)) for t in orphan_trades)
                        orphan_fee = sum(float(t.get('commission', 0)) for t in orphan_trades)
                        net_pnl = orphan_pnl - orphan_fee
                        pnl_loaded = len(orphan_trades) > 0
                    except Exception as pnl_err:
                        print(f"  ⚠️ Could not fetch orphan-leg close PnL for {orphan_symbol}: {pnl_err}")

                    pair_info.position_status = 0
                    pair_info.qty1 = 0
                    pair_info.qty2 = 0
                    pair_info.entry_price1 = 0
                    pair_info.entry_price2 = 0
                    pair_info.is_trading = False
                    self.mark_pair_wait_for_next_candle(pair_info, reason='orphan_restart')

                    pnl1 = orphan_pnl if orphan_symbol == pair_info.symbol1 else 0.0
                    pnl2 = orphan_pnl if orphan_symbol == pair_info.symbol2 else 0.0
                    fee1 = orphan_fee if orphan_symbol == pair_info.symbol1 else 0.0
                    fee2 = orphan_fee if orphan_symbol == pair_info.symbol2 else 0.0

                    if pair_info.db_id:
                        upd = {
                            'id': pair_info.db_id,
                            'position_status': 0,
                            'qty1': 0,
                            'qty2': 0,
                            'entry_price1': 0,
                            'entry_price2': 0,
                            'close_time': int(time.time()),
                            'close_reason': 'orphan_restart',
                        }
                        if pnl_loaded:
                            upd.update({
                                'close_pnl': net_pnl,
                                'pnl1': pnl1,
                                'pnl2': pnl2,
                                'fee1': fee1,
                                'fee2': fee2,
                            })
                        await db.update_pair(upd)

                    if pair_info.current_trade_id:
                        await db.close_trade_record(
                            pair_info.current_trade_id,
                            status='CLOSED_ORPHAN',
                            close_reason='orphan_restart',
                            pnl=net_pnl if pnl_loaded else None,
                            fee1=fee1 if pnl_loaded else None,
                            fee2=fee2 if pnl_loaded else None,
                        )
                        pair_info.current_trade_id = None

                    externally_closed_pairs.append({
                        'pair': f"{pair_info.symbol1}-{pair_info.symbol2}",
                        'pnl': net_pnl if pnl_loaded else None
                    })
                    print(f"  ✅ Fixed: {pair_info.symbol1}-{pair_info.symbol2} orphan leg {orphan_symbol} closed")
                
                elif action == 'close_db_with_pnl' and len(fix) >= 5:
                    # Mark as closed in DB with PnL info
                    net_pnl = fix[2]
                    pnl1 = fix[3]
                    pnl2 = fix[4]
                    fee1 = fix[5] if len(fix) >= 6 else 0.0
                    fee2 = fix[6] if len(fix) >= 7 else 0.0
                    
                    pair_info.position_status = 0
                    pair_info.qty1 = 0
                    pair_info.qty2 = 0
                    pair_info.entry_price1 = 0
                    pair_info.entry_price2 = 0
                    pair_info.is_trading = False
                    
                    if pair_info.db_id:
                        import time as time_mod
                        await db.update_pair({
                            'id': pair_info.db_id,
                            'position_status': 0,
                            'qty1': 0,
                            'qty2': 0,
                            'entry_price1': 0,
                            'entry_price2': 0,
                            'close_time': int(time_mod.time()),
                            'close_pnl': net_pnl,
                            'close_reason': 'orphan_restart',
                            'pnl1': pnl1,
                            'pnl2': pnl2,
                            'fee1': fee1,
                            'fee2': fee2,
                        })
                    
                    # Close trade record
                    if pair_info.current_trade_id:
                        await db.close_trade_record(
                                pair_info.current_trade_id,
                                status='CLOSED_ORPHAN',
                                close_reason='orphan_restart',
                                pnl=net_pnl,
                                fee1=fee1,
                                fee2=fee2,
                            )
                        pair_info.current_trade_id = None
                    
                    print(f"  ✅ Fixed: {pair_info.symbol1}-{pair_info.symbol2} orphan closed with PnL={net_pnl:.2f}")

            if externally_closed_pairs:
                known = [x for x in externally_closed_pairs if x['pnl'] is not None]
                unknown_count = len(externally_closed_pairs) - len(known)
                total_external_pnl = sum(x['pnl'] for x in known)
                total_emoji = "🟢" if total_external_pnl >= 0 else "🔴"

                lines = [f"⚡ <b>External Close Detected</b>",
                         f"Pairs closed on exchange: <b>{len(externally_closed_pairs)}</b>"]
                if known:
                    lines.append(f"💰 Total Realized PnL: {total_emoji} <b>{total_external_pnl:+.2f} USDT</b>")
                if unknown_count:
                    lines.append(f"ℹ️ PnL unavailable for {unknown_count} pair(s).")

                preview = externally_closed_pairs[:12]
                for item in preview:
                    if item['pnl'] is None:
                        lines.append(f"• {item['pair']}: n/a")
                    else:
                        e = "🟢" if item['pnl'] >= 0 else "🔴"
                        lines.append(f"• {item['pair']}: {e} {item['pnl']:+.2f} USDT")
                if len(externally_closed_pairs) > len(preview):
                    lines.append(f"... and {len(externally_closed_pairs) - len(preview)} more")

                await self._notify("\n".join(lines))
            
            active_count = self.count_active_positions()
            print(f"🔄 Reconciliation complete. Active pairs in DB: {active_count}")
            
            # Cleanup orphaned algo orders immediately
            await self._cleanup_orphaned_algo_orders()
            
        except Exception as e:
            print(f"❌ Error during reconciliation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self._reconcile_lock.locked():
                try:
                    self._reconcile_lock.release()
                except Exception:
                    pass

    def _get_exchange_pair_pnl(self, pair_info: PairInfo, price1: float = 0, price2: float = 0) -> float:
        """
        Get unrealized PnL for a pair from EXCHANGE cache (updated every 15s).
        This is the SINGLE SOURCE OF TRUTH for all PnL-based decisions.
        Falls back to manual calc only if cache is empty (first 15s after startup).
        """
        s1, s2 = pair_info.symbol1, pair_info.symbol2
        pnl1 = self._exchange_pnl_cache.get(s1)
        pnl2 = self._exchange_pnl_cache.get(s2)
        
        if pnl1 is not None and pnl2 is not None:
            return pnl1 + pnl2
        
        # Fallback: manual calc (only during first 15s before cache is populated)
        if price1 > 0 and price2 > 0 and pair_info.entry_price1 > 0 and pair_info.entry_price2 > 0:
            side1 = 1 if pair_info.position_status == 1 else -1
            side2 = -side1
            manual_pnl1 = (price1 - pair_info.entry_price1) * pair_info.qty1 * side1
            manual_pnl2 = (price2 - pair_info.entry_price2) * pair_info.qty2 * side2
            return manual_pnl1 + manual_pnl2
        
        return 0.0

    async def _cleanup_orphaned_algo_orders(self):
        """
        Clean up orphaned algo orders (STOP/TAKE_PROFIT).
        Uses EXCHANGE as source of truth:
        1. Cancel orders for symbols without positions on exchange
        2. Cancel extra orders if more than 2 per symbol (1 SL + 1 TP)
        3. Sync local state with exchange
        """
        try:
            # Get REAL positions from exchange (source of truth)
            positions_risk = await self.client.get_position_risk()
            exchange_positions = set()
            pnl_update = {}
            for pos in positions_risk:
                amt = float(pos.get('positionAmt', 0))
                if amt != 0:
                    sym = pos['symbol']
                    exchange_positions.add(sym)
                    # Cache unrealized PnL from exchange (source of truth for all PnL decisions)
                    pnl_update[sym] = float(pos.get('unRealizedProfit', 0))
            # Atomic update: replace old cache with fresh data
            self._exchange_pnl_cache = pnl_update
            
            # Sync local state: clear pairs that don't have positions on exchange
            for pair_info in list(self.active_pairs.values()):
                if pair_info.position_status != 0:
                    leg1_exists = pair_info.symbol1 in exchange_positions
                    leg2_exists = pair_info.symbol2 in exchange_positions
                    
                    if not leg1_exists and not leg2_exists:
                        print(f"🧹 Syncing stale pair: {pair_info.symbol1}-{pair_info.symbol2}")
                        pair_info.position_status = 0
                        pair_info.is_trading = False
                        self.mark_pair_wait_for_next_candle(pair_info, reason='stale_symbols')
            
            # Get all algo orders (using fixed endpoint /fapi/v1/openAlgoOrders)
            algo_orders = await self.client.get_algo_orders()
            if isinstance(algo_orders, dict):
                algo_orders = algo_orders.get('orders', [])
            if not isinstance(algo_orders, list):
                algo_orders = []
            if not algo_orders:
                return
            
            # Group orders by symbol
            orders_by_symbol = {}
            for order in algo_orders:
                order_type = str(order.get('orderType') or order.get('type') or order.get('o') or '').upper()
                status = order.get('algoStatus') or order.get('status', '')
                if order_type in ['STOP', 'STOP_MARKET', 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET'] and status == 'NEW':
                    sym = order['symbol']
                    if sym not in orders_by_symbol:
                        orders_by_symbol[sym] = []
                    orders_by_symbol[sym].append(order)
            
            # Find orphaned orders
            orphaned = []
            for sym, orders in orders_by_symbol.items():
                if sym not in exchange_positions:
                    # Symbol has no position on exchange - all its orders are orphaned
                    orphaned.extend(orders)
                    print(f"  🗑️ {sym}: {len(orders)} orders orphaned (no position)")
                elif len(orders) > 2:
                    # Too many orders for this symbol - keep first 2 (oldest by algoId)
                    orders.sort(key=lambda x: int(x.get('algoId', 0)))
                    extra = orders[2:]
                    orphaned.extend(extra)
                    print(f"  🗑️ {sym}: {len(extra)} extra orders (keeping 2)")
            
            if orphaned:
                print(f"🗑️ Cancelling {len(orphaned)} orphaned algo orders...")
                for o in orphaned:
                    try:
                        await self.client.cancel_algo_order(algoId=o['algoId'])
                    except Exception as e:
                        print(f"  ⚠️ Failed to cancel algoId {o.get('algoId')}: {e}")
                
                print(f"✅ Orphaned orders cleanup completed")
                
        except Exception as e:
            print(f"⚠️ Error cleaning up orphaned orders: {e}")

    async def _restore_pair_protection(self, pair_info: PairInfo, max_attempts: int = 2) -> bool:
        """
        Rebuild full SL/TP protection for an already-open pair.
        max_attempts includes one retry (e.g. 2 = try + retry once).
        """
        if pair_info.position_status == 0:
            return True

        s1, s2 = pair_info.symbol1, pair_info.symbol2
        s1_info = self.all_symbols.get(s1)
        s2_info = self.all_symbols.get(s2)
        if not s1_info or not s2_info:
            print(f"⚠️ Cannot restore protection: missing symbol metadata for {s1}-{s2}")
            return False

        # Build ATR with fallback to config min percentages (atr=0 fallback in utils).
        data1 = self.all_data.get(s1)
        data2 = self.all_data.get(s2)
        atr1 = 0.0
        atr2 = 0.0
        try:
            if data1 and len(data1.close) > 1:
                atr1 = utils.calculate_atr(list(data1.high), list(data1.low), list(data1.close))
            if data2 and len(data2.close) > 1:
                atr2 = utils.calculate_atr(list(data2.high), list(data2.low), list(data2.close))
        except Exception as atr_err:
            print(f"⚠️ ATR calc error while restoring protection for {s1}-{s2}: {atr_err}")

        direction = pair_info.position_status
        leg1_side = 'LONG' if direction == 1 else 'SHORT'
        leg2_side = 'SHORT' if direction == 1 else 'LONG'
        close_side1 = 'SELL' if direction == 1 else 'BUY'
        close_side2 = 'BUY' if direction == 1 else 'SELL'

        sl1, tp1, _, _ = utils.calculate_hardware_stops(pair_info.entry_price1, leg1_side, atr1, self.config)
        sl2, tp2, _, _ = utils.calculate_hardware_stops(pair_info.entry_price2, leg2_side, atr2, self.config)
        sl1 = round(sl1, s1_info.tick_size)
        sl2 = round(sl2, s2_info.tick_size)
        tp1 = round(tp1, s1_info.tick_size)
        tp2 = round(tp2, s2_info.tick_size)
        if sl1 <= 0 or sl2 <= 0 or tp1 <= 0 or tp2 <= 0:
            print(f"WARN: Invalid restore prices for {s1}-{s2}: sl1={sl1}, tp1={tp1}, sl2={sl2}, tp2={tp2}")
            return False

        pair_key = frozenset([s1, s2])
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"🛡️ Restore protection attempt {attempt}/{max_attempts} for {s1}-{s2}")

                # Cancel any stale regular stop orders.
                await asyncio.gather(
                    self.client.cancel_open_orders(symbol=s1),
                    self.client.cancel_open_orders(symbol=s2),
                    return_exceptions=True
                )

                # Cancel stale algo orders for both symbols.
                try:
                    algo_orders = await self.client.get_algo_orders()
                    if isinstance(algo_orders, dict) and 'orders' in algo_orders:
                        algo_orders = algo_orders.get('orders', [])
                    for o in (algo_orders or []):
                        sym = o.get('symbol')
                        if sym not in (s1, s2):
                            continue
                        status = o.get('algoStatus') or o.get('status', '')
                        if status != 'NEW':
                            continue
                        o_type = str(o.get('orderType') or o.get('type') or o.get('o') or '').upper()
                        if o_type in ('STOP', 'STOP_MARKET', 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET'):
                            aid = o.get('algoId')
                            if aid is not None:
                                await self.client.cancel_algo_order(algoId=aid)
                except Exception as clean_err:
                    print(f"⚠️ Cleanup before protection restore failed for {s1}-{s2}: {clean_err}")

                tasks = [
                    self.client.new_algo_order(symbol=s1, side=close_side1, type='STOP_MARKET',
                                               triggerPrice=sl1, quantity=pair_info.qty1, reduceOnly='true'),
                    self.client.new_algo_order(symbol=s2, side=close_side2, type='STOP_MARKET',
                                               triggerPrice=sl2, quantity=pair_info.qty2, reduceOnly='true'),
                    self.client.new_algo_order(symbol=s1, side=close_side1, type='TAKE_PROFIT_MARKET',
                                               triggerPrice=tp1, quantity=pair_info.qty1, reduceOnly='true'),
                    self.client.new_algo_order(symbol=s2, side=close_side2, type='TAKE_PROFIT_MARKET',
                                               triggerPrice=tp2, quantity=pair_info.qty2, reduceOnly='true'),
                ]
                task_meta = [(s1, 'STOP'), (s2, 'STOP'), (s1, 'TAKE_PROFIT'), (s2, 'TAKE_PROFIT')]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                ok_ids = []
                for r in results:
                    if isinstance(r, Exception):
                        raise r
                    if isinstance(r, dict) and 'algoId' in r:
                        ok_ids.append(str(r['algoId']))

                if len(ok_ids) != len(task_meta):
                    raise RuntimeError(f"Expected {len(task_meta)} algo orders, got {len(ok_ids)}")

                # Replace local algo mapping for this pair.
                to_remove = [aid for aid, info in self.algo_orders.items() if info.get('pair_key') == pair_key]
                for aid in to_remove:
                    self.algo_orders.pop(aid, None)
                for aid, (sym, typ) in zip(ok_ids, task_meta):
                    self.algo_orders[aid] = {'pair_key': pair_key, 'symbol': sym, 'type': typ}

                print(f"OK: Protection restored for {s1}-{s2} ({len(task_meta)} orders)")
                return True
            except Exception as restore_err:
                print(f"⚠️ Protection restore attempt {attempt} failed for {s1}-{s2}: {restore_err}")
                if attempt < max_attempts:
                    await asyncio.sleep(1)

        return False

    async def restore_protection_for_symbol(self, symbol: str, max_attempts: int = 2) -> bool:
        """
        Public helper for runtime repair after SL/TP cancel events.
        Returns True if protection was restored for at least one open pair containing symbol.
        """
        restored_any = False
        for pair_info in list(self._pairs_with_symbol(symbol)):
            if pair_info.position_status == 0 or pair_info.is_trading:
                continue
            ok = await self._restore_pair_protection(pair_info, max_attempts=max_attempts)
            restored_any = restored_any or ok
        return restored_any

    async def handle_sl_tp_triggered(self, symbol: str, order_type: str = 'STOP'):
        """
        Called when a hardware SL/TP order is filled (via WebSocket).
        Finds the pair containing this symbol and closes the other leg.
        order_type: 'STOP', 'STOP_MARKET' for SL; 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET' for TP
        """
        is_tp = 'TAKE_PROFIT' in order_type.upper() if order_type else False
        
        for pair_info in list(self.active_pairs.values()):
            if pair_info.position_status == 0:
                continue
            if symbol in (pair_info.symbol1, pair_info.symbol2):
                other_symbol = pair_info.symbol2 if symbol == pair_info.symbol1 else pair_info.symbol1
                
                # VERIFY SL vs TP using actual PnL (order_type from WS can be unreliable for algo orders)
                # If the triggered leg closed with PROFIT, it's a TP; if LOSS, it's SL
                try:
                    entry_price = pair_info.entry_price1 if symbol == pair_info.symbol1 else pair_info.entry_price2
                    qty = pair_info.qty1 if symbol == pair_info.symbol1 else pair_info.qty2
                    is_s1 = symbol == pair_info.symbol1
                    
                    # Determine position direction for this leg
                    if is_s1:
                        side_dir = 1 if pair_info.position_status == 1 else -1  # s1: long if status=1
                    else:
                        side_dir = -1 if pair_info.position_status == 1 else 1  # s2: short if status=1
                    
                    # Get actual PnL from recent trades (source of truth)
                    try:
                        # Wait briefly for trade to register
                        await asyncio.sleep(0.5)
                        start_ms = self._trade_window_start_ms(pair_info)
                        recent_trades = await self._fetch_account_trades_window(
                            symbol=symbol,
                            start_ms=start_ms,
                            max_records=1000
                        )
                        
                        if recent_trades:
                            # Sum realized PnL of recent trades for this symbol
                            leg_pnl = sum(float(t.get('realizedPnl', 0)) for t in recent_trades)
                            
                            # Override order_type classification based on actual PnL
                            if leg_pnl > 0:
                                is_tp = True
                                print(f"📊 PnL verification: {symbol} PnL={leg_pnl:+.2f} → confirmed TAKE PROFIT")
                            else:
                                is_tp = False
                                print(f"📊 PnL verification: {symbol} PnL={leg_pnl:+.2f} → confirmed STOP LOSS")
                        else:
                            # Fallback to manual calc if no trades found (rare)
                            close_price = self.last_prices.get(symbol, 0)
                            if close_price > 0 and entry_price > 0:
                                leg_pnl = (close_price - entry_price) * qty * side_dir
                                if leg_pnl > 0:
                                    is_tp = True
                                else:
                                    is_tp = False
                                print(f"⚠️ Exchange trades not found, manual PnL: {leg_pnl:.2f} ({'TP' if is_tp else 'SL'})")
                            else:
                                print(f"📊 PnL verification skipped (no trades & missing price data)")
                    except Exception as e:
                        print(f"⚠️ PnL verification error: {e}. Using order_type={order_type}")
                except Exception as e:
                    print(f"⚠️ PnL verification error: {e}. Using order_type={order_type}")
                
                close_reason = 'hardware_tp' if is_tp else 'hardware_sl'
                tp_or_sl = 'TP' if is_tp else 'SL'
                msg = f"🎯 Hardware {tp_or_sl} triggered on {symbol}! Closing {other_symbol}"
                print(msg)
                # Don't notify here - _execute_trade will send full close notification with PnL
                
                # Force close the pair (cancels algo orders + closes other leg if needed)
                pair_info.close_handled = True
                pair_info.is_trading = True
                pair_info._triggered_symbol = symbol  # Tell _execute_trade which leg is already closed
                await self._execute_trade(pair_info, 0, close_reason=close_reason)
                break

    async def _periodic_leg_sync_loop(self):
        """BACKUP: Periodically check leg sync and cleanup orphaned orders every 15 seconds.
        Primary sync is handled by userdata WebSocket."""
        while True:
            await asyncio.sleep(15)  # Backup check (15s — primary sync via userdata WS)
            try:
                await self._check_leg_synchronization()
                await self._cleanup_orphaned_algo_orders()
                await self._cleanup_idle_pairs()  # Remove old idle pairs
            except Exception as e:
                print(f"⚠️ Leg sync/cleanup error: {e}")

    async def _cleanup_idle_pairs(self):
        """
        Remove old/excess idle pairs to prevent accumulation.
        Idle pairs = position_status == 0 and not is_trading.
        Configurable via TG: max_idle_pairs, idle_timeout_hours.
        """
        max_idle = getattr(self.config, 'max_idle_pairs', 150) or 150
        timeout_hours = getattr(self.config, 'idle_timeout_hours', 48) or 48
        timeout_sec = timeout_hours * 3600
        now = time.time()
        
        # Collect idle pairs (no open position, not currently trading)
        idle_pairs = []
        for pair_set, pair_info in list(self.active_pairs.items()):
            if pair_info.position_status == 0 and not pair_info.is_trading:
                idle_pairs.append((pair_set, pair_info))
        
        removed_count = 0
        
        # 1. Remove timed-out pairs first
        for pair_set, pair_info in idle_pairs:
            if pair_info.discovered_at > 0 and (now - pair_info.discovered_at) > timeout_sec:
                await self._remove_idle_pair(pair_set, 'timeout')
                removed_count += 1
        
        # 2. Recalculate idle count after timeout cleanup
        idle_pairs = [(ps, pi) for ps, pi in list(self.active_pairs.items())
                      if pi.position_status == 0 and not pi.is_trading]
        
        # 3. Remove excess pairs (weakest first) if still over limit
        if len(idle_pairs) > max_idle:
            # Lower quality_score is removed first, then older discovery time.
            idle_pairs.sort(
                key=lambda x: (
                    float(getattr(x[1], 'quality_score', 0.0) or 0.0),
                    x[1].discovered_at if x[1].discovered_at > 0 else float('inf')
                )
            )
            excess = len(idle_pairs) - max_idle
            for pair_set, pair_info in idle_pairs[:excess]:
                await self._remove_idle_pair(pair_set, 'limit')
                removed_count += 1
        
        if removed_count > 0:
            print(f"🗑️ Cleaned up {removed_count} idle pairs (limit: {max_idle}, timeout: {timeout_hours}h)")
    
    async def _remove_idle_pair(self, pair_set: frozenset, reason: str):
        """
        Remove an idle pair from active_pairs and DB.
        Does NOT close any positions (pair must be idle).
        """
        pair_info = self.active_pairs.get(pair_set)
        if not pair_info:
            return
        
        s1, s2 = pair_info.symbol1, pair_info.symbol2
        
        # Safety check: never remove pairs with open positions
        if pair_info.position_status != 0 or pair_info.is_trading:
            print(f"⚠️ Cannot remove {s1}-{s2}: has open position or is trading")
            return
        
        # Remove from active_pairs and symbol index
        self._unregister_pair(pair_info)
        del self.active_pairs[pair_set]
        
        # Remove markPrice subscription if symbols not used by other pairs
        self._cleanup_unused_subscription(s1)
        self._cleanup_unused_subscription(s2)
        
        # Delete from DB
        if pair_info.db_id:
            await db.archive_pair(pair_info.db_id, reason=f"idle_{reason}")
        
        print(f"  🗑️ Removed idle pair {s1}-{s2} (reason: {reason})")
    
    def _cleanup_unused_subscription(self, symbol: str):
        """Remove symbol from subscribed set if no other pairs use it."""
        # Check if any other pair uses this symbol
        for pair_info in self.active_pairs.values():
            if symbol in (pair_info.symbol1, pair_info.symbol2):
                return  # Still in use
        
        # Not used anymore - remove from tracked subscriptions
        if symbol in self._subscribed_mark_symbols:
            self._subscribed_mark_symbols.discard(symbol)
            # Trigger websocket resync so removed symbols are truly unsubscribed.
            # Callback is async; run in background to avoid blocking cleanup path.
            if self._subscribe_mark_callback:
                try:
                    self.loop.create_task(self._subscribe_mark_callback(list(self._subscribed_mark_symbols)))
                except Exception:
                    pass

    async def _close_leg_reduce_only(self, symbol: str, side: str, quantity: float) -> dict | None:
        """
        Close one futures leg with reduceOnly.
        Primary path: MARKET.
        Fallback for -4131 (PERCENT_PRICE): LIMIT IOC with safe bounded price.
        """
        qty = abs(float(quantity or 0))
        if qty <= 0:
            return

        try:
            return await self.client.new_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=qty,
                reduceOnly='true'
            )
        except Exception as e:
            is_percent_price = getattr(e, 'error_code', None) == -4131 or 'PERCENT_PRICE' in str(e)
            if not is_percent_price:
                raise
            print(f"⚠️ MARKET reduceOnly failed for {symbol} ({e}). Trying LIMIT IOC fallback...")

        sym_info = self.client.symbols.get(symbol) if getattr(self.client, 'symbols', None) else None
        tick_digits = int(getattr(sym_info, 'tick_size', 3) or 3)
        multiplier_up = float(getattr(sym_info, 'multiplier_up', 0) or 0)
        multiplier_down = float(getattr(sym_info, 'multiplier_down', 0) or 0)

        last_err = None
        for _ in range(2):
            try:
                book = await self.client.book_ticker(symbol=symbol)
                bid = float(book.get('bidPrice', 0) or 0)
                ask = float(book.get('askPrice', 0) or 0)

                mark_data = await self.client.mark_price(symbol=symbol)
                mark = float(mark_data.get('markPrice', 0) or 0)
                if mark <= 0:
                    mark = ask if ask > 0 else bid
                if mark <= 0:
                    raise RuntimeError(f"Could not get valid mark/bbo for {symbol}")

                lower = mark * multiplier_down if multiplier_down > 0 else 0.0
                upper = mark * multiplier_up if multiplier_up > 0 else float('inf')

                if side == 'SELL':
                    price = bid if bid > 0 else mark
                else:
                    price = ask if ask > 0 else mark

                if lower > 0:
                    price = max(price, lower)
                if upper < float('inf'):
                    price = min(price, upper)

                price = round(price, tick_digits)
                if price <= 0:
                    raise RuntimeError(f"Invalid fallback price for {symbol}: {price}")

                result = await self.client.new_order(
                    symbol=symbol,
                    side=side,
                    type='LIMIT',
                    timeInForce='IOC',
                    quantity=qty,
                    price=price,
                    reduceOnly='true',
                    newOrderRespType='RESULT'
                )
                print(f"✅ Closed {symbol} via LIMIT IOC fallback at {price}")
                return result
            except Exception as ioc_err:
                last_err = ioc_err
                await asyncio.sleep(0.2)

        raise last_err if last_err is not None else RuntimeError(f"Failed to close {symbol} with fallback path")

    async def _check_leg_synchronization(self):
        """Check that both legs of each active pair are open."""
        try:
            account = await self.client.account()
            pnl_update = {}
            pos_by_symbol = {}
            for pos in account['positions']:
                amt = float(pos['positionAmt'])
                if amt != 0:
                    sym = pos['symbol']
                    pos_by_symbol[sym] = amt
                    # Populate PnL cache immediately on startup/sync
                    # Handle both casing styles just in case
                    pnl = float(pos.get('unrealizedProfit', 0) or pos.get('unRealizedProfit', 0))
                    pnl_update[sym] = pnl
                    
            # Update exchange caches
            self._exchange_positions_cache = {s: abs(q) for s, q in pos_by_symbol.items()}
            self._exchange_pnl_cache = pnl_update  # Immediate PnL source of truth
            self._exchange_position_count = len(pos_by_symbol)
            
            externally_closed_now = []
            for pair_info in list(self.active_pairs.values()):
                if pair_info.is_trading:
                    continue

                leg1_open = pair_info.symbol1 in pos_by_symbol
                leg2_open = pair_info.symbol2 in pos_by_symbol

                if pair_info.position_status == 0:
                    if leg1_open != leg2_open:
                        orphan_symbol = pair_info.symbol1 if leg1_open else pair_info.symbol2
                        owner_pair = self._find_symbol_owner_pair(orphan_symbol, exclude_pair=pair_info)
                        if owner_pair:
                            now_conflict = time.time()
                            last_warn = float(getattr(pair_info, '_orphan_conflict_warn_ts', 0) or 0)
                            # Throttle to avoid log spam in 15s sync loop.
                            if now_conflict - last_warn >= 120:
                                pair_info._orphan_conflict_warn_ts = now_conflict
                                print(
                                    f"ℹ️ Skip orphan close for CLOSED pair {pair_info.symbol1}-{pair_info.symbol2}: "
                                    f"{orphan_symbol} belongs to active pair {owner_pair}"
                                )
                            continue
                        orphan_amt = pos_by_symbol.get(orphan_symbol, 0)
                        if orphan_amt != 0:
                            print(f"⚠️ Orphan leg on CLOSED pair {pair_info.symbol1}-{pair_info.symbol2}: {orphan_symbol} amt={orphan_amt}. Closing...")
                            try:
                                await self.client.cancel_open_orders(pair_info.symbol1)
                            except Exception:
                                pass
                            try:
                                await self.client.cancel_open_orders(pair_info.symbol2)
                            except Exception:
                                pass
                            try:
                                close_side = 'SELL' if orphan_amt > 0 else 'BUY'
                                await self._close_leg_reduce_only(
                                    symbol=orphan_symbol,
                                    side=close_side,
                                    quantity=abs(orphan_amt)
                                )
                                self._exchange_positions_cache.pop(orphan_symbol, None)
                                self._exchange_pnl_cache.pop(orphan_symbol, None)
                                self._exchange_position_count = len(self._exchange_positions_cache)
                                self.mark_pair_wait_for_next_candle(pair_info, reason='orphan_restart')
                                if pair_info.current_trade_id:
                                    try:
                                        await db.close_trade_record(
                                            pair_info.current_trade_id,
                                            status='CLOSED_ORPHAN',
                                            close_reason='orphan_restart',
                                        )
                                    except Exception:
                                        pass
                                    pair_info.current_trade_id = None
                                if pair_info.db_id:
                                    await db.update_pair({
                                        'id': pair_info.db_id,
                                        'position_status': 0,
                                        'qty1': 0,
                                        'qty2': 0,
                                        'entry_price1': 0,
                                        'entry_price2': 0,
                                        'close_time': int(time.time()),
                                        'close_reason': 'orphan_restart',
                                    })
                                await self._notify(
                                    f"⚠️ <b>Orphan Leg Closed</b>\n"
                                    f"Pair: <b>{pair_info.symbol1}/{pair_info.symbol2}</b>\n"
                                    f"Closed leg: <b>{orphan_symbol}</b>"
                                )
                            except Exception as orphan_err:
                                print(f"❌ Failed to close orphan leg {orphan_symbol}: {orphan_err}")
                    continue

                if not leg1_open and not leg2_open:
                    s1, s2 = pair_info.symbol1, pair_info.symbol2
                    pnl1 = 0.0
                    pnl2 = 0.0
                    total_pnl = 0.0
                    net_pnl = 0.0
                    fee1 = 0.0
                    fee2 = 0.0
                    pnl_loaded = False
                    try:
                        start_ms = self._trade_window_start_ms(pair_info, default_lookback_sec=3600, buffer_sec=180)
                        trades1 = await self.client.get_account_trades(symbol=s1, startTime=start_ms, limit=100)
                        trades2 = await self.client.get_account_trades(symbol=s2, startTime=start_ms, limit=100)
                        pnl1 = sum(float(t.get('realizedPnl', 0)) for t in trades1)
                        pnl2 = sum(float(t.get('realizedPnl', 0)) for t in trades2)
                        fee1 = sum(float(t.get('commission', 0)) for t in trades1)
                        fee2 = sum(float(t.get('commission', 0)) for t in trades2)
                        total_pnl = pnl1 + pnl2
                        net_pnl = total_pnl - (fee1 + fee2)
                        pnl_loaded = (len(trades1) + len(trades2)) > 0
                    except Exception as pnl_err:
                        print(f"⚠️ Could not fetch external-close PnL for {s1}-{s2}: {pnl_err}")

                    pair_info.position_status = 0
                    pair_info.qty1 = 0
                    pair_info.qty2 = 0
                    pair_info.entry_price1 = 0
                    pair_info.entry_price2 = 0
                    pair_info.is_trading = False
                    self.mark_pair_wait_for_next_candle(pair_info, reason='external')
                    pair_info.close_handled = True

                    if pair_info.db_id:
                        pair_update = {
                            'id': pair_info.db_id,
                            'position_status': 0,
                            'qty1': 0,
                            'qty2': 0,
                            'entry_price1': 0,
                            'entry_price2': 0,
                            'close_time': int(time.time()),
                            'close_reason': 'external'
                        }
                        if pnl_loaded:
                            pair_update.update({
                                'close_pnl': net_pnl,
                                'pnl1': pnl1,
                                'pnl2': pnl2,
                                'fee1': fee1,
                                'fee2': fee2,
                            })
                        await db.update_pair(pair_update)

                    if pair_info.current_trade_id:
                        await db.close_trade_record(
                            pair_info.current_trade_id,
                            status='CLOSED_EXTERNAL',
                            close_reason='external',
                            pnl=net_pnl if pnl_loaded else None,
                            fee1=fee1 if pnl_loaded else None,
                            fee2=fee2 if pnl_loaded else None,
                        )
                        if pnl_loaded:
                            await self._persist_pair_executions(
                                pair_info, trades1, trades2, phase='EXTERNAL_CLOSE_SYNC', trade_id=pair_info.current_trade_id
                            )
                        pair_info.current_trade_id = None

                    externally_closed_now.append({
                        'pair': f"{s1}-{s2}",
                        'pnl': net_pnl if pnl_loaded else None
                    })
                    print(f"⚡ External close detected: {s1}-{s2}")
                    continue
                
                if leg1_open != leg2_open:
                    # One leg closed unexpectedly - need to close the other and report PnL
                    s1, s2 = pair_info.symbol1, pair_info.symbol2
                    closed_leg = s1 if not leg1_open else s2
                    remaining_leg = s2 if not leg1_open else s1
                    remaining_qty = pos_by_symbol.get(remaining_leg, 0)
                    
                    print(f"⚡ Desync detected: {s1}-{s2}. {closed_leg} closed, closing {remaining_leg}...")
                    
                    # INVESTIGATE: Why was the closed leg closed?
                    desync_reason = ''
                    try:
                        # Check recent orders for the closed leg to determine cause
                        now_ms = int(time.time() * 1000)
                        recent_orders = await self.client.get_all_orders(symbol=closed_leg, limit=10)
                        
                        # Find the most recent FILLED order that could have closed it
                        close_candidates = []
                        for o in recent_orders:
                            if o.get('status') == 'FILLED' and o.get('updateTime', 0) > now_ms - 300_000:
                                close_candidates.append(o)
                        
                        if close_candidates:
                            close_candidates.sort(key=lambda x: x.get('updateTime', 0), reverse=True)
                            trigger_order = close_candidates[0]
                            o_type = str(trigger_order.get('type') or trigger_order.get('origType') or '')
                            
                            if 'STOP' in o_type:
                                desync_reason = f'Hardware SL triggered on {closed_leg}'
                            elif 'TAKE_PROFIT' in o_type:
                                desync_reason = f'Hardware TP triggered on {closed_leg}'
                            elif o_type == 'MARKET':
                                reduce_only = trigger_order.get('reduceOnly', False)
                                if reduce_only:
                                    desync_reason = f'Bot/reduceOnly market close on {closed_leg}'
                                else:
                                    desync_reason = f'Manual market close on {closed_leg}'
                            elif o_type == 'LIMIT':
                                desync_reason = f'Limit order closed {closed_leg}'
                            elif 'TRAILING' in o_type:
                                desync_reason = f'Trailing stop on {closed_leg}'
                            else:
                                desync_reason = f'{o_type} order on {closed_leg}'
                        else:
                            # Check algo orders
                            try:
                                algo_orders = await self.client.get_algo_orders()
                                if isinstance(algo_orders, dict):
                                    algo_orders = algo_orders.get('orders', [])
                                if not isinstance(algo_orders, list):
                                    algo_orders = []
                                triggered_algos = [a for a in algo_orders 
                                                   if a.get('symbol') == closed_leg 
                                                    and a.get('algoStatus') in ('TRIGGERED', 'FINISHED')]
                                if triggered_algos:
                                    algo_type = triggered_algos[0].get('orderType', 'ALGO')
                                    desync_reason = f'Algo {algo_type} triggered on {closed_leg}'
                                else:
                                    desync_reason = f'Unknown cause (no recent orders for {closed_leg})'
                            except Exception:
                                desync_reason = f'Unknown cause (could not query orders for {closed_leg})'
                    except Exception as e:
                        desync_reason = f'Could not determine cause: {str(e)[:50]}'
                    
                    print(f"  🔍 Desync cause: {desync_reason}")
                    
                    pair_info.close_handled = True  # Prevent WS handler from sending duplicate notification
                    pair_info.is_trading = True
                    
                    try:
                        # Cancel any remaining orders
                        await self.client.cancel_open_orders(s1)
                        await self.client.cancel_open_orders(s2)
                        
                        # Close remaining leg if it has position
                        if remaining_qty != 0:
                            close_side = 'SELL' if remaining_qty > 0 else 'BUY'
                            await self._close_leg_reduce_only(
                                symbol=remaining_leg,
                                side=close_side,
                                quantity=remaining_qty
                            )
                            print(f"✅ Closed remaining leg {remaining_leg}")
                        
                        # Wait for trade data to be available
                        await asyncio.sleep(1)
                        
                        # Fetch actual PnL from recent trades
                        start_ms = self._trade_window_start_ms(pair_info)
                        
                        trades1 = await self._fetch_account_trades_window(s1, start_ms, max_records=2000)
                        trades2 = await self._fetch_account_trades_window(s2, start_ms, max_records=2000)
                        
                        print(f"📊 Trades for {s1}: {len(trades1)} entries")
                        print(f"📊 Trades for {s2}: {len(trades2)} entries")
                        
                        # Sum realized PnL (already includes fees)
                        pnl1 = sum(float(t.get('realizedPnl', 0)) for t in trades1)
                        pnl2 = sum(float(t.get('realizedPnl', 0)) for t in trades2)
                        fee1 = sum(float(t.get('commission', 0)) for t in trades1)
                        fee2 = sum(float(t.get('commission', 0)) for t in trades2)
                        total_pnl = pnl1 + pnl2
                        total_fees = fee1 + fee2
                        net_pnl = total_pnl - total_fees
                        
                        pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"
                        e1 = '🟢' if pnl1 >= 0 else '🔴'
                        e2 = '🟢' if pnl2 >= 0 else '🔴'
                        
                        # Update memory state
                        pair_info.position_status = 0
                        pair_info.qty1 = 0
                        pair_info.qty2 = 0
                        pair_info.is_trading = False
                        
                        # Calculate Z-score for notification
                        close_zscore = 0.0
                        try:
                            p1 = self.last_prices.get(s1, 0)
                            p2 = self.last_prices.get(s2, 0)
                            import math
                            if p1 > 0 and p2 > 0:
                                close_zscore = self._calc_realtime_zscore(pair_info, p1, p2)
                                if math.isnan(close_zscore):
                                    close_zscore = 0.0
                        except Exception:
                            pass
                        
                        # Fallback: recalculate from historical data if realtime failed
                        if close_zscore == 0.0 and isinstance(self.all_data, dict) and s1 in self.all_data and s2 in self.all_data:
                            try:
                                _d1 = self.all_data[s1]
                                _d2 = self.all_data[s2]
                                if len(_d1.close) >= self.min_data_points and len(_d2.close) >= self.min_data_points:
                                    _lp1 = np.log(list(_d1.close)[-self.min_data_points:])
                                    _lp2 = np.log(list(_d2.close)[-self.min_data_points:])
                                    _spread = _lp1 - pair_info.hedge_ratio * _lp2
                                    _mean = np.mean(_spread)
                                    _std = np.std(_spread)
                                    if _std > 0:
                                        close_zscore = float((_spread[-1] - _mean) / _std)
                            except Exception:
                                pass
                        
                        if close_zscore == 0.0:
                            close_zscore = pair_info.last_z_score or 0
                        close_beta = getattr(pair_info, 'beta_btc', 0) or 0
                        close_pval = getattr(pair_info, 'last_pvalue', 0) or 0
                        
                        # Recalculate beta & p-value fresh if they're 0 (stale after restart)
                        if (close_beta == 0 or close_pval == 0) and isinstance(self.all_data, dict) and s1 in self.all_data and s2 in self.all_data:
                            try:
                                _d1 = self.all_data[s1]
                                _d2 = self.all_data[s2]
                                if len(_d1.close) >= self.min_data_points and len(_d2.close) >= self.min_data_points:
                                    _lp1 = np.log(list(_d1.close)[-self.min_data_points:])
                                    _lp2 = np.log(list(_d2.close)[-self.min_data_points:])
                                    _, _, _, _pval = utils.calculate_cointegration(_lp1, _lp2, strict_hl=False)
                                    if close_pval == 0 and not np.isnan(_pval):
                                        close_pval = float(_pval)
                                    if close_beta == 0 and isinstance(self.all_data, dict) and 'BTCUSDT' in self.all_data:
                                        _btc = self.all_data['BTCUSDT']
                                        if len(_btc.close) >= self.min_data_points:
                                            _lbtc = np.log(list(_btc.close)[-self.min_data_points:])
                                            _sr = np.diff(_lp1) - pair_info.hedge_ratio * np.diff(_lp2)
                                            _br = np.diff(_lbtc)
                                            _beta = utils.calculate_pair_beta(_sr, _br)
                                            if not np.isnan(_beta):
                                                close_beta = float(_beta)
                            except Exception as e:
                                print(f"⚠️ Fresh beta/pval calc error at desync close: {e}")
                        
                        close_hl = self._format_half_life(pair_info.half_life) if pair_info.half_life and pair_info.half_life > 0 else 'N/A'
                        hedge = getattr(pair_info, 'hedge_ratio', 0) or 0
                        
                        # Update trade record if available
                        if pair_info.current_trade_id:
                            try:
                                await db.close_trade_record(
                                    pair_info.current_trade_id,
                                    status='CLOSED',
                                    close_reason='desync',
                                    pnl=net_pnl,
                                    close_z=close_zscore,
                                    fee1=fee1,
                                    fee2=fee2,
                                )
                                await self._persist_pair_executions(
                                    pair_info, trades1, trades2, phase='DESYNC_CLOSE', trade_id=pair_info.current_trade_id
                                )
                            except Exception:
                                pass
                            pair_info.current_trade_id = None
                        
                        # Save beta/pvalue to DB for analysis
                        if pair_info.db_id:
                            await db.update_pair({
                                'id': pair_info.db_id,
                                'position_status': 0,
                                'qty1': 0,
                                'qty2': 0,
                                'close_time': int(time.time()),
                                'close_pnl': net_pnl,
                                'close_reason': 'desync',
                                'pnl1': pnl1,
                                'pnl2': pnl2,
                                'fee1': fee1,
                                'fee2': fee2,
                                'beta_btc': close_beta,
                                'last_pvalue': close_pval,
                            })
                        
                        # Send detailed notification with CAUSE
                        done_msg = (f"⚡ <b>Pair Closed (Desync):</b> {s1}/{s2}\n"
                                    f"🔍 Cause: {desync_reason}\n\n"
                                    f"📊 Z: {close_zscore:+.2f} | β: {close_beta:.3f} | p: {close_pval:.4f}\n"
                                    f"⏳ HL: {close_hl} | Hedge: {hedge:.4f}\n"
                                    f"💵 PnL: {pnl_emoji} <b>{net_pnl:.2f} USDT</b>\n"
                                    f"   {e1} {s1}: {pnl1:+.2f} USDT\n"
                                    f"   {e2} {s2}: {pnl2:+.2f} USDT\n"
                                    f"💸 Fees: {total_fees:.4f} USDT")
                        print(done_msg.replace('<b>', '').replace('</b>', ''))
                        reply_to = await self._resolve_reply_to_message_id(pair_info)
                        await self._notify(done_msg, reply_to)
                        
                        # WAIT FOR CANDLE: pair-local same-candle guard
                        self.mark_pair_wait_for_next_candle(pair_info, reason='desync')
                        
                    except Exception as e:
                        print(f"⚠️ Desync close error for {s1}-{s2}: {e}")
                        import traceback
                        traceback.print_exc()
                        pair_info.is_trading = False
            if externally_closed_now:
                known = [x for x in externally_closed_now if x['pnl'] is not None]
                unknown_count = len(externally_closed_now) - len(known)
                total_external_pnl = sum(x['pnl'] for x in known)
                total_emoji = "🟢" if total_external_pnl >= 0 else "🔴"

                lines = [
                    "⚡ <b>Positions Closed Externally</b>",
                    f"Pairs: <b>{len(externally_closed_now)}</b>"
                ]
                if known:
                    lines.append(f"💰 Total Realized PnL: {total_emoji} <b>{total_external_pnl:+.2f} USDT</b>")
                if unknown_count:
                    lines.append(f"ℹ️ PnL unavailable for {unknown_count} pair(s).")

                for item in externally_closed_now[:12]:
                    if item['pnl'] is None:
                        lines.append(f"• {item['pair']}: n/a")
                    else:
                        e = "🟢" if item['pnl'] >= 0 else "🔴"
                        lines.append(f"• {item['pair']}: {e} {item['pnl']:+.2f} USDT")
                if len(externally_closed_now) > 12:
                    lines.append(f"... and {len(externally_closed_now) - 12} more")

                await self._notify("\n".join(lines))

        except Exception as e:
            print(f"⚠️ Leg sync error: {e}")

    async def initialize_all_symbols_data(self, target_symbols=None, concurrency=20, run_discovery=True):
        """
        Loads historical data for specified symbols with controlled concurrency.
        Prioritizes active pairs and priority pairs.
        """
        raw_symbols = target_symbols if target_symbols else list(self.all_symbols.keys())
        symbols_to_load = []
        seen_symbols = set()
        for raw_sym in raw_symbols:
            sym = str(raw_sym or '').strip().upper()
            if not sym or sym in seen_symbols:
                continue
            if sym not in self.all_symbols:
                continue
            if not _is_tradeable_usdt_symbol(sym):
                continue
            seen_symbols.add(sym)
            symbols_to_load.append(sym)
        dropped_symbols = max(0, len(raw_symbols) - len(symbols_to_load))
        if dropped_symbols:
            print(f"⏭️ Warmup skipped {dropped_symbols} invalid/unavailable symbols.")
        #print(f"Initializing history for {len(symbols_to_load)} symbols (Concurrency: {concurrency})...")
        start_time = time.time()
        slow_history: list[tuple[float, str, int]] = []
        history_errors = 0
        
        # 1. Identify priority symbols
        priority_symbols = set()
        
        # Active pairs
        for pair in self.active_pairs.values():
            priority_symbols.add(pair.symbol1)
            priority_symbols.add(pair.symbol2)
            
        # Priority file - resolve relative to script directory
        priority_file_path = getattr(self.config, 'priority_pairs_file', 'best_pairs.json')
        if priority_file_path and not os.path.isabs(priority_file_path):
             script_dir = os.path.dirname(os.path.abspath(__file__))
             priority_file_path = os.path.join(script_dir, priority_file_path)
             
        if priority_file_path and os.path.exists(priority_file_path):
            try:
                for s1, s2 in self._load_priority_pairs_list():
                    if s1 in symbols_to_load:
                        priority_symbols.add(s1)
                    if s2 in symbols_to_load:
                        priority_symbols.add(s2)
            except Exception:
                pass
            
        # Sort symbols: priority first, then others
        other_symbols = [s for s in symbols_to_load if s not in priority_symbols]
        sorted_symbols = list(priority_symbols) + other_symbols
        
        # CRITICAL: Always include BTCUSDT for market beta calculation.
        # Put it first so early discovery has beta data ready.
        if 'BTCUSDT' in sorted_symbols:
            sorted_symbols = ['BTCUSDT'] + [s for s in sorted_symbols if s != 'BTCUSDT']
        elif 'BTCUSDT' in self.all_symbols:
            sorted_symbols = ['BTCUSDT'] + sorted_symbols
            print("📈 Added BTCUSDT for market beta calculation")
        
        print(f"Priority symbols: {len(priority_symbols)}, Others: {len(other_symbols)}")
        
        # 2. Batch processing with semaphore
        sem = asyncio.Semaphore(concurrency)
        
        async def load_safe(symbol):
            nonlocal history_errors
            async with sem:
                # Check if data exists AND has enough candles for analysis
                has_enough = (
                    symbol in self.all_data
                    and len(self.all_data[symbol].ts) >= self.min_data_points
                )
                if not has_enough:
                    self.all_data[symbol] = Data(maxlen=self.max_len)
                    sym_t0 = time.time()
                    await self._initialize_history(symbol)
                    elapsed = time.time() - sym_t0
                    candles = len(self.all_data[symbol].ts) if symbol in self.all_data else 0
                    if candles <= 0:
                        history_errors += 1
                    slow_history.append((elapsed, symbol, candles))
                
        batch_size = max(20, int(concurrency) * 3)
        loaded_count = 0
        early_discovery_done = False
        early_threshold = max(30, min(120, len(sorted_symbols) // 3 if sorted_symbols else 30))

        for i in range(0, len(sorted_symbols), batch_size):
            batch_t0 = time.time()
            batch = sorted_symbols[i:i + batch_size]
            tasks = [load_safe(s) for s in batch]
            if tasks:
                await asyncio.gather(*tasks)
            loaded_count += len(batch)
            batch_elapsed = time.time() - batch_t0
            print(
                f"Warmup batch {i // batch_size + 1}: "
                f"{loaded_count}/{len(sorted_symbols)} symbols, "
                f"batch_time={batch_elapsed:.2f}s"
            )

            if run_discovery and not early_discovery_done and loaded_count >= early_threshold:
                early_discovery_done = True
                print(
                    f"Early Discovery after warmup batch ({loaded_count}/{len(sorted_symbols)} symbols) "
                    f"[priority-only fast pass; full scan runs after warmup]..."
                )
                try:
                    await self._discover_new_pairs(priority_only=True)
                except Exception as e:
                    print(f"Early Discovery failed (continuing): {e}")
        
        elapsed = time.time() - start_time
        print(f"✅ History initialization finished in {elapsed:.2f}s.")
        if slow_history:
            slow_history.sort(key=lambda x: x[0], reverse=True)
            print("Top slow history symbols:")
            for sec, sym, candles in slow_history[:10]:
                print(f"  {sym}: {sec:.2f}s ({candles} candles)")
            ready = sum(1 for _, _, candles in slow_history if candles >= self.min_data_points)
            print(
                f"History summary: loaded={len(slow_history)}, "
                f"ready_for_analysis={ready}, errors_or_empty={history_errors}"
            )
        
        # CRITICAL: Ensure BTCUSDT is loaded for beta calculation
        if 'BTCUSDT' not in self.all_data:
            print("📈 Loading BTCUSDT for beta calculation...")
            self.all_data['BTCUSDT'] = Data(maxlen=self.max_len)
            await self._initialize_history('BTCUSDT')
        
        btc_len = len(self.all_data.get('BTCUSDT', Data()).close) if 'BTCUSDT' in self.all_data else 0
        print(f"📊 BTCUSDT data: {btc_len} candles loaded")
        
        # Optional heavy step: full discovery. Can be deferred for quick startup.
        if run_discovery:
            print("🔍 Running initial Discovery...")
            d0 = time.time()
            await self._discover_new_pairs()
            print(f"Initial Discovery finished in {time.time() - d0:.2f}s")
        
        
        # Force run analysis for test_mode
        test_mode = getattr(self.config, 'test_mode', False)
        if isinstance(test_mode, str):
            test_mode = test_mode.lower() in ('true', '1', 'yes')
        if test_mode and self.active_pairs:
            print("🧪 TEST MODE: Force running initial analysis...")
            await asyncio.sleep(1)  # Small delay to ensure data is ready
            # Trigger analysis for each test pair
            for pair_set, pair_info in list(self.active_pairs.items()):
                if pair_info.position_status == 0:
                    # Trigger analysis for this pair
                    s1, s2 = pair_info.symbol1, pair_info.symbol2
                    if s1 in self.all_data and s2 in self.all_data:
                        print(f"  Analyzing {s1}-{s2}...")
                        await self._check_signals_for_active_pairs(s1)

    def start_background_warmup(self, target_symbols, concurrency=20, run_discovery=True):
        """Start history warmup (and optional discovery) in background (non-blocking)."""
        if self._warmup_task is not None and not self._warmup_task.done():
            return
        self._warmup_task = self.loop.create_task(
            self.initialize_all_symbols_data(target_symbols, concurrency=concurrency, run_discovery=run_discovery)
        )
        def _warmup_done(task):
            try:
                exc = task.exception()
                if exc:
                    print(f"⚠️ Background warmup failed: {type(exc).__name__}: {exc}")
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"⚠️ Warmup done-callback error: {e}")
        self._warmup_task.add_done_callback(_warmup_done)

    async def add_kline_main(self, kline_data):
        """
        Processes kline from MAIN timeframe (discovery + validation).
        Triggers full analysis including cointegration tests.
        """
        symbol = kline_data['s']
        
        if symbol not in self.all_data:
            self.all_data[symbol] = Data(maxlen=self.max_len)
            await self._initialize_history(symbol)

        added = self.all_data[symbol].add_kline(
            kline_data['t'],
            kline_data['o'],
            kline_data['h'],
            kline_data['l'],
            kline_data['c']
        )

        if added:
            self._progress_kline_added += 1
            self._progress_last_symbol = symbol
            try:
                self._progress_last_kline_open_ts = int(kline_data.get('t', 0) or 0)
            except Exception:
                self._progress_last_kline_open_ts = 0
            # Full analysis: discovery + signal check
            self.loop.create_task(self.run_analysis(symbol))
            self._log_main_tf_progress()

    def _log_main_tf_progress(self):
        """Periodic heartbeat for closed-candle processing progress."""
        now = time.time()
        # Keep it low-noise and adaptive to timeframe.
        tf = str(getattr(self, 'timeframe', '1h') or '1h').strip().lower()
        if tf.endswith('d'):
            min_interval_sec = 1800  # 30 min for daily TF
        elif tf.endswith('4h'):
            min_interval_sec = 600   # 10 min for 4h TF
        elif tf.endswith('h'):
            min_interval_sec = 300   # 5 min for hourly TFs
        else:
            min_interval_sec = 60    # keep 1 min for minute-based TFs
        if now - self._progress_last_log_ts < min_interval_sec:
            return
        self._progress_last_log_ts = now
        last_pair = self._progress_last_pair or "-"
        last_pair_ts = self._progress_last_pair_ts or 0
        last_symbol = self._progress_last_symbol or "-"
        last_kline_ts = self._progress_last_kline_open_ts or 0
        print(
            f"🫀 Main-TF progress: closed_klines={self._progress_kline_added}/60s, "
            f"analysis_runs={self._progress_analysis_runs}/60s, "
            f"coint_evals={self._progress_coint_evals}/60s, "
            f"last_symbol={last_symbol}, last_kline_open_ts={last_kline_ts}, "
            f"last_pair={last_pair}, last_pair_ts={last_pair_ts}"
        )
        self._progress_kline_added = 0
        self._progress_analysis_runs = 0
        self._progress_coint_evals = 0

    def _pair_key(self, symbol1: str, symbol2: str) -> frozenset:
        return frozenset([str(symbol1 or '').strip().upper(), str(symbol2 or '').strip().upper()])

    def _record_discovery_reject(self, symbol1: str, symbol2: str, reason: str, now_ts: float | None = None):
        """Track repeated discovery rejects and add temporary cooldown for noisy pairs."""
        now_ts = now_ts if now_ts is not None else time.time()
        pair_key = self._pair_key(symbol1, symbol2)
        reason_key = str(reason or 'rejected').strip().lower()
        if not reason_key:
            reason_key = 'rejected'

        self._diag_reject_reason_counts[reason_key] = self._diag_reject_reason_counts.get(reason_key, 0) + 1

        prev = self._pair_reject_state.get(pair_key)
        if prev and str(prev.get('reason', '')).strip().lower() == reason_key:
            next_count = int(prev.get('count', 0) or 0) + 1
            blocked_until = float(prev.get('blocked_until', 0.0) or 0.0)
        else:
            next_count = 1
            blocked_until = 0.0

        repeats_to_block = int(getattr(self.config, 'discovery_reject_repeat_count', 3) or 3)
        cooldown_hours = float(getattr(self.config, 'discovery_reject_cooldown_hours', 12.0) or 12.0)
        cooldown_sec = max(0.0, cooldown_hours * 3600.0)

        if repeats_to_block > 0 and cooldown_sec > 0 and next_count >= repeats_to_block:
            blocked_until = max(blocked_until, now_ts + cooldown_sec)
            print(
                f"⏳ Discovery anti-repeat cooldown: {symbol1}-{symbol2} "
                f"reason={reason_key}, repeats={next_count}, cooldown={int(cooldown_sec // 3600)}h"
            )

        self._pair_reject_state[pair_key] = {
            'reason': reason_key,
            'count': next_count,
            'updated_at': float(now_ts),
            'blocked_until': float(blocked_until),
        }

    def _reject_block_info(self, pair_key: frozenset, now_ts: float | None = None) -> tuple[bool, str, int]:
        now_ts = now_ts if now_ts is not None else time.time()
        state = self._pair_reject_state.get(pair_key)
        if not state:
            return False, '', 0
        blocked_until = float(state.get('blocked_until', 0.0) or 0.0)
        if blocked_until <= now_ts:
            return False, '', 0
        reason = str(state.get('reason', 'rejected') or 'rejected')
        left_sec = max(0, int(blocked_until - now_ts))
        return True, reason, left_sec

    def _clear_reject_state(self, symbol1: str, symbol2: str):
        self._pair_reject_state.pop(self._pair_key(symbol1, symbol2), None)

    def _cleanup_reject_state(self, now_ts: float | None = None):
        """Prune old reject-cache entries to keep memory bounded."""
        if not self._pair_reject_state:
            return
        now_ts = now_ts if now_ts is not None else time.time()
        ttl_hours = float(getattr(self.config, 'discovery_reject_state_ttl_hours', 72.0) or 72.0)
        ttl_sec = max(3600.0, ttl_hours * 3600.0)
        for key, state in list(self._pair_reject_state.items()):
            updated_at = float(state.get('updated_at', 0.0) or 0.0)
            blocked_until = float(state.get('blocked_until', 0.0) or 0.0)
            is_stale = (now_ts - updated_at) > ttl_sec
            if is_stale and blocked_until <= now_ts:
                self._pair_reject_state.pop(key, None)

    async def _maybe_trigger_stagnation_full_scan(self, now_ts: float):
        """Run one deeper full-scan only after long stagnation and no open positions."""
        if not self._initialized:
            return
        if self.count_active_positions() > 0:
            return
        if self._discovery_task is not None and not self._discovery_task.done():
            return

        watch_hours = float(getattr(self.config, 'stagnation_watchdog_hours', 12.0) or 12.0)
        if watch_hours <= 0:
            return
        watch_sec = watch_hours * 3600.0
        anchor_ts = max(float(getattr(self, '_last_pair_found_ts', 0.0) or 0.0), float(self._init_complete_time or 0.0))
        if anchor_ts <= 0:
            return
        if (now_ts - anchor_ts) < watch_sec:
            return

        cooldown_sec = int(
            getattr(self.config, 'stagnation_watchdog_cooldown_sec', max(3600, int(watch_sec))) or max(3600, int(watch_sec))
        )
        if (now_ts - float(self._stagnation_last_full_scan_ts or 0.0)) < cooldown_sec:
            return

        ready_symbols = sum(1 for d in self.all_data.values() if len(d.ts) >= self.min_data_points)
        if ready_symbols < max(20, int(len(self.all_symbols) * 0.1)):
            return

        idle_pairs = sum(1 for pi in self.active_pairs.values() if pi.position_status == 0 and not pi.is_trading)
        self._stagnation_last_full_scan_ts = now_ts
        self._last_discovery_time = now_ts
        print(
            f"🧭 Stagnation watchdog: no new pairs for {int((now_ts - anchor_ts) // 3600)}h, "
            f"open=0, idle={idle_pairs}. Triggering deep full discovery."
        )
        await self._notify(
            "🧭 <b>Stagnation Watchdog</b>\n"
            f"No new pairs for ~{int((now_ts - anchor_ts) // 3600)}h.\n"
            f"Open: 0 | Idle: {idle_pairs}\n"
            "Action: run one deep full-scan (temporary, low-frequency)."
        )
        self._discovery_task = self.loop.create_task(self._discover_new_pairs(priority_only=False, force_full_scan=True))

    async def _maybe_send_discovery_diagnostics(self, now_ts: float):
        """Rare TG heartbeat with discovery health and reject reasons."""
        report_sec = int(getattr(self.config, 'discovery_diag_interval_sec', 43200) or 43200)
        if report_sec <= 0:
            return
        last_ts = float(getattr(self, '_diag_last_report_ts', 0.0) or 0.0)
        if last_ts > 0 and (now_ts - last_ts) < report_sec:
            return

        open_count = self.count_active_positions()
        idle_count = sum(1 for pi in self.active_pairs.values() if pi.position_status == 0 and not pi.is_trading)
        last_found = float(getattr(self, '_last_pair_found_ts', 0.0) or 0.0)
        no_new_hours = (now_ts - last_found) / 3600.0 if last_found > 0 else 0.0
        top_rejects = sorted(self._diag_reject_reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        rejects_line = ", ".join(f"{k}:{v}" for k, v in top_rejects) if top_rejects else "none"
        window_h = (now_ts - last_ts) / 3600.0 if last_ts > 0 else 0.0

        msg = (
            "📊 <b>Discovery Health</b>\n"
            f"Window: {window_h:.1f}h\n"
            f"Discovery runs: {int(self._diag_discovery_runs)}\n"
            f"New pairs found: {int(self._diag_discovery_new_pairs)}\n"
            f"Now: open={open_count}, idle={idle_count}\n"
            f"No new pairs for: {no_new_hours:.1f}h\n"
            f"Top reject reasons: {rejects_line}"
        )
        await self._notify(msg)

        self._diag_last_report_ts = now_ts
        self._diag_discovery_runs = 0
        self._diag_discovery_new_pairs = 0
        self._diag_reject_reason_counts.clear()

    async def _discovery_health_loop(self):
        """Low-frequency housekeeping for stagnation safety and diagnostics."""
        while True:
            try:
                await asyncio.sleep(30)
                now_ts = time.time()
                self._cleanup_reject_state(now_ts)
                await self._maybe_trigger_stagnation_full_scan(now_ts)
                await self._maybe_send_discovery_diagnostics(now_ts)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️ Discovery health loop error (continuing): {e}")

    # Legacy method for backward compatibility (single TF mode)
    async def add_kline(self, kline_data):
        """Legacy method - calls add_kline_main for backward compatibility."""
        await self.add_kline_main(kline_data)

    async def _initialize_history(self, symbol):
        """
        Loads historical data to initialize deques.
        """
        #print(f"Initializing history for {symbol}...")
        symbol = str(symbol or '').strip().upper()
        if not _is_tradeable_usdt_symbol(symbol) or symbol not in self.all_symbols:
            print(f"⏭️ Skip history init for invalid symbol: {symbol}")
            self.all_data.pop(symbol, None)
            return
        try:
            t0 = time.time()
            klines = await self.client.klines(symbol, self.timeframe, limit=self.max_len)
            data = self.all_data[symbol]
            for k in klines:
                data.add_kline(k[0], k[1], k[2], k[3], k[4])
            elapsed = time.time() - t0
            if elapsed >= 5.0:
                print(f"Slow history fetch: {symbol} took {elapsed:.2f}s ({len(data.ts)} candles)")
            #print(f"History for {symbol} initialized with {len(data.ts)} candles.")
        except Exception as e:
            print(f"Error initializing history for {symbol}: {e}")
            if symbol in self.all_data:
                del self.all_data[symbol]

    async def run_analysis(self, updated_symbol: str):
        """
        Runs analysis for pairs containing the updated symbol.
        """
        self._progress_analysis_runs += 1
        # 1. Check signals for active pairs
        await self._check_signals_for_active_pairs(updated_symbol)

        # 2. Periodically run discovery (every 10 minutes)
        #    Skip if background warmup still loading -- discovery would see incomplete ready_set.
        now = time.time()
        warmup_running = self._warmup_task is not None and not self._warmup_task.done()
        if now - self._last_discovery_time > 600 and not warmup_running:
            if self._discovery_task is None or self._discovery_task.done():
                self._last_discovery_time = now
                self._discovery_task = self.loop.create_task(self._discover_new_pairs())


    async def _check_signals_for_active_pairs(self, updated_symbol: str):
        """
        Checks for trading signals and handles pair rotation.
        """
        symbol_pairs = list(self._pairs_with_symbol(updated_symbol))
        if not symbol_pairs:
            return

        for pair_info in symbol_pairs:
            if pair_info.is_trading:
                continue
            
            s1, s2 = pair_info.symbol1, pair_info.symbol2
            pair_set = frozenset([s1, s2])
            # Pair can be removed concurrently during this loop.
            if pair_set not in self.active_pairs:
                continue
            
            if s1 not in self.all_data or s2 not in self.all_data:
                continue
            
            data1 = self.all_data[s1]
            data2 = self.all_data[s2]

            if len(data1.close) < self.min_data_points or len(data2.close) < self.min_data_points:
                continue

            if data1.ts[-1] != data2.ts[-1]:
                continue

            # Evaluate coint metrics at most once per common closed candle for this pair.
            # Without this guard, repeated ad-hoc checks on the same candle can distort
            # coint_streak_bars semantics ("consecutive closed candles").
            common_close_ts = int(data1.ts[-1])
            last_eval_ts = int(getattr(pair_info, '_last_coint_eval_ts', 0) or 0)
            if common_close_ts <= last_eval_ts:
                continue
            pair_info._last_coint_eval_ts = common_close_ts
            self._progress_coint_evals += 1
            self._progress_last_pair = f"{s1}-{s2}"
            self._progress_last_pair_ts = common_close_ts

            log_prices1 = np.log(list(data1.close)[-self.min_data_points:])
            log_prices2 = np.log(list(data2.close)[-self.min_data_points:])

            # Dynamic recalculation of cointegration (with configurable p-value threshold)
            p_value_threshold = getattr(self.config, 'p_value_threshold', 0.05) or 0.05
            flag, hedge, hl, pval = utils.calculate_cointegration(log_prices1, log_prices2, p_value_threshold, strict_hl=False)
            if flag == 1:
                pair_info.coint_streak_bars = int(getattr(pair_info, 'coint_streak_bars', 0) or 0) + 1
                pair_info.coint_broken_count = 0
            else:
                pair_info.coint_streak_bars = 0
                if pair_info.position_status != 0:
                    pair_info.coint_broken_count = int(getattr(pair_info, 'coint_broken_count', 0) or 0) + 1
                else:
                    pair_info.coint_broken_count = 0

            # === MARKET NEUTRALITY CHECK ===
            # Calculate beta to BTC to ensure pair is market-neutral
            beta_btc = np.nan
            beta_threshold = getattr(self.config, 'beta_threshold', 0.11) or 0.11
            
            if flag == 1 and 'BTCUSDT' in self.all_data:
                btc_data = self.all_data['BTCUSDT']
                if len(btc_data.close) >= self.min_data_points:
                    log_btc = np.log(list(btc_data.close)[-self.min_data_points:])
                    # Spread returns = d(log1) - hedge * d(log2)
                    spread_returns = np.diff(log_prices1) - hedge * np.diff(log_prices2)
                    btc_returns = np.diff(log_btc)
                    beta_btc = utils.calculate_pair_beta(spread_returns, btc_returns)
                    
                    if not np.isnan(beta_btc) and abs(beta_btc) >= beta_threshold:
                        # Only reject/set flag=0 if the pair is IDLE (no open position)
                        # For active trades, we let _check_realtime_exit handle beta drift
                        if pair_info.position_status == 0:
                            print(f"⚠️ {s1}-{s2} rejected: beta_btc={beta_btc:.3f} >= {beta_threshold} (not market-neutral)")
                            flag = 0  # Mark as not cointegrated (only for idle pairs)
                        else:
                            # For trading pairs, just log warning - RT exit will handle PnL-based closure
                            print(f"🛡️ {s1}-{s2} beta drift detected: |beta|={abs(beta_btc):.3f} (above limit {beta_threshold}). Handling via RT monitoring.")
            
            # === HEDGE RATIO BOUNDS CHECK ===
            # Reject pairs with |hedge| outside configured bounds (too unbalanced positions)
            if flag == 1:
                hedge_min = getattr(self.config, 'hedge_min', 0.3) or 0.3
                hedge_max = getattr(self.config, 'hedge_max', 3.0) or 3.0
                abs_hedge = abs(hedge) if not np.isnan(hedge) else 0.0
                if abs_hedge < hedge_min or abs_hedge > hedge_max:
                    if pair_info.position_status == 0:
                        print(f"⚠️ {s1}-{s2} rejected: |hedge|={abs_hedge:.4f} outside [{hedge_min}, {hedge_max}] (positions would be unbalanced)")
                        flag = 0
                    else:
                        print(f"⚠️ {s1}-{s2} hedge drift: |hedge|={abs_hedge:.4f} outside [{hedge_min}, {hedge_max}]")
            
            # Store beta for display (ALWAYS, even if rejected)
            pair_info.beta_btc = beta_btc if not np.isnan(beta_btc) else 0.0
            pair_info.last_pvalue = pval if not np.isnan(pval) else 0.0
            # Persist to DB for restart recovery & analysis
            if pair_info.db_id and pair_info.position_status != 0:
                try:
                    await db.update_pair({
                        'id': pair_info.db_id,
                        'beta_btc': pair_info.beta_btc,
                        'last_pvalue': pair_info.last_pvalue
                    })
                except Exception as _db_e:
                    print(f"⚠️ DB beta/pval save failed: {_db_e}")
            # Pair rotation: if cointegration breaks
            if flag == 0:
                print(f"⚠️ Pair {s1}-{s2} correlation broken (pval: {pval:.4f}, HL: {hl}). Removing...")
                
                if pair_info.position_status != 0:
                    broken_bars = int(getattr(pair_info, 'coint_broken_count', 0) or 0)
                    grace_bars = max(0, int(getattr(self.config, 'coint_broken_grace_bars', 0) or 0))
                    if grace_bars > 0 and broken_bars <= grace_bars:
                        print(
                            f"⏳ GRACE PERIOD (coint bars): Skipping broken_coint close for {s1}-{s2} "
                            f"(broken bars {broken_bars}/{grace_bars})"
                        )
                        continue
                    leg1_open = abs(float(self._exchange_positions_cache.get(s1, 0.0) or 0.0)) > 0
                    leg2_open = abs(float(self._exchange_positions_cache.get(s2, 0.0) or 0.0)) > 0
                    if not leg1_open or not leg2_open:
                        print(
                            f"Skipping broken_coint close for {s1}-{s2}: "
                            f"exchange legs present={int(leg1_open)}/{int(leg2_open)}. "
                            "Waiting for sync/external-close flow."
                        )
                        continue

                    # GRACE PERIOD 1: Skip broken_coint closures during warmup
                    # After bot restart, data may not be fully loaded yet
                    grace_elapsed = time.time() - self._init_complete_time
                    if grace_elapsed < self._broken_coint_grace_sec:
                        print(f"⏳ GRACE PERIOD (init): Skipping broken_coint close for {s1}-{s2} (init {grace_elapsed:.0f}s ago, need {self._broken_coint_grace_sec}s)")
                        continue
                    
                    # GRACE PERIOD 2: Skip broken_coint closures for freshly opened trades
                    # Cointegration re-test with slightly different data can give false negatives
                    trade_open_time = getattr(pair_info, '_trade_open_time', 0)
                    trade_age = time.time() - trade_open_time
                    if trade_open_time > 0 and trade_age < 60:
                        print(f"⏳ GRACE PERIOD (trade): Skipping broken_coint close for {s1}-{s2} (trade opened {trade_age:.0f}s ago, need 60s)")
                        continue
                    
                    print(f"🚨 Broken Correlation on {s1}-{s2} (Pval: {pval:.3f}). Force Closing Position!")
                    # Don't send notification here - _execute_trade will send full close message with PnL
                    if not getattr(pair_info, 'tg_message_id', 0):
                        restored_reply = await self._resolve_reply_to_message_id(pair_info)
                        if restored_reply:
                            print(f"ℹ️ Restored missing tg_message_id for {s1}-{s2}: {restored_reply}")
                        else:
                            print(f"⚠️ tg_message_id missing for {s1}-{s2}; close alert may be non-threaded")
                    pair_info.close_handled = True
                    pair_info.is_trading = True
                    
                    # CRITICAL: Await close before removing from active_pairs to avoid zombie positions
                    try:
                        await self._execute_trade(pair_info, 0, close_reason='broken_coint')
                    except Exception as e:
                        print(f"❌ Failed to close broken pair {s1}-{s2}: {e}. Keeping in active list to retry.")
                        continue # Do not delete pair if close failed

                if pair_info.db_id:
                    reason = f"Broken cointegration (flag={flag}, p={pval:.4f})"
                    # CRITICAL: Await this to prevent pool exhaustion (was create_task)
                    try:
                        await db.log_pair_history_event(
                            symbol1=s1,
                            symbol2=s2,
                            event_type='BROKEN',
                            timestamp_ms=int(time.time() * 1000),
                            hedge_ratio=hedge,
                            half_life=hl,
                            reason=reason,
                            pair_id=pair_info.db_id,
                            beta_btc=pair_info.beta_btc,
                            pvalue=pval,
                        )
                        await db.archive_pair(pair_info.db_id, reason='broken_coint')
                    except Exception as e:
                        print(f"⚠️ Failed to update DB content for broken pair {s1}-{s2}: {e}")
                
                if pair_set in self.active_pairs:
                    self._unregister_pair(pair_info)
                    del self.active_pairs[pair_set]
                continue

            # Update parameters
            pair_info.hedge_ratio = hedge
            pair_info.half_life = hl
            self._update_quality_score_cache(pair_info)
            
            if pair_info.db_id:
                # CRITICAL: Await DB update to prevent pool exhaustion (was create_task)
                await db.update_pair({
                    'id': pair_info.db_id,
                    'hedge_ratio': hedge,
                    'half_life': hl
                })

            spread = log_prices1 - pair_info.hedge_ratio * log_prices2
            z_score = utils.calculate_z_last(spread)
            if z_score is None:
                continue
            
            pair_info.last_z_score = z_score

            # Circuit Breaker Logic (candle-close backup — primary is in on_ticker_update)
            if pair_info.position_status != 0 and pair_info.entry_price1 > 0 and pair_info.entry_price2 > 0:
                current_price1 = list(data1.close)[-1]
                current_price2 = list(data2.close)[-1]

                # Use EXCHANGE PnL (source of truth)
                total_pnl = self._get_exchange_pair_pnl(pair_info, current_price1, current_price2)

                notional = (pair_info.entry_price1 * pair_info.qty1) + (pair_info.entry_price2 * pair_info.qty2)
                leverage = self.config.leverage if self.config and self.config.leverage else 20
                margin = notional / leverage  # Actual deployed capital
                circuit_breaker_pct = getattr(self.config, 'circuit_breaker_pct', 0.20) or 0.20

                if notional > 0:
                    roi_notional = total_pnl / notional
                    if roi_notional < -circuit_breaker_pct:
                        roi_margin = total_pnl / margin if margin > 0 else 0
                        cb_msg = (f"🚨 <b>CIRCUIT BREAKER TRIGGERED</b> on {s1}-{s2}!\n"
                                  f"Loss: {roi_notional*100:.2f}% of notional ({total_pnl:.2f} USDT)\n"
                                  f"Margin: {roi_margin*100:.2f}% | Leverage: {leverage}x\n"
                                  f"Force Closing...")
                        print(cb_msg)
                        reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                        await self._notify(cb_msg, reply_to)
                        pair_info.close_handled = True
                        pair_info.is_trading = True
                        await self._execute_trade(pair_info, 0, close_reason='circuit')
                        continue
                
                # === BETA DRIFT MONITORING (candle-close backup — primary is in on_ticker_update) ===
                # Check if open position has become correlated with market
                if pair_info.position_status != 0 and pair_info.beta_btc != 0:
                    # Respect grace period (same as primary RT check)
                    trade_open_time = getattr(pair_info, '_trade_open_time', 0)
                    if trade_open_time > 0 and time.time() - trade_open_time < 120:
                        pass  # Too early — beta not yet stable
                    else:
                        beta_alert_threshold = getattr(self.config, 'beta_alert_threshold', 0.15) or 0.15
                        
                        if abs(pair_info.beta_btc) >= beta_alert_threshold:
                            # Use EXCHANGE PnL (source of truth)
                            total_pnl = self._get_exchange_pair_pnl(pair_info, current_price1, current_price2)
                            
                            if total_pnl > 0:
                                # Positive PnL - auto close
                                pair_info._beta_at_trigger = pair_info.beta_btc
                                beta_msg = (f"⚠️ <b>BETA DRIFT</b> on {s1}-{s2}!\n"
                                            f"Beta: {pair_info.beta_btc:.3f} (threshold: {beta_alert_threshold})\n"
                                            f"PnL: +{total_pnl:.2f} USDT. Auto-closing...")
                                print(beta_msg)
                                await self._notify(beta_msg)
                                pair_info.close_handled = True
                                pair_info.is_trading = True
                                await self._execute_trade(pair_info, 0, close_reason='beta_drift')
                                continue
                            else:
                                # Negative PnL - notify user (but don't close)
                                beta_warn = (f"⚠️ <b>BETA DRIFT WARNING</b> on {s1}-{s2}!\n"
                                             f"Beta: {pair_info.beta_btc:.3f} (threshold: {beta_alert_threshold})\n"
                                             f"PnL: {total_pnl:.2f} USDT. Consider manual close.")
                                print(beta_warn)
                                await self._notify(beta_warn)
                                # Don't continue - let position stay open

                if pair_info.position_status != 0:
                    due_time_exit, hold_bars, hold_limit = self._time_exit_due(pair_info)
                    if due_time_exit:
                        print(
                            f"⏱️ TIME EXIT (candle backup) on {s1}-{s2}. "
                            f"hold_bars={hold_bars} >= limit={hold_limit}. Closing..."
                        )
                        pair_info.close_handled = True
                        pair_info.is_trading = True
                        await self._execute_trade(pair_info, 0, close_reason='time_exit')
                        continue

                z_entry = self.config.z_entry if self.config and self.config.z_entry else 1.9
                z_exit = self.config.z_exit if self.config and self.config.z_exit is not None else 0.0
                z_stop = self.config.z_stop if self.config and self.config.z_stop else 4.0
                
                # Test mode flag
                test_mode = getattr(self.config, 'test_mode', False)
                if test_mode and isinstance(test_mode, str):
                    test_mode = test_mode.lower() in ('true', '1', 'yes')
                
                # Signal logic
                if pair_info.position_status == 0:
                    # Pair-local same-candle guard.
                    if self._is_pair_reentry_blocked_same_candle(pair_info):
                        continue
                    if getattr(pair_info, '_wait_for_candle', False):
                        pair_info._wait_for_candle = False
                        print(f"✅ {s1}-{s2}: New candle closed, pair eligible for re-entry")
                    
                    # Check position limits before opening
                    if not self.can_open_new_position(s1, s2):
                        continue
                    
                    # Test mode: force open without strict signal window (for sandbox checks).
                    if test_mode:
                        test_direction = 1 if z_score <= 0 else -1
                        print(f"🧪 TEST MODE: Force opening {s1}-{s2} (z={z_score:.2f}, dir={'LONG' if test_direction == 1 else 'SHORT'})")
                        pair_info.is_trading = True
                        self.loop.create_task(self._execute_trade(pair_info, test_direction))
                        continue

                    # Live fallback entry on candle-close.
                    # Protects from missing entries if markPrice WS is unstable.
                    if abs(z_score) >= z_entry and abs(z_score) < getattr(self.config, 'z_entry_max', 2.5):
                        # Queue as pending candidate and let ranked confirmation loop open top pairs.
                        pair_info.pending_signal = z_score
                        # Make candle fallback eligible immediately in confirmation loop.
                        pair_info.pending_since = time.time() - max(1, int(getattr(self.config, 'signal_confirm_sec', 10) or 10))
                        pair_info.pending_source = 'candle'
                        print(f"⚡ CANDLE SIGNAL queued {s1}-{s2}: Z={z_score:.2f} (will be ranked against other candidates)")
                        continue
                
                elif pair_info.position_status == 1: # Long spread
                    # Candle-close Z-score exit (BACKUP — primary is in on_ticker_update)
                    if z_score >= z_exit:
                        print(f"💰 TAKE PROFIT (Long) on {s1}-{s2}. Z: {z_score:.2f} >= {z_exit}. Closing...")
                        pair_info.close_handled = True
                        pair_info.is_trading = True
                        await self._execute_trade(pair_info, 0, close_reason='z_tp')
                    elif z_score <= -z_stop:
                        print(f"🛑 STOP LOSS (Long) on {s1}-{s2}. Z: {z_score:.2f} <= -{z_stop}. Closing...")
                        pair_info.close_handled = True
                        pair_info.is_trading = True
                        await self._execute_trade(pair_info, 0, close_reason='z_sl')

                elif pair_info.position_status == -1: # Short spread
                    if z_score <= -z_exit:
                        print(f"💰 TAKE PROFIT (Short) on {s1}-{s2}. Z: {z_score:.2f} <= {-z_exit}. Closing...")
                        pair_info.close_handled = True
                        pair_info.is_trading = True
                        await self._execute_trade(pair_info, 0, close_reason='z_tp')
                    elif z_score >= z_stop:
                        print(f"🛑 STOP LOSS (Short) on {s1}-{s2}. Z: {z_score:.2f} >= {z_stop}. Closing...")
                        pair_info.close_handled = True
                        pair_info.is_trading = True
                        await self._execute_trade(pair_info, 0, close_reason='z_sl')

    async def _discover_new_pairs(self, priority_only: bool = False, force_full_scan: bool = False):
        """
        Finds new cointegrated pairs using parallel processing.
        """
        best_pairs_only = self._best_pairs_only_enabled()
        effective_priority_only = priority_only or best_pairs_only
        if best_pairs_only:
            mode = "BEST-PAIRS-ONLY"
        else:
            mode = "PRIORITY-ONLY" if priority_only else ("FULL-DEEP" if force_full_scan else "FULL")
        print(f"Starting discovery process for new cointegrated pairs ({mode}, PARALLEL)...")
        start_time = time.time()
        self._diag_discovery_runs += 1
        self._cleanup_reject_state(start_time)
        
        ready_symbols = []
        data_snapshot = {}
        
        for s, data in self.all_data.items():
            if len(data.ts) >= self.min_data_points:
                ready_symbols.append(s)
                prices = list(data.close)[-self.min_data_points:]
                data_snapshot[s] = np.log(prices)
        
        if len(ready_symbols) < 2:
            print("Not enough symbols with sufficient data to find pairs.")
            return

        # Early pre-filter by minNotional feasibility (before cointegration search).
        # If a symbol cannot fit into pair budget even with allowed bump, any pair with
        # this symbol will fail at execution stage. Drop it early to save CPU.
        try:
            capital = self.config.capital if self.config and self.config.capital else 1000.0
            max_notional_pct = self.config.max_notional_pct if self.config and self.config.max_notional_pct else 0.1
            max_order_bump = getattr(self.config, 'max_order_bump', 1.5) or 1.5
            hedge_min = getattr(self.config, 'hedge_min', 0.3) or 0.3
            hedge_max = getattr(self.config, 'hedge_max', 3.0) or 3.0

            pair_budget = float(capital) * float(max_notional_pct)
            print(
                f"Discovery prefilter params: capital={float(capital):.2f}, "
                f"max_notional_pct={float(max_notional_pct):.4f}, "
                f"pair_budget=${pair_budget:.2f}, max_order_bump={float(max_order_bump):.2f}"
            )
            # Maximum share one leg can receive under configured hedge bounds.
            max_leg_share = max(
                1.0 / (1.0 + abs(float(hedge_min))),
                abs(float(hedge_max)) / (1.0 + abs(float(hedge_max)))
            )
            max_leg_notional = pair_budget * max_leg_share

            filtered_ready = []
            dropped_symbols = []
            for s in ready_symbols:
                sinfo = self.all_symbols.get(s)
                # FIX: If symbol is not in all_symbols, it has no step_size/tick_size
                # and cannot be traded → drop from discovery to prevent 'Symbol info not found' errors.
                if sinfo is None:
                    dropped_symbols.append((s, 0.0, 0.0))
                    continue
                min_notional = float(getattr(sinfo, 'notional', 0.0) or 0.0)
                # Same safety margin as in execution path.
                required_min = min_notional * 1.1
                if required_min <= 0:
                    filtered_ready.append(s)
                    continue
                # At execution, trade is skipped if bump exceeds max_order_bump:
                # required_min / calc_notional > max_order_bump  -> skip
                # So minimally viable calc_notional is required_min / max_order_bump.
                min_viable_calc = required_min / float(max_order_bump)
                if max_leg_notional + 1e-9 < min_viable_calc:
                    dropped_symbols.append((s, required_min, min_viable_calc))
                else:
                    filtered_ready.append(s)

            if dropped_symbols:
                not_in_syms = [(s, r, v) for s, r, v in dropped_symbols if r == 0.0 and v == 0.0]
                notional_drops = [(s, r, v) for s, r, v in dropped_symbols if not (r == 0.0 and v == 0.0)]
                if not_in_syms:
                    print(
                        f"⛔ Discovery prefilter: dropped {len(not_in_syms)} symbols "
                        f"not found in all_symbols (no step_size/tick_size info): "
                        f"{', '.join(s for s, _, _ in not_in_syms[:10])}"
                        + (f" and {len(not_in_syms)-10} more" if len(not_in_syms) > 10 else "")
                    )
                if notional_drops:
                    print(
                        f"⛔ Discovery prefilter: dropped {len(notional_drops)} symbols by minNotional "
                        f"(max leg ${max_leg_notional:.2f}, bump<= {max_order_bump}x)"
                    )
                    for s, required_min, min_viable in notional_drops[:20]:
                        print(
                            f"  - {s}: min_required=${required_min:.2f}, "
                            f"min_viable_calc=${min_viable:.2f} > max_leg=${max_leg_notional:.2f}"
                        )
                    if len(notional_drops) > 20:
                        print(f"  ... and {len(notional_drops) - 20} more symbols")

            # Fail-safe: if filter is too aggressive, keep original universe.
            # Better to spend extra CPU than silently run with zero candidates.
            min_symbols_after_filter = max(10, int(len(ready_symbols) * 0.10))
            if len(filtered_ready) < 2 or len(filtered_ready) < min_symbols_after_filter:
                print(
                    f"⚠️ Discovery prefilter fallback: kept original universe "
                    f"({len(filtered_ready)} after filter is too low from {len(ready_symbols)})."
                )
            else:
                ready_symbols = filtered_ready
                data_snapshot = {s: data_snapshot[s] for s in ready_symbols if s in data_snapshot}
        except Exception as prefilter_err:
            print(f"⚠️ Discovery minNotional prefilter skipped: {prefilter_err}")

        if len(ready_symbols) < 2:
            print("Not enough symbols after minNotional prefilter to find pairs.")
            return

        print(f"Analyzing {len(ready_symbols)} symbols using {self.min_data_points} candles.")
        ready_set = set(ready_symbols)
        checked_pairs = set()
        candidates_to_process = []
        now_discovery_ts = start_time
        reject_cooldown_skipped = 0
        pair_blacklist_keys = self._load_pair_blacklist_keys() if self._pair_blacklist_enabled() else set()
        if pair_blacklist_keys:
            print(f"⛔ Pair blacklist active: {len(pair_blacklist_keys)} pairs")

        # Pre-filter pairs that already exist in DB to avoid duplicate discovery noise
        # and unnecessary stats/beta calculations for known pairs.
        existing_db_keys = set()
        try:
            existing_db_keys = await db.get_active_pair_keys()
            for sym1, sym2 in existing_db_keys:
                checked_pairs.add(frozenset([sym1, sym2]))
            if existing_db_keys:
                print(f"Loaded {len(existing_db_keys)} active pairs from DB for duplicate pre-filter.")
        except Exception as e:
            print(f"⚠️ Could not preload active pair keys from DB: {e}")
        
        # --- 1. Load and process Priority Pairs ---
        priority_file_path = getattr(self.config, 'priority_pairs_file', 'best_pairs.json')
        # Handle path resolution - relative to script directory
        if priority_file_path and not os.path.isabs(priority_file_path):
             script_dir = os.path.dirname(os.path.abspath(__file__))
             priority_file_path = os.path.join(script_dir, priority_file_path)

        priority_pairs = []
        priority_existing_checked = 0
        # Symbols currently occupied by open positions (do not touch them for priority inspection).
        occupied_symbols = set()
        for _pi in self.active_pairs.values():
            if getattr(_pi, 'position_status', 0) != 0:
                occupied_symbols.add(_pi.symbol1)
                occupied_symbols.add(_pi.symbol2)

        if priority_file_path and os.path.exists(priority_file_path):
            try:
                for s1, s2 in self._load_priority_pairs_list(ready_set=ready_set):
                    # Hard rule from user: skip priority pair if any symbol is currently in an open trade.
                    if s1 in occupied_symbols or s2 in occupied_symbols:
                        continue

                    pair_set = frozenset([s1, s2])
                    pair_key = _canonical_pair_key(s1, s2)
                    if pair_key in pair_blacklist_keys:
                        continue
                    # If pair already exists in memory (idle), force immediate inspection with top priority.
                    if pair_set in self.active_pairs:
                        pi = self.active_pairs.get(pair_set)
                        if pi and getattr(pi, 'position_status', 0) == 0 and not getattr(pi, 'is_trading', False):
                            # Keep TG responsive: schedule, do not await.
                            self.loop.create_task(self._check_signals_for_active_pairs(s1))
                            priority_existing_checked += 1
                        continue

                    if pair_set not in checked_pairs:
                        # Priority pairs skip anti-repeat cooldown — always check
                        priority_pairs.append((s1, s2))
                        checked_pairs.add(pair_set)

                if priority_pairs:
                    print(f"⭐ Found {len(priority_pairs)} valid candidates from priority list.")
                    candidates_to_process.extend(priority_pairs)
                if priority_existing_checked:
                    print(f"⭐ Priority inspection queued for {priority_existing_checked} existing idle pairs.")
            except Exception as e:
                print(f"⚠️ Error loading priority pairs from {priority_file_path}: {e}")
        else:
             print(f"Info: Priority file not found at {priority_file_path}")

        # --- 2. Generate standard combinations ---
        added_count = 0
        truncated_by_cap = False
        if not effective_priority_only:
            discovery_shards = int(getattr(self.config, 'discovery_shards', 1) or 1)
            if discovery_shards < 1:
                discovery_shards = 1
            discovery_max_pairs = int(getattr(self.config, 'discovery_max_pairs_per_cycle', 0) or 0)
            if discovery_max_pairs < 0:
                discovery_max_pairs = 0

            if force_full_scan:
                deep_scan_cap = int(
                    getattr(
                        self.config,
                        'stagnation_full_scan_max_pairs',
                        max(20000, discovery_max_pairs if discovery_max_pairs > 0 else 0),
                    ) or max(20000, discovery_max_pairs if discovery_max_pairs > 0 else 0)
                )
                discovery_shards = 1
                if deep_scan_cap > 0:
                    discovery_max_pairs = max(deep_scan_cap, discovery_max_pairs)
                print(
                    f"🧭 Deep scan mode: shards=1, non-priority cap="
                    f"{discovery_max_pairs if discovery_max_pairs > 0 else 'unlimited'}"
                )

            round_idx = int(getattr(self, '_discovery_round_idx', 0) or 0)
            shard_idx = round_idx % discovery_shards
            self._discovery_round_idx = round_idx + 1
            if discovery_shards > 1:
                print(
                    f"Discovery sharding: shard {shard_idx + 1}/{discovery_shards} "
                    f"(round={round_idx + 1})"
                )
            if discovery_max_pairs > 0:
                print(f"Discovery cap: max {discovery_max_pairs} non-priority pairs this cycle")

            all_combinations = itertools.combinations(ready_symbols, 2)
            for combo_idx, p in enumerate(all_combinations):
                if discovery_shards > 1 and (combo_idx % discovery_shards) != shard_idx:
                    continue
                pair_set = frozenset(p)
                pair_key = _canonical_pair_key(p[0], p[1])
                if pair_key in pair_blacklist_keys:
                    continue
                if pair_set not in self.active_pairs and pair_set not in checked_pairs:
                    blocked, _, _ = self._reject_block_info(pair_set, now_discovery_ts)
                    if blocked:
                        reject_cooldown_skipped += 1
                        continue
                    candidates_to_process.append(p)
                    added_count += 1
                    if discovery_max_pairs > 0 and added_count >= discovery_max_pairs:
                        truncated_by_cap = True
                        break
        else:
            if best_pairs_only:
                print("Best-pairs-only discovery: skipping non-priority combinations.")
            else:
                print("Priority-only discovery: skipping non-priority pair combinations for fast startup.")
                
        total_pairs = len(candidates_to_process)
        print(f"Total pairs to check: {total_pairs} (Priority: {len(priority_pairs)}, Others: {added_count})")
        if reject_cooldown_skipped:
            print(f"⏭️ Discovery anti-repeat: skipped {reject_cooldown_skipped} pairs on temporary reject cooldown.")
        if truncated_by_cap:
            print("⚡ Discovery non-priority list truncated by cap for this cycle.")
        
        if total_pairs == 0:
            return

        worker_count = max(1, int(getattr(self.executor, "_max_workers", 1)))
        # On Windows spawn mode, repeatedly pickling large data_snapshot per chunk is expensive.
        # Keep chunk count close to worker count to minimize serialization overhead.
        if worker_count <= 1:
            chunk_size = total_pairs
        else:
            target_chunks = min(worker_count, total_pairs)
            chunk_size = max(1, int(math.ceil(total_pairs / target_chunks)))
        # Priority pairs are first in the list, so they stay in the first chunks.
        chunks = [candidates_to_process[i:i + chunk_size] for i in range(0, total_pairs, chunk_size)]
        print(f"Split into {len(chunks)} chunks for parallel processing (workers={worker_count}, chunk_size={chunk_size}).")
        
        tasks = []
        for chunk in chunks:
            task = self.loop.run_in_executor(
                self.executor, 
                utils.batch_process_pairs, 
                chunk, 
                data_snapshot, 
                self.min_data_points,
                self.config.timeframe,  # Pass timeframe for half-life limits
                getattr(self.config, 'hl_min_days', 2.0) or 2.0,
                getattr(self.config, 'hl_max_days', 5.0) or 5.0,
                getattr(self.config, 'hedge_min', 0.3) or 0.3,
                getattr(self.config, 'hedge_max', 3.0) or 3.0,
                getattr(self.config, 'p_value_threshold', 0.05) or 0.05,
            )
            tasks.append(task)
        
        results_list = []
        completed = 0
        total_chunks = len(tasks)
        for fut in asyncio.as_completed(tasks):
            chunk_result = await fut
            results_list.append(chunk_result)
            completed += 1
            if completed == 1 or completed % 5 == 0 or completed == total_chunks:
                print(f"Discovery progress: {completed}/{total_chunks} chunks completed.")
        
        existing_db_keys_canonical = set(existing_db_keys or set())
        new_pairs_count = 0
        batch_idx = 0
        for batch_results in results_list:
            batch_idx += 1
            # Yield to event loop every batch to keep TG responsive
            if batch_idx % 3 == 0:
                await asyncio.sleep(0)
            
            for res in batch_results:
                s1, s2, hedge, hl, pval = res
                try:
                    pair_set = frozenset([s1, s2])
                    canonical_key = tuple(sorted((s1, s2)))
                    # Final duplicate check before touching DB (race condition protection)
                    if pair_set in self.active_pairs:
                        print(f"  ⚠️ Skipping duplicate (race condition): {s1}-{s2}")
                        continue
                    if canonical_key in existing_db_keys_canonical:
                        print(f"  ⚠️ Skipping duplicate (already active in DB): {s1}-{s2}")
                        continue

                    new_pair = db.Pairs(
                        symbol1=s1, 
                        symbol2=s2, 
                        hedge_ratio=hedge, 
                        half_life=hl,
                        position_status=0
                    )
                    try:
                        await db.add_pair(new_pair)
                    except db.DuplicateActivePairError:
                        print(f"  ⚠️ Skipping duplicate in DB: {s1}-{s2}")
                        continue
                    existing_db_keys_canonical.add(canonical_key)
                    
                    # === BETA CHECK BEFORE ADDING TO ACTIVE PAIRS ===
                    # Calculate beta to BTC to ensure pair is market-neutral
                    beta_btc = 0.0
                    beta_threshold = getattr(self.config, 'beta_threshold', 0.11) or 0.11
                    test_mode = getattr(self.config, 'test_mode', False)
                    if isinstance(test_mode, str):
                        test_mode = test_mode.lower() in ('true', '1', 'yes')
                    btc_ready = (
                        'BTCUSDT' in self.all_data
                        and len(self.all_data['BTCUSDT'].close) >= self.min_data_points
                    )
                    if not test_mode and not btc_ready:
                        print(f"⚠️ {s1}-{s2} deferred at discovery: BTCUSDT history is not ready for beta check")
                        self._record_discovery_reject(s1, s2, 'beta_data_not_ready')
                        try:
                            await db.archive_pair(new_pair.id, reason='beta_data_not_ready')
                        except Exception:
                            pass
                        continue
                    
                    if 'BTCUSDT' in self.all_data and s1 in data_snapshot and s2 in data_snapshot:
                        try:
                            btc_data = self.all_data['BTCUSDT']
                            if len(btc_data.close) >= self.min_data_points:
                                log_btc = np.log(list(btc_data.close)[-self.min_data_points:])
                                log_p1 = data_snapshot[s1]
                                log_p2 = data_snapshot[s2]
                                # Spread returns = d(log1) - hedge * d(log2)
                                spread_returns = np.diff(log_p1) - hedge * np.diff(log_p2)
                                btc_returns = np.diff(log_btc)
                                beta_btc = utils.calculate_pair_beta(spread_returns, btc_returns)
                        except Exception as e:
                            print(f"⚠️ Beta calc error for {s1}-{s2}: {e}")
                    
                    # Reject pair if beta is too high (skip in test_mode)
                    if not test_mode and not np.isnan(beta_btc) and abs(beta_btc) >= beta_threshold:
                        print(f"⚠️ {s1}-{s2} REJECTED at discovery: |beta|={abs(beta_btc):.3f} >= {beta_threshold}")
                        self._record_discovery_reject(s1, s2, 'beta_rejected')
                        try:
                            await db.log_pair_history_event(
                                symbol1=s1,
                                symbol2=s2,
                                event_type='BETA_REJECTED',
                                timestamp_ms=int(time.time() * 1000),
                                hedge_ratio=hedge,
                                half_life=hl,
                                reason='Discovery rejected by beta threshold',
                                pair_id=new_pair.id,
                                beta_btc=beta_btc,
                                pvalue=pval,
                            )
                        except Exception:
                            pass
                        # Keep historical integrity: archive instead of delete to avoid orphan history rows
                        try:
                            await db.archive_pair(new_pair.id, reason='beta_rejected')
                        except:
                            pass
                        continue  # Skip this pair
                    elif test_mode and not np.isnan(beta_btc) and abs(beta_btc) >= beta_threshold:
                        print(f"🧪 TEST MODE: {s1}-{s2} |beta|={abs(beta_btc):.3f} >= {beta_threshold} - ALLOWED for testing")
                    
                    print(f"✅ FOUND: {s1}-{s2} | HL: {hl:.2f}, P: {pval:.4f}, Beta: {beta_btc:.3f}, Hedge: {hedge:.4f}")
                    self._clear_reject_state(s1, s2)
                    try:
                        await db.log_pair_history_event(
                            symbol1=s1,
                            symbol2=s2,
                            event_type='FOUND',
                            timestamp_ms=int(time.time() * 1000),
                            hedge_ratio=hedge,
                            half_life=hl,
                            reason='Discovery passed',
                            pair_id=new_pair.id,
                            beta_btc=beta_btc,
                            pvalue=pval,
                        )
                    except Exception as hist_err:
                        print(f"⚠️ Could not write PairHistory FOUND for {s1}-{s2}: {hist_err}")
                    
                    pair_info = PairInfo(
                        symbol1=s1, 
                        symbol2=s2, 
                        hedge_ratio=hedge, 
                        half_life=hl,
                        db_id=new_pair.id
                    )
                    pair_info.beta_btc = beta_btc
                    pair_info.discovered_at = time.time()  # Track when pair was discovered
                    pair_info.last_pvalue = pval if not np.isnan(pval) else 0.0
                    # Seed streak from loaded history so restart does not bias coint phase to 0/1 bar.
                    min_streak = int(getattr(self.config, 'coint_stability_min_bars', 2) or 2)
                    seed_lookback = max(6, min_streak + 2)
                    seeded_streak, seeded_eval_ts = self._estimate_coint_streak_from_history(
                        pair_info, max_recent_bars=seed_lookback
                    )
                    if seeded_streak > 0:
                        pair_info.coint_streak_bars = int(seeded_streak)
                    if seeded_eval_ts > 0:
                        pair_info._last_coint_eval_ts = int(seeded_eval_ts)
                    self._update_quality_score_cache(pair_info)
                    self.active_pairs[pair_set] = pair_info
                    self._register_pair(pair_info)
                    new_pairs_count += 1
                    
                    # Subscribe to real-time markPrice for this new pair
                    await self._subscribe_new_pair_realtime(s1, s2)
                    
                    # TEST MODE: Auto-open trade immediately (skip z_entry check)
                    test_mode = getattr(self.config, 'test_mode', False)
                    if isinstance(test_mode, str):
                        test_mode = test_mode.lower() in ('true', '1', 'yes')
                    
                    if test_mode and self.can_open_new_position(s1, s2):
                        # Calculate current Z-score to determine direction
                        z_stop = getattr(self.config, 'z_stop', 4.0) or 4.0
                        z_exit = getattr(self.config, 'z_exit', 0.0) or 0.0
                        
                        # Calculate Z-score from available data
                        try:
                            if s1 in self.all_data and s2 in self.all_data:
                                data1 = self.all_data[s1]
                                data2 = self.all_data[s2]
                                if len(data1.close) >= 50 and len(data2.close) >= 50:
                                    log_p1 = np.log(list(data1.close)[-50:])
                                    log_p2 = np.log(list(data2.close)[-50:])
                                    spread = log_p1 - pair_info.hedge_ratio * log_p2
                                    z_score = utils.calculate_z_last(spread)
                                    
                                    # Don't open if z_score is already at stop/exit levels
                                    if z_score is not None and abs(z_score) < z_stop and abs(z_score) > z_exit:
                                        direction = 1 if z_score < 0 else -1
                                        print(f"🧪 TEST MODE AUTO-OPEN: {s1}-{s2} Z={z_score:.2f} -> {'LONG' if direction == 1 else 'SHORT'}")
                                        pair_info.entry_z_score = z_score
                                        pair_info.is_trading = True
                                        self.loop.create_task(self._execute_trade(pair_info, direction))
                                    else:
                                        print(f"🧪 TEST: {s1}-{s2} Z={z_score:.2f} at stop/exit level, skipping auto-open")
                        except Exception as e:
                            print(f"🧪 TEST: Could not auto-open {s1}-{s2}: {e}")
                    elif self.can_open_new_position(s1, s2):
                        # IMMEDIATE ENTRY CHECK: Don't wait for 5m candle!
                        print(f"⚡ Checking immediate entry for found pair {s1}-{s2}...")
                        self.loop.create_task(self._check_signals_for_active_pairs(s1))
                        
                except Exception as e:
                    print(f"Error adding pair {s1}-{s2}: {e}")

        elapsed = time.time() - start_time
        print(f"Discovery process finished in {elapsed:.2f}s. Found {new_pairs_count} new pairs.")
        self._diag_discovery_new_pairs += int(new_pairs_count)
        if new_pairs_count > 0:
            self._last_pair_found_ts = time.time()
        # Keep priority file fresh in background (throttled internally).
        self.loop.create_task(self._refresh_best_pairs(force=False, reason='post_discovery'))

    async def _notify(self, message, reply_to_msg_id=None, reply_markup=None):
        """Sends a notification via the configured callback. Returns msg_id for reply threading."""
        if self.notify_callback:
            try:
                if isinstance(message, str):
                    message = _repair_mojibake_text(message)
                return await self.notify_callback(message, reply_to_msg_id, reply_markup)
            except Exception as e:
                print(f"Error in _notify: {e}")
        return None

    async def _resolve_reply_to_message_id(self, pair_info: PairInfo):
        """
        Best-effort resolver for TG thread id.
        Uses in-memory value first, then falls back to DB (important after restarts/sync drift).
        """
        tg_id = int(getattr(pair_info, 'tg_message_id', 0) or 0)
        if tg_id > 0:
            return tg_id

        db_id = int(getattr(pair_info, 'db_id', 0) or 0)
        if db_id <= 0:
            return None

        try:
            db_pair = await db.get_pair_by_id(db_id)
            db_tg_id = int(getattr(db_pair, 'tg_message_id', 0) or 0) if db_pair else 0
            if db_tg_id > 0:
                pair_info.tg_message_id = db_tg_id
                return db_tg_id
        except Exception as e:
            print(f"⚠️ Failed to resolve tg_message_id from DB for pair_id={db_id}: {e}")

        return None

    def _format_half_life(self, hl_bars: float) -> str:
        """Format half-life stored in bars into human-readable hours/days for the current timeframe."""
        try:
            hl_bars = float(hl_bars or 0.0)
        except Exception:
            hl_bars = 0.0
        if hl_bars <= 0:
            return "N/A"

        tf_hours = self._timeframe_seconds_local() / 3600.0
        hl_hours = hl_bars * tf_hours
        if hl_hours >= 24:
            days = int(hl_hours // 24)
            hours = int(hl_hours % 24)
            return f"{days}d {hours}h" if hours > 0 else f"{days}d"
        hours = int(hl_hours)
        mins = int((hl_hours - hours) * 60)
        return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"

    def _trade_window_start_ms(self, pair_info: PairInfo, default_lookback_sec: int = 300, buffer_sec: int = 120) -> int:
        """
        Build safer startTime for userTrades queries.
        Prefer pair open_time to avoid mixing unrelated fills.
        """
        now_ms = int(time.time() * 1000)
        open_time = int(getattr(pair_info, 'open_time', 0) or 0)
        if open_time > 0:
            start_sec = max(0, open_time - buffer_sec)
            return start_sec * 1000
        return now_ms - (default_lookback_sec * 1000)

    def _clamp01(self, x: float) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        if v < 0:
            return 0.0
        if v > 1:
            return 1.0
        return v

    def _update_quality_score_cache(self, pair_info: PairInfo):
        """
        Update lightweight pair quality score from cached candle-close metrics.
        No heavy computations here by design.
        """
        pval = float(getattr(pair_info, 'last_pvalue', 0.0) or 0.0)
        beta = abs(float(getattr(pair_info, 'beta_btc', 0.0) or 0.0))
        hedge = abs(float(getattr(pair_info, 'hedge_ratio', 0.0) or 0.0))
        hl = float(getattr(pair_info, 'half_life', 0.0) or 0.0)

        # Normalize to [0,1] quality components (higher is better).
        # p-value: 0 is best, 0.05+ is poor.
        p_quality = 1.0 - self._clamp01(pval / 0.05) if pval > 0 else 0.0
        # Beta: 0 is best, 0.15+ is poor.
        beta_quality = 1.0 - self._clamp01(beta / 0.15)
        # Hedge balance: ideal near 1.0, degrade as distance grows.
        hedge_quality = 1.0 - self._clamp01(abs(hedge - 1.0) / 1.0)
        # Half-life quality: prefer roughly 6h..72h in current timeframe units.
        if hl <= 0:
            hl_quality = 0.0
        elif hl < 6:
            hl_quality = self._clamp01(hl / 6.0)
        elif hl > 72:
            hl_quality = 1.0 - self._clamp01((hl - 72.0) / 120.0)
        else:
            hl_quality = 1.0

        # Recent adverse outcomes penalty (cached counters on PairInfo, optional).
        fail_penalty = float(getattr(pair_info, '_recent_fail_penalty', 0.0) or 0.0)
        fail_penalty = self._clamp01(fail_penalty)

        score = (
            0.38 * p_quality +
            0.32 * beta_quality +
            0.20 * hedge_quality +
            0.10 * hl_quality
        )
        score = max(0.0, score - 0.25 * fail_penalty)

        pair_info.quality_score = float(score)
        pair_info.quality_updated_at = time.time()

    def _update_pair_quality_penalty_on_close(self, pair_info: PairInfo, close_reason: str | None):
        reason = (close_reason or '').strip().lower()
        penalty = float(getattr(pair_info, '_recent_fail_penalty', 0.0) or 0.0)
        if reason in {'z_sl', 'hardware_sl', 'broken_coint', 'beta_critical', 'circuit', 'desync'}:
            penalty = min(1.0, penalty + 0.35)
        elif reason in {'z_tp', 'hardware_tp'}:
            penalty = max(0.0, penalty - 0.20)
        else:
            penalty = max(0.0, penalty - 0.05)
        pair_info._recent_fail_penalty = penalty

    def _apply_close_cooldown(self, pair_info: PairInfo, close_reason: str | None):
        """Apply per-pair cooldown after stop-loss exits to prevent instant re-entry loops."""
        reason = (close_reason or '').strip().lower()
        if reason in {'z_sl', 'hardware_sl'}:
            cooldown_sec = int(getattr(self.config, 'sl_reentry_cooldown_sec', 0) or 0)
            if cooldown_sec <= 0:
                cooldown_sec = int(getattr(self.config, 'close_retry_cooldown_sec', 30) or 30)
            pair_info._close_cooldown_until = time.time() + max(1, cooldown_sec)
        else:
            pair_info._close_cooldown_until = 0.0

    async def _fetch_account_trades_window(
        self,
        symbol: str,
        start_ms: int,
        *,
        max_records: int = 3000,
        page_limit: int = 1000,
    ) -> list:
        """
        Fetch user trades in pages by moving startTime cursor forward.
        Reduces risk of missing fills when trade count exceeds small limits.
        """
        out = []
        seen = set()
        cursor = int(max(0, start_ms))
        hard_cap = int(max(1, max_records))
        limit = int(min(1000, max(1, page_limit)))
        for _ in range(20):  # hard safety cap for API calls per fetch
            if len(out) >= hard_cap:
                break
            batch = await self.client.get_account_trades(symbol=symbol, startTime=cursor, limit=limit)
            if not batch:
                break
            max_time = cursor
            for t in batch:
                tid = t.get('id')
                ttime = int(t.get('time', 0) or t.get('T', 0) or 0)
                key = (tid, ttime)
                if key in seen:
                    continue
                seen.add(key)
                out.append(t)
                if ttime > max_time:
                    max_time = ttime
                if len(out) >= hard_cap:
                    break
            if len(batch) < limit:
                break
            if max_time <= cursor:
                break
            cursor = max_time + 1
        return out

    def _build_execution_rows(self, pair_info: PairInfo, symbol: str, trades: list, phase: str, trade_id: int | None = None):
        rows = []
        if not trades:
            return rows
        pair_id = getattr(pair_info, 'db_id', None)
        for t in trades:
            try:
                rows.append({
                    'trade_id': trade_id,
                    'pair_id': pair_id,
                    'symbol': symbol,
                    'phase': phase,
                    'side': t.get('side') or t.get('S'),
                    'order_id': int(t.get('orderId')) if t.get('orderId') is not None else None,
                    'exchange_trade_id': int(t.get('id')) if t.get('id') is not None else None,
                    'price': float(t.get('price', 0) or 0),
                    'qty': float(t.get('qty', 0) or t.get('executedQty', 0) or 0),
                    'quote_qty': float(t.get('quoteQty', 0) or 0),
                    'realized_pnl': float(t.get('realizedPnl', 0) or 0),
                    'commission': float(t.get('commission', 0) or 0),
                    'commission_asset': str(t.get('commissionAsset', '') or ''),
                    'event_time': int(t.get('time', 0) or t.get('T', 0) or 0),
                    'is_buyer': bool(t.get('buyer', False)),
                    'is_maker': bool(t.get('maker', False)),
                })
            except Exception:
                continue
        return rows

    async def _persist_pair_executions(self, pair_info: PairInfo, trades1: list, trades2: list, phase: str, trade_id: int | None = None):
        rows = []
        rows.extend(self._build_execution_rows(pair_info, pair_info.symbol1, trades1, phase, trade_id))
        rows.extend(self._build_execution_rows(pair_info, pair_info.symbol2, trades2, phase, trade_id))
        if not rows:
            return
        try:
            await db.add_trade_executions(rows)
        except Exception as e:
            print(f"⚠️ Could not persist trade executions for {pair_info.symbol1}-{pair_info.symbol2} [{phase}]: {e}")

    async def _set_leverage(self, symbol, leverage):
        """Sets leverage for the symbol if not already set. Returns True on success."""
        if not leverage or leverage < 1:
            return True
        if self.leverage_cache.get(symbol) == leverage:
            return True
        try:
            print(f"⚖️ Setting leverage {leverage}x for {symbol}...")
            await self.client.change_leverage(symbol=symbol, leverage=leverage)
            self.leverage_cache[symbol] = leverage
            return True
        except Exception as e:
            print(f"⚠️ Failed to set leverage for {symbol}: {e}")
            return False

    def _estimate_coint_streak_from_history(self, pair_info: PairInfo, max_recent_bars: int = 8) -> tuple[int, int]:
        """
        Estimate consecutive cointegration streak from already loaded historical candles.
        Used to avoid restart bias when coint_streak would otherwise start from 0/1.
        Returns: (streak_bars, latest_eval_ts_ms)
        """
        try:
            s1, s2 = pair_info.symbol1, pair_info.symbol2
            d1 = self.all_data.get(s1)
            d2 = self.all_data.get(s2)
            if not d1 or not d2:
                return 0, 0

            closes1 = list(d1.close)
            closes2 = list(d2.close)
            ts1 = list(d1.ts)
            ts2 = list(d2.ts)
            n = min(len(closes1), len(closes2), len(ts1), len(ts2))
            need = int(self.min_data_points or 0)
            if n < need or need <= 0:
                return 0, 0

            closes1 = closes1[-n:]
            closes2 = closes2[-n:]
            ts1 = ts1[-n:]
            ts2 = ts2[-n:]

            p_value_threshold = getattr(self.config, 'p_value_threshold', 0.05) or 0.05
            max_recent = max(1, int(max_recent_bars or 1))
            max_shift = min(max_recent - 1, n - need)

            streak = 0
            latest_eval_ts = 0
            for shift in range(0, max_shift + 1):
                end = n - shift
                start = end - need
                if start < 0:
                    break

                ts1_end = int(ts1[end - 1])
                ts2_end = int(ts2[end - 1])
                eval_ts = ts1_end if ts1_end == ts2_end else min(ts1_end, ts2_end)
                if shift == 0:
                    latest_eval_ts = eval_ts
                if ts1_end != ts2_end:
                    break

                lp1 = np.log(np.asarray(closes1[start:end], dtype=float))
                lp2 = np.log(np.asarray(closes2[start:end], dtype=float))
                flag, _, _, _ = utils.calculate_cointegration(lp1, lp2, p_value_threshold, strict_hl=False)
                if flag == 1:
                    streak += 1
                else:
                    break

            return int(streak), int(latest_eval_ts)
        except Exception:
            return 0, 0

    def _priority_file_path(self) -> str:
        priority_file_path = getattr(self.config, 'priority_pairs_file', 'best_pairs.json')
        if priority_file_path and not os.path.isabs(priority_file_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            priority_file_path = os.path.join(script_dir, priority_file_path)
        return priority_file_path or ''

    def _pair_blacklist_file_path(self) -> str:
        pair_blacklist_file = getattr(self.config, 'pair_blacklist_file', '')
        if pair_blacklist_file and not os.path.isabs(pair_blacklist_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            pair_blacklist_file = os.path.join(script_dir, pair_blacklist_file)
        return pair_blacklist_file or ''

    def _best_pairs_only_enabled(self) -> bool:
        return _as_bool(getattr(self.config, 'best_pairs_only', False), False)

    def _pair_blacklist_enabled(self) -> bool:
        if not self._best_pairs_only_enabled():
            return True
        return _as_bool(getattr(self.config, 'pair_blacklist_enabled', True), True)

    def _auto_refresh_priority_pairs_enabled(self) -> bool:
        return _as_bool(getattr(self.config, 'auto_refresh_priority_pairs', False), False)

    def _read_pair_file_entries(self, path: str, *, sort_by_score: bool = False) -> list[tuple[str, str]]:
        if not path or not os.path.exists(path):
            return []

        try:
            suffix = os.path.splitext(path)[1].lower()
            raw_entries = []
            if suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    raw_entries = _extract_pair_entries(json.load(f))
            elif suffix in ('.csv', '.tsv'):
                with open(path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f, delimiter='\t' if suffix == '.tsv' else ',')
                    for row in reader:
                        raw_entries.append(row)
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    raw_entries = [line.strip() for line in f if line.strip()]
        except Exception:
            return []

        if sort_by_score and raw_entries and all(isinstance(x, dict) for x in raw_entries):
            raw_entries = sorted(raw_entries, key=lambda x: float(x.get('score', 0.0) or 0.0), reverse=True)

        out: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in raw_entries:
            parsed = _parse_pair_text(item)
            if parsed is None:
                continue
            key = _canonical_pair_key(parsed[0], parsed[1])
            if key in seen:
                continue
            seen.add(key)
            out.append(parsed)
        return out

    def _load_priority_pairs_cache(self) -> tuple[list[tuple[str, str]], set[tuple[str, str]]]:
        path = self._priority_file_path()
        try:
            mtime = os.path.getmtime(path) if path and os.path.exists(path) else None
        except Exception:
            mtime = None
        if path != self._priority_pairs_cache_path or mtime != self._priority_pairs_cache_mtime:
            entries = self._read_pair_file_entries(path, sort_by_score=True)
            self._priority_pairs_cache_path = path
            self._priority_pairs_cache_mtime = mtime
            self._priority_pairs_cache_entries = entries
            self._priority_pairs_cache_keys = {_canonical_pair_key(s1, s2) for s1, s2 in entries}
        return list(self._priority_pairs_cache_entries), set(self._priority_pairs_cache_keys)

    def _load_pair_blacklist_keys(self) -> set[tuple[str, str]]:
        path = self._pair_blacklist_file_path()
        try:
            mtime = os.path.getmtime(path) if path and os.path.exists(path) else None
        except Exception:
            mtime = None
        if path != self._pair_blacklist_cache_path or mtime != self._pair_blacklist_cache_mtime:
            entries = self._read_pair_file_entries(path, sort_by_score=False)
            self._pair_blacklist_cache_path = path
            self._pair_blacklist_cache_mtime = mtime
            self._pair_blacklist_cache_keys = {_canonical_pair_key(s1, s2) for s1, s2 in entries}
        return set(self._pair_blacklist_cache_keys)

    def _is_pair_trade_allowed(self, symbol1: str, symbol2: str) -> bool:
        key = _canonical_pair_key(symbol1, symbol2)
        if self._pair_blacklist_enabled() and key in self._load_pair_blacklist_keys():
            return False
        if self._best_pairs_only_enabled():
            _, priority_keys = self._load_priority_pairs_cache()
            return key in priority_keys
        return True

    def _load_priority_pairs_list(self, ready_set: set | None = None) -> list[tuple[str, str]]:
        """
        Load priority pairs from file.
        Supports both formats:
        - ["SYM1-SYM2", ...] (legacy)
        - [{"pair":"SYM1-SYM2","score":...,...}, ...] (rich)
        """
        entries, _ = self._load_priority_pairs_cache()
        out: list[tuple[str, str]] = []
        seen = set()
        for s1, s2 in entries:
            if ready_set is not None and (s1 not in ready_set or s2 not in ready_set):
                continue
            key = tuple(sorted((s1, s2)))
            if key in seen:
                continue
            seen.add(key)
            out.append((s1, s2))
        return out

    def _pair_priority_score(self, stat: dict, min_trades: int) -> float:
        trades = int(stat.get('trades', 0) or 0)
        if trades <= 0:
            return -999.0
        wins = int(stat.get('wins', 0) or 0)
        tp_wins = int(stat.get('tp_wins', 0) or 0)
        bad_closes = int(stat.get('bad_closes', 0) or 0)
        net_pnl = float(stat.get('net_pnl', 0.0) or 0.0)
        avg_pnl = float(stat.get('avg_pnl', 0.0) or 0.0)
        sum_pos = float(stat.get('sum_pos', 0.0) or 0.0)
        sum_neg_abs = float(stat.get('sum_neg_abs', 0.0) or 0.0)

        winrate = wins / trades
        tp_rate = tp_wins / trades
        bad_rate = bad_closes / trades
        pf = sum_pos / max(1e-9, sum_neg_abs)
        pf_score = min(1.5, pf / 2.0)
        pnl_score = max(0.0, math.tanh(net_pnl / 5.0))
        avg_score = max(0.0, math.tanh(avg_pnl / 1.0))
        confidence = min(1.0, trades / max(1.0, float(min_trades)))

        score = (
            0.30 * winrate +
            0.20 * tp_rate +
            0.20 * pf_score +
            0.20 * pnl_score +
            0.10 * avg_score
        )
        score = score * confidence - (0.25 * bad_rate)
        if net_pnl <= 0 and trades < (min_trades + 2):
            score -= 0.25
        return float(score)

    async def _refresh_best_pairs(self, force: bool = False, reason: str = ''):
        """
        Rebuild best_pairs.json from closed trade performance.
        Promotes stable profitable pairs, demotes/removes degrading pairs.
        """
        if not self._auto_refresh_priority_pairs_enabled():
            return
        now = time.time()
        refresh_sec = 300.0  # throttle expensive rebuilds
        if not force and (now - self._best_pairs_last_refresh) < refresh_sec:
            return

        async with self._best_pairs_refresh_lock:
            now = time.time()
            if not force and (now - self._best_pairs_last_refresh) < refresh_sec:
                return
            self._best_pairs_last_refresh = now

            path = self._priority_file_path()
            if not path:
                return

            try:
                # Hysteresis for best_pairs stability:
                # enter is stricter than keep/remove to prevent borderline flip-flop.
                enter_min_trades = 4
                enter_min_wins = 3
                enter_min_winrate = 0.55
                keep_min_trades_for_eval = 6
                remove_max_winrate = 0.45
                remove_if_net_nonpositive = True
                if not hasattr(db, 'get_closed_trade_stats_by_pair'):
                    print("⚠️ best_pairs refresh skipped: db.get_closed_trade_stats_by_pair is missing")
                    return
                stats = await db.get_closed_trade_stats_by_pair()
                existing_entries = []
                if os.path.exists(path):
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            raw = json.load(f)
                        if isinstance(raw, list):
                            for x in raw:
                                if isinstance(x, str) and '-' in x:
                                    existing_entries.append({'pair': str(x).strip()})
                                elif isinstance(x, dict):
                                    p = str(x.get('pair', '') or '').strip()
                                    if '-' in p:
                                        item = dict(x)
                                        item['pair'] = p
                                        existing_entries.append(item)
                    except Exception:
                        existing_entries = []

                def _canon(pair_str: str):
                    parts = [p.strip().upper() for p in pair_str.split('-', 1)]
                    if len(parts) != 2 or not parts[0] or not parts[1]:
                        return None
                    return tuple(sorted(parts))

                ranked = []
                stat_keys = set()
                stat_by_key = {}
                for st in stats:
                    s1 = str(st.get('symbol1', '') or '').strip().upper()
                    s2 = str(st.get('symbol2', '') or '').strip().upper()
                    key = None
                    if s1 and s2:
                        key = tuple(sorted((s1, s2)))
                        stat_keys.add(key)
                        stat_by_key[key] = st
                    trades = int(st.get('trades', 0) or 0)
                    net_pnl = float(st.get('net_pnl', 0.0) or 0.0)
                    wins = int(st.get('wins', 0) or 0)
                    winrate = (wins / trades) if trades > 0 else 0.0
                    score = self._pair_priority_score(st, min_trades=enter_min_trades)
                    if (
                        trades >= enter_min_trades
                        and wins >= enter_min_wins
                        and winrate >= enter_min_winrate
                        and net_pnl > 0
                        and score > 0
                    ):
                        ranked.append((score, st))

                ranked.sort(key=lambda x: (-x[0], -float(x[1].get('net_pnl', 0.0) or 0.0), -int(x[1].get('trades', 0) or 0)))
                payload = []
                for score, item in ranked:
                    trades = int(item.get('trades', 0) or 0)
                    wins = int(item.get('wins', 0) or 0)
                    tp_wins = int(item.get('tp_wins', 0) or 0)
                    bad = int(item.get('bad_closes', 0) or 0)
                    sum_pos = float(item.get('sum_pos', 0.0) or 0.0)
                    sum_neg_abs = float(item.get('sum_neg_abs', 0.0) or 0.0)
                    pf = sum_pos / max(1e-9, sum_neg_abs)
                    payload.append({
                        'pair': f"{item['symbol1']}-{item['symbol2']}",
                        'score': round(float(score), 6),
                        'trade_count': trades,
                        'net_pnl': round(float(item.get('net_pnl', 0.0) or 0.0), 8),
                        'avg_pnl': round(float(item.get('avg_pnl', 0.0) or 0.0), 8),
                        'win_rate': round((wins / trades) if trades > 0 else 0.0, 6),
                        'tp_rate': round((tp_wins / trades) if trades > 0 else 0.0, 6),
                        'bad_rate': round((bad / trades) if trades > 0 else 0.0, 6),
                        'profit_factor': round(float(pf), 6),
                        'last_close_time': int(item.get('last_close_time', 0) or 0),
                        'source': 'bot_performance',
                    })

                # Preserve existing entries that are not yet present in runtime stats (manual/seed/history).
                # If a pair already has runtime stats, keep it in best_pairs only when it passes strict quality gates.
                existing_by_key = {}
                for entry in existing_entries:
                    p = str(entry.get('pair', '') or '').strip()
                    ck = _canon(p)
                    if ck is None:
                        continue
                    if ck in stat_by_key:
                        st = stat_by_key.get(ck) or {}
                        trades = int(st.get('trades', 0) or 0)
                        wins = int(st.get('wins', 0) or 0)
                        net_pnl = float(st.get('net_pnl', 0.0) or 0.0)
                        winrate = (wins / trades) if trades > 0 else 0.0
                        # Remove only when we have enough data and clearly degraded behavior.
                        enough_data_to_judge = trades >= keep_min_trades_for_eval
                        degraded = (
                            (remove_if_net_nonpositive and net_pnl <= 0)
                            or (winrate < remove_max_winrate)
                        )
                        if enough_data_to_judge and degraded:
                            continue
                    # Keep full existing object (score/metrics), only ensure pair field exists.
                    existing_by_key[ck] = entry if isinstance(entry, dict) else {'pair': p, 'source': 'manual_seed'}

                # Keep ranked order first, then append remaining preserved entries.
                merged = []
                seen = set()
                for item in payload:
                    pair_txt = item.get('pair', '') if isinstance(item, dict) else str(item)
                    ck = _canon(pair_txt)
                    if ck is None or ck in seen:
                        continue
                    seen.add(ck)
                    merged.append(item)
                for ck, item in existing_by_key.items():
                    if ck in seen:
                        continue
                    seen.add(ck)
                    merged.append(item)
                payload = merged

                # Safety: never wipe priority list if rebuild produced nothing.
                if not payload and existing_entries:
                    payload = existing_entries
                    print("⚠️ best_pairs rebuild produced empty list, preserving existing file entries.")

                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                tmp_path = f"{path}.tmp"
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp_path, path)

                preview = ", ".join([(x.get('pair', '') if isinstance(x, dict) else str(x)) for x in payload[:5]]) if payload else "empty"
                print(
                    f"Updated best_pairs.json ({reason or 'periodic'}): "
                    f"{len(payload)} pairs, from {len(stats)} tracked pairs. Top: {preview}"
                )
            except Exception as e:
                print(f"⚠️ Could not rebuild best_pairs.json: {e}")

    def _add_to_best_pairs(self, symbol1: str, symbol2: str):
        """
        Legacy trigger kept for compatibility.
        Now schedules full best_pairs rebuild from performance stats.
        """
        if not self._auto_refresh_priority_pairs_enabled():
            return
        try:
            self.loop.create_task(self._refresh_best_pairs(force=True, reason=f"tp_close:{symbol1}-{symbol2}"))
        except Exception as e:
            print(f"⚠️ Could not schedule best_pairs refresh: {e}")

    def _timeframe_seconds_local(self) -> int:
        tf = (self.timeframe or '1h').strip().lower()
        try:
            if tf.endswith('m'):
                return max(60, int(tf[:-1]) * 60)
            if tf.endswith('h'):
                return max(3600, int(tf[:-1]) * 3600)
            if tf.endswith('d'):
                return max(86400, int(tf[:-1]) * 86400)
        except Exception:
            pass
        return 3600

    def _max_hold_bars_from_days(self) -> int:
        try:
            candles_per_day = float(utils.CANDLES_PER_DAY.get(str(self.timeframe), 24) or 24)
        except Exception:
            candles_per_day = 24.0
        max_hold_days = float(getattr(self.config, 'max_hold_days', 30.0) or 30.0)
        return max(1, int(round(max_hold_days * candles_per_day)))

    def _compute_entry_hold_limit_bars(self, hl_bars: float) -> int:
        max_hold_bars = self._max_hold_bars_from_days()
        hl_bars = float(hl_bars or 0.0)
        if hl_bars > 0:
            hold_multiplier = float(getattr(self.config, 'hold_multiplier', 3.0) or 3.0)
            hl_limit = max(1, int(round(hl_bars * hold_multiplier)))
            return min(hl_limit, max_hold_bars)
        return max_hold_bars

    def _time_exit_hold_limit_bars(self, pair_info: PairInfo) -> int:
        frozen_limit = int(getattr(pair_info, 'entry_hold_limit_bars', 0) or 0)
        if frozen_limit > 0:
            return frozen_limit
        hl_bars = float(getattr(pair_info, 'entry_half_life_bars', 0.0) or 0.0)
        if hl_bars <= 0:
            hl_bars = float(getattr(pair_info, 'half_life', 0.0) or 0.0)
        return self._compute_entry_hold_limit_bars(hl_bars)

    def _position_hold_bars(self, pair_info: PairInfo, now_ts: float | None = None) -> int:
        if now_ts is None:
            now_ts = time.time()
        tf_sec = max(1, int(self._timeframe_seconds_local() or 3600))
        open_ts = float(getattr(pair_info, '_trade_open_time', 0.0) or 0.0)
        if open_ts <= 0:
            open_ts = float(int(getattr(pair_info, 'open_time', 0) or 0))
        if open_ts <= 0:
            return 0
        elapsed = max(0.0, float(now_ts) - open_ts)
        return max(0, int(elapsed // tf_sec))

    def _time_exit_due(self, pair_info: PairInfo, now_ts: float | None = None) -> tuple[bool, int, int]:
        hold_bars = self._position_hold_bars(pair_info, now_ts=now_ts)
        hold_limit = self._time_exit_hold_limit_bars(pair_info)
        return hold_bars >= hold_limit, hold_bars, hold_limit

    async def _ensure_entry_time_exit_state(self, pair_info: PairInfo, persist: bool = False) -> bool:
        changed = False
        if float(getattr(pair_info, 'entry_half_life_bars', 0.0) or 0.0) <= 0:
            pair_info.entry_half_life_bars = float(getattr(pair_info, 'half_life', 0.0) or 0.0)
            changed = True
        if int(getattr(pair_info, 'entry_hold_limit_bars', 0) or 0) <= 0:
            pair_info.entry_hold_limit_bars = self._compute_entry_hold_limit_bars(pair_info.entry_half_life_bars)
            changed = True
        if persist and changed and pair_info.db_id:
            try:
                await db.update_pair({
                    'id': pair_info.db_id,
                    'entry_half_life_bars': pair_info.entry_half_life_bars,
                    'entry_hold_limit_bars': pair_info.entry_hold_limit_bars,
                })
            except Exception as e:
                print(f"?? Failed to persist time-exit state for {pair_info.symbol1}-{pair_info.symbol2}: {e}")
        return changed

    def _floor_to_candle_ts_ms(self, ts_ms: int) -> int:
        tf_ms = self._timeframe_seconds_local() * 1000
        if tf_ms <= 0:
            tf_ms = 3600 * 1000
        return (int(ts_ms) // tf_ms) * tf_ms

    def _latest_closed_candle_ts_ms(self, pair_info: PairInfo) -> int:
        d1 = self.all_data.get(pair_info.symbol1)
        d2 = self.all_data.get(pair_info.symbol2)
        ts1 = int(d1.ts[-1]) if d1 and len(d1.ts) > 0 else 0
        ts2 = int(d2.ts[-1]) if d2 and len(d2.ts) > 0 else 0
        if ts1 and ts2:
            return min(ts1, ts2)
        return ts1 or ts2 or 0

    def mark_pair_wait_for_next_candle(self, pair_info: PairInfo, reason: str = ''):
        latest_ts = self._latest_closed_candle_ts_ms(pair_info)
        if latest_ts <= 0:
            latest_ts = self._floor_to_candle_ts_ms(int(time.time() * 1000))
        pair_info.reentry_block_candle_ts = int(latest_ts)
        key = frozenset([pair_info.symbol1, pair_info.symbol2])
        prev = int(self._reentry_block_by_pair.get(key, 0) or 0)
        if pair_info.reentry_block_candle_ts > prev:
            self._reentry_block_by_pair[key] = pair_info.reentry_block_candle_ts
        pair_info._wait_for_candle = True
        print(f"⏸️ {pair_info.symbol1}-{pair_info.symbol2}: Re-entry blocked until next candle close (reason: {reason or 'close'}, candle_ts={pair_info.reentry_block_candle_ts})")

    def _is_pair_reentry_blocked_same_candle(self, pair_info: PairInfo) -> bool:
        key = frozenset([pair_info.symbol1, pair_info.symbol2])
        block_ts = int(getattr(pair_info, 'reentry_block_candle_ts', 0) or 0)
        if block_ts <= 0:
            block_ts = int(self._reentry_block_by_pair.get(key, 0) or 0)
            if block_ts > 0:
                pair_info.reentry_block_candle_ts = block_ts
        if block_ts <= 0:
            return False
        latest_ts = self._latest_closed_candle_ts_ms(pair_info)
        if latest_ts <= 0:
            return True
        if latest_ts <= block_ts:
            pair_info._wait_for_candle = True
            return True
        pair_info.reentry_block_candle_ts = 0
        pair_info._wait_for_candle = False
        self._reentry_block_by_pair.pop(key, None)
        return False

    async def _trigger_immediate_analysis(self):
        """
        Triggers immediate analysis of all pairs when a slot becomes available.
        This allows opening new trades without waiting for the next candle close.
        """
        await asyncio.sleep(0.5)  # Small delay to ensure state is updated
        
        # Check if we have available slots
        max_pairs = getattr(self.config, 'max_active_pairs', 5) or 5
        current_active = self.count_active_positions()
        
        if current_active >= max_pairs:
            return  # No slots available
        
        print(f"🔍 Immediate analysis: {current_active}/{max_pairs} slots used. Scanning for opportunities...")
        
        # Analyze all pairs with data
        analyzed = 0
        opened_before = self.count_active_positions()
        for pair_set, pair_info in list(self.active_pairs.items()):
            if pair_info.position_status != 0:
                continue  # Skip pairs with open positions
            
            # Skip pairs in cooldown (recently closed by SL)
            if getattr(pair_info, '_close_cooldown_until', 0) > time.time():
                continue
            
            # Skip pairs waiting for next candle close before re-entry
            if getattr(pair_info, '_wait_for_candle', False) or self._is_pair_reentry_blocked_same_candle(pair_info):
                continue
            
            s1, s2 = pair_info.symbol1, pair_info.symbol2
            if isinstance(self.all_data, dict) and s1 in self.all_data and s2 in self.all_data:
                await self._check_signals_for_active_pairs(s1)
                analyzed += 1
                
                # Check if we filled all slots
                if self.count_active_positions() >= max_pairs:
                    break
        
        print(f"🔍 Immediate analysis complete. Checked {analyzed} pairs.")

        opened_after = self.count_active_positions()

        # If quick scan did not fill slots enough, run discovery immediately so OTHER pairs can
        # be traded without waiting for candle on recently closed pairs.
        need_more_pairs = opened_after < max_pairs
        weak_progress = (analyzed == 0) or (opened_after <= opened_before + 1)
        if need_more_pairs and weak_progress:
            now = time.time()
            if self._discovery_task is None or self._discovery_task.done():
                if now - self._last_discovery_time > 30:
                    self._last_discovery_time = now
                    print("🔍 Immediate scan made weak progress. Triggering discovery for alternative pairs...")
                    self._discovery_task = self.loop.create_task(self._discover_new_pairs())

    def is_symbol_locked(self, symbol: str, exclude_pair=None) -> bool:
        """Check if symbol is already in an active position or being opened (in any pair)."""
        for pair_info in self.active_pairs.values():
            # Skip the pair we're currently trying to open (prevents self-blocking)
            if exclude_pair is not None and pair_info is exclude_pair:
                continue
            # DC-3: Also check is_trading — pair may be in the process of opening
            # (position_status is now set AFTER order success, not tentatively in lock)
            if pair_info.position_status != 0 or pair_info.is_trading:
                if symbol in (pair_info.symbol1, pair_info.symbol2):
                    return True
        return False

    def _find_symbol_owner_pair(self, symbol: str, exclude_pair=None) -> str:
        """
        Return owner pair name if symbol is currently used by another OPEN/TRADING pair.
        Used to prevent false orphan cleanup on stale CLOSED pairs.
        """
        for pair_info in self.active_pairs.values():
            if exclude_pair is not None and pair_info is exclude_pair:
                continue
            if pair_info.position_status != 0 or pair_info.is_trading:
                if symbol in (pair_info.symbol1, pair_info.symbol2):
                    return f"{pair_info.symbol1}-{pair_info.symbol2}"
        return ''

    def count_active_positions(self, exclude_pair=None) -> int:
        """Count the number of currently open or being-opened pairs."""
        count = 0
        for pair_info in self.active_pairs.values():
            # Skip the pair we're currently trying to open (prevents self-blocking)
            if exclude_pair is not None and pair_info is exclude_pair:
                continue
            # DC-3: Also count pairs being opened (is_trading=True)
            if pair_info.position_status != 0 or pair_info.is_trading:
                count += 1
        return count

    async def _fetch_pair_live_position_amts(self, s1: str, s2: str, retries: int = 2, delay_sec: float = 0.35) -> dict | None:
        """Read signed live position amounts for two symbols from exchange."""
        last_err = None
        for attempt in range(max(0, int(retries)) + 1):
            try:
                data = await self.client.get_position_risk()
                out = {s1: 0.0, s2: 0.0}
                for pos in data:
                    sym = pos.get('symbol')
                    if sym in out:
                        out[sym] = float(pos.get('positionAmt', 0) or 0)
                return out
            except Exception as e:
                last_err = e
                if attempt < retries:
                    await asyncio.sleep(max(0.05, float(delay_sec)))
        print(f"⚠️ Could not fetch live pair positions for {s1}-{s2}: {last_err}")
        return None

    def _sync_pair_exchange_cache(self, s1: str, s2: str, live_amts: dict | None):
        """Sync internal exchange caches from signed live amounts."""
        if not isinstance(live_amts, dict):
            return
        amt1 = abs(float(live_amts.get(s1, 0.0) or 0.0))
        amt2 = abs(float(live_amts.get(s2, 0.0) or 0.0))
        if amt1 > 0:
            self._exchange_positions_cache[s1] = amt1
        else:
            self._exchange_positions_cache.pop(s1, None)
        if amt2 > 0:
            self._exchange_positions_cache[s2] = amt2
        else:
            self._exchange_positions_cache.pop(s2, None)
        self._exchange_position_count = len(self._exchange_positions_cache)

    async def _confirm_pair_closed_on_exchange(self, pair_info: PairInfo, close_reason_tag: str) -> bool:
        """
        Final close gate: only allow DB/memory close when BOTH legs are actually zero on exchange.
        """
        s1, s2 = pair_info.symbol1, pair_info.symbol2
        live_amts = await self._fetch_pair_live_position_amts(s1, s2, retries=2, delay_sec=0.4)
        if live_amts is None:
            now_ts = time.time()
            last_warn = float(getattr(pair_info, '_last_close_verify_warn_ts', 0) or 0)
            if now_ts - last_warn >= 60:
                pair_info._last_close_verify_warn_ts = now_ts
                warn_msg = (
                    f"⚠️ <b>Close verification delayed</b>: {s1}/{s2}\n"
                    f"Tag: <code>{close_reason_tag}</code>\n"
                    f"Could not confirm leg status from exchange, close finalization is blocked."
                )
                reply_to = await self._resolve_reply_to_message_id(pair_info)
                await self._notify(warn_msg, reply_to)
            return False

        self._sync_pair_exchange_cache(s1, s2, live_amts)
        amt1_signed = float(live_amts.get(s1, 0.0) or 0.0)
        amt2_signed = float(live_amts.get(s2, 0.0) or 0.0)
        amt1 = abs(amt1_signed)
        amt2 = abs(amt2_signed)
        if amt1 <= 0 and amt2 <= 0:
            return True

        # Keep pair OPEN in state if close did not fully complete.
        pair_info.qty1 = amt1
        pair_info.qty2 = amt2
        if pair_info.position_status == 0:
            if amt1_signed > 0 and amt2_signed < 0:
                pair_info.position_status = 1
            elif amt1_signed < 0 and amt2_signed > 0:
                pair_info.position_status = -1
            elif amt1 > 0 or amt2 > 0:
                pair_info.position_status = 1

        now_ts = time.time()
        last_warn = float(getattr(pair_info, '_last_close_verify_warn_ts', 0) or 0)
        detail = f"{s1}:{amt1_signed:+.8f}, {s2}:{amt2_signed:+.8f}"
        print(f"⚠️ Close NOT finalized for {s1}-{s2} [{close_reason_tag}] -> {detail}")
        if now_ts - last_warn >= 60:
            pair_info._last_close_verify_warn_ts = now_ts
            warn_msg = (
                f"⚠️ <b>Close not finalized on exchange</b>: {s1}/{s2}\n"
                f"Tag: <code>{close_reason_tag}</code>\n"
                f"Remaining: <code>{detail}</code>\n"
                f"Bot keeps pair OPEN and will retry close/sync."
            )
            reply_to = await self._resolve_reply_to_message_id(pair_info)
            await self._notify(warn_msg, reply_to)
        return False

    async def _refresh_exchange_position_count(self):
        """Refresh cached exchange position count from exchange API."""
        try:
            positions_risk = await self.client.get_position_risk()
            positions = {}
            for pos in positions_risk:
                amt = abs(float(pos.get('positionAmt', 0)))
                if amt > 0:
                    positions[pos['symbol']] = amt
            self._exchange_positions_cache = positions
            self._exchange_position_count = len(positions)
            return len(positions)
        except Exception as e:
            print(f"⚠️ Failed to refresh exchange position count: {e}")
            return self._exchange_position_count

    def _is_symbol_temporarily_blocked(self, symbol: str) -> bool:
        until = self._symbol_block_until.get(symbol, 0)
        if until <= 0:
            return False
        if time.time() >= until:
            self._symbol_block_until.pop(symbol, None)
            return False
        return True

    def _set_symbol_cooldown(self, symbol: str, seconds: int, reason: str):
        until = time.time() + max(1, int(seconds))
        prev = self._symbol_block_until.get(symbol, 0)
        self._symbol_block_until[symbol] = max(prev, until)
        left = int(self._symbol_block_until[symbol] - time.time())
        print(f"[COOLDOWN] Symbol {symbol} blocked for {left}s ({reason})")

    def _can_open_new_position_reason(self, s1: str, s2: str, exclude_pair=None) -> tuple[bool, str]:
        """Check if we can open a new position for this pair.
        
        Args:
            exclude_pair: PairInfo to exclude from checks (prevents self-blocking
                          when the pair being opened has is_trading=True already set).
        """
        # Check if trading is enabled
        trade_mode = getattr(self.config, 'trade_mode', True)
        if trade_mode is not None and str(trade_mode).lower() in ('false', '0', 'no'):
            return False, 'trade_mode_disabled'

        if not self._is_pair_trade_allowed(s1, s2):
            return False, 'pair_not_allowed'
        
        # Check max active pairs limit (local memory)
        max_pairs = getattr(self.config, 'max_active_pairs', 5) or 5
        if self.count_active_positions(exclude_pair=exclude_pair) >= max_pairs:
            return False, 'local_max_pairs'
        
        # Keep scalar counter consistent with current symbol cache to avoid stale blocks.
        cached_positions = len(self._exchange_positions_cache or {})
        if cached_positions != self._exchange_position_count:
            self._exchange_position_count = cached_positions

        # SAFETY: Also check exchange position cache (positions / 2 = pairs)
        # Each pair opens 2 positions, so max positions = max_pairs * 2
        max_exchange_positions = max_pairs * 2
        if self._exchange_position_count >= max_exchange_positions:
            print(f"🚫 Exchange position limit: {self._exchange_position_count}/{max_exchange_positions} positions on exchange")
            return False, 'exchange_position_limit'
        
        # Symbol cooldown after insufficient margin/capital/order-limit failures
        if self._is_symbol_temporarily_blocked(s1):
            return False, f'symbol_cooldown:{s1}'
        if self._is_symbol_temporarily_blocked(s2):
            return False, f'symbol_cooldown:{s2}'

        # Check symbol lock - each symbol can only be in one active pair
        if self.is_symbol_locked(s1, exclude_pair=exclude_pair):
            return False, f'symbol_locked:{s1}'
        if self.is_symbol_locked(s2, exclude_pair=exclude_pair):
            return False, f'symbol_locked:{s2}'
        
        return True, 'ok'

    def can_open_new_position(self, s1: str, s2: str, exclude_pair=None) -> bool:
        allowed, _ = self._can_open_new_position_reason(s1, s2, exclude_pair=exclude_pair)
        return allowed

    async def _execute_trade(self, pair_info: PairInfo, direction: int, close_reason: str = None):
        """
        Executes a trade order.
        direction: 1 for long spread, -1 for short spread, 0 for close.
        close_reason: Why position is being closed (z_tp, z_sl, circuit, broken_coint, hardware_sl, hardware_tp, manual, desync)
        """
        s1 = pair_info.symbol1
        s2 = pair_info.symbol2
        leverage = self.config.leverage if self.config and self.config.leverage else 20

        # For OPENING trades: acquire lock and re-check limit
        if direction != 0:
            async with self._trade_lock:
                # CRITICAL: Re-check limit inside lock to prevent race condition
                # exclude_pair=pair_info: don't let the pair block itself (is_trading already True)
                if not self.can_open_new_position(s1, s2, exclude_pair=pair_info):
                    print(f"🚫 Trade blocked by lock: {s1}-{s2} (limit reached or symbol locked)")
                    pair_info.is_trading = False
                    return
                
                # CRITICAL: Also verify against LIVE exchange positions (prevents 25-position bug)
                max_pairs = getattr(self.config, 'max_active_pairs', 5) or 5
                try:
                    live_count = await self._refresh_exchange_position_count()
                    if live_count >= max_pairs * 2:
                        print(f"🚫 Trade blocked by EXCHANGE limit: {live_count}/{max_pairs * 2} positions on exchange for {s1}-{s2}")
                        pair_info.is_trading = False
                        return
                except Exception as e:
                    print(f"⚠️ Could not verify exchange positions: {e}. Blocking trade open for safety.")
                    pair_info.is_trading = False
                    return
                
                # Mark as opening INSIDE lock — only set is_trading flag.
                # position_status is set AFTER successful order execution to prevent phantom state.
                pair_info._pending_direction = direction  # Used by can_open_new_position check
            
            # Now proceed with actual execution (lock released for API calls)
            lev1_ok = await self._set_leverage(s1, leverage)
            lev2_ok = await self._set_leverage(s2, leverage)
            
            if not lev1_ok or not lev2_ok:
                print(f"[X] Trade aborted for {s1}-{s2}: leverage setting failed")
                pair_info.position_status = 0
                pair_info.is_trading = False
                # Cooldown to prevent immediate retry
                pair_info.pending_signal = None
                pair_info.pending_since = None
                pair_info.pending_source = ''
                pair_info._leverage_fail_until = time.time() + 600  # 10 min cooldown
                return
        
        try:
            if direction == 0:
                if pair_info.position_status == 0:
                    return

                close_reason_tag = str(close_reason).strip().lower() if close_reason is not None else ''
                if not close_reason_tag:
                    close_reason_tag = 'unknown'
                print(f"EXECUTING CLOSE for {s1}-{s2} (reason: {close_reason_tag})")
                
                # Store close reason IMMEDIATELY so external handlers can see it
                pair_info.last_close_reason = close_reason_tag
                
                side1_close = 'SELL' if pair_info.position_status == 1 else 'BUY'
                side2_close = 'BUY' if pair_info.position_status == 1 else 'SELL'
                qty1_close = pair_info.qty1
                qty2_close = pair_info.qty2
                
                # FAST PATH: For SL/TP triggered closes, one leg is already closed
                # Close the other leg IMMEDIATELY, cancel orders AFTER
                is_hardware_close = close_reason_tag in ('hardware_sl', 'hardware_tp')
                
                if is_hardware_close:
                    # One leg already closed by exchange - close the other one ASAP
                    # Determine which leg is still open
                    triggered_symbol = getattr(pair_info, '_triggered_symbol', None)
                    if triggered_symbol == s1:
                        # s1 closed by SL/TP, close s2
                        if qty2_close and qty2_close > 0:
                            try:
                                await self._close_leg_reduce_only(
                                    symbol=s2,
                                    side=side2_close,
                                    quantity=qty2_close
                                )
                                print(f"✅ FAST closed {s2} (qty={qty2_close})")
                            except Exception as e:
                                print(f"⚠️ Fast close {s2} failed: {e}")
                    elif triggered_symbol == s2:
                        # s2 closed by SL/TP, close s1
                        if qty1_close and qty1_close > 0:
                            try:
                                await self._close_leg_reduce_only(
                                    symbol=s1,
                                    side=side1_close,
                                    quantity=qty1_close
                                )
                                print(f"✅ FAST closed {s1} (qty={qty1_close})")
                            except Exception as e:
                                print(f"⚠️ Fast close {s1} failed: {e}")
                    else:
                        # Unknown which leg triggered - close both using stored qty
                        close_tasks = []
                        if qty1_close and qty1_close > 0:
                            close_tasks.append(self._close_leg_reduce_only(
                                symbol=s1,
                                side=side1_close,
                                quantity=qty1_close
                            ))
                        if qty2_close and qty2_close > 0:
                            close_tasks.append(self._close_leg_reduce_only(
                                symbol=s2,
                                side=side2_close,
                                quantity=qty2_close
                            ))
                        if close_tasks:
                            results = await asyncio.gather(*close_tasks, return_exceptions=True)
                            for r in results:
                                if isinstance(r, Exception):
                                    print(f"⚠️ Close error: {r}")
                    
                    # Cancel remaining algo/SL/TP orders AFTER closing
                    try:
                        await asyncio.gather(
                            self.client.cancel_open_orders(symbol=s1),
                            self.client.cancel_open_orders(symbol=s2),
                            return_exceptions=True
                        )
                    except Exception as e:
                        print(f"⚠️ Cancel orders error: {e}")

                    # Exchange is source of truth: finalize close only after both legs are really zero.
                    if not await self._confirm_pair_closed_on_exchange(pair_info, close_reason_tag):
                        return
                
                    # === PnL CALCULATION & NOTIFICATION for hardware close ===
                    # Save values BEFORE zeroing state (needed for PnL calc & notification)
                    saved_entry1 = pair_info.entry_price1
                    saved_entry2 = pair_info.entry_price2
                    saved_qty1 = pair_info.qty1
                    saved_qty2 = pair_info.qty2
                    saved_status = pair_info.position_status
                    saved_trade_id = pair_info.current_trade_id
                    
                    # Fetch actual close prices from recent trades
                    close_price1 = 0.0
                    close_price2 = 0.0
                    try:
                        await asyncio.sleep(0.5)  # Brief delay for trade data availability
                        start_ms = self._trade_window_start_ms(pair_info)
                        
                        trades1 = await self.client.get_account_trades(symbol=s1, startTime=start_ms, limit=50)
                        trades2 = await self.client.get_account_trades(symbol=s2, startTime=start_ms, limit=50)
                        
                        if trades1:
                            close_price1 = float(trades1[-1].get('price', 0))
                        if trades2:
                            close_price2 = float(trades2[-1].get('price', 0))
                    except Exception as e:
                        print(f"⚠️ Could not fetch close prices for hardware close: {e}")
                    
                    # Fallback to last_prices if trades unavailable
                    if close_price1 == 0:
                        close_price1 = self.last_prices.get(s1, saved_entry1)
                    if close_price2 == 0:
                        close_price2 = self.last_prices.get(s2, saved_entry2)
                    
                    # Calculate PnL using EXCHANGE data (source of truth)
                    try:
                        start_ms_pnl = self._trade_window_start_ms(pair_info)
                        trades_pnl_s1 = await self._fetch_account_trades_window(s1, start_ms_pnl, max_records=3000)
                        trades_pnl_s2 = await self._fetch_account_trades_window(s2, start_ms_pnl, max_records=3000)
                        if trades_pnl_s1 or trades_pnl_s2:
                            pnl1 = sum(float(t.get('realizedPnl', 0)) for t in trades_pnl_s1)
                            pnl2 = sum(float(t.get('realizedPnl', 0)) for t in trades_pnl_s2)
                        else:
                            raise ValueError("No trades found")
                    except Exception as pnl_err:
                        print(f"⚠️ Exchange PnL fetch failed for HW close ({pnl_err}), using manual calc")
                        side1_dir = 1 if saved_status == 1 else -1
                        side2_dir = -side1_dir
                        pnl1 = (close_price1 - saved_entry1) * saved_qty1 * side1_dir
                        pnl2 = (close_price2 - saved_entry2) * saved_qty2 * side2_dir
                    total_pnl = pnl1 + pnl2

                    # Calculate close z-score BEFORE DB update.
                    close_zscore = pair_info.last_z_score or 0
                    try:
                        p1 = self.last_prices.get(s1, 0)
                        p2 = self.last_prices.get(s2, 0)
                        if p1 > 0 and p2 > 0:
                            close_zscore = self._calc_realtime_zscore(pair_info, p1, p2)
                            import math
                            if math.isnan(close_zscore):
                                close_zscore = pair_info.last_z_score or 0
                    except Exception:
                        close_zscore = pair_info.last_z_score or 0
                    
                    # Update trade record in DB
                    # Calculate fees from recent trades
                    _hw_fee1, _hw_fee2 = 0.0, 0.0
                    try:
                        _hw_fee1 = sum(float(t.get('commission', 0)) for t in trades_pnl_s1)
                        _hw_fee2 = sum(float(t.get('commission', 0)) for t in trades_pnl_s2)
                    except Exception:
                        pass
                    net_pnl = total_pnl - (_hw_fee1 + _hw_fee2)
                    
                    if saved_trade_id:
                        try:
                            await db.close_trade_record(
                                saved_trade_id,
                                status='CLOSED',
                                close_reason=close_reason_tag,
                                close_price_1=close_price1,
                                close_price_2=close_price2,
                                pnl=net_pnl,
                                close_z=close_zscore,
                                fee1=_hw_fee1,
                                fee2=_hw_fee2,
                            )
                            await self._persist_pair_executions(
                                pair_info, trades_pnl_s1, trades_pnl_s2, phase='CLOSE_HARDWARE', trade_id=saved_trade_id
                            )
                        except Exception as e:
                            print(f"⚠️ Trade record update failed: {e}")
                    
                    # Build and send close notification
                    HW_REASONS = {
                        'hardware_sl': '🛡️ Hardware Stop Loss',
                        'hardware_tp': '🛡️ Hardware Take Profit',
                    }
                    reason_text = HW_REASONS.get(close_reason_tag, f'🛡️ Hardware {close_reason_tag}')
                    
                    pnl_emoji = "🟢" if net_pnl > 0 else "🔴"
                    e1 = '🟢' if pnl1 >= 0 else '🔴'
                    e2 = '🟢' if pnl2 >= 0 else '🔴'
                    
                    # Get current stats for notification
                    try:
                        p1 = self.last_prices.get(s1, 0)
                        p2 = self.last_prices.get(s2, 0)
                        if p1 > 0 and p2 > 0:
                            close_zscore = self._calc_realtime_zscore(pair_info, p1, p2)
                            import math
                            if math.isnan(close_zscore):
                                close_zscore = pair_info.last_z_score or 0
                        else:
                            close_zscore = pair_info.last_z_score or 0
                    except Exception:
                        close_zscore = pair_info.last_z_score or 0
                    
                    close_beta = getattr(pair_info, '_beta_at_trigger', None)
                    if close_beta is None:
                        close_beta = getattr(pair_info, 'beta_btc', 0) or 0
                    else:
                        pair_info._beta_at_trigger = None  # Reset after use
                    close_pval = getattr(pair_info, 'last_pvalue', 0) or 0
                    
                    # Recalculate beta & p-value fresh if they're 0 (stale after restart)
                    if (close_beta == 0 or close_pval == 0) and isinstance(self.all_data, dict) and s1 in self.all_data and s2 in self.all_data:
                        try:
                            _d1 = self.all_data[s1]
                            _d2 = self.all_data[s2]
                            if len(_d1.close) >= self.min_data_points and len(_d2.close) >= self.min_data_points:
                                _lp1 = np.log(list(_d1.close)[-self.min_data_points:])
                                _lp2 = np.log(list(_d2.close)[-self.min_data_points:])
                                _, _, _, _pval = utils.calculate_cointegration(_lp1, _lp2, strict_hl=False)
                                if close_pval == 0 and not np.isnan(_pval):
                                    close_pval = float(_pval)
                                if close_beta == 0 and isinstance(self.all_data, dict) and 'BTCUSDT' in self.all_data:
                                    _btc = self.all_data['BTCUSDT']
                                    if len(_btc.close) >= self.min_data_points:
                                        _lbtc = np.log(list(_btc.close)[-self.min_data_points:])
                                        _sr = np.diff(_lp1) - pair_info.hedge_ratio * np.diff(_lp2)
                                        _br = np.diff(_lbtc)
                                        _beta = utils.calculate_pair_beta(_sr, _br)
                                        if not np.isnan(_beta):
                                            close_beta = float(_beta)
                        except Exception as e:
                            print(f"\u26a0\ufe0f Fresh beta/pval calc error at close: {e}")
                    close_hl = self._format_half_life(pair_info.half_life) if pair_info.half_life and pair_info.half_life > 0 else 'N/A'
                    
                    full_msg = f"{reason_text}: <b>{s1}/{s2}</b>\n"
                    full_msg += f"🏷️ Tag: <code>{close_reason_tag}</code>\n\n"
                    full_msg += f"📊 Z: {close_zscore:+.2f} | β: {close_beta:.3f} | p: {close_pval:.4f}\n"
                    full_msg += f"⏳ HL: {close_hl} | Hedge: {pair_info.hedge_ratio:.4f}\n"
                    full_msg += f"💵 PnL: {pnl_emoji} <b>{net_pnl:+.2f} USDT</b>\n"
                    full_msg += f"💸 Fees: {_hw_fee1 + _hw_fee2:.4f} USDT\n"
                    full_msg += f"   {e1} {s1}: {pnl1:+.2f} | {e2} {s2}: {pnl2:+.2f}\n"
                    
                    print(full_msg.replace('<b>', '').replace('</b>', ''))
                    reply_to = await self._resolve_reply_to_message_id(pair_info)
                    await self._notify(full_msg, reply_to)
                    
                    # State cleanup for hardware close
                    self._update_pair_quality_penalty_on_close(pair_info, close_reason_tag)
                    self._update_quality_score_cache(pair_info)
                    pair_info.position_status = 0
                    pair_info.qty1 = 0
                    pair_info.qty2 = 0
                    pair_info.entry_price1 = 0
                    pair_info.entry_price2 = 0
                    pair_info.current_trade_id = None
                    self._apply_close_cooldown(pair_info, close_reason_tag)
                    
                    # Update exchange position cache
                    self._exchange_positions_cache.pop(s1, None)
                    self._exchange_positions_cache.pop(s2, None)
                    self._exchange_position_count = len(self._exchange_positions_cache)
                    
                    # WAIT FOR CANDLE: pair-local same-candle guard
                    self.mark_pair_wait_for_next_candle(pair_info, reason=close_reason_tag)
                    
                    # Update DB (includes close_pnl + market neutrality metrics)
                    if pair_info.db_id:
                        await db.update_pair({
                            'id': pair_info.db_id,
                            'position_status': 0,
                            'qty1': 0,
                            'qty2': 0,
                            'entry_price1': 0,
                            'entry_price2': 0,
                            'close_time': int(time.time()),
                            'close_pnl': net_pnl,
                            'close_reason': close_reason_tag,
                            'pnl1': pnl1,
                            'pnl2': pnl2,
                            'fee1': _hw_fee1,
                            'fee2': _hw_fee2,
                            'beta_btc': close_beta,
                            'last_pvalue': close_pval,
                        })
                    
                    # Trigger re-analysis for freed slot
                    self.loop.create_task(self._trigger_immediate_analysis())
                    return
                
                else:
                    # NORMAL CLOSE PATH: cancel orders first, then close both legs
                    # Cancel all open orders (including algo SL/TP) before closing
                    try:
                        results = await asyncio.gather(
                            self.client.cancel_open_orders(symbol=s1),
                            self.client.cancel_open_orders(symbol=s2),
                            return_exceptions=True
                        )
                        for i, res in enumerate(results):
                            if isinstance(res, Exception):
                                print(f"⚠️ Cancel orders error for {[s1, s2][i]}: {res}")
                            else:
                                print(f"🗑️ Cancelled orders for {[s1, s2][i]}")
                    except Exception as e:
                        print(f"⚠️ Could not cancel orders: {e}")

                errors = []  # BUG-3 FIX: initialize before conditional block to prevent NameError
                try:
                    if not is_hardware_close:
                        # Normal path: check which legs still have open positions
                        account = await self.client.account()
                        open_positions = {}
                        for pos in account.get('positions', []):
                            amt = float(pos.get('positionAmt', 0))
                            if amt != 0:
                                open_positions[pos['symbol']] = amt
                        
                        leg1_exists = s1 in open_positions
                        leg2_exists = s2 in open_positions
                        
                        close_tasks = []
                        close_symbols = []
                        
                        if leg1_exists:
                            close_tasks.append(self._close_leg_reduce_only(
                                symbol=s1,
                                side=side1_close,
                                quantity=abs(open_positions[s1])
                            ))
                            close_symbols.append(s1)
                        else:
                            print(f"ℹ️ {s1} already closed, skipping")
                            
                        if leg2_exists:
                            close_tasks.append(self._close_leg_reduce_only(
                                symbol=s2,
                                side=side2_close,
                                quantity=abs(open_positions[s2])
                            ))
                            close_symbols.append(s2)
                        else:
                            print(f"ℹ️ {s2} already closed, skipping")
                        
                        if close_tasks:
                            results = await asyncio.gather(*close_tasks, return_exceptions=True)
                        else:
                            results = []
                        
                        # Check for errors
                        errors = []
                        for i, res in enumerate(results):
                            if isinstance(res, Exception):
                                sym = close_symbols[i]
                                # Simplify error message
                                err_str = str(res)
                                if 'ReduceOnly' in err_str:
                                    errors.append(f"{sym}: already closed")
                                else:
                                    errors.append(f"{sym}: {err_str[:50]}")
                            else:
                                if close_symbols:
                                    print(f"✅ Closed {close_symbols[i]}")
                
                    if errors:
                        err_msg = f"⚠️ Close {s1}-{s2}: {', '.join(errors)}"
                        print(err_msg)
                        await self._notify(err_msg)
                    else:
                        # Exchange is source of truth: finalize close only after both legs are really zero.
                        if not await self._confirm_pair_closed_on_exchange(pair_info, close_reason_tag):
                            return

                        reason_text = CLOSE_REASONS.get(close_reason_tag, '\u2753 Unknown')
                    
                        def get_price(order):
                            if not isinstance(order, dict):
                                return 0.0
                            if 'avgPrice' in order and float(order['avgPrice']) > 0:
                                return float(order['avgPrice'])
                            if 'cummulativeQuoteQty' in order and 'executedQty' in order and float(order['executedQty']) > 0:
                                return float(order['cummulativeQuoteQty']) / float(order['executedQty'])
                            return 0.0

                        # Safely get prices - MAP by symbol (results array may not match s1/s2 order!)
                        close_prices = {}
                        for i, res in enumerate(results):
                            if not isinstance(res, Exception) and i < len(close_symbols):
                                close_prices[close_symbols[i]] = get_price(res)
                        
                        # For legs already closed by exchange (SL/TP trigger), fetch actual close price
                        for sym in [s1, s2]:
                            if sym not in close_prices:
                                try:
                                    start_ms = self._trade_window_start_ms(pair_info)
                                    trades = await self._fetch_account_trades_window(sym, start_ms, max_records=1000)
                                    if trades:
                                        # Last trade price is the close price
                                        close_prices[sym] = float(trades[-1].get('price', 0))
                                        print(f"📊 Fetched close price for {sym} from trades: {close_prices[sym]}")
                                    else:
                                        close_prices[sym] = self.last_prices.get(sym, 0) or (pair_info.entry_price1 if sym == s1 else pair_info.entry_price2)
                                except Exception as e:
                                    print(f"⚠️ Could not fetch close price for {sym}: {e}")
                                    close_prices[sym] = self.last_prices.get(sym, 0) or (pair_info.entry_price1 if sym == s1 else pair_info.entry_price2)
                        
                        close_price1 = close_prices.get(s1, pair_info.entry_price1)
                        close_price2 = close_prices.get(s2, pair_info.entry_price2)
                    
                        # BUG-5 FIX: Use exchange realizedPnl for consistency with other close paths
                        # Manual calc is kept as fallback only
                        pnl1 = 0.0
                        pnl2 = 0.0
                        try:
                            await asyncio.sleep(0.5)  # Brief delay for trade data availability
                            start_ms_pnl = self._trade_window_start_ms(pair_info)
                            trades_s1 = await self._fetch_account_trades_window(s1, start_ms_pnl, max_records=3000)
                            trades_s2 = await self._fetch_account_trades_window(s2, start_ms_pnl, max_records=3000)
                            if trades_s1 or trades_s2:
                                pnl1 = sum(float(t.get('realizedPnl', 0)) for t in trades_s1)
                                pnl2 = sum(float(t.get('realizedPnl', 0)) for t in trades_s2)
                            else:
                                raise ValueError("No trades found, using manual calc")
                        except Exception as pnl_err:
                            print(f"⚠️ Exchange PnL fetch failed ({pnl_err}), using manual calc")
                            side1_dir = 1 if pair_info.position_status == 1 else -1
                            side2_dir = -side1_dir
                            pnl1 = (close_price1 - pair_info.entry_price1) * pair_info.qty1 * side1_dir
                            pnl2 = (close_price2 - pair_info.entry_price2) * pair_info.qty2 * side2_dir
                        total_pnl = pnl1 + pnl2


                        # Calculate fees from recent trades (BEFORE trade record update)
                        _norm_fee1, _norm_fee2 = 0.0, 0.0
                        try:
                            _norm_fee1 = sum(float(t.get('commission', 0)) for t in trades_s1) if trades_s1 else 0.0
                            _norm_fee2 = sum(float(t.get('commission', 0)) for t in trades_s2) if trades_s2 else 0.0
                        except Exception:
                            pass
                        net_pnl = total_pnl - (_norm_fee1 + _norm_fee2)
                        pnl_emoji = "🟢" if net_pnl > 0 else "🔴"

                        # Calculate real-time Z-score BEFORE DB update (was causing UnboundLocalError)
                        close_zscore = 0.0
                        try:
                            p1 = self.last_prices.get(s1, 0)
                            p2 = self.last_prices.get(s2, 0)
                            if p1 > 0 and p2 > 0:
                                close_zscore = self._calc_realtime_zscore(pair_info, p1, p2)
                                import math
                                if math.isnan(close_zscore):
                                    close_zscore = pair_info.last_z_score or 0
                            else:
                                close_zscore = pair_info.last_z_score or 0
                        except Exception:
                            close_zscore = pair_info.last_z_score or 0

                        if pair_info.current_trade_id:
                            await db.close_trade_record(
                                pair_info.current_trade_id,
                                status='CLOSED',
                                close_reason=close_reason_tag,
                                close_price_1=close_price1,
                                close_price_2=close_price2,
                                pnl=net_pnl,
                                close_z=close_zscore if close_zscore else 0.0,
                                fee1=_norm_fee1,
                                fee2=_norm_fee2,
                            )
                            await self._persist_pair_executions(
                                pair_info, trades_s1, trades_s2, phase='CLOSE', trade_id=pair_info.current_trade_id
                            )
                    
                        pair_info.current_trade_id = None
                        pair_info.position_status = 0
                        pair_info.qty1 = 0
                        pair_info.qty2 = 0
                        pair_info.close_handled = True  # Mark as handled to prevent duplicate notification
                        pair_info.last_close_reason = close_reason_tag
                        self._apply_close_cooldown(pair_info, close_reason_tag)
                        pair_info.entry_price1 = 0
                        pair_info.entry_price2 = 0
                        
                        # Update exchange position cache
                        self._exchange_positions_cache.pop(s1, None)
                        self._exchange_positions_cache.pop(s2, None)
                        self._exchange_position_count = len(self._exchange_positions_cache)
                    
                        # Cancel all algo orders for this pair - track per symbol/type
                        cleanup_status = []
                        try:
                            algo_orders = await self.client.get_algo_orders()
                            if isinstance(algo_orders, dict):
                                algo_orders = algo_orders.get('orders', [])
                            if not isinstance(algo_orders, list):
                                algo_orders = []

                            # Track which orders exist for each symbol
                            orders_by_sym = {s1: [], s2: []}
                            for o in algo_orders:
                                sym = o.get('symbol')
                                if not sym:
                                    continue
                                if sym in orders_by_sym:
                                    order_type = str(o.get('type') or o.get('orderType') or '')
                                    order_type_up = order_type.upper()
                                    if 'STOP' in order_type_up:
                                        orders_by_sym[sym].append(('SL', o.get('algoId')))
                                    elif 'TAKE_PROFIT' in order_type_up:
                                        orders_by_sym[sym].append(('TP', o.get('algoId')))
                                    else:
                                        orders_by_sym[sym].append((order_type or 'ORDER', o.get('algoId')))
                            
                            # Cancel each order and track result
                            for sym in [s1, s2]:
                                for order_type, algo_id in orders_by_sym[sym]:
                                    if algo_id is None:
                                        cleanup_status.append(f"  ⚠️ {sym} {order_type} - missing algoId")
                                        continue
                                    try:
                                        await self.client.cancel_algo_order(algoId=algo_id)
                                        cleanup_status.append(f"  ✅ {sym} {order_type} cancelled")
                                    except Exception as e:
                                        cleanup_status.append(f"  ⚠️ {sym} {order_type} - {str(e)[:20]}")
                            
                            if not orders_by_sym[s1] and not orders_by_sym[s2]:
                                cleanup_status.append("  ℹ️ No orders found")
                                
                        except Exception as e:
                            cleanup_status.append(f"  ❌ Failed: {str(e)[:30]}")
                        
                        # Use beta_at_trigger if available (set by beta_drift/beta_critical close)
                        # This prevents confusing TG messages showing current (already-changed) beta
                        close_beta = getattr(pair_info, '_beta_at_trigger', None)
                        if close_beta is None:
                            close_beta = getattr(pair_info, 'beta_btc', 0) or 0
                        else:
                            pair_info._beta_at_trigger = None  # Reset after use
                        
                        # Per-position PnL with emoji
                        e1 = '🟢' if pnl1 >= 0 else '🔴'
                        e2 = '🟢' if pnl2 >= 0 else '🔴'
                        
                        # Build enhanced close message
                        cleanup_msg = "\n".join(cleanup_status) if cleanup_status else "  ℹ️ No cleanup needed"
                        close_pval = getattr(pair_info, 'last_pvalue', 0) or 0
                        
                        # Recalculate beta & p-value fresh if they're 0 (stale after restart)
                        if (close_beta == 0 or close_pval == 0) and isinstance(self.all_data, dict) and s1 in self.all_data and s2 in self.all_data:
                            try:
                                _d1 = self.all_data[s1]
                                _d2 = self.all_data[s2]
                                if len(_d1.close) >= self.min_data_points and len(_d2.close) >= self.min_data_points:
                                    _lp1 = np.log(list(_d1.close)[-self.min_data_points:])
                                    _lp2 = np.log(list(_d2.close)[-self.min_data_points:])
                                    _, _, _, _pval = utils.calculate_cointegration(_lp1, _lp2, strict_hl=False)
                                    if close_pval == 0 and not np.isnan(_pval):
                                        close_pval = float(_pval)
                                    if close_beta == 0 and isinstance(self.all_data, dict) and 'BTCUSDT' in self.all_data:
                                        _btc = self.all_data['BTCUSDT']
                                        if len(_btc.close) >= self.min_data_points:
                                            _lbtc = np.log(list(_btc.close)[-self.min_data_points:])
                                            _sr = np.diff(_lp1) - pair_info.hedge_ratio * np.diff(_lp2)
                                            _br = np.diff(_lbtc)
                                            _beta = utils.calculate_pair_beta(_sr, _br)
                                            if not np.isnan(_beta):
                                                close_beta = float(_beta)
                            except Exception as e:
                                print(f"\u26a0\ufe0f Fresh beta/pval calc error at close: {e}")
                        close_hl = self._format_half_life(pair_info.half_life) if pair_info.half_life and pair_info.half_life > 0 else 'N/A'
                        full_msg = f"{reason_text}: <b>{s1}/{s2}</b>\n"
                        full_msg += f"🏷️ Tag: <code>{close_reason_tag}</code>\n\n"
                        full_msg += f"📊 Z: {close_zscore:+.2f} | β: {close_beta:.3f} | p: {close_pval:.4f}\n"
                        full_msg += f"⏳ HL: {close_hl} | Hedge: {pair_info.hedge_ratio:.4f}\n"
                        full_msg += f"💵 PnL: {pnl_emoji} <b>{net_pnl:+.2f} USDT</b>\n"
                        full_msg += f"   {e1} {s1}: {pnl1:+.2f} | {e2} {s2}: {pnl2:+.2f}\n"
                        full_msg += f"💸 Fees: {_norm_fee1 + _norm_fee2:.4f} USDT\n\n"
                        full_msg += f"🛡️ Order Cleanup:\n{cleanup_msg}"
                        
                        print(full_msg.replace('<b>', '').replace('</b>', ''))
                        # Reply to original open message if available
                        reply_to = await self._resolve_reply_to_message_id(pair_info)
                        await self._notify(full_msg, reply_to)
                        
                        # Update DB with close details + market neutrality metrics
                        self._update_pair_quality_penalty_on_close(pair_info, close_reason_tag)
                        self._update_quality_score_cache(pair_info)
                        if pair_info.db_id:
                            await db.update_pair({
                                'id': pair_info.db_id,
                                'position_status': 0,
                                'qty1': 0,
                                'qty2': 0,
                                'entry_price1': 0,
                                'entry_price2': 0,
                                'close_time': int(time.time()),
                                'close_pnl': net_pnl,
                                'close_reason': close_reason_tag,
                                'pnl1': pnl1,
                                'pnl2': pnl2,
                                'fee1': _norm_fee1,
                                'fee2': _norm_fee2,
                                'beta_btc': close_beta,
                                'last_pvalue': close_pval,
                            })

                        
                        # AUTO-ADD to best_pairs.json on successful TP only
                        # BUG-7 FIX: Don't add pairs from forced closes (circuit, beta_drift, etc.)
                        if close_reason_tag in ('z_tp', 'hardware_tp'):
                            self._add_to_best_pairs(s1, s2)
                        
                        # WAIT FOR CANDLE: re-entry block is ALWAYS pair-local.
                        # Only this closed pair is blocked; other pairs may still open.
                        self.mark_pair_wait_for_next_candle(pair_info, reason=close_reason_tag)
                        
                        # IMMEDIATE RE-ANALYSIS: Trigger search for new trades now that slot is free
                        print(f"🔄 Slot freed after closing {s1}-{s2}. Triggering immediate re-analysis...")
                        self.loop.create_task(self._trigger_immediate_analysis())
                        
                except Exception as e:
                    print(f"FATAL ERROR closing position for {s1}-{s2}: {e}")
                    # Conservative path: finalize close only if exchange confirms both legs are zero.
                    try:
                        close_confirmed = await self._confirm_pair_closed_on_exchange(pair_info, close_reason_tag)
                        if close_confirmed:
                            if pair_info.current_trade_id:
                                try:
                                    await db.close_trade_record(
                                        pair_info.current_trade_id,
                                        status='CLOSED_ERROR',
                                        close_reason=close_reason_tag,
                                    )
                                    pair_info.current_trade_id = None
                                except Exception as trade_close_err:
                                    print(f"⚠️ Failed to mark trade CLOSED_ERROR for {s1}-{s2}: {trade_close_err}")
                            pair_info.position_status = 0
                            pair_info.qty1 = 0
                            pair_info.qty2 = 0
                            pair_info.entry_price1 = 0
                            pair_info.entry_price2 = 0
                            pair_info.close_handled = True
                            pair_info.last_close_reason = close_reason_tag
                            self._apply_close_cooldown(pair_info, close_reason_tag)
                            self.mark_pair_wait_for_next_candle(pair_info, reason=close_reason_tag)
                            self._exchange_positions_cache.pop(s1, None)
                            self._exchange_positions_cache.pop(s2, None)
                            self._exchange_position_count = len(self._exchange_positions_cache)
                            err_msg = (
                                f"⚠️ Close error {s1}-{s2} [tag={close_reason_tag}]: "
                                f"{type(e).__name__}: {e}\n"
                                f"Position is closed on exchange, state saved as CLOSED_ERROR."
                            )
                        else:
                            err_msg = (
                                f"⚠️ Close flow failed {s1}-{s2} [tag={close_reason_tag}]: "
                                f"{type(e).__name__}: {e}\n"
                                f"Pair remains OPEN until exchange confirms full close."
                            )
                        reply_to = await self._resolve_reply_to_message_id(pair_info)
                        await self._notify(err_msg, reply_to)
                    except Exception:
                        pass
                return

            hedge = pair_info.hedge_ratio
            s1_info = self.all_symbols.get(s1)
            s2_info = self.all_symbols.get(s2)

            if not s1_info or not s2_info:
                missing = s1 if not s1_info else s2
                print(
                    f"⛔ Symbol info not found for {missing} (not in all_symbols). "
                    f"Pair {s1}-{s2} skipped. Will retry after next symbol refresh."
                )
                # Do NOT touch position_status (pair is idle, not open).
                # Apply a cooldown so the pair doesn't spam entry attempts
                # until all_symbols is refreshed (every ~1h in load_symbols_loop).
                pair_info.is_trading = False
                pair_info.pending_signal = None
                pair_info.pending_since = None
                pair_info.pending_source = ''
                pair_info._leverage_fail_until = time.time() + 900  # 15-min cooldown
                return

            try:
                s1_price = float((await self.client.ticker_price(s1))['price'])
                s2_price = float((await self.client.ticker_price(s2))['price'])
            except Exception as e:
                print(f"ERROR: Could not fetch prices for {s1}/{s2}. E: {e}")
                pair_info.position_status = 0
                return
            
            data1 = self.all_data.get(s1)
            data2 = self.all_data.get(s2)
        
            if data1 is None or data2 is None:
                print(f"ERROR: Data not found for {s1} or {s2} during execution.")
                pair_info.position_status = 0
                return

            # BUG-4 FIX: Use self.min_data_points instead of hardcoded COINT_WINDOW
            # Ensures same lookback window as _calc_realtime_zscore and _check_signals
            log_prices1 = np.log(list(data1.close)[-self.min_data_points:])
            log_prices2 = np.log(list(data2.close)[-self.min_data_points:])

            # === FRESH HEDGE RATIO: Recalculate from current data before sizing ===
            # Between signal detection and execution, hedge_ratio may have drifted.
            # This also serves as a final cointegration gate — abort if pair broke.
            try:
                p_val_thresh = getattr(self.config, 'p_value_threshold', 0.05) or 0.05
                fresh_flag, fresh_hedge, fresh_hl, fresh_pval = utils.calculate_cointegration(
                    log_prices1, log_prices2, p_value_threshold=p_val_thresh, strict_hl=False
                )
                if fresh_flag == 1 and not np.isnan(fresh_hedge):
                    old_hedge = pair_info.hedge_ratio
                    if abs(fresh_hedge - old_hedge) > 0.001:
                        print(f"🔄 Hedge refresh for {s1}-{s2}: {old_hedge:.4f} → {fresh_hedge:.4f}")
                    hedge = fresh_hedge
                    pair_info.hedge_ratio = fresh_hedge
                    pair_info.last_pvalue = fresh_pval
                else:
                    print(f"⚠️ Fresh cointegration FAILED for {s1}-{s2} (flag={fresh_flag}, p={fresh_pval:.4f}). Aborting trade.")
                    pair_info.position_status = 0
                    pair_info.is_trading = False
                    pair_info.pending_signal = None
                    pair_info.pending_since = None
                    pair_info.pending_source = ''
                    return
            except Exception as e:
                print(f"⚠️ Hedge refresh error for {s1}-{s2}: {e}. Using existing hedge={hedge:.4f}")

            # === STRICT BETA CHECK per user request: Enforce beta_threshold=0.11 ===
            try:
                beta_threshold = getattr(self.config, 'beta_threshold', 0.11) or 0.11
                current_beta = getattr(pair_info, 'beta_btc', 0.0)
                btc_data_ready = False
                test_mode = getattr(self.config, 'test_mode', False)
                if isinstance(test_mode, str):
                    test_mode = test_mode.lower() in ('true', '1', 'yes')
                # Recalculate beta right before entry to avoid stale neutrality check.
                if 'BTCUSDT' in self.all_data:
                    btc_data = self.all_data.get('BTCUSDT')
                    if btc_data and len(btc_data.close) >= self.min_data_points:
                        btc_data_ready = True

                        log_btc = np.log(list(btc_data.close)[-self.min_data_points:])
                        spread_returns = np.diff(log_prices1) - hedge * np.diff(log_prices2)
                        btc_returns = np.diff(log_btc)
                        fresh_beta = utils.calculate_pair_beta(spread_returns, btc_returns)
                        if not np.isnan(fresh_beta):
                            current_beta = float(fresh_beta)
                            pair_info.beta_btc = current_beta
                if not btc_data_ready and not test_mode:
                    warn_msg = f"⛔ BETA CHECK SKIPPED: BTCUSDT data not ready for {s1}-{s2}. Aborting entry."
                    print(warn_msg)
                    pair_info.position_status = 0
                    pair_info.is_trading = False
                    pair_info.pending_signal = None
                    pair_info.pending_since = None
                    pair_info.pending_source = ''
                    # Avoid retry spam in same candle when beta context is unavailable.
                    self.mark_pair_wait_for_next_candle(pair_info, reason='beta_data_not_ready')
                    return
                if not np.isnan(current_beta) and abs(current_beta) >= beta_threshold:
                    warn_msg = f"⛔ BETA REJECT: {s1}-{s2} beta={current_beta:.3f} >= {beta_threshold}. Aborting entry."
                    print(warn_msg)
                    pair_info.position_status = 0
                    pair_info.is_trading = False
                    pair_info.pending_signal = None
                    pair_info.pending_since = None
                    pair_info.pending_source = ''
                    # Pair-level cooldown: retry only after next closed candle.
                    self.mark_pair_wait_for_next_candle(pair_info, reason='beta_reject')
                    return
            except Exception as e:
                print(f"⚠️ Beta check error: {e}")

            # === HEDGE RATIO BOUNDS CHECK ===
            # Prevent opening wildly unbalanced positions (e.g. $5 vs $92)
            try:
                hedge_min = getattr(self.config, 'hedge_min', 0.3) or 0.3
                hedge_max = getattr(self.config, 'hedge_max', 3.0) or 3.0
                abs_hedge = abs(hedge) if not np.isnan(hedge) else 0.0
                if abs_hedge < hedge_min or abs_hedge > hedge_max:
                    warn_msg = f"⛔ HEDGE REJECT: {s1}-{s2} |hedge|={abs_hedge:.4f} outside [{hedge_min}, {hedge_max}]. Positions would be unbalanced. Aborting entry."
                    print(warn_msg)
                    pair_info.position_status = 0
                    pair_info.is_trading = False
                    pair_info.pending_signal = None
                    pair_info.pending_since = None
                    pair_info.pending_source = ''
                    reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                    await self._notify(warn_msg, reply_to)
                    return
            except Exception as e:
                print(f"⚠️ Hedge bounds check error: {e}")

            capital = self.config.capital if self.config and self.config.capital else 1000.0
            max_notional = self.config.max_notional_pct if self.config and self.config.max_notional_pct else 0.1

            dollar1, dollar2 = utils.vol_parity_notional(
                log_prices1, 
                log_prices2, 
                hedge,
                capital=capital,
                max_notional_per_pair=max_notional
            )
        
            qty1_dollar = dollar1 * direction
            qty2_dollar = dollar2 * -direction

            qty1 = qty1_dollar / s1_price
            qty2 = qty2_dollar / s2_price
            side1 = 'BUY' if qty1 > 0 else 'SELL'
            side2 = 'BUY' if qty2 > 0 else 'SELL'
            qty1_rounded = utils.round_down(abs(qty1), s1_info.step_size)
            qty2_rounded = utils.round_down(abs(qty2), s2_info.step_size)

            min_notional1 = s1_info.notional * 1.1
            min_notional2 = s2_info.notional * 1.1
            calculated_notional1 = qty1_rounded * s1_price
            calculated_notional2 = qty2_rounded * s2_price
            
            # Get max_order_bump threshold from config (default 1.5)
            max_order_bump = getattr(self.config, 'max_order_bump', 1.5) or 1.5
            
            # Check if we should skip trade due to size constraints
            if utils.should_skip_trade(min_notional1, calculated_notional1, max_order_bump):
                print(f"SKIP: Trade for {s1}-{s2} cancelled - {s1} below min notional with excessive bump required")
                cooldown_sec = int(getattr(self.config, 'min_notional_skip_cooldown_sec', 900) or 900)
                pair_info.pending_signal = None
                pair_info.pending_since = None
                pair_info.pending_source = ''
                pair_info._leverage_fail_until = time.time() + cooldown_sec
                self._set_symbol_cooldown(s1, cooldown_sec, 'min_notional_bump')
                pair_info.position_status = 0
                pair_info.is_trading = False
                return
            
            if utils.should_skip_trade(min_notional2, calculated_notional2, max_order_bump):
                print(f"SKIP: Trade for {s1}-{s2} cancelled - {s2} below min notional with excessive bump required")
                cooldown_sec = int(getattr(self.config, 'min_notional_skip_cooldown_sec', 900) or 900)
                pair_info.pending_signal = None
                pair_info.pending_since = None
                pair_info.pending_source = ''
                pair_info._leverage_fail_until = time.time() + cooldown_sec
                self._set_symbol_cooldown(s2, cooldown_sec, 'min_notional_bump')
                pair_info.position_status = 0
                pair_info.is_trading = False
                return
            
            # Apply small bumps if needed (within threshold)
            if calculated_notional1 < min_notional1:
                print(f"INFO: {s1} qty slightly bumped to meet min notional")
                qty1_rounded = utils.round_up(min_notional1 / s1_price, s1_info.step_size)
        
            if calculated_notional2 < min_notional2:
                print(f"INFO: {s2} qty slightly bumped to meet min notional")
                qty2_rounded = utils.round_up(min_notional2 / s2_price, s2_info.step_size)

            # === REBALANCE: Maintain vol parity ratio after min_notional bumps ===
            # If either leg was bumped to meet min notional, scale the other
            # leg by the same factor to preserve the vol parity hedge ratio
            final_notional1 = qty1_rounded * s1_price
            final_notional2 = qty2_rounded * s2_price
            
            if calculated_notional1 > 0 and calculated_notional2 > 0:
                # Calculate bump factor for each leg (1.0 = no bump)
                bump1 = final_notional1 / calculated_notional1
                bump2 = final_notional2 / calculated_notional2
                
                # If either leg was bumped (>1% tolerance for rounding), apply max factor to both
                max_bump = max(bump1, bump2)
                if max_bump > 1.01:
                    # Scale up the leg that had a smaller bump
                    if bump1 < max_bump:
                        needed1 = calculated_notional1 * max_bump
                        new_qty1 = utils.round_up(needed1 / s1_price, s1_info.step_size)
                        if new_qty1 > qty1_rounded:
                            print(f"INFO: Rebalanced {s1} ${final_notional1:.2f} → ${new_qty1 * s1_price:.2f} (bump {max_bump:.2f}x)")
                            qty1_rounded = new_qty1
                    if bump2 < max_bump:
                        needed2 = calculated_notional2 * max_bump
                        new_qty2 = utils.round_up(needed2 / s2_price, s2_info.step_size)
                        if new_qty2 > qty2_rounded:
                            print(f"INFO: Rebalanced {s2} ${final_notional2:.2f} → ${new_qty2 * s2_price:.2f} (bump {max_bump:.2f}x)")
                            qty2_rounded = new_qty2
                    
                    # Safety: total notional shouldn't exceed 3x the original pair budget
                    total_after = qty1_rounded * s1_price + qty2_rounded * s2_price
                    pair_budget = capital * max_notional
                    if total_after > pair_budget * 3:
                        print(f"SKIP: Rebalanced total ${total_after:.2f} exceeds 3x pair budget ${pair_budget:.2f} for {s1}-{s2}")
                        pair_info.position_status = 0
                        pair_info.is_trading = False
                        return

            print(f"EXECUTING TRADE for {s1}-{s2}:")
            print(f"  {side1} {qty1_rounded} {s1} at {s1_price}")
            print(f"  {side2} {qty2_rounded} {s2} at {s2_price}")

            # === PRE-FLIGHT: Validate both positions can be opened ===
            try:
                preflight_ok = True
                failed_preflight_symbol = None
                for sym, qty, price in [(s1, qty1_rounded, s1_price), (s2, qty2_rounded, s2_price)]:
                    notional = qty * price
                    brackets = await self.client.leverage_brackets(symbol=sym)
                    if brackets:
                        # Find the bracket for our leverage level
                        bracket_data = brackets[0] if isinstance(brackets, list) else brackets
                        bracket_list = bracket_data.get('brackets', [])
                        bracket_notional_cap = None
                        for b in bracket_list:
                            if b.get('initialLeverage', 0) >= leverage:
                                bracket_notional_cap = float(b.get('notionalCap', 0))
                                break
                        
                        if bracket_notional_cap and notional > bracket_notional_cap:
                            print(f"🚫 PRE-FLIGHT FAIL: {sym} notional ${notional:.2f} exceeds max ${bracket_notional_cap:.2f} at {leverage}x leverage")
                            preflight_ok = False
                            failed_preflight_symbol = sym
                            break
                        
                        # Also check if leverage is even supported
                        max_lev = max((b.get('initialLeverage', 0) for b in bracket_list), default=0)
                        if leverage > max_lev:
                            print(f"🚫 PRE-FLIGHT FAIL: {sym} max leverage is {max_lev}x, requested {leverage}x")
                            preflight_ok = False
                            failed_preflight_symbol = sym
                            break
                
                if not preflight_ok:
                    print(f"🚫 Trade aborted for {s1}-{s2}: pre-flight validation failed")
                    pair_info.position_status = 0
                    pair_info.is_trading = False
                    pair_info.pending_signal = None
                    pair_info.pending_since = None
                    pair_info.pending_source = ''
                    pair_info._leverage_fail_until = time.time() + 600  # 10 min cooldown
                    if failed_preflight_symbol:
                        self._set_symbol_cooldown(failed_preflight_symbol, 900, 'preflight_limit')
                    return
                    
            except Exception as e:
                print(f"⚠️ Pre-flight check warning (proceeding): {e}")

            try:
                task1 = self.loop.create_task(
                    self.client.new_order(symbol=s1, side=side1, type='MARKET', quantity=qty1_rounded, newOrderRespType='RESULT')
                )
                task2 = self.loop.create_task(
                    self.client.new_order(symbol=s2, side=side2, type='MARKET', quantity=qty2_rounded, newOrderRespType='RESULT')
                )
            
                results = await asyncio.gather(task1, task2, return_exceptions=True)
                has_error = False
                executed_orders = []
                failed_symbols = []
                for idx, res in enumerate(results):
                    if isinstance(res, Exception):
                        print(f"ERROR placing order: {res}")
                        has_error = True
                        failed_symbols.append(s1 if idx == 0 else s2)
                    else:
                        executed_orders.append(res)
            
                if has_error:
                    print("ERROR: Orders failed. Reverting executed legs...")
                    
                    # CRITICAL: Prevent WS handler from spamming "Manual Close" during revert
                    pair_info.close_handled = True
                    pair_info.is_trading = True
                    
                    revert_tasks = []
                    for executed in executed_orders:
                        try:
                            exec_symbol = executed['symbol']
                            exec_qty = float(executed['executedQty'])
                            exec_side = executed['side']
                            revert_side = 'SELL' if exec_side == 'BUY' else 'BUY'
                            revert_tasks.append(
                                self._close_leg_reduce_only(
                                    symbol=exec_symbol,
                                    side=revert_side,
                                    quantity=exec_qty
                                )
                            )
                        except Exception as rev_e:
                            print(f"  CRITICAL: Failed to prepare revert {exec_symbol}: {rev_e}")
                
                    if revert_tasks:
                        revert_results = await asyncio.gather(*revert_tasks, return_exceptions=True)
                        for rr in revert_results:
                            if isinstance(rr, Exception):
                                print(f"  WARNING: Revert order error: {rr}")

                    # Verify no residual positions remain after rollback; force-close if needed
                    try:
                        verify_positions = await self.client.get_position_risk()
                        for vp in verify_positions:
                            sym = vp.get('symbol')
                            if sym not in (s1, s2):
                                continue
                            amt = float(vp.get('positionAmt', 0))
                            if amt == 0:
                                continue
                            close_side = 'SELL' if amt > 0 else 'BUY'
                            await self._close_leg_reduce_only(
                                symbol=sym,
                                side=close_side,
                                quantity=abs(amt)
                            )
                            print(f"  Emergency rollback close executed for {sym} (qty={abs(amt)})")
                    except Exception as verify_err:
                        print(f"  WARNING: Rollback verification failed: {verify_err}")

                    residual_legs = []
                    try:
                        post_verify_positions = await self.client.get_position_risk()
                        for vp in post_verify_positions:
                            sym = vp.get('symbol')
                            if sym not in (s1, s2):
                                continue
                            amt = float(vp.get('positionAmt', 0))
                            if amt != 0:
                                residual_legs.append((sym, amt))
                    except Exception as post_verify_err:
                        print(f"  WARNING: Post-rollback verification failed: {post_verify_err}")
                    if residual_legs:
                        details = ", ".join([f"{sym}:{amt:+g}" for sym, amt in residual_legs])
                        print(f"  CRITICAL: Residual legs remain after rollback for {s1}-{s2}: {details}")
                        await self._notify(
                            f"🚨 <b>Rollback residual detected</b>\n"
                            f"Pair: <b>{s1}/{s2}</b>\n"
                            f"Residual: <code>{details}</code>\n"
                            f"Bot will retry cleanup via sync loop."
                        )
                
                    pair_info.position_status = 0
                    pair_info.is_trading = False
                    pair_info.close_handled = False  # Reset after revert completes
                    
                    # Cooldown to prevent immediate retry loop
                    pair_info.pending_signal = None
                    pair_info.pending_since = None
                    pair_info.pending_source = ''
                    pair_info._leverage_fail_until = time.time() + 600  # 10 min cooldown
                    
                    # Block failed symbols to stop retrying same bad symbol across pair combinations
                    if failed_symbols:
                        insufficient_markers = (
                            'insufficient', 'margin', 'balance', 'not enough', 'min notional',
                            'notional', 'max position', 'position limit', 'risk limit'
                        )
                        for idx, res in enumerate(results):
                            if not isinstance(res, Exception):
                                continue
                            msg = str(res).lower()
                            if any(marker in msg for marker in insufficient_markers):
                                self._set_symbol_cooldown(s1 if idx == 0 else s2, 900, 'insufficient_capital')
                else:
                    # DC-3 FIX: Set position_status ONLY after confirmed order success
                    # (removed tentative set from inside _trade_lock to prevent phantom state)
                    pair_info.position_status = direction
                    pair_info.qty1 = float(executed_orders[0]['executedQty'])
                    pair_info.qty2 = float(executed_orders[1]['executedQty'])
                    
                    # CRITICAL: Reset failure flags and set trade open time
                    # Prevents stale state from instantly closing new trades
                    pair_info._beta_critical_triggered = False
                    pair_info._beta_at_trigger = None
                    pair_info._trade_open_time = time.time()
                    pair_info.entry_half_life_bars = float(getattr(pair_info, 'half_life', 0.0) or 0.0)
                    pair_info.entry_hold_limit_bars = self._compute_entry_hold_limit_bars(pair_info.entry_half_life_bars)
                    
                    # CRITICAL: Update exchange position cache immediately
                    # This prevents race condition where another task checks limit before cache refreshes
                    self._exchange_positions_cache[s1] = pair_info.qty1
                    self._exchange_positions_cache[s2] = pair_info.qty2
                    self._exchange_position_count = len(self._exchange_positions_cache)
                
                    def get_price(order):
                        if not isinstance(order, dict):
                            return 0.0
                        if 'avgPrice' in order and float(order['avgPrice']) > 0:
                            return float(order['avgPrice'])
                        if 'cummulativeQuoteQty' in order and 'executedQty' in order and float(order['executedQty']) > 0:
                            return float(order['cummulativeQuoteQty']) / float(order['executedQty'])
                        return 0.0

                    pair_info.entry_price1 = get_price(executed_orders[0])
                    pair_info.entry_price2 = get_price(executed_orders[1])
                    
                    # Set open time
                    from datetime import datetime
                    pair_info.open_time = int(time.time())
                    open_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Determine which leg is LONG/SHORT
                    if direction == 1:  # Long spread: BUY s1, SELL s2
                        long_sym, long_qty, long_price = s1, pair_info.qty1, pair_info.entry_price1
                        short_sym, short_qty, short_price = s2, pair_info.qty2, pair_info.entry_price2
                    else:  # Short spread: SELL s1, BUY s2
                        short_sym, short_qty, short_price = s1, pair_info.qty1, pair_info.entry_price1
                        long_sym, long_qty, long_price = s2, pair_info.qty2, pair_info.entry_price2
                    
                    # Calculate beta if not already set (for newly discovered pairs)
                    if pair_info.beta_btc == 0.0 and 'BTCUSDT' in self.all_data:
                        try:
                            data1 = self.all_data.get(s1)
                            data2 = self.all_data.get(s2)
                            btc_data = self.all_data['BTCUSDT']
                            if data1 and data2 and len(btc_data.close) >= self.min_data_points:
                                log1 = np.log(list(data1.close)[-self.min_data_points:])
                                log2 = np.log(list(data2.close)[-self.min_data_points:])
                                log_btc = np.log(list(btc_data.close)[-self.min_data_points:])
                                spread_returns = np.diff(log1) - pair_info.hedge_ratio * np.diff(log2)
                                btc_returns = np.diff(log_btc)
                                beta = utils.calculate_pair_beta(spread_returns, btc_returns)
                                if not np.isnan(beta):
                                    pair_info.beta_btc = beta
                        except Exception as e:
                            print(f"⚠️ Beta calculation error: {e}")
                    
                    entry_hours = float(getattr(pair_info, 'entry_expected_hours', 0.0) or 0.0)
                    streak_bars = int(getattr(pair_info, 'entry_coint_streak_bars', 0) or 0)
                    candles_per_day = float(utils.CANDLES_PER_DAY.get(str(self.timeframe), 24) or 24)
                    bar_hours = 24.0 / candles_per_day if candles_per_day > 0 else 1.0
                    streak_hours = streak_bars * bar_hours
                    hl_bars = float(getattr(pair_info, 'half_life', 0.0) or 0.0)
                    streak_hl_ratio = (streak_bars / hl_bars) if hl_bars > 0 else 0.0

                    success_msg = (f"🚀 <b>Trade OPENED:</b> {s1}-{s2}\n"
                                   f"📅 {open_dt}\n\n"
                                   f"📈 LONG: {long_qty} {long_sym} @ {long_price:.4f}\n"
                                   f"     💰 ${long_qty * long_price:.2f}\n"
                                   f"📉 SHORT: {short_qty} {short_sym} @ {short_price:.4f}\n"
                                   f"     💰 ${short_qty * short_price:.2f}\n\n"
                                   f"⚖️ Hedge: {pair_info.hedge_ratio:.4f} | Z: {pair_info.entry_z_score:.2f}\n"
                                   f"📊 Beta: {pair_info.beta_btc:.3f} | p-value: {pair_info.last_pvalue:.4f}\n"
                                   # Format half-life as readable hours/days
                                   f"⏳ Half-life: {self._format_half_life(pair_info.half_life)}\n"
                                   f"🧠 E[T→|Z| target]: {entry_hours:.1f}h\n"
                                   f"🔁 Coint stability: {streak_bars} bars ({streak_hours:.1f}h, {streak_hl_ratio:.2f}x HL)")
                    print(success_msg.replace('<b>', '').replace('</b>', ''))
                    # Save msg_id for reply threading on close
                    msg_id = await self._notify(success_msg)
                    if msg_id:
                        pair_info.tg_message_id = msg_id
                        # Update in DB
                        if pair_info.db_id:
                            await db.update_pair({'id': pair_info.db_id, 'tg_message_id': msg_id})
                
                    # === HARDWARE SL/TP PLACEMENT ===
                    try:
                        # Calculate ATR for each symbol
                        atr1 = utils.calculate_atr(
                            list(data1.high), 
                            list(data1.low), 
                            list(data1.close)
                        )
                        atr2 = utils.calculate_atr(
                            list(data2.high), 
                            list(data2.low), 
                            list(data2.close)
                        )
                        
                        # Determine side for each leg
                        leg1_side = 'LONG' if direction == 1 else 'SHORT'
                        leg2_side = 'SHORT' if direction == 1 else 'LONG'
                        
                        # Calculate SL prices
                        sl1, tp1, sl1_pct, tp1_pct = utils.calculate_hardware_stops(
                            pair_info.entry_price1, leg1_side, atr1, self.config
                        )
                        sl2, tp2, sl2_pct, tp2_pct = utils.calculate_hardware_stops(
                            pair_info.entry_price2, leg2_side, atr2, self.config
                        )
                        
                        # Round stop prices to tick_size (tick_size is int: digit count)
                        sl1 = round(sl1, s1_info.tick_size)
                        sl2 = round(sl2, s2_info.tick_size)
                        
                        # Determine close sides
                        sl_side1 = 'SELL' if direction == 1 else 'BUY'
                        sl_side2 = 'BUY' if direction == 1 else 'SELL'
                        
                        # Round TP prices to tick_size
                        tp1 = round(tp1, s1_info.tick_size)
                        tp2 = round(tp2, s2_info.tick_size)
                        

                        print(f"🛡️ Placing SL/TP (Algo): {s1} SL@{sl1} TP@{tp1}, {s2} SL@{sl2} TP@{tp2}")

                        if sl1 <= 0 or sl2 <= 0 or tp1 <= 0 or tp2 <= 0:
                            warn_msg = (f"WARN: CRITICAL: Invalid protection prices for {s1}-{s2}! "
                                       f"sl1={sl1}, tp1={tp1}, sl2={sl2}, tp2={tp2}. Force closing position.")
                            print(warn_msg)
                            print(f"  Entry prices: {pair_info.entry_price1}, {pair_info.entry_price2}")
                            print(f"  ATR values: {atr1}, {atr2}")
                            reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                            await self._notify(warn_msg, reply_to)
                            pair_info.close_handled = True
                            pair_info.is_trading = True
                            await self._execute_trade(pair_info, 0, close_reason='hardware_sl')
                            return

                        protection_tasks = [
                            self.client.new_algo_order(symbol=s1, side=sl_side1, type='STOP_MARKET',
                                                       triggerPrice=sl1, quantity=pair_info.qty1, reduceOnly='true'),
                            self.client.new_algo_order(symbol=s2, side=sl_side2, type='STOP_MARKET',
                                                       triggerPrice=sl2, quantity=pair_info.qty2, reduceOnly='true'),
                            self.client.new_algo_order(symbol=s1, side=sl_side1, type='TAKE_PROFIT_MARKET',
                                                       triggerPrice=tp1, quantity=pair_info.qty1, reduceOnly='true'),
                            self.client.new_algo_order(symbol=s2, side=sl_side2, type='TAKE_PROFIT_MARKET',
                                                       triggerPrice=tp2, quantity=pair_info.qty2, reduceOnly='true'),
                        ]
                        task_meta = [(s1, 'STOP'), (s2, 'STOP'), (s1, 'TAKE_PROFIT'), (s2, 'TAKE_PROFIT')]

                        results = await asyncio.gather(*protection_tasks, return_exceptions=True)

                        successful_algo_ids = []
                        failed_count = 0
                        for res in results:
                            if isinstance(res, Exception):
                                print(f"WARN: Failed to place protection order: {res}")
                                failed_count += 1
                            elif isinstance(res, dict) and 'algoId' in res:
                                successful_algo_ids.append(res['algoId'])

                        expected_orders = len(task_meta)
                        if failed_count == 0 and len(successful_algo_ids) == expected_orders:
                            print("🛡️ Protection placed successfully (4 orders)")
                            pair_key = frozenset([s1, s2])
                            for aid, (sym, typ) in zip(successful_algo_ids, task_meta):
                                aid_str = str(aid)
                                self.algo_orders[aid_str] = {'pair_key': pair_key, 'symbol': sym, 'type': typ}
                        else:
                            warn_msg = (
                                f"WARN: CRITICAL: Protection incomplete for {s1}-{s2} "
                                f"(ok={len(successful_algo_ids)}/{max(1, expected_orders)}, failed={failed_count}). Force closing!"
                            )
                            print(warn_msg)
                            reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                            await self._notify(warn_msg, reply_to)
                            
                            if successful_algo_ids:
                                try:
                                    cancel_tasks = [self.client.cancel_algo_order(algoId=aid) for aid in successful_algo_ids]
                                    await asyncio.gather(*cancel_tasks, return_exceptions=True)
                                    print(f"INFO: Cancelled {len(successful_algo_ids)} partial algo orders")
                                except Exception as ce:
                                    print(f"WARN: Could not cancel partial orders: {ce}")
                            
                            pair_info.close_handled = True
                            pair_info.is_trading = True
                            await self._execute_trade(pair_info, 0, close_reason='hardware_sl')
                            
                    except Exception as e:
                        warn_msg = f"WARN: CRITICAL ERROR placing hardware protection for {s1}-{s2}: {e}. Force closing position!"
                        print(warn_msg)
                        reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                        await self._notify(warn_msg, reply_to)
                        
                        pair_info.close_handled = True
                        pair_info.is_trading = True
                        await self._execute_trade(pair_info, 0, close_reason='hardware_sl')
                    # === END HARDWARE SL/TP ===
                    # If entry got auto-closed during protection placement, skip OPEN persistence.
                    if pair_info.position_status == 0:
                        print(f"⚠️ {s1}-{s2}: position already closed during protection flow, skipping OPEN trade persistence.")
                        return
                
                    if pair_info.db_id:
                        # Await DB update for safety (includes market neutrality metrics)
                        try:
                            await db.update_pair({
                                'id': pair_info.db_id,
                                'position_status': pair_info.position_status,
                                'qty1': pair_info.qty1,
                                'qty2': pair_info.qty2,
                                'entry_price1': pair_info.entry_price1,
                                'entry_price2': pair_info.entry_price2,
                                'beta_btc': pair_info.beta_btc,
                                'last_pvalue': pair_info.last_pvalue,
                                'entry_z_score': pair_info.entry_z_score,
                                'entry_half_life_bars': pair_info.entry_half_life_bars,
                                'entry_hold_limit_bars': pair_info.entry_hold_limit_bars,
                                'open_time': int(time.time()),
                            })
                        except Exception as dbe:
                            print(f"⚠️ DB Update failed: {dbe}")

                    try:
                        trade = db.Trades(
                            pair_id=pair_info.db_id,
                            direction=direction,
                            status='OPEN',
                            open_time=int(time.time() * 1000),
                            entry_price_1=pair_info.entry_price1,
                            entry_price_2=pair_info.entry_price2,
                            qty1=pair_info.qty1,
                            qty2=pair_info.qty2,
                            pnl=0.0,
                            hedge_ratio=pair_info.hedge_ratio,
                            beta_btc=pair_info.beta_btc,
                            pvalue=pair_info.last_pvalue,
                            entry_z=pair_info.entry_z_score,
                        )
                        pair_info.current_trade_id = await db.add_trade(trade)
                        try:
                            # Persist entry fills for post-trade order analytics.
                            now_ms = int(time.time() * 1000)
                            start_ms_open = now_ms - 180_000
                            order_id_s1 = int(executed_orders[0].get('orderId')) if executed_orders and len(executed_orders) > 0 and executed_orders[0].get('orderId') is not None else None
                            order_id_s2 = int(executed_orders[1].get('orderId')) if executed_orders and len(executed_orders) > 1 and executed_orders[1].get('orderId') is not None else None
                            open_trades_s1 = await self._fetch_account_trades_window(s1, start_ms_open, max_records=1500)
                            open_trades_s2 = await self._fetch_account_trades_window(s2, start_ms_open, max_records=1500)
                            if order_id_s1 is not None:
                                open_trades_s1 = [t for t in open_trades_s1 if int(t.get('orderId', -1)) == order_id_s1]
                            if order_id_s2 is not None:
                                open_trades_s2 = [t for t in open_trades_s2 if int(t.get('orderId', -1)) == order_id_s2]
                            await self._persist_pair_executions(
                                pair_info,
                                open_trades_s1,
                                open_trades_s2,
                                phase='OPEN',
                                trade_id=pair_info.current_trade_id
                            )
                        except Exception as fill_err:
                            print(f"⚠️ Could not persist OPEN executions for {s1}-{s2}: {fill_err}")
                    except Exception as e:
                        # Do not keep live positions without a trade row: close immediately for audit safety.
                        print(f"CRITICAL: could not create OPEN trade record for {s1}-{s2}: {e}")
                        try:
                            alert = (f"🚨 <b>Trade audit safety stop</b>\n"
                                     f"Pair: {s1}-{s2}\n"
                                     f"Reason: failed to persist OPEN trade record.\n"
                                     f"Action: position will be closed immediately to avoid untracked exposure.")
                            reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                            await self._notify(alert, reply_to)
                        except Exception:
                            pass
                        pair_info.close_handled = True
                        pair_info.is_trading = True
                        await self._execute_trade(pair_info, 0, close_reason='audit_fail')
                        return
            
            except Exception as e:
                print(f"FATAL ERROR during trade execution for {s1}-{s2}: {e}")
                traceback.print_exc()
                pair_info.position_status = 0
        finally:
            pair_info.is_trading = False

    # === REAL-TIME Z-SCORE MONITORING ===
    
    def _pairs_with_symbol(self, symbol: str) -> list:
        """Returns all PairInfo objects containing the given symbol. O(1) via index."""
        return list(self._symbol_to_pairs.get(symbol, []))

    def _register_pair(self, pair_info: PairInfo):
        """Add pair to _symbol_to_pairs index. Call after adding to active_pairs."""
        for sym in (pair_info.symbol1, pair_info.symbol2):
            self._symbol_to_pairs.setdefault(sym, []).append(pair_info)

    def _unregister_pair(self, pair_info: PairInfo):
        """Remove pair from _symbol_to_pairs index. Call before deleting from active_pairs."""
        for sym in (pair_info.symbol1, pair_info.symbol2):
            pairs_list = self._symbol_to_pairs.get(sym)
            if pairs_list:
                try:
                    pairs_list.remove(pair_info)
                except ValueError:
                    pass
                if not pairs_list:
                    del self._symbol_to_pairs[sym]
    
    def _calc_realtime_zscore(self, pair_info: PairInfo, price1: float, price2: float) -> float:
        """
        Calculate Z-score using historical data + current real-time prices.
        Uses the closed candles for mean/std, but current price for the last point.
        """
        try:
            data1 = self.all_data.get(pair_info.symbol1)
            data2 = self.all_data.get(pair_info.symbol2)
            if not data1 or not data2:
                return np.nan
            if len(data1.close) < self.min_data_points or len(data2.close) < self.min_data_points:
                return np.nan
            
            # Get historical log prices
            log1 = np.log(list(data1.close)[-self.min_data_points:])
            log2 = np.log(list(data2.close)[-self.min_data_points:])
            
            # Calculate spread on historical data
            historical_spread = log1 - pair_info.hedge_ratio * log2
            mean = np.mean(historical_spread)
            std = np.std(historical_spread)
            if std == 0 or np.isnan(std):
                return np.nan
            
            # Current spread using real-time prices
            current_log1 = np.log(price1)
            current_log2 = np.log(price2)
            current_spread = current_log1 - pair_info.hedge_ratio * current_log2
            
            # Z-score
            z_score = (current_spread - mean) / std
            return float(z_score)
        except Exception:
            return np.nan
    
    async def on_ticker_update(self, symbol: str, price: float):
        """
        Called when a price update is received from WebSocket ticker (~1s markPrice).
        Handles BOTH:
          - Entry signal detection (for idle pairs)
          - EXIT monitoring: Z-score TP/SL, circuit breaker, beta drift (for open positions)
          - BTC Market Shock protection (closes ALL positions on flash crash)
        """
        self.last_prices[symbol] = price
        
        # Track BTC price for Market Shock Protector
        if symbol == 'BTCUSDT':
            self._btc_price_history.append((time.time(), price))
            
            # Check for BTC shock (only if we have enough history and not in cooldown)
            if len(self._btc_price_history) >= 10 and not self._btc_shock_triggered:
                await self._check_btc_shock()
        
        # Check all pairs containing this symbol
        for pair_info in self._pairs_with_symbol(symbol):
            if pair_info.is_trading:
                continue  # Trade in progress, skip
            
            # Get both prices
            price1 = self.last_prices.get(pair_info.symbol1)
            price2 = self.last_prices.get(pair_info.symbol2)
            if not price1 or not price2:
                continue
            
            # Calculate real-time Z-score
            z_score = self._calc_realtime_zscore(pair_info, price1, price2)
            if np.isnan(z_score):
                continue
            
            # Update last_z_score with real-time value
            pair_info.last_z_score = z_score
            
            # ====== OPEN POSITIONS: Real-time EXIT monitoring ======
            if pair_info.position_status != 0:
                await self._check_realtime_exit(pair_info, z_score, price1, price2)
                continue
            
            # ====== IDLE PAIRS: Entry signal detection ======
            z_entry = getattr(self.config, 'z_entry', 1.9) or 1.9
            z_entry_max = getattr(self.config, 'z_entry_max', 2.5) or 2.5
            
            # Skip entry if pair is in cooldown after SL
            if getattr(pair_info, '_close_cooldown_until', 0) > time.time():
                continue
            
            # Skip if pair is waiting for next candle close before re-entry
            if getattr(pair_info, '_wait_for_candle', False) or self._is_pair_reentry_blocked_same_candle(pair_info):
                continue
            
            # Check if signal (between z_entry and z_entry_max)
            # Reject if already too extreme - spread may be broken
            if abs(z_score) >= z_entry and abs(z_score) < z_entry_max:
                if pair_info.pending_signal is None:
                    # Start confirmation timer
                    pair_info.pending_signal = z_score
                    pair_info.pending_since = time.time()
                    pair_info.pending_source = 'realtime'
            else:
                # Signal went away - reset
                if pair_info.pending_signal is not None:
                    pair_info.pending_signal = None
                    pair_info.pending_since = None
                    pair_info.pending_source = ''
    
    async def _check_btc_shock(self):
        """
        BTC Market Shock Protector.
        If BTC moves > btc_crash_pct within btc_crash_window seconds,
        force-close ALL open positions immediately.
        """
        btc_crash_pct = getattr(self.config, 'btc_crash_pct', 0.05) or 0.05  # 5% default
        btc_crash_window = getattr(self.config, 'btc_crash_window', 300) or 300  # 5 min default
        
        now = time.time()
        
        # Cooldown: don't re-trigger for 5 minutes after a shock
        if now < self._btc_shock_cooldown:
            return
        
        current_price = self._btc_price_history[-1][1]
        
        # Find the price from ~btc_crash_window seconds ago
        reference_price = None
        for ts, p in self._btc_price_history:
            if now - ts <= btc_crash_window:
                reference_price = p
                break
        
        if reference_price is None or reference_price == 0:
            return
        
        btc_change = (current_price - reference_price) / reference_price
        
        if abs(btc_change) >= btc_crash_pct:
            direction = '📉 CRASH' if btc_change < 0 else '📈 PUMP'
            self._btc_shock_triggered = True
            self._btc_shock_cooldown = now + btc_crash_window  # Cooldown
            
            shock_msg = (f"💥 <b>BTC MARKET SHOCK</b>!\n"
                         f"{direction}: BTC {btc_change*100:+.2f}% in {btc_crash_window//60:.0f} min\n"
                         f"Price: {reference_price:.2f} → {current_price:.2f}\n"
                         f"🚨 Force-closing ALL open positions...")
            print(shock_msg)
            await self._notify(shock_msg)
            
            # Close ALL open positions
            closed_count = 0
            for pair_info in list(self.active_pairs.values()):
                if pair_info.position_status != 0 and not pair_info.is_trading:
                    s1, s2 = pair_info.symbol1, pair_info.symbol2
                    print(f"💥 BTC Shock: closing {s1}-{s2}")
                    pair_info.close_handled = True
                    pair_info.is_trading = True
                    try:
                        await self._execute_trade(pair_info, 0, close_reason='btc_shock')
                        closed_count += 1
                    except Exception as e:
                        print(f"❌ Failed to close {s1}-{s2} during BTC shock: {e}")
            
            result_msg = f"💥 BTC Shock: closed {closed_count} positions"
            print(result_msg)
            await self._notify(result_msg)
            
            # Reset flag after all closures complete
            self._btc_shock_triggered = False
    
    async def _check_realtime_exit(self, pair_info: PairInfo, z_score: float, price1: float, price2: float):
        """
        Real-time exit monitoring for open positions. Called on every markPrice update (~1s).
        Checks: Z-score TP/SL, Circuit Breaker, Beta Drift (with live recalc every 10s).
        """
        s1, s2 = pair_info.symbol1, pair_info.symbol2
        now_ts = time.time()
        close_retry_sec = int(getattr(self.config, 'close_retry_cooldown_sec', 30) or 30)
        next_close_try_ts = float(getattr(pair_info, '_next_close_try_ts', 0) or 0)
        if next_close_try_ts > now_ts:
            return

        def _arm_close_retry():
            pair_info._next_close_try_ts = time.time() + max(5, close_retry_sec)
        
        # --- 1. Z-Score TP / SL (instant, every tick) ---
        z_exit = self.config.z_exit if self.config and self.config.z_exit is not None else 0.0
        z_stop = self.config.z_stop if self.config and self.config.z_stop else 4.0
        
        if pair_info.position_status == 1:  # Long spread
            if z_score >= z_exit:
                print(f"💰 RT TAKE PROFIT (Long) on {s1}-{s2}. Z: {z_score:.2f} >= {z_exit}. Closing...")
                pair_info.close_handled = True
                pair_info.is_trading = True
                _arm_close_retry()
                await self._execute_trade(pair_info, 0, close_reason='z_tp')
                return
            elif z_score <= -z_stop:
                print(f"🛑 RT STOP LOSS (Long) on {s1}-{s2}. Z: {z_score:.2f} <= -{z_stop}. Closing...")
                pair_info.close_handled = True
                pair_info.is_trading = True
                _arm_close_retry()
                await self._execute_trade(pair_info, 0, close_reason='z_sl')
                return
        
        elif pair_info.position_status == -1:  # Short spread
            if z_score <= -z_exit:
                print(f"💰 RT TAKE PROFIT (Short) on {s1}-{s2}. Z: {z_score:.2f} <= {-z_exit}. Closing...")
                pair_info.close_handled = True
                pair_info.is_trading = True
                _arm_close_retry()
                await self._execute_trade(pair_info, 0, close_reason='z_tp')
                return
            elif z_score >= z_stop:
                print(f"🛑 RT STOP LOSS (Short) on {s1}-{s2}. Z: {z_score:.2f} >= {z_stop}. Closing...")
                pair_info.close_handled = True
                pair_info.is_trading = True
                _arm_close_retry()
                await self._execute_trade(pair_info, 0, close_reason='z_sl')
                return
        
        # --- 2. Circuit Breaker (instant, every tick) ---
        if pair_info.entry_price1 > 0 and pair_info.entry_price2 > 0:
            # Use EXCHANGE PnL (source of truth) — no manual calculations
            total_pnl = self._get_exchange_pair_pnl(pair_info, price1, price2)
            
            notional = (pair_info.entry_price1 * pair_info.qty1) + (pair_info.entry_price2 * pair_info.qty2)
            leverage = self.config.leverage if self.config and self.config.leverage else 20
            margin = notional / leverage  # Actual deployed capital
            circuit_breaker_pct = getattr(self.config, 'circuit_breaker_pct', 0.20) or 0.20
            
            if notional > 0:
                roi_notional = total_pnl / notional
                if roi_notional < -circuit_breaker_pct:
                    roi_margin = total_pnl / margin if margin > 0 else 0
                    cb_msg = (f"🚨 <b>RT CIRCUIT BREAKER</b> on {s1}-{s2}!\n"
                              f"Loss: {roi_notional*100:.2f}% of notional ({total_pnl:.2f} USDT)\n"
                              f"Margin: {roi_margin*100:.2f}% | Leverage: {leverage}x\n"
                              f"Force Closing...")
                    print(cb_msg)
                    reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                    await self._notify(cb_msg, reply_to)
                    pair_info.close_handled = True
                    pair_info.is_trading = True
                    _arm_close_retry()
                    await self._execute_trade(pair_info, 0, close_reason='circuit')
                    return
        
        # --- 3. Beta Drift CHECK (every tick, using last known beta from candle close) ---
        # NOTE: Cointegration (ADF test) and beta recalculation are done ONLY at candle close
        # in _check_signals_for_active_pairs(). Between candle closes, the data (200 closed candles)
        # is identical, so re-running ADF 360x/hour wastes CPU and produces identical results.
        # Real-time protection between candles is provided by: Z-score TP/SL, Circuit Breaker,
        # BTC Shock Protector, Hardware SL/TP, and this Beta Drift threshold check.
        
        # GRACE PERIOD: Skip beta checks for 120s after trade opens.
        # Beta needs several candle-close recalculations to stabilize after entry.
        trade_open_time = getattr(pair_info, '_trade_open_time', 0)
        if trade_open_time > 0 and time.time() - trade_open_time < 120:
            pass  # Too early — beta not yet stable, skip beta check
        elif pair_info.beta_btc != 0:
            beta_alert_threshold = getattr(self.config, 'beta_alert_threshold', 0.15) or 0.15
            beta_critical = getattr(self.config, 'beta_critical', 1.0) or 1.0
            
            abs_beta = abs(pair_info.beta_btc)
            
            if abs_beta >= beta_critical:
                # Beta critical is checked on every tick, but the value only changes at candle close.
                # So we only need to trigger once when candle-close recalc puts beta above critical.
                # Use a flag to avoid spamming force-close attempts on every tick.
                if not getattr(pair_info, '_beta_critical_triggered', False):
                    pair_info._beta_critical_triggered = True
                    pair_info._beta_at_trigger = pair_info.beta_btc
                    
                    # Use EXCHANGE PnL (source of truth)
                    total_pnl = self._get_exchange_pair_pnl(pair_info, price1, price2)
                    
                    beta_msg = (f"🚨 <b>RT BETA CRITICAL</b> on {s1}-{s2}!\n"
                                f"Beta: {pair_info.beta_btc:.3f} (critical: {beta_critical})\n"
                                f"PnL: {total_pnl:+.2f} USDT. Force-closing...")
                    print(beta_msg)
                    reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                    await self._notify(beta_msg, reply_to)
                    pair_info.close_handled = True
                    pair_info.is_trading = True
                    _arm_close_retry()
                    await self._execute_trade(pair_info, 0, close_reason='beta_critical')
                    return
            
            elif abs_beta >= beta_alert_threshold:
                # Reset beta critical flag (beta dropped below critical)
                pair_info._beta_critical_triggered = False
                
                # Use EXCHANGE PnL (source of truth) — no manual calculations
                total_pnl = self._get_exchange_pair_pnl(pair_info, price1, price2)
                
                if total_pnl > 0:
                    pair_info._beta_at_trigger = pair_info.beta_btc
                    
                    beta_msg = (f"⚠️ <b>RT BETA DRIFT</b> on {s1}-{s2}!\n"
                                f"Beta: {pair_info.beta_btc:.3f} (threshold: {beta_alert_threshold})\n"
                                f"PnL: +{total_pnl:.2f} USDT. Auto-closing...")
                    print(beta_msg)
                    reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                    await self._notify(beta_msg, reply_to)
                    pair_info.close_handled = True
                    pair_info.is_trading = True
                    _arm_close_retry()
                    await self._execute_trade(pair_info, 0, close_reason='beta_drift')
                    return
                else:
                    # Negative PnL - warn but don't spam (throttle warnings to every 60s)
                    now_warn = time.time()
                    last_warn = getattr(pair_info, '_last_beta_warn', 0)
                    if now_warn - last_warn >= 60:
                        pair_info._last_beta_warn = now_warn
                        beta_warn = (f"⚠️ <b>RT BETA DRIFT WARNING</b> on {s1}-{s2}!\n"
                                     f"Beta: {pair_info.beta_btc:.3f} (threshold: {beta_alert_threshold})\n"
                                     f"PnL: {total_pnl:.2f} USDT. Consider manual close.")
                        print(beta_warn)
                        reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                        await self._notify(beta_warn, reply_to)
            else:
                # Beta is within normal range — reset flag
                pair_info._beta_critical_triggered = False

        due_time_exit, hold_bars, hold_limit = self._time_exit_due(pair_info, now_ts=now_ts)
        if due_time_exit:
            print(
                f"⏱️ RT TIME EXIT on {s1}-{s2}. "
                f"hold_bars={hold_bars} >= limit={hold_limit}. Closing..."
            )
            pair_info.close_handled = True
            pair_info.is_trading = True
            _arm_close_retry()
            await self._execute_trade(pair_info, 0, close_reason='time_exit')
            return
    
    async def _signal_confirmation_loop(self):
        """
        Periodically checks for confirmed signals (held for N seconds).
        """
        confirm_sec = getattr(self.config, 'signal_confirm_sec', 10) or 10
        last_status_log = 0
        
        while True:
            await asyncio.sleep(1)  # Check every second
            
            try:
                # Periodic status log (every 60s) to show bot is alive
                now = time.time()
                if now - last_status_log >= 60:
                    last_status_log = now
                    idle_count = sum(1 for pi in self.active_pairs.values() if pi.position_status == 0 and not pi.is_trading)
                    open_count = sum(1 for pi in self.active_pairs.values() if pi.position_status != 0)
                    pending_count = sum(1 for pi in self.active_pairs.values() if pi.pending_signal is not None)
                    print(f"📊 Signal monitor: {idle_count} idle pairs, {open_count} open positions, {pending_count} pending signals")
                    # Diagnostic snapshot at monitor cadence (not every tick).
                    z_entry_dbg = getattr(self.config, 'z_entry', 1.9) or 1.9
                    z_entry_max_dbg = getattr(self.config, 'z_entry_max', 2.5) or 2.5
                    confirm_sec_dbg = getattr(self.config, 'signal_confirm_sec', 10) or 10
                    details = []
                    for pi in self.active_pairs.values():
                        if pi.position_status != 0 or pi.is_trading:
                            continue
                        s1 = pi.symbol1
                        s2 = pi.symbol2
                        p1 = self.last_prices.get(s1)
                        p2 = self.last_prices.get(s2)
                        z = np.nan
                        if p1 and p2:
                            z = self._calc_realtime_zscore(pi, p1, p2)
                        if np.isnan(z):
                            z_txt = "nan"
                            near_entry = False
                        else:
                            z_txt = f"{z:+.2f}"
                            near_entry = abs(z) >= (z_entry_dbg * 0.8)

                        if getattr(pi, '_close_cooldown_until', 0) > now:
                            reason = f"sl_cd:{int(getattr(pi, '_close_cooldown_until', 0) - now)}s"
                        elif getattr(pi, '_wait_for_candle', False) or self._is_pair_reentry_blocked_same_candle(pi):
                            reason = "wait_candle"
                        elif pi.pending_signal is not None:
                            elapsed = int(now - (pi.pending_since or now))
                            left = max(0, int(confirm_sec_dbg - elapsed))
                            reason = f"pending:{left}s"
                        elif np.isnan(z):
                            reason = "no_rt_z"
                        elif abs(z) >= z_entry_max_dbg:
                            reason = f"z>{z_entry_max_dbg:.2f}"
                        elif abs(z) >= z_entry_dbg:
                            reason = "entry_window"
                        else:
                            reason = "z_low"

                        details.append((near_entry, abs(z) if not np.isnan(z) else -1.0, f"  - {s1}-{s2}: z={z_txt}, reason={reason}"))

                    details.sort(key=lambda x: (not x[0], -x[1]))
                    max_lines = 20
                    for _, _, line in details[:max_lines]:
                        print(line)
                    if len(details) > max_lines:
                        print(f"  ... and {len(details) - max_lines} more idle pairs")
                z_entry = getattr(self.config, 'z_entry', 1.9) or 1.9
                z_entry_max = getattr(self.config, 'z_entry_max', 2.5) or 2.5
                coint_stability_min_bars = int(getattr(self.config, 'coint_stability_min_bars', 2) or 2)
                entry_target_abs_z = float(getattr(self.config, 'entry_et_target_abs_z', 0.5) or 0.5)
                candles_per_day = float(utils.CANDLES_PER_DAY.get(str(self.timeframe), 24) or 24)
                bar_hours = 24.0 / candles_per_day if candles_per_day > 0 else 1.0
                ready_candidates = []
                processed_pending = []

                for pair_info in list(self.active_pairs.values()):
                    if pair_info.position_status != 0 or pair_info.is_trading:
                        continue
                    if pair_info.pending_signal is None or pair_info.pending_since is None:
                        continue

                    now_ts = time.time()
                    elapsed = now_ts - pair_info.pending_since
                    if elapsed < confirm_sec:
                        continue

                    def _log_pending_reject(message: str, cooldown_sec: int = 15):
                        key = f"_last_pending_reject_log_{message}"
                        prev = float(getattr(pair_info, key, 0.0) or 0.0)
                        if now_ts - prev >= cooldown_sec:
                            setattr(pair_info, key, now_ts)
                            print(f"⏭️ Pending rejected {pair_info.symbol1}-{pair_info.symbol2}: {message}")

                    # Primary: realtime z from markPrice. Fallback for candle-origin signals.
                    price1 = self.last_prices.get(pair_info.symbol1)
                    price2 = self.last_prices.get(pair_info.symbol2)
                    current_z = np.nan
                    if price1 and price2:
                        current_z = self._calc_realtime_zscore(pair_info, price1, price2)
                    elif getattr(pair_info, 'pending_source', '') == 'candle':
                        current_z = float(getattr(pair_info, 'last_z_score', np.nan))

                    processed_pending.append(pair_info)
                    if np.isnan(current_z):
                        _log_pending_reject("no realtime z-score")
                        continue

                    # Must remain inside entry window and keep original direction.
                    if abs(current_z) >= z_entry and abs(current_z) < z_entry_max and (current_z * pair_info.pending_signal > 0):
                        streak_bars = int(getattr(pair_info, 'coint_streak_bars', 0) or 0)
                        if streak_bars < coint_stability_min_bars:
                            # Restart-safe seed: infer streak from loaded history when runtime counters are cold.
                            last_eval_seed = int(getattr(pair_info, '_last_coint_eval_ts', 0) or 0)
                            if last_eval_seed <= 0:
                                seed_lookback = max(6, coint_stability_min_bars + 2)
                                seeded_streak, seeded_eval_ts = self._estimate_coint_streak_from_history(
                                    pair_info, max_recent_bars=seed_lookback
                                )
                                if seeded_streak > streak_bars:
                                    pair_info.coint_streak_bars = int(seeded_streak)
                                    streak_bars = int(seeded_streak)
                                if seeded_eval_ts > last_eval_seed:
                                    pair_info._last_coint_eval_ts = int(seeded_eval_ts)
                                    last_eval_seed = int(seeded_eval_ts)
                                if seeded_streak > 0 and not bool(getattr(pair_info, '_coint_seeded_logged', False)):
                                    setattr(pair_info, '_coint_seeded_logged', True)
                                    print(
                                        f"🔁 Seeded coint streak from history for {pair_info.symbol1}-{pair_info.symbol2}: "
                                        f"{seeded_streak} bars (eval_ts={seeded_eval_ts})"
                                    )
                        if streak_bars < coint_stability_min_bars:
                            ts1 = int(self.all_data.get(pair_info.symbol1).ts[-1]) if self.all_data.get(pair_info.symbol1) and self.all_data.get(pair_info.symbol1).ts else 0
                            ts2 = int(self.all_data.get(pair_info.symbol2).ts[-1]) if self.all_data.get(pair_info.symbol2) and self.all_data.get(pair_info.symbol2).ts else 0
                            last_eval = int(getattr(pair_info, '_last_coint_eval_ts', 0) or 0)
                            _log_pending_reject(
                                f"coint_streak={streak_bars} < min={coint_stability_min_bars} "
                                f"(last_eval_ts={last_eval}, s1_ts={ts1}, s2_ts={ts2})"
                            )
                            continue
                        expected_bars = utils.expected_reversion_bars(
                            abs_z_now=abs(float(current_z)),
                            abs_z_target=entry_target_abs_z,
                            half_life_bars=float(getattr(pair_info, 'half_life', 0.0) or 0.0),
                        )
                        pair_info.entry_expected_hours = float(expected_bars * bar_hours)
                        pair_info.entry_coint_streak_bars = streak_bars
                        score = float(getattr(pair_info, 'quality_score', 0.0) or 0.0)
                        updated_at = float(getattr(pair_info, 'quality_updated_at', 0.0) or 0.0)
                        ready_candidates.append((score, updated_at, pair_info, current_z))
                    elif abs(current_z) >= z_entry_max:
                        print(f"⚠️ {pair_info.symbol1}-{pair_info.symbol2}: Z={current_z:.2f} exceeds z_entry_max={z_entry_max}. Skipping entry (spread may be broken).")
                    else:
                        if current_z * pair_info.pending_signal <= 0:
                            _log_pending_reject(
                                f"signal direction flipped (pending={pair_info.pending_signal:+.2f}, now={current_z:+.2f})"
                            )
                        elif abs(current_z) < z_entry:
                            _log_pending_reject(f"|z| dropped below z_entry ({abs(current_z):.2f} < {z_entry:.2f})")
                        else:
                            _log_pending_reject("outside entry window")

                # Reset matured pending signals after evaluation cycle.
                for pair_info in processed_pending:
                    pair_info.pending_signal = None
                    pair_info.pending_since = None
                    pair_info.pending_source = ''

                if not ready_candidates:
                    continue

                # Rank by cached quality score (higher is better), then recency of metrics.
                ready_candidates.sort(key=lambda x: (-x[0], -x[1]))
                max_pairs = getattr(self.config, 'max_active_pairs', 5) or 5
                free_slots = max(0, int(max_pairs - self.count_active_positions()))
                if free_slots <= 0:
                    continue

                for score, _, pair_info, current_z in ready_candidates:
                    if free_slots <= 0:
                        break
                    if not self._is_pair_trade_allowed(pair_info.symbol1, pair_info.symbol2):
                        print(f"⏭️ Ranked entry blocked {pair_info.symbol1}-{pair_info.symbol2}: pair list filter")
                        continue
                    can_open, open_reason = self._can_open_new_position_reason(pair_info.symbol1, pair_info.symbol2)
                    if not can_open:
                        print(f"⏭️ Ranked entry blocked {pair_info.symbol1}-{pair_info.symbol2}: {open_reason}")
                        continue

                    # Check cooldown from failed leverage/trade
                    fail_until = getattr(pair_info, '_leverage_fail_until', 0)
                    if fail_until and time.time() < fail_until:
                        print(f"⏭️ Ranked entry blocked {pair_info.symbol1}-{pair_info.symbol2}: leverage_fail_cooldown")
                        continue

                    # Check cooldown after stop-loss close
                    close_cooldown = getattr(pair_info, '_close_cooldown_until', 0)
                    if close_cooldown and time.time() < close_cooldown:
                        remaining = int(close_cooldown - time.time())
                        print(f"⏸️ {pair_info.symbol1}-{pair_info.symbol2}: Entry blocked by SL cooldown ({remaining}s remaining)")
                        continue

                    # Check if pair is waiting for next candle close
                    if getattr(pair_info, '_wait_for_candle', False) or self._is_pair_reentry_blocked_same_candle(pair_info):
                        print(f"⏭️ Ranked entry blocked {pair_info.symbol1}-{pair_info.symbol2}: wait_for_next_candle")
                        continue

                    direction = 1 if current_z < 0 else -1
                    pair_info.entry_z_score = current_z
                    print(f"✅ Ranked entry: {pair_info.symbol1}-{pair_info.symbol2} | score={score:.3f} | Z={current_z:.2f}. Opening...")
                    pair_info.is_trading = True
                    self.loop.create_task(self._execute_trade(pair_info, direction))
                    free_slots -= 1
            except Exception as e:
                print(f"⚠️ Signal confirmation loop error (continuing): {e}")
    
    async def _subscribe_new_pair_realtime(self, symbol1: str, symbol2: str):
        """
        Subscribe to markPrice streams for a newly discovered pair.
        This enables real-time Z-score monitoring for new pairs.
        """
        if not self._subscribe_mark_callback:
            return  # No callback configured - skip silently
        
        new_symbols = []
        
        # Only subscribe if not already subscribed
        if symbol1 not in self._subscribed_mark_symbols:
            new_symbols.append(symbol1)
            self._subscribed_mark_symbols.add(symbol1)
        
        if symbol2 not in self._subscribed_mark_symbols:
            new_symbols.append(symbol2)
            self._subscribed_mark_symbols.add(symbol2)
        
        # Subscribe to markPrice for new symbols
        if new_symbols:
            try:
                await self._subscribe_mark_callback(new_symbols)
                print(f"🔔 Subscribed to markPrice for new pair: {symbol1}-{symbol2} (symbols: {new_symbols})")
            except Exception as e:
                print(f"⚠️ Failed to subscribe markPrice for {symbol1}-{symbol2}: {e}")
    
    def start_realtime_monitoring(self):
        """Start the signal confirmation loop."""
        if self._signal_confirmation_task is None:
            self._signal_confirmation_task = self.loop.create_task(self._signal_confirmation_loop())
            print("🔄 Started real-time signal confirmation loop") 
        if self._health_task is None:
            self._health_task = self.loop.create_task(self._discovery_health_loop())
            print("🧭 Started discovery health watchdog loop")
