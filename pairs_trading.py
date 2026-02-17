from collections import deque
import numpy as np
from dataclasses import dataclass, field
import utils
import asyncio
import itertools
import time
import traceback
import json
import os
from concurrent.futures import ProcessPoolExecutor
import db
import math


MAX_LEN = 500
COINT_WINDOW = 200

# Canonical close reason mapping (used in pairs_trading and main.py)
CLOSE_REASONS = {
    'z_tp': '💰 Z-Score Take Profit',
    'z_sl': '🛑 Z-Score Stop Loss',
    'circuit': '🔴 Circuit Breaker',
    'broken_coint': '🚨 Broken Correlation',
    'hardware_sl': '🛡️ Hardware SL',
    'hardware_tp': '🛡️ Hardware TP',
    'manual': '👤 Manual Close',
    'desync': '⚠️ Leg Desync',
    'beta_drift': '📉 Beta Drift',
    'beta_critical': '🚨 Beta Critical',
    'btc_shock': '💥 BTC Market Shock',
    'external': '⚡ External Close',
    'orphan_restart': '🔄 Orphan on Restart',
    'stale_symbols': '⏳ Stale Symbols',
    'manual_partial': '👤 Manual Close (1 leg)',
}

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
    # Market neutrality
    beta_btc: float = 0.0      # Beta to BTC (should be near 0 for market-neutral)
    last_pvalue: float = 0.0   # Last p-value from cointegration test
    # Signal confirmation (for real-time mode)
    pending_signal: float = None  # Pending Z-score signal awaiting confirmation
    pending_since: float = None   # Time when signal started
    # Idle pair management
    discovered_at: float = field(default_factory=time.time)  # When pair was discovered
    # Close tracking - prevents duplicate notifications
    close_handled: bool = False    # True if bot already processed close notification
    last_close_reason: str = ''    # Reason for last close (for debugging)
    # Cooldown after stop-loss to prevent immediate re-entry
    _close_cooldown_until: float = 0.0  # Unix timestamp: skip entry signals until this time
    # Wait-for-candle: after ANY close, block re-entry until next candle closes
    _wait_for_candle: bool = False  # True = pair just closed, wait for next candle before re-entry

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
        # O(1) symbol → list[PairInfo] index (maintained by _register_pair/_unregister_pair)
        self._symbol_to_pairs: dict[str, list[PairInfo]] = {}
        self.leverage_cache = {} # {symbol: leverage_int}
        self._discovery_task = None
        self._last_discovery_time = 0
        self._cleanup_task = None  # Periodic orphaned orders cleanup
        self._last_cleanup_time = 0
        
        # CRITICAL: Lock to prevent race condition when opening trades
        self._trade_lock = asyncio.Lock()
        
        # CPU Pool for heavy computations
        self.executor = ProcessPoolExecutor(max_workers=None)
        
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
        
        # Update exchange position cache
        await self._refresh_exchange_position_count()
        
        self._initialized = True
        self._init_complete_time = time.time()
        max_pairs = getattr(self.config, 'max_active_pairs', 5) or 5
        print(f"✅ PairsManager initialized. Max active pairs: {max_pairs}")
        
        # Start periodic leg sync loop (BACKUP only - primary sync is via WebSocket)
        self._leg_sync_task = self.loop.create_task(self._periodic_leg_sync_loop())
        print("🔄 Started backup leg sync loop (every 30s, primary via WebSocket)")

    async def _load_state_from_db(self):
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
                        info = PairInfo(
                            symbol1=p.symbol1,
                            symbol2=p.symbol2,
                            hedge_ratio=p.hedge_ratio,
                            half_life=p.half_life,
                            position_status=p.position_status,
                            qty1=p.qty1,
                            qty2=p.qty2,
                            entry_price1=p.entry_price1,
                            entry_price2=p.entry_price2,
                            db_id=p.id,
                            tg_message_id=getattr(p, 'tg_message_id', 0) or 0
                        )
                        # Restore market neutrality metrics from DB (survive restart)
                        info.beta_btc = getattr(p, 'beta_btc', 0.0) or 0.0
                        info.last_pvalue = getattr(p, 'last_pvalue', 0.0) or 0.0
                        info.entry_z_score = getattr(p, 'entry_z_score', 0.0) or 0.0
                        
                        last_trade = await db.get_last_open_trade_for_pair(p.id)
                        if last_trade:
                            info.current_trade_id = last_trade.id
                        
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
                                await self.client.new_order(
                                    symbol=remaining_sym,
                                    side=close_side,
                                    type='MARKET',
                                    quantity=remaining_qty,
                                    reduceOnly='true'
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
                        closed_count += 1
            
            print(f"  Restored {restored_count} pairs, marked {closed_count} stale pairs as CLOSED")
            
            # Continue with full reconciliation (orphan handling, unknown positions, etc.)
            # MUST be inside try block - if DB load failed, we must NOT reconcile with empty active_pairs
            await self._reconcile_with_exchange()
            
            # Add test pairs if test_mode is enabled
            await self._add_test_pairs()
            
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
        print("🔄 Reconciling DB with exchange positions...")
        try:
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
                        await self._load_state_from_db()
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
                
                for symbol in unknown_positions:
                    pos_data = open_on_exchange[symbol]
                    qty = pos_data['qty']
                    side = pos_data['side']
                    unrealized_pnl = pos_data['unrealized_pnl']
                    
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
                                    info = PairInfo(
                                        symbol1=db_pair.symbol1,
                                        symbol2=db_pair.symbol2,
                                        hedge_ratio=db_pair.hedge_ratio,
                                        half_life=db_pair.half_life,
                                        position_status=db_pair.position_status,
                                        qty1=db_pair.qty1,
                                        qty2=db_pair.qty2,
                                        entry_price1=db_pair.entry_price1,
                                        entry_price2=db_pair.entry_price2,
                                        db_id=db_pair.id,
                                        tg_message_id=getattr(db_pair, 'tg_message_id', 0) or 0
                                    )
                                    self.active_pairs[pair_set] = info
                                    self._register_pair(info)
                                    tracked_symbols.add(db_pair.symbol1)
                                    tracked_symbols.add(db_pair.symbol2)
                                    print(f"      🔄 RECOVERED from DB: {db_pair.symbol1}-{db_pair.symbol2} (was missed during load)")
                                    await self._notify(
                                        f"🔄 <b>PAIR RECOVERED</b>\n\n"
                                        f"Pair: {db_pair.symbol1}-{db_pair.symbol2}\n"
                                        f"Was missed during DB load but found on exchange + DB.\n"
                                        f"Restored to active trading."
                                    )
                                continue  # Skip closing - recovered
                    
                    # Only close if truly unknown (not found in DB either)
                    pnl_emoji = "🔴" if unrealized_pnl < 0 else "🟢"
                    await self._notify(
                        f"🚨 <b>UNKNOWN POSITION DETECTED</b>\n\n"
                        f"Symbol: {symbol} ({side})\n"
                        f"Qty: {qty}\n"
                        f"💵 Unrealized PnL: {pnl_emoji} <b>{unrealized_pnl:.2f} USDT</b>\n\n"
                        f"⏱️ <b>Closing...</b>"
                    )
                    
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
                            await self._notify(f"✅ Unknown position {symbol} was already closed.")
                        else:
                            await self.client.cancel_open_orders(symbol)
                            
                            close_side = 'SELL' if side == 'LONG' else 'BUY'
                            await self.client.new_order(
                                symbol=symbol,
                                side=close_side,
                                type='MARKET',
                                quantity=qty,
                                reduceOnly='true'
                            )
                            print(f"      ✅ Closed unknown position {symbol}")
                            
                            # Fetch PnL
                            await asyncio.sleep(1)
                            import time as time_mod
                            now_ms = int(time_mod.time() * 1000)
                            start_ms = now_ms - 300_000
                            trades = await self.client.get_account_trades(symbol=symbol, startTime=start_ms, limit=50)
                            pnl = sum(float(t.get('realizedPnl', 0)) for t in trades)
                            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                            
                            await self._notify(f"⚡ <b>Unknown Position Closed:</b> {symbol}\n"
                                               f"💵 PnL: {pnl_emoji} <b>{pnl:.2f} USDT</b>")
                    except Exception as e:
                        print(f"      ⚠️ Failed to close {symbol}: {e}")
                        await self._notify(f"🚨 FAILED to close: {symbol}\nError: {e}")
            
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
                                    await self.client.new_order(
                                        symbol=remaining_sym,
                                        side=close_side,
                                        type='MARKET',
                                        quantity=remaining_qty,
                                        reduceOnly='true'
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
                                total_pnl = pnl1 + pnl2
                                pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
                                
                                # Update DB with PnL
                                pairs_to_fix.append((pair_info, 'close_db_with_pnl', total_pnl, pnl1, pnl2))
                                
                                # Notify with details
                                await self._notify(f"⚡ <b>Orphan Closed (Restart):</b> {s1}-{s2}\n\n"
                                                   f"💵 PnL: {pnl_emoji} <b>{total_pnl:.2f} USDT</b>\n"
                                                   f"   {s1}: {pnl1:+.2f} USDT\n"
                                                   f"   {s2}: {pnl2:+.2f} USDT")
                            except Exception as e:
                                print(f"      ⚠️ Failed to close orphan: {e}")
                                pairs_to_fix.append((pair_info, 'close_db'))
                                await self._notify(f"🚨 ORPHAN FAILED: {s1}-{s2}\n{remaining_sym} still open!\nError: {e}")
                    else:
                        # Both open - check algo orders for SL/TP protection
                        # Use pre-fetched all_algo_orders (not API call per pair!)
                        s1_orders = [o for o in all_algo_orders if o.get('symbol') == s1]
                        s2_orders = [o for o in all_algo_orders if o.get('symbol') == s2]
                        
                        has_sl1 = any(o.get('orderType') == 'STOP' for o in s1_orders)
                        has_sl2 = any(o.get('orderType') == 'STOP' for o in s2_orders)
                        has_tp1 = any(o.get('orderType') == 'TAKE_PROFIT' for o in s1_orders)
                        has_tp2 = any(o.get('orderType') == 'TAKE_PROFIT' for o in s2_orders)
                        
                        missing = []
                        if not has_sl1: missing.append(f"{s1} SL")
                        if not has_sl2: missing.append(f"{s2} SL")
                        if not has_tp1: missing.append(f"{s1} TP")
                        if not has_tp2: missing.append(f"{s2} TP")
                        
                        if missing:
                            print(f"  ⚠️ MISSING PROTECTION for {s1}-{s2}: {', '.join(missing)}")
                            await self._notify(f"⚠️ <b>MISSING PROTECTION:</b> {s1}-{s2}\n"
                                               f"Missing: {', '.join(missing)}\n\n"
                                               f"Bot will attempt to restore SL/TP...")
                            
                            # TODO: Implement SL/TP restoration
                            # For now, just notify - restoration requires recalculating ATR
                else:
                    # DB says position is CLOSED
                    if s1_open or s2_open:
                        # But exchange has open position!
                        print(f"  ⚠️ MISMATCH: {s1}-{s2} marked CLOSED in DB but has positions on exchange!")
                        # Update DB to reflect reality
                        if s1_open and s2_open:
                            pairs_to_fix.append((pair_info, 'open_db', open_on_exchange.get(s1), open_on_exchange.get(s2)))
            
            # Apply fixes
            for fix in pairs_to_fix:
                pair_info = fix[0]
                action = fix[1]
                
                if action == 'close_db':
                    # Mark as closed in DB
                    pair_info.position_status = 0
                    pair_info.qty1 = 0
                    pair_info.qty2 = 0
                    pair_info.entry_price1 = 0
                    pair_info.entry_price2 = 0
                    
                    if pair_info.db_id:
                        await db.update_pair({
                            'id': pair_info.db_id,
                            'position_status': 0,
                            'qty1': 0,
                            'qty2': 0,
                            'entry_price1': 0,
                            'entry_price2': 0
                        })
                    
                    # Close any open trade records
                    if pair_info.current_trade_id:
                        await db.update_trade_fields(
                            pair_info.current_trade_id,
                            status='CLOSED_MANUAL',
                            close_time=int(time.time() * 1000)
                        )
                        pair_info.current_trade_id = None
                    
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
                
                elif action == 'close_db_with_pnl' and len(fix) >= 5:
                    # Mark as closed in DB with PnL info
                    total_pnl = fix[2]
                    pnl1 = fix[3]
                    pnl2 = fix[4]
                    
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
                            'close_pnl': total_pnl,
                            'close_reason': 'orphan_restart',
                            'pnl1': pnl1,
                            'pnl2': pnl2
                        })
                    
                    # Close trade record
                    if pair_info.current_trade_id:
                        await db.update_trade_fields(
                            pair_info.current_trade_id,
                            status='CLOSED_ORPHAN',
                            close_time=int(time.time() * 1000)
                        )
                        pair_info.current_trade_id = None
                    
                    print(f"  ✅ Fixed: {pair_info.symbol1}-{pair_info.symbol2} orphan closed with PnL={total_pnl:.2f}")
            
            active_count = self.count_active_positions()
            print(f"🔄 Reconciliation complete. Active pairs in DB: {active_count}")
            
            # Cleanup orphaned algo orders immediately
            await self._cleanup_orphaned_algo_orders()
            
        except Exception as e:
            print(f"❌ Error during reconciliation: {e}")
            import traceback
            traceback.print_exc()

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
            
            # Get all algo orders (using fixed endpoint /fapi/v1/openAlgoOrders)
            algo_orders = await self.client.get_algo_orders()
            if not algo_orders:
                return
            
            # Handle case where response is dict with 'orders' key
            if isinstance(algo_orders, dict) and 'orders' in algo_orders:
                algo_orders = algo_orders['orders']
            
            # Group orders by symbol
            orders_by_symbol = {}
            for order in algo_orders:
                order_type = order.get('orderType') or order.get('type', '')
                status = order.get('algoStatus') or order.get('status', '')
                if order_type in ['STOP', 'TAKE_PROFIT'] and status == 'NEW':
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
                        now_ms = int(time.time() * 1000)
                        start_ms = now_ms - 300_000  # Last 5 minutes
                        recent_trades = await self.client.get_account_trades(symbol=symbol, startTime=start_ms, limit=20)
                        
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
        
        # 3. Remove excess pairs (oldest first) if still over limit
        if len(idle_pairs) > max_idle:
            # Sort by discovered_at (oldest first)
            idle_pairs.sort(key=lambda x: x[1].discovered_at if x[1].discovered_at > 0 else float('inf'))
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
            await db.delete_pair(pair_info.db_id)
        
        print(f"  🗑️ Removed idle pair {s1}-{s2} (reason: {reason})")
    
    def _cleanup_unused_subscription(self, symbol: str):
        """Remove symbol from subscribed set if no other pairs use it."""
        # Check if any other pair uses this symbol
        for pair_info in self.active_pairs.values():
            if symbol in (pair_info.symbol1, pair_info.symbol2):
                return  # Still in use
        
        # Not used anymore - remove from tracked subscriptions
        self._subscribed_mark_symbols.discard(symbol)

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
            
            for pair_info in list(self.active_pairs.values()):
                if pair_info.position_status == 0 or pair_info.is_trading:
                    continue
                
                # VALIDATION: Skip stale pairs where symbols don't exist
                if pair_info.symbol1 not in self.all_symbols or pair_info.symbol2 not in self.all_symbols:
                    print(f"⚠️ Skipping stale pair {pair_info.symbol1}-{pair_info.symbol2}: symbols not in trading list. Cleaning up...")
                    pair_info.position_status = 0
                    pair_info.qty1 = 0
                    pair_info.qty2 = 0
                    if pair_info.db_id:
                        await db.update_pair({
                            'id': pair_info.db_id,
                            'position_status': 0,
                            'qty1': 0,
                            'qty2': 0,
                            'close_reason': 'stale_symbols'
                        })
                    continue
                    
                leg1_open = pair_info.symbol1 in pos_by_symbol
                leg2_open = pair_info.symbol2 in pos_by_symbol
                
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
                            o_type = trigger_order.get('type', '') or trigger_order.get('origType', '')
                            
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
                            await self.client.new_order(
                                symbol=remaining_leg, 
                                side=close_side, 
                                type='MARKET',
                                quantity=abs(remaining_qty), 
                                reduceOnly='true'
                            )
                            print(f"✅ Closed remaining leg {remaining_leg}")
                        
                        # Wait for trade data to be available
                        await asyncio.sleep(1)
                        
                        # Fetch actual PnL from recent trades
                        now_ms = int(time.time() * 1000)
                        start_ms = now_ms - 300_000  # Last 5 minutes
                        
                        trades1 = await self.client.get_account_trades(symbol=s1, startTime=start_ms, limit=50)
                        trades2 = await self.client.get_account_trades(symbol=s2, startTime=start_ms, limit=50)
                        
                        print(f"📊 Trades for {s1}: {len(trades1)} entries")
                        print(f"📊 Trades for {s2}: {len(trades2)} entries")
                        
                        # Sum realized PnL (already includes fees)
                        pnl1 = sum(float(t.get('realizedPnl', 0)) for t in trades1)
                        pnl2 = sum(float(t.get('realizedPnl', 0)) for t in trades2)
                        fee1 = sum(float(t.get('commission', 0)) for t in trades1)
                        fee2 = sum(float(t.get('commission', 0)) for t in trades2)
                        total_pnl = pnl1 + pnl2
                        total_fees = fee1 + fee2
                        
                        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
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
                        if close_zscore == 0.0 and s1 in self.all_data and s2 in self.all_data:
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
                        if (close_beta == 0 or close_pval == 0) and s1 in self.all_data and s2 in self.all_data:
                            try:
                                _d1 = self.all_data[s1]
                                _d2 = self.all_data[s2]
                                if len(_d1.close) >= self.min_data_points and len(_d2.close) >= self.min_data_points:
                                    _lp1 = np.log(list(_d1.close)[-self.min_data_points:])
                                    _lp2 = np.log(list(_d2.close)[-self.min_data_points:])
                                    _, _, _, _pval = utils.calculate_cointegration(_lp1, _lp2, strict_hl=False)
                                    if close_pval == 0 and not np.isnan(_pval):
                                        close_pval = float(_pval)
                                    if close_beta == 0 and 'BTCUSDT' in self.all_data:
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
                                await db.update_trade_fields(
                                    pair_info.current_trade_id,
                                    status='CLOSED',
                                    close_time=int(time.time() * 1000),
                                    pnl=total_pnl,
                                    close_z=close_zscore,
                                    fee1=fee1,
                                    fee2=fee2,
                                    close_reason='desync',
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
                                'close_pnl': total_pnl,
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
                                    f"💵 PnL: {pnl_emoji} <b>{total_pnl:.2f} USDT</b>\n"
                                    f"   {e1} {s1}: {pnl1:+.2f} USDT\n"
                                    f"   {e2} {s2}: {pnl2:+.2f} USDT\n"
                                    f"💸 Fees (included): {total_fees:.4f} USDT")
                        print(done_msg.replace('<b>', '').replace('</b>', ''))
                        reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                        await self._notify(done_msg, reply_to)
                        
                        # WAIT FOR CANDLE: Block re-entry until next candle close
                        pair_info._wait_for_candle = True
                        print(f"⏸️ {s1}-{s2}: Re-entry blocked until next candle close (reason: desync)")
                        
                    except Exception as e:
                        print(f"⚠️ Desync close error for {s1}-{s2}: {e}")
                        import traceback
                        traceback.print_exc()
                        pair_info.is_trading = False
        except Exception as e:
            print(f"⚠️ Leg sync error: {e}")

    async def _add_test_pairs(self):
        """Add test pairs to active_pairs when test_mode is enabled."""
        test_mode = getattr(self.config, 'test_mode', False)
        if isinstance(test_mode, str):
            test_mode = test_mode.lower() in ('true', '1', 'yes')
        
        if not test_mode:
            return
            
        test_pairs_str = getattr(self.config, 'test_pairs', '') or ''
        test_pairs = [p.strip() for p in test_pairs_str.split(',') if p.strip()]
        
        if not test_pairs:
            print("⚠️ Test mode enabled but no test_pairs configured")
            return
            
        print(f"🧪 TEST MODE: Adding {len(test_pairs)} test pairs...")
        
        for pair_str in test_pairs:
            parts = pair_str.split('-')
            if len(parts) != 2:
                continue
            s1, s2 = parts[0].strip(), parts[1].strip()
            pair_set = frozenset([s1, s2])
            
            # Skip if already exists
            if pair_set in self.active_pairs:
                print(f"  Test pair {s1}-{s2} already in active_pairs")
                continue
            
            # Add to DB and active_pairs
            try:
                new_pair = db.Pairs(
                    symbol1=s1,
                    symbol2=s2,
                    hedge_ratio=1.0,  # Default hedge for test
                    half_life=24.0,   # Default half-life
                    position_status=0
                )
                await db.add_pair(new_pair)
                
                new_info = PairInfo(
                    symbol1=s1,
                    symbol2=s2,
                    hedge_ratio=1.0,
                    half_life=24.0,
                    db_id=new_pair.id
                )
                self.active_pairs[pair_set] = new_info
                self._register_pair(new_info)
                print(f"  ✅ Added test pair: {s1}-{s2}")
            except Exception as e:
                print(f"  ⚠️ Error adding test pair {s1}-{s2}: {e}")

    async def initialize_all_symbols_data(self, target_symbols=None, concurrency=20):
        """
        Loads historical data for specified symbols with controlled concurrency.
        Prioritizes active pairs and priority pairs.
        """
        symbols_to_load = target_symbols if target_symbols else list(self.all_symbols.keys())
        #print(f"Initializing history for {len(symbols_to_load)} symbols (Concurrency: {concurrency})...")
        start_time = time.time()
        
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
                with open(priority_file_path, 'r') as f:
                    file_pairs = json.load(f)
                    if isinstance(file_pairs, list):
                        for p_str in file_pairs:
                            parts = p_str.split('-')
                            if len(parts) == 2:
                                s1, s2 = parts[0].strip(), parts[1].strip()
                                # Only add if it's in the target list
                                if s1 in symbols_to_load: priority_symbols.add(s1)
                                if s2 in symbols_to_load: priority_symbols.add(s2)
            except: pass
            
        # Sort symbols: priority first, then others
        other_symbols = [s for s in symbols_to_load if s not in priority_symbols]
        sorted_symbols = list(priority_symbols) + other_symbols
        
        # CRITICAL: Always include BTCUSDT for market beta calculation
        if 'BTCUSDT' not in sorted_symbols and 'BTCUSDT' in self.all_symbols:
            sorted_symbols.append('BTCUSDT')
            print("📈 Added BTCUSDT for market beta calculation")
        
        print(f"Priority symbols: {len(priority_symbols)}, Others: {len(other_symbols)}")
        
        # 2. Batch processing with semaphore
        sem = asyncio.Semaphore(concurrency)
        
        async def load_safe(symbol):
            async with sem:
                # Check if data exists
                if symbol not in self.all_data:
                    self.all_data[symbol] = Data(maxlen=self.max_len)
                    await self._initialize_history(symbol)
                
        tasks = [load_safe(s) for s in sorted_symbols]
        if tasks:
            await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        print(f"✅ History initialization finished in {elapsed:.2f}s.")
        
        # CRITICAL: Ensure BTCUSDT is loaded for beta calculation
        if 'BTCUSDT' not in self.all_data:
            print("📈 Loading BTCUSDT for beta calculation...")
            self.all_data['BTCUSDT'] = Data(maxlen=self.max_len)
            await self._initialize_history('BTCUSDT')
        
        btc_len = len(self.all_data.get('BTCUSDT', Data()).close) if 'BTCUSDT' in self.all_data else 0
        print(f"📊 BTCUSDT data: {btc_len} candles loaded")
        
        # CRITICAL: Run Discovery to find cointegrated pairs BEFORE checking signals
        print("🔍 Running initial Discovery...")
        await self._discover_new_pairs()
        
        
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
            # Full analysis: discovery + signal check
            self.loop.create_task(self.run_analysis(symbol))

    # Legacy method for backward compatibility (single TF mode)
    async def add_kline(self, kline_data):
        """Legacy method - calls add_kline_main for backward compatibility."""
        await self.add_kline_main(kline_data)

    async def _initialize_history(self, symbol):
        """
        Loads historical data to initialize deques.
        """
        #print(f"Initializing history for {symbol}...")
        try:
            klines = await self.client.klines(symbol, self.timeframe, limit=self.max_len)
            data = self.all_data[symbol]
            for k in klines:
                data.add_kline(k[0], k[1], k[2], k[3], k[4])
            #print(f"History for {symbol} initialized with {len(data.ts)} candles.")
        except Exception as e:
            print(f"Error initializing history for {symbol}: {e}")
            if symbol in self.all_data:
                del self.all_data[symbol]

    async def run_analysis(self, updated_symbol: str):
        """
        Runs analysis for pairs containing the updated symbol.
        """
        # 1. Check signals for active pairs
        await self._check_signals_for_active_pairs(updated_symbol)

        # 2. Periodically run discovery (every 10 minutes)
        now = time.time()
        if now - self._last_discovery_time > 600:
            if self._discovery_task is None or self._discovery_task.done():
                self._last_discovery_time = now
                self._discovery_task = self.loop.create_task(self._discover_new_pairs())


    async def _check_signals_for_active_pairs(self, updated_symbol: str):
        """
        Checks for trading signals and handles pair rotation.
        """
        current_pairs = list(self.active_pairs.items())

        for pair_set, pair_info in current_pairs:
            if pair_info.is_trading:
                continue
                
            if updated_symbol in pair_set:
                s1, s2 = pair_info.symbol1, pair_info.symbol2
                
                if s1 not in self.all_data or s2 not in self.all_data:
                    continue
                
                data1 = self.all_data[s1]
                data2 = self.all_data[s2]

                if len(data1.close) < self.min_data_points or len(data2.close) < self.min_data_points:
                    continue

                if data1.ts[-1] != data2.ts[-1]:
                    continue

                log_prices1 = np.log(list(data1.close)[-self.min_data_points:])
                log_prices2 = np.log(list(data2.close)[-self.min_data_points:])

                # Dynamic recalculation of cointegration (with configurable p-value threshold)
                p_value_threshold = getattr(self.config, 'p_value_threshold', 0.05) or 0.05
                flag, hedge, hl, pval = utils.calculate_cointegration(log_prices1, log_prices2, p_value_threshold, strict_hl=False)

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
                # Check if this is a test pair that should not be removed
                is_protected_test_pair = False
                test_mode = getattr(self.config, 'test_mode', False)
                if isinstance(test_mode, str):
                    test_mode = test_mode.lower() in ('true', '1', 'yes')
                if test_mode:
                    test_pairs_str = getattr(self.config, 'test_pairs', '') or ''
                    t_pairs = [p.strip() for p in test_pairs_str.split(',')]
                    if f"{s1}-{s2}" in t_pairs or f"{s2}-{s1}" in t_pairs:
                        is_protected_test_pair = True
                        # Use default hedge for test pairs if calculation failed
                        if flag == 0 or np.isnan(hedge):
                            hedge = 1.0
                        if np.isnan(hl) or hl > 200:
                            hl = 24.0

                # Pair rotation: if cointegration breaks (skip for protected test pairs)
                if flag == 0 and not is_protected_test_pair:
                    print(f"⚠️ Pair {s1}-{s2} correlation broken (pval: {pval:.4f}, HL: {hl}). Removing...")
                    
                    if pair_info.position_status != 0:
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
                        pair_info.close_handled = True
                        pair_info.is_trading = True
                        
                        # CRITICAL: Await close before removing from active_pairs to avoid zombie positions
                        try:
                            await self._execute_trade(pair_info, 0, close_reason='broken_coint')
                        except Exception as e:
                            print(f"❌ Failed to close broken pair {s1}-{s2}: {e}. Keeping in active list to retry.")
                            continue # Do not delete pair if close failed

                    if pair_info.db_id:
                        reason = f"Broken. Flag={flag}, HL={hl:.2f}, Pval={pval:.4f}"
                        history_item = db.PairHistory(
                            symbol1=s1,
                            symbol2=s2,
                            event_type='BROKEN',
                            timestamp=int(time.time() * 1000),
                            hedge_ratio=hedge,
                            half_life=hl,
                            reason=reason
                        )
                        # CRITICAL: Await this to prevent pool exhaustion (was create_task)
                        try:
                            await db.add_pair_history(history_item)
                            await db.delete_pair(pair_info.db_id)
                        except Exception as e:
                            print(f"⚠️ Failed to update DB content for broken pair {s1}-{s2}: {e}")
                    
                    if pair_set in self.active_pairs:
                        self._unregister_pair(pair_info)
                        del self.active_pairs[pair_set]
                    continue

                # Update parameters
                pair_info.hedge_ratio = hedge
                pair_info.half_life = hl
                
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

                z_entry = self.config.z_entry if self.config and self.config.z_entry else 1.9
                z_exit = self.config.z_exit if self.config and self.config.z_exit is not None else 0.0
                z_stop = self.config.z_stop if self.config and self.config.z_stop else 4.0
                
                # Test mode flag
                test_mode = getattr(self.config, 'test_mode', False)
                if test_mode and isinstance(test_mode, str):
                    test_mode = test_mode.lower() in ('true', '1', 'yes')
                
                # Signal logic
                if pair_info.position_status == 0:
                    # CANDLE CLOSE: Reset _wait_for_candle flag
                    # This is the ONLY place where the flag is reset — a new candle has closed,
                    # so the pair is now eligible for re-entry with fresh parameters.
                    if getattr(pair_info, '_wait_for_candle', False):
                        pair_info._wait_for_candle = False
                        print(f"✅ {s1}-{s2}: New candle closed, pair eligible for re-entry")
                    
                    # Check position limits before opening
                    if not self.can_open_new_position(s1, s2):
                        continue
                    
                    # In test_mode: force open trades without z-score signals (only on slow TFs)
                    if test_mode:
                        test_pairs_str = getattr(self.config, 'test_pairs', '') or ''
                        test_pairs = [p.strip() for p in test_pairs_str.split(',') if p.strip()]
                        pair_key = f"{s1}-{s2}"
                        reverse_key = f"{s2}-{s1}"
                        
                        # Force open only on slow timeframes (15m+) where signals are rare
                        slow_timeframes = ['15m', '30m', '1h', '2h', '4h', '1d']
                        should_force = self.timeframe in slow_timeframes
                        
                        if (pair_key in test_pairs or reverse_key in test_pairs) and should_force:
                            print(f"🧪 TEST MODE: Force opening {s1}-{s2} (z={z_score:.2f}, tf={self.timeframe})")
                            pair_info.is_trading = True
                            self.loop.create_task(self._execute_trade(pair_info, 1))
                            continue
                    
                    # NOTE: Entry signals are now handled by real-time Z-score monitoring
                    # (on_ticker_update + _signal_confirmation_loop)
                    # This discovery loop only validates cointegration and updates parameters
                    pass
                
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

    async def _discover_new_pairs(self):
        """
        Finds new cointegrated pairs using parallel processing.
        """
        print("Starting discovery process for new cointegrated pairs (PARALLEL)...")
        start_time = time.time()
        
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

        print(f"Analyzing {len(ready_symbols)} symbols using {self.min_data_points} candles.")
        ready_set = set(ready_symbols)
        checked_pairs = set()
        candidates_to_process = []
        
        # --- 1. Load and process Priority Pairs ---
        priority_file_path = getattr(self.config, 'priority_pairs_file', 'best_pairs.json')
        # Handle path resolution - relative to script directory
        if priority_file_path and not os.path.isabs(priority_file_path):
             script_dir = os.path.dirname(os.path.abspath(__file__))
             priority_file_path = os.path.join(script_dir, priority_file_path)

        priority_pairs = []
        if priority_file_path and os.path.exists(priority_file_path):
            try:
                with open(priority_file_path, 'r') as f:
                    file_pairs = json.load(f)
                    if isinstance(file_pairs, list):
                        for p_str in file_pairs:
                            parts = p_str.split('-')
                            if len(parts) == 2:
                                s1, s2 = parts[0].strip(), parts[1].strip()
                                if s1 in ready_set and s2 in ready_set:
                                    pair_set = frozenset([s1, s2])
                                    if pair_set not in self.active_pairs and pair_set not in checked_pairs:
                                        priority_pairs.append((s1, s2))
                                        checked_pairs.add(pair_set)
                
                if priority_pairs:
                    print(f"⭐ Found {len(priority_pairs)} valid candidates from priority list.")
                    candidates_to_process.extend(priority_pairs)
            except Exception as e:
                print(f"⚠️ Error loading priority pairs from {priority_file_path}: {e}")
        else:
             print(f"Info: Priority file not found at {priority_file_path}")

        # --- 2. Generate standard combinations ---
        all_combinations = itertools.combinations(ready_symbols, 2)
        added_count = 0
        for p in all_combinations:
            pair_set = frozenset(p)
            if pair_set not in self.active_pairs and pair_set not in checked_pairs:
                candidates_to_process.append(p)
                added_count += 1
                
        total_pairs = len(candidates_to_process)
        print(f"Total pairs to check: {total_pairs} (Priority: {len(priority_pairs)}, Others: {added_count})")
        
        if total_pairs == 0:
            return

        CHUNK_SIZE = 5000
        # Priority pairs are first in the list, so they will be in the first chunks
        chunks = [candidates_to_process[i:i + CHUNK_SIZE] for i in range(0, total_pairs, CHUNK_SIZE)]
        print(f"Split into {len(chunks)} chunks for parallel processing.")
        
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
            )
            tasks.append(task)
        
        results_list = await asyncio.gather(*tasks)
        
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
                    new_pair = db.Pairs(
                        symbol1=s1, 
                        symbol2=s2, 
                        hedge_ratio=hedge, 
                        half_life=hl,
                        position_status=0
                    )
                    await db.add_pair(new_pair)
                    
                    history_item = db.PairHistory(
                        symbol1=s1, 
                        symbol2=s2, 
                        event_type='FOUND',
                        timestamp=int(time.time() * 1000),
                        hedge_ratio=hedge,
                        half_life=hl,
                        reason='Discovery'
                    )
                    # CRITICAL: Await this to prevent flooding DB pool with 13k+ tasks
                    await db.add_pair_history(history_item)
                    
                    pair_set = frozenset([s1, s2])
                    
                    # === BETA CHECK BEFORE ADDING TO ACTIVE PAIRS ===
                    # Calculate beta to BTC to ensure pair is market-neutral
                    beta_btc = 0.0
                    beta_threshold = getattr(self.config, 'beta_threshold', 0.11) or 0.11
                    
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
                    test_mode = getattr(self.config, 'test_mode', False)
                    if isinstance(test_mode, str):
                        test_mode = test_mode.lower() in ('true', '1', 'yes')
                    
                    if not test_mode and not np.isnan(beta_btc) and abs(beta_btc) >= beta_threshold:
                        print(f"⚠️ {s1}-{s2} REJECTED at discovery: |beta|={abs(beta_btc):.3f} >= {beta_threshold}")
                        # Remove from DB since we just added it
                        try:
                            await db.delete_pair(new_pair.id)
                        except:
                            pass
                        continue  # Skip this pair
                    elif test_mode and not np.isnan(beta_btc) and abs(beta_btc) >= beta_threshold:
                        print(f"🧪 TEST MODE: {s1}-{s2} |beta|={abs(beta_btc):.3f} >= {beta_threshold} - ALLOWED for testing")
                    
                    print(f"✅ FOUND: {s1}-{s2} | HL: {hl:.2f}, P: {pval:.4f}, Beta: {beta_btc:.3f}, Hedge: {hedge:.4f}")
                    
                    # Final duplicate check before adding (race condition protection)
                    if pair_set in self.active_pairs:
                        print(f"  ⚠️ Skipping duplicate (race condition): {s1}-{s2}")
                        continue
                    
                    pair_info = PairInfo(
                        symbol1=s1, 
                        symbol2=s2, 
                        hedge_ratio=hedge, 
                        half_life=hl,
                        db_id=new_pair.id
                    )
                    pair_info.beta_btc = beta_btc
                    pair_info.discovered_at = time.time()  # Track when pair was discovered
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

    async def _notify(self, message, reply_to_msg_id=None, reply_markup=None):
        """Sends a notification via the configured callback. Returns msg_id for reply threading."""
        if self.notify_callback:
            try:
                return await self.notify_callback(message, reply_to_msg_id, reply_markup)
            except Exception as e:
                print(f"Error in _notify: {e}")
        return None

    def _format_half_life(self, hl_hours: float) -> str:
        """Format half-life in human-readable format (e.g., '1d 6h' or '16h 48m')."""
        if hl_hours >= 24:
            days = int(hl_hours // 24)
            hours = int(hl_hours % 24)
            return f"{days}d {hours}h" if hours > 0 else f"{days}d"
        else:
            hours = int(hl_hours)
            mins = int((hl_hours - hours) * 60)
            return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"

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

    def _add_to_best_pairs(self, symbol1: str, symbol2: str):
        """
        Add a successfully traded pair to best_pairs.json for priority loading.
        Only adds if the pair doesn't already exist.
        """
        try:
            priority_file_path = getattr(self.config, 'priority_pairs_file', 'best_pairs.json')
            if priority_file_path and not os.path.isabs(priority_file_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                priority_file_path = os.path.join(script_dir, priority_file_path)
            
            if not priority_file_path:
                return
            
            # Load existing pairs
            existing_pairs = []
            if os.path.exists(priority_file_path):
                with open(priority_file_path, 'r') as f:
                    existing_pairs = json.load(f)
            
            # Create pair string
            pair_str = f"{symbol1}-{symbol2}"
            pair_str_rev = f"{symbol2}-{symbol1}"
            
            # Check if already exists
            if pair_str not in existing_pairs and pair_str_rev not in existing_pairs:
                existing_pairs.append(pair_str)
                with open(priority_file_path, 'w') as f:
                    json.dump(existing_pairs, f, indent=2)
                print(f"✅ Added {pair_str} to best_pairs.json")
        except Exception as e:
            print(f"⚠️ Could not add to best_pairs: {e}")

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
        for pair_set, pair_info in list(self.active_pairs.items()):
            if pair_info.position_status != 0:
                continue  # Skip pairs with open positions
            
            # Skip pairs in cooldown (recently closed by SL)
            if getattr(pair_info, '_close_cooldown_until', 0) > time.time():
                continue
            
            # Skip pairs waiting for next candle close before re-entry
            if getattr(pair_info, '_wait_for_candle', False):
                continue
            
            s1, s2 = pair_info.symbol1, pair_info.symbol2
            if s1 in self.all_data and s2 in self.all_data:
                await self._check_signals_for_active_pairs(s1)
                analyzed += 1
                
                # Check if we filled all slots
                if self.count_active_positions() >= max_pairs:
                    break
        
        print(f"🔍 Immediate analysis complete. Checked {analyzed} pairs.")

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

    async def _refresh_exchange_position_count(self):
        """Refresh cached exchange position count from exchange API."""
        try:
            account = await self.client.account()
            positions = {}
            for pos in account.get('positions', []):
                amt = abs(float(pos.get('positionAmt', 0)))
                if amt > 0:
                    positions[pos['symbol']] = amt
            self._exchange_positions_cache = positions
            # Count pairs: each pair = 2 positions, so positions / 2 = pairs
            self._exchange_position_count = len(positions)
            return len(positions)
        except Exception as e:
            print(f"⚠️ Failed to refresh exchange position count: {e}")
            return self._exchange_position_count

    def can_open_new_position(self, s1: str, s2: str, exclude_pair=None) -> bool:
        """Check if we can open a new position for this pair.
        
        Args:
            exclude_pair: PairInfo to exclude from checks (prevents self-blocking
                          when the pair being opened has is_trading=True already set).
        """
        # Check if trading is enabled
        trade_mode = getattr(self.config, 'trade_mode', True)
        if trade_mode is not None and str(trade_mode).lower() in ('false', '0', 'no'):
            return False
        
        # Check max active pairs limit (local memory)
        max_pairs = getattr(self.config, 'max_active_pairs', 5) or 5
        if self.count_active_positions(exclude_pair=exclude_pair) >= max_pairs:
            return False
        
        # SAFETY: Also check exchange position cache (positions / 2 = pairs)
        # Each pair opens 2 positions, so max positions = max_pairs * 2
        max_exchange_positions = max_pairs * 2
        if self._exchange_position_count >= max_exchange_positions:
            print(f"🚫 Exchange position limit: {self._exchange_position_count}/{max_exchange_positions} positions on exchange")
            return False
        
        # Check symbol lock - each symbol can only be in one active pair
        if self.is_symbol_locked(s1, exclude_pair=exclude_pair) or self.is_symbol_locked(s2, exclude_pair=exclude_pair):
            return False
        
        return True

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
                    print(f"⚠️ Could not verify exchange positions: {e}. Proceeding with local check only.")
                
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
                pair_info._leverage_fail_until = time.time() + 600  # 10 min cooldown
                return
        
        try:
            if direction == 0:
                if pair_info.position_status == 0:
                    return

                print(f"EXECUTING CLOSE for {s1}-{s2} (reason: {close_reason})")
                
                # Store close reason IMMEDIATELY so external handlers can see it
                pair_info.last_close_reason = close_reason or 'unknown'
                
                side1_close = 'SELL' if pair_info.position_status == 1 else 'BUY'
                side2_close = 'BUY' if pair_info.position_status == 1 else 'SELL'
                qty1_close = pair_info.qty1
                qty2_close = pair_info.qty2
                
                # FAST PATH: For SL/TP triggered closes, one leg is already closed
                # Close the other leg IMMEDIATELY, cancel orders AFTER
                is_hardware_close = close_reason in ('hardware_sl', 'hardware_tp')
                
                if is_hardware_close:
                    # One leg already closed by exchange - close the other one ASAP
                    # Determine which leg is still open
                    triggered_symbol = getattr(pair_info, '_triggered_symbol', None)
                    if triggered_symbol == s1:
                        # s1 closed by SL/TP, close s2
                        if qty2_close and qty2_close > 0:
                            try:
                                await self.client.new_order(
                                    symbol=s2, side=side2_close, type='MARKET',
                                    quantity=qty2_close, reduceOnly='true', newOrderRespType='RESULT'
                                )
                                print(f"✅ FAST closed {s2} (qty={qty2_close})")
                            except Exception as e:
                                print(f"⚠️ Fast close {s2} failed: {e}")
                    elif triggered_symbol == s2:
                        # s2 closed by SL/TP, close s1
                        if qty1_close and qty1_close > 0:
                            try:
                                await self.client.new_order(
                                    symbol=s1, side=side1_close, type='MARKET',
                                    quantity=qty1_close, reduceOnly='true', newOrderRespType='RESULT'
                                )
                                print(f"✅ FAST closed {s1} (qty={qty1_close})")
                            except Exception as e:
                                print(f"⚠️ Fast close {s1} failed: {e}")
                    else:
                        # Unknown which leg triggered - close both using stored qty
                        close_tasks = []
                        if qty1_close and qty1_close > 0:
                            close_tasks.append(self.client.new_order(
                                symbol=s1, side=side1_close, type='MARKET',
                                quantity=qty1_close, reduceOnly='true', newOrderRespType='RESULT'
                            ))
                        if qty2_close and qty2_close > 0:
                            close_tasks.append(self.client.new_order(
                                symbol=s2, side=side2_close, type='MARKET',
                                quantity=qty2_close, reduceOnly='true', newOrderRespType='RESULT'
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
                        now_ms = int(time.time() * 1000)
                        start_ms = now_ms - 300_000  # Last 5 minutes
                        
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
                        now_ms = int(time.time() * 1000)
                        start_ms_pnl = now_ms - 300_000  # Last 5 minutes
                        trades_pnl_s1 = await self.client.get_account_trades(symbol=s1, startTime=start_ms_pnl, limit=50)
                        trades_pnl_s2 = await self.client.get_account_trades(symbol=s2, startTime=start_ms_pnl, limit=50)
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
                    
                    # Update trade record in DB
                    # Calculate fees from recent trades
                    _hw_fee1, _hw_fee2 = 0.0, 0.0
                    try:
                        _hw_fee1 = sum(float(t.get('commission', 0)) for t in trades_pnl_s1)
                        _hw_fee2 = sum(float(t.get('commission', 0)) for t in trades_pnl_s2)
                    except Exception:
                        pass
                    
                    if saved_trade_id:
                        try:
                            await db.update_trade_fields(
                                saved_trade_id,
                                status='CLOSED',
                                close_time=int(time.time() * 1000),
                                close_price_1=close_price1,
                                close_price_2=close_price2,
                                pnl=total_pnl,
                                close_z=close_zscore,
                                fee1=_hw_fee1,
                                fee2=_hw_fee2,
                                close_reason=close_reason or 'unknown',
                            )
                        except Exception as e:
                            print(f"⚠️ Trade record update failed: {e}")
                    
                    # Build and send close notification
                    HW_REASONS = {
                        'hardware_sl': '🛡️ Hardware Stop Loss',
                        'hardware_tp': '🛡️ Hardware Take Profit',
                    }
                    reason_text = HW_REASONS.get(close_reason, f'🛡️ Hardware {close_reason}')
                    
                    pnl_emoji = "🟢" if total_pnl > 0 else "🔴"
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
                    if (close_beta == 0 or close_pval == 0) and s1 in self.all_data and s2 in self.all_data:
                        try:
                            _d1 = self.all_data[s1]
                            _d2 = self.all_data[s2]
                            if len(_d1.close) >= self.min_data_points and len(_d2.close) >= self.min_data_points:
                                _lp1 = np.log(list(_d1.close)[-self.min_data_points:])
                                _lp2 = np.log(list(_d2.close)[-self.min_data_points:])
                                _, _, _, _pval = utils.calculate_cointegration(_lp1, _lp2, strict_hl=False)
                                if close_pval == 0 and not np.isnan(_pval):
                                    close_pval = float(_pval)
                                if close_beta == 0 and 'BTCUSDT' in self.all_data:
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
                    
                    full_msg = f"{reason_text}: <b>{s1}/{s2}</b>\n\n"
                    full_msg += f"📊 Z: {close_zscore:+.2f} | β: {close_beta:.3f} | p: {close_pval:.4f}\n"
                    full_msg += f"⏳ HL: {close_hl} | Hedge: {pair_info.hedge_ratio:.4f}\n"
                    full_msg += f"💵 PnL: {pnl_emoji} <b>{total_pnl:+.2f} USDT</b>\n"
                    full_msg += f"   {e1} {s1}: {pnl1:+.2f} | {e2} {s2}: {pnl2:+.2f}\n"
                    
                    print(full_msg.replace('<b>', '').replace('</b>', ''))
                    reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                    await self._notify(full_msg, reply_to)
                    
                    # State cleanup for hardware close
                    pair_info.position_status = 0
                    pair_info.qty1 = 0
                    pair_info.qty2 = 0
                    pair_info.entry_price1 = 0
                    pair_info.entry_price2 = 0
                    pair_info.current_trade_id = None
                    
                    # Update exchange position cache
                    self._exchange_positions_cache.pop(s1, None)
                    self._exchange_positions_cache.pop(s2, None)
                    self._exchange_position_count = len(self._exchange_positions_cache)
                    
                    # WAIT FOR CANDLE: After ANY close, block re-entry until next candle closes
                    pair_info._wait_for_candle = True
                    print(f"⏸️ {s1}-{s2}: Re-entry blocked until next candle close (reason: {close_reason})")
                    
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
                            'close_pnl': total_pnl,
                            'close_reason': close_reason or 'unknown',
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
                            close_tasks.append(self.client.new_order(
                                symbol=s1, side=side1_close, type='MARKET', 
                                quantity=abs(open_positions[s1]), reduceOnly='true', newOrderRespType='RESULT'
                            ))
                            close_symbols.append(s1)
                        else:
                            print(f"ℹ️ {s1} already closed, skipping")
                            
                        if leg2_exists:
                            close_tasks.append(self.client.new_order(
                                symbol=s2, side=side2_close, type='MARKET', 
                                quantity=abs(open_positions[s2]), reduceOnly='true', newOrderRespType='RESULT'
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
                        # Close reason mapping
                        CLOSE_REASONS = {
                            'z_tp': '💰 Z-Score Take Profit',
                            'z_sl': '🛑 Z-Score Stop Loss',
                            'circuit': '🔴 Circuit Breaker',
                            'broken_coint': '🚨 Broken Correlation',
                            'hardware_sl': '🛡️ Hardware Stop Loss',
                            'hardware_tp': '🛡️ Hardware Take Profit',
                            'manual': '👤 Manual Close',
                            'desync': '⚠️ Leg Desync',
                            'beta_drift': '📉 Beta Drift',
                            'beta_critical': '🚨 Beta Critical',
                            'external': '⚡ External Close',
                            'orphan_restart': '🔄 Orphan on Restart',
                            'stale_symbols': '⏳ Stale Symbols',
                        }
                        reason_text = CLOSE_REASONS.get(close_reason, '❓ Unknown') if close_reason else '❓ Unknown'
                    
                        def get_price(order):
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
                                    now_ms = int(time.time() * 1000)
                                    start_ms = now_ms - 300_000  # Last 5 minutes
                                    trades = await self.client.get_account_trades(symbol=sym, startTime=start_ms, limit=50)
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
                            now_ms_pnl = int(time.time() * 1000)
                            start_ms_pnl = now_ms_pnl - 300_000
                            trades_s1 = await self.client.get_account_trades(symbol=s1, startTime=start_ms_pnl, limit=50)
                            trades_s2 = await self.client.get_account_trades(symbol=s2, startTime=start_ms_pnl, limit=50)
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
                    
                        pnl_emoji = "🟢" if total_pnl > 0 else "🔴"


                        # Calculate fees from recent trades (BEFORE trade record update)
                        _norm_fee1, _norm_fee2 = 0.0, 0.0
                        try:
                            _norm_fee1 = sum(float(t.get('commission', 0)) for t in trades_s1) if trades_s1 else 0.0
                            _norm_fee2 = sum(float(t.get('commission', 0)) for t in trades_s2) if trades_s2 else 0.0
                        except Exception:
                            pass

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
                            await db.update_trade_fields(
                                pair_info.current_trade_id,
                                status='CLOSED',
                                close_time=int(time.time() * 1000),
                                close_price_1=close_price1,
                                close_price_2=close_price2,
                                pnl=total_pnl,
                                close_z=close_zscore if close_zscore else 0.0,
                                fee1=_norm_fee1,
                                fee2=_norm_fee2,
                                close_reason=close_reason or 'unknown',
                            )
                    
                        pair_info.current_trade_id = None
                        pair_info.position_status = 0
                        pair_info.qty1 = 0
                        pair_info.qty2 = 0
                        pair_info.close_handled = True  # Mark as handled to prevent duplicate notification
                        pair_info.last_close_reason = close_reason or 'unknown'
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
                            
                            # Track which orders exist for each symbol
                            orders_by_sym = {s1: [], s2: []}
                            for o in algo_orders:
                                sym = o['symbol']
                                if sym in orders_by_sym:
                                    order_type = o.get('type', '') or o.get('orderType', '')
                                    if 'STOP' in order_type.upper():
                                        orders_by_sym[sym].append(('SL', o['algoId']))
                                    elif 'TAKE_PROFIT' in order_type.upper():
                                        orders_by_sym[sym].append(('TP', o['algoId']))
                                    else:
                                        orders_by_sym[sym].append((order_type or 'ORDER', o['algoId']))
                            
                            # Cancel each order and track result
                            for sym in [s1, s2]:
                                for order_type, algo_id in orders_by_sym[sym]:
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
                        if (close_beta == 0 or close_pval == 0) and s1 in self.all_data and s2 in self.all_data:
                            try:
                                _d1 = self.all_data[s1]
                                _d2 = self.all_data[s2]
                                if len(_d1.close) >= self.min_data_points and len(_d2.close) >= self.min_data_points:
                                    _lp1 = np.log(list(_d1.close)[-self.min_data_points:])
                                    _lp2 = np.log(list(_d2.close)[-self.min_data_points:])
                                    _, _, _, _pval = utils.calculate_cointegration(_lp1, _lp2, strict_hl=False)
                                    if close_pval == 0 and not np.isnan(_pval):
                                        close_pval = float(_pval)
                                    if close_beta == 0 and 'BTCUSDT' in self.all_data:
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
                        full_msg = f"{reason_text}: <b>{s1}/{s2}</b>\n\n"
                        full_msg += f"📊 Z: {close_zscore:+.2f} | β: {close_beta:.3f} | p: {close_pval:.4f}\n"
                        full_msg += f"⏳ HL: {close_hl} | Hedge: {pair_info.hedge_ratio:.4f}\n"
                        full_msg += f"💵 PnL: {pnl_emoji} <b>{total_pnl:+.2f} USDT</b>\n"
                        full_msg += f"   {e1} {s1}: {pnl1:+.2f} | {e2} {s2}: {pnl2:+.2f}\n\n"
                        full_msg += f"🛡️ Order Cleanup:\n{cleanup_msg}"
                        
                        print(full_msg.replace('<b>', '').replace('</b>', ''))
                        # Reply to original open message if available
                        reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                        await self._notify(full_msg, reply_to)
                        
                        # Update DB with close details + market neutrality metrics
                        if pair_info.db_id:
                            await db.update_pair({
                                'id': pair_info.db_id,
                                'position_status': 0,
                                'qty1': 0,
                                'qty2': 0,
                                'entry_price1': 0,
                                'entry_price2': 0,
                                'close_time': int(time.time()),
                                'close_pnl': total_pnl,
                                'close_reason': close_reason or 'unknown',
                                'pnl1': pnl1,
                                'pnl2': pnl2,
                                'fee1': _norm_fee1,
                                'fee2': _norm_fee2,
                                'beta_btc': close_beta,
                                'last_pvalue': close_pval,
                            })

                        
                        # AUTO-ADD to best_pairs.json on successful TP only
                        # BUG-7 FIX: Don't add pairs from forced closes (circuit, beta_drift, etc.)
                        if close_reason in ('z_tp', 'hardware_tp'):
                            self._add_to_best_pairs(s1, s2)
                        
                        # WAIT FOR CANDLE: After ANY close, block re-entry until next candle closes
                        # The pair can only re-enter when _check_signals_for_active_pairs resets the flag
                        pair_info._wait_for_candle = True
                        print(f"⏸️ {s1}-{s2}: Re-entry blocked until next candle close (reason: {close_reason})")
                        
                        # IMMEDIATE RE-ANALYSIS: Trigger search for new trades now that slot is free
                        print(f"🔄 Slot freed after closing {s1}-{s2}. Triggering immediate re-analysis...")
                        self.loop.create_task(self._trigger_immediate_analysis())
                        
                except Exception as e:
                    print(f"FATAL ERROR closing position for {s1}-{s2}: {e}")
                    # Ensure state cleanup even on error — position IS closed on exchange
                    try:
                        pair_info.position_status = 0
                        pair_info.qty1 = 0
                        pair_info.qty2 = 0
                        pair_info.entry_price1 = 0
                        pair_info.entry_price2 = 0
                        pair_info.close_handled = True
                        pair_info.last_close_reason = close_reason or 'unknown'
                        pair_info.current_trade_id = None
                        pair_info._wait_for_candle = True
                        self._exchange_positions_cache.pop(s1, None)
                        self._exchange_positions_cache.pop(s2, None)
                        self._exchange_position_count = len(self._exchange_positions_cache)
                        # Send error notification to TG
                        err_msg = f"⚠️ Close error {s1}-{s2}: {e}\nPosition closed on exchange but notification failed."
                        await self._notify(err_msg)
                    except Exception:
                        pass
                return

            hedge = pair_info.hedge_ratio
            s1_info = self.all_symbols.get(s1)
            s2_info = self.all_symbols.get(s2)

            if not s1_info or not s2_info:
                print(f"ERROR: Symbol info not found for {s1} or {s2}")
                pair_info.position_status = 0
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
                    return
            except Exception as e:
                print(f"⚠️ Hedge refresh error for {s1}-{s2}: {e}. Using existing hedge={hedge:.4f}")

            # === STRICT BETA CHECK per user request: Enforce beta_threshold=0.11 ===
            try:
                beta_threshold = getattr(self.config, 'beta_threshold', 0.11) or 0.11
                current_beta = getattr(pair_info, 'beta_btc', 0.0)
                if not np.isnan(current_beta) and abs(current_beta) >= beta_threshold:
                    warn_msg = f"⛔ BETA REJECT: {s1}-{s2} beta={current_beta:.3f} >= {beta_threshold}. Aborting entry."
                    print(warn_msg)
                    pair_info.position_status = 0
                    pair_info.is_trading = False
                    pair_info.pending_signal = None
                    pair_info.pending_since = None
                    # Notify TG to explain why signal was rejected
                    reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                    await self._notify(warn_msg, reply_to)
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
            
            # Get min_order_bump threshold from config (default 1.5)
            min_order_bump = getattr(self.config, 'min_order_bump', 1.5) or 1.5
            
            # Check if we should skip trade due to size constraints
            if utils.should_skip_trade(min_notional1, calculated_notional1, min_order_bump):
                print(f"SKIP: Trade for {s1}-{s2} cancelled - {s1} below min notional with excessive bump required")
                pair_info.position_status = 0
                pair_info.is_trading = False
                return
            
            if utils.should_skip_trade(min_notional2, calculated_notional2, min_order_bump):
                print(f"SKIP: Trade for {s1}-{s2} cancelled - {s2} below min notional with excessive bump required")
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
                            break
                        
                        # Also check if leverage is even supported
                        max_lev = max((b.get('initialLeverage', 0) for b in bracket_list), default=0)
                        if leverage > max_lev:
                            print(f"🚫 PRE-FLIGHT FAIL: {sym} max leverage is {max_lev}x, requested {leverage}x")
                            preflight_ok = False
                            break
                
                if not preflight_ok:
                    print(f"🚫 Trade aborted for {s1}-{s2}: pre-flight validation failed")
                    pair_info.position_status = 0
                    pair_info.is_trading = False
                    pair_info.pending_signal = None
                    pair_info.pending_since = None
                    pair_info._leverage_fail_until = time.time() + 600  # 10 min cooldown
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
                for res in results:
                    if isinstance(res, Exception):
                        print(f"ERROR placing order: {res}")
                        has_error = True
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
                                self.client.new_order(symbol=exec_symbol, side=revert_side, type='MARKET', quantity=exec_qty, reduceOnly='true')
                            )
                        except Exception as rev_e:
                            print(f"  CRITICAL: Failed to prepare revert {exec_symbol}: {rev_e}")
                
                    if revert_tasks:
                        await asyncio.gather(*revert_tasks, return_exceptions=True)
                
                    pair_info.position_status = 0
                    pair_info.is_trading = False
                    pair_info.close_handled = False  # Reset after revert completes
                    
                    # Cooldown to prevent immediate retry loop
                    pair_info.pending_signal = None
                    pair_info.pending_since = None
                    pair_info._leverage_fail_until = time.time() + 600  # 10 min cooldown
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
                    
                    # CRITICAL: Update exchange position cache immediately
                    # This prevents race condition where another task checks limit before cache refreshes
                    self._exchange_positions_cache[s1] = pair_info.qty1
                    self._exchange_positions_cache[s2] = pair_info.qty2
                    self._exchange_position_count = len(self._exchange_positions_cache)
                
                    def get_price(order):
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
                    
                    success_msg = (f"🚀 <b>Trade OPENED:</b> {s1}-{s2}\n"
                                   f"📅 {open_dt}\n\n"
                                   f"📈 LONG: {long_qty} {long_sym} @ {long_price:.4f}\n"
                                   f"     💰 ${long_qty * long_price:.2f}\n"
                                   f"📉 SHORT: {short_qty} {short_sym} @ {short_price:.4f}\n"
                                   f"     💰 ${short_qty * short_price:.2f}\n\n"
                                   f"⚖️ Hedge: {pair_info.hedge_ratio:.4f} | Z: {pair_info.entry_z_score:.2f}\n"
                                   f"📊 Beta: {pair_info.beta_btc:.3f} | p-value: {pair_info.last_pvalue:.4f}\n"
                                   # Format half-life as readable hours/days
                                   f"⏳ Half-life: {self._format_half_life(pair_info.half_life)}")
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
                        
                        # Calculate limit prices with 1% slippage
                        sl1_limit = round(sl1 * (0.99 if sl_side1 == 'SELL' else 1.01), s1_info.tick_size)
                        sl2_limit = round(sl2 * (0.99 if sl_side2 == 'SELL' else 1.01), s2_info.tick_size)
                        tp1_limit = round(tp1 * (1.01 if sl_side1 == 'SELL' else 0.99), s1_info.tick_size)
                        tp2_limit = round(tp2 * (1.01 if sl_side2 == 'SELL' else 0.99), s2_info.tick_size)

                        print(f"🛡️ Placing SL/TP (Algo): {s1} SL@{sl1} TP@{tp1}, {s2} SL@{sl2} TP@{tp2}")
                        
                        # Validate all prices are positive before placing orders
                        if sl1 <= 0 or sl2 <= 0 or tp1 <= 0 or tp2 <= 0:
                            warn_msg = (f"⚠️ CRITICAL: Invalid SL/TP prices for {s1}-{s2}! "
                                       f"sl1={sl1}, sl2={sl2}, tp1={tp1}, tp2={tp2}. "
                                       f"Force closing position.")
                            print(warn_msg)
                            print(f"  Entry prices: {pair_info.entry_price1}, {pair_info.entry_price2}")
                            print(f"  ATR values: {atr1}, {atr2}")
                            reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                            await self._notify(warn_msg, reply_to)
                            # Force close — can't leave positions unprotected
                            pair_info.close_handled = True
                            pair_info.is_trading = True
                            await self._execute_trade(pair_info, 0, close_reason='hardware_sl')
                            return
                        
                        # Use algo orders
                        protection_tasks = [
                            # SL Orders (STOP via algo endpoint)
                            self.client.new_algo_order(symbol=s1, side=sl_side1, type='STOP',
                                                       triggerPrice=sl1, price=sl1_limit,
                                                       quantity=pair_info.qty1, timeInForce='GTC', reduceOnly='true'),
                            self.client.new_algo_order(symbol=s2, side=sl_side2, type='STOP',
                                                       triggerPrice=sl2, price=sl2_limit,
                                                       quantity=pair_info.qty2, timeInForce='GTC', reduceOnly='true'),
                            # TP Orders (TAKE_PROFIT via algo endpoint)
                            self.client.new_algo_order(symbol=s1, side=sl_side1, type='TAKE_PROFIT',
                                                       triggerPrice=tp1, price=tp1_limit,
                                                       quantity=pair_info.qty1, timeInForce='GTC', reduceOnly='true'),
                            self.client.new_algo_order(symbol=s2, side=sl_side2, type='TAKE_PROFIT',
                                                       triggerPrice=tp2, price=tp2_limit,
                                                       quantity=pair_info.qty2, timeInForce='GTC', reduceOnly='true'),
                        ]
                        
                        results = await asyncio.gather(*protection_tasks, return_exceptions=True)
                        
                        # Collect successful order algoIds for potential cancellation
                        successful_algo_ids = []
                        failed_count = 0
                        for res in results:
                            if isinstance(res, Exception):
                                print(f"⚠️ WARN: Failed to place protection order: {res}")
                                failed_count += 1
                            elif isinstance(res, dict) and 'algoId' in res:
                                successful_algo_ids.append(res['algoId'])
                        
                        if failed_count == 0 and len(successful_algo_ids) == 4:
                            print(f"🛡️ Protection placed successfully (4 orders)")
                            # Store algo order mapping for ALGO_UPDATE event handling
                            pair_key = frozenset([s1, s2])
                            for i, aid in enumerate(successful_algo_ids):
                                aid_str = str(aid)  # Ensure consistent string keys
                                if i == 0:
                                    self.algo_orders[aid_str] = {'pair_key': pair_key, 'symbol': s1, 'type': 'STOP'}
                                elif i == 1:
                                    self.algo_orders[aid_str] = {'pair_key': pair_key, 'symbol': s2, 'type': 'STOP'}
                                elif i == 2:
                                    self.algo_orders[aid_str] = {'pair_key': pair_key, 'symbol': s1, 'type': 'TAKE_PROFIT'}
                                elif i == 3:
                                    self.algo_orders[aid_str] = {'pair_key': pair_key, 'symbol': s2, 'type': 'TAKE_PROFIT'}
                        elif failed_count > 0:
                            warn_msg = f"⚠️ CRITICAL: Protection partially FAILED for {s1}-{s2} ({failed_count}/4 failed). Force closing!"
                            print(warn_msg)
                            reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                            await self._notify(warn_msg, reply_to)
                            
                            # Cancel successfully placed orders using algoId from results
                            if successful_algo_ids:
                                try:
                                    cancel_tasks = [self.client.cancel_algo_order(algoId=aid) for aid in successful_algo_ids]
                                    await asyncio.gather(*cancel_tasks, return_exceptions=True)
                                    print(f"🗑️ Cancelled {len(successful_algo_ids)} partial algo orders")
                                except Exception as ce:
                                    print(f"⚠️ Could not cancel partial orders: {ce}")
                            
                            # Force close position
                            pair_info.close_handled = True  # Prevent duplicate notification from WS handler
                            pair_info.is_trading = True
                            await self._execute_trade(pair_info, 0, close_reason='hardware_sl')
                            
                    except Exception as e:
                        warn_msg = f"⚠️ CRITICAL ERROR placing hardware SL for {s1}-{s2}: {e}. Force closing position!"
                        print(warn_msg)
                        reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                        await self._notify(warn_msg, reply_to)
                        
                        # Force close position (algo orders will be cancelled by _execute_trade)
                        pair_info.close_handled = True  # Prevent duplicate notification from WS handler
                        pair_info.is_trading = True
                        await self._execute_trade(pair_info, 0, close_reason='hardware_sl')
                    # === END HARDWARE SL/TP ===
                
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
                    except Exception as e:
                        print(f"Error creating trade record: {e}")
            
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
            if getattr(pair_info, '_wait_for_candle', False):
                continue
            
            # Check if signal (between z_entry and z_entry_max)
            # Reject if already too extreme - spread may be broken
            if abs(z_score) >= z_entry and abs(z_score) < z_entry_max:
                if pair_info.pending_signal is None:
                    # Start confirmation timer
                    pair_info.pending_signal = z_score
                    pair_info.pending_since = time.time()
            else:
                # Signal went away - reset
                if pair_info.pending_signal is not None:
                    pair_info.pending_signal = None
                    pair_info.pending_since = None
    
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
        
        # --- 1. Z-Score TP / SL (instant, every tick) ---
        z_exit = self.config.z_exit if self.config and self.config.z_exit is not None else 0.0
        z_stop = self.config.z_stop if self.config and self.config.z_stop else 4.0
        
        if pair_info.position_status == 1:  # Long spread
            if z_score >= z_exit:
                print(f"💰 RT TAKE PROFIT (Long) on {s1}-{s2}. Z: {z_score:.2f} >= {z_exit}. Closing...")
                pair_info.close_handled = True
                pair_info.is_trading = True
                await self._execute_trade(pair_info, 0, close_reason='z_tp')
                return
            elif z_score <= -z_stop:
                print(f"🛑 RT STOP LOSS (Long) on {s1}-{s2}. Z: {z_score:.2f} <= -{z_stop}. Closing...")
                pair_info.close_handled = True
                pair_info.is_trading = True
                await self._execute_trade(pair_info, 0, close_reason='z_sl')
                return
        
        elif pair_info.position_status == -1:  # Short spread
            if z_score <= -z_exit:
                print(f"💰 RT TAKE PROFIT (Short) on {s1}-{s2}. Z: {z_score:.2f} <= {-z_exit}. Closing...")
                pair_info.close_handled = True
                pair_info.is_trading = True
                await self._execute_trade(pair_info, 0, close_reason='z_tp')
                return
            elif z_score >= z_stop:
                print(f"🛑 RT STOP LOSS (Short) on {s1}-{s2}. Z: {z_score:.2f} >= {z_stop}. Closing...")
                pair_info.close_handled = True
                pair_info.is_trading = True
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
                
                for pair_info in list(self.active_pairs.values()):
                    if pair_info.position_status != 0 or pair_info.is_trading:
                        continue
                    
                    if pair_info.pending_signal is not None and pair_info.pending_since is not None:
                        elapsed = time.time() - pair_info.pending_since
                        
                        if elapsed >= confirm_sec:
                            # Re-check current Z-score
                            price1 = self.last_prices.get(pair_info.symbol1)
                            price2 = self.last_prices.get(pair_info.symbol2)
                            
                            if price1 and price2:
                                current_z = self._calc_realtime_zscore(pair_info, price1, price2)
                                z_entry = getattr(self.config, 'z_entry', 1.9) or 1.9
                                z_entry_max = getattr(self.config, 'z_entry_max', 2.5) or 2.5
                                
                                # Check signal still valid and in same direction
                                # Also reject if Z-score exceeds entry window (spread may be broken)
                                if abs(current_z) >= z_entry and abs(current_z) < z_entry_max and (current_z * pair_info.pending_signal > 0):
                                    # Check can open
                                    if self.can_open_new_position(pair_info.symbol1, pair_info.symbol2):
                                        # Check cooldown from failed leverage/trade
                                        fail_until = getattr(pair_info, '_leverage_fail_until', 0)
                                        if fail_until and time.time() < fail_until:
                                            continue
                                        
                                        # Check cooldown after stop-loss close
                                        close_cooldown = getattr(pair_info, '_close_cooldown_until', 0)
                                        if close_cooldown and time.time() < close_cooldown:
                                            remaining = int(close_cooldown - time.time())
                                            print(f"⏸️ {pair_info.symbol1}-{pair_info.symbol2}: Entry blocked by SL cooldown ({remaining}s remaining)")
                                            continue
                                        
                                        # Check if pair is waiting for next candle close
                                        if getattr(pair_info, '_wait_for_candle', False):
                                            continue
                                        
                                        direction = 1 if current_z < 0 else -1
                                        pair_info.entry_z_score = current_z
                                        print(f"✅ Signal CONFIRMED for {pair_info.symbol1}-{pair_info.symbol2}: Z={current_z:.2f}. Opening position...")
                                        pair_info.is_trading = True
                                        self.loop.create_task(self._execute_trade(pair_info, direction))
                                elif abs(current_z) >= z_entry_max:
                                    print(f"⚠️ {pair_info.symbol1}-{pair_info.symbol2}: Z={current_z:.2f} exceeds z_entry_max={z_entry_max}. Skipping entry (spread may be broken).")
                            
                            # Reset pending after confirmed check (timer expired)
                            pair_info.pending_signal = None
                            pair_info.pending_since = None
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