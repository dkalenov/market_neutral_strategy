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


MAX_LEN = 500
COINT_WINDOW = 200

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
    # Signal confirmation (for real-time mode)
    pending_signal: float = None  # Pending Z-score signal awaiting confirmation
    pending_since: float = None   # Time when signal started

class PairsManager:
    """
    Manages symbol data, finds cointegrated pairs, and generates signals.
    Supports Multi-Timeframe (MTF) mode: main TF for discovery, entry TF for faster signals.
    """
    def __init__(self, client, loop, all_symbols, timeframe='1h', entry_timeframe=None, min_data_points=200, notify_callback=None, config_info=None):
        self.client = client
        self.loop = loop
        self.all_symbols = all_symbols
        self.timeframe = timeframe
        self.entry_timeframe = entry_timeframe or timeframe  # Default to main TF if not specified
        self.min_data_points = min_data_points
        self.notify_callback = notify_callback
        self.config = config_info
        
        self.max_len = int(min_data_points * 2.5)
        
        # Main TF data (for discovery + validation)
        self.all_data: dict[str, Data] = {}
        # Entry TF data (for faster signal detection) - only used if MTF mode
        self.entry_data: dict[str, Data] = {}
        
        self.active_pairs: dict[frozenset, PairInfo] = {}
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
        
        # NOTE: Initialization is now EXPLICIT - call await pairs_manager.initialize() in main.py
        self._initialized = False

    async def initialize(self):
        """
        MUST be called after creation and awaited before any trading.
        Loads state from DB and reconciles with exchange.
        """
        if self._initialized:
            return
        await self._load_state_from_db()
        self._initialized = True
        max_pairs = getattr(self.config, 'max_active_pairs', 5) or 5
        print(f"✅ PairsManager initialized. Max active pairs: {max_pairs}")
        
        # Start periodic leg sync loop (BACKUP only - primary sync is via WebSocket)
        self._leg_sync_task = self.loop.create_task(self._periodic_leg_sync_loop())
        print("🔄 Started backup leg sync loop (every 30s, primary via WebSocket)")

    async def _load_state_from_db(self):
        print("Restoring active positions from DB...")
        try:
            pairs = await db.get_all_pairs()
            
            # Only restore pairs with ACTIVE positions (status != 0)
            active_count = 0
            for p in pairs:
                # Skip pairs without open positions - they'll be re-discovered
                if p.position_status == 0:
                    continue
                    
                pair_set = frozenset([p.symbol1, p.symbol2])
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

                last_trade = await db.get_last_open_trade_for_pair(p.id)
                if last_trade:
                    info.current_trade_id = last_trade.id
                    print(f"  ✅ Restored: {p.symbol1}-{p.symbol2} | Trade ID: {last_trade.id}")
                else:
                    print(f"  ⚠️ Restored: {p.symbol1}-{p.symbol2} | No trade record")

                self.active_pairs[pair_set] = info
                active_count += 1
            
            print(f"Restored {active_count} active pairs from DB.")
        except Exception as e:
            print(f"Error loading state from DB: {e}")
        
        # CRITICAL: Reconcile DB state with actual exchange positions
        await self._reconcile_with_exchange()
        
        # Add test pairs if test_mode is enabled
        await self._add_test_pairs()

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

            # Note: algo orders (STOP/TAKE_PROFIT) cannot be fetched on mainnet due to API limitation

            print(f"  Exchange has {len(open_on_exchange)} open positions and {len(all_open_orders)} regular orders.")
            
            # Warn about positions on exchange that are NOT in our DB
            tracked_symbols = set()
            for pair_info in self.active_pairs.values():
                if pair_info.position_status != 0:
                    tracked_symbols.add(pair_info.symbol1)
                    tracked_symbols.add(pair_info.symbol2)
            
            unknown_positions = [s for s in open_on_exchange.keys() if s not in tracked_symbols]
            if unknown_positions:
                print(f"  🚨 UNKNOWN POSITIONS on exchange (not tracked by bot): {unknown_positions}")
                await self._notify(f"🚨 WARNING: {len(unknown_positions)} unknown positions on exchange: {unknown_positions[:5]}... Close manually!")
            
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
                        # Mark as closed in DB to prevent sync loop from trying to close
                        print(f"  🚨 ORPHAN: {s1}-{s2} has mismatched legs! {s1}:{s1_open}, {s2}:{s2_open}")
                        print(f"      → Marking as CLOSED in DB. Close remaining leg manually!")
                        pairs_to_fix.append((pair_info, 'close_db'))
                        remaining_sym = s1 if s1_open else s2
                        await self._notify(f"🚨 ORPHAN POSITION: {s1}-{s2}\n{remaining_sym} is still open on exchange!\nClose manually or it will remain open.")
                    else:
                        # Both open - check for SL orders (STOP type in algo orders)
                        has_sl1 = any(o.get('type') == 'STOP' for o in orders_by_symbol.get(s1, []))
                        has_sl2 = any(o.get('type') == 'STOP' for o in orders_by_symbol.get(s2, []))
                        if not has_sl1 or not has_sl2:
                            print(f"  ⚠️ WARNING: {s1}-{s2} is open but lacks SL orders! (s1:{has_sl1}, s2:{has_sl2})")
                            await self._notify(f"⚠️ PROTECT: {s1}-{s2} is open but lacks hardware SL protection on exchange!")
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
            
            active_count = self.count_active_positions()
            print(f"🔄 Reconciliation complete. Active pairs in DB: {active_count}")
            
        except Exception as e:
            print(f"❌ Error during reconciliation: {e}")
            import traceback
            traceback.print_exc()

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
            account = await self.client.account()
            exchange_positions = set()
            for pos in account.get('positions', []):
                if float(pos.get('positionAmt', 0)) != 0:
                    exchange_positions.add(pos['symbol'])
            
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

    async def handle_sl_tp_triggered(self, symbol: str):
        """
        Called when a hardware SL/TP order is filled (via WebSocket).
        Finds the pair containing this symbol and closes the other leg.
        """
        for pair_info in list(self.active_pairs.values()):
            if pair_info.position_status == 0:
                continue
            if symbol in (pair_info.symbol1, pair_info.symbol2):
                other_symbol = pair_info.symbol2 if symbol == pair_info.symbol1 else pair_info.symbol1
                msg = f"🚨 SL/TP triggered on {symbol}! Closing other leg {other_symbol}"
                print(msg)
                await self._notify(msg)
                
                # Force close the pair (cancels algo orders + closes other leg if needed)
                pair_info.is_trading = True
                await self._execute_trade(pair_info, 0, close_reason='hardware_sl')
                break

    async def _periodic_leg_sync_loop(self):
        """BACKUP: Periodically check leg sync and cleanup orphaned orders every 30 seconds.
        Primary sync is handled by userdata WebSocket."""
        while True:
            await asyncio.sleep(30)  # Backup check
            try:
                await self._check_leg_synchronization()
                await self._cleanup_orphaned_algo_orders()
            except Exception as e:
                print(f"⚠️ Leg sync/cleanup error: {e}")

    async def _check_leg_synchronization(self):
        """Check that both legs of each active pair are open."""
        try:
            account = await self.client.account()
            pos_by_symbol = {}
            for pos in account['positions']:
                amt = float(pos['positionAmt'])
                if amt != 0:
                    pos_by_symbol[pos['symbol']] = amt
            
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
                    # One leg closed unexpectedly
                    closed_leg = pair_info.symbol1 if not leg1_open else pair_info.symbol2
                    remaining_leg = pair_info.symbol2 if not leg1_open else pair_info.symbol1
                    remaining_qty = pos_by_symbol.get(remaining_leg, 0)
                    
                    msg = f"⚠️ <b>External Close Detected:</b>\n"
                    msg += f"Pair: {pair_info.symbol1}-{pair_info.symbol2}\n"
                    msg += f"❌ Closed: {closed_leg}\n"
                    msg += f"🔄 Closing: {remaining_leg} ({abs(remaining_qty):.4f})"
                    print(msg.replace('<b>', '').replace('</b>', ''))
                    await self._notify(msg)
                    
                    pair_info.is_trading = True
                    await self._execute_trade(pair_info, 0, close_reason='desync')
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
                
                self.active_pairs[pair_set] = PairInfo(
                    symbol1=s1,
                    symbol2=s2,
                    hedge_ratio=1.0,
                    half_life=24.0,
                    db_id=new_pair.id
                )
                print(f"  ✅ Added test pair: {s1}-{s2}")
            except Exception as e:
                print(f"  ⚠️ Error adding test pair {s1}-{s2}: {e}")

    async def initialize_all_symbols_data(self, target_symbols=None, concurrency=20):
        """
        Loads historical data for specified symbols with controlled concurrency.
        Prioritizes active pairs and priority pairs.
        """
        symbols_to_load = target_symbols if target_symbols else list(self.all_symbols.keys())
        print(f"Initializing history for {len(symbols_to_load)} symbols (Concurrency: {concurrency})...")
        start_time = time.time()
        
        # 1. Identify priority symbols
        priority_symbols = set()
        
        # Active pairs
        for pair in self.active_pairs.values():
            priority_symbols.add(pair.symbol1)
            priority_symbols.add(pair.symbol2)
            
        # Priority file
        priority_file_path = getattr(self.config, 'priority_pairs_file', 'market_neutral/best_pairs.json')
        if priority_file_path and not os.path.isabs(priority_file_path):
             priority_file_path = os.path.join(os.getcwd(), priority_file_path)
             
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
        
        # Initialize entry_data for all symbols that have main TF data (for MTF mode)
        if self.entry_timeframe != self.timeframe:
            print(f"📊 Initializing entry TF ({self.entry_timeframe}) data for discovered pairs...")
            symbols_to_init = set()
            for pair_info in self.active_pairs.values():
                symbols_to_init.add(pair_info.symbol1)
                symbols_to_init.add(pair_info.symbol2)
            
            for sym in symbols_to_init:
                if sym not in self.entry_data and sym in self.all_symbols:
                    self.entry_data[sym] = Data(maxlen=100)
                    await self._initialize_entry_history(sym)
            print(f"📊 Entry TF data initialized for {len(symbols_to_init)} symbols.")
        
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

    async def add_kline_entry(self, kline_data):
        """
        Processes kline from ENTRY timeframe (faster signal detection).
        Only checks signals for ALREADY DISCOVERED pairs.
        """
        symbol = kline_data['s']
        
        # Store entry TF data for the symbol
        if symbol not in self.entry_data:
            self.entry_data[symbol] = Data(maxlen=100)  # Smaller buffer for entry TF
            await self._initialize_entry_history(symbol)

        added = self.entry_data[symbol].add_kline(
            kline_data['t'],
            kline_data['o'],
            kline_data['h'],
            kline_data['l'],
            kline_data['c']
        )

        if added:
            # Only check signals (no discovery) for active pairs containing this symbol
            self.loop.create_task(self._check_entry_signals(symbol))

    async def _initialize_entry_history(self, symbol):
        """Loads historical data for entry timeframe."""
        try:
            klines = await self.client.klines(symbol, self.entry_timeframe, limit=100)
            data = self.entry_data[symbol]
            for k in klines:
                data.add_kline(k[0], k[1], k[2], k[3], k[4])
        except Exception as e:
            if symbol in self.entry_data:
                del self.entry_data[symbol]

    async def _check_entry_signals(self, updated_symbol: str):
        """
        Checks entry signals using ENTRY timeframe data.
        Uses main TF cointegration parameters but entry TF prices for faster signals.
        """
        for pair_set, pair_info in list(self.active_pairs.items()):
            if pair_info.is_trading or pair_info.position_status != 0:
                continue
            
            s1, s2 = pair_info.symbol1, pair_info.symbol2
            if updated_symbol not in (s1, s2):
                continue
            
            # Check if we have entry data for both symbols
            if s1 not in self.entry_data or s2 not in self.entry_data:
                continue
            
            data1 = self.entry_data[s1]
            data2 = self.entry_data[s2]
            
            if len(data1.close) < 30 or len(data2.close) < 30:
                continue
            
            # Use main TF hedge ratio but entry TF prices
            log_prices1 = np.log(list(data1.close)[-50:])
            log_prices2 = np.log(list(data2.close)[-50:])
            
            spread = log_prices1 - pair_info.hedge_ratio * log_prices2
            z_score = utils.calculate_z_last(spread)
            
            if z_score is None:
                continue
            
            pair_info.last_z_score = z_score
            
            z_entry = self.config.z_entry if self.config and self.config.z_entry else 2.0
            
            # Check position limits
            if not self.can_open_new_position(s1, s2):
                continue
            
            # Entry signals
            if z_score < -z_entry:
                print(f"🚀 [ENTRY TF] LONG Signal on {s1}-{s2}. Z: {z_score:.2f}. Opening...")
                pair_info.entry_z_score = z_score  # Save Z-score at entry
                pair_info.is_trading = True
                self.loop.create_task(self._execute_trade(pair_info, 1))
            elif z_score > z_entry:
                print(f"🔥 [ENTRY TF] SHORT Signal on {s1}-{s2}. Z: {z_score:.2f}. Opening...")
                pair_info.entry_z_score = z_score  # Save Z-score at entry
                pair_info.is_trading = True
                self.loop.create_task(self._execute_trade(pair_info, -1))

    # Legacy method for backward compatibility (single TF mode)
    async def add_kline(self, kline_data):
        """Legacy method - calls add_kline_main for backward compatibility."""
        await self.add_kline_main(kline_data)

    async def _initialize_history(self, symbol):
        """
        Loads historical data to initialize deques.
        """
        print(f"Initializing history for {symbol}...")
        try:
            klines = await self.client.klines(symbol, self.timeframe, limit=self.max_len)
            data = self.all_data[symbol]
            for k in klines:
                data.add_kline(k[0], k[1], k[2], k[3], k[4])
            print(f"History for {symbol} initialized with {len(data.ts)} candles.")
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
                flag, hedge, hl, pval = utils.calculate_cointegration(log_prices1, log_prices2, p_value_threshold)

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
                            print(f"⚠️ {s1}-{s2} rejected: beta_btc={beta_btc:.3f} >= {beta_threshold} (not market-neutral)")
                            flag = 0  # Mark as not cointegrated
                
                # Store beta for display
                pair_info.beta_btc = beta_btc if not np.isnan(beta_btc) else 0.0

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
                if (flag == 0 or hl > 200) and not is_protected_test_pair:
                    print(f"⚠️ Pair {s1}-{s2} correlation broken (pval: {pval:.4f}, HL: {hl}). Removing...")
                    
                    if pair_info.position_status != 0:
                        warn_msg = f"🚨 <b>Broken Correlation</b> on {s1}-{s2} (Pval: {pval:.3f}). Force Closing Position!"
                        print(warn_msg)
                        self.loop.create_task(self._notify(warn_msg))
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

                # Circuit Breaker Logic
                if pair_info.position_status != 0 and pair_info.entry_price1 > 0 and pair_info.entry_price2 > 0:
                    current_price1 = list(data1.close)[-1]
                    current_price2 = list(data2.close)[-1]
                    
                    side1 = 1 if pair_info.position_status == 1 else -1
                    side2 = -1 if pair_info.position_status == 1 else 1

                    pnl1 = (current_price1 - pair_info.entry_price1) * pair_info.qty1 * side1
                    pnl2 = (current_price2 - pair_info.entry_price2) * pair_info.qty2 * side2
                    total_pnl = pnl1 + pnl2
                    
                    initial_investment = (pair_info.entry_price1 * pair_info.qty1) + (pair_info.entry_price2 * pair_info.qty2)
                    circuit_breaker_pct = getattr(self.config, 'circuit_breaker_pct', 0.20) or 0.20
                    
                    if initial_investment > 0:
                        roi = total_pnl / initial_investment
                        if roi < -circuit_breaker_pct:
                            cb_msg = (f"🚨 <b>CIRCUIT BREAKER TRIGGERED</b> on {s1}-{s2}!\n"
                                      f"Loss: {roi*100:.2f}% ({total_pnl:.2f} USDT). Force Closing...")
                            print(cb_msg)
                            self.loop.create_task(self._notify(cb_msg))
                            pair_info.is_trading = True
                            self.loop.create_task(self._execute_trade(pair_info, 0, close_reason='circuit'))
                            # pair_info.position_status = 0 # Will be set in execute_trade
                            continue
                
                # === BETA DRIFT MONITORING ===
                # Check if open position has become correlated with market
                if pair_info.position_status != 0 and pair_info.beta_btc != 0:
                    beta_alert_threshold = getattr(self.config, 'beta_alert_threshold', 0.15) or 0.15
                    
                    if abs(pair_info.beta_btc) >= beta_alert_threshold:
                        # Calculate current PnL
                        current_price1 = list(data1.close)[-1]
                        current_price2 = list(data2.close)[-1]
                        side1 = 1 if pair_info.position_status == 1 else -1
                        side2 = -1 if pair_info.position_status == 1 else 1
                        pnl1 = (current_price1 - pair_info.entry_price1) * pair_info.qty1 * side1
                        pnl2 = (current_price2 - pair_info.entry_price2) * pair_info.qty2 * side2
                        total_pnl = pnl1 + pnl2
                        
                        if total_pnl > 0:
                            # Positive PnL - auto close
                            beta_msg = (f"⚠️ <b>BETA DRIFT</b> on {s1}-{s2}!\n"
                                        f"Beta: {pair_info.beta_btc:.3f} (threshold: {beta_alert_threshold})\n"
                                        f"PnL: +{total_pnl:.2f} USDT. Auto-closing...")
                            print(beta_msg)
                            self.loop.create_task(self._notify(beta_msg))
                            pair_info.is_trading = True
                            self.loop.create_task(self._execute_trade(pair_info, 0, close_reason='beta_drift'))
                            continue
                        else:
                            # Negative PnL - notify user (but don't close)
                            beta_warn = (f"⚠️ <b>BETA DRIFT WARNING</b> on {s1}-{s2}!\n"
                                         f"Beta: {pair_info.beta_btc:.3f} (threshold: {beta_alert_threshold})\n"
                                         f"PnL: {total_pnl:.2f} USDT. Consider manual close.")
                            print(beta_warn)
                            self.loop.create_task(self._notify(beta_warn))
                            # Don't continue - let position stay open

                z_entry = self.config.z_entry if self.config and self.config.z_entry else 2.0
                z_exit = self.config.z_exit if self.config and self.config.z_exit is not None else 0.0
                z_stop = self.config.z_stop if self.config and self.config.z_stop else 4.0
                
                # Test mode flag
                test_mode = getattr(self.config, 'test_mode', False)
                if test_mode and isinstance(test_mode, str):
                    test_mode = test_mode.lower() in ('true', '1', 'yes')
                
                # Signal logic
                if pair_info.position_status == 0:
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
                    if z_score >= z_exit:
                        print(f"💰 TAKE PROFIT (Long) on {s1}-{s2}. Z: {z_score:.2f} >= {z_exit}. Closing...")
                        pair_info.is_trading = True
                        self.loop.create_task(self._execute_trade(pair_info, 0, close_reason='z_tp'))
                    elif z_score <= -z_stop:
                        print(f"🛑 STOP LOSS (Long) on {s1}-{s2}. Z: {z_score:.2f} <= -{z_stop}. Closing...")
                        pair_info.is_trading = True
                        self.loop.create_task(self._execute_trade(pair_info, 0, close_reason='z_sl'))

                elif pair_info.position_status == -1: # Short spread
                    if z_score <= -z_exit:
                        print(f"💰 TAKE PROFIT (Short) on {s1}-{s2}. Z: {z_score:.2f} <= {-z_exit}. Closing...")
                        pair_info.is_trading = True
                        self.loop.create_task(self._execute_trade(pair_info, 0, close_reason='z_tp'))
                    elif z_score >= z_stop:
                        print(f"🛑 STOP LOSS (Short) on {s1}-{s2}. Z: {z_score:.2f} >= {z_stop}. Closing...")
                        pair_info.is_trading = True
                        self.loop.create_task(self._execute_trade(pair_info, 0, close_reason='z_sl'))

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
        priority_file_path = getattr(self.config, 'priority_pairs_file', 'market_neutral/best_pairs.json')
        # Handle path resolution
        if priority_file_path and not os.path.isabs(priority_file_path):
             priority_file_path = os.path.join(os.getcwd(), priority_file_path)

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
                getattr(self.config, 'hl_max_days', 5.0) or 5.0
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
                    print(f"✅ FOUND: {s1}-{s2} | HL: {hl:.2f}, P: {pval:.4f}")
                    
                    self.active_pairs[pair_set] = PairInfo(
                        symbol1=s1, 
                        symbol2=s2, 
                        hedge_ratio=hedge, 
                        half_life=hl,
                        db_id=new_pair.id
                    )
                    new_pairs_count += 1
                    
                    # IMMEDIATE ENTRY CHECK: Don't wait for 5m candle!
                    # Check signals right now using the data we just downloaded
                    if self.can_open_new_position(s1, s2):
                        print(f"⚡ Checking immediate entry for found pair {s1}-{s2}...")
                        self.loop.create_task(self._check_signals_for_active_pairs(s1))
                        
                except Exception as e:
                    print(f"Error adding pair {s1}-{s2}: {e}")

        elapsed = time.time() - start_time
        print(f"Discovery process finished in {elapsed:.2f}s. Found {new_pairs_count} new pairs.")

    async def _notify(self, message, reply_to_msg_id=None):
        """Sends a notification via the configured callback. Returns msg_id for reply threading."""
        if self.notify_callback:
            try:
                return await self.notify_callback(message, reply_to_msg_id)
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
        """Sets leverage for the symbol if not already set."""
        if not leverage or leverage < 1:
            return
        if self.leverage_cache.get(symbol) == leverage:
            return
        try:
            print(f"⚖️ Setting leverage {leverage}x for {symbol}...")
            await self.client.change_leverage(symbol=symbol, leverage=leverage)
            self.leverage_cache[symbol] = leverage
        except Exception as e:
            print(f"⚠️ Failed to set leverage for {symbol}: {e}")

    def _add_to_best_pairs(self, symbol1: str, symbol2: str):
        """
        Add a successfully traded pair to best_pairs.json for priority loading.
        Only adds if the pair doesn't already exist.
        """
        try:
            priority_file_path = getattr(self.config, 'priority_pairs_file', 'market_neutral/best_pairs.json')
            if priority_file_path and not os.path.isabs(priority_file_path):
                priority_file_path = os.path.join(os.getcwd(), priority_file_path)
            
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
            
            s1, s2 = pair_info.symbol1, pair_info.symbol2
            if s1 in self.all_data and s2 in self.all_data:
                await self._check_signals_for_active_pairs(s1)
                analyzed += 1
                
                # Check if we filled all slots
                if self.count_active_positions() >= max_pairs:
                    break
        
        print(f"🔍 Immediate analysis complete. Checked {analyzed} pairs.")

    def is_symbol_locked(self, symbol: str) -> bool:
        """Check if symbol is already in an active position (in any pair)."""
        for pair_info in self.active_pairs.values():
            if pair_info.position_status != 0:
                if symbol in (pair_info.symbol1, pair_info.symbol2):
                    return True
        return False

    def count_active_positions(self) -> int:
        """Count the number of currently open pairs."""
        count = 0
        for pair_info in self.active_pairs.values():
            if pair_info.position_status != 0:
                count += 1
        return count

    def can_open_new_position(self, s1: str, s2: str) -> bool:
        """Check if we can open a new position for this pair."""
        # Check if trading is enabled
        trade_mode = getattr(self.config, 'trade_mode', True)
        if trade_mode is not None and str(trade_mode).lower() in ('false', '0', 'no'):
            return False
        
        # Check max active pairs limit
        max_pairs = getattr(self.config, 'max_active_pairs', 5) or 5
        if self.count_active_positions() >= max_pairs:
            return False
        
        # Check symbol lock - each symbol can only be in one active pair
        if self.is_symbol_locked(s1) or self.is_symbol_locked(s2):
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
                if not self.can_open_new_position(s1, s2):
                    print(f"🚫 Trade blocked by lock: {s1}-{s2} (limit reached or symbol locked)")
                    pair_info.is_trading = False
                    return
                
                # Mark as opening INSIDE lock before releasing
                pair_info.position_status = direction  # Tentatively set to prevent other trades
            
            # Now proceed with actual execution (lock released for API calls)
            await self._set_leverage(s1, leverage)
            await self._set_leverage(s2, leverage)
        
        try:
            if direction == 0:
                if pair_info.position_status == 0:
                    return

                print(f"EXECUTING CLOSE for {s1}-{s2}")
                
                # Cancel all open orders (including algo SL/TP) before closing
                try:
                    results = await asyncio.gather(
                        self.client.cancel_open_orders(symbol=s1),
                        self.client.cancel_open_orders(symbol=s2),
                        return_exceptions=True
                    )
                    # Log any errors
                    for i, res in enumerate(results):
                        if isinstance(res, Exception):
                            print(f"⚠️ Cancel orders error for {[s1, s2][i]}: {res}")
                        else:
                            print(f"🗑️ Cancelled orders for {[s1, s2][i]}")
                except Exception as e:
                    print(f"⚠️ Could not cancel orders: {e}")
                
                side1_close = 'SELL' if pair_info.position_status == 1 else 'BUY'
                side2_close = 'BUY' if pair_info.position_status == 1 else 'SELL'
                qty1_close = pair_info.qty1
                qty2_close = pair_info.qty2

                try:
                    # Check which legs still have open positions
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
                        }
                        reason_text = CLOSE_REASONS.get(close_reason, '❓ Unknown') if close_reason else '❓ Unknown'
                    
                        def get_price(order):
                            if 'avgPrice' in order and float(order['avgPrice']) > 0:
                                return float(order['avgPrice'])
                            if 'cummulativeQuoteQty' in order and 'executedQty' in order and float(order['executedQty']) > 0:
                                return float(order['cummulativeQuoteQty']) / float(order['executedQty'])
                            return 0.0

                        # Safely get prices (results may be smaller if some legs already closed)
                        close_price1 = get_price(results[0]) if len(results) > 0 and not isinstance(results[0], Exception) else pair_info.entry_price1
                        close_price2 = get_price(results[1]) if len(results) > 1 and not isinstance(results[1], Exception) else pair_info.entry_price2
                    
                        side1_dir = 1 if pair_info.position_status == 1 else -1
                        side2_dir = -1 if pair_info.position_status == 1 else 1
                        pnl1 = (close_price1 - pair_info.entry_price1) * pair_info.qty1 * side1_dir
                        pnl2 = (close_price2 - pair_info.entry_price2) * pair_info.qty2 * side2_dir
                        total_pnl = pnl1 + pnl2
                    
                        pnl_emoji = "🟢" if total_pnl > 0 else "🔴"


                        if pair_info.current_trade_id:
                            await db.update_trade_fields(
                                pair_info.current_trade_id,
                                status='CLOSED',
                                close_time=int(time.time() * 1000),
                                close_price_1=close_price1,
                                close_price_2=close_price2,
                                pnl=total_pnl,
                            )
                    
                        pair_info.current_trade_id = None
                        pair_info.position_status = 0
                        pair_info.qty1 = 0
                        pair_info.qty2 = 0
                        pair_info.entry_price1 = 0
                        pair_info.entry_price2 = 0
                    
                        # Cancel all algo orders for this pair - track per symbol/type
                        cleanup_status = []
                        try:
                            algo_orders = await self.client.get_algo_orders()
                            
                            # Track which orders exist for each symbol
                            orders_by_sym = {s1: [], s2: []}
                            for o in algo_orders:
                                sym = o['symbol']
                                if sym in orders_by_sym:
                                    order_type = o.get('type', '')
                                    if 'STOP' in order_type.upper():
                                        orders_by_sym[sym].append(('SL', o['algoId']))
                                    else:
                                        orders_by_sym[sym].append(('TP', o['algoId']))
                            
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
                        
                        # Build enhanced close message
                        cleanup_msg = "\n".join(cleanup_status) if cleanup_status else "  ℹ️ No cleanup needed"
                        full_msg = f"✅ Position Closed: {s1}/{s2}\n"
                        full_msg += f"📊 Reason: {reason_text}\n\n"
                        full_msg += f"🛡️ Order Cleanup:\n{cleanup_msg}\n\n"
                        full_msg += f"💵 PnL: {pnl_emoji} <b>{total_pnl:.2f} USDT</b>"
                        
                        print(full_msg.replace('<b>', '').replace('</b>', ''))
                        # Reply to original open message if available
                        reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                        await self._notify(full_msg, reply_to)
                        
                        # Update DB with close details
                        if pair_info.db_id:
                            import time
                            await db.update_pair({
                                'id': pair_info.db_id,
                                'position_status': 0,
                                'qty1': 0,
                                'qty2': 0,
                                'entry_price1': 0,
                                'entry_price2': 0,
                                'close_time': int(time.time()),
                                'close_pnl': total_pnl,
                                'close_reason': close_reason or 'unknown'
                            })

                        
                        # AUTO-ADD to best_pairs.json on successful TP
                        if close_reason in ('z_tp', 'hardware_tp') or total_pnl > 0:
                            self._add_to_best_pairs(s1, s2)
                        
                        # IMMEDIATE RE-ANALYSIS: Trigger search for new trades now that slot is free
                        print(f"🔄 Slot freed after closing {s1}-{s2}. Triggering immediate re-analysis...")
                        self.loop.create_task(self._trigger_immediate_analysis())
                        
                except Exception as e:
                    print(f"FATAL ERROR closing position for {s1}-{s2}: {e}")
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

            log_prices1 = np.log(list(data1.close)[-COINT_WINDOW:])
            log_prices2 = np.log(list(data2.close)[-COINT_WINDOW:])

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

            print(f"EXECUTING TRADE for {s1}-{s2}:")
            print(f"  {side1} {qty1_rounded} {s1} at {s1_price}")
            print(f"  {side2} {qty2_rounded} {s2} at {s2_price}")

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
                else:
                    pair_info.position_status = direction
                    pair_info.qty1 = float(executed_orders[0]['executedQty'])
                    pair_info.qty2 = float(executed_orders[1]['executedQty'])
                
                    def get_price(order):
                        if 'avgPrice' in order and float(order['avgPrice']) > 0:
                            return float(order['avgPrice'])
                        if 'cummulativeQuoteQty' in order and 'executedQty' in order and float(order['executedQty']) > 0:
                            return float(order['cummulativeQuoteQty']) / float(order['executedQty'])
                        return 0.0

                    pair_info.entry_price1 = get_price(executed_orders[0])
                    pair_info.entry_price2 = get_price(executed_orders[1])
                    
                    # Set open time
                    import time
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
                                   f"📊 Beta: {pair_info.beta_btc:.3f}\n"
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
                            print(f"⚠️ WARN: Invalid SL/TP prices (sl1={sl1}, sl2={sl2}, tp1={tp1}, tp2={tp2}). ATR may be missing.")
                            print(f"  Entry prices: {pair_info.entry_price1}, {pair_info.entry_price2}")
                            print(f"  ATR values: {atr1}, {atr2}")
                            # Skip placing protection but don't force close - allow software stops
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
                        elif failed_count > 0:
                            warn_msg = f"⚠️ CRITICAL: Protection partially FAILED for {s1}-{s2} ({failed_count}/4 failed). Force closing!"
                            print(warn_msg)
                            await self._notify(warn_msg)
                            
                            # Cancel successfully placed orders using algoId from results
                            if successful_algo_ids:
                                try:
                                    cancel_tasks = [self.client.cancel_algo_order(algoId=aid) for aid in successful_algo_ids]
                                    await asyncio.gather(*cancel_tasks, return_exceptions=True)
                                    print(f"🗑️ Cancelled {len(successful_algo_ids)} partial algo orders")
                                except Exception as ce:
                                    print(f"⚠️ Could not cancel partial orders: {ce}")
                            
                            # Force close position
                            pair_info.is_trading = True
                            self.loop.create_task(self._execute_trade(pair_info, 0, close_reason='hardware_sl'))
                            
                    except Exception as e:
                        warn_msg = f"⚠️ CRITICAL ERROR placing hardware SL for {s1}-{s2}: {e}. Force closing position!"
                        print(warn_msg)
                        await self._notify(warn_msg)
                        
                        # Force close position (algo orders will be cancelled by _execute_trade)
                        pair_info.is_trading = True
                        self.loop.create_task(self._execute_trade(pair_info, 0, close_reason='hardware_sl'))
                    # === END HARDWARE SL/TP ===
                
                    if pair_info.db_id:
                        # Await DB update for safety
                        try:
                            await db.update_pair({
                                'id': pair_info.db_id,
                                'position_status': pair_info.position_status,
                                'qty1': pair_info.qty1,
                                'qty2': pair_info.qty2,
                                'entry_price1': pair_info.entry_price1,
                                'entry_price2': pair_info.entry_price2
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
                            pnl=0.0
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
        """Returns all PairInfo objects containing the given symbol."""
        result = []
        for pair_set, pair_info in self.active_pairs.items():
            if symbol in pair_set:
                result.append(pair_info)
        return result
    
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
        Called when a price update is received from WebSocket ticker.
        Updates last_prices and checks for entry signals with confirmation.
        """
        self.last_prices[symbol] = price
        
        # Check all pairs containing this symbol
        for pair_info in self._pairs_with_symbol(symbol):
            if pair_info.position_status != 0 or pair_info.is_trading:
                continue  # Already in position or trading
            
            # Get both prices
            price1 = self.last_prices.get(pair_info.symbol1)
            price2 = self.last_prices.get(pair_info.symbol2)
            if not price1 or not price2:
                continue
            
            # Calculate real-time Z-score
            z_score = self._calc_realtime_zscore(pair_info, price1, price2)
            if np.isnan(z_score):
                continue
            
            z_entry = getattr(self.config, 'z_entry', 2.0) or 2.0
            z_entry_max = getattr(self.config, 'z_entry_max', 3.0) or 3.0
            
            # Check if signal (between z_entry and z_entry_max)
            # Reject if already too extreme - spread may be broken
            if abs(z_score) >= z_entry and abs(z_score) < z_entry_max:
                if pair_info.pending_signal is None:
                    # Start confirmation timer
                    pair_info.pending_signal = z_score
                    pair_info.pending_since = time.time()
                    print(f"🔔 Signal detected for {pair_info.symbol1}-{pair_info.symbol2}: Z={z_score:.2f}. Confirming...")
            else:
                # Signal went away - reset
                if pair_info.pending_signal is not None:
                    pair_info.pending_signal = None
                    pair_info.pending_since = None
    
    async def _signal_confirmation_loop(self):
        """
        Periodically checks for confirmed signals (held for N seconds).
        """
        confirm_sec = getattr(self.config, 'signal_confirm_sec', 10) or 10
        
        while True:
            await asyncio.sleep(1)  # Check every second
            
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
                            z_entry = getattr(self.config, 'z_entry', 2.0) or 2.0
                            z_entry_max = getattr(self.config, 'z_entry_max', 3.0) or 3.0
                            
                            # Check signal still valid and in same direction
                            # Also reject if Z-score exceeds entry window (spread may be broken)
                            if abs(current_z) >= z_entry and abs(current_z) < z_entry_max and (current_z * pair_info.pending_signal > 0):
                                # Check can open
                                if self.can_open_new_position(pair_info.symbol1, pair_info.symbol2):
                                    direction = 1 if current_z < 0 else -1
                                    pair_info.entry_z_score = current_z
                                    print(f"✅ Signal CONFIRMED for {pair_info.symbol1}-{pair_info.symbol2}: Z={current_z:.2f}. Opening position...")
                                    pair_info.is_trading = True
                                    self.loop.create_task(self._execute_trade(pair_info, direction))
                            elif abs(current_z) >= z_entry_max:
                                print(f"⚠️ {pair_info.symbol1}-{pair_info.symbol2}: Z={current_z:.2f} exceeds z_entry_max={z_entry_max}. Skipping entry (spread may be broken).")
                        
                        # Reset pending after check
                        pair_info.pending_signal = None
                        pair_info.pending_since = None
    
    def start_realtime_monitoring(self):
        """Start the signal confirmation loop."""
        if self._signal_confirmation_task is None:
            self._signal_confirmation_task = self.loop.create_task(self._signal_confirmation_loop())
            print("🔄 Started real-time signal confirmation loop")