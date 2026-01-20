from collections import deque
import numpy as np
from dataclasses import dataclass, field
import utils
import asyncio
import itertools
import time
import traceback
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

class PairsManager:
    """
    Manages symbol data, finds cointegrated pairs, and generates signals.
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
        
        self.all_data: dict[str, Data] = {}
        self.active_pairs: dict[frozenset, PairInfo] = {}
        self.leverage_cache = {} # {symbol: leverage_int}
        self._discovery_task = None
        self._last_discovery_time = 0
        
        # CPU Pool for heavy computations
        self.executor = ProcessPoolExecutor(max_workers=None)
        
        # Restore state from DB on startup
        self.loop.create_task(self._load_state_from_db())

    async def _load_state_from_db(self):
        # #region agent log
        import os
        import json
        import time
        log_path = r"c:\Users\Dmitrii\Trading strategies\Market_neutral_strategy\.cursor\debug.log"
        def log_instrument(location, message, data=None):
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    entry = {
                        "id": f"log_{int(time.time()*1000)}_db",
                        "timestamp": int(time.time()*1000),
                        "location": location,
                        "message": message,
                        "data": data or {},
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "DB_STATE_2"
                    }
                    f.write(json.dumps(entry) + '\n')
            except: pass
        # #endregion

        log_instrument("pairs_trading.py:_load_state_from_db", "Starting state restoration from DB")
        print("Restoring state from DB...")
        try:
            pairs = await db.get_all_pairs()
            log_instrument("pairs_trading.py:_load_state_from_db", "Retrieved pairs from DB", {"pairs_count": len(pairs)})

            for p in pairs:
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
                    db_id=p.id
                )

                if p.position_status != 0:
                    last_trade = await db.get_last_open_trade_for_pair(p.id)
                    if last_trade:
                        info.current_trade_id = last_trade.id
                        log_instrument("pairs_trading.py:_load_state_from_db", f"Attached open trade for pair {p.id}", {"trade_id": last_trade.id})
                        print(f"  Attached open trade ID: {last_trade.id}")
                    else:
                        log_instrument("pairs_trading.py:_load_state_from_db", f"Position active but no trade found for pair {p.id}", {"pair_id": p.id, "status": p.position_status})
                        print(f"  WARN: Position active but no open trade found in DB for pair {p.id}")

                self.active_pairs[pair_set] = info
                log_instrument("pairs_trading.py:_load_state_from_db", f"Restored pair {p.symbol1}/{p.symbol2}", {"status": p.position_status, "db_id": p.id})
                print(f"Restored pair {p.symbol1}/{p.symbol2} (Status: {p.position_status}, ID: {p.id})")
        except Exception as e:
            log_instrument("pairs_trading.py:_load_state_from_db", "State restoration failed", {"error": str(e), "error_type": type(e).__name__})
            print(f"Error loading state from DB: {e}")

    async def add_kline(self, kline_data):
        """
        Processes a new kline from websocket.
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
            # #region agent log
            import os
            import json
            import time
            log_path = r"c:\Users\Dmitrii\Trading strategies\Market_neutral_strategy\.cursor\debug.log"
            def log_instrument(location, message, data=None):
                try:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        entry = {
                            "id": f"log_{int(time.time()*1000)}_async",
                            "timestamp": int(time.time()*1000),
                            "location": location,
                            "message": message,
                            "data": data or {},
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "ASYNC_TASK_5"
                        }
                        f.write(json.dumps(entry) + '\n')
                except: pass
            # #endregion

            log_instrument("pairs_trading.py:add_kline", f"Creating analysis task for {symbol}", {"data_points": len(self.all_data[symbol].close)})
            self.loop.create_task(self.run_analysis(symbol))

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

                # Dynamic recalculation of cointegration
                flag, hedge, hl, pval = utils.calculate_cointegration(log_prices1, log_prices2)

                # Pair rotation: if cointegration breaks
                if flag == 0 or hl > 200:
                    print(f"⚠️ Pair {s1}-{s2} correlation broken (pval: {pval:.4f}, HL: {hl}). Removing...")
                    
                    if pair_info.position_status != 0:
                        warn_msg = f"🚨 <b>Broken Correlation</b> on {s1}-{s2} (Pval: {pval:.3f}). Force Closing Position!"
                        print(warn_msg)
                        self.loop.create_task(self._notify(warn_msg))
                        pair_info.is_trading = True
                        self.loop.create_task(self._execute_trade(pair_info, 0))
                    
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
                        self.loop.create_task(db.add_pair_history(history_item))
                        await db.delete_pair(pair_info.db_id)
                    
                    if pair_set in self.active_pairs:
                        del self.active_pairs[pair_set]
                    continue

                # Update parameters
                pair_info.hedge_ratio = hedge
                pair_info.half_life = hl
                
                if pair_info.db_id:
                    self.loop.create_task(db.update_pair({
                        'id': pair_info.db_id,
                        'hedge_ratio': hedge,
                        'half_life': hl
                    }))

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
                    HARD_STOP_PCT = 0.20 
                    
                    if initial_investment > 0:
                        roi = total_pnl / initial_investment
                        if roi < -HARD_STOP_PCT:
                            cb_msg = (f"🚨 <b>CIRCUIT BREAKER TRIGGERED</b> on {s1}-{s2}!\n"
                                      f"Loss: {roi*100:.2f}% ({total_pnl:.2f} USDT). Force Closing...")
                            print(cb_msg)
                            self.loop.create_task(self._notify(cb_msg))
                            pair_info.is_trading = True
                            self.loop.create_task(self._execute_trade(pair_info, 0))
                            # pair_info.position_status = 0 # Will be set in execute_trade
                            continue

                z_entry = self.config.z_entry if self.config and self.config.z_entry else 2.0
                z_exit = self.config.z_exit if self.config and self.config.z_exit is not None else 0.0
                z_stop = self.config.z_stop if self.config and self.config.z_stop else 4.0
                
                # Signal logic
                if pair_info.position_status == 0:
                    if z_score < -z_entry:
                        print(f"🚀 LONG Signal on {s1}-{s2} spread. Z: {z_score:.2f}. Opening...")
                        pair_info.is_trading = True
                        self.loop.create_task(self._execute_trade(pair_info, 1))
                    elif z_score > z_entry:
                        print(f"🔥 SHORT Signal on {s1}-{s2} spread. Z: {z_score:.2f}. Opening...")
                        pair_info.is_trading = True
                        self.loop.create_task(self._execute_trade(pair_info, -1))
                
                elif pair_info.position_status == 1: # Long spread
                    if z_score >= z_exit:
                        print(f"💰 TAKE PROFIT (Long) on {s1}-{s2}. Z: {z_score:.2f} >= {z_exit}. Closing...")
                        pair_info.is_trading = True
                        self.loop.create_task(self._execute_trade(pair_info, 0))
                    elif z_score <= -z_stop:
                        print(f"🛑 STOP LOSS (Long) on {s1}-{s2}. Z: {z_score:.2f} <= -{z_stop}. Closing...")
                        pair_info.is_trading = True
                        self.loop.create_task(self._execute_trade(pair_info, 0))

                elif pair_info.position_status == -1: # Short spread
                    if z_score <= -z_exit:
                        print(f"💰 TAKE PROFIT (Short) on {s1}-{s2}. Z: {z_score:.2f} <= {-z_exit}. Closing...")
                        pair_info.is_trading = True
                        self.loop.create_task(self._execute_trade(pair_info, 0))
                    elif z_score >= z_stop:
                        print(f"🛑 STOP LOSS (Short) on {s1}-{s2}. Z: {z_score:.2f} >= {z_stop}. Closing...")
                        pair_info.is_trading = True
                        self.loop.create_task(self._execute_trade(pair_info, 0))

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

        all_combinations = list(itertools.combinations(ready_symbols, 2))
        candidates = [pair for pair in all_combinations if frozenset(pair) not in self.active_pairs]
        total_pairs = len(candidates)
        print(f"Total pairs to check: {total_pairs}")
        
        if total_pairs == 0:
            return

        CHUNK_SIZE = 5000
        chunks = [candidates[i:i + CHUNK_SIZE] for i in range(0, total_pairs, CHUNK_SIZE)]
        print(f"Split into {len(chunks)} chunks for parallel processing.")
        
        tasks = []
        for chunk in chunks:
            task = self.loop.run_in_executor(
                self.executor, 
                utils.batch_process_pairs, 
                chunk, 
                data_snapshot, 
                self.min_data_points
            )
            tasks.append(task)
        
        results_list = await asyncio.gather(*tasks)
        
        new_pairs_count = 0
        for batch_results in results_list:
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
                    self.loop.create_task(db.add_pair_history(history_item))
                    
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
                except Exception as e:
                    print(f"Error adding pair {s1}-{s2}: {e}")

        elapsed = time.time() - start_time
        print(f"Discovery process finished in {elapsed:.2f}s. Found {new_pairs_count} new pairs.")

    async def _notify(self, message):
        """Sends a notification via the configured callback."""
        if self.notify_callback:
            try:
                await self.notify_callback(message)
            except Exception as e:
                print(f"Error in _notify: {e}")

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

    async def _execute_trade(self, pair_info: PairInfo, direction: int):
        """
        Executes a trade order.
        direction: 1 for long spread, -1 for short spread, 0 for close.
        """
        s1 = pair_info.symbol1
        s2 = pair_info.symbol2
        leverage = self.config.leverage if self.config and self.config.leverage else 20

        if direction != 0:
            await self._set_leverage(s1, leverage)
            await self._set_leverage(s2, leverage)
        
        try:
            # #region agent log
            import os
            import json
            import time
            log_path = r"c:\Users\Dmitrii\Trading strategies\Market_neutral_strategy\.cursor\debug.log"
            def log_instrument(location, message, data=None):
                try:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        entry = {
                            "id": f"log_{int(time.time()*1000)}_order",
                            "timestamp": int(time.time()*1000),
                            "location": location,
                            "message": message,
                            "data": data or {},
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "ORDER_EXEC_4"
                        }
                        f.write(json.dumps(entry) + '\n')
                except: pass
            # #endregion

            if direction == 0:
                if pair_info.position_status == 0:
                    return

                log_instrument("pairs_trading.py:_execute_trade", f"Starting position close for {s1}-{s2}", {"status": pair_info.position_status, "qty1": pair_info.qty1, "qty2": pair_info.qty2})
                print(f"EXECUTING CLOSE for {s1}-{s2}")
                side1_close = 'SELL' if pair_info.position_status == 1 else 'BUY'
                side2_close = 'BUY' if pair_info.position_status == 1 else 'SELL'
                qty1_close = pair_info.qty1
                qty2_close = pair_info.qty2

                try:
                    log_instrument("pairs_trading.py:_execute_trade", "Submitting close orders", {"s1_side": side1_close, "s2_side": side2_close})
                    task1 = self.loop.create_task(
                        self.client.new_order(symbol=s1, side=side1_close, type='MARKET', quantity=qty1_close, newOrderRespType='RESULT')
                    )
                    task2 = self.loop.create_task(
                        self.client.new_order(symbol=s2, side=side2_close, type='MARKET', quantity=qty2_close, newOrderRespType='RESULT')
                    )
                    results = await asyncio.gather(task1, task2, return_exceptions=True)
                    log_instrument("pairs_trading.py:_execute_trade", "Close orders completed", {"results_count": len(results)})
                
                    if any(isinstance(res, Exception) for res in results):
                        err_msg = f"❌ ERROR closing position for {s1}-{s2}. Manual intervention required. Errors: {results}"
                        print(err_msg)
                        await self._notify(err_msg)
                    else:
                        msg = f"✅ SUCCESS: Position closed for {s1}-{s2}"
                    
                        def get_price(order):
                            if 'avgPrice' in order and float(order['avgPrice']) > 0:
                                return float(order['avgPrice'])
                            if 'cummulativeQuoteQty' in order and 'executedQty' in order and float(order['executedQty']) > 0:
                                return float(order['cummulativeQuoteQty']) / float(order['executedQty'])
                            return 0.0

                        close_price1 = get_price(results[0])
                        close_price2 = get_price(results[1])
                    
                        side1_dir = 1 if pair_info.position_status == 1 else -1
                        side2_dir = -1 if pair_info.position_status == 1 else 1
                        pnl1 = (close_price1 - pair_info.entry_price1) * pair_info.qty1 * side1_dir
                        pnl2 = (close_price2 - pair_info.entry_price2) * pair_info.qty2 * side2_dir
                        total_pnl = pnl1 + pnl2
                    
                        pnl_emoji = "🟢" if total_pnl > 0 else "🔴"
                        msg += f"\n  Realized PnL: {pnl_emoji} <b>{total_pnl:.2f} USDT</b>"
                        print(msg)
                        await self._notify(msg)

                        if pair_info.current_trade_id:
                            self.loop.create_task(db.update_trade_fields(
                                pair_info.current_trade_id, 
                                status='CLOSED',
                                close_time=int(time.time() * 1000),
                                close_price_1=close_price1,
                                close_price_2=close_price2,
                                pnl=total_pnl,
                            ))
                    
                        pair_info.current_trade_id = None
                        pair_info.position_status = 0
                        pair_info.qty1 = 0
                        pair_info.qty2 = 0
                        pair_info.entry_price1 = 0
                        pair_info.entry_price2 = 0
                    
                        if pair_info.db_id:
                            self.loop.create_task(db.update_pair({
                                'id': pair_info.db_id,
                                'position_status': 0,
                                'qty1': 0,
                                'qty2': 0,
                                'entry_price1': 0,
                                'entry_price2': 0
                            }))
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
            if qty1_rounded * s1_price < min_notional1:
                print(f"WARN: {s1} qty {qty1_rounded} below min notional {min_notional1}. Bumping up...")
                qty1_rounded = utils.round_up(min_notional1 / s1_price, s1_info.step_size)
        
            min_notional2 = s2_info.notional * 1.1
            if qty2_rounded * s2_price < min_notional2:
                 print(f"WARN: {s2} qty {qty2_rounded} below min notional {min_notional2}. Bumping up...")
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
                                self.client.new_order(symbol=exec_symbol, side=revert_side, type='MARKET', quantity=exec_qty)
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

                    success_msg = (f"🚀 <b>Trade OPENED:</b> {s1}-{s2}\n"
                                   f"Direction: {'LONG' if direction == 1 else 'SHORT'} Spread\n"
                                   f"Entry 1: {pair_info.qty1} {s1} @ {pair_info.entry_price1}\n"
                                   f"Entry 2: {pair_info.qty2} {s2} @ {pair_info.entry_price2}")
                    print(success_msg)
                    await self._notify(success_msg)
                
                    if pair_info.db_id:
                        log_instrument("pairs_trading.py:_execute_trade", "Updating pair in DB", {"db_id": pair_info.db_id, "status": pair_info.position_status})
                        self.loop.create_task(db.update_pair({
                            'id': pair_info.db_id,
                            'position_status': pair_info.position_status,
                            'qty1': pair_info.qty1,
                            'qty2': pair_info.qty2,
                            'entry_price1': pair_info.entry_price1,
                            'entry_price2': pair_info.entry_price2
                        }))

                    try:
                        log_instrument("pairs_trading.py:_execute_trade", "Creating trade record in DB", {"pair_id": pair_info.db_id})
                        trade = db.Trades(
                            pair_id=pair_info.db_id,
                            symbol1=s1,
                            symbol2=s2,
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
                        log_instrument("pairs_trading.py:_execute_trade", "Trade record created", {"trade_id": pair_info.current_trade_id})
                    except Exception as e:
                        log_instrument("pairs_trading.py:_execute_trade", "Trade record creation failed", {"error": str(e), "error_type": type(e).__name__})
                        print(f"Error creating trade record: {e}")
            
            except Exception as e:
                print(f"FATAL ERROR during trade execution for {s1}-{s2}: {e}")
                traceback.print_exc()
                pair_info.position_status = 0
        finally:
            pair_info.is_trading = False
