import asyncio
import configparser
import traceback
import os
from dotenv import load_dotenv
import binance
import pairs_trading
import db
import tg


client: binance.Futures

all_symbols: dict[str, binance.SymbolFutures] = {}
positions = {}
websockets_list: list[binance.futures.WebsocketAsync] = []
userdata_ws: binance.futures.WebsocketAsync
pairs_manager: pairs_trading.PairsManager

# TG notification globals
tg_channel_global = ''
tg_admins_global = ''

async def send_tg_notification(message, reply_to_message_id=None, reply_markup=None):
    """Send notification to TG channel or admins. Returns message_id for reply threading."""
    if not tg.bot:
        print("⚠️ TG: bot not initialized")
        return None
    
    msg_id = None
    # Priority: send to channel if configured, otherwise to admins
    if tg_channel_global:
        try:
            sent = await tg.bot.send_message(
                tg_channel_global, message, parse_mode='HTML',
                reply_to_message_id=reply_to_message_id,
                reply_markup=reply_markup
            )
            msg_id = sent.message_id
            print(f"📨 TG sent to channel, msg_id={msg_id}, reply_to={reply_to_message_id}")
        except Exception as e:
            print(f"Error sending TG to channel: {e}")
    elif tg_admins_global:
        admins = [int(admin_id) for admin_id in tg_admins_global.split(',') if admin_id.strip()]
        for admin_id in admins:
            try:
                sent = await tg.bot.send_message(
                    admin_id, message, parse_mode='HTML',
                    reply_to_message_id=reply_to_message_id,
                    reply_markup=reply_markup
                )
                if msg_id is None:
                    msg_id = sent.message_id
                print(f"📨 TG sent to {admin_id}, msg_id={msg_id}, reply_to={reply_to_message_id}")
            except Exception as e:
                print(f"Error sending TG to {admin_id}: {e}")
    else:
        print("⚠️ TG: no channel or admins configured")
    
    return msg_id


async def main():
    global client
    global pairs_manager
    global all_symbols
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Connect to DB
    ini_config = configparser.ConfigParser()
    ini_config.read('market_neutral/config.ini')
    
    session = await db.connect(
        host=ini_config['DB']['host'],
        port=ini_config['DB']['port'],
        user=ini_config['DB']['user'],
        password=ini_config['DB']['password'],
        db_name=ini_config['DB']['db_name']
    )

    # Load config from DB
    conf = await db.load_config()

    # Load secrets from environment variables (NOT from config.ini or DB)
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    tg_token = os.getenv('TG_TOKEN')
    tg_admins = os.getenv('TG_ADMINS', '')
    
    if not api_key or not api_secret:
        print("ERROR: BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env file!")
        return
    
    if not tg_token:
        print("WARNING: TG_TOKEN not set in .env file. Telegram bot will not work.")

    # Initialize global TG notification settings
    global tg_channel_global, tg_admins_global
    tg_channel_global = os.getenv('TG_CHANNEL', '').strip()
    tg_admins_global = tg_admins

    # Create Binance client
    client = binance.Futures(api_key=api_key,
                             secret_key=api_secret,
                             asynced=True,
                             testnet=ini_config.getboolean('BOT', 'testnet'))
    
    # CRITICAL: Sync time with server
    # Note: Original library does not support sync_time. Ensure system clock is accurate.
    # await client.sync_time()
    
    # Init pairs manager
    loop = asyncio.get_running_loop()
    
    # Default values
    timeframe = conf.timeframe if conf.timeframe else '1h'
    
    # Robust window_size logic:
    # 1. Check if user provided a valid override
    use_manual_window = False
    window_size_val = 0
    
    if conf.window_size:
        raw_val = str(conf.window_size).strip().lower()
        if raw_val not in ('', 'none', 'null', '0', 'default', 'auto'):
            try:
                window_size_val = int(raw_val)
                if window_size_val > 0:
                    use_manual_window = True
            except ValueError:
                print(f"⚠️ Invalid window_size '{conf.window_size}' in config. Using auto-calculation.")

    if use_manual_window:
        window_size = window_size_val
        print(f"⚙️ Using manual window_size: {window_size}")
    else:
        # Auto-selection of window size based on timeframe
        if timeframe == '1m':
            window_size = 720  # 12 hours
        elif timeframe == '5m':
            window_size = 576  # 2 days
        elif timeframe == '15m':
            window_size = 480  # 5 days
        elif timeframe == '1h':
            window_size = 336  # 2 weeks
        elif timeframe == '4h':
            window_size = 180  # 30 days
        elif timeframe == '1d':
            window_size = 90   # 90 days
        else:
            window_size = 336  # Default as for 1h
        print(f"⚙️ Auto-calculated window_size: {window_size} (for {timeframe})")
    
    # Single TF Mode: entry timeframe = main timeframe
    # This ensures Z-score consistency and avoids conflicting signals
    entry_timeframe = timeframe  # Changed from separate entry TF to single TF mode
    
    # 1. Load symbols (with error handling for bad filter data from Binance)
    print("Initial loading of market symbols...")
    try:
        all_symbols = await client.load_symbols()
    except ValueError as e:
        # Fallback: Manual loading with skipping problematic symbols
        print(f"⚠️ Standard load failed ({e}). Using safe loader...")
        raw_info = await client.exchange_info()
        all_symbols = {}
        for s_data in raw_info['symbols']:
            try:
                # Check for zero stepSize before parsing
                skip = False
                for f in s_data.get('filters', []):
                    for k, v in f.items():
                        if k in ('stepSize', 'tickSize', 'minQty') and v == '0' or v == '0.0':
                            skip = True
                            break
                if skip:
                    continue
                sym_obj = binance.SymbolFutures(s_data)
                all_symbols[sym_obj.symbol] = sym_obj
            except Exception:
                continue  # Skip problematic symbols
    print(f"Loaded {len(all_symbols)} symbols.")
    
    # 1.5 VOLUME FILTER: Keep only top N symbols by 24h volume
    max_symbols = int(conf.max_symbols) if conf.max_symbols else 150
    blacklist = set((conf.blacklist or '').split(','))
    
    try:
        print(f"📈 Filtering top {max_symbols} symbols by 24h volume...")
        tickers = await client.ticker_24hr_price_change()
        
        # Filter to USDT pairs with volume, exclude blacklist
        valid_tickers = []
        for t in tickers:
            sym = t.get('symbol', '')
            if sym.endswith('USDT') and sym not in blacklist and sym in all_symbols:
                try:
                    vol = float(t.get('quoteVolume', 0))
                    valid_tickers.append((sym, vol))
                except:
                    continue
        
        # Sort by volume descending and keep top N
        valid_tickers.sort(key=lambda x: x[1], reverse=True)
        top_symbols = set(sym for sym, vol in valid_tickers[:max_symbols])
        
        # Filter all_symbols to only include top volume symbols
        filtered_symbols = {s: obj for s, obj in all_symbols.items() if s in top_symbols}
        print(f"✅ Filtered to {len(filtered_symbols)} symbols (from {len(all_symbols)}, blacklist: {len(blacklist)})")
        all_symbols = filtered_symbols
    except Exception as e:
        print(f"⚠️ Volume filter failed ({e}). Using all symbols.")
    
    # 2. Create pairs manager AFTER loading symbols
    pairs_manager = pairs_trading.PairsManager(
        client, 
        loop, 
        all_symbols, 
        timeframe=timeframe, 
        entry_timeframe=entry_timeframe,  # NEW: faster TF for entry signals
        min_data_points=window_size,
        notify_callback=send_tg_notification,
        config_info=conf
    )

    # CRITICAL: Initialize TG bot FIRST so notifications work during reconciliation
    await tg.init_bot()

    # CRITICAL: Initialize pairs manager (loads DB state + reconciles with exchange)
    await pairs_manager.initialize()

    # 3. Start background symbol updates
    loop.create_task(load_symbols_loop())
    
    # 4. Connect to websockets (both timeframes)
    loop.create_task(connect_ws(timeframe, entry_timeframe))
    
    # Run Telegram bot
    await tg.run(session, client, pairs_manager)


# Service to refresh all trading pairs every hour
async def load_symbols_loop():
    global all_symbols
    while True:
        try:
            await asyncio.sleep(3600)
            
            print("Refreshing market symbols...")
            all_symbols = await client.load_symbols()
            print(f"Refreshed {len(all_symbols)} symbols.")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error loading symbols: {e}")
            traceback.print_exc()


# Connect to websockets
async def connect_ws(timeframe='1h', entry_timeframe=None):
    global websockets_list
    global userdata_ws
    global pairs_manager

    print("Connecting to websockets...")

    # COLLECT ALL USDT PAIRS FOR SCANNER
    target_symbols = []
    
    # Load config for blacklist
    conf = await db.load_config()

    # Get blacklist from DB (which already contains defaults if initialized)
    FULL_BLACKLIST = set()
    if conf and conf.blacklist:
        FULL_BLACKLIST = set([s.strip().upper() for s in conf.blacklist.split(',') if s.strip()])

    # Test mode: whitelist test_pairs symbols (bypass blacklist)
    TEST_WHITELIST = set()
    test_mode = getattr(conf, 'test_mode', False)
    if isinstance(test_mode, str):
        test_mode = test_mode.lower() in ('true', '1', 'yes')
    if test_mode:
        test_pairs_str = getattr(conf, 'test_pairs', '') or ''
        for pair_str in test_pairs_str.split(','):
            parts = pair_str.strip().split('-')
            if len(parts) == 2:
                TEST_WHITELIST.add(parts[0].strip().upper())
                TEST_WHITELIST.add(parts[1].strip().upper())
        if TEST_WHITELIST:
            print(f"🧪 TEST MODE: Whitelisting symbols: {TEST_WHITELIST}")

    # 1. All USDT pairs from market
    for s_name, s_info in all_symbols.items():
        # Filter 1: Active and PERPETUAL only
        try:
            if getattr(s_info, 'contract_type', None) != 'PERPETUAL': continue
            if getattr(s_info, 'status', None) != 'TRADING': continue
            if getattr(s_info, 'quote_asset', None) != 'USDT': continue
        except:
            if not s_name.endswith('USDT'): continue

        # Filter 2: ASCII only (exclude Chinese chars and weird symbols)
        if not s_name.isascii():
            continue

        # Filter 3: Exclude blacklist (but allow TEST_WHITELIST in test_mode)
        if s_name in FULL_BLACKLIST and s_name not in TEST_WHITELIST:
            continue
            
        # Filter 4: Exclude stablecoins, USDC, leverage tokens, and special tokens
        BAD_PATTERNS = [
            'UPUSDT', 'DOWNUSDT', 'BEAR', 'BULL',  # Leveraged tokens
            'DAI', 'TUSD', 'USDP', 'FDUSD', 'USDC',  # Stablecoins
            'EURUSDT', 'GBPUSDT',  # Fiat pairs
            '3L', '3S', '2L', '2S',  # Leverage tokens
            'LEVERAGE'
        ]
        if any(pattern in s_name for pattern in BAD_PATTERNS):
            continue
            
        # Filter 5: Exclude symbols with underscore
        if '_' in s_name:
            continue

        target_symbols.append(s_name)
    
    target_symbols.sort()
    print(f"Subscribing to {len(target_symbols)} high-quality symbols (Filtered for PERPETUAL USDT-M).")

    # 2. Add symbols from DB (ONLY pairs with open positions)
    db_pairs = await db.get_all_pairs()
    active_db_count = 0
    for p in db_pairs:
        if p.position_status != 0:  # Only add if position is open
            if p.symbol1 not in target_symbols:
                target_symbols.append(p.symbol1)
            if p.symbol2 not in target_symbols:
                target_symbols.append(p.symbol2)
            active_db_count += 1
            
    print(f"Subscribing to {len(target_symbols)} symbols (Filtered: {len(target_symbols) - active_db_count * 2}, DB active: {active_db_count} pairs)...")

    # Optimization: Pre-load historical data using batch processing
    if pairs_manager:
        await pairs_manager.initialize_all_symbols_data(target_symbols)

    # MAIN TIMEFRAME: for discovery (cointegration tests)
    main_streams = [f"{symbol.lower()}@kline_{timeframe}" for symbol in target_symbols]
    
    # ENTRY TIMEFRAME: for faster entry signals (only if different from main)
    if entry_timeframe and entry_timeframe != timeframe:
        entry_streams = [f"{symbol.lower()}@kline_{entry_timeframe}" for symbol in target_symbols]
        print(f"MTF Mode: Main TF={timeframe} (discovery), Entry TF={entry_timeframe} (signals)")
    else:
        entry_streams = []
        print(f"Single TF Mode: {timeframe}")

    # Start websockets for MAIN timeframe
    chunk_size = 100
    streams_list = [main_streams[i:i + chunk_size] for i in range(0, len(main_streams), chunk_size)]

    for i, stream_list in enumerate(streams_list):
        try:
            ws = await client.websocket(stream_list, on_message=ws_msg_main, on_error=ws_error)
            websockets_list.append(ws)
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error subscribing to main TF chunk {i+1}: {e}")

    print(f"Connected to main TF kline websockets ({len(websockets_list)} connections).")
    
    # Start websockets for ENTRY timeframe (if MTF mode)
    if entry_streams:
        entry_streams_list = [entry_streams[i:i + chunk_size] for i in range(0, len(entry_streams), chunk_size)]
        entry_ws_count = 0
        for i, stream_list in enumerate(entry_streams_list):
            try:
                ws = await client.websocket(stream_list, on_message=ws_msg_entry, on_error=ws_error)
                websockets_list.append(ws)
                entry_ws_count += 1
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error subscribing to entry TF chunk {i+1}: {e}")
        print(f"Connected to entry TF kline websockets ({entry_ws_count} connections).")

    # Userdata websocket
    try:
        userdata_ws = await client.websocket_userdata(on_message=ws_user_msg, on_error=ws_error)
        print("Connected to userdata websocket.")
    except Exception as e:
        print(f"Could not connect to userdata stream: {e}")
    
    # Start real-time signal confirmation loop
    if pairs_manager:
        pairs_manager.start_realtime_monitoring()
        
        # Subscribe to markPrice stream for real-time price updates (for active pairs only)
        # Get symbols from active pairs
        active_symbols = set()
        for pair_info in pairs_manager.active_pairs.values():
            active_symbols.add(pair_info.symbol1)
            active_symbols.add(pair_info.symbol2)
        
        # Define markPrice handler FIRST (before any subscription)
        async def ws_mark_price(ws, msg):
            """Handle markPrice updates for real-time Z-score."""
            if 'data' not in msg:
                return
            data = msg['data']
            symbol = data.get('s')
            price = float(data.get('p', 0))
            if symbol and price > 0 and pairs_manager:
                await pairs_manager.on_ticker_update(symbol, price)
        
        # Track already subscribed symbols in pairs_manager
        pairs_manager._subscribed_mark_symbols = set(active_symbols)
        
        # Create callback for dynamic subscription (used when new pairs are discovered)
        async def subscribe_new_marks(symbols):
            """Subscribe to markPrice streams for new symbols dynamically."""
            if not symbols:
                return
            streams = [f"{s.lower()}@markPrice@1s" for s in symbols]
            try:
                ws = await client.websocket(streams, on_message=ws_mark_price, on_error=ws_error)
                websockets_list.append(ws)
            except Exception as e:
                print(f"⚠️ Failed to subscribe markPrice for {symbols}: {e}")
        
        # Set the callback on pairs_manager so it can subscribe new pairs
        pairs_manager._subscribe_mark_callback = subscribe_new_marks
        
        # Subscribe to initial symbols
        if active_symbols:
            mark_streams = [f"{sym.lower()}@markPrice@1s" for sym in active_symbols]
            mark_chunks = [mark_streams[i:i + chunk_size] for i in range(0, len(mark_streams), chunk_size)]
            
            for marks in mark_chunks:
                try:
                    ws = await client.websocket(marks, on_message=ws_mark_price, on_error=ws_error)
                    websockets_list.append(ws)
                except Exception as e:
                    print(f"Error subscribing to markPrice: {e}")
            
            print(f"Connected to markPrice websocket ({len(active_symbols)} initial symbols).")
        else:
            print("ℹ️ No active pairs at startup - markPrice will be subscribed dynamically.")


# Disconnect from websockets
async def disconnect_ws():
    global websockets_list
    global userdata_ws
    print("Disconnecting from websockets...")
    for ws in websockets_list:
        try:
            await ws.close()
        except:
            pass
    try:
        if 'userdata_ws' in globals() and userdata_ws:
            await userdata_ws.close()
    except:
        pass


# Handle websocket errors
async def ws_error(ws, error):
    print(f"WS ERROR: {error}")
    traceback.print_exc()


# Handle MAIN timeframe kline messages (for discovery + validation)
async def ws_msg_main(ws, msg):
    if 'data' not in msg:
        return
    
    kline = msg['data']['k']
    
    # Only process on candle close (discovery needs complete candles)
    if kline['x']:
        await pairs_manager.add_kline_main(kline)


# Handle ENTRY timeframe kline messages (for faster signal detection)
async def ws_msg_entry(ws, msg):
    if 'data' not in msg:
        return
    
    kline = msg['data']['k']
    
    # Only process on candle close (even for entry, we use closed 15m candles for stability)
    if kline['x']:
        await pairs_manager.add_kline_entry(kline)


# Handle userdata messages
async def ws_user_msg(ws, msg):
    """Handle userdata messages including order updates and position changes."""
    global pairs_manager
    
    event_type = msg.get('e')
    
    # DEBUG: Log all userdata events
    if event_type in ('ACCOUNT_UPDATE', 'ORDER_TRADE_UPDATE'):
        print(f"📡 UserData WS: {event_type} received")
    
    # Check for ACCOUNT_UPDATE (position changes - including manual closes)
    if event_type == 'ACCOUNT_UPDATE':
        positions = msg.get('a', {}).get('P', [])
        print(f"📡 ACCOUNT_UPDATE: {len(positions)} positions in update")
        
        # Notify about ALL position changes immediately
        closed_symbols = []
        for pos in positions:
            sym = pos.get('s')
            amt = float(pos.get('pa', 0))
            if amt == 0:
                closed_symbols.append(sym)
        
        # Note: Detailed notifications will be sent per-pair below
        # Skip simple "Position Changes" message - too noisy
        
        processed_pairs = set()  # Track pairs we've already handled in this update
        
        for pos in positions:
            symbol = pos.get('s')
            position_amt = float(pos.get('pa', 0))
            
            # Position closed (amount = 0)
            if position_amt == 0 and pairs_manager:
                # Check if this symbol is part of an active pair
                for pair_set, pair_info in list(pairs_manager.active_pairs.items()):
                    # Skip if already processed in this batch
                    if pair_set in processed_pairs:
                        continue
                        
                    if pair_info.position_status != 0 and symbol in [pair_info.symbol1, pair_info.symbol2]:
                        s1, s2 = pair_info.symbol1, pair_info.symbol2
                        other_symbol = s2 if symbol == s1 else s1
                        
                        # VALIDATION: Skip stale pairs where symbols don't exist anymore
                        if s1 not in pairs_manager.all_symbols or s2 not in pairs_manager.all_symbols:
                            print(f"⚠️ Skipping stale pair {s1}-{s2}: symbols not in trading list. Cleaning up...")
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
                        
                        # Check if both legs were closed in this same update (bulk close)
                        other_closed_in_batch = any(
                            p.get('s') == other_symbol and float(p.get('pa', 0)) == 0 
                            for p in positions
                        )
                        
                        pnl = float(pos.get('up', 0))
                        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                        
                        # Mark as processed in THIS batch only (prevents duplicate handling in same message)
                        processed_pairs.add(pair_set)
                        
                        # Set is_trading to prevent duplicate handling from pairs_trading leg sync
                        pair_info.is_trading = True
                        
                        if other_closed_in_batch:
                            # Check if bot already handled this close (prevent duplicate notification)
                            if getattr(pair_info, 'close_handled', False):
                                print(f"ℹ️ {s1}-{s2} close already handled by bot (reason: {getattr(pair_info, 'last_close_reason', 'unknown')}), skipping external notification")
                                pair_info.close_handled = False  # Reset for next trade
                                continue
                            
                            # Both legs closed together - fetch actual PnL and cleanup
                            print(f"⚡ Both legs of {s1}-{s2} closed externally. Fetching PnL...")
                            try:
                                await client.cancel_open_orders(s1)
                                await client.cancel_open_orders(s2)
                                
                                # Small delay to ensure trade data is available
                                import asyncio
                                await asyncio.sleep(1)
                                
                                # Fetch actual PnL from recent trades (more reliable than Income API)
                                import time as time_mod
                                now_ms = int(time_mod.time() * 1000)
                                start_ms = now_ms - 300_000  # Last 5 minutes
                                
                                trades1 = await client.get_account_trades(symbol=s1, startTime=start_ms, limit=50)
                                trades2 = await client.get_account_trades(symbol=s2, startTime=start_ms, limit=50)
                                
                                # Debug: show what API returned
                                print(f"📊 Trades for {s1}: {len(trades1)} entries")
                                print(f"📊 Trades for {s2}: {len(trades2)} entries")
                                
                                # Sum realized PnL and commissions from trades
                                pnl1 = sum(float(t.get('realizedPnl', 0)) for t in trades1)
                                pnl2 = sum(float(t.get('realizedPnl', 0)) for t in trades2)
                                fee1 = sum(float(t.get('commission', 0)) for t in trades1)
                                fee2 = sum(float(t.get('commission', 0)) for t in trades2)
                                total_pnl = pnl1 + pnl2
                                total_fees = fee1 + fee2
                                
                                # realizedPnl from Binance ALREADY includes fee deduction
                                # So net_pnl = total_pnl (fees shown for info only)
                                net_pnl = total_pnl
                                
                                # Determine close reason from trade data
                                close_type = '❓ Unknown'
                                close_hint = '\n💡 Check exchange for details'
                                # Define close reason mapping
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
                                    'external': '⚡ External Close',
                                    'orphan_restart': '🔄 Orphan on Restart',
                                    'stale_symbols': '⏳ Stale Symbols',
                                    'manual_partial': '👤 Manual Close (1 leg)',
                                }
                                
                                # FIRST: Check if bot stored exact reason
                                stored_reason = getattr(pair_info, 'last_close_reason', '')
                                if stored_reason and stored_reason in CLOSE_REASONS:
                                    close_type = CLOSE_REASONS[stored_reason]
                                    close_hint = ''
                                    print(f"📋 Using stored reason: {stored_reason} -> {close_type}")
                                else:
                                    # FALLBACK: Query orders to detect external close type
                                    try:
                                        orders1 = await client.get_all_orders(symbol=s1, limit=15)
                                        orders2 = await client.get_all_orders(symbol=s2, limit=15)
                                        
                                        now_ms = int(time_mod.time() * 1000)
                                        recent_orders = [o for o in orders1 + orders2 
                                                        if o.get('status') == 'FILLED' and o.get('updateTime', 0) > now_ms - 300_000]
                                        
                                        if recent_orders:
                                            recent_orders.sort(key=lambda x: x.get('updateTime', 0), reverse=True)
                                            o = recent_orders[0]
                                            o_type = o.get('type', '') or o.get('origType', '')
                                            
                                            if 'STOP' in o_type:
                                                close_type = '🛡️ Hardware SL'
                                            elif 'TAKE_PROFIT' in o_type:
                                                close_type = '🛡️ Hardware TP'
                                            elif o_type == 'MARKET':
                                                close_type = '👤 Manual Market' if not o.get('reduceOnly') else '🤖 Bot Close'
                                            elif o_type == 'LIMIT':
                                                close_type = '📊 Limit Order'
                                            elif 'TRAILING' in o_type:
                                                close_type = '📈 Trailing Stop'
                                            else:
                                                close_type = f'⚡ {o_type}'
                                            print(f"� Detected: {o_type} -> {close_type}")
                                        else:
                                            close_type = '⚡ External Close'
                                            close_hint = ' (no orders found)'
                                            print(f"⚠️ No orders for {s1}/{s2}")
                                    except Exception as e:
                                        print(f"⚠️ Query error: {e}")
                                        close_type = '⚡ External'
                                
                                pnl_emoji = '🟢' if net_pnl >= 0 else '🔴'
                                # Get Z-score and Beta from pair_info
                                zscore = getattr(pair_info, 'zscore', 0) or 0
                                beta = getattr(pair_info, 'beta', 0) or 0
                                e1 = '🟢' if pnl1 >= 0 else '🔴'
                                e2 = '🟢' if pnl2 >= 0 else '🔴'
                                msg_text = (f"{close_type}: <b>{s1}-{s2}</b>\n\n"
                                            f"📊 Z: {zscore:+.2f} | β: {beta:.3f}\n"
                                            f"💵 PnL: {pnl_emoji} <b>{net_pnl:+.2f} USDT</b>\n"
                                            f"   {e1} {s1}: {pnl1:+.2f} | {e2} {s2}: {pnl2:+.2f}\n"
                                            f"💸 Fees: {total_fees:.4f} USDT{close_hint}")
                                
                                reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                                await send_tg_notification(msg_text, reply_to)
                                
                                # Update memory state
                                pair_info.position_status = 0
                                pair_info.qty1 = 0
                                pair_info.qty2 = 0
                                pair_info.is_trading = False
                                
                                # Update DB with PnL details
                                if pair_info.db_id:
                                    await db.update_pair({
                                        'id': pair_info.db_id,
                                        'position_status': 0,
                                        'qty1': 0,
                                        'qty2': 0,
                                        'close_time': int(time_mod.time()),
                                        'close_pnl': net_pnl,
                                        'close_reason': stored_reason if stored_reason else 'external',
                                        'pnl1': pnl1,
                                        'pnl2': pnl2,
                                        'fee1': fee1,
                                        'fee2': fee2
                                    })
                            except Exception as e:
                                print(f"⚠️ Cleanup error: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            # Only one leg closed - user manually closed one position
                            print(f"⚡ External close detected: {symbol} in pair {s1}-{s2}. Closing {other_symbol}...")
                            pair_info.last_close_reason = 'manual_partial'  # User manually closed one leg
                            try:
                                await client.cancel_open_orders(s1)
                                await client.cancel_open_orders(s2)
                                
                                # Use get_position_risk to get position info
                                positions_data = await client.get_position_risk(symbol=other_symbol)
                                other_pos = positions_data[0] if positions_data else {}
                                other_amt = float(other_pos.get('positionAmt', 0))
                                
                                if other_amt != 0:
                                    close_side = 'SELL' if other_amt > 0 else 'BUY'
                                    await client.new_order(symbol=other_symbol, side=close_side, type='MARKET', 
                                                          quantity=abs(other_amt), reduceOnly='true')
                                    print(f"✅ Closed remaining leg {other_symbol}")
                                
                                # Small delay to ensure trade data is available
                                import asyncio
                                await asyncio.sleep(1)
                                
                                # Fetch actual PnL from recent trades (more reliable than Income API)
                                import time as time_mod
                                now_ms = int(time_mod.time() * 1000)
                                start_ms = now_ms - 300_000  # Last 5 minutes
                                
                                trades1 = await client.get_account_trades(symbol=s1, startTime=start_ms, limit=50)
                                trades2 = await client.get_account_trades(symbol=s2, startTime=start_ms, limit=50)
                                
                                # Debug: show what API returned
                                print(f"📊 Trades for {s1}: {len(trades1)} entries")
                                print(f"📊 Trades for {s2}: {len(trades2)} entries")
                                
                                # Sum realized PnL and commissions from trades
                                pnl1 = sum(float(t.get('realizedPnl', 0)) for t in trades1)
                                pnl2 = sum(float(t.get('realizedPnl', 0)) for t in trades2)
                                fee1 = sum(float(t.get('commission', 0)) for t in trades1)
                                fee2 = sum(float(t.get('commission', 0)) for t in trades2)
                                total_pnl = pnl1 + pnl2
                                total_fees = fee1 + fee2
                                
                                # realizedPnl from Binance ALREADY includes fee deduction
                                net_pnl = total_pnl
                                
                                pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"
                                
                                # Update memory state
                                pair_info.position_status = 0
                                pair_info.qty1 = 0
                                pair_info.qty2 = 0
                                pair_info.is_trading = False
                                
                                # Get stored reason from pair_info (set by bot when closing)
                                stored_reason = getattr(pair_info, 'last_close_reason', '')
                                
                                # Update DB with PnL details
                                if pair_info.db_id:
                                    await db.update_pair({
                                        'id': pair_info.db_id,
                                        'position_status': 0,
                                        'qty1': 0,
                                        'qty2': 0,
                                        'close_time': int(time_mod.time()),
                                        'close_pnl': net_pnl,
                                        'close_reason': stored_reason if stored_reason else 'external',
                                        'pnl1': pnl1,
                                        'pnl2': pnl2,
                                        'fee1': fee1,
                                        'fee2': fee2
                                    })
                                
                                # Define close reason mapping
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
                                    'external': '⚡ External Close',
                                    'orphan_restart': '🔄 Orphan on Restart',
                                    'stale_symbols': '⏳ Stale Symbols',
                                    'manual_partial': '👤 Manual Close (1 leg)',
                                }
                                
                                # Default values (in case no reason is found)
                                close_type = '❓ Unknown'
                                close_hint = ''
                                
                                # FIRST: Check if bot stored exact reason
                                stored_reason = getattr(pair_info, 'last_close_reason', '')
                                if stored_reason and stored_reason in CLOSE_REASONS:
                                    close_type = CLOSE_REASONS[stored_reason]
                                    close_hint = ''
                                    print(f"📋 Using stored reason: {stored_reason} -> {close_type}")
                                else:
                                    # FALLBACK: Query orders to detect external close type
                                    try:
                                        orders1 = await client.get_all_orders(symbol=s1, limit=15)
                                        orders2 = await client.get_all_orders(symbol=s2, limit=15)
                                        
                                        now_ms = int(time_mod.time() * 1000)
                                        recent_orders = []
                                        for o in orders1 + orders2:
                                            if o.get('status') == 'FILLED' and o.get('updateTime', 0) > now_ms - 300_000:
                                                recent_orders.append(o)
                                        
                                        if recent_orders:
                                            # Sort by time, newest first
                                            recent_orders.sort(key=lambda x: x.get('updateTime', 0), reverse=True)
                                            o = recent_orders[0]
                                            o_type = o.get('type', '') or o.get('origType', '')
                                            
                                            if 'STOP' in o_type:
                                                close_type = '🛡️ Hardware SL'
                                            elif 'TAKE_PROFIT' in o_type:
                                                close_type = '🛡️ Hardware TP'
                                            elif o_type == 'MARKET':
                                                # Check if reduceOnly - that's bot closing
                                                if o.get('reduceOnly', False):
                                                    close_type = '🤖 Bot Close (reason unknown)'
                                                else:
                                                    close_type = '👤 Manual Market Order'
                                            elif o_type == 'LIMIT':
                                                close_type = '📊 Limit Order Filled'
                                            elif 'TRAILING' in o_type:
                                                close_type = '📈 Trailing Stop'
                                            else:
                                                close_type = f'⚡ Order: {o_type}'
                                            
                                            print(f"� Detected from orders: {o_type} -> {close_type}")
                                        else:
                                            # No recent orders found - truly external or WebSocket miss
                                            close_type = '⚡ External Close'
                                            close_hint = ' (no matching orders)'
                                            print(f"⚠️ No recent orders found for {s1}/{s2} - external close")
                                            
                                    except Exception as e:
                                        print(f"⚠️ Could not query orders: {e}")
                                        close_type = '⚡ External Close'
                                        close_hint = ' (query failed)'
                                
                                pnl_emoji = '🟢' if net_pnl >= 0 else '🔴'
                                # Get Z-score and Beta from pair_info
                                zscore = getattr(pair_info, 'zscore', 0) or 0
                                beta = getattr(pair_info, 'beta', 0) or 0
                                e1 = '🟢' if pnl1 >= 0 else '🔴'
                                e2 = '🟢' if pnl2 >= 0 else '🔴'
                                done_msg = (f"{close_type}: <b>{s1}-{s2}</b>\n\n"
                                            f"📊 Z: {zscore:+.2f} | β: {beta:.3f}\n"
                                            f"💵 PnL: {pnl_emoji} <b>{net_pnl:+.2f} USDT</b>\n"
                                            f"   {e1} {s1}: {pnl1:+.2f} | {e2} {s2}: {pnl2:+.2f}\n"
                                            f"💸 Fees: {total_fees:.4f} USDT{close_hint}")
                                reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                                await send_tg_notification(done_msg, reply_to)
                                
                            except Exception as e:
                                print(f"⚠️ External close handling error for {s1}-{s2}: {e}")
                                import traceback
                                traceback.print_exc()
                        break
    
    # Check for ORDER_TRADE_UPDATE (order filled/canceled)
    if msg.get('e') == 'ORDER_TRADE_UPDATE':
        order = msg.get('o', {})
        symbol = order.get('s')
        order_type = order.get('ot')  # Original order type: STOP, TAKE_PROFIT, MARKET, etc.
        status = order.get('X')       # FILLED, CANCELED, NEW, etc.
        side = order.get('S')         # BUY, SELL
        order_id = order.get('i')     # Order ID
        
        # Check if this is a filled SL/TP order (hardware stop triggered)
        if status == 'FILLED' and order_type in ('STOP', 'TAKE_PROFIT', 'STOP_MARKET', 'TAKE_PROFIT_MARKET'):
            print(f"🎯 Hardware SL/TP triggered: {symbol} {order_type} FILLED")
            
            # Notify pairs_manager to close the other leg
            if pairs_manager:
                try:
                    await pairs_manager.handle_sl_tp_triggered(symbol, order_type)
                except Exception as e:
                    print(f"⚠️ Error handling SL/TP trigger: {e}")
        
        # CANCELED order - notify user and trigger cleanup
        elif status == 'CANCELED' and order_type in ('STOP', 'TAKE_PROFIT', 'STOP_MARKET', 'TAKE_PROFIT_MARKET'):
            print(f"⚠️ SL/TP CANCELED: {symbol} {order_type} by user/system")
            
            # Find which pair this order belongs to
            if pairs_manager:
                for pair_info in pairs_manager.active_pairs.values():
                    if pair_info.position_status != 0 and symbol in [pair_info.symbol1, pair_info.symbol2]:
                        s1, s2 = pair_info.symbol1, pair_info.symbol2
                        other_symbol = s2 if symbol == s1 else s1
                        
                        # Notify user about manual order cancellation
                        cancel_msg = (f"⚠️ <b>Order CANCELED:</b> {symbol}\n"
                                      f"Type: {order_type}\n"
                                      f"Pair: {s1}-{s2}\n"
                                      f"⏳ Checking pair integrity...")
                        try:
                            await send_tg_notification(cancel_msg)
                        except Exception as e:
                            print(f"⚠️ TG notify error: {e}")
                        
                        # Trigger immediate leg sync check for this pair
                        try:
                            await pairs_manager._check_and_sync_legs()
                        except Exception as e:
                            print(f"⚠️ Leg sync error after cancel: {e}")
                        break


if __name__ == '__main__':
    try:
        print("Starting market neutral bot...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user.")
