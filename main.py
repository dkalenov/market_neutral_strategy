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

    # Function to send TG notifications
    async def send_tg_notification(message):
        if tg.bot and tg_admins:
             # Send to all admins
             admins = [int(admin_id) for admin_id in tg_admins.split(',') if admin_id.strip()]
             for admin_id in admins:
                 try:
                     await tg.bot.send_message(admin_id, message)
                 except Exception as e:
                     print(f"Error sending TG message to {admin_id}: {e}")

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
    
    # Entry timeframe for faster signals (default: 15m)
    entry_timeframe = conf.entry_timeframe if conf.entry_timeframe else '15m'
    
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

    # 2. Add symbols from DB (pairs that are already active)
    db_pairs = await db.get_all_pairs()
    for p in db_pairs:
        if p.symbol1 not in target_symbols:
            target_symbols.append(p.symbol1)
        if p.symbol2 not in target_symbols:
            target_symbols.append(p.symbol2)
            
    print(f"Subscribing to {len(target_symbols)} symbols (Market + DB active)...")

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
    """Handle userdata messages including order updates."""
    global pairs_manager
    
    # Check for ORDER_TRADE_UPDATE (order filled/canceled)
    if msg.get('e') == 'ORDER_TRADE_UPDATE':
        order = msg.get('o', {})
        symbol = order.get('s')
        order_type = order.get('ot')  # Original order type: STOP, TAKE_PROFIT, MARKET, etc.
        status = order.get('X')       # FILLED, CANCELED, NEW, etc.
        
        # Check if this is a filled SL/TP order (hardware stop triggered)
        if status == 'FILLED' and order_type in ('STOP', 'TAKE_PROFIT'):
            print(f"🎯 Hardware SL/TP triggered: {symbol} {order_type} FILLED")
            
            # Notify pairs_manager to close the other leg
            if pairs_manager:
                try:
                    await pairs_manager.handle_sl_tp_triggered(symbol)
                except Exception as e:
                    print(f"⚠️ Error handling SL/TP trigger: {e}")


if __name__ == '__main__':
    try:
        print("Starting market neutral bot...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user.")
