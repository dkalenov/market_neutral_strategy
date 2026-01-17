import asyncio
import configparser
import traceback
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

    # Initial sync from config.ini to DB if DB is empty or has placeholders
    sync_needed = False
    updates = {}

    def needs_sync(val, key_name=None):
        if val is None:
            return True
        if not isinstance(val, str):
            return False
        # If it's a key/secret/token, it should be long. 32 chars means it was truncated.
        if key_name in ['api_key', 'api_secret', 'tg_token'] and len(val) <= 32:
            return True
        return val.lower() in ['missing', 'none', '', 'отсутствует']

    # Get values from config.ini
    api_key_ini = ini_config.get('API', 'KEY', fallback=None) if ini_config.has_section('API') else None
    api_secret_ini = ini_config.get('API', 'SECRET', fallback=None) if ini_config.has_section('API') else None
    tg_token_ini = ini_config.get('TG', 'TOKEN', fallback=None) if ini_config.has_section('TG') else None
    tg_admins_ini = ini_config.get('TG', 'ADMINS', fallback=None) if ini_config.has_section('TG') else None

    if needs_sync(conf.api_key, 'api_key') or (api_key_ini and conf.api_key != api_key_ini):
        if api_key_ini:
            updates['api_key'] = api_key_ini
            sync_needed = True
    if needs_sync(conf.api_secret, 'api_secret') or (api_secret_ini and conf.api_secret != api_secret_ini):
        if api_secret_ini:
            updates['api_secret'] = api_secret_ini
            sync_needed = True
    if needs_sync(conf.tg_token, 'tg_token') or (tg_token_ini and conf.tg_token != tg_token_ini):
        if tg_token_ini:
            updates['tg_token'] = tg_token_ini
            sync_needed = True
    if needs_sync(conf.tg_admins, 'tg_admins') or (tg_admins_ini and conf.tg_admins != tg_admins_ini):
        if tg_admins_ini:
            updates['tg_admins'] = tg_admins_ini
            sync_needed = True

    # Sync DB config to table
    db_params = ['host', 'port', 'user', 'password', 'db_name']
    for param in db_params:
        ini_val = ini_config.get('DB', param, fallback=None)
        db_key_name = f"db_{param}" if param != 'db_name' else 'db_name'
        current_db_val = getattr(conf, db_key_name, None)
        
        if ini_val and str(current_db_val) != str(ini_val):
            updates[db_key_name] = ini_val
            sync_needed = True

    if sync_needed:
        print("Syncing credentials from config.ini to database...")
        await db.config_update(**updates)
        conf = await db.load_config() # Reload synced config

    # Function to send TG notifications
    async def send_tg_notification(message):
        if tg.bot and conf.tg_admins:
             # Send to all admins
             admins = [int(admin_id) for admin_id in conf.tg_admins.split(',') if admin_id] if conf.tg_admins else []
             for admin_id in admins:
                 try:
                     await tg.bot.send_message(admin_id, message)
                 except Exception as e:
                     print(f"Error sending TG message to {admin_id}: {e}")

    # Create Binance client
    client = binance.Futures(api_key=conf.api_key,
                             secret_key=conf.api_secret,
                             asynced=True,
                             testnet=ini_config.getboolean('BOT', 'testnet'))
    # Init pairs manager
    loop = asyncio.get_running_loop()
    
    # Default values
    timeframe = conf.timeframe if conf.timeframe else '1h'
    
    if conf.window_size:
        window_size = conf.window_size
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
    
    # 1. Load symbols
    print("Initial loading of market symbols...")
    all_symbols = await client.load_symbols()
    print(f"Loaded {len(all_symbols)} symbols.")
    
    # 2. Create pairs manager AFTER loading symbols
    pairs_manager = pairs_trading.PairsManager(
        client, 
        loop, 
        all_symbols, 
        timeframe=timeframe, 
        min_data_points=window_size,
        notify_callback=send_tg_notification,
        config_info=conf
    )

    # 3. Start background symbol updates
    loop.create_task(load_symbols_loop())
    
    # 4. Connect to websockets
    loop.create_task(connect_ws(timeframe))
    
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
async def connect_ws(timeframe='1h'):
    global websockets_list
    global userdata_ws

    # #region agent log
    import os
    import json
    import time
    log_path = r"c:\Users\Dmitrii\Trading strategies\Market_neutral_strategy\.cursor\debug.log"
    def log_instrument(location, message, data=None):
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                entry = {
                    "id": f"log_{int(time.time()*1000)}_ws",
                    "timestamp": int(time.time()*1000),
                    "location": location,
                    "message": message,
                    "data": data or {},
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "WS_STABILITY_3"
                }
                f.write(json.dumps(entry) + '\n')
        except: pass
    # #endregion

    log_instrument("main.py:connect_ws", "Starting websocket connections", {"timeframe": timeframe})
    print("Connecting to websockets...")

    # COLLECT ALL USDT PAIRS FOR SCANNER
    target_symbols = []
    
    # Load config for blacklist
    conf = await db.load_config()

    # Get blacklist from DB (which already contains defaults if initialized)
    FULL_BLACKLIST = set()
    if conf and conf.blacklist:
        FULL_BLACKLIST = set([s.strip().upper() for s in conf.blacklist.split(',') if s.strip()])

    # 1. All USDT pairs from market
    for s_name, s_info in all_symbols.items():
        # Filter 1: Active and PERPETUAL only
        try:
            if getattr(s_info, 'contract_type', None) != 'PERPETUAL': continue
            if getattr(s_info, 'status', None) != 'TRADING': continue
            if getattr(s_info, 'quote_asset', None) != 'USDT': continue
        except:
            if not s_name.endswith('USDT'): continue

        # Filter 2: Exclude blacklist
        if s_name in FULL_BLACKLIST:
            continue
            
        # Filter 3: Exclude stablecoins and special tokens
        if any(x in s_name for x in ['UPUSDT', 'DOWNUSDT', 'BEAR', 'BULL', 'DAI', 'TUSD', 'USDP', 'FDUSD', 'EURUSDT', 'GBPUSDT']):
            continue
            
        # Filter 4: Exclude symbols with underscore
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

    streams = [f"{symbol.lower()}@kline_{timeframe}" for symbol in target_symbols]

    # Start websockets
    chunk_size = 100
    streams_list = [streams[i:i + chunk_size] for i in range(0, len(streams), chunk_size)]
    log_instrument("main.py:connect_ws", f"Starting websocket connections with {len(streams_list)} chunks")

    for i, stream_list in enumerate(streams_list):
        try:
            log_instrument("main.py:connect_ws", f"Connecting chunk {i+1}/{len(streams_list)}", {"chunk_size": len(stream_list)})
            ws = await client.websocket(stream_list, on_message=ws_msg, on_error=ws_error)
            websockets_list.append(ws)
            log_instrument("main.py:connect_ws", f"Chunk {i+1} connected successfully")
            await asyncio.sleep(0.1)
        except Exception as e:
            log_instrument("main.py:connect_ws", f"Failed to connect chunk {i+1}", {"error": str(e), "error_type": type(e).__name__})
            print(f"Error subscribing to chunk: {e}")

    log_instrument("main.py:connect_ws", f"Kline websockets setup complete", {"connections": len(websockets_list)})
    print(f"Connected to kline websockets ({len(websockets_list)} connections).")

    # Userdata websocket
    try:
        log_instrument("main.py:connect_ws", "Connecting userdata websocket")
        userdata_ws = await client.websocket_userdata(on_message=ws_user_msg, on_error=ws_error)
        log_instrument("main.py:connect_ws", "Userdata websocket connected successfully")
        print("Connected to userdata websocket.")
    except Exception as e:
        log_instrument("main.py:connect_ws", "Userdata websocket connection failed", {"error": str(e), "error_type": type(e).__name__})
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


# Handle kline messages
async def ws_msg(ws, msg):
    if 'data' not in msg:
        return
    
    kline = msg['data']['k']
    
    # If kline closed
    if kline['x']:
        await pairs_manager.add_kline(kline)


# Handle userdata messages
async def ws_user_msg(ws, msg):
    print(f"Userdata message: {msg}")


if __name__ == '__main__':
    try:
        print("Starting market neutral bot...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user.")
