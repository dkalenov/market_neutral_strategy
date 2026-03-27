import asyncio
import configparser
import traceback
import os
import time as time_mod
import math
import sys
import builtins
from collections import deque
import aiohttp
import csv
import json
from urllib.parse import urlsplit, parse_qs
from dotenv import load_dotenv
import binance
import pairs_trading
from pairs_trading import CLOSE_REASONS
import db
import tg


client: binance.Futures

all_symbols: dict[str, binance.SymbolFutures] = {}
websockets_list: list[binance.futures.WebsocketAsync] = []
main_kline_wss: list[binance.futures.WebsocketAsync] = []
userdata_ws: binance.futures.WebsocketAsync
pairs_manager: pairs_trading.PairsManager

# TG notification globals
tg_channel_global = ''
tg_admins_global = ''
# Anti-spam cache for noisy untracked-close websocket events
untracked_close_alerts: dict[str, float] = {}
# Anti-spam cache for unknown/external open position websocket events
untracked_open_alerts: dict[str, float] = {}
# Short-lived suppression for symbols that were just handled by pair close logic.
# Prevents false "UNTRACKED POSITION CLOSED" on the next ACCOUNT_UPDATE tick.
recently_handled_close_symbols: dict[str, float] = {}
_orig_print = builtins.print
_main_timeframe_global: str = '1h'

# WS health/watchdog state
_ws_last_main_msg_ts: float = 0.0
_ws_last_mark_msg_ts: float = 0.0
_ws_last_user_msg_ts: float = 0.0
_ws_error_ts: deque = deque(maxlen=512)
_ws_recover_lock: asyncio.Lock | None = None
_ws_last_recover_ts: float = 0.0
_main_kline_reload_lock: asyncio.Lock | None = None

_BAD_SYMBOL_PATTERNS = (
    'UPUSDT', 'DOWNUSDT', 'BEAR', 'BULL',
    'DAI', 'TUSD', 'USDP', 'FDUSD', 'USDC',
    'EURUSDT', 'GBPUSDT',
    '3L', '3S', '2L', '2S',
    'LEVERAGE',
)


def _is_tradeable_usdt_symbol_name(symbol: str) -> bool:
    """Unified runtime symbol filter for ws subscriptions/warmup candidates."""
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


def _cfg_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _priority_file_path_from_config(conf) -> str:
    path = getattr(conf, 'priority_pairs_file', '') or ''
    if path and not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)
    return path


def _extract_priority_symbols_from_file(path: str) -> set[str]:
    symbols: set[str] = set()
    if not path or not os.path.exists(path):
        return symbols
    try:
        suffix = os.path.splitext(path)[1].lower()
        raw_entries = []
        if suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                raw_entries = payload.get('pairs', []) or payload.get('items', []) or []
            elif isinstance(payload, list):
                raw_entries = payload
        elif suffix in ('.csv', '.tsv'):
            with open(path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f, delimiter='\t' if suffix == '.tsv' else ',')
                raw_entries = list(reader)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                raw_entries = [line.strip() for line in f if line.strip()]
    except Exception:
        return symbols

    for item in raw_entries:
        parsed = pairs_trading._parse_pair_text(item)
        if not parsed:
            continue
        s1, s2 = parsed
        # Priority symbols are pre-vetted in backtest — skip _is_tradeable filter
        if s1 and s1.endswith('USDT'):
            symbols.add(s1)
        if s2 and s2.endswith('USDT'):
            symbols.add(s2)
    return symbols


def _configure_console_encoding():
    """
    Best-effort UTF-8 console setup for Windows to avoid mojibake in logs.
    Safe no-op on non-Windows platforms.
    """
    try:
        if os.name == 'nt':
            try:
                import ctypes
                ctypes.windll.kernel32.SetConsoleOutputCP(65001)
                ctypes.windll.kernel32.SetConsoleCP(65001)
            except Exception:
                pass
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def _fix_mojibake_text(s: str) -> str:
    """
    Best-effort repair for UTF-8 text decoded as latin1/cp1252.
    """
    if not isinstance(s, str) or not s:
        return s
    if not any(ch in s for ch in ("\u00f0", "\u00e2", "\u00ce", "\u00d0", "\u00d1", "\u00c3")):
        return s

    def _score(text: str) -> int:
        bad = sum(text.count(x) for x in ("\u00f0", "\u00e2", "\u00ce", "\u00c3", "\u00c2", "\u00ef\u00b8", "\u00e2\u2020"))
        return -bad

    candidates = [s]
    for enc in ("latin-1", "cp1252"):
        cur = s
        for _ in range(2):
            try:
                nxt = cur.encode(enc, errors="strict").decode("utf-8", errors="strict")
            except Exception:
                break
            if nxt == cur:
                break
            candidates.append(nxt)
            cur = nxt

    return max(candidates, key=_score)


def _install_print_mojibake_fix():
    """Patch print so console logs are auto-normalized."""
    def _fixed_print(*args, **kwargs):
        fixed_args = [(_fix_mojibake_text(a) if isinstance(a, str) else a) for a in args]
        _orig_print(*fixed_args, **kwargs)
    builtins.print = _fixed_print

async def send_tg_notification(message, reply_to_message_id=None, reply_markup=None):
    """Send notification to TG channel or admins. Returns message_id for reply threading."""
    if isinstance(message, str):
        message = _fix_mojibake_text(message)
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


async def _persist_pair_executions_main(pair_info, trades1, trades2, phase: str):
    """Persist fill-level executions for closure paths handled in main.py."""
    if not pair_info:
        return
    rows = []
    trade_id = getattr(pair_info, 'current_trade_id', None)
    pair_id = getattr(pair_info, 'db_id', None)
    now_ms = int(time_mod.time() * 1000)
    for symbol, trades in ((pair_info.symbol1, trades1), (pair_info.symbol2, trades2)):
        for t in trades or []:
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
                    'created_at': now_ms,
                })
            except Exception:
                continue
    if rows:
        try:
            await db.add_trade_executions(rows)
        except Exception as e:
            print(f"⚠️ Could not persist executions [{phase}] for {pair_info.symbol1}-{pair_info.symbol2}: {e}")


async def _fetch_account_trades_window_main(symbol: str, start_ms: int, *, max_records: int = 3000, page_limit: int = 1000):
    """Paged user-trades fetch to avoid truncating PnL/fill history on busy closes."""
    out = []
    seen = set()
    cursor = int(max(0, start_ms))
    hard_cap = int(max(1, max_records))
    limit = int(min(1000, max(1, page_limit)))
    for _ in range(20):
        if len(out) >= hard_cap:
            break
        batch = await client.get_account_trades(symbol=symbol, startTime=cursor, limit=limit)
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


async def main():
    global client
    global pairs_manager
    global all_symbols
    global _main_timeframe_global
    global _ws_recover_lock
    global _main_kline_reload_lock

    startup_marks: list[tuple[str, float]] = []
    startup_profile = os.getenv('STARTUP_PROFILE', 'true').strip().lower() not in ('0', 'false', 'no')
    bg_tasks: list[asyncio.Task] = []

    def _mark(stage: str):
        if startup_profile:
            startup_marks.append((stage, time_mod.perf_counter()))

    def _print_startup_profile(final_stage: str):
        if not startup_profile:
            return
        _mark(final_stage)
        if len(startup_marks) < 2:
            return
        print("=== STARTUP PROFILE ===")
        prev_name, prev_ts = startup_marks[0]
        for name, ts in startup_marks[1:]:
            print(f"  {prev_name} -> {name}: {ts - prev_ts:.2f}s")
            prev_name, prev_ts = name, ts
        print(f"  TOTAL: {startup_marks[-1][1] - startup_marks[0][1]:.2f}s")
        print("=======================")
    _mark('start')
    
    # Load environment variables from .env file
    _configure_console_encoding()
    load_dotenv()
    _mark('env_loaded')
    
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
    _mark('db_connected')

    # Load config from DB
    conf = await db.load_config()
    _mark('config_loaded')
    # Non-destructive integrity audit (warn-only). Helps detect historical DB drift early.
    try:
        integrity = await db.audit_data_integrity(sample_limit=10)
        issues = []
        if integrity.get('duplicate_active_pairs', 0) > 0:
            issues.append(f"duplicate_active_pairs={integrity['duplicate_active_pairs']}")
        if integrity.get('open_trades_without_pair', 0) > 0:
            issues.append(f"open_trades_without_pair={integrity['open_trades_without_pair']}")
        if integrity.get('open_trades_with_closed_pair', 0) > 0:
            issues.append(f"open_trades_with_closed_pair={integrity['open_trades_with_closed_pair']}")
        if integrity.get('pairs_with_multiple_open_trades', 0) > 0:
            issues.append(f"pairs_with_multiple_open_trades={integrity['pairs_with_multiple_open_trades']}")
        if issues:
            print("CRITICAL DB integrity issues detected: " + ", ".join(issues))
            sample = integrity.get('duplicate_active_pairs_sample', [])
            for item in sample[:5]:
                print(f"  DUP: {item.get('symbol1')}-{item.get('symbol2')} x{item.get('count')}")
    except Exception as e:
        print(f"⚠️ DB integrity audit failed: {e}")

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
    loop = asyncio.get_running_loop()
    tg_task = None
    if tg_token:
        try:
            # Bring control channel online before heavy exchange bootstrap.
            tg_ready = await tg.init_bot()
            if tg_ready:
                tg.attach_runtime(session, None, None)
                tg_task = loop.create_task(tg.run(session, None, None))
                print("TG control channel started before exchange checks.")
        except Exception as e:
            print(f"⚠️ Early TG startup failed: {e}")
    _mark('tg_early_started')

    try:
        # Create Binance client
        client = binance.Futures(api_key=api_key,
                                 secret_key=api_secret,
                                 asynced=True,
                                 testnet=ini_config.getboolean('BOT', 'testnet'))
        tg.attach_runtime(session, client, None)
        _mark('client_created')
        
        # CRITICAL: Sync time with server to avoid -1021 timestamp errors.
        try:
            await client._sync_time_async()
        except Exception as e:
            print(f"⚠️ Initial time sync failed: {e}")
        _mark('time_synced')
        
        # Init pairs manager
        
        # Default values
        timeframe = conf.timeframe if conf.timeframe else '1h'
        _main_timeframe_global = timeframe
        if _ws_recover_lock is None:
            _ws_recover_lock = asyncio.Lock()
        if _main_kline_reload_lock is None:
            _main_kline_reload_lock = asyncio.Lock()
        
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
                            if k in ('stepSize', 'tickSize', 'minQty') and (v == '0' or v == '0.0'):
                                skip = True
                                break
                    if skip:
                        continue
                    sym_obj = binance.SymbolFutures(s_data)
                    all_symbols[sym_obj.symbol] = sym_obj
                except Exception:
                    continue  # Skip problematic symbols
        loaded_raw_count = len(all_symbols)
        all_symbols_unfiltered = dict(all_symbols)  # Keep unfiltered copy for priority lookups
        all_symbols = {s: obj for s, obj in all_symbols.items() if _is_tradeable_usdt_symbol_name(s)}
        print(f"Loaded {len(all_symbols)} symbols (raw: {loaded_raw_count}).")
        _mark('symbols_loaded')
        
        # 1.5 Symbol universe selection
        max_symbols = int(conf.max_symbols) if conf.max_symbols else 150
        blacklist = {s.strip().upper() for s in (conf.blacklist or '').split(',') if s.strip()}
        best_pairs_only = _cfg_bool(getattr(conf, 'best_pairs_only', False), False)
        priority_file_path = _priority_file_path_from_config(conf)
        
        try:
            top_symbols = set()

            if best_pairs_only:
                priority_symbols = _extract_priority_symbols_from_file(priority_file_path)
                # Use unfiltered symbols — priority pairs are pre-vetted in backtest
                top_symbols.update(s for s in priority_symbols if s in all_symbols_unfiltered and s not in blacklist)
                print(
                    f"🎯 BEST-PAIRS-ONLY universe: {len(top_symbols)} symbols from priority file"
                    f"{f' ({priority_file_path})' if priority_file_path else ''}"
                )
            else:
                print(f"📈 Filtering top {max_symbols} symbols by 24h volume...")
                tickers = await client.ticker_24hr_price_change()
                
                # Filter to USDT pairs with volume, exclude blacklist
                valid_tickers = []
                for t in tickers:
                    sym = t.get('symbol', '')
                    if (
                        _is_tradeable_usdt_symbol_name(sym)
                        and sym not in blacklist
                        and sym in all_symbols
                    ):
                        try:
                            vol = float(t.get('quoteVolume', 0))
                            valid_tickers.append((sym, vol))
                        except:
                            continue
                
                # Sort by volume descending and keep top N
                valid_tickers.sort(key=lambda x: x[1], reverse=True)
                top_symbols.update(sym for sym, vol in valid_tickers[:max_symbols])

            # Always keep BTCUSDT for market-beta calculations.
            if 'BTCUSDT' in all_symbols:
                top_symbols.add('BTCUSDT')
            
            # SAFETY: Always keep symbols that currently have open positions (DB or exchange)
            protected_symbols = set()
            try:
                db_pairs = await db.get_all_pairs()
                for p in db_pairs:
                    if getattr(p, 'position_status', 0) != 0:
                        s1 = str(getattr(p, 'symbol1', '') or '').strip().upper()
                        s2 = str(getattr(p, 'symbol2', '') or '').strip().upper()
                        if _is_tradeable_usdt_symbol_name(s1):
                            protected_symbols.add(s1)
                        if _is_tradeable_usdt_symbol_name(s2):
                            protected_symbols.add(s2)
            except Exception as e:
                print(f"Could not load protected symbols from DB: {e}")
            try:
                exchange_positions = await client.get_position_risk()
                for pos in exchange_positions:
                    sym = str(pos.get('symbol', '') or '').strip().upper()
                    if abs(float(pos.get('positionAmt', 0))) > 0 and _is_tradeable_usdt_symbol_name(sym):
                        protected_symbols.add(sym)
            except Exception as e:
                print(f"Could not load protected symbols from exchange: {e}")
            
            top_symbols.update(s for s in protected_symbols if s in all_symbols)
            
            # Final symbol universe: use filtered + unfiltered priority symbols
            filtered_symbols = {s: obj for s, obj in all_symbols.items() if s in top_symbols}
            # Add priority symbols that passed best_pairs vetting but were dropped by pattern filter
            for s in top_symbols:
                if s not in filtered_symbols and s in all_symbols_unfiltered:
                    filtered_symbols[s] = all_symbols_unfiltered[s]
            print(f"✅ Filtered to {len(filtered_symbols)} symbols (from {len(all_symbols)}, blacklist: {len(blacklist)}, protected: {len(protected_symbols)})")
            all_symbols = filtered_symbols
        except Exception as e:
            print(f"⚠️ Symbol universe selection failed ({e}). Using all symbols.")
        _mark('volume_filter_done')
        
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
    
        # CRITICAL: Initialize pairs manager (loads DB state + reconciles with exchange)
        await pairs_manager.initialize()
        tg.attach_runtime(session, client, pairs_manager)
        _mark('pairs_manager_initialized')
    
        # 3. Start background symbol updates
        bg_tasks.append(loop.create_task(load_symbols_loop()))
        bg_tasks.append(loop.create_task(sync_exchange_time_loop()))
        bg_tasks.append(loop.create_task(ws_health_watchdog_loop()))
        
        bg_tasks.append(loop.create_task(connect_ws(timeframe)))
        _mark('background_tasks_started')
        _print_startup_profile('main_ready')
        
        # Run Telegram bot
        if tg_task:
            await tg_task
        else:
            await tg.run(session, client, pairs_manager)
    finally:
        for t in bg_tasks:
            if t and not t.done():
                t.cancel()
        if bg_tasks:
            await asyncio.gather(*bg_tasks, return_exceptions=True)
        try:
            await disconnect_ws()
        except Exception:
            pass
        try:
            if tg_task and not tg_task.done():
                tg_task.cancel()
                await asyncio.gather(tg_task, return_exceptions=True)
        except Exception:
            pass
        try:
            if client:
                await client.close()
        except Exception:
            pass
        try:
            if tg.bot and getattr(tg.bot, "session", None):
                await tg.bot.session.close()
        except Exception:
            pass

# Service to refresh all trading pairs every hour
async def load_symbols_loop():
    global all_symbols
    while True:
        try:
            await asyncio.sleep(3600)
            
            print("Refreshing market symbols...")
            new_symbols = await client.load_symbols()
            raw_refresh_count = len(new_symbols)
            new_symbols_unfiltered = dict(new_symbols)  # Keep unfiltered copy for priority lookups
            new_symbols = {s: obj for s, obj in new_symbols.items() if _is_tradeable_usdt_symbol_name(s)}
            if len(new_symbols) != raw_refresh_count:
                print(f"⏭️ Refresh dropped {raw_refresh_count - len(new_symbols)} invalid symbols.")
            
            # Apply symbol universe selection (same as initial load in main())
            conf = await db.load_config()
            max_symbols = int(conf.max_symbols) if conf.max_symbols else 150
            blacklist = {s.strip().upper() for s in (conf.blacklist or '').split(',') if s.strip()}
            best_pairs_only = _cfg_bool(getattr(conf, 'best_pairs_only', False), False)
            priority_file_path = _priority_file_path_from_config(conf)
            
            try:
                top_symbols = set()
                if best_pairs_only:
                    priority_symbols = _extract_priority_symbols_from_file(priority_file_path)
                    # Use unfiltered symbols — priority pairs are pre-vetted in backtest
                    top_symbols.update(s for s in priority_symbols if s in new_symbols_unfiltered and s not in blacklist)
                    print(
                        f"🎯 Refresh BEST-PAIRS-ONLY universe: {len(top_symbols)} symbols from priority file"
                        f"{f' ({priority_file_path})' if priority_file_path else ''}"
                    )
                else:
                    tickers = await client.ticker_24hr_price_change()
                    valid_tickers = []
                    for t in tickers:
                        sym = t.get('symbol', '')
                        if (
                            _is_tradeable_usdt_symbol_name(sym)
                            and sym not in blacklist
                            and sym in new_symbols
                        ):
                            try:
                                vol = float(t.get('quoteVolume', 0))
                                valid_tickers.append((sym, vol))
                            except Exception:
                                continue
                    valid_tickers.sort(key=lambda x: x[1], reverse=True)
                    top_symbols.update(sym for sym, vol in valid_tickers[:max_symbols])

                # Always keep BTCUSDT for market-beta calculations.
                if 'BTCUSDT' in new_symbols:
                    top_symbols.add('BTCUSDT')
                
                # SAFETY: Preserve symbols used by active/open positions
                protected_symbols = set()
                if pairs_manager:
                    for pair_info in pairs_manager.active_pairs.values():
                        if getattr(pair_info, 'position_status', 0) != 0 or getattr(pair_info, 'is_trading', False):
                            s1 = str(getattr(pair_info, 'symbol1', '') or '').strip().upper()
                            s2 = str(getattr(pair_info, 'symbol2', '') or '').strip().upper()
                            if _is_tradeable_usdt_symbol_name(s1):
                                protected_symbols.add(s1)
                            if _is_tradeable_usdt_symbol_name(s2):
                                protected_symbols.add(s2)
                try:
                    exchange_positions = await client.get_position_risk()
                    for pos in exchange_positions:
                        sym = str(pos.get('symbol', '') or '').strip().upper()
                        if abs(float(pos.get('positionAmt', 0))) > 0 and _is_tradeable_usdt_symbol_name(sym):
                            protected_symbols.add(sym)
                except Exception as e:
                    print(f"Could not refresh protected symbols from exchange: {e}")
                
                top_symbols.update(s for s in protected_symbols if s in new_symbols)
                filtered_symbols = {s: obj for s, obj in new_symbols.items() if s in top_symbols}
                # Add priority symbols that passed best_pairs vetting but were dropped by pattern filter
                for s in top_symbols:
                    if s not in filtered_symbols and s in new_symbols_unfiltered:
                        filtered_symbols[s] = new_symbols_unfiltered[s]
                print(f"✅ Refreshed {len(filtered_symbols)} symbols (from {len(new_symbols)}, blacklist: {len(blacklist)}, protected: {len(protected_symbols)})")
                new_symbols = filtered_symbols
            except Exception as e:
                print(f"⚠️ Symbol universe selection failed during refresh ({e}). Using all symbols.")
            
            # Update BOTH global and pairs_manager references
            all_symbols = new_symbols
            if pairs_manager:
                pairs_manager.all_symbols = new_symbols
                print(f"✅ pairs_manager.all_symbols updated ({len(new_symbols)} symbols)")
                # CRITICAL: refresh MAIN kline subscriptions to the new symbol universe.
                # Without this, bot can stay on stale streams until full WS reconnect.
                try:
                    refresh_ws_symbols = set(new_symbols.keys())
                    for pair_info in pairs_manager.active_pairs.values():
                        if getattr(pair_info, 'position_status', 0) != 0 or getattr(pair_info, 'is_trading', False):
                            refresh_ws_symbols.add(pair_info.symbol1)
                            refresh_ws_symbols.add(pair_info.symbol2)
                    if 'BTCUSDT' in all_symbols:
                        refresh_ws_symbols.add('BTCUSDT')
                    await _rebuild_main_kline_subscriptions(
                        sorted(refresh_ws_symbols),
                        _main_timeframe_global,
                        reason='symbol_refresh'
                    )
                except Exception as ws_refresh_err:
                    print(f"⚠️ Could not rebuild MAIN kline WS after refresh: {ws_refresh_err}")
                # Refresh path: warm up history for refreshed universe in background
                # and run a fast priority discovery without waiting for next candle close.
                try:
                    refresh_symbols = sorted(new_symbols.keys())
                    refresh_warmup_conc = int(getattr(conf, 'refresh_warmup_concurrency', 8) or 8)
                    if pairs_manager._warmup_task is None or pairs_manager._warmup_task.done():
                        pairs_manager.start_background_warmup(
                            refresh_symbols,
                            concurrency=refresh_warmup_conc,
                            run_discovery=False
                        )
                        print(
                            f"⚡ Refresh warmup scheduled for {len(refresh_symbols)} symbols "
                            f"(concurrency={refresh_warmup_conc}, discovery=off)."
                        )

                    async def _post_refresh_priority_discovery(wait_task):
                        try:
                            if wait_task is not None:
                                await wait_task
                        except Exception as warm_err:
                            print(f"⚠️ Refresh warmup failed before priority discovery: {warm_err}")
                        try:
                            if pairs_manager and (pairs_manager._discovery_task is None or pairs_manager._discovery_task.done()):
                                pairs_manager._last_discovery_time = time_mod.time()
                                pairs_manager._discovery_task = pairs_manager.loop.create_task(
                                    pairs_manager._discover_new_pairs(priority_only=True)
                                )
                                print("⚡ Post-refresh priority discovery scheduled.")
                        except Exception as disc_err:
                            print(f"⚠️ Could not schedule post-refresh priority discovery: {disc_err}")

                    pairs_manager.loop.create_task(_post_refresh_priority_discovery(pairs_manager._warmup_task))
                except Exception as refresh_bg_err:
                    print(f"⚠️ Could not schedule post-refresh warmup/discovery: {refresh_bg_err}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error loading symbols: {e}")
            traceback.print_exc()


async def sync_exchange_time_loop():
    """Periodic server time sync to prevent signed-request timestamp drift (-1021)."""
    while True:
        try:
            await asyncio.sleep(300)
            if client:
                await client._sync_time_async()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️ Time sync loop error: {e}")


def _timeframe_to_seconds(tf: str) -> int:
    tf = (tf or '1h').strip().lower()
    if tf.endswith('m'):
        try:
            return max(60, int(tf[:-1]) * 60)
        except Exception:
            return 3600
    if tf.endswith('h'):
        try:
            return max(3600, int(tf[:-1]) * 3600)
        except Exception:
            return 3600
    if tf.endswith('d'):
        try:
            return max(86400, int(tf[:-1]) * 86400)
        except Exception:
            return 86400
    return 3600


def _format_half_life_for_display(hl_bars: float, timeframe: str) -> str:
    try:
        hl_bars = float(hl_bars or 0.0)
    except Exception:
        hl_bars = 0.0
    if hl_bars <= 0:
        return 'N/A'
    hl_hours = hl_bars * (_timeframe_to_seconds(timeframe) / 3600.0)
    if hl_hours >= 24:
        hl_d = int(hl_hours // 24)
        hl_h = int(hl_hours % 24)
        return f"{hl_d}d {hl_h}h" if hl_h > 0 else f"{hl_d}d"
    hl_h = int(hl_hours)
    hl_m = int((hl_hours - hl_h) * 60)
    return f"{hl_h}h {hl_m}m" if hl_m > 0 else f"{hl_h}h"


async def _rebuild_main_kline_subscriptions(symbols: list[str], timeframe: str, reason: str = "runtime") -> bool:
    """
    Rebuild only MAIN timeframe kline subscriptions (discovery/validation stream).
    MarkPrice and user-data streams are managed separately.
    """
    global websockets_list
    global main_kline_wss
    global _ws_last_main_msg_ts
    global _main_kline_reload_lock

    if not client:
        return False

    if _main_kline_reload_lock is None:
        _main_kline_reload_lock = asyncio.Lock()

    async with _main_kline_reload_lock:
        requested = sorted({s for s in (symbols or []) if s in all_symbols})
        if not requested:
            print(f"⚠️ MAIN kline rebuild skipped ({reason}): no symbols.")
            return False

        streams = [f"{s.lower()}@kline_{timeframe}" for s in requested]
        kline_chunk_size = 80
        chunks = [streams[i:i + kline_chunk_size] for i in range(0, len(streams), kline_chunk_size)]

        new_main_wss: list[binance.futures.WebsocketAsync] = []
        try:
            for chunk in chunks:
                ws = await client.websocket(chunk, on_message=ws_msg_main, on_error=ws_error)
                new_main_wss.append(ws)
                await asyncio.sleep(0.05)
        except Exception as e:
            for ws in new_main_wss:
                try:
                    await ws.close()
                except Exception:
                    pass
            print(f"⚠️ MAIN kline rebuild failed ({reason}): {e}")
            return False

        old_main = list(main_kline_wss)
        main_kline_wss = new_main_wss

        for ws in old_main:
            try:
                await ws.close()
            except Exception:
                pass
            try:
                websockets_list.remove(ws)
            except ValueError:
                pass

        for ws in new_main_wss:
            if ws not in websockets_list:
                websockets_list.append(ws)

        if pairs_manager:
            pairs_manager._subscribed_main_symbols = set(requested)
        _ws_last_main_msg_ts = time_mod.time()
        print(
            f"✅ MAIN kline WS rebuilt ({reason}): {len(requested)} symbols, "
            f"{len(new_main_wss)} connections."
        )
        return True


async def _recover_ws_stack(reason: str):
    global _ws_last_recover_ts
    global _ws_last_main_msg_ts
    global _ws_last_mark_msg_ts
    global _ws_last_user_msg_ts
    global _ws_error_ts
    now = time_mod.time()
    # Anti-flap: don't restart too often.
    if now - _ws_last_recover_ts < 120:
        return
    if _ws_recover_lock is None:
        return
    async with _ws_recover_lock:
        now = time_mod.time()
        if now - _ws_last_recover_ts < 120:
            return
        _ws_last_recover_ts = now
        print(f"🛠️ WS watchdog recovery started: {reason}")
        try:
            await send_tg_notification(
                f"⚠️ <b>WS Watchdog</b>: reconnect storm/stale stream detected.\n"
                f"Reason: <code>{reason}</code>\n"
                f"Action: restarting WS stack automatically."
            )
        except Exception:
            pass
        try:
            await disconnect_ws()
            await asyncio.sleep(2)
            await connect_ws(_main_timeframe_global)
            now2 = time_mod.time()
            _ws_last_main_msg_ts = now2
            _ws_last_mark_msg_ts = now2
            _ws_last_user_msg_ts = now2
            _ws_error_ts.clear()
            await send_tg_notification("✅ <b>WS Watchdog</b>: WS stack restarted successfully.")
        except Exception as e:
            print(f"❌ WS watchdog recovery failed: {e}")
            try:
                await send_tg_notification(f"🚨 <b>WS Watchdog restart failed</b>: <code>{e}</code>")
            except Exception:
                pass


async def ws_health_watchdog_loop():
    """Autonomous WS health monitor with self-healing restart."""
    global _ws_last_main_msg_ts
    global _ws_last_mark_msg_ts
    global _ws_last_user_msg_ts
    while True:
        try:
            await asyncio.sleep(30)
            now = time_mod.time()

            # During startup, prime timestamps to avoid false positives.
            if _ws_last_main_msg_ts <= 0:
                _ws_last_main_msg_ts = now
            if _ws_last_mark_msg_ts <= 0:
                _ws_last_mark_msg_ts = now
            if _ws_last_user_msg_ts <= 0:
                _ws_last_user_msg_ts = now

            # Reconnect storm detector.
            recent_errors = [t for t in _ws_error_ts if now - t <= 180]
            if len(recent_errors) >= 12:
                await _recover_ws_stack(f"reconnect_storm:{len(recent_errors)}/180s")
                continue

            # Stale markPrice detector (critical for realtime entries/exits).
            # Only check when symbols are actually subscribed.
            mark_subscribed = 0
            if pairs_manager:
                mark_subscribed = len(getattr(pairs_manager, '_subscribed_mark_symbols', set()) or set())
            if mark_subscribed > 0 and (now - _ws_last_mark_msg_ts) > 180:
                await _recover_ws_stack(f"markprice_stale:{int(now - _ws_last_mark_msg_ts)}s")
                continue

            # Main kline detector.
            tf_sec = _timeframe_to_seconds(_main_timeframe_global)
            # Kline stream usually updates intra-candle, but allow a wide threshold.
            max_main_silence = min(max(300, tf_sec // 2), 1800)
            if (now - _ws_last_main_msg_ts) > max_main_silence:
                await _recover_ws_stack(f"main_kline_stale:{int(now - _ws_last_main_msg_ts)}s")
                continue
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️ WS watchdog loop error: {e}")


# Connect to websockets
async def connect_ws(timeframe='1h'):
    global websockets_list
    global main_kline_wss
    global userdata_ws
    global pairs_manager
    global _ws_last_main_msg_ts
    global _ws_last_mark_msg_ts
    global _ws_last_user_msg_ts

    ws_t0 = time_mod.perf_counter()
    print("Connecting to websockets...")
    now_ts = time_mod.time()
    _ws_last_main_msg_ts = now_ts
    _ws_last_mark_msg_ts = now_ts
    _ws_last_user_msg_ts = now_ts

    # COLLECT ALL USDT PAIRS FOR SCANNER
    target_symbols = []
    
    # Load config for blacklist
    conf = await db.load_config()
    t_cfg = time_mod.perf_counter()

    # Get blacklist from DB (which already contains defaults if initialized)
    FULL_BLACKLIST = set()
    if conf and conf.blacklist:
        FULL_BLACKLIST = set([s.strip().upper() for s in conf.blacklist.split(',') if s.strip()])

    # 1. All USDT pairs from market
    total_loaded_symbols = len(all_symbols)
    pass_filter1_count = 0
    for s_name, s_info in all_symbols.items():
        # Filter 1: Active and PERPETUAL only
        try:
            if getattr(s_info, 'contract_type', None) != 'PERPETUAL': continue
            if getattr(s_info, 'status', None) != 'TRADING': continue
            if getattr(s_info, 'quote_asset', None) != 'USDT': continue
            pass_filter1_count += 1
        except:
            if not s_name.endswith('USDT'): continue
            pass_filter1_count += 1

        # Filter 2: Strict symbol eligibility
        if not _is_tradeable_usdt_symbol_name(s_name):
            continue

        # Filter 3: Exclude blacklist
        if s_name in FULL_BLACKLIST:
            continue

        target_symbols.append(s_name)
    
    print(f"Filter #1 (PERPETUAL+TRADING+USDT): {pass_filter1_count} / {total_loaded_symbols}")
    target_symbols.sort()
    print(f"Subscribing to {len(target_symbols)} high-quality symbols (Filtered for PERPETUAL USDT-M).")

    # 2. Add symbols from DB (ONLY pairs with open positions)
    db_pairs = await db.get_all_pairs()
    t_db_pairs = time_mod.perf_counter()
    active_db_count = 0
    for p in db_pairs:
        if p.position_status != 0:  # Only add if position is open
            s1 = str(getattr(p, 'symbol1', '') or '').strip().upper()
            s2 = str(getattr(p, 'symbol2', '') or '').strip().upper()
            if s1 in all_symbols and _is_tradeable_usdt_symbol_name(s1) and s1 not in target_symbols:
                target_symbols.append(s1)
            if s2 in all_symbols and _is_tradeable_usdt_symbol_name(s2) and s2 not in target_symbols:
                target_symbols.append(s2)
            active_db_count += 1

    # Startup audit: prove websocket/warmup universe contains only eligible symbols.
    invalid_target_symbols = [s for s in target_symbols if not _is_tradeable_usdt_symbol_name(s)]
    if invalid_target_symbols:
        sample = ", ".join(sorted(set(invalid_target_symbols))[:10])
        print(
            f"⚠️ Symbol universe audit failed: {len(invalid_target_symbols)} invalid symbols in target list. "
            f"Sample: {sample}"
        )
    else:
        print(f"✅ Symbol universe audit passed: {len(target_symbols)} symbols are valid.")
            
    print(f"Subscribing to {len(target_symbols)} symbols (Filtered: {len(target_symbols) - active_db_count * 2}, DB active: {active_db_count} pairs)...")

    # Quick startup: preload only critical symbols (open pairs + BTC), full warmup in background
    if pairs_manager:
        critical_symbols = set()
        for pair_info in pairs_manager.active_pairs.values():
            if getattr(pair_info, 'position_status', 0) != 0:
                critical_symbols.add(pair_info.symbol1)
                critical_symbols.add(pair_info.symbol2)
        if 'BTCUSDT' in target_symbols:
            critical_symbols.add('BTCUSDT')
        if critical_symbols:
            await pairs_manager.initialize_all_symbols_data(
                sorted(critical_symbols),
                concurrency=10,
                run_discovery=False
            )

    print(f"Single TF Mode: {timeframe}")

    # MAIN TIMEFRAME: for discovery (cointegration tests)
    await _rebuild_main_kline_subscriptions(target_symbols, timeframe, reason='startup')
    t_main_ws = time_mod.perf_counter()
    print(f"Connected to main TF kline websockets ({len(main_kline_wss)} connections).")

    # Userdata websocket
    try:
        userdata_ws = await client.websocket_userdata(on_message=ws_user_msg, on_error=ws_error)
        print("Connected to userdata websocket.")
    except Exception as e:
        print(f"Could not connect to userdata stream: {e}")
    t_user_ws = time_mod.perf_counter()
    
    # Start real-time signal confirmation loop
    if pairs_manager:
        pairs_manager.start_realtime_monitoring()
        
        # Subscribe to markPrice stream for real-time price updates (for active pairs only)
        # Get symbols from active pairs
        active_symbols = set()
        for pair_info in pairs_manager.active_pairs.values():
            active_symbols.add(pair_info.symbol1)
            active_symbols.add(pair_info.symbol2)
        # Always subscribe BTCUSDT for Market Shock Protector logic.
        active_symbols.add('BTCUSDT')
        
        # Define markPrice handler FIRST (before any subscription)
        async def ws_mark_price(ws, msg):
            """Handle markPrice updates for real-time Z-score."""
            global _ws_last_mark_msg_ts
            if 'data' not in msg:
                return
            data = msg['data']
            symbol = data.get('s')
            price = float(data.get('p', 0))
            if symbol and price > 0 and pairs_manager:
                _ws_last_mark_msg_ts = time_mod.time()
                await pairs_manager.on_ticker_update(symbol, price)
        
        # Track already subscribed symbols in pairs_manager
        pairs_manager._subscribed_mark_symbols = set(active_symbols)
        mark_subscribe_lock = asyncio.Lock()
        dynamic_mark_symbols: set[str] = set()
        dynamic_mark_wss: list[binance.futures.WebsocketAsync] = []
        mark_chunk_size = 35
        mark_max_symbols = int(getattr(conf, 'markprice_max_symbols', 120) or 120)
        
        # Create callback for dynamic subscription (used when new pairs are discovered)
        async def subscribe_new_marks(symbols):
            """Rebuild dynamic markPrice subscriptions without leaking websocket connections."""
            async with mark_subscribe_lock:
                nonlocal dynamic_mark_symbols
                requested_symbols = set(symbols or [])
                protected_symbols = {'BTCUSDT'}
                if pairs_manager:
                    for pi in pairs_manager.active_pairs.values():
                        if getattr(pi, 'position_status', 0) != 0:
                            protected_symbols.add(pi.symbol1)
                            protected_symbols.add(pi.symbol2)
                # Keep already tracked idle symbols; otherwise each "new pair" update
                # would replace the full realtime universe with only that pair.
                desired_symbols = set(dynamic_mark_symbols) | requested_symbols | protected_symbols
                if len(desired_symbols) > mark_max_symbols:
                    protected_sorted = sorted(desired_symbols & protected_symbols)
                    requested_sorted = sorted((desired_symbols & requested_symbols) - set(protected_sorted))
                    existing_sorted = sorted(desired_symbols - set(protected_sorted) - set(requested_sorted))
                    allowed_others = max(0, mark_max_symbols - len(protected_sorted))
                    desired_symbols = set(protected_sorted + requested_sorted[:allowed_others])
                    if len(desired_symbols) < mark_max_symbols:
                        remaining = mark_max_symbols - len(desired_symbols)
                        desired_symbols.update(existing_sorted[:remaining])
                if desired_symbols == dynamic_mark_symbols:
                    return
                dynamic_mark_symbols = desired_symbols
                pairs_manager._subscribed_mark_symbols = set(desired_symbols)
                streams = [f"{s.lower()}@markPrice@1s" for s in sorted(desired_symbols)]
                chunks = [streams[i:i + mark_chunk_size] for i in range(0, len(streams), mark_chunk_size)]
                old_wss = list(dynamic_mark_wss)
                dynamic_mark_wss.clear()
                try:
                    for ws in old_wss:
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        try:
                            websockets_list.remove(ws)
                        except ValueError:
                            pass
                    for marks in chunks:
                        ws = await client.websocket(marks, on_message=ws_mark_price, on_error=ws_error)
                        dynamic_mark_wss.append(ws)
                        websockets_list.append(ws)
                except Exception as e:
                    print(f"⚠️ Failed to rebuild markPrice streams (requested={len(requested_symbols)}, subscribed={len(desired_symbols)}): {e}")
        
        # Set the callback on pairs_manager so it can subscribe new pairs
        pairs_manager._subscribe_mark_callback = subscribe_new_marks
        
        # Subscribe to initial symbols
        if active_symbols:
            await subscribe_new_marks(sorted(active_symbols))
            print(f"Connected to markPrice websocket (requested={len(active_symbols)}, cap={mark_max_symbols}).")
        else:
            print("ℹ️ No active pairs at startup - markPrice will be subscribed dynamically.")
        
        # Heavy warmup+discovery is moved to background to avoid blocking startup
        pairs_manager.start_background_warmup(target_symbols, concurrency=20)
    t_done = time_mod.perf_counter()
    print(
        f"WS startup profile: "
        f"config={t_cfg - ws_t0:.2f}s, "
        f"db_pairs={t_db_pairs - t_cfg:.2f}s, "
        f"main_ws={t_main_ws - t_db_pairs:.2f}s, "
        f"userdata_ws={t_user_ws - t_main_ws:.2f}s, "
        f"tail={t_done - t_user_ws:.2f}s, "
        f"total={t_done - ws_t0:.2f}s"
    )


# Disconnect from websockets
async def disconnect_ws():
    global websockets_list
    global main_kline_wss
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
    websockets_list.clear()
    main_kline_wss.clear()
    userdata_ws = None


# Handle websocket errors
async def ws_error(ws, error):
    global _ws_error_ts
    # Network hiccups are expected; avoid noisy full tracebacks on every reconnect.
    if isinstance(error, (asyncio.TimeoutError, TimeoutError, aiohttp.ClientError, ConnectionError)):
        _ws_error_ts.append(time_mod.time())
        err_txt = str(error)
        if 'wss://' in err_txt and 'streams=' in err_txt:
            try:
                start = err_txt.find('wss://')
                url = err_txt[start:].split(' ', 1)[0]
                parsed = urlsplit(url)
                q = parse_qs(parsed.query)
                streams_raw = q.get('streams', [''])[0]
                streams_cnt = len([s for s in streams_raw.split('/') if s]) if streams_raw else 0
                err_txt = f"{parsed.scheme}://{parsed.netloc}{parsed.path} (streams={streams_cnt})"
            except Exception:
                if len(err_txt) > 240:
                    err_txt = err_txt[:240] + '...'
        elif len(err_txt) > 240:
            err_txt = err_txt[:240] + '...'
        print(f"WS reconnect: {type(error).__name__}: {err_txt}")
        return
    # Aiohttp WS error payload may arrive as plain text-like object.
    err_txt = str(error)
    if 'No PONG received' in err_txt:
        _ws_error_ts.append(time_mod.time())
        print(f"WS reconnect: ServerTimeoutError: {err_txt}")
        return
    print(f"WS ERROR: {error}")


# Handle MAIN timeframe kline messages (for discovery + validation)
async def ws_msg_main(ws, msg):
    global _ws_last_main_msg_ts
    if 'data' not in msg:
        return
    _ws_last_main_msg_ts = time_mod.time()
    
    kline = msg['data']['k']
    
    # Only process on candle close (discovery needs complete candles)
    if kline['x']:
        await pairs_manager.add_kline_main(kline)



# Handle userdata messages
async def ws_user_msg(ws, msg):
    """Handle userdata messages including order updates and position changes."""
    global pairs_manager
    global _ws_last_user_msg_ts
    _ws_last_user_msg_ts = time_mod.time()
    
    event_type = msg.get('e')
    
    # DEBUG: Log all userdata events
    if event_type in ('ACCOUNT_UPDATE', 'ORDER_TRADE_UPDATE', 'ALGO_UPDATE'):
        print(f"📡 UserData WS: {event_type} received")
    
    # Check for ACCOUNT_UPDATE (position changes - including manual closes)
    if event_type == 'ACCOUNT_UPDATE':
        global recently_handled_close_symbols
        positions = msg.get('a', {}).get('P', [])
        print(f"📡 ACCOUNT_UPDATE: {len(positions)} positions in update")

        # Cleanup stale suppression entries.
        now_ts_global = time_mod.time()
        recently_handled_close_symbols = {
            s: ts for s, ts in recently_handled_close_symbols.items() if now_ts_global - ts < 180
        }

        # Snapshot of known-open symbols BEFORE applying this update.
        # Used to avoid false "UNTRACKED POSITION CLOSED" alerts for symbols that
        # were never tracked as open by the bot.
        known_open_before = set()
        if pairs_manager:
            known_open_before.update((pairs_manager._exchange_positions_cache or {}).keys())
            known_open_before.update((pairs_manager._exchange_pnl_cache or {}).keys())
        
        # Notify about ALL position changes immediately
        for pos in positions:
            sym = pos.get('s')
            amt = float(pos.get('pa', 0))
            up = float(pos.get('up', 0))  # unrealizedProfit from exchange
            if amt == 0:
                # Clear PnL cache for closed position
                if pairs_manager:
                    pairs_manager._exchange_pnl_cache.pop(sym, None)
                    pairs_manager._exchange_positions_cache.pop(sym, None)
            else:
                # Update PnL cache in real-time from WebSocket (instant, no API call)
                if pairs_manager:
                    pairs_manager._exchange_pnl_cache[sym] = up
                    pairs_manager._exchange_positions_cache[sym] = abs(amt)

        # Keep cached exchange position counter in sync with websocket position cache.
        # Otherwise can_open_new_position() may see stale "limit reached" state.
        if pairs_manager:
            pairs_manager._exchange_position_count = len(pairs_manager._exchange_positions_cache)
        
        # Note: Detailed notifications will be sent per-pair below
        # Skip simple "Position Changes" message - too noisy
        for pos in positions:
            symbol = pos.get('s')
            position_amt = float(pos.get('pa', 0))
            if position_amt == 0:
                continue
            if symbol in known_open_before:
                continue
            if pairs_manager and any(
                symbol in (pi.symbol1, pi.symbol2) and (getattr(pi, 'position_status', 0) != 0 or getattr(pi, 'is_trading', False))
                for pi in pairs_manager.active_pairs.values()
            ):
                continue
            last_open_alert = untracked_open_alerts.get(symbol, 0)
            now_open_ts = time_mod.time()
            if now_open_ts - last_open_alert < 600:
                continue
            untracked_open_alerts[symbol] = now_open_ts
            side = 'LONG' if position_amt > 0 else 'SHORT'
            up = float(pos.get('up', 0) or 0.0)
            try:
                entry_price = float(pos.get('ep', 0) or 0.0)
            except Exception:
                entry_price = 0.0
            open_msg = (
                f"⚡ <b>UNTRACKED POSITION OPENED</b>\n\n"
                f"Symbol: <b>{symbol}</b>\n"
                f"Side: <b>{side}</b>\n"
                f"Qty: <b>{abs(position_amt)}</b>\n"
                + (f"Entry: <b>{entry_price}</b>\n" if entry_price > 0 else "")
                + f"Unrealized PnL: <b>{up:+.2f} USDT</b>\n"
                f"Source: Exchange websocket reported a position not tracked by bot active_pairs."
            )
            await send_tg_notification(open_msg)
        
        processed_pairs = set()  # Track pairs we've already handled in this update
        
        # PHASE 1: Pre-mark ALL affected pairs to prevent leg sync from racing
        # Collect pairs to process BEFORE doing any async work
        pairs_to_process = []  # List of (pair_set, pair_info, symbol, other_symbol, other_closed_in_batch)
        
        for pos in positions:
            symbol = pos.get('s')
            position_amt = float(pos.get('pa', 0))
            
            if position_amt == 0 and pairs_manager:
                for pair_set, pair_info in list(pairs_manager.active_pairs.items()):
                    if pair_set in processed_pairs:
                        continue
                    
                    if pair_info.position_status != 0 and symbol in [pair_info.symbol1, pair_info.symbol2]:
                        s1, s2 = pair_info.symbol1, pair_info.symbol2
                        other_symbol = s2 if symbol == s1 else s1
                        
                        
                        other_closed_in_batch = any(
                            p.get('s') == other_symbol and float(p.get('pa', 0)) == 0 
                            for p in positions
                        )
                        
                        processed_pairs.add(pair_set)
                        # CRITICAL: Set is_trading NOW to prevent leg sync from racing
                        pair_info.is_trading = True
                        pair_info._is_trading_since = time_mod.time()
                        
                        pairs_to_process.append((pair_set, pair_info, symbol, other_symbol, other_closed_in_batch))
                        break  # Found the pair for this symbol
        
        # PHASE 2: Process each pair (now safe from leg sync races)
        for pair_set, pair_info, symbol, other_symbol, other_closed_in_batch in pairs_to_process:
            s1, s2 = pair_info.symbol1, pair_info.symbol2
            
            if other_closed_in_batch:
                # Check if bot already handled this close AND sent its own notification
                stored_reason = getattr(pair_info, 'last_close_reason', '')
                bot_close_reasons = ('manual', 'z_tp', 'z_sl', 'circuit', 'broken_coint', 
                                     'hardware_sl', 'hardware_tp', 'beta_drift', 'beta_critical',
                                     'btc_shock', 'desync', 'orphan_restart', 'stale_symbols', 'audit_fail')
                if getattr(pair_info, 'close_handled', False) and stored_reason in bot_close_reasons:
                    print(f"ℹ️ {s1}-{s2} close already handled by bot (reason: {stored_reason}), skipping external notification")
                    now_mark = time_mod.time()
                    recently_handled_close_symbols[s1] = now_mark
                    recently_handled_close_symbols[s2] = now_mark
                    pair_info.close_handled = False  # Reset for next trade
                    pair_info.is_trading = False
                    continue
                
                # Both legs closed together - fetch actual PnL and cleanup
                print(f"⚡ Both legs of {s1}-{s2} closed externally. Fetching PnL...")
                try:
                    await asyncio.gather(
                        client.cancel_open_orders(s1),
                        client.cancel_open_orders(s2),
                        return_exceptions=True
                    )
                    
                    # Small delay to ensure trade data is available
                    await asyncio.sleep(1)
                    
                    # Fetch actual PnL from recent trades
                    now_ms = int(time_mod.time() * 1000)
                    open_time = int(getattr(pair_info, 'open_time', 0) or 0)
                    start_ms = (max(0, open_time - 120) * 1000) if open_time > 0 else (now_ms - 300_000)
                    
                    trades1, trades2 = await asyncio.gather(
                        _fetch_account_trades_window_main(s1, start_ms, max_records=3000),
                        _fetch_account_trades_window_main(s2, start_ms, max_records=3000)
                    )
                    
                    print(f"📊 Trades for {s1}: {len(trades1)} entries")
                    print(f"📊 Trades for {s2}: {len(trades2)} entries")
                    
                    pnl1 = sum(float(t.get('realizedPnl', 0)) for t in trades1)
                    pnl2 = sum(float(t.get('realizedPnl', 0)) for t in trades2)
                    fee1 = sum(float(t.get('commission', 0)) for t in trades1)
                    fee2 = sum(float(t.get('commission', 0)) for t in trades2)
                    close_price1 = float(trades1[-1].get('price', 0)) if trades1 else 0.0
                    close_price2 = float(trades2[-1].get('price', 0)) if trades2 else 0.0
                    total_pnl = pnl1 + pnl2
                    total_fees = fee1 + fee2
                    net_pnl = total_pnl - total_fees
                    
                    # Determine close reason
                    close_type = '❓ Unknown'
                    close_hint = '\n💡 Check exchange for details'
                    
                    stored_reason = getattr(pair_info, 'last_close_reason', '')
                    if stored_reason and stored_reason in CLOSE_REASONS:
                        close_type = CLOSE_REASONS[stored_reason]
                        close_hint = ''
                        print(f"📋 Using stored reason: {stored_reason} -> {close_type}")
                    else:
                        try:
                            orders1, orders2 = await asyncio.gather(
                                client.get_all_orders(symbol=s1, limit=15),
                                client.get_all_orders(symbol=s2, limit=15)
                            )
                            
                            now_ms = int(time_mod.time() * 1000)
                            recent_orders = []
                            for o in orders1 + orders2:
                                if o.get('status') == 'FILLED' and o.get('updateTime', 0) > now_ms - 300_000:
                                    recent_orders.append(o)
                            
                            if recent_orders:
                                recent_orders.sort(key=lambda x: x.get('updateTime', 0), reverse=True)
                                o = recent_orders[0]
                                o_type = str(o.get('type') or o.get('origType') or '')
                                
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
                                print(f"📋 Detected: {o_type} -> {close_type}")
                            else:
                                close_type = '⚡ External Close'
                                close_hint = ' (no orders found)'
                                print(f"⚠️ No orders for {s1}/{s2}")
                        except Exception as e:
                            print(f"⚠️ Query error: {e}")
                            close_type = '⚡ External'
                    
                    pnl_emoji = '🟢' if net_pnl >= 0 else '🔴'
                    try:
                        p1 = pairs_manager.last_prices.get(s1, 0)
                        p2 = pairs_manager.last_prices.get(s2, 0)
                        if p1 > 0 and p2 > 0:
                            zscore = pairs_manager._calc_realtime_zscore(pair_info, p1, p2)
                            if math.isnan(zscore):
                                zscore = getattr(pair_info, 'last_z_score', 0) or 0
                        else:
                            zscore = getattr(pair_info, 'last_z_score', 0) or 0
                    except Exception:
                        zscore = getattr(pair_info, 'last_z_score', 0) or 0
                    beta = getattr(pair_info, 'beta_btc', 0) or 0
                    close_pval = getattr(pair_info, 'last_pvalue', 0) or 0
                    hl = getattr(pair_info, 'half_life', 0) or 0
                    close_hl = _format_half_life_for_display(hl, _main_timeframe_global)
                    hedge = getattr(pair_info, 'hedge_ratio', 0) or 0
                    e1 = '🟢' if pnl1 >= 0 else '🔴'
                    e2 = '🟢' if pnl2 >= 0 else '🔴'
                    close_tag = (stored_reason or 'external').strip().lower()
                    done_msg = (f"{close_type}: <b>{s1}/{s2}</b>\n"
                                f"🏷️ Tag: <code>{close_tag}</code>\n\n"
                                f"📊 Z: {zscore:+.2f} | β: {beta:.3f} | p: {close_pval:.4f}\n"
                                f"⏳ HL: {close_hl} | Hedge: {hedge:.4f}\n"
                                f"💵 PnL: {pnl_emoji} <b>{net_pnl:+.2f} USDT</b>\n"
                                f"   {e1} {s1}: {pnl1:+.2f} | {e2} {s2}: {pnl2:+.2f}\n"
                                f"💸 Fees: {total_fees:.4f} USDT{close_hint}")
                    
                    reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                    await send_tg_notification(done_msg, reply_to)
                    now_mark = time_mod.time()
                    recently_handled_close_symbols[s1] = now_mark
                    recently_handled_close_symbols[s2] = now_mark
                    
                    # Update memory state
                    pair_info.position_status = 0
                    pair_info.qty1 = 0
                    pair_info.qty2 = 0
                    pair_info.is_trading = False
                    if pairs_manager:
                        pairs_manager.mark_pair_wait_for_next_candle(pair_info, reason=stored_reason if stored_reason else 'external')
                    if pairs_manager:
                        pairs_manager.loop.create_task(pairs_manager._trigger_immediate_analysis())
                    
                    # Update DB
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
                    if pair_info.current_trade_id:
                        try:
                            await db.close_trade_record(
                                pair_info.current_trade_id,
                                status='CLOSED_EXTERNAL',
                                close_reason=stored_reason if stored_reason else 'external',
                                pnl=net_pnl,
                                fee1=fee1,
                                fee2=fee2,
                                close_z=zscore if zscore else 0.0,
                                close_price_1=close_price1 if close_price1 > 0 else None,
                                close_price_2=close_price2 if close_price2 > 0 else None,
                            )
                            await _persist_pair_executions_main(pair_info, trades1, trades2, phase='EXTERNAL_CLOSE_WS')
                        except Exception as trade_err:
                            print(f"⚠️ Trade update failed for {s1}-{s2}: {trade_err}")
                        pair_info.current_trade_id = None
                except Exception as e:
                    print(f"⚠️ Cleanup error: {e}")
                    import traceback
                    traceback.print_exc()
                    pair_info.is_trading = False
            else:
                # Only one leg closed - check if bot is already handling this
                stored_reason = getattr(pair_info, 'last_close_reason', '')
                if getattr(pair_info, 'close_handled', False) and stored_reason in ('manual', 'z_tp', 'z_sl', 'circuit', 'broken_coint', 
                        'hardware_sl', 'hardware_tp', 'beta_drift', 'beta_critical',
                        'btc_shock', 'desync', 'orphan_restart', 'stale_symbols', 'audit_fail'):
                    print(f"ℹ️ {s1}-{s2} close already handled by bot (reason: {stored_reason}), skipping single-leg handler")
                    now_mark = time_mod.time()
                    recently_handled_close_symbols[s1] = now_mark
                    recently_handled_close_symbols[s2] = now_mark
                    pair_info.is_trading = False
                    continue
                
                # External close - user manually closed one position
                print(f"⚡ External close detected: {symbol} in pair {s1}-{s2}. Closing {other_symbol} IMMEDIATELY...")
                pair_info.last_close_reason = 'manual_partial'
                try:
                    close_exec_note = ""

                    # Query exchange first: stored qty can be stale during rapid external closes.
                    positions_data = await client.get_position_risk(symbol=other_symbol)
                    other_pos = positions_data[0] if positions_data else {}
                    other_amt = float(other_pos.get('positionAmt', 0))

                    if other_amt != 0:
                        close_side = 'SELL' if other_amt > 0 else 'BUY'
                        try:
                            await asyncio.wait_for(
                                client.new_order(
                                    symbol=other_symbol,
                                    side=close_side,
                                    type='MARKET',
                                    quantity=abs(other_amt),
                                    reduceOnly='true'
                                ),
                                timeout=15
                            )
                            print(f"✅ Closed remaining leg {other_symbol} (qty={abs(other_amt)}, side={close_side}, reduceOnly=true)")
                        except Exception as close_err:
                            err_txt = str(close_err)
                            if "-2022" in err_txt or "ReduceOnly Order is rejected" in err_txt:
                                # Typical race: leg may already be zero. Re-check before fallback.
                                verify_data = await client.get_position_risk(symbol=other_symbol)
                                verify_pos = verify_data[0] if verify_data else {}
                                verify_amt = float(verify_pos.get('positionAmt', 0))
                                if verify_amt == 0:
                                    pair_info.last_close_reason = 'external'
                                    close_exec_note = "ℹ️ Remaining leg was already closed on exchange."
                                    print(f"ℹ️ {other_symbol} already closed after reduceOnly reject (-2022).")
                                else:
                                    verify_side = 'SELL' if verify_amt > 0 else 'BUY'
                                    await asyncio.wait_for(
                                        client.new_order(
                                            symbol=other_symbol,
                                            side=verify_side,
                                            type='MARKET',
                                            quantity=abs(verify_amt),
                                            reduceOnly='true'
                                        ),
                                        timeout=15
                                    )
                                    close_exec_note = "ℹ️ reduceOnly rejected, closed with fallback MARKET order."
                                    print(f"⚠️ reduceOnly rejected for {other_symbol}; fallback MARKET close succeeded.")
                            else:
                                close_exec_note = f"⚠️ Could not close remaining leg: {close_err}"
                                print(f"⚠️ Failed to close remaining leg {other_symbol}: {close_err}")
                    else:
                        pair_info.last_close_reason = 'external'
                        close_exec_note = "ℹ️ Remaining leg was already closed on exchange."
                        print(f"ℹ️ Remaining leg {other_symbol} already at zero position.")
                    
                    # THEN cancel remaining algo/SL/TP orders (non-critical, can be slower)
                    try:
                        await asyncio.gather(
                            client.cancel_open_orders(s1),
                            client.cancel_open_orders(s2),
                            return_exceptions=True
                        )
                    except Exception as cancel_err:
                        print(f"⚠️ Cancel orders error (non-critical): {cancel_err}")
                    
                    await asyncio.sleep(1)
                    
                    # Verify remaining leg really closed before finalizing pair state.
                    verify_data = await client.get_position_risk(symbol=other_symbol)
                    verify_pos = verify_data[0] if verify_data else {}
                    remaining_amt = float(verify_pos.get('positionAmt', 0))
                    if remaining_amt != 0:
                        warn_msg = (f"🚨 <b>External close handling incomplete</b>\n\n"
                                    f"Pair: {s1}/{s2}\n"
                                    f"Closed leg: {symbol}\n"
                                    f"Remaining leg still OPEN: {other_symbol}\n"
                                    f"Qty: <b>{abs(remaining_amt):.8f}</b>\n\n"
                                    f"Please close {other_symbol} manually.")
                        await send_tg_notification(warn_msg, pair_info.tg_message_id if pair_info.tg_message_id else None)
                        pair_info.is_trading = False
                        continue

                    now_ms = int(time_mod.time() * 1000)
                    open_time = int(getattr(pair_info, 'open_time', 0) or 0)
                    start_ms = (max(0, open_time - 120) * 1000) if open_time > 0 else (now_ms - 300_000)
                    
                    trades1, trades2 = await asyncio.gather(
                        _fetch_account_trades_window_main(s1, start_ms, max_records=3000),
                        _fetch_account_trades_window_main(s2, start_ms, max_records=3000)
                    )
                    
                    print(f"📊 Trades for {s1}: {len(trades1)} entries")
                    print(f"📊 Trades for {s2}: {len(trades2)} entries")
                    
                    pnl1 = sum(float(t.get('realizedPnl', 0)) for t in trades1)
                    pnl2 = sum(float(t.get('realizedPnl', 0)) for t in trades2)
                    fee1 = sum(float(t.get('commission', 0)) for t in trades1)
                    fee2 = sum(float(t.get('commission', 0)) for t in trades2)
                    close_price1 = float(trades1[-1].get('price', 0)) if trades1 else 0.0
                    close_price2 = float(trades2[-1].get('price', 0)) if trades2 else 0.0
                    total_pnl = pnl1 + pnl2
                    total_fees = fee1 + fee2
                    net_pnl = total_pnl - total_fees
                    
                    pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"
                    
                    # Update memory state
                    pair_info.position_status = 0
                    pair_info.qty1 = 0
                    pair_info.qty2 = 0
                    pair_info.is_trading = False
                    stored_reason = getattr(pair_info, 'last_close_reason', '')
                    if pairs_manager:
                        pairs_manager.mark_pair_wait_for_next_candle(pair_info, reason=stored_reason if stored_reason else 'external')
                    if pairs_manager:
                        pairs_manager.loop.create_task(pairs_manager._trigger_immediate_analysis())

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
                    if pair_info.current_trade_id:
                        try:
                            await db.close_trade_record(
                                pair_info.current_trade_id,
                                status='CLOSED_EXTERNAL',
                                close_reason=stored_reason if stored_reason else 'external',
                                pnl=net_pnl,
                                fee1=fee1,
                                fee2=fee2,
                                close_z=pair_info.last_z_score if pair_info.last_z_score else 0.0,
                                close_price_1=close_price1 if close_price1 > 0 else None,
                                close_price_2=close_price2 if close_price2 > 0 else None,
                            )
                            await _persist_pair_executions_main(pair_info, trades1, trades2, phase='MANUAL_PARTIAL_CLOSE_WS')
                        except Exception as trade_err:
                            print(f"⚠️ Trade update failed for {s1}-{s2}: {trade_err}")
                        pair_info.current_trade_id = None
                    
                    close_type = '❓ Unknown'
                    close_hint = ''
                    if close_exec_note:
                        close_hint = f"\n{close_exec_note}"
                    
                    if stored_reason and stored_reason in CLOSE_REASONS:
                        close_type = CLOSE_REASONS[stored_reason]
                        print(f"📋 Using stored reason: {stored_reason} -> {close_type}")
                    else:
                        try:
                            orders1, orders2 = await asyncio.gather(
                                client.get_all_orders(symbol=s1, limit=15),
                                client.get_all_orders(symbol=s2, limit=15)
                            )
                            
                            now_time = int(time_mod.time() * 1000)
                            recent_orders = []
                            for o in orders1 + orders2:
                                if o.get('status') == 'FILLED' and o.get('updateTime', 0) > now_time - 300_000:
                                     recent_orders.append(o)
                            
                            if recent_orders:
                                recent_orders.sort(key=lambda x: x.get('updateTime', 0), reverse=True)
                                o = recent_orders[0]
                                o_type = str(o.get('type') or o.get('origType') or '')
                                
                                if 'STOP' in o_type:
                                    close_type = '🛡️ Hardware SL'
                                elif 'TAKE_PROFIT' in o_type:
                                    close_type = '🛡️ Hardware TP'
                                elif o_type == 'MARKET':
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
                                print(f"📋 Detected from orders: {o_type} -> {close_type}")
                            else:
                                close_type = '⚡ External Close'
                                close_hint += ' (no matching orders)'
                                print(f"⚠️ No recent orders found for {s1}/{s2}")
                        except Exception as e:
                            print(f"⚠️ Could not query orders: {e}")
                            close_type = '⚡ External Close'
                            close_hint += ' (query failed)'
                    
                    pnl_emoji = '🟢' if net_pnl >= 0 else '🔴'
                    try:
                        p1 = pairs_manager.last_prices.get(s1, 0)
                        p2 = pairs_manager.last_prices.get(s2, 0)
                        if p1 > 0 and p2 > 0:
                            zscore = pairs_manager._calc_realtime_zscore(pair_info, p1, p2)
                            if math.isnan(zscore):
                                zscore = getattr(pair_info, 'last_z_score', 0) or 0
                        else:
                            zscore = getattr(pair_info, 'last_z_score', 0) or 0
                    except Exception:
                        zscore = getattr(pair_info, 'last_z_score', 0) or 0
                    beta = getattr(pair_info, 'beta_btc', 0) or 0
                    close_pval = getattr(pair_info, 'last_pvalue', 0) or 0
                    hl = getattr(pair_info, 'half_life', 0) or 0
                    close_hl = _format_half_life_for_display(hl, _main_timeframe_global)
                    hedge = getattr(pair_info, 'hedge_ratio', 0) or 0
                    e1 = '🟢' if pnl1 >= 0 else '🔴'
                    e2 = '🟢' if pnl2 >= 0 else '🔴'
                    close_tag = (stored_reason or 'external').strip().lower()
                    done_msg = (f"{close_type}: <b>{s1}/{s2}</b>\n"
                                f"🏷️ Tag: <code>{close_tag}</code>\n\n"
                                f"📊 Z: {zscore:+.2f} | β: {beta:.3f} | p: {close_pval:.4f}\n"
                                f"⏳ HL: {close_hl} | Hedge: {hedge:.4f}\n"
                                f"💵 PnL: {pnl_emoji} <b>{net_pnl:+.2f} USDT</b>\n"
                                f"   {e1} {s1}: {pnl1:+.2f} | {e2} {s2}: {pnl2:+.2f}\n"
                                f"💸 Fees: {total_fees:.4f} USDT{close_hint}")
                    reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                    await send_tg_notification(done_msg, reply_to)
                    now_mark = time_mod.time()
                    recently_handled_close_symbols[s1] = now_mark
                    recently_handled_close_symbols[s2] = now_mark
                    
                except Exception as e:
                    print(f"⚠️ External close handling error for {s1}-{s2}: {e}")
                    import traceback
                    traceback.print_exc()
                    pair_info.is_trading = False

        # PHASE 3: Fallback notifications for untracked position closes
        # This handles the case where DB failed to load but exchange positions exist
        # Build set of ALL symbols handled in Phase 2 (both the manually closed + bot-closed orphan)
        handled_symbols = set()
        for _, pinfo, sym, other_sym, _ in pairs_to_process:
            handled_symbols.add(sym)
            handled_symbols.add(other_sym)
        
        for pos in positions:
            symbol = pos.get('s')
            position_amt = float(pos.get('pa', 0))
            if position_amt == 0:
                was_handled = symbol in handled_symbols
                if not was_handled:
                    # If symbol belongs to any known pair (even already closed in this cycle),
                    # it's not truly "untracked" and should not trigger fallback alert.
                    if pairs_manager and any(symbol in (pi.symbol1, pi.symbol2) for pi in pairs_manager.active_pairs.values()):
                        continue

                    # Skip if symbol was just handled by pair-close logic in the last 3 minutes.
                    last_handled = recently_handled_close_symbols.get(symbol, 0)
                    if time_mod.time() - last_handled < 180:
                        continue

                    # Only alert if this symbol was known as open before the update.
                    # Prevents false alerts from noisy ACCOUNT_UPDATE payloads.
                    if symbol not in known_open_before:
                        continue

                    # Anti-spam: suppress duplicate alerts for same symbol within 10 minutes.
                    now_ts = time_mod.time()
                    last_ts = untracked_close_alerts.get(symbol, 0)
                    if now_ts - last_ts < 600:
                        continue
                    untracked_close_alerts[symbol] = now_ts

                    msg_txt = (f"⚡ <b>UNTRACKED POSITION CLOSED</b>\n\n"
                               f"Symbol: <b>{symbol}</b>\n"
                               f"Notice: This position was closed but was NOT tracked by the bot's active_pairs list.\n"
                               f"Cause: Manual close or DB sync issue.")
                    await send_tg_notification(msg_txt)
    
    
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
                        
                        # Skip if pair is already being processed for closure
                        # (e.g. bulk close on exchange cancels orders then closes positions)
                        if getattr(pair_info, 'is_trading', False):
                            print(f"ℹ️ {s1}-{s2} already being processed, skipping cancel handler")
                            break
                        
                        # Notify user about manual order cancellation
                        cancel_msg = (f"⚠️ <b>Order CANCELED:</b> {symbol}\n"
                                      f"Type: {order_type}\n"
                                      f"Pair: {s1}-{s2}\n"
                                      f"⏳ Checking pair integrity...")
                        try:
                            await send_tg_notification(cancel_msg)
                        except Exception as e:
                            print(f"⚠️ TG notify error: {e}")
                        
                        # Try restoring protection immediately (1 retry), then fallback to leg sync.
                        try:
                            restored = await pairs_manager.restore_protection_for_symbol(symbol, max_attempts=2)
                            if not restored:
                                await pairs_manager._check_leg_synchronization()
                        except Exception as e:
                            print(f"⚠️ Leg sync error after cancel: {e}")
                        break
    
    # Check for ALGO_UPDATE (algo order triggered/finished - SL/TP via algo endpoint)
    if event_type == 'ALGO_UPDATE':
        algo_data = msg.get('o', {})
        algo_id = str(algo_data.get('aid', ''))  # Algo order ID (Binance field: "aid")
        algo_status = algo_data.get('X', '')     # Algo Status (Binance field: "X"): NEW, CANCELED, TRIGGERING, TRIGGERED, FINISHED, REJECTED, EXPIRED
        algo_symbol = algo_data.get('s', '')     # Symbol (Binance field: "s")
        algo_type = algo_data.get('o', '')       # Order Type (Binance field: "o"): STOP, TAKE_PROFIT, etc.
        
        print(f"📡 ALGO_UPDATE: {algo_symbol} {algo_type} {algo_status} (algoId={algo_id})")
        
        if algo_status in ('TRIGGERING', 'TRIGGERED') and pairs_manager:
            # Check if this algoId is tracked
            algo_info = pairs_manager.algo_orders.get(algo_id)
            if algo_info:
                order_type = algo_info.get('type', algo_type)
                pair_key = algo_info.get('pair_key')
                symbol = algo_info.get('symbol', algo_symbol)
                
                is_tp = 'TAKE_PROFIT' in order_type.upper() if order_type else False
                tp_or_sl = 'TP' if is_tp else 'SL'
                
                print(f"🎯 Algo {tp_or_sl} triggered: {symbol} (algoId={algo_id})")
                
                try:
                    await pairs_manager.handle_sl_tp_triggered(symbol, order_type)
                    
                    # Clean up all algo orders for this pair
                    if pair_key:
                        to_remove = [aid for aid, info in pairs_manager.algo_orders.items()
                                     if info.get('pair_key') == pair_key]
                        for aid in to_remove:
                            del pairs_manager.algo_orders[aid]
                        print(f"🗑️ Cleaned up {len(to_remove)} algo order mappings for pair")
                except Exception as e:
                    print(f"⚠️ Error handling algo SL/TP trigger: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"ℹ️ Algo order {algo_id} not tracked (may be from previous session)")
                # Fallback: try to match by symbol
                if algo_status == 'TRIGGERING':
                    try:
                        await pairs_manager.handle_sl_tp_triggered(algo_symbol, algo_type)
                    except Exception as e:
                        print(f"⚠️ Fallback algo handler error: {e}")
        
        elif algo_status == 'CANCELED' and pairs_manager:
            # Remove from tracking
            if algo_id in pairs_manager.algo_orders:
                del pairs_manager.algo_orders[algo_id]
                print(f"🗑️ Removed canceled algo order {algo_id} from tracking")


if __name__ == '__main__':
    try:
        _configure_console_encoding()
        _install_print_mojibake_fix()
        print("Starting market neutral bot...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user.")
