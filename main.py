import asyncio
import configparser
import traceback
import os
import time as time_mod
import math
import sys
import builtins
import csv
import gzip
import shutil
from collections import deque
import aiohttp
from urllib.parse import urlsplit, parse_qs
from dotenv import load_dotenv
import binance
import pairs_trading
from pairs_trading import CLOSE_REASONS
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
# Anti-spam cache for noisy untracked-close websocket events
untracked_close_alerts: dict[str, float] = {}
# Short-lived suppression for symbols that were just handled by pair close logic.
# Prevents false "UNTRACKED POSITION CLOSED" on the next ACCOUNT_UPDATE tick.
recently_handled_close_symbols: dict[str, float] = {}
_pair_history_last_alert_key: str = ''
_orig_print = builtins.print
_pair_history_last_backup_key: str = ''
_main_timeframe_global: str = '1h'

# WS health/watchdog state
_ws_last_main_msg_ts: float = 0.0
_ws_last_mark_msg_ts: float = 0.0
_ws_last_user_msg_ts: float = 0.0
_ws_error_ts: deque = deque(maxlen=512)
_ws_recover_lock: asyncio.Lock | None = None
_ws_last_recover_ts: float = 0.0


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


_CP1252_UNICODE_TO_BYTE = {
    0x20AC: 0x80, 0x201A: 0x82, 0x0192: 0x83, 0x201E: 0x84, 0x2026: 0x85,
    0x2020: 0x86, 0x2021: 0x87, 0x02C6: 0x88, 0x2030: 0x89, 0x0160: 0x8A,
    0x2039: 0x8B, 0x0152: 0x8C, 0x017D: 0x8E, 0x2018: 0x91, 0x2019: 0x92,
    0x201C: 0x93, 0x201D: 0x94, 0x2022: 0x95, 0x2013: 0x96, 0x2014: 0x97,
    0x02DC: 0x98, 0x2122: 0x99, 0x0161: 0x9A, 0x203A: 0x9B, 0x0153: 0x9C,
    0x017E: 0x9E, 0x0178: 0x9F,
}


def _repair_mojibake_once(s: str):
    if not isinstance(s, str) or not s:
        return s, False
    data = bytearray()
    used_cp1252_map = False
    for ch in s:
        code = ord(ch)
        if code <= 0xFF:
            data.append(code)
            continue
        mapped = _CP1252_UNICODE_TO_BYTE.get(code)
        if mapped is None:
            return s, False
        used_cp1252_map = True
        data.append(mapped)
    # If no cp1252-only chars and no control-range chars, likely already clean text.
    if not used_cp1252_map and not any(0x80 <= b <= 0x9F for b in data):
        return s, False
    try:
        fixed = bytes(data).decode('utf-8')
    except Exception:
        return s, False
    return fixed, (fixed != s)


def _fix_mojibake_text(s: str) -> str:
    """Repair UTF-8 text that was decoded as Latin-1/CP1252 (possibly multiple passes)."""
    out = s
    for _ in range(2):
        out2, changed = _repair_mojibake_once(out)
        if not changed:
            break
        out = out2
    return out


def _install_print_mojibake_fix():
    """Patch builtins.print so runtime logs are auto-repaired if mojibake appears."""
    def _fixed_print(*args, **kwargs):
        fixed_args = [(_fix_mojibake_text(a) if isinstance(a, str) else a) for a in args]
        _orig_print(*fixed_args, **kwargs)
    builtins.print = _fixed_print

async def send_tg_notification(message, reply_to_message_id=None, reply_markup=None):
    """Send notification to TG channel or admins. Returns message_id for reply threading."""
    if isinstance(message, str):
        message = _fix_mojibake_text(message)
    if not tg.bot:
        print("âš ï¸ TG: bot not initialized")
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
            print(f"ðŸ“¨ TG sent to channel, msg_id={msg_id}, reply_to={reply_to_message_id}")
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
                print(f"ðŸ“¨ TG sent to {admin_id}, msg_id={msg_id}, reply_to={reply_to_message_id}")
            except Exception as e:
                print(f"Error sending TG to {admin_id}: {e}")
    else:
        print("âš ï¸ TG: no channel or admins configured")
    
    return msg_id


async def main():
    global client
    global pairs_manager
    global all_symbols
    global _main_timeframe_global
    global _ws_recover_lock
    
    # Load environment variables from .env file
    _configure_console_encoding()
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
    
    # CRITICAL: Sync time with server to avoid -1021 timestamp errors.
    try:
        await client._sync_time_async()
    except Exception as e:
        print(f"⚠️ Initial time sync failed: {e}")
    
    # Init pairs manager
    loop = asyncio.get_running_loop()
    
    # Default values
    timeframe = conf.timeframe if conf.timeframe else '1h'
    _main_timeframe_global = timeframe
    if _ws_recover_lock is None:
        _ws_recover_lock = asyncio.Lock()
    
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
                print(f"âš ï¸ Invalid window_size '{conf.window_size}' in config. Using auto-calculation.")

    if use_manual_window:
        window_size = window_size_val
        print(f"âš™ï¸ Using manual window_size: {window_size}")
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
        print(f"âš™ï¸ Auto-calculated window_size: {window_size} (for {timeframe})")
    
    
    # 1. Load symbols (with error handling for bad filter data from Binance)
    print("Initial loading of market symbols...")
    try:
        all_symbols = await client.load_symbols()
    except ValueError as e:
        # Fallback: Manual loading with skipping problematic symbols
        print(f"âš ï¸ Standard load failed ({e}). Using safe loader...")
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
    print(f"Loaded {len(all_symbols)} symbols.")
    
    # 1.5 VOLUME FILTER: Keep only top N symbols by 24h volume
    max_symbols = int(conf.max_symbols) if conf.max_symbols else 150
    blacklist = {s.strip().upper() for s in (conf.blacklist or '').split(',') if s.strip()}
    
    try:
        print(f"ðŸ“ˆ Filtering top {max_symbols} symbols by 24h volume...")
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
        
        # SAFETY: Always keep symbols that currently have open positions (DB or exchange)
        protected_symbols = set()
        try:
            db_pairs = await db.get_all_pairs()
            for p in db_pairs:
                if getattr(p, 'position_status', 0) != 0:
                    protected_symbols.add(p.symbol1)
                    protected_symbols.add(p.symbol2)
        except Exception as e:
            print(f"Could not load protected symbols from DB: {e}")
        try:
            exchange_positions = await client.get_position_risk()
            for pos in exchange_positions:
                if abs(float(pos.get('positionAmt', 0))) > 0:
                    protected_symbols.add(pos.get('symbol', ''))
        except Exception as e:
            print(f"Could not load protected symbols from exchange: {e}")
        
        top_symbols.update(s for s in protected_symbols if s in all_symbols)
        
        # Filter all_symbols to only include top volume symbols
        filtered_symbols = {s: obj for s, obj in all_symbols.items() if s in top_symbols}
        print(f"âœ… Filtered to {len(filtered_symbols)} symbols (from {len(all_symbols)}, blacklist: {len(blacklist)}, protected: {len(protected_symbols)})")
        all_symbols = filtered_symbols
    except Exception as e:
        print(f"âš ï¸ Volume filter failed ({e}). Using all symbols.")
    
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

    # CRITICAL: Initialize TG bot FIRST so notifications work during reconciliation
    await tg.init_bot()

    # CRITICAL: Initialize pairs manager (loads DB state + reconciles with exchange)
    await pairs_manager.initialize()

    # 3. Start background symbol updates
    loop.create_task(load_symbols_loop())
    loop.create_task(pair_history_retention_notice_loop())
    loop.create_task(sync_exchange_time_loop())
    loop.create_task(ws_health_watchdog_loop())
    
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
            new_symbols = await client.load_symbols()
            
            # Apply volume filter + blacklist (same as initial load in main())
            conf = await db.load_config()
            max_symbols = int(conf.max_symbols) if conf.max_symbols else 150
            blacklist = {s.strip().upper() for s in (conf.blacklist or '').split(',') if s.strip()}
            
            try:
                tickers = await client.ticker_24hr_price_change()
                valid_tickers = []
                for t in tickers:
                    sym = t.get('symbol', '')
                    if sym.endswith('USDT') and sym not in blacklist and sym in new_symbols:
                        try:
                            vol = float(t.get('quoteVolume', 0))
                            valid_tickers.append((sym, vol))
                        except Exception:
                            continue
                valid_tickers.sort(key=lambda x: x[1], reverse=True)
                top_symbols = set(sym for sym, vol in valid_tickers[:max_symbols])
                
                # SAFETY: Preserve symbols used by active/open positions
                protected_symbols = set()
                if pairs_manager:
                    for pair_info in pairs_manager.active_pairs.values():
                        if getattr(pair_info, 'position_status', 0) != 0 or getattr(pair_info, 'is_trading', False):
                            protected_symbols.add(pair_info.symbol1)
                            protected_symbols.add(pair_info.symbol2)
                try:
                    exchange_positions = await client.get_position_risk()
                    for pos in exchange_positions:
                        if abs(float(pos.get('positionAmt', 0))) > 0:
                            protected_symbols.add(pos.get('symbol', ''))
                except Exception as e:
                    print(f"Could not refresh protected symbols from exchange: {e}")
                
                top_symbols.update(s for s in protected_symbols if s in new_symbols)
                filtered_symbols = {s: obj for s, obj in new_symbols.items() if s in top_symbols}
                print(f"âœ… Refreshed {len(filtered_symbols)} symbols (from {len(new_symbols)}, blacklist: {len(blacklist)}, protected: {len(protected_symbols)})")
                new_symbols = filtered_symbols
            except Exception as e:
                print(f"âš ï¸ Volume filter failed during refresh ({e}). Using all symbols.")
            
            # Update BOTH global and pairs_manager references
            all_symbols = new_symbols
            if pairs_manager:
                pairs_manager.all_symbols = new_symbols
                print(f"âœ… pairs_manager.all_symbols updated ({len(new_symbols)} symbols)")
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


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ('true', '1', 'yes', 'on')


async def pair_history_retention_notice_loop():
    """
    Non-destructive retention monitor:
    - sends TG reminder when pair_history rows are close to retention threshold
    - sends TG warning when rows are already older than retention horizon
    No deletion is performed here.
    """
    global _pair_history_last_alert_key
    global _pair_history_last_backup_key
    while True:
        sleep_sec = 6 * 3600
        try:
            conf = await db.load_config()
            retention_days = int(getattr(conf, 'pair_history_retention_days', 365) or 365)
            warn_days = int(getattr(conf, 'pair_history_warn_days', 14) or 14)
            warn_days = max(1, min(warn_days, retention_days))
            cleanup_enabled = _to_bool(getattr(conf, 'pair_history_cleanup_enabled', False))
            interval_h = int(getattr(conf, 'pair_history_check_interval_hours', 6) or 6)
            sleep_sec = max(1, interval_h) * 3600

            old_count = await db.count_pair_history_older_than_days(retention_days)
            warn_from = max(1, retention_days - warn_days)
            near_count = await db.count_pair_history_age_between_days(warn_from, retention_days)

            alert_kind = ''
            if old_count > 0:
                alert_kind = f'over:{retention_days}:{old_count}:{int(cleanup_enabled)}'
                msg = (
                    f"📦 <b>DB Retention Notice</b>\n\n"
                    f"Table: <b>pair_history</b>\n"
                    f"Older than retention ({retention_days}d): <b>{old_count}</b> rows\n"
                    f"Cleanup enabled: <b>{'YES' if cleanup_enabled else 'NO'}</b>\n\n"
                    f"Recommendation: export/backup data before enabling cleanup."
                )
            elif near_count > 0:
                alert_kind = f'near:{warn_from}-{retention_days}:{near_count}:{int(cleanup_enabled)}'
                msg = (
                    f"📦 <b>DB Retention Warning</b>\n\n"
                    f"Table: <b>pair_history</b>\n"
                    f"Will reach retention in ≤ {warn_days} days: <b>{near_count}</b> rows\n"
                    f"Retention horizon: <b>{retention_days} days</b>\n"
                    f"Cleanup enabled: <b>{'YES' if cleanup_enabled else 'NO'}</b>\n\n"
                    f"Recommendation: export/backup data on server."
                )
            else:
                msg = ''

            # Send at most once per UTC day per alert snapshot
            if msg:
                day_key = time_mod.strftime('%Y-%m-%d', time_mod.gmtime())
                key = f"{day_key}:{alert_kind}"
                if key != _pair_history_last_alert_key:
                    await send_tg_notification(msg)
                    _pair_history_last_alert_key = key

            # Optional 2-slot rotating backup (current/prev), non-destructive.
            backup_enabled = _to_bool(getattr(conf, 'pair_history_backup_enabled', False))
            if backup_enabled and (old_count > 0 or near_count > 0):
                backup_interval_h = int(getattr(conf, 'pair_history_backup_interval_hours', 24) or 24)
                backup_day_key = time_mod.strftime('%Y-%m-%d', time_mod.gmtime())
                backup_key = f"{backup_day_key}:{backup_interval_h}:{retention_days}:{warn_days}"
                if backup_key != _pair_history_last_backup_key:
                    cutoff_days = max(1, retention_days - warn_days)
                    backup_rows, backup_path = await _backup_pair_history_rotating(conf, cutoff_days=cutoff_days)
                    if backup_rows > 0:
                        await send_tg_notification(
                            f"💾 <b>pair_history backup done</b>\n\n"
                            f"Rows exported: <b>{backup_rows}</b>\n"
                            f"Age filter: older than <b>{cutoff_days}</b> days\n"
                            f"File: <code>{backup_path}</code>\n"
                            f"Rotation: current + prev"
                        )
                    _pair_history_last_backup_key = backup_key

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️ Retention monitor error: {e}")
        await asyncio.sleep(sleep_sec)


async def _backup_pair_history_rotating(conf, cutoff_days: int = 351):
    """
    Export old pair_history rows into a rotating 2-file backup set:
    - pair_history_backup_current.csv.gz
    - pair_history_backup_prev.csv.gz
    """
    cutoff_days = max(1, int(cutoff_days))
    cutoff_ms = int((time_mod.time() - cutoff_days * 86400) * 1000)

    backup_dir = getattr(conf, 'pair_history_backup_dir', 'market_neutral/backups') or 'market_neutral/backups'
    if not os.path.isabs(backup_dir):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        backup_dir = os.path.join(root_dir, backup_dir)
    os.makedirs(backup_dir, exist_ok=True)

    new_path = os.path.join(backup_dir, 'pair_history_backup_new.csv.gz')
    current_path = os.path.join(backup_dir, 'pair_history_backup_current.csv.gz')
    prev_path = os.path.join(backup_dir, 'pair_history_backup_prev.csv.gz')

    written = 0
    last_id = 0
    with gzip.open(new_path, mode='wt', encoding='utf-8', newline='') as gz:
        writer = csv.writer(gz)
        writer.writerow([
            'id', 'symbol1', 'symbol2', 'event_type', 'timestamp',
            'hedge_ratio', 'half_life', 'pair_id', 'trade_id',
            'z_score', 'beta_btc', 'pvalue', 'reason'
        ])
        while True:
            batch = await db.fetch_pair_history_batch_before_ts(cutoff_ms=cutoff_ms, last_id=last_id, limit=5000)
            if not batch:
                break
            for row in batch:
                writer.writerow([
                    row.id, row.symbol1, row.symbol2, row.event_type, row.timestamp,
                    row.hedge_ratio, row.half_life,
                    getattr(row, 'pair_id', None),
                    getattr(row, 'trade_id', None),
                    getattr(row, 'z_score', None),
                    getattr(row, 'beta_btc', None),
                    getattr(row, 'pvalue', None),
                    row.reason
                ])
                last_id = row.id
                written += 1
            if len(batch) < 5000:
                break

    # Rotate: current -> prev, new -> current
    try:
        if os.path.exists(prev_path):
            os.remove(prev_path)
        if os.path.exists(current_path):
            shutil.move(current_path, prev_path)
        shutil.move(new_path, current_path)
    finally:
        if os.path.exists(new_path):
            os.remove(new_path)

    return written, current_path


# Connect to websockets
async def connect_ws(timeframe='1h'):
    global websockets_list
    global userdata_ws
    global pairs_manager
    global _ws_last_main_msg_ts
    global _ws_last_mark_msg_ts
    global _ws_last_user_msg_ts

    print("Connecting to websockets...")
    now_ts = time_mod.time()
    _ws_last_main_msg_ts = now_ts
    _ws_last_mark_msg_ts = now_ts
    _ws_last_user_msg_ts = now_ts

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

        # Filter 2: ASCII only (exclude Chinese chars and weird symbols)
        if not s_name.isascii():
            continue

        # Filter 3: Exclude blacklist
        if s_name in FULL_BLACKLIST:
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

    # MAIN TIMEFRAME: for discovery (cointegration tests)
    main_streams = [f"{symbol.lower()}@kline_{timeframe}" for symbol in target_symbols]
    
    print(f"Single TF Mode: {timeframe}")

    # Start websockets for MAIN timeframe (slightly smaller chunks for better stability)
    kline_chunk_size = 80
    streams_list = [main_streams[i:i + kline_chunk_size] for i in range(0, len(main_streams), kline_chunk_size)]

    for i, stream_list in enumerate(streams_list):
        try:
            ws = await client.websocket(stream_list, on_message=ws_msg_main, on_error=ws_error)
            websockets_list.append(ws)
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error subscribing to main TF chunk {i+1}: {e}")

    print(f"Connected to main TF kline websockets ({len(websockets_list)} connections).")

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
                desired_symbols = set(requested_symbols) | protected_symbols
                if len(desired_symbols) > mark_max_symbols:
                    protected_sorted = sorted(desired_symbols & protected_symbols)
                    other_sorted = sorted(desired_symbols - set(protected_sorted))
                    allowed_others = max(0, mark_max_symbols - len(protected_sorted))
                    desired_symbols = set(protected_sorted + other_sorted[:allowed_others])
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
                    print(f"âš ï¸ Failed to rebuild markPrice streams (requested={len(requested_symbols)}, subscribed={len(desired_symbols)}): {e}")
        
        # Set the callback on pairs_manager so it can subscribe new pairs
        pairs_manager._subscribe_mark_callback = subscribe_new_marks
        
        # Subscribe to initial symbols
        if active_symbols:
            await subscribe_new_marks(sorted(active_symbols))
            print(f"Connected to markPrice websocket (requested={len(active_symbols)}, cap={mark_max_symbols}).")
        else:
            print("â„¹ï¸ No active pairs at startup - markPrice will be subscribed dynamically.")
        
        # Heavy warmup+discovery is moved to background to avoid blocking startup
        pairs_manager.start_background_warmup(target_symbols, concurrency=20)


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
    websockets_list.clear()
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
        print(f"ðŸ“¡ UserData WS: {event_type} received")
    
    # Check for ACCOUNT_UPDATE (position changes - including manual closes)
    if event_type == 'ACCOUNT_UPDATE':
        global recently_handled_close_symbols
        positions = msg.get('a', {}).get('P', [])
        print(f"ðŸ“¡ ACCOUNT_UPDATE: {len(positions)} positions in update")

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
        closed_symbols = []
        for pos in positions:
            sym = pos.get('s')
            amt = float(pos.get('pa', 0))
            up = float(pos.get('up', 0))  # unrealizedProfit from exchange
            if amt == 0:
                closed_symbols.append(sym)
                # Clear PnL cache for closed position
                if pairs_manager:
                    pairs_manager._exchange_pnl_cache.pop(sym, None)
                    pairs_manager._exchange_positions_cache.pop(sym, None)
            else:
                # Update PnL cache in real-time from WebSocket (instant, no API call)
                if pairs_manager:
                    pairs_manager._exchange_pnl_cache[sym] = up
                    pairs_manager._exchange_positions_cache[sym] = abs(amt)
        
        # Note: Detailed notifications will be sent per-pair below
        # Skip simple "Position Changes" message - too noisy
        
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
                                     'btc_shock', 'desync', 'orphan_restart', 'stale_symbols')
                if getattr(pair_info, 'close_handled', False) and stored_reason in bot_close_reasons:
                    print(f"â„¹ï¸ {s1}-{s2} close already handled by bot (reason: {stored_reason}), skipping external notification")
                    now_mark = time_mod.time()
                    recently_handled_close_symbols[s1] = now_mark
                    recently_handled_close_symbols[s2] = now_mark
                    pair_info.close_handled = False  # Reset for next trade
                    pair_info.is_trading = False
                    continue
                
                # Both legs closed together - fetch actual PnL and cleanup
                print(f"âš¡ Both legs of {s1}-{s2} closed externally. Fetching PnL...")
                try:
                    await client.cancel_open_orders(s1)
                    await client.cancel_open_orders(s2)
                    
                    # Small delay to ensure trade data is available
                    await asyncio.sleep(1)
                    
                    # Fetch actual PnL from recent trades
                    now_ms = int(time_mod.time() * 1000)
                    open_time = int(getattr(pair_info, 'open_time', 0) or 0)
                    start_ms = (max(0, open_time - 120) * 1000) if open_time > 0 else (now_ms - 300_000)
                    
                    trades1 = await client.get_account_trades(symbol=s1, startTime=start_ms, limit=50)
                    trades2 = await client.get_account_trades(symbol=s2, startTime=start_ms, limit=50)
                    
                    print(f"ðŸ“Š Trades for {s1}: {len(trades1)} entries")
                    print(f"ðŸ“Š Trades for {s2}: {len(trades2)} entries")
                    
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
                    close_type = 'â“ Unknown'
                    close_hint = '\nðŸ’¡ Check exchange for details'
                    
                    stored_reason = getattr(pair_info, 'last_close_reason', '')
                    if stored_reason and stored_reason in CLOSE_REASONS:
                        close_type = CLOSE_REASONS[stored_reason]
                        close_hint = ''
                        print(f"ðŸ“‹ Using stored reason: {stored_reason} -> {close_type}")
                    else:
                        try:
                            orders1 = await client.get_all_orders(symbol=s1, limit=15)
                            orders2 = await client.get_all_orders(symbol=s2, limit=15)
                            
                            now_ms = int(time_mod.time() * 1000)
                            recent_orders = []
                            for o in orders1 + orders2:
                                if o.get('status') == 'FILLED' and o.get('updateTime', 0) > now_ms - 300_000:
                                    recent_orders.append(o)
                            
                            if recent_orders:
                                recent_orders.sort(key=lambda x: x.get('updateTime', 0), reverse=True)
                                o = recent_orders[0]
                                o_type = o.get('type', '') or o.get('origType', '')
                                
                                if 'STOP' in o_type:
                                    close_type = 'ðŸ›¡ï¸ Hardware SL'
                                elif 'TAKE_PROFIT' in o_type:
                                    close_type = 'ðŸ›¡ï¸ Hardware TP'
                                elif o_type == 'MARKET':
                                    close_type = 'ðŸ‘¤ Manual Market' if not o.get('reduceOnly') else 'ðŸ¤– Bot Close'
                                elif o_type == 'LIMIT':
                                    close_type = 'ðŸ“Š Limit Order'
                                elif 'TRAILING' in o_type:
                                    close_type = 'ðŸ“ˆ Trailing Stop'
                                else:
                                    close_type = f'âš¡ {o_type}'
                                print(f"ðŸ“‹ Detected: {o_type} -> {close_type}")
                            else:
                                close_type = 'âš¡ External Close'
                                close_hint = ' (no orders found)'
                                print(f"âš ï¸ No orders for {s1}/{s2}")
                        except Exception as e:
                            print(f"âš ï¸ Query error: {e}")
                            close_type = 'âš¡ External'
                    
                    pnl_emoji = 'ðŸŸ¢' if net_pnl >= 0 else 'ðŸ”´'
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
                    # Format half-life
                    hl = getattr(pair_info, 'half_life', 0) or 0
                    if hl > 0:
                        if hl >= 24:
                            hl_d = int(hl // 24)
                            hl_h = int(hl % 24)
                            close_hl = f"{hl_d}d {hl_h}h" if hl_h > 0 else f"{hl_d}d"
                        else:
                            hl_h = int(hl)
                            hl_m = int((hl - hl_h) * 60)
                            close_hl = f"{hl_h}h {hl_m}m" if hl_m > 0 else f"{hl_h}h"
                    else:
                        close_hl = 'N/A'
                    hedge = getattr(pair_info, 'hedge_ratio', 0) or 0
                    e1 = 'ðŸŸ¢' if pnl1 >= 0 else 'ðŸ”´'
                    e2 = 'ðŸŸ¢' if pnl2 >= 0 else 'ðŸ”´'
                    done_msg = (f"{close_type}: <b>{s1}/{s2}</b>\n\n"
                                f"ðŸ“Š Z: {zscore:+.2f} | Î²: {beta:.3f} | p: {close_pval:.4f}\n"
                                f"â³ HL: {close_hl} | Hedge: {hedge:.4f}\n"
                                f"ðŸ’µ PnL: {pnl_emoji} <b>{net_pnl:+.2f} USDT</b>\n"
                                f"   {e1} {s1}: {pnl1:+.2f} | {e2} {s2}: {pnl2:+.2f}\n"
                                f"ðŸ’¸ Fees: {total_fees:.4f} USDT{close_hint}")
                    
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
                    pair_info._wait_for_candle = True  # Block re-entry until next candle
                    
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
                        except Exception as trade_err:
                            print(f"âš ï¸ Trade update failed for {s1}-{s2}: {trade_err}")
                        pair_info.current_trade_id = None
                except Exception as e:
                    print(f"âš ï¸ Cleanup error: {e}")
                    import traceback
                    traceback.print_exc()
                    pair_info.is_trading = False
            else:
                # Only one leg closed - check if bot is already handling this
                stored_reason = getattr(pair_info, 'last_close_reason', '')
                if getattr(pair_info, 'close_handled', False) and stored_reason in ('manual', 'z_tp', 'z_sl', 'circuit', 'broken_coint', 
                        'hardware_sl', 'hardware_tp', 'beta_drift', 'beta_critical',
                        'btc_shock', 'desync', 'orphan_restart', 'stale_symbols'):
                    print(f"â„¹ï¸ {s1}-{s2} close already handled by bot (reason: {stored_reason}), skipping single-leg handler")
                    now_mark = time_mod.time()
                    recently_handled_close_symbols[s1] = now_mark
                    recently_handled_close_symbols[s2] = now_mark
                    pair_info.is_trading = False
                    continue
                
                # External close - user manually closed one position
                print(f"âš¡ External close detected: {symbol} in pair {s1}-{s2}. Closing {other_symbol} IMMEDIATELY...")
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
                            await client.new_order(
                                symbol=other_symbol,
                                side=close_side,
                                type='MARKET',
                                quantity=abs(other_amt),
                                reduceOnly='true'
                            )
                            print(f"âœ… Closed remaining leg {other_symbol} (qty={abs(other_amt)}, side={close_side}, reduceOnly=true)")
                        except Exception as close_err:
                            err_txt = str(close_err)
                            if "-2022" in err_txt or "ReduceOnly Order is rejected" in err_txt:
                                # Typical race: leg may already be zero. Re-check before fallback.
                                verify_data = await client.get_position_risk(symbol=other_symbol)
                                verify_pos = verify_data[0] if verify_data else {}
                                verify_amt = float(verify_pos.get('positionAmt', 0))
                                if verify_amt == 0:
                                    pair_info.last_close_reason = 'external'
                                    close_exec_note = "â„¹ï¸ Remaining leg was already closed on exchange."
                                    print(f"â„¹ï¸ {other_symbol} already closed after reduceOnly reject (-2022).")
                                else:
                                    verify_side = 'SELL' if verify_amt > 0 else 'BUY'
                                    await client.new_order(
                                        symbol=other_symbol,
                                        side=verify_side,
                                        type='MARKET',
                                        quantity=abs(verify_amt)
                                    )
                                    close_exec_note = "â„¹ï¸ reduceOnly rejected, closed with fallback MARKET order."
                                    print(f"âš ï¸ reduceOnly rejected for {other_symbol}; fallback MARKET close succeeded.")
                            else:
                                close_exec_note = f"âš ï¸ Could not close remaining leg: {close_err}"
                                print(f"âš ï¸ Failed to close remaining leg {other_symbol}: {close_err}")
                    else:
                        pair_info.last_close_reason = 'external'
                        close_exec_note = "â„¹ï¸ Remaining leg was already closed on exchange."
                        print(f"â„¹ï¸ Remaining leg {other_symbol} already at zero position.")
                    
                    # THEN cancel remaining algo/SL/TP orders (non-critical, can be slower)
                    try:
                        await client.cancel_open_orders(s1)
                        await client.cancel_open_orders(s2)
                    except Exception as cancel_err:
                        print(f"âš ï¸ Cancel orders error (non-critical): {cancel_err}")
                    
                    await asyncio.sleep(1)
                    
                    # Verify remaining leg really closed before finalizing pair state.
                    verify_data = await client.get_position_risk(symbol=other_symbol)
                    verify_pos = verify_data[0] if verify_data else {}
                    remaining_amt = float(verify_pos.get('positionAmt', 0))
                    if remaining_amt != 0:
                        warn_msg = (f"ðŸš¨ <b>External close handling incomplete</b>\n\n"
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
                    
                    trades1 = await client.get_account_trades(symbol=s1, startTime=start_ms, limit=50)
                    trades2 = await client.get_account_trades(symbol=s2, startTime=start_ms, limit=50)
                    
                    print(f"ðŸ“Š Trades for {s1}: {len(trades1)} entries")
                    print(f"ðŸ“Š Trades for {s2}: {len(trades2)} entries")
                    
                    pnl1 = sum(float(t.get('realizedPnl', 0)) for t in trades1)
                    pnl2 = sum(float(t.get('realizedPnl', 0)) for t in trades2)
                    fee1 = sum(float(t.get('commission', 0)) for t in trades1)
                    fee2 = sum(float(t.get('commission', 0)) for t in trades2)
                    close_price1 = float(trades1[-1].get('price', 0)) if trades1 else 0.0
                    close_price2 = float(trades2[-1].get('price', 0)) if trades2 else 0.0
                    total_pnl = pnl1 + pnl2
                    total_fees = fee1 + fee2
                    net_pnl = total_pnl - total_fees
                    
                    pnl_emoji = "ðŸŸ¢" if net_pnl >= 0 else "ðŸ”´"
                    
                    # Update memory state
                    pair_info.position_status = 0
                    pair_info.qty1 = 0
                    pair_info.qty2 = 0
                    pair_info.is_trading = False
                    pair_info._wait_for_candle = True  # Block re-entry until next candle
                    
                    stored_reason = getattr(pair_info, 'last_close_reason', '')
                    
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
                        except Exception as trade_err:
                            print(f"âš ï¸ Trade update failed for {s1}-{s2}: {trade_err}")
                        pair_info.current_trade_id = None
                    
                    close_type = 'â“ Unknown'
                    close_hint = ''
                    if close_exec_note:
                        close_hint = f"\n{close_exec_note}"
                    
                    if stored_reason and stored_reason in CLOSE_REASONS:
                        close_type = CLOSE_REASONS[stored_reason]
                        print(f"ðŸ“‹ Using stored reason: {stored_reason} -> {close_type}")
                    else:
                        try:
                            orders1 = await client.get_all_orders(symbol=s1, limit=15)
                            orders2 = await client.get_all_orders(symbol=s2, limit=15)
                            
                            now_time = int(time_mod.time() * 1000)
                            recent_orders = []
                            for o in orders1 + orders2:
                                if o.get('status') == 'FILLED' and o.get('updateTime', 0) > now_time - 300_000:
                                     recent_orders.append(o)
                            
                            if recent_orders:
                                recent_orders.sort(key=lambda x: x.get('updateTime', 0), reverse=True)
                                o = recent_orders[0]
                                o_type = o.get('type', '') or o.get('origType', '')
                                
                                if 'STOP' in o_type:
                                    close_type = 'ðŸ›¡ï¸ Hardware SL'
                                elif 'TAKE_PROFIT' in o_type:
                                    close_type = 'ðŸ›¡ï¸ Hardware TP'
                                elif o_type == 'MARKET':
                                    if o.get('reduceOnly', False):
                                        close_type = 'ðŸ¤– Bot Close (reason unknown)'
                                    else:
                                        close_type = 'ðŸ‘¤ Manual Market Order'
                                elif o_type == 'LIMIT':
                                    close_type = 'ðŸ“Š Limit Order Filled'
                                elif 'TRAILING' in o_type:
                                    close_type = 'ðŸ“ˆ Trailing Stop'
                                else:
                                    close_type = f'âš¡ Order: {o_type}'
                                print(f"ðŸ“‹ Detected from orders: {o_type} -> {close_type}")
                            else:
                                close_type = 'âš¡ External Close'
                                close_hint += ' (no matching orders)'
                                print(f"âš ï¸ No recent orders found for {s1}/{s2}")
                        except Exception as e:
                            print(f"âš ï¸ Could not query orders: {e}")
                            close_type = 'âš¡ External Close'
                            close_hint += ' (query failed)'
                    
                    pnl_emoji = 'ðŸŸ¢' if net_pnl >= 0 else 'ðŸ”´'
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
                    # Format half-life
                    hl = getattr(pair_info, 'half_life', 0) or 0
                    if hl > 0:
                        if hl >= 24:
                            hl_d = int(hl // 24)
                            hl_h = int(hl % 24)
                            close_hl = f"{hl_d}d {hl_h}h" if hl_h > 0 else f"{hl_d}d"
                        else:
                            hl_h = int(hl)
                            hl_m = int((hl - hl_h) * 60)
                            close_hl = f"{hl_h}h {hl_m}m" if hl_m > 0 else f"{hl_h}h"
                    else:
                        close_hl = 'N/A'
                    hedge = getattr(pair_info, 'hedge_ratio', 0) or 0
                    e1 = 'ðŸŸ¢' if pnl1 >= 0 else 'ðŸ”´'
                    e2 = 'ðŸŸ¢' if pnl2 >= 0 else 'ðŸ”´'
                    done_msg = (f"{close_type}: <b>{s1}/{s2}</b>\n\n"
                                f"ðŸ“Š Z: {zscore:+.2f} | Î²: {beta:.3f} | p: {close_pval:.4f}\n"
                                f"â³ HL: {close_hl} | Hedge: {hedge:.4f}\n"
                                f"ðŸ’µ PnL: {pnl_emoji} <b>{net_pnl:+.2f} USDT</b>\n"
                                f"   {e1} {s1}: {pnl1:+.2f} | {e2} {s2}: {pnl2:+.2f}\n"
                                f"ðŸ’¸ Fees: {total_fees:.4f} USDT{close_hint}")
                    reply_to = pair_info.tg_message_id if pair_info.tg_message_id else None
                    await send_tg_notification(done_msg, reply_to)
                    now_mark = time_mod.time()
                    recently_handled_close_symbols[s1] = now_mark
                    recently_handled_close_symbols[s2] = now_mark
                    
                except Exception as e:
                    print(f"âš ï¸ External close handling error for {s1}-{s2}: {e}")
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

                    msg_txt = (f"âš¡ <b>UNTRACKED POSITION CLOSED</b>\n\n"
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
            print(f"ðŸŽ¯ Hardware SL/TP triggered: {symbol} {order_type} FILLED")
            
            # Notify pairs_manager to close the other leg
            if pairs_manager:
                try:
                    await pairs_manager.handle_sl_tp_triggered(symbol, order_type)
                except Exception as e:
                    print(f"âš ï¸ Error handling SL/TP trigger: {e}")
        
        # CANCELED order - notify user and trigger cleanup
        elif status == 'CANCELED' and order_type in ('STOP', 'TAKE_PROFIT', 'STOP_MARKET', 'TAKE_PROFIT_MARKET'):
            print(f"âš ï¸ SL/TP CANCELED: {symbol} {order_type} by user/system")
            
            # Find which pair this order belongs to
            if pairs_manager:
                for pair_info in pairs_manager.active_pairs.values():
                    if pair_info.position_status != 0 and symbol in [pair_info.symbol1, pair_info.symbol2]:
                        s1, s2 = pair_info.symbol1, pair_info.symbol2
                        
                        # Skip if pair is already being processed for closure
                        # (e.g. bulk close on exchange cancels orders then closes positions)
                        if getattr(pair_info, 'is_trading', False):
                            print(f"â„¹ï¸ {s1}-{s2} already being processed, skipping cancel handler")
                            break
                        
                        other_symbol = s2 if symbol == s1 else s1
                        
                        # Notify user about manual order cancellation
                        cancel_msg = (f"âš ï¸ <b>Order CANCELED:</b> {symbol}\n"
                                      f"Type: {order_type}\n"
                                      f"Pair: {s1}-{s2}\n"
                                      f"â³ Checking pair integrity...")
                        try:
                            await send_tg_notification(cancel_msg)
                        except Exception as e:
                            print(f"âš ï¸ TG notify error: {e}")
                        
                        # Try restoring protection immediately (1 retry), then fallback to leg sync.
                        try:
                            restored = await pairs_manager.restore_protection_for_symbol(symbol, max_attempts=2)
                            if not restored:
                                await pairs_manager._check_leg_synchronization()
                        except Exception as e:
                            print(f"âš ï¸ Leg sync error after cancel: {e}")
                        break
    
    # Check for ALGO_UPDATE (algo order triggered/finished - SL/TP via algo endpoint)
    if event_type == 'ALGO_UPDATE':
        algo_data = msg.get('o', {})
        algo_id = str(algo_data.get('aid', ''))  # Algo order ID (Binance field: "aid")
        algo_status = algo_data.get('X', '')     # Algo Status (Binance field: "X"): NEW, CANCELED, TRIGGERING, TRIGGERED, FINISHED, REJECTED, EXPIRED
        algo_symbol = algo_data.get('s', '')     # Symbol (Binance field: "s")
        algo_type = algo_data.get('o', '')       # Order Type (Binance field: "o"): STOP, TAKE_PROFIT, etc.
        
        print(f"ðŸ“¡ ALGO_UPDATE: {algo_symbol} {algo_type} {algo_status} (algoId={algo_id})")
        
        if algo_status in ('TRIGGERING', 'TRIGGERED') and pairs_manager:
            # Check if this algoId is tracked
            algo_info = pairs_manager.algo_orders.get(algo_id)
            if algo_info:
                order_type = algo_info.get('type', algo_type)
                pair_key = algo_info.get('pair_key')
                symbol = algo_info.get('symbol', algo_symbol)
                
                is_tp = 'TAKE_PROFIT' in order_type.upper() if order_type else False
                tp_or_sl = 'TP' if is_tp else 'SL'
                
                print(f"ðŸŽ¯ Algo {tp_or_sl} triggered: {symbol} (algoId={algo_id})")
                
                try:
                    await pairs_manager.handle_sl_tp_triggered(symbol, order_type)
                    
                    # Clean up all algo orders for this pair
                    if pair_key:
                        to_remove = [aid for aid, info in pairs_manager.algo_orders.items()
                                     if info.get('pair_key') == pair_key]
                        for aid in to_remove:
                            del pairs_manager.algo_orders[aid]
                        print(f"ðŸ—‘ï¸ Cleaned up {len(to_remove)} algo order mappings for pair")
                except Exception as e:
                    print(f"âš ï¸ Error handling algo SL/TP trigger: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"â„¹ï¸ Algo order {algo_id} not tracked (may be from previous session)")
                # Fallback: try to match by symbol
                if algo_status == 'TRIGGERING':
                    try:
                        await pairs_manager.handle_sl_tp_triggered(algo_symbol, algo_type)
                    except Exception as e:
                        print(f"âš ï¸ Fallback algo handler error: {e}")
        
        elif algo_status == 'CANCELED' and pairs_manager:
            # Remove from tracking
            if algo_id in pairs_manager.algo_orders:
                del pairs_manager.algo_orders[algo_id]
                print(f"ðŸ—‘ï¸ Removed canceled algo order {algo_id} from tracking")


if __name__ == '__main__':
    try:
        _configure_console_encoding()
        _install_print_mojibake_fix()
        print("Starting market neutral bot...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user.")

