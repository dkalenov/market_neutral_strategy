import asyncio
import configparser
import binance
import db
import os
from aiogram import Bot, Dispatcher, F, types, BaseMiddleware
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton)


# Initialize bot and dispatcher
bot: Bot = None  # Will be initialized in run()
dp = Dispatcher()

class AuthMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        if isinstance(event, Message):
            print(f"TG: Received message from {event.from_user.id} (@{event.from_user.username}): {event.text}")
            if event.from_user.id not in tg_admins:
                print(f"TG: Unauthorized access from {event.from_user.id}")
                await event.answer(f"Unauthorized. Your ID: {event.from_user.id}")
                return
        elif isinstance(event, CallbackQuery):
            print(f"TG: Received callback from {event.from_user.id} (@{event.from_user.username}): {event.data}")
            if event.from_user.id not in tg_admins:
                print(f"TG: Unauthorized access from {event.from_user.id}")
                await event.answer("Unauthorized", show_alert=True)
                return
        return await handler(event, data)

dp.message.middleware(AuthMiddleware())
dp.callback_query.middleware(AuthMiddleware())

# Load config
config = configparser.ConfigParser()
config.read('market_neutral/config.ini')
# Parse admin list
tg_admins = []

# Function to run the bot
async def init_bot():
    """Initialize TG bot (call this BEFORE pairs_manager.initialize for notifications to work)."""
    global bot
    global tg_admins
    
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
    
    tg_token = os.getenv('TG_TOKEN')
    tg_admins_str = os.getenv('TG_ADMINS', '')
    
    if not tg_token:
        print("ERROR: TG_TOKEN not set in .env file!")
        return False
    
    # Parse admins
    tg_admins = [int(admin_id) for admin_id in tg_admins_str.split(',') if admin_id.strip()]
    
    # Initialize bot
    try:
        bot = Bot(token=tg_token, default=DefaultBotProperties(parse_mode='HTML'))
        print(f"TG: Bot initialized. Token: {tg_token[:10]}...")
        print(f"TG: Authorized admins: {tg_admins}")
        return True
    except Exception as e:
        print(f"TG: Bot initialization failed: {e}")
        return False


async def run(_session, _client: binance.Futures, _pairs_manager):
    # Pass parameters from main
    global bot
    global dp
    global session
    global client
    global pairs_manager
    global tg_admins

    session = _session
    client = _client
    pairs_manager = _pairs_manager

    # Initialize bot if not already done
    if not bot:
        success = await init_bot()
        if not success:
            return

    # Delete webhook
    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception as e:
        print(f"TG: Webhook deletion failed: {e}")

    await send_startup_message()
    
    print("TG: Starting polling...")
    try:
        # Start polling
        await dp.start_polling(bot)
    except Exception as e:
        print(f"TG: Polling failed with an error: {e}")
    finally:
        # Close session
        print("TG: Closing bot session...")
        try:
            await bot.session.close()
            print("TG: Bot session closed.")
        except Exception as e:
            print(f"TG: Session close error: {e}")


# Bot states
class States(StatesGroup):
    main_menu = State()
    trades = State()
    pairs = State()
    settings = State()
    add_pair = State()
    delete_pair = State()
    change_keys = State()
    restart = State()
    blacklist = State()
    hardware_sltp = State()
    close_positions = State()

# Helper function to answer messages or callbacks
async def answer(message: Message | CallbackQuery, text, reply_markup=None):
    if isinstance(message, CallbackQuery):
        await message.answer()
        message = message.message
    await message.answer(text, reply_markup=reply_markup)


# Main menu
@dp.message(Command("start", "menu"))
@dp.message(F.text == "Main Menu")
async def start(message: Message, state: FSMContext):
    # Set state
    await state.set_state(States.main_menu)
    # Create keyboard
    keyboard = ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="Statistics"), KeyboardButton(text="Settings")],
        [KeyboardButton(text="Blacklist"), KeyboardButton(text="🔴 Close Positions")],
        [KeyboardButton(text="Main Menu")]
    ], resize_keyboard=True)
    # Send message
    await answer(message, "Main Menu", reply_markup=keyboard)


# Callback handler for inline "start" button (Main Menu from inline keyboards)
@dp.callback_query(F.data == "start")
async def start_callback(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await state.set_state(States.main_menu)
    # Create keyboard
    keyboard = ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="Statistics"), KeyboardButton(text="Settings")],
        [KeyboardButton(text="Blacklist"), KeyboardButton(text="🔴 Close Positions")],
        [KeyboardButton(text="Main Menu")]
    ], resize_keyboard=True)
    await callback.message.answer("Main Menu", reply_markup=keyboard)
    await callback.answer()




@dp.message(F.text == "Statistics")
@dp.callback_query(F.data == "trades")
async def open_trades(message: Message | CallbackQuery, state: FSMContext):
    await state.set_state(States.trades)
    text = "📊 <b>Open Trades:</b>"
    
    if pairs_manager:
        try:
            # Fetch live positions from exchange for PnL
            positions = await client.get_position_risk()
            pnl_by_symbol = {}
            for pos in positions:
                sym = pos.get('symbol', '')
                qty = abs(float(pos.get('positionAmt', 0)))
                if qty > 0:
                    pnl_by_symbol[sym] = {
                        'pnl': float(pos.get('unRealizedProfit', 0)),
                        'entry': float(pos.get('entryPrice', 0)),
                        'mark': float(pos.get('markPrice', 0)),
                        'side': 'LONG' if float(pos.get('positionAmt', 0)) > 0 else 'SHORT'
                    }
            
            # Get active pairs with PnL
            has_trades = False
            total_pnl = 0.0
            
            for pair_info in pairs_manager.active_pairs.values():
                if pair_info.position_status != 0:
                    has_trades = True
                    s1, s2 = pair_info.symbol1, pair_info.symbol2
                    
                    # Get PnL for each leg
                    pnl1 = pnl_by_symbol.get(s1, {}).get('pnl', 0)
                    pnl2 = pnl_by_symbol.get(s2, {}).get('pnl', 0)
                    pair_pnl = pnl1 + pnl2
                    total_pnl += pair_pnl
                    
                    # Direction
                    direction = "🟢 LONG" if pair_info.position_status == 1 else "🔴 SHORT"
                    pnl_emoji = "🟢" if pair_pnl >= 0 else "🔴"
                    
                    # Mark price and unrealized PnL
                    s1_info = pnl_by_symbol.get(s1, {})
                    s2_info = pnl_by_symbol.get(s2, {})
                    
                    text += f"\n\n<b>{direction}</b> {s1}/{s2}"
                    
                    # Statistical info
                    import math
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
                    pval = getattr(pair_info, 'last_pvalue', 0) or 0
                    hl = getattr(pair_info, 'half_life', 0) or 0
                    if hl > 0:
                        if hl >= 24:
                            hl_d = int(hl // 24)
                            hl_h = int(hl % 24)
                            hl_str = f"{hl_d}d {hl_h}h" if hl_h > 0 else f"{hl_d}d"
                        else:
                            hl_h = int(hl)
                            hl_m = int((hl - hl_h) * 60)
                            hl_str = f"{hl_h}h {hl_m}m" if hl_m > 0 else f"{hl_h}h"
                    else:
                        hl_str = 'N/A'
                    hedge = getattr(pair_info, 'hedge_ratio', 0) or 0
                    
                    text += f"\n  📊 Z: {zscore:+.2f} | β: {beta:.3f} | p: {pval:.4f}"
                    text += f"\n  ⏳ HL: {hl_str} | Hedge: {hedge:.4f}"
                    text += f"\n  {s1}: {s1_info.get('side', '?')} @ {s1_info.get('entry', 0):.4f}"
                    text += f" → {pnl1:+.2f}"
                    text += f"\n  {s2}: {s2_info.get('side', '?')} @ {s2_info.get('entry', 0):.4f}"
                    text += f" → {pnl2:+.2f}"
                    text += f"\n  💰 Pair PnL: {pnl_emoji} <b>{pair_pnl:+.2f} USDT</b>"
            
            if has_trades:
                # Add total PnL summary
                total_emoji = "🟢" if total_pnl >= 0 else "🔴"
                text += f"\n\n━━━━━━━━━━━━━━━━"
                text += f"\n💎 <b>Total Unrealized: {total_emoji} {total_pnl:+.2f} USDT</b>"
                
                keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="🔄 Refresh", callback_data="trades")]
                ])
                await answer(message, text, reply_markup=keyboard)
            else:
                await answer(message, "📊 No open trades")
        except Exception as e:
            await answer(message, f"⚠️ Error fetching data: {e}")
    else:
        await answer(message, "⚠️ Pairs manager not initialized")

# Settings
@dp.message(F.text == "Settings")
@dp.callback_query(F.data == "settings")
async def settings(message: Message, state: FSMContext):
    # Load config from DB
    conf = await db.load_config()
    # Set state
    await state.set_state(States.settings)
    # Get test mode status
    test_mode = getattr(conf, 'test_mode', False)
    if isinstance(test_mode, str):
        test_mode = test_mode.lower() in ('true', '1', 'yes')
    
    # Get trade mode status
    trade_mode = getattr(conf, 'trade_mode', True)
    if isinstance(trade_mode, str):
        trade_mode = trade_mode.lower() in ('true', '1', 'yes')
    
    # Create keyboard
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"🔄 Trading: {'ON ✅' if trade_mode else 'OFF ❌'}", callback_data="toggle_trade_mode")],
        [InlineKeyboardButton(text="Strategy Settings", callback_data="strategy_settings")],
        [InlineKeyboardButton(text="Risk Settings", callback_data="risk_settings")],
        [InlineKeyboardButton(text="📋 Manage Blacklist", callback_data="manage_blacklist")],
        [InlineKeyboardButton(text=f"🧪 Test Mode: {'ON ✅' if test_mode else 'OFF ❌'}", callback_data="toggle_test_mode")],
        [InlineKeyboardButton(text="Change Keys/Tokens", callback_data="change_keys")],
        [InlineKeyboardButton(text="Restart Bot", callback_data="restart")]
    ])
    # Format message
    api_key = f"{conf.api_key[:4]}...{conf.api_key[-4:]}" if conf.api_key else "Missing"
    tg_token = f"{conf.tg_token[:10]}..." if conf.tg_token else "Missing"
    
    tf = conf.timeframe if conf.timeframe else "1h (default)"
    win = conf.window_size if conf.window_size else "200 (default)"
    
    cap = conf.capital if conf.capital else "1000 (default)"
    lev = conf.leverage if conf.leverage else "20 (default)"
    risk = f"{conf.max_notional_pct*100}%" if conf.max_notional_pct else "10% (default)"
    z_in = conf.z_entry if conf.z_entry else "1.9 (default)"
    z_in_max = conf.z_entry_max if conf.z_entry_max else "2.5 (default)"
    z_ex = conf.z_exit if conf.z_exit is not None else "0.0 (default)"
    z_out = conf.z_stop if conf.z_stop else "4.0 (default)"

    text = (f"<b>Basic Settings:</b>\n"
            f"API KEY: <b>{api_key}</b>\n"
            f"TG_TOKEN: <b>{tg_token}</b>\n\n"
            f"<b>Strategy Parameters:</b>\n"
            f"Timeframe: <b>{tf}</b>\n"
            f"Window: <b>{win}</b>\n\n"
            f"<b>Risk Management:</b>\n"
            f"Capital: <b>{cap} USDT</b>\n"
            f"Leverage: <b>x{lev}</b>\n"
            f"Max Risk/Pair: <b>{risk}</b>\n"
            f"Z-Entry Window: <b>{z_in} - {z_in_max}</b>\n"
            f"Z-Exit: <b>{z_ex}</b> (TP)\n"
            f"Z-Stop: <b>{z_out}</b> (SL)\n")
    # Send message
    await answer(message, text, reply_markup=keyboard)


@dp.callback_query(F.data == "strategy_settings")
async def strategy_settings_menu(callback: CallbackQuery, state: FSMContext):
    conf = await db.load_config()
    hl_min = getattr(conf, 'hl_min_days', 2.0) or 2.0
    hl_max = getattr(conf, 'hl_max_days', 5.0) or 5.0
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Set Timeframe", callback_data="set_timeframe")],
        [InlineKeyboardButton(text="Set Window Size", callback_data="set_window")],
        [InlineKeyboardButton(text=f"⏱️ Half-Life: {hl_min}-{hl_max} days", callback_data="set_half_life")],
        [InlineKeyboardButton(text="Back", callback_data="settings")]
    ])
    await answer(callback, "Choose parameter to change:", reply_markup=keyboard)


@dp.callback_query(F.data == "set_timeframe")
async def set_timeframe(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="timeframe")
    await answer(callback, "Enter new Timeframe (e.g., 15m, 1h, 4h):")

@dp.callback_query(F.data == "set_window")
async def set_window(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="window")
    await answer(callback, "Enter new Window Size (e.g., 200, 300, 400):")

@dp.callback_query(F.data == "set_half_life")
async def set_half_life(callback: CallbackQuery, state: FSMContext):
    conf = await db.load_config()
    hl_min = getattr(conf, 'hl_min_days', 2.0) or 2.0
    hl_max = getattr(conf, 'hl_max_days', 5.0) or 5.0
    await state.set_state(States.settings)
    await state.update_data(waiting_for="half_life")
    await answer(callback, 
        f"⏱️ <b>Half-Life Range</b>\n\n"
        f"Current: {hl_min}-{hl_max} days\n\n"
        f"Enter new range as <code>min-max</code> (e.g., 2-5):\n\n"
        f"• Min: fastest mean-reversion (2+ recommended)\n"
        f"• Max: slowest before capital locked (5-7 typical)")


@dp.callback_query(F.data == "risk_settings")
async def risk_settings_menu(callback: CallbackQuery, state: FSMContext):
    conf = await db.load_config()
    max_pairs = getattr(conf, 'max_active_pairs', 5) or 5
    max_symbols = getattr(conf, 'max_symbols', 150) or 150
    max_idle = getattr(conf, 'max_idle_pairs', 150) or 150
    idle_timeout = getattr(conf, 'idle_timeout_hours', 48) or 48
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Capital (USDT)", callback_data="set_capital")],
        [InlineKeyboardButton(text="Leverage (x)", callback_data="set_leverage")],
        [InlineKeyboardButton(text="Max % Per Pair", callback_data="set_max_notional")],
        [InlineKeyboardButton(text="Z-Entry", callback_data="set_z_entry")],
        [InlineKeyboardButton(text="Z-Entry Max", callback_data="set_z_entry_max")],
        [InlineKeyboardButton(text="Z-Exit", callback_data="set_z_exit")],
        [InlineKeyboardButton(text="Z-Stop", callback_data="set_z_stop")],
        [InlineKeyboardButton(text="🛡️ Hardware SL/TP", callback_data="hardware_sltp")],
        [InlineKeyboardButton(text=f"📊 Max Pairs: {max_pairs}", callback_data="set_max_pairs")],
        [InlineKeyboardButton(text=f"📈 Max Symbols: {max_symbols}", callback_data="set_max_symbols")],
        [InlineKeyboardButton(text=f"🗑️ Max Idle: {max_idle}", callback_data="set_max_idle"),
         InlineKeyboardButton(text=f"⏰ Timeout: {idle_timeout}h", callback_data="set_idle_timeout")],
        [InlineKeyboardButton(text="Back", callback_data="settings")]
    ])
    await answer(callback, "Risk Management Settings:\n\n"
                           "<b>Capital</b>: Your purchasing power.\n"
                           "<b>Hardware SL/TP</b>: ATR-based stop orders.\n"
                           f"<b>Max Pairs</b>: {max_pairs} concurrent trades.\n"
                           f"<b>Max Symbols</b>: Filter top {max_symbols} by volume.\n\n"
                           f"<b>Idle Pair Cleanup:</b>\n"
                           f"  • Max idle pairs: {max_idle}\n"
                           f"  • Timeout: {idle_timeout}h", reply_markup=keyboard)

@dp.callback_query(F.data == "set_capital")
async def set_capital_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="capital")
    await answer(callback, "Enter Capital in USDT (Effective deposit including leverage, e.g., 10000):")

@dp.callback_query(F.data == "set_leverage")
async def set_leverage_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="leverage")
    await answer(callback, "Enter Leverage, e.g., 5, 10, 20:")

@dp.callback_query(F.data == "set_max_notional")
async def set_max_notional_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="max_notional")
    await answer(callback, "Enter Max % per pair (e.g., 0.1 for 10%):")

@dp.callback_query(F.data == "set_z_entry")
async def set_z_entry_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="z_entry")
    await answer(callback, "Enter Z-Score for ENTRY (lower bound of entry window, e.g., 1.9):")

@dp.callback_query(F.data == "set_z_exit")
async def set_z_exit_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="z_exit")
    await answer(callback, "Enter Z-Score for EXIT (Take Profit), usually 0:")

@dp.callback_query(F.data == "set_z_entry_max")
async def set_z_entry_max_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="z_entry_max")
    await answer(callback, "Enter Z-Score MAX for ENTRY (upper bound of entry window, e.g., 2.5):")

@dp.callback_query(F.data == "set_z_stop")
async def set_z_stop_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="z_stop")
    await answer(callback, "Enter Z-Score for STOP (e.g., 4.0):")

@dp.callback_query(F.data == "set_max_pairs")
async def set_max_pairs_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="max_active_pairs")
    await answer(callback, "Enter maximum number of concurrent pairs (e.g., 5):")

@dp.callback_query(F.data == "set_max_symbols")
async def set_max_symbols_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="max_symbols")
    await answer(callback, "Enter maximum number of symbols by volume (50-300, e.g., 150):\n\n⚠️ Requires restart to take effect.")

@dp.callback_query(F.data == "set_max_idle")
async def set_max_idle_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="max_idle_pairs")
    await answer(callback, "🗑️ <b>Max Idle Pairs</b>\n\nIdle pairs = co-integrated but no open position.\n\nEnter max number (e.g., 150):")

@dp.callback_query(F.data == "set_idle_timeout")
async def set_idle_timeout_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="idle_timeout_hours")
    await answer(callback, "⏰ <b>Idle Timeout (hours)</b>\n\nRemove pairs idle for more than X hours.\n\nEnter hours (e.g., 48):")

@dp.callback_query(F.data == "toggle_trade_mode")
async def toggle_trade_mode_cb(callback: CallbackQuery, state: FSMContext):
    conf = await db.load_config()
    current = getattr(conf, 'trade_mode', True)
    if isinstance(current, str):
        current = current.lower() in ('true', '1', 'yes')
    
    new_value = 'false' if current else 'true'
    await db.config_update(trade_mode=new_value)
    
    # Reload config in pairs_manager
    if pairs_manager and hasattr(pairs_manager, 'config'):
        pairs_manager.config = await db.load_config()
    
    if new_value == 'true':
        status = "🔄 Trading ENABLED - New positions will be opened"
    else:
        status = "🔄 Trading DISABLED - No new positions will be opened"
    
    await callback.answer(status, show_alert=True)
    # Refresh menu - go back to settings
    await settings(callback, state)

@dp.callback_query(F.data == "toggle_test_mode")
async def toggle_test_mode_cb(callback: CallbackQuery, state: FSMContext):
    conf = await db.load_config()
    current = getattr(conf, 'test_mode', False)
    if isinstance(current, str):
        current = current.lower() in ('true', '1', 'yes')
    
    new_value = 'false' if current else 'true'
    await db.config_update(test_mode=new_value)
    
    # Reload config in pairs_manager
    if pairs_manager and hasattr(pairs_manager, 'config'):
        pairs_manager.config = await db.load_config()
    
    status = "🧪 Test Mode ENABLED" if new_value == 'true' else "🧪 Test Mode DISABLED"
    await callback.answer(status)
    # Refresh menu - go back to settings
    await settings(callback, state)


# === HARDWARE SL/TP MENU ===
@dp.callback_query(F.data == "hardware_sltp")
async def hardware_sltp_menu(callback: CallbackQuery, state: FSMContext):
    conf = await db.load_config()
    sl_atr = getattr(conf, 'sl_atr_mult', 2.5) or 2.5
    sl_min = getattr(conf, 'sl_min_pct', 0.10) or 0.10
    sl_max = getattr(conf, 'sl_max_pct', 0.30) or 0.30
    tp_atr = getattr(conf, 'tp_atr_mult', 4.0) or 4.0
    tp_min = getattr(conf, 'tp_min_pct', 0.15) or 0.15
    tp_max = getattr(conf, 'tp_max_pct', 0.50) or 0.50
    cb_pct = getattr(conf, 'circuit_breaker_pct', 0.20) or 0.20
    p_val = getattr(conf, 'p_value_threshold', 0.05) or 0.05
    bump = getattr(conf, 'min_order_bump', 1.5) or 1.5
    beta_crit = getattr(conf, 'beta_critical', 1.0) or 1.0
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"SL ATR Mult: {sl_atr}", callback_data="set_sl_atr")],
        [InlineKeyboardButton(text=f"SL Min: {sl_min*100:.0f}%", callback_data="set_sl_min"),
         InlineKeyboardButton(text=f"SL Max: {sl_max*100:.0f}%", callback_data="set_sl_max")],
        [InlineKeyboardButton(text=f"TP ATR Mult: {tp_atr}", callback_data="set_tp_atr")],
        [InlineKeyboardButton(text=f"TP Min: {tp_min*100:.0f}%", callback_data="set_tp_min"),
         InlineKeyboardButton(text=f"TP Max: {tp_max*100:.0f}%", callback_data="set_tp_max")],
        [InlineKeyboardButton(text=f"Circuit Breaker: {cb_pct*100:.0f}% notional", callback_data="set_circuit_breaker")],
        [InlineKeyboardButton(text=f"P-Value Threshold: {p_val}", callback_data="set_p_value")],
        [InlineKeyboardButton(text=f"Min Order Bump: {bump}x", callback_data="set_min_bump")],
        [InlineKeyboardButton(text=f"⚠️ Beta Critical: {beta_crit}", callback_data="set_beta_critical")],
        [InlineKeyboardButton(text="Back", callback_data="risk_settings")]
    ])
    await answer(callback, "🛡️ <b>Hardware SL/TP Settings</b>\n\n"
                           "<b>Stop-Loss (ATR-based):</b>\n"
                           f"  ATR Multiplier: {sl_atr}x\n"
                           f"  Range: {sl_min*100:.0f}% - {sl_max*100:.0f}%\n\n"
                           "<b>Take-Profit (ATR-based):</b>\n"
                           f"  ATR Multiplier: {tp_atr}x\n"
                           f"  Range: {tp_min*100:.0f}% - {tp_max*100:.0f}%\n\n"
                           f"<b>Circuit Breaker:</b> {cb_pct*100:.0f}% of margin (={cb_pct/(conf.leverage or 20)*100:.1f}% notional at {conf.leverage or 20}x)\n"
                           f"<b>Min Order Bump:</b> {bump}x max increase\n"
                           f"<b>Beta Critical:</b> {beta_crit} (force-close if |β| ≥ this)", reply_markup=keyboard)

@dp.callback_query(F.data == "set_sl_atr")
async def set_sl_atr_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.hardware_sltp)
    await state.update_data(waiting_for="sl_atr_mult")
    await answer(callback, "Enter SL ATR Multiplier (e.g., 2.5):")

@dp.callback_query(F.data == "set_sl_min")
async def set_sl_min_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.hardware_sltp)
    await state.update_data(waiting_for="sl_min_pct")
    await answer(callback, "Enter Min SL % as decimal (e.g., 0.10 for 10%):")

@dp.callback_query(F.data == "set_sl_max")
async def set_sl_max_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.hardware_sltp)
    await state.update_data(waiting_for="sl_max_pct")
    await answer(callback, "Enter Max SL % as decimal (e.g., 0.30 for 30%):")

@dp.callback_query(F.data == "set_tp_atr")
async def set_tp_atr_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.hardware_sltp)
    await state.update_data(waiting_for="tp_atr_mult")
    await answer(callback, "Enter TP ATR Multiplier (e.g., 4.0):")

@dp.callback_query(F.data == "set_tp_min")
async def set_tp_min_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.hardware_sltp)
    await state.update_data(waiting_for="tp_min_pct")
    await answer(callback, "Enter Min TP % as decimal (e.g., 0.15 for 15%):")

@dp.callback_query(F.data == "set_tp_max")
async def set_tp_max_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.hardware_sltp)
    await state.update_data(waiting_for="tp_max_pct")
    await answer(callback, "Enter Max TP % as decimal (e.g., 0.50 for 50%):")

@dp.callback_query(F.data == "set_circuit_breaker")
async def set_circuit_breaker_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.hardware_sltp)
    await state.update_data(waiting_for="circuit_breaker_pct")
    conf = await db.load_config()
    lev = conf.leverage if conf.leverage else 20
    await answer(callback, 
        f"Enter Circuit Breaker as % of <b>notional</b> (total position size).\n\n"
        f"Examples:\n"
        f"  0.20 = 20% → close if pair loses 20% of position value\n"
        f"  0.10 = 10% → more aggressive protection\n\n"
        f"Leverage does NOT affect this threshold.\n"
        f"Enter as decimal (e.g., 0.20 for 20%):")

@dp.callback_query(F.data == "set_p_value")
async def set_p_value_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.hardware_sltp)
    await state.update_data(waiting_for="p_value_threshold")
    await answer(callback, "Enter P-Value threshold (e.g., 0.05). Pair closes if p-value > this:")

@dp.callback_query(F.data == "set_min_bump")
async def set_min_bump_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.hardware_sltp)
    await state.update_data(waiting_for="min_order_bump")
    await answer(callback, "Enter Max Order Bump ratio (e.g., 1.5 means 50% max increase):")

@dp.callback_query(F.data == "set_beta_critical")
async def set_beta_critical_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.hardware_sltp)
    await state.update_data(waiting_for="beta_critical")
    await answer(callback, "⚠️ <b>Beta Critical Threshold</b>\n\n"
                           "Force-close position if |beta| ≥ this value, regardless of PnL.\n\n"
                           "• 1.0 = pair moves MORE than BTC itself\n"
                           "• 0.5 = moderate directional exposure\n\n"
                           "Enter value (e.g., 1.0):")

@dp.message(StateFilter(States.hardware_sltp))
async def process_hardware_sltp_settings(message: Message, state: FSMContext):
    data = await state.get_data()
    waiting_for = data.get("waiting_for")
    value = message.text.strip()
    
    try:
        float_value = float(value)
        if float_value <= 0:
            await message.answer("Value must be positive!")
            return
        
        await db.config_update(**{waiting_for: str(float_value)})
        await message.answer(f"✅ {waiting_for} updated to {float_value}")
        
        # Reload pairs_manager config if available
        if pairs_manager and hasattr(pairs_manager, 'config'):
            pairs_manager.config = await db.load_config()
        
        await state.clear()
        # Return to hardware_sltp menu
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Back to Hardware SL/TP", callback_data="hardware_sltp")],
            [InlineKeyboardButton(text="Main Menu", callback_data="start")]
        ])
        await message.answer("Setting saved!", reply_markup=keyboard)
    except ValueError:
        await message.answer("Invalid input! Please enter a valid number.")


# Handle settings input
@dp.message(States.settings)
async def process_strategy_settings(message: Message, state: FSMContext):
    data = await state.get_data()
    waiting_for = data.get("waiting_for")
    
    if not waiting_for:
        return
    
    value = message.text.strip()
    
    try:
        if waiting_for == "timeframe":
            valid_tfs = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']
            if value not in valid_tfs:
                await answer(message, f"Invalid format. Allowed: {', '.join(valid_tfs)}")
                return
            await db.config_update(timeframe=value)
            await answer(message, f"Timeframe changed to <b>{value}</b>.")
            
        elif waiting_for == "window":
            if not value.isdigit() or int(value) < 50:
                await answer(message, "Value must be a number > 50.")
                return
            await db.config_update(window_size=int(value))
            await answer(message, f"Window Size changed to <b>{value}</b>.")

        elif waiting_for == "capital":
            val = float(value)
            await db.config_update(capital=val)
            await answer(message, f"Capital: <b>{val} USDT</b>")

        elif waiting_for == "leverage":
            val = int(value)
            if val < 1 or val > 125:
                await answer(message, "Leverage must be between 1 and 125.")
                return
            await db.config_update(leverage=val)
            await answer(message, f"Leverage: <b>x{val}</b>\n(Applied to new trades)")

        elif waiting_for == "max_notional":
            val = float(value)
            if val <= 0 or val > 1:
                await answer(message, "Value must be between 0.01 and 1.0")
                return
            await db.config_update(max_notional_pct=val)
            await answer(message, f"Max Risk Per Pair: <b>{val*100}%</b>")

        elif waiting_for == "z_entry":
            val = float(value)
            await db.config_update(z_entry=val)
            await answer(message, f"Z-Entry: <b>{val}</b>")

        elif waiting_for == "z_exit":
            val = float(value)
            await db.config_update(z_exit=val)
            await answer(message, f"Z-Exit: <b>{val}</b> (Take Profit)")

        elif waiting_for == "z_entry_max":
            val = float(value)
            await db.config_update(z_entry_max=val)
            await answer(message, f"Z-Entry Max: <b>{val}</b> (Upper bound for entry window)")

        elif waiting_for == "z_stop":
            val = float(value)
            await db.config_update(z_stop=val)
            await answer(message, f"Z-Stop: <b>{val}</b>")

        elif waiting_for == "max_active_pairs":
            val = int(value)
            if val < 1 or val > 100:
                await answer(message, "Value must be between 1 and 100.")
                return
            await db.config_update(max_active_pairs=val)
            await answer(message, f"Max Active Pairs: <b>{val}</b>")
            
        elif waiting_for == "half_life":
            # Parse "min-max" format
            parts = value.replace(' ', '').split('-')
            if len(parts) != 2:
                await answer(message, "Error: use format <code>min-max</code> (e.g., 2-5)")
                return
            min_days = float(parts[0])
            max_days = float(parts[1])
            if min_days < 0.5 or max_days < min_days or max_days > 30:
                await answer(message, "Error: min must be ≥0.5, max ≥ min, max ≤30")
                return
            await db.config_update(hl_min_days=min_days, hl_max_days=max_days)
            await answer(message, f"⏱️ Half-Life: <b>{min_days}-{max_days} days</b>")
        
        elif waiting_for == "max_symbols":
            val = int(value)
            if val < 50 or val > 300:
                await answer(message, "Value must be between 50 and 300.")
                return
            await db.config_update(max_symbols=val)
            await answer(message, f"📈 Max Symbols: <b>{val}</b>\n⚠️ Requires restart to take effect.")
        
        elif waiting_for == "max_idle_pairs":
            val = int(value)
            if val < 10 or val > 500:
                await answer(message, "Value must be between 10 and 500.")
                return
            await db.config_update(max_idle_pairs=val)
            await answer(message, f"🗑️ Max Idle Pairs: <b>{val}</b>")
        
        elif waiting_for == "idle_timeout_hours":
            val = float(value)
            if val < 1 or val > 168:  # 1 hour to 1 week
                await answer(message, "Value must be between 1 and 168 hours.")
                return
            await db.config_update(idle_timeout_hours=val)
            await answer(message, f"⏰ Idle Timeout: <b>{val}h</b>")
            
        await answer(message, "<b>IMPORTANT:</b> Restart the bot to apply some settings.")

    except ValueError:
        await answer(message, "Error: enter a valid number.")
        return
    
    await state.update_data(waiting_for=None)
    await settings(message, state)


# Change keys
@dp.callback_query(States.settings, F.data == "change_keys")
async def change_keys(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.change_keys)
    await state.update_data({})
    await callback.answer()
    await callback.message.answer("Enter API KEY (or 'skip' to skip):")

@dp.message(States.change_keys)
async def change_keys_value(message: Message, state: FSMContext):
    data = await state.get_data()
    
    if 'api_key' not in data:
        if message.text.lower() != 'skip':
            await state.update_data(api_key=message.text)
        else:
            await state.update_data(api_key=None)
        await answer(message, "Enter SECRET KEY (or 'skip' to skip):")
        await message.delete()
        return
    
    if 'api_secret' not in data:
        if message.text.lower() != 'skip':
            await state.update_data(api_secret=message.text)
        else:
            await state.update_data(api_secret=None)
        await answer(message, "Enter TG TOKEN (or 'skip' to skip):")
        await message.delete()
        return

    if 'tg_token' not in data:
        if message.text.lower() != 'skip':
            await state.update_data(tg_token=message.text)
        else:
            await state.update_data(tg_token=None)
        await answer(message, "Enter TG ADMINS (comma separated, or 'skip' to skip):")
        await message.delete()
        return
    
    if 'tg_admins' not in data:
        if message.text.lower() != 'skip':
            await state.update_data(tg_admins=message.text)
        else:
            await state.update_data(tg_admins=None)
        await answer(message, "Enter TG CHANNEL ID for trade notifications (or 'skip' to skip):")
        await message.delete()
        return
    
    if 'tg_channel' not in data:
        if message.text.lower() != 'skip':
            await state.update_data(tg_channel=message.text)
        else:
            await state.update_data(tg_channel=None)
        
        new_conf = await state.get_data()
        update_data = {k: v for k, v in new_conf.items() if v is not None}
        
        if update_data:
            await db.config_update(**update_data)
            await answer(message, "Keys and tokens changed successfully. Restarting bot.")
            await restart_yes(message, state)
        else:
            await answer(message, "No changes made.")
            await settings(message, state)


@dp.callback_query(F.data == "restart")
async def restart(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.restart)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Yes", callback_data=f"restart_yes")],
        [InlineKeyboardButton(text="No", callback_data="settings")]
    ])
    await callback.answer()
    await callback.message.answer(f"Are you sure you want to restart the bot?", reply_markup=keyboard)

@dp.callback_query(States.restart, F.data == "restart_yes")
async def restart_yes(callback: CallbackQuery | Message, state: FSMContext):
    try:
        await answer(callback, 'Restarting...')
    finally:
        os._exit(0)


# --- Blacklist Management ---

@dp.message(F.text == "Blacklist")
@dp.callback_query(F.data == "manage_blacklist")
async def blacklist_menu(event: Message | CallbackQuery, state: FSMContext):
    await state.set_state(States.blacklist)
    conf = await db.load_config()
    current_bl = conf.blacklist if conf.blacklist else ""
    split_list = [s.strip() for s in current_bl.split(',') if s.strip()]
    full_bl_str = ", ".join(sorted(split_list)) if split_list else "(empty)"

    text = (
        "<b>Blacklist Management</b>\n\n"
        "Symbols in this list are ignored during pair discovery.\n\n"
        f"<b>Current Blacklist:</b>\n<code>{full_bl_str}</code>\n\n"
        "To <b>add</b> symbols, send them separated by commas (e.g., PEPE, FLOKI).\n"
        "To <b>remove</b> symbols, send them with minus (e.g., -PEPE, -FLOKI).\n"
        "To <b>clear</b> list, type 'clear'."
    )
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Clear All", callback_data="bl_clear")],
        [InlineKeyboardButton(text="Back", callback_data="settings")]
    ])
    
    if isinstance(event, CallbackQuery):
        await event.answer()
        try:
            await event.message.edit_text(text, reply_markup=keyboard, parse_mode='HTML')
        except:
            await event.message.answer(text, reply_markup=keyboard, parse_mode='HTML')
    else:
        await event.answer(text, reply_markup=keyboard, parse_mode='HTML')

@dp.message(StateFilter(States.blacklist))
async def process_blacklist_update(message: Message, state: FSMContext):
    # Skip processing for menu buttons - let other handlers deal with them
    menu_buttons = ["Statistics", "Settings", "Blacklist", "Main Menu", "🔴 Close Positions"]
    if message.text in menu_buttons:
        await state.clear()  # Exit blacklist state
        await answer(message, "No changes made.")
        return
    
    if message.text.lower() == 'clear':
        await db.config_update(blacklist="")
        await answer(message, "Blacklist cleared.")
    else:
        conf = await db.load_config()
        existing = set(conf.blacklist.split(',')) if conf.blacklist else set()
        
        inputs = [s.strip().upper() for s in message.text.split(',') if s.strip()]
        to_add = []
        to_remove = []

        for s in inputs:
            if s.startswith('-'):
                # Remove
                clean_s = s[1:]
                if clean_s in existing:
                    existing.remove(clean_s)
                    to_remove.append(clean_s)
            else:
                # Add
                if s not in existing:
                    existing.add(s)
                    to_add.append(s)
        
        updated_list = sorted(list(existing))
        await db.config_update(blacklist=",".join(updated_list))
        
        msg = ""
        if to_add:
            msg += f"Added: {', '.join(to_add)}\n"
        if to_remove:
            msg += f"Removed: {', '.join(to_remove)}\n"
        if not msg:
            msg = "No changes made."
            
        await answer(message, msg)
    
    await blacklist_menu(message, state)

@dp.callback_query(StateFilter(States.blacklist), F.data == "bl_clear")
async def bl_clear_cb(callback: CallbackQuery, state: FSMContext):
    await db.config_update(blacklist="")
    await callback.answer("List cleared")
    await blacklist_menu(callback, state)


# --- Close Positions Menu ---

@dp.message(F.text == "🔴 Close Positions")
@dp.callback_query(F.data == "close_positions")
async def close_positions_menu(event: Message | CallbackQuery, state: FSMContext):
    """Show list of active pairs with open positions - verified against exchange."""
    await state.set_state(States.close_positions)
    
    # Show loading message first (API call can be slow)
    if isinstance(event, CallbackQuery):
        await event.answer("⏳ Loading positions...")
    
    keyboard = []
    has_positions = False
    
    if pairs_manager:
        # Get real positions from exchange (source of truth)
        try:
            positions = await pairs_manager.client.get_position_risk()
            exchange_positions = set()
            for pos in positions:
                if float(pos.get('positionAmt', 0)) != 0:
                    exchange_positions.add(pos['symbol'])
        except Exception as e:
            print(f"Error fetching exchange positions: {e}")
            exchange_positions = set()
        
        for pair_info in pairs_manager.active_pairs.values():
            if pair_info.position_status != 0:
                # Verify BOTH legs exist on exchange
                leg1_exists = pair_info.symbol1 in exchange_positions
                leg2_exists = pair_info.symbol2 in exchange_positions
                
                if leg1_exists or leg2_exists:
                    direction = "LONG" if pair_info.position_status == 1 else "SHORT"
                    status = "⚠️" if not (leg1_exists and leg2_exists) else ""
                    btn_text = f"❌ {status}{pair_info.symbol1}/{pair_info.symbol2} ({direction})"
                    callback_data = f"close_pair:{pair_info.symbol1}:{pair_info.symbol2}"
                    keyboard.append([InlineKeyboardButton(text=btn_text, callback_data=callback_data)])
                    has_positions = True
                else:
                    # Position closed on exchange but not in local state - clean up
                    pair_info.position_status = 0
                    print(f"🧹 Auto-cleaned stale pair: {pair_info.symbol1}-{pair_info.symbol2}")
    
    if has_positions:
        keyboard.append([InlineKeyboardButton(text="🔴 CLOSE ALL POSITIONS", callback_data="close_all_confirm")])
    
    keyboard.append([InlineKeyboardButton(text="🔄 Refresh", callback_data="close_positions")])
    keyboard.append([InlineKeyboardButton(text="⬅️ Back", callback_data="start")])
    
    text = "🔴 <b>Close Positions</b>\n\n"
    if has_positions:
        text += "Select a position to close, or close all:"
    else:
        text += "No open positions on exchange."
    
    if isinstance(event, CallbackQuery):
        await event.answer()
        await event.message.answer(text, reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard))
    else:
        await event.answer(text, reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard))


@dp.callback_query(F.data.startswith("close_pair:"))
async def close_pair_handler(callback: CallbackQuery, state: FSMContext):
    """Close a specific pair position."""
    parts = callback.data.split(":")
    if len(parts) != 3:
        await callback.answer("Invalid pair data", show_alert=True)
        return
    
    s1, s2 = parts[1], parts[2]
    
    if not pairs_manager:
        await callback.answer("Pairs manager not available", show_alert=True)
        return
    
    # Find the pair
    pair_info = None
    for pi in pairs_manager.active_pairs.values():
        if pi.symbol1 == s1 and pi.symbol2 == s2:
            pair_info = pi
            break
    
    if not pair_info or pair_info.position_status == 0:
        await callback.answer("Position not found or already closed", show_alert=True)
        return
    
    await callback.answer(f"Closing {s1}/{s2}...")
    await callback.message.answer(f"⏳ Closing position {s1}/{s2}...")
    
    try:
        pair_info.close_handled = True
        pair_info.is_trading = True
        await pairs_manager._execute_trade(pair_info, 0, close_reason='manual')
        await callback.message.answer(f"✅ Position {s1}/{s2} closed successfully!")
    except Exception as e:
        await callback.message.answer(f"❌ Error closing {s1}/{s2}: {e}")
    
    # Return to close menu
    await close_positions_menu(callback, state)


@dp.callback_query(F.data == "close_all_confirm")
async def close_all_confirm_handler(callback: CallbackQuery, state: FSMContext):
    """Ask for confirmation before closing all positions."""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Yes, Close All", callback_data="close_all_yes")],
        [InlineKeyboardButton(text="❌ Cancel", callback_data="close_positions")]
    ])
    await callback.answer()
    await callback.message.answer(
        "⚠️ <b>Are you sure you want to close ALL positions?</b>\n\n"
        "This will immediately close all active pair trades.",
        reply_markup=keyboard
    )


@dp.callback_query(F.data == "close_all_yes")
async def close_all_yes_handler(callback: CallbackQuery, state: FSMContext):
    """Close all open positions."""
    if not pairs_manager:
        await callback.answer("Pairs manager not available", show_alert=True)
        return
    
    await callback.answer("Closing all positions...")
    await callback.message.answer("⏳ Closing all positions...")
    
    closed = 0
    failed = 0
    
    pairs_to_close = [
        pi for pi in pairs_manager.active_pairs.values() 
        if pi.position_status != 0
    ]
    
    for pair_info in pairs_to_close:
        try:
            pair_info.close_handled = True
            pair_info.is_trading = True
            await pairs_manager._execute_trade(pair_info, 0, close_reason='manual')
            closed += 1
        except Exception as e:
            print(f"Error closing {pair_info.symbol1}-{pair_info.symbol2}: {e}")
            failed += 1
    
    result_msg = f"✅ Closed {closed} positions."
    if failed > 0:
        result_msg += f"\n⚠️ Failed to close {failed} positions."
    
    await callback.message.answer(result_msg)
    await close_positions_menu(callback, state)


async def send_startup_message():
    """
    Sends a startup message to all admins.
    """
    for admin_id in tg_admins:
        try:
            await bot.send_message(admin_id, "Bot started successfully!")
        except Exception as e:
            print(f"Could not send startup message to admin {admin_id}: {e}")