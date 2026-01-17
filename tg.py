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
bot: Bot
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
async def run(_session, _client: binance.Futures, _pairs_manager):
    # Pass parameters from main
    global bot
    global dp
    global session
    global client
    global pairs_manager
    global tg_admins

    # #region agent log
    import os
    import json
    import time
    log_path = r"c:\Users\Dmitrii\Trading strategies\Market_neutral_strategy\.cursor\debug.log"
    def log_instrument(location, message, data=None):
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                entry = {
                    "id": f"log_{int(time.time()*1000)}_tg",
                    "timestamp": int(time.time()*1000),
                    "location": location,
                    "message": message,
                    "data": data or {},
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "TG_BOT_6"
                }
                f.write(json.dumps(entry) + '\n')
        except: pass
    # #endregion

    session = _session
    client = _client
    pairs_manager = _pairs_manager

    log_instrument("tg.py:run", "Starting Telegram bot initialization")

    # Initialize bot
    try:
        bot = Bot(token=config['TG']['TOKEN'], default=DefaultBotProperties(parse_mode='HTML'))
        log_instrument("tg.py:run", "Bot initialized successfully")
    except Exception as e:
        log_instrument("tg.py:run", "Bot initialization failed", {"error": str(e)})
        raise

    # Parse admins
    admins_str = config.get('TG', 'ADMINS', fallback='')
    tg_admins = [int(admin_id) for admin_id in admins_str.split(',') if admin_id]
    log_instrument("tg.py:run", "Parsed admin IDs", {"admins_count": len(tg_admins), "admins": tg_admins})

    # Delete webhook
    print(f"TG: Token used: {config['TG']['TOKEN'][:10]}...")
    print(f"TG: Authorized admins: {tg_admins}")
    print("TG: Deleting webhook...")
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        log_instrument("tg.py:run", "Webhook deleted successfully")
    except Exception as e:
        log_instrument("tg.py:run", "Webhook deletion failed", {"error": str(e)})
        print(f"TG: Webhook deletion failed: {e}")

    print("TG: Starting polling...")
    await send_startup_message()
    try:
        # Start polling
        log_instrument("tg.py:run", "Starting polling")
        await dp.start_polling(bot)
    except Exception as e:
        log_instrument("tg.py:run", "Polling failed", {"error": str(e), "error_type": type(e).__name__})
        print(f"TG: Polling failed with an error: {e}")
    finally:
        # Close session
        print("TG: Closing bot session...")
        try:
            await bot.session.close()
            log_instrument("tg.py:run", "Bot session closed successfully")
            print("TG: Bot session closed.")
        except Exception as e:
            log_instrument("tg.py:run", "Bot session close failed", {"error": str(e)})
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
        [KeyboardButton(text="Statistics"), KeyboardButton(text="Pairs")],
        [KeyboardButton(text="Settings"), KeyboardButton(text="Blacklist")],
        [KeyboardButton(text="Main Menu")]
    ], resize_keyboard=True)
    # Send message
    await answer(message, "Main Menu", reply_markup=keyboard)


# Trading pairs
@dp.message(F.text == "Pairs")
@dp.callback_query(F.data == "pairs")
async def list_pairs(message: Message | CallbackQuery, state: FSMContext):
    # Set state
    await state.set_state(States.pairs)
    # Load pairs from DB
    pairs = await db.get_all_pairs()
    keyboard = []
    # Create keyboard
    for pair in pairs:
        keyboard.append([InlineKeyboardButton(text=f"{pair.symbol1}/{pair.symbol2}", callback_data=f"pair:{pair.id}")])
    # Add pair button
    keyboard.append([InlineKeyboardButton(text="Add Pair", callback_data="add_pair")])
    await answer(message, "Trading Pairs List", reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard))


# Pair edit menu
@dp.callback_query(F.data.startswith("pair:"))
async def pair_menu(callback: CallbackQuery, state: FSMContext):
    pair_id = int(callback.data.split(':')[1])
    await state.update_data(pair_id=pair_id)

    pairs = await db.get_all_pairs()
    pair = next((p for p in pairs if p.id == pair_id), None)

    if not pair:
        await answer(callback, "Pair not found.")
        return

    text = (f"Trading Pair: <b>{pair.symbol1}/{pair.symbol2}</b>\n"
            f"Hedge Ratio: <b>{pair.hedge_ratio}</b>\n"
            f"Half-life: <b>{pair.half_life}</b>")

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Delete Pair", callback_data="delete_pair")],
        [InlineKeyboardButton(text="Back", callback_data="pairs")]
    ])
    await answer(callback, text, reply_markup=keyboard)


# Add pair
@dp.callback_query(States.pairs, F.data == "add_pair")
async def add_pair(callback: CallbackQuery, state: FSMContext):
    # Set state
    await state.set_state(States.add_pair)
    # Send message
    await answer(callback, "Enter pair in format SYMBOL1/SYMBOL2")

# Process add pair
@dp.message(States.add_pair)
async def add_pair_value(message: Message, state: FSMContext):
    try:
        symbol1, symbol2 = message.text.upper().split('/')
    except ValueError:
        await answer(message, "Invalid format. Enter pair as SYMBOL1/SYMBOL2")
        return

    new_pair = db.Pairs(symbol1=symbol1, symbol2=symbol2)
    await db.add_pair(new_pair)
    
    await answer(message, f"Pair <b>{symbol1}/{symbol2}</b> added successfully.")
    await list_pairs(message, state)


# Delete pair
@dp.callback_query(F.data == "delete_pair")
async def delete_pair(callback: CallbackQuery, state: FSMContext):
    # Set state
    await state.set_state(States.delete_pair)
    # Load data
    data = await state.get_data()
    # Create keyboard
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Yes", callback_data="delete_pair_yes"),
         InlineKeyboardButton(text="No", callback_data=f"pair:{data['pair_id']}")]
    ])
    # Send message
    await answer(callback, f"Are you sure you want to delete this pair?", reply_markup=keyboard)

# Confirm delete pair
@dp.callback_query(States.delete_pair, F.data == "delete_pair_yes")
async def delete_pair_yes(callback: CallbackQuery, state: FSMContext):
    # Load state
    data = await state.get_data()
    pair_id = data['pair_id']
    await db.delete_pair(pair_id)
    await answer(callback, f"Pair deleted successfully")
    await list_pairs(callback, state)


@dp.message(F.text == "Statistics")
@dp.callback_query(F.data == "trades")
async def open_trades(message: Message | CallbackQuery, state: FSMContext):
    await state.set_state(States.trades)
    text = "Open Trades:"
    trades = await db.get_open_trades()
    for trade in trades:
        direction = "LONG" if trade.direction == 1 else "SHORT"
        text += (f"\n\n<b>{direction}</b> #{trade.pair_id}\n"
                 f"Symbol 1: <b>{trade.qty1} @ {trade.entry_price_1}</b>\n"
                 f"Symbol 2: <b>{trade.qty2} @ {trade.entry_price_2}</b>\n"
                 f"PNL: <b>{round(trade.pnl, 2)} USDT</b>")
    if "\n" in text:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Refresh", callback_data="trades")]
        ])
        await answer(message, text, reply_markup=keyboard)
    else:
        await answer(message, "No open trades")

# Settings
@dp.message(F.text == "Settings")
@dp.callback_query(F.data == "settings")
async def settings(message: Message, state: FSMContext):
    # Load config from DB
    conf = await db.load_config()
    # Set state
    await state.set_state(States.settings)
    # Create keyboard
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Strategy Settings", callback_data="strategy_settings")],
        [InlineKeyboardButton(text="Risk Settings", callback_data="risk_settings")],
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
    z_in = conf.z_entry if conf.z_entry else "2.0 (default)"
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
            f"Z-Entry: <b>{z_in}</b>\n"
            f"Z-Exit: <b>{z_ex}</b> (TP)\n"
            f"Z-Stop: <b>{z_out}</b> (SL)\n")
    # Send message
    await answer(message, text, reply_markup=keyboard)


@dp.callback_query(F.data == "strategy_settings")
async def strategy_settings_menu(callback: CallbackQuery, state: FSMContext):
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Set Timeframe", callback_data="set_timeframe")],
        [InlineKeyboardButton(text="Set Window Size", callback_data="set_window")],
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

@dp.callback_query(F.data == "risk_settings")
async def risk_settings_menu(callback: CallbackQuery, state: FSMContext):
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Capital (USDT)", callback_data="set_capital")],
        [InlineKeyboardButton(text="Leverage (x)", callback_data="set_leverage")],
        [InlineKeyboardButton(text="Max % Per Pair", callback_data="set_max_notional")],
        [InlineKeyboardButton(text="Z-Entry", callback_data="set_z_entry")],
        [InlineKeyboardButton(text="Z-Exit", callback_data="set_z_exit")],
        [InlineKeyboardButton(text="Z-Stop", callback_data="set_z_stop")],
        [InlineKeyboardButton(text="Back", callback_data="settings")]
    ])
    await answer(callback, "Risk Management Settings:\n\n"
                           "<b>Capital</b>: Your purchasing power (Equity * Leverage).\n"
                           "<b>Leverage</b>: Exchange margin leverage.\n"
                           "<b>Max %</b>: Limit per pair from Capital.", reply_markup=keyboard)

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
    await answer(callback, "Enter Z-Score for ENTRY (e.g., 2.0):")

@dp.callback_query(F.data == "set_z_exit")
async def set_z_exit_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="z_exit")
    await answer(callback, "Enter Z-Score for EXIT (Take Profit), usually 0:")

@dp.callback_query(F.data == "set_z_stop")
async def set_z_stop_cb(callback: CallbackQuery, state: FSMContext):
    await state.set_state(States.settings)
    await state.update_data(waiting_for="z_stop")
    await answer(callback, "Enter Z-Score for STOP (e.g., 4.0):")


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

        elif waiting_for == "z_stop":
            val = float(value)
            await db.config_update(z_stop=val)
            await answer(message, f"Z-Stop: <b>{val}</b>")
            
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
        await event.message.edit_text(text, reply_markup=keyboard)
    else:
        await event.answer(text, reply_markup=keyboard)

@dp.message(States.blacklist)
async def process_blacklist_update(message: Message, state: FSMContext):
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

@dp.callback_query(States.blacklist, F.data == "bl_clear")
async def bl_clear_cb(callback: CallbackQuery, state: FSMContext):
    await db.config_update(blacklist="")
    await callback.answer("List cleared")
    await blacklist_menu(callback, state)


async def send_startup_message():
    """
    Sends a startup message to all admins.
    """
    for admin_id in tg_admins:
        try:
            await bot.send_message(admin_id, "Bot started successfully!")
        except Exception as e:
            print(f"Could not send startup message to admin {admin_id}: {e}")