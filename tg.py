import asyncio
import configparser
import binance
import db
import os
from aiogram import Bot, Dispatcher, F, types
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton)


# создаем бота и диспетчер
bot: Bot
dp = Dispatcher()
# загружаем конфиг
config = configparser.ConfigParser()
config.read('market_neutral/config.ini')
# парсим список админов
tg_admins = []

# фунция для запуска бота
async def run(_session, _client: binance.Futures, _pairs_manager):
    # передаем функции и параметры из файла main
    global bot
    global dp
    global session
    global client
    global pairs_manager
    global tg_admins
    
    session = _session
    client = _client
    pairs_manager = _pairs_manager
    
    # инициализируем бота
    bot = Bot(token=config['TG']['TOKEN'], default=DefaultBotProperties(parse_mode='HTML'))
    
    # парсим админов
    tg_admins = [int(admin_id) for admin_id in config['TG']['ADMINS'].split(',') if admin_id]
    # удаляем старые сообщения
    print("TG: Deleting webhook...")
    await bot.delete_webhook(drop_pending_updates=True)
    print("TG: Starting polling...")
    await send_startup_message()
    try:
        # запускаем бота
        await dp.start_polling(bot)
    except Exception as e:
        print(f"TG: Polling failed with an error: {e}")
    finally:
        # закрываем сессию
        print("TG: Closing bot session...")
        await bot.session.close()
        print("TG: Bot session closed.")


# класс для хранения состояний бота
class States(StatesGroup):
    main_menu = State()
    trades = State()
    pairs = State()
    settings = State()
    add_pair = State()
    delete_pair = State()
    change_keys = State()
    restart = State()

# функция для ответа на сообщение или коллбек
async def answer(message: Message | CallbackQuery, text, reply_markup=None):
    if isinstance(message, CallbackQuery):
        await message.answer()
        message = message.message
    await message.answer(text, reply_markup=reply_markup)


# функция для пропуска команд от других людей (не от админов)
@dp.message(~F.from_user.id.in_(tg_admins))
async def skip(_):
    pass


# главное меню
@dp.message(Command("start", "menu"))
@dp.message(F.text == "Главное меню")
async def start(message: Message, state: FSMContext):
    # устанавливаем состояние
    await state.set_state(States.main_menu)
    # создаем клавиатуру
    keyboard = ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="Открытые сделки"), KeyboardButton(text="Торговые пары")],
        [KeyboardButton(text="Настройки"), KeyboardButton(text="Главное меню")]
    ], resize_keyboard=True)
    # отправляем сообщение
    await answer(message, "Главное меню", reply_markup=keyboard)


# торговые пары
@dp.message(F.text == "Торговые пары")
@dp.callback_query(F.data == "pairs")
async def list_pairs(message: Message | CallbackQuery, state: FSMContext):
    # устанавливаем состояние
    await state.set_state(States.pairs)
    # загружаем список торговых пар из базы данных
    pairs = await db.get_all_pairs()
    keyboard = []
    # создаем клавиатуру
    for pair in pairs:
        keyboard.append([InlineKeyboardButton(text=f"{pair.symbol1}/{pair.symbol2}", callback_data=f"pair:{pair.id}")])
    # добавляем кнопку для добавления пары
    keyboard.append([InlineKeyboardButton(text="Добавить пару", callback_data="add_pair")])
    await answer(message, "Список торговых пар", reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard))


# меню для редактирования пары
@dp.callback_query(F.data.startswith("pair:"))
async def pair_menu(callback: CallbackQuery, state: FSMContext):
    pair_id = int(callback.data.split(':')[1])
    await state.update_data(pair_id=pair_id)

    pairs = await db.get_all_pairs()
    pair = next((p for p in pairs if p.id == pair_id), None)

    if not pair:
        await answer(callback, "Пара не найдена.")
        return

    text = (f"Торговая пара: <b>{pair.symbol1}/{pair.symbol2}</b>\n"
            f"Hedge Ratio: <b>{pair.hedge_ratio}</b>\n"
            f"Half-life: <b>{pair.half_life}</b>")

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Удалить пару", callback_data="delete_pair")],
        [InlineKeyboardButton(text="Назад", callback_data="pairs")]
    ])
    await answer(callback, text, reply_markup=keyboard)


# добавление пары
@dp.callback_query(States.pairs, F.data == "add_pair")
async def add_pair(callback: CallbackQuery, state: FSMContext):
    # устанавливаем состояние
    await state.set_state(States.add_pair)
    # отправляем сообщение
    await answer(callback, "Введите пару в формате SYMBOL1/SYMBOL2")

# добавление пары
@dp.message(States.add_pair)
async def add_pair_value(message: Message, state: FSMContext):
    try:
        symbol1, symbol2 = message.text.upper().split('/')
    except ValueError:
        await answer(message, "Неверный формат. Введите пару в формате SYMBOL1/SYMBOL2")
        return

    # ToDo: Add check if symbols exist on exchange
    
    new_pair = db.Pairs(symbol1=symbol1, symbol2=symbol2)
    await db.add_pair(new_pair)
    
    await answer(message, f"Пара <b>{symbol1}/{symbol2}</b> успешно добавлена.")
    await list_pairs(message, state)


# удаление пары
@dp.callback_query(F.data == "delete_pair")
async def delete_pair(callback: CallbackQuery, state: FSMContext):
    # устанавливаем состояние
    await state.set_state(States.delete_pair)
    # загружаем данные
    data = await state.get_data()
    # создаем клавиатуру
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Да", callback_data="delete_pair_yes"),
         InlineKeyboardButton(text="Нет", callback_data=f"pair:{data['pair_id']}")]
    ])
    # отправляем сообщение
    await answer(callback, f"Вы уверены, что хотите удалить эту пару?", reply_markup=keyboard)

# удаление пары
@dp.callback_query(States.delete_pair, F.data == "delete_pair_yes")
async def delete_pair_yes(callback: CallbackQuery, state: FSMContext):
    # загружаем состояние
    data = await state.get_data()
    pair_id = data['pair_id']
    await db.delete_pair(pair_id)
    await answer(callback, f"Пара успешно удалена")
    await list_pairs(callback, state)


@dp.message(F.text == "Открытые сделки")
@dp.callback_query(F.data == "trades")
async def open_trades(message: Message | CallbackQuery, state: FSMContext):
    await state.set_state(States.trades)
    text = "Открытые сделки:"
    trades = await db.get_open_trades()
    for trade in trades:
        direction = "LONG" if trade.direction == 1 else "SHORT"
        text += (f"\n\n<b>{direction}</b> #{trade.pair_id}\n"
                 f"Symbol 1: <b>{trade.qty1} @ {trade.entry_price_1}</b>\n"
                 f"Symbol 2: <b>{trade.qty2} @ {trade.entry_price_2}</b>\n"
                 f"PNL: <b>{round(trade.pnl, 2)} USDT</b>")
    if "\n" in text:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Обновить", callback_data="trades")]
        ])
        await answer(message, text, reply_markup=keyboard)
    else:
        await answer(message, "Открытых сделок нет")

# настройки
@dp.message(F.text == "Настройки")
@dp.callback_query(F.data == "settings")
async def settings(message: Message, state: FSMContext):
    # загружаем настройки из базы данных
    conf = await db.load_config()
    # устанавливаем состояние
    await state.set_state(States.settings)
    # создаем клавиатуру
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Изменить ключи/токены", callback_data="change_keys")],
        [InlineKeyboardButton(text="Перезагрузить бота", callback_data="restart")]
    ])
    # формируем сообщение
    api_key = f"{conf.api_key[:4]}...{conf.api_key[-4:]}" if conf.api_key else "Отсутствует"
    tg_token = f"{conf.tg_token[:10]}..." if conf.tg_token else "Отсутствует"

    text = (f"Основные настройки:\n"
            f"API KEY: <b>{api_key}</b>\n"
            f"TG_TOKEN: <b>{tg_token}</b>\n"
            f"TG_ADMINS: <b>{conf.tg_admins}</b>\n")
    # отправляем сообщение
    await answer(message, text, reply_markup=keyboard)


# изменение ключей
@dp.callback_query(States.settings, F.data == "change_keys")
async def change_keys(callback: CallbackQuery, state: FSMContext):
    # устанавливаем состояние для изменения ключей
    await state.set_state(States.change_keys)
    await state.update_data({})
    # отвечаем на callback
    await callback.answer()
    # запрашиваем API KEY
    await callback.message.answer("Введите API KEY (или 'skip' для пропуска):")

# функция для изменения ключей
@dp.message(States.change_keys)
async def change_keys_value(message: Message, state: FSMContext):
    data = await state.get_data()
    
    if 'api_key' not in data:
        if message.text.lower() != 'skip':
            await state.update_data(api_key=message.text)
        else:
            await state.update_data(api_key=None)
        await answer(message, "Введите SECRET KEY (или 'skip' для пропуска):")
        await message.delete()
        return
    
    if 'api_secret' not in data:
        if message.text.lower() != 'skip':
            await state.update_data(api_secret=message.text)
        else:
            await state.update_data(api_secret=None)
        await answer(message, "Введите TG TOKEN (или 'skip' для пропуска):")
        await message.delete()
        return

    if 'tg_token' not in data:
        if message.text.lower() != 'skip':
            await state.update_data(tg_token=message.text)
        else:
            await state.update_data(tg_token=None)
        await answer(message, "Введите TG ADMINS (через запятую, или 'skip' для пропуска):")
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
            await answer(message, "Ключи и токены успешно изменены, перезагружаю бота.")
            await restart_yes(message, state)
        else:
            await answer(message, "Никаких изменений не было внесено.")
            await settings(message, state)


@dp.callback_query(F.data == "restart")
async def restart(callback: CallbackQuery, state: FSMContext):
    # устанавливаем состояние для подтверждения перезагрузки
    await state.set_state(States.restart)
    # формируем клавиатуру с подтверждением удаления
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Да", callback_data=f"restart_yes")],
        [InlineKeyboardButton(text="Нет", callback_data="settings")]
    ])
    await callback.answer()
    # отправляем сообщение с подтверждением
    await callback.message.answer(f"Вы действительно хотите перезагрузить бота?", reply_markup=keyboard)

# функция перезагрузки бота
@dp.callback_query(States.restart, F.data == "restart_yes")
async def restart_yes(callback: CallbackQuery | Message, state: FSMContext):
    try:
        # отправляем сообщение о перезагрузке
        await answer(callback, 'Перезагрузка...')
    finally:
        # перезагружаем бота (просто завершаем процесс, systemd на сервере сам перезапустит сервис)
        os._exit(0)


async def send_startup_message():
    """
    Sends a startup message to all admins.
    """
    for admin_id in tg_admins:
        try:
            await bot.send_message(admin_id, "✅ Bot started successfully!")
        except Exception as e:
            print(f"Could not send startup message to admin {admin_id}: {e}")