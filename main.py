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
    
    # Подключаемся к БД
    ini_config = configparser.ConfigParser()
    ini_config.read('market_neutral/config.ini')
    
    session = await db.connect(
        host=ini_config['DB']['host'],
        port=ini_config['DB']['port'],
        user=ini_config['DB']['user'],
        password=ini_config['DB']['password'],
        db_name=ini_config['DB']['db_name']
    )

    # Загружаем конфиг из БД
    conf = await db.load_config()

    # Функция для отправки уведомлений в TG
    async def send_tg_notification(message):
        if tg.bot and conf.tg_admins:
             # Отправляем всем админам
             admins = [int(admin_id) for admin_id in conf.tg_admins.split(',') if admin_id]
             for admin_id in admins:
                 try:
                     await tg.bot.send_message(admin_id, message)
                 except Exception as e:
                     print(f"Error sending TG message to {admin_id}: {e}")

    # создаем клиента для Binance
    client = binance.Futures(api_key=conf.api_key,
                             secret_key=conf.api_secret,
                             asynced=True,
                             testnet=ini_config.getboolean('BOT', 'testnet'))
    # создаем менеджер пар
    loop = asyncio.get_running_loop()
    
    # Дефолтные значения если в конфиге пусто
    timeframe = conf.timeframe if conf.timeframe else '1h'
    
    if conf.window_size:
        window_size = conf.window_size
    else:
        # Автоматический подбор оптимального окна для разных ТФ
        if timeframe == '1m':
            window_size = 720  # 12 часов (полусуточный цикл, минимизация шума)
        elif timeframe == '5m':
            window_size = 576  # 2 суток (2 * 288 свечей)
        elif timeframe == '15m':
            window_size = 480  # 5 суток (рабочая торговая неделя)
        elif timeframe == '1h':
            window_size = 336  # 14 дней (2 недели, цикл "средней" краткосрочности)
        elif timeframe == '4h':
            window_size = 180  # 30 дней (месячный тренд)
        elif timeframe == '1d':
            window_size = 90   # 90 дней (квартальный тренд, избегаем старых структурных разрывов)
        else:
            window_size = 336  # Дефолт как для 1h
    
    # 1. Сначала загружаем символы (синхронно для старта)
    print("Initial loading of market symbols...")
    all_symbols = await client.load_symbols()
    print(f"Loaded {len(all_symbols)} symbols.")
    
    # 2. Создаём менеджер пар ПОСЛЕ загрузки символов (чтобы он получил заполненный dict)
    pairs_manager = pairs_trading.PairsManager(
        client, 
        loop, 
        all_symbols, 
        timeframe=timeframe, 
        min_data_points=window_size,
        notify_callback=send_tg_notification,
        config_info=conf
    )

    # 2. Запускаем фоновое обновление символов
    loop.create_task(load_symbols_loop())
    
    # 3. Подключаемся к вебсокетам (теперь all_symbols заполнен)
    loop.create_task(connect_ws(timeframe))
    
    # Запускаем телеграм бота
    await tg.run(session, client, pairs_manager)


# сервис для загрузки всех торговых пар каждый час
async def load_symbols_loop():
    global all_symbols
    # вечный цикл
    while True:
        try:
            # ждем 1 час
            await asyncio.sleep(3600)
            
            # загружаем все торговые пары
            print("Refreshing market symbols...")
            all_symbols = await client.load_symbols()
            print(f"Refreshed {len(all_symbols)} symbols.")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error loading symbols: {e}")
            traceback.print_exc()


# подключение к вебсокетам
async def connect_ws(timeframe='1h'):
    global websockets_list
    global userdata_ws

    print("Connecting to websockets...")

    # СОБИРАЕМ ВСЕ USDT ПАРЫ ДЛЯ СКАНЕРА
    # Бот должен слушать весь рынок, чтобы искать новые коинтеграции
    target_symbols = []
    
    # 1. Все USDT пары из маркета
    for s_name in all_symbols.keys():
        if s_name.endswith('USDT'):
            target_symbols.append(s_name)
    
    # 2. Добавляем те, что есть в БД (на случай если их нет в маркете/делистнули, но надо закрыть)
    db_pairs = await db.get_all_pairs()
    for p in db_pairs:
        if p.symbol1 not in target_symbols:
            target_symbols.append(p.symbol1)
        if p.symbol2 not in target_symbols:
            target_symbols.append(p.symbol2)
            
    print(f"Subscribing to {len(target_symbols)} symbols (Market + DB active)...")

    streams = [f"{symbol.lower()}@kline_{timeframe}" for symbol in target_symbols]

    # запускаем вебсокеты
    chunk_size = 100 # Ограничение Binance (обычно 1024, но 100 безопаснее для URL length)
    streams_list = [streams[i:i + chunk_size] for i in range(0, len(streams), chunk_size)]
    for stream_list in streams_list:
        try:
            ws = await client.websocket(stream_list, on_message=ws_msg, on_error=ws_error)
            websockets_list.append(ws)
            # Небольшая пауза между подключениями, чтобы не спамить
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error subscribing to chunk: {e}")

    print(f"Connected to kline websockets ({len(websockets_list)} connections).")

    # Подключение к вебсокету userdata
    try:
        userdata_ws = await client.websocket_userdata(on_message=ws_user_msg, on_error=ws_error)
        print("Connected to userdata websocket.")
    except Exception as e:
        print(f"Could not connect to userdata stream. API keys may be missing. Error: {e}")


# отключение от вебсокетов
async def disconnect_ws():
    global websockets_list
    global userdata_ws
    print("Disconnecting from websockets...")
    # перебираем вебсокеты
    for ws in websockets_list:
        try:
            # закрываем вебсокет
            await ws.close()
        except:
            pass
    try:
        # закрываем вебсокет userdata
        if 'userdata_ws' in globals() and userdata_ws:
            await userdata_ws.close()
    except:
        pass


# обработка ошибок вебсокета
async def ws_error(ws, error):
    print(f"WS ERROR: {error}")
    traceback.print_exc()


# обработка сообщений вебсокета
async def ws_msg(ws, msg):
    if 'data' not in msg:
        return
    
    kline = msg['data']['k']
    
    # Если свеча закрылась
    if kline['x']:
        # print(f"Kline closed for {kline['s']}: C={kline['c']}")
        # Передаем данные в менеджер пар
        await pairs_manager.add_kline(kline)


# обработчик сообщений вебсокета пользователя
async def ws_user_msg(ws, msg):
    print(f"Userdata message: {msg}")


if __name__ == '__main__':
    try:
        print("Starting market neutral bot...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user.")
