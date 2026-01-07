import asyncio
import configparser
import traceback
import binance
import pairs_trading
import db
import tg

# клиент для Binance
client: binance.Futures
# словарь для хранения всех торговых пар
all_symbols: dict[str, binance.SymbolFutures] = {}
# список открытых позиций
positions = {}
# список вебсокетов
websockets_list: list[binance.futures.WebsocketAsync] = []
# вебсокет юзердата
userdata_ws: binance.futures.WebsocketAsync
# менеджер пар
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

    # создаем клиента для Binance
    client = binance.Futures(api_key=conf.api_key,
                             secret_key=conf.api_secret,
                             asynced=True,
                             testnet=ini_config.getboolean('BOT', 'testnet'))
    # создаем менеджер пар
    loop = asyncio.get_running_loop()
    pairs_manager = pairs_trading.PairsManager(client, loop, all_symbols)
    # загружаем все торговые пары
    asyncio.create_task(load_symbols())
    # подключаемся к вебсокетам
    asyncio.create_task(connect_ws())
    
    # Запускаем телеграм бота
    await tg.run(session, client, pairs_manager)


# сервис для загрузки всех торговых пар каждый час
async def load_symbols():
    global all_symbols
    # вечный цикл
    while True:
        try:
            # загружаем все торговые пары
            print("Loading all market symbols...")
            all_symbols = await client.load_symbols()
            print(f"Loaded {len(all_symbols)} symbols.")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error loading symbols: {e}")
            traceback.print_exc()
        # ждем 1 час
        await asyncio.sleep(3600)


# подключение к вебсокетам
async def connect_ws():
    global websockets_list
    global userdata_ws

    print("Connecting to websockets...")

    # Список символов для отслеживания
    pairs = await db.get_all_pairs()
    target_symbols = set()
    for pair in pairs:
        target_symbols.add(pair.symbol1)
        target_symbols.add(pair.symbol2)

    streams = [f"{symbol.lower()}@kline_1m" for symbol in list(target_symbols)]

    # запускаем вебсокеты
    chunk_size = 100
    streams_list = [streams[i:i + chunk_size] for i in range(0, len(streams), chunk_size)]
    for stream_list in streams_list:
        websockets_list.append(await client.websocket(stream_list, on_message=ws_msg, on_error=ws_error))
    print("Connected to kline websocket.")

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
