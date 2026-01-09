from sqlalchemy import Column, Integer, BigInteger, String, Float, Boolean, select, delete, update, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

# создаем базовый класс для работы с БД
Base = declarative_base()
Session: async_sessionmaker


# создаем таблицу с основными настройками
class Config(Base):
    __tablename__ = 'config'
    key = Column(String, primary_key=True)
    value = Column(String)


# создаем таблицу для хранения пар
class Pairs(Base):
    __tablename__ = 'pairs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol1 = Column(String)
    symbol2 = Column(String)
    hedge_ratio = Column(Float)
    half_life = Column(Float)
    
    
# создаем таблицу для хранения сделок
class Trades(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_id = Column(Integer)
    status = Column(String) # OPEN, CLOSED
    open_time = Column(BigInteger)
    close_time = Column(BigInteger)
    direction = Column(Integer) # 1 for long spread, -1 for short spread
    qty1 = Column(Float)
    qty2 = Column(Float)
    entry_price_1 = Column(Float)
    entry_price_2 = Column(Float)
    close_price_1 = Column(Float)
    close_price_2 = Column(Float)
    pnl = Column(Float, default=0)
    

# класс для хранения конфигации
class ConfigInfo:
    api_key: str
    api_secret: str
    tg_token: str
    tg_admins: str
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str

    def __init__(self, data):
        for key in self.__class__.__annotations__:
            setattr(self, key, None)
        # преобразование данных
        for key, value in data.items():
            try:
                # пытаемся преобразовать в int
                value = int(value)
            except:
                # если не получилось
                try:
                    # пытаемся преобразовать в float
                    value = float(value)
                except:
                    # если не получилось, то пропускаем и он будет строкой
                    pass
            # присваиваем значение значению класса
            setattr(self, key, value)


async def connect(host, port, user, password, db_name):
    # глобальная переменная сессии
    global Session
    # создаем асинхронный движок для работы с БД используя конфигурацию из config.py
    engine = create_async_engine(f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}")
    async with engine.begin() as conn:
        # создаем таблицы
        await conn.run_sync(Base.metadata.create_all)
    # создаем сессию, expire_on_commit=False - чтобы она не уничтожалась после коммита
    Session = async_sessionmaker(engine, expire_on_commit=False)
    # возвращаем сессию
    return Session


async def load_config():
    # создаем пустые поля в БД
    for key in ConfigInfo.__annotations__.keys():
        try:
            async with Session() as s:
                s.add(Config(key=key, value=None))
                await s.commit()
        except:
            pass
    # создание сессии для работы с БД
    async with Session() as session:
        # загрузка конфигурации
        result = (await session.execute(select(Config))).scalars().all()
        # преобразование данных
        data = {row.key: row.value for row in result}
        # возвращаем результат
        return ConfigInfo(data)

# функция для обновления конфигурации
async def config_update(**kwargs):
    # создание сессии для работы с БД
    async with Session() as s:
        # перебираем все параметры
        for key, value in kwargs.items():
            # записываем изменения в БД
            await s.execute(update(Config).where(Config.key == key).values(value=value))
        # записываем изменения в БД
        await s.commit()

# функция для получения всех пар
async def get_all_pairs():
    async with Session() as s:
        # получение всех пар
        pairs = await s.execute(select(Pairs))
        # возвращаем результат
        return pairs.scalars().all()

# функция для добавления пары
async def add_pair(pair):
    # создание сессии для работы с БД
    async with Session() as s:
        # добавляем новую пару
        s.add(pair)
        # записываем изменения в БД
        await s.commit()

# функция для удаления пары
async def delete_pair(pair_id):
    # создание сессии для работы с БД
    async with Session() as s:
        # удаляем пару
        await s.execute(delete(Pairs).where(Pairs.id == pair_id))
        # записываем изменения в БД
        await s.commit()

# функция для получения открытых трейдов
async def get_open_trades():
    # создание сессии для работы с БД
    async with Session() as s:
        # получаем открытые трейды
        trades = await s.execute(select(Trades).where(Trades.status == 'OPEN'))
        # возвращаем результат
        return trades.scalars().all()

# функция для добавления сделки
async def add_trade(trade):
    # создание сессии для работы с БД
    async with Session() as s:
        # добавляем новую сделку
        s.add(trade)
        # записываем изменения в БД
        await s.commit()

# функция для обновления сделки
async def update_trade(trade):
    # создание сессии для работы с БД
    async with Session() as s:
        # обновляем сделку
        s.add(trade)
        # записываем изменения в БД
        await s.commit()
