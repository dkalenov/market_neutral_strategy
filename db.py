from sqlalchemy import Column, Integer, BigInteger, String, Float, Boolean, select, delete, update, JSON, text
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

# создаем таблицу для хранения истории коинтеграции пар
class PairHistory(Base):
    __tablename__ = 'pair_history'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol1 = Column(String)
    symbol2 = Column(String)
    event_type = Column(String) # 'FOUND', 'BROKEN'
    timestamp = Column(BigInteger)
    hedge_ratio = Column(Float)
    half_life = Column(Float)
    reason = Column(String, nullable=True) # Причина удаления


# создаем таблицу для хранения пар
class Pairs(Base):
    __tablename__ = 'pairs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol1 = Column(String)
    symbol2 = Column(String)
    hedge_ratio = Column(Float)
    half_life = Column(Float)
    position_status = Column(Integer, default=0)
    qty1 = Column(Float, default=0.0)
    qty2 = Column(Float, default=0.0)
    entry_price1 = Column(Float, default=0.0)
    entry_price2 = Column(Float, default=0.0)
    
    
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
    timeframe: str
    window_size: int
    # Risk & Strategy Params
    capital: float
    max_notional_pct: float
    leverage: int
    z_entry: float
    z_exit: float
    z_stop: float

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
    
    # Автоматическая миграция — добавляем недостающие колонки
    await run_migrations(engine)
    
    # возвращаем сессию
    return Session


async def run_migrations(engine):
    """Автоматически добавляет недостающие колонки в таблицы"""
    migrations = [
        # Таблица pairs — новые колонки
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS position_status INTEGER DEFAULT 0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS qty1 FLOAT DEFAULT 0.0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS qty2 FLOAT DEFAULT 0.0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS entry_price1 FLOAT DEFAULT 0.0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS entry_price2 FLOAT DEFAULT 0.0;",
    ]
    
    async with engine.begin() as conn:
        for sql in migrations:
            try:
                await conn.execute(text(sql))
            except Exception as e:
                # Игнорируем ошибки (колонка уже существует и т.д.)
                pass


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

# функция для обновления полей сделки по ID
async def update_trade_fields(trade_id, **kwargs):
    async with Session() as s:
        await s.execute(update(Trades).where(Trades.id == trade_id).values(**kwargs))
        await s.commit()

# функция для обновления пары
# Принимает dict с обязательным ключом 'id' и остальными полями для обновления
async def update_pair(data: dict):
    # Создаём копию, чтобы не мутировать исходный словарь
    data = data.copy()
    # создание сессии для работы с БД
    async with Session() as s:
        pair_id = data.pop('id')  # Извлекаем ID из копии словаря
        # обновляем пару
        await s.execute(update(Pairs).where(Pairs.id == pair_id).values(**data))
        # записываем изменения в БД
        await s.commit()

# функция для добавления истории пары
async def add_pair_history(history_item):
    async with Session() as s:
        s.add(history_item)
        await s.commit()

# функция для обновления пары
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

# функция для получения последней открытой сделки по ID пары
async def get_last_open_trade_for_pair(pair_id):
    async with Session() as s:
        trade = await s.execute(
            select(Trades)
            .where(Trades.pair_id == pair_id)
            .where(Trades.status == 'OPEN')
            .order_by(Trades.id.desc())
            .limit(1)
        )
        return trade.scalar_one_or_none()

# функция для добавления сделки
async def add_trade(trade):
    # создание сессии для работы с БД
    async with Session() as s:
        # добавляем новую сделку
        s.add(trade)
        # записываем изменения в БД
        await s.commit()
        # Обновляем объект, чтобы получить ID
        await s.refresh(trade)
        return trade.id

# функция для обновления сделки
async def update_trade(trade):
    # создание сессии для работы с БД
    async with Session() as s:
        # обновляем сделку
        s.add(trade)
        # записываем изменения в БД
        await s.commit()
