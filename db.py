from sqlalchemy import Column, Integer, BigInteger, String, Float, Boolean, select, delete, update, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

# Create base class for DB operations
Base = declarative_base()
Session: async_sessionmaker = None


# Table for main configuration keys
class Config(Base):
    __tablename__ = 'config'
    key = Column(String, primary_key=True)
    value = Column(String)

# Table for cointegration pairs history (logs)
class PairHistory(Base):
    __tablename__ = 'pair_history'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol1 = Column(String)
    symbol2 = Column(String)
    event_type = Column(String) # 'FOUND', 'BROKEN'
    timestamp = Column(BigInteger)
    hedge_ratio = Column(Float)
    half_life = Column(Float)
    reason = Column(String, nullable=True) # Removal reason

# Table for active trading pairs
class Pairs(Base):
    __tablename__ = 'pairs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol1 = Column(String)
    symbol2 = Column(String)
    hedge_ratio = Column(Float)
    half_life = Column(Float)
    position_status = Column(Integer, default=0) # 0: none, 1: long spread, -1: short spread
    qty1 = Column(Float, default=0.0)
    qty2 = Column(Float, default=0.0)
    entry_price1 = Column(Float, default=0.0)
    entry_price2 = Column(Float, default=0.0)
    
    
# Table for trade execution records
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
    

# Helper class for configuration data
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
    blacklist: str

    def __init__(self, data):
        for key in self.__class__.__annotations__:
            setattr(self, key, None)
        # Parse and convert data types
        for key, value in data.items():
            try:
                # Try integer
                value = int(value)
            except:
                try:
                    # Try float
                    value = float(value)
                except:
                    # Keep as string
                    pass
            setattr(self, key, value)


async def connect(host, port, user, password, db_name):
    global Session
    # #region agent log
    import os
    import json
    import time
    log_path = r"c:\Users\Dmitrii\Trading strategies\Market_neutral_strategy\.cursor\debug.log"
    def log_instrument(location, message, data=None):
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                entry = {
                    "id": f"log_{int(time.time()*1000)}_db",
                    "timestamp": int(time.time()*1000),
                    "location": location,
                    "message": message,
                    "data": data or {},
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "DB_CONN_1"
                }
                f.write(json.dumps(entry) + '\n')
        except: pass
    # #endregion

    log_instrument("db.py:connect", "Starting database connection", {"host": host, "port": port, "db": db_name})
    try:
        # Create async engine for PostgreSQL using asyncpg
        engine = create_async_engine(f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}")
        log_instrument("db.py:connect", "Engine created successfully")
        async with engine.begin() as conn:
            # Create tables
            await conn.run_sync(Base.metadata.create_all)
            log_instrument("db.py:connect", "Tables created successfully")
        # Create session maker
        Session = async_sessionmaker(engine, expire_on_commit=False)
        log_instrument("db.py:connect", "Session maker created")

        # Auto-migration for missing columns
        await run_migrations(engine)
        log_instrument("db.py:connect", "Migrations completed")

        return Session
    except Exception as e:
        log_instrument("db.py:connect", "Database connection failed", {"error": str(e), "error_type": type(e).__name__})
        raise


async def run_migrations(engine):
    """Automatically adds missing columns to tables"""
    migrations = [
        # Table pairs — new columns
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS position_status INTEGER DEFAULT 0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS qty1 FLOAT DEFAULT 0.0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS qty2 FLOAT DEFAULT 0.0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS entry_price1 FLOAT DEFAULT 0.0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS entry_price2 FLOAT DEFAULT 0.0;",
        # Table config — increase value length
        "ALTER TABLE config ALTER COLUMN value TYPE TEXT;",
    ]
    
    async with engine.begin() as conn:
        for sql in migrations:
            try:
                await conn.execute(text(sql))
            except Exception as e:
                # Column might already exist
                pass


async def load_config():
    # #region agent log
    import os
    import json
    import time
    log_path = r"c:\Users\Dmitrii\Trading strategies\Market_neutral_strategy\.cursor\debug.log"
    def log_instrument(location, message, data=None):
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                entry = {
                    "id": f"log_{int(time.time()*1000)}_db",
                    "timestamp": int(time.time()*1000),
                    "location": location,
                    "message": message,
                    "data": data or {},
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "DB_CONN_1"
                }
                f.write(json.dumps(entry) + '\n')
        except: pass
    # #endregion

    log_instrument("db.py:load_config", "Starting config load")
    if Session is None:
        raise RuntimeError("DB Session not initialized. Call db.connect() first.")
    try:
        # Default values
        DEFAULTS = {
            'timeframe': '1h',
            'window_size': '200',
            'capital': '1000',
            'leverage': '20',
            'max_notional_pct': '0.1',
            'z_entry': '2.0',
            'z_exit': '0.0',
            'z_stop': '4.0',
            'blacklist': 'BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,LTCUSDT,USDCUSDT,BTCDOMUSDT,DEFIUSDT'
        }

        # Ensure all expected config keys exist in DB and have defaults if empty
        for key in ConfigInfo.__annotations__.keys():
            try:
                async with Session() as s:
                    # Check if key exists
                    result = await s.execute(select(Config).where(Config.key == key))
                    existing = result.scalar_one_or_none()
                    
                    default_val = DEFAULTS.get(key, None)
                    
                    if existing is None:
                        # Insert new
                        s.add(Config(key=key, value=default_val))
                        await s.commit()
                        log_instrument("db.py:load_config", f"Config key created: {key} with default {default_val}")
                    elif (existing.value is None or existing.value == "") and default_val is not None:
                        # Update existing empty/null value with default
                        await s.execute(update(Config).where(Config.key == key).values(value=default_val))
                        await s.commit()
                        log_instrument("db.py:load_config", f"Config key updated with default: {key} -> {default_val}")
            except Exception as e:
                log_instrument("db.py:load_config", f"Failed to ensure/update config key {key}", {"error": str(e)})
                pass
        # Load all configuration from DB
        async with Session() as session:
            result = (await session.execute(select(Config))).scalars().all()
            data = {row.key: row.value for row in result}
            log_instrument("db.py:load_config", "Config loaded successfully", {"keys_count": len(data)})
            return ConfigInfo(data)
    except Exception as e:
        log_instrument("db.py:load_config", "Config load failed", {"error": str(e), "error_type": type(e).__name__})
        raise


async def config_update(**kwargs):
    # Update configuration parameters in DB
    async with Session() as s:
        for key, value in kwargs.items():
            # Convert to string for DB storage
            db_value = str(value) if value is not None else None
            await s.execute(update(Config).where(Config.key == key).values(value=db_value))
        await s.commit()


async def get_all_pairs():
    async with Session() as s:
        pairs = await s.execute(select(Pairs))
        return pairs.scalars().all()


async def add_pair(pair):
    async with Session() as s:
        s.add(pair)
        await s.commit()


async def update_trade_fields(trade_id, **kwargs):
    async with Session() as s:
        await s.execute(update(Trades).where(Trades.id == trade_id).values(**kwargs))
        await s.commit()


# Updates pair by ID using dict of fields
async def update_pair(data: dict):
    data = data.copy()
    async with Session() as s:
        pair_id = data.pop('id')
        result = await s.execute(update(Pairs).where(Pairs.id == pair_id).values(**data))
        if result.rowcount == 0:
            print(f"WARN: update_pair failed - Pair ID {pair_id} not found.")
        await s.commit()


async def add_pair_history(history_item):
    async with Session() as s:
        s.add(history_item)
        await s.commit()


async def delete_pair(pair_id):
    async with Session() as s:
        await s.execute(delete(Pairs).where(Pairs.id == pair_id))
        await s.commit()


async def get_open_trades():
    async with Session() as s:
        trades = await s.execute(select(Trades).where(Trades.status == 'OPEN'))
        return trades.scalars().all()


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


async def add_trade(trade):
    async with Session() as s:
        s.add(trade)
        await s.commit()
        await s.refresh(trade)
        return trade.id


async def update_trade(trade):
    async with Session() as s:
        s.add(trade)
        await s.commit()
