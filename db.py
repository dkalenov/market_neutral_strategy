import time
from sqlalchemy import Column, Integer, BigInteger, String, Float, Boolean, select, delete, update, text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.exc import IntegrityError

# Create base class for DB operations
Base = declarative_base()
Session: async_sessionmaker = None


class DuplicateActivePairError(Exception):
    """Raised when inserting duplicate active pair blocked by unique index."""
    pass


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
    pair_id = Column(Integer, nullable=True)      # FK-like reference to pairs.id (not enforced)
    trade_id = Column(Integer, nullable=True)     # FK-like reference to trades.id (not enforced)
    z_score = Column(Float, default=0.0)          # Snapshot metric at event time
    beta_btc = Column(Float, default=0.0)         # Snapshot beta at event time
    pvalue = Column(Float, default=0.0)           # Snapshot p-value at event time
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
    # TG message tracking
    tg_message_id = Column(Integer, default=0)  # For reply threading
    # Trade lifecycle
    open_time = Column(BigInteger, default=0)   # Unix timestamp
    close_time = Column(BigInteger, default=0)  # Unix timestamp
    last_close_candle_ts = Column(BigInteger, default=0)  # Candle guard anchor (ms)
    close_pnl = Column(Float, default=0.0)      # Realized PnL
    close_reason = Column(String, default='')   # z_tp, z_sl, hw_sl, hw_tp, broken, manual, external
    fee1 = Column(Float, default=0.0)           # Fees leg 1
    fee2 = Column(Float, default=0.0)           # Fees leg 2
    pnl1 = Column(Float, default=0.0)           # Realized PnL leg 1
    pnl2 = Column(Float, default=0.0)           # Realized PnL leg 2
    # Market neutrality metrics (persisted for analysis & restart recovery)
    beta_btc = Column(Float, default=0.0)       # Last beta to BTC
    last_pvalue = Column(Float, default=0.0)    # Last cointegration p-value
    entry_z_score = Column(Float, default=0.0)  # Z-score at trade entry
    is_archived = Column(Boolean, default=False)  # Soft-delete marker for historical integrity
    
    
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
    # Extended trade metadata (for post-trade analysis)
    hedge_ratio = Column(Float, default=0.0)    # Hedge ratio at time of trade
    beta_btc = Column(Float, default=0.0)       # Beta to BTC at entry
    pvalue = Column(Float, default=0.0)         # P-value at entry
    entry_z = Column(Float, default=0.0)        # Z-score at entry
    close_z = Column(Float, default=0.0)        # Z-score at close
    fee1 = Column(Float, default=0.0)           # Fees on leg 1
    fee2 = Column(Float, default=0.0)           # Fees on leg 2
    close_reason = Column(String, default='')   # Why trade was closed


class TradeExecutions(Base):
    """
    Per-fill execution log for post-trade analytics/auditing.
    One row ~= one exchange trade fill (from get_account_trades).
    """
    __tablename__ = 'trade_executions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, nullable=True)
    pair_id = Column(Integer, nullable=True)
    symbol = Column(String, nullable=False)
    phase = Column(String, nullable=False)  # OPEN, CLOSE, EXTERNAL_CLOSE, DESYNC_CLOSE, etc.
    side = Column(String, nullable=True)    # BUY / SELL
    order_id = Column(BigInteger, nullable=True)
    exchange_trade_id = Column(BigInteger, nullable=True)
    price = Column(Float, default=0.0)
    qty = Column(Float, default=0.0)
    quote_qty = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    commission_asset = Column(String, default='')
    event_time = Column(BigInteger, default=0)  # exchange trade timestamp (ms)
    is_buyer = Column(Boolean, default=False)
    is_maker = Column(Boolean, default=False)
    created_at = Column(BigInteger, default=0)  # bot insert timestamp (ms)


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
    z_entry_max: float  # Upper bound for entry Z-score (default 3.0)
    z_exit: float
    z_stop: float
    blacklist: str
    # Hardware SL/TP Parameters (ATR-based)
    sl_atr_mult: float      # ATR multiplier for stop-loss (default 2.5)
    sl_min_pct: float       # Minimum SL distance in % (default 0.10)
    sl_max_pct: float       # Maximum SL distance in % (default 0.30)
    tp_atr_mult: float      # ATR multiplier for take-profit (default 4.0)
    tp_min_pct: float       # Minimum TP distance in % (default 0.15)
    tp_max_pct: float       # Maximum TP distance in % (default 0.50)
    circuit_breaker_pct: float  # Max loss as % of notional before force-close (default 0.20)
    p_value_threshold: float    # Max p-value for correlation validity (default 0.05)
    min_order_bump: float   # Max allowed order size increase ratio (default 1.5)
    # Position Management (Phase 2)
    max_active_pairs: int   # Maximum concurrent open pairs (default 5)
    test_mode: bool         # Force trades without signals on testnet (default False)
    priority_pairs_file: str # Path to JSON file with priority pairs (default 'market_neutral/best_pairs.json')
    # Symbol Filtering
    max_symbols: int        # Top N symbols by 24h volume (default 150)
    tg_channel: str         # TG channel ID for trade notifications (default '')
    # Half-Life Limits (in days)
    hl_min_days: float      # Minimum half-life in days (default 2.0)
    hl_max_days: float      # Maximum half-life in days (default 5.0)
    # Market Neutrality (Phase 3)
    beta_threshold: float   # Max |beta_btc| for pair acceptance (default 0.11)
    beta_alert_threshold: float  # Alert if |beta| > this for open positions (default 0.15)
    beta_critical: float    # Force-close if |beta| > this, regardless of PnL (default 1.0)
    signal_confirm_sec: int # Signal confirmation time in seconds (default 10)
    trade_mode: bool        # If True, allow opening new positions (default True)
    # Hedge Ratio Bounds (market neutrality)
    hedge_min: float        # Minimum |hedge_ratio| for pair acceptance (default 0.3)
    hedge_max: float        # Maximum |hedge_ratio| for pair acceptance (default 3.0)
    # Idle Pair Management
    max_idle_pairs: int     # Maximum idle pairs without positions (default 150)
    idle_timeout_hours: float  # Remove idle pairs older than X hours (default 48)
    # Realtime markPrice load control
    markprice_max_symbols: int              # Max symbols in markPrice realtime subscription (default 120)

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
    try:
        # Create async engine for PostgreSQL using asyncpg
        engine = create_async_engine(f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}")
        async with engine.begin() as conn:
            # Create tables
            await conn.run_sync(Base.metadata.create_all)
        # Create session maker
        Session = async_sessionmaker(engine, expire_on_commit=False)

        # Auto-migration for missing columns
        await run_migrations(engine)

        return Session
    except Exception as e:
        print(f"Database connection failed: {e}")
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
        # TG notification tracking columns
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS tg_message_id INTEGER DEFAULT 0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS open_time BIGINT DEFAULT 0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS close_time BIGINT DEFAULT 0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS last_close_candle_ts BIGINT DEFAULT 0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS close_pnl FLOAT DEFAULT 0.0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS close_reason VARCHAR(100) DEFAULT '';",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS fee1 FLOAT DEFAULT 0.0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS fee2 FLOAT DEFAULT 0.0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS pnl1 FLOAT DEFAULT 0.0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS pnl2 FLOAT DEFAULT 0.0;",
        # Pairs — market neutrality metrics (persisted for analysis & restart recovery)
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS beta_btc FLOAT DEFAULT 0.0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS last_pvalue FLOAT DEFAULT 0.0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS entry_z_score FLOAT DEFAULT 0.0;",
        "ALTER TABLE pairs ADD COLUMN IF NOT EXISTS is_archived BOOLEAN DEFAULT FALSE;",
        # Trades — extended metadata for post-trade analysis
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS hedge_ratio FLOAT DEFAULT 0.0;",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS beta_btc FLOAT DEFAULT 0.0;",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS pvalue FLOAT DEFAULT 0.0;",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS entry_z FLOAT DEFAULT 0.0;",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS close_z FLOAT DEFAULT 0.0;",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS fee1 FLOAT DEFAULT 0.0;",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS fee2 FLOAT DEFAULT 0.0;",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS close_reason VARCHAR(100) DEFAULT '';",
        # Pair history — structured analytics fields
        "ALTER TABLE pair_history ADD COLUMN IF NOT EXISTS pair_id INTEGER;",
        "ALTER TABLE pair_history ADD COLUMN IF NOT EXISTS trade_id INTEGER;",
        "ALTER TABLE pair_history ADD COLUMN IF NOT EXISTS z_score FLOAT DEFAULT 0.0;",
        "ALTER TABLE pair_history ADD COLUMN IF NOT EXISTS beta_btc FLOAT DEFAULT 0.0;",
        "ALTER TABLE pair_history ADD COLUMN IF NOT EXISTS pvalue FLOAT DEFAULT 0.0;",
        # Cleanup deprecated pair_history columns (kept in older builds)
        "ALTER TABLE pair_history DROP COLUMN IF EXISTS metric;",
        "ALTER TABLE pair_history DROP COLUMN IF EXISTS metric_value;",
        "ALTER TABLE pair_history DROP COLUMN IF EXISTS metric_threshold;",
        "ALTER TABLE pair_history DROP COLUMN IF EXISTS details;",
        # Fill-level executions table for order analytics
        """
        CREATE TABLE IF NOT EXISTS trade_executions (
            id SERIAL PRIMARY KEY,
            trade_id INTEGER NULL,
            pair_id INTEGER NULL,
            symbol VARCHAR(32) NOT NULL,
            phase VARCHAR(32) NOT NULL,
            side VARCHAR(8) NULL,
            order_id BIGINT NULL,
            exchange_trade_id BIGINT NULL,
            price FLOAT DEFAULT 0.0,
            qty FLOAT DEFAULT 0.0,
            quote_qty FLOAT DEFAULT 0.0,
            realized_pnl FLOAT DEFAULT 0.0,
            commission FLOAT DEFAULT 0.0,
            commission_asset VARCHAR(16) DEFAULT '',
            event_time BIGINT DEFAULT 0,
            is_buyer BOOLEAN DEFAULT FALSE,
            is_maker BOOLEAN DEFAULT FALSE,
            created_at BIGINT DEFAULT 0
        );
        """,
        # Table config — increase value length
        "ALTER TABLE config ALTER COLUMN value TYPE TEXT;",
        # Performance indexes (safe, idempotent)
        "CREATE INDEX IF NOT EXISTS idx_pairs_is_archived ON pairs (is_archived);",
        "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);",
        "CREATE INDEX IF NOT EXISTS idx_trades_pair_id ON trades (pair_id);",
        "CREATE INDEX IF NOT EXISTS idx_trade_exec_trade_id ON trade_executions (trade_id);",
        "CREATE INDEX IF NOT EXISTS idx_trade_exec_pair_id ON trade_executions (pair_id);",
        "CREATE INDEX IF NOT EXISTS idx_trade_exec_symbol_time ON trade_executions (symbol, event_time);",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_exec_unique_fill "
        "ON trade_executions (COALESCE(trade_id, -1), symbol, phase, exchange_trade_id) "
        "WHERE exchange_trade_id IS NOT NULL;",
        "CREATE INDEX IF NOT EXISTS idx_pair_history_ts ON pair_history (timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_pair_history_event ON pair_history (event_type);",
        "CREATE INDEX IF NOT EXISTS idx_pair_history_symbols ON pair_history (symbol1, symbol2);",
        # Prevent duplicate ACTIVE pairs regardless of symbol order.
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_pairs_unique_active "
        "ON pairs (LEAST(symbol1, symbol2), GREATEST(symbol1, symbol2)) "
        "WHERE is_archived = FALSE;",
        # Cleanup deprecated config keys
        "DELETE FROM config WHERE key = 'test_pairs';",
    ]
    
    async with engine.begin() as conn:
        for sql in migrations:
            try:
                await conn.execute(text(sql))
            except Exception as e:
                sql_head = sql.strip().split('\n', 1)[0]
                if len(sql_head) > 120:
                    sql_head = sql_head[:120] + "..."
                print(f"⚠️ Migration skipped/failed: {sql_head} | {e}")


async def load_config():
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
            'z_entry': '1.9',
            'z_entry_max': '2.5',  # Upper bound for entry window
            'z_exit': '0.0',
            'z_stop': '4.0',
            'blacklist': 'BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,LTCUSDT,USDCUSDT,BTCDOMUSDT,DEFIUSDT',
            # Hardware SL/TP defaults
            'sl_atr_mult': '2.5',
            'sl_min_pct': '0.10',
            'sl_max_pct': '0.30',
            'tp_atr_mult': '4.0',
            'tp_min_pct': '0.15',
            'tp_max_pct': '0.50',
            'circuit_breaker_pct': '0.50',
            'p_value_threshold': '0.05',
            'min_order_bump': '1.5',
            # Position Management (Phase 2)
            'max_active_pairs': '5',
            'test_mode': 'false',
            'priority_pairs_file': 'best_pairs.json',
            # Symbol Filtering
            'max_symbols': '150',
            'tg_channel': '',
            # Half-Life Limits (in days) - optimized for 1h/4h TFs
            'hl_min_days': '0.25',   # Min 6 hours (1h=6 candles, 4h=1.5 candles)
            'hl_max_days': '2.0',    # Max 2 days (1h=48 candles, 4h=12 candles)
            'beta_threshold': '0.11',        # Max |beta_btc| for pair acceptance
            'beta_alert_threshold': '0.15',  # Alert if |beta| > this for open positions
            'beta_critical': '1.0',          # Force-close if |beta| > this regardless of PnL
            'signal_confirm_sec': '10',      # Signal confirmation time in seconds
            'trade_mode': 'true',            # Allow opening new positions
            # Hedge Ratio Bounds (market neutrality)
            'hedge_min': '0.3',              # Min |hedge| — below this positions are too unbalanced
            'hedge_max': '3.0',              # Max |hedge| — above this positions are too unbalanced
            # Idle Pair Management
            'max_idle_pairs': '150',         # Maximum idle pairs without positions
            'idle_timeout_hours': '48',      # Remove idle pairs older than X hours
            'markprice_max_symbols': '120',
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
                    elif (existing.value is None or existing.value == "") and default_val is not None:
                        # Update existing empty/null value with default
                        await s.execute(update(Config).where(Config.key == key).values(value=default_val))
                        await s.commit()
            except Exception as e:
                print(f"⚠️ Config default init failed for key '{key}': {e}")
        # Load all configuration from DB
        async with Session() as session:
            result = (await session.execute(select(Config))).scalars().all()
            data = {row.key: row.value for row in result}
            return ConfigInfo(data)
    except Exception as e:
        print(f"Config load failed: {e}")
        raise


async def config_update(**kwargs):
    # Update configuration parameters in DB
    async with Session() as s:
        for key, value in kwargs.items():
            # Convert to string for DB storage
            db_value = str(value) if value is not None else None
            await s.execute(update(Config).where(Config.key == key).values(value=db_value))
        await s.commit()


async def get_all_pairs(include_archived: bool = False):
    async with Session() as s:
        stmt = select(Pairs)
        if not include_archived:
            stmt = stmt.where(Pairs.is_archived.is_(False))
        pairs = await s.execute(stmt)
        return pairs.scalars().all()


async def get_active_pair_keys() -> set[tuple[str, str]]:
    """
    Return canonical symbol keys for all active (non-archived) pairs.
    Key format: (min(symbol1, symbol2), max(symbol1, symbol2)).
    """
    async with Session() as s:
        rows = await s.execute(
            select(Pairs.symbol1, Pairs.symbol2).where(Pairs.is_archived.is_(False))
        )
        result = set()
        for sym1, sym2 in rows.all():
            if not sym1 or not sym2:
                continue
            a, b = sorted((str(sym1), str(sym2)))
            result.add((a, b))
        return result


async def active_pair_exists(symbol1: str, symbol2: str) -> bool:
    """Check if an active (non-archived) pair exists regardless of symbol order."""
    sym_lo, sym_hi = sorted((symbol1, symbol2))
    async with Session() as s:
        stmt = (
            select(Pairs.id)
            .where(Pairs.is_archived.is_(False))
            .where(func.least(Pairs.symbol1, Pairs.symbol2) == sym_lo)
            .where(func.greatest(Pairs.symbol1, Pairs.symbol2) == sym_hi)
            .limit(1)
        )
        result = await s.execute(stmt)
        return result.scalar_one_or_none() is not None


async def add_pair(pair):
    async with Session() as s:
        s.add(pair)
        try:
            await s.commit()
        except IntegrityError as e:
            await s.rollback()
            if 'idx_pairs_unique_active' in str(e):
                raise DuplicateActivePairError from e
            raise


async def update_trade_fields(trade_id, **kwargs):
    async with Session() as s:
        await s.execute(update(Trades).where(Trades.id == trade_id).values(**kwargs))
        await s.commit()


# Updates pair by ID using dict of fields
async def update_pair(data: dict):
    data = data.copy()
    pos = data.get('position_status', None)
    if pos == 0 and 'last_close_candle_ts' not in data:
        close_time = data.get('close_time', int(time.time()))
        try:
            close_time = int(close_time)
        except Exception:
            close_time = int(time.time())
        # Normalize to milliseconds for candle-boundary math in strategy layer.
        data['last_close_candle_ts'] = close_time if close_time > 1_000_000_000_000 else close_time * 1000
    elif pos is not None and pos != 0 and 'last_close_candle_ts' not in data:
        # Opening/active state clears close-candle guard marker.
        data['last_close_candle_ts'] = 0

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


async def log_pair_history_event(
    *,
    symbol1: str,
    symbol2: str,
    event_type: str,
    timestamp_ms: int | None = None,
    hedge_ratio: float | None = None,
    half_life: float | None = None,
    reason: str | None = None,
    pair_id: int | None = None,
    trade_id: int | None = None,
    z_score: float | None = None,
    beta_btc: float | None = None,
    pvalue: float | None = None,
):
    """
    Structured PairHistory event writer.
    Keeps reason as human-readable text but stores analyzable numeric/context fields.
    """
    item = PairHistory(
        symbol1=symbol1,
        symbol2=symbol2,
        event_type=event_type,
        timestamp=timestamp_ms if timestamp_ms is not None else int(time.time() * 1000),
        hedge_ratio=float(hedge_ratio) if hedge_ratio is not None else 0.0,
        half_life=float(half_life) if half_life is not None else 0.0,
        reason=reason,
        pair_id=pair_id,
        trade_id=trade_id,
        z_score=float(z_score) if z_score is not None else 0.0,
        beta_btc=float(beta_btc) if beta_btc is not None else 0.0,
        pvalue=float(pvalue) if pvalue is not None else 0.0,
    )
    await add_pair_history(item)


async def delete_pair(pair_id):
    async with Session() as s:
        await s.execute(delete(Pairs).where(Pairs.id == pair_id))
        await s.commit()


async def archive_pair(pair_id: int, reason: str = ''):
    """Soft-delete pair row to preserve trade history references."""
    values = {
        'is_archived': True,
        'position_status': 0,
        'qty1': 0.0,
        'qty2': 0.0,
        'entry_price1': 0.0,
        'entry_price2': 0.0,
        'close_time': int(time.time()),
    }
    if reason:
        values['close_reason'] = reason
    async with Session() as s:
        await s.execute(update(Pairs).where(Pairs.id == pair_id).values(**values))
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


async def close_trade_record(
    trade_id: int,
    *,
    status: str = 'CLOSED',
    close_reason: str = 'unknown',
    close_time_ms: int | None = None,
    pnl: float | None = None,
    fee1: float | None = None,
    fee2: float | None = None,
    close_z: float | None = None,
    close_price_1: float | None = None,
    close_price_2: float | None = None,
):
    """Unified safe closer for Trades rows."""
    data = {
        'status': status,
        'close_time': close_time_ms if close_time_ms is not None else int(time.time() * 1000),
        'close_reason': close_reason,
    }
    if pnl is not None:
        data['pnl'] = pnl
    if fee1 is not None:
        data['fee1'] = fee1
    if fee2 is not None:
        data['fee2'] = fee2
    if close_z is not None:
        data['close_z'] = close_z
    if close_price_1 is not None:
        data['close_price_1'] = close_price_1
    if close_price_2 is not None:
        data['close_price_2'] = close_price_2
    await update_trade_fields(trade_id, **data)


async def add_trade_executions(rows: list[dict]):
    """
    Bulk insert execution/fill rows. Best-effort (single transaction).
    Expected row keys align with TradeExecutions model fields.
    """
    if not rows:
        return
    now_ms = int(time.time() * 1000)
    objects = []
    for row in rows:
        data = dict(row)
        if 'created_at' not in data or data['created_at'] is None:
            data['created_at'] = now_ms
        objects.append(TradeExecutions(**data))
    async with Session() as s:
        s.add_all(objects)
        try:
            await s.commit()
        except IntegrityError:
            await s.rollback()
            # Fallback: insert one by one, skipping duplicates.
            for obj in objects:
                try:
                    s.add(obj)
                    await s.commit()
                except IntegrityError:
                    await s.rollback()
