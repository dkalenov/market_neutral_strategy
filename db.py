import time
import os
import csv
import gzip
import shutil
from sqlalchemy import Column, Integer, BigInteger, String, Float, Boolean, select, delete, update, JSON, text, func
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
    # TG message tracking
    tg_message_id = Column(Integer, default=0)  # For reply threading
    # Trade lifecycle
    open_time = Column(BigInteger, default=0)   # Unix timestamp
    close_time = Column(BigInteger, default=0)  # Unix timestamp
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
    # Pair history retention monitoring (alerts only by default)
    pair_history_retention_days: int        # Retention horizon in days (default 365)
    pair_history_warn_days: int             # Warn when records are within this many days to retention (default 14)
    pair_history_cleanup_enabled: bool      # Future switch for cleanup job (default False)
    pair_history_check_interval_hours: int  # How often to check and alert (default 6)
    # Pair history backup (2-file rotation: current + prev)
    pair_history_backup_enabled: bool       # Enable periodic backup snapshots (default False)
    pair_history_backup_interval_hours: int # Backup check interval (default 24)
    pair_history_backup_dir: str            # Backup dir path (default 'market_neutral/backups')
    # Full DB backup (manual/triggered)
    db_backup_dir: str                      # Full DB backup dir path (default 'market_neutral/backups/db')
    db_backup_max_copies: int               # Rotating backup copies (default 2)
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
        # Table config — increase value length
        "ALTER TABLE config ALTER COLUMN value TYPE TEXT;",
        # Performance indexes (safe, idempotent)
        "CREATE INDEX IF NOT EXISTS idx_pairs_is_archived ON pairs (is_archived);",
        "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);",
        "CREATE INDEX IF NOT EXISTS idx_trades_pair_id ON trades (pair_id);",
        # Prevent duplicate ACTIVE pairs regardless of symbol order.
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_pairs_unique_active "
        "ON pairs (LEAST(symbol1, symbol2), GREATEST(symbol1, symbol2)) "
        "WHERE is_archived = FALSE;",
    ]
    
    async with engine.begin() as conn:
        for sql in migrations:
            try:
                await conn.execute(text(sql))
            except Exception as e:
                # Column might already exist
                pass


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
            # Hardware SL/TP Parameters (ATR-based) - values are decimal fractions
            'sl_atr_mult': '2.5',
            'sl_min_pct': '0.10',     # 10% min SL distance
            'sl_max_pct': '0.30',     # 30% max SL distance
            'tp_atr_mult': '4.0',
            'tp_min_pct': '0.15',     # 15% min TP distance
            'tp_max_pct': '0.50',     # 50% max TP distance
            'circuit_breaker_pct': '0.50',
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
            # Pair history retention monitoring (alerts only; no auto-delete in this build)
            'pair_history_retention_days': '365',
            'pair_history_warn_days': '14',
            'pair_history_cleanup_enabled': 'false',
            'pair_history_check_interval_hours': '6',
            'pair_history_backup_enabled': 'false',
            'pair_history_backup_interval_hours': '24',
            'pair_history_backup_dir': 'market_neutral/backups',
            'db_backup_dir': 'market_neutral/backups/db',
            'db_backup_max_copies': '2',
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
                pass
        # Load all configuration from DB
        async with Session() as session:
            # Remove deprecated key that is no longer used by strategy logic.
            await session.execute(delete(Config).where(Config.key == 'test_pairs'))
            await session.commit()
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


async def count_pair_history_older_than_days(days: int) -> int:
    """Count pair_history rows older than N days."""
    days = max(1, int(days))
    cutoff_ms = int((time.time() - days * 86400) * 1000)
    async with Session() as s:
        result = await s.execute(
            select(func.count()).select_from(PairHistory).where(PairHistory.timestamp < cutoff_ms)
        )
        return int(result.scalar() or 0)


async def count_pair_history_age_between_days(min_days: int, max_days: int) -> int:
    """
    Count pair_history rows with age in [min_days, max_days).
    Useful for "warning window" alerts before retention boundary.
    """
    min_days = max(0, int(min_days))
    max_days = max(min_days + 1, int(max_days))
    now = time.time()
    newer_than_ms = int((now - min_days * 86400) * 1000)
    older_than_ms = int((now - max_days * 86400) * 1000)
    async with Session() as s:
        result = await s.execute(
            select(func.count())
            .select_from(PairHistory)
            .where(PairHistory.timestamp < newer_than_ms)
            .where(PairHistory.timestamp >= older_than_ms)
        )
        return int(result.scalar() or 0)


async def fetch_pair_history_batch_before_ts(cutoff_ms: int, last_id: int = 0, limit: int = 5000):
    """
    Fetch pair_history rows in ascending id batches where timestamp < cutoff_ms.
    Returns list[PairHistory].
    """
    cutoff_ms = int(cutoff_ms)
    last_id = int(last_id or 0)
    limit = max(1, int(limit))
    async with Session() as s:
        result = await s.execute(
            select(PairHistory)
            .where(PairHistory.timestamp < cutoff_ms)
            .where(PairHistory.id > last_id)
            .order_by(PairHistory.id.asc())
            .limit(limit)
        )
        return result.scalars().all()


def _resolve_backup_dir(path_value: str) -> str:
    path_value = (path_value or '').strip() or 'market_neutral/backups/db'
    if os.path.isabs(path_value):
        return path_value
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    return os.path.join(root_dir, path_value)


async def _dump_table_csv_gz(table_name: str, model, target_dir: str, batch_size: int = 5000) -> int:
    columns = [c.name for c in model.__table__.columns]
    out_path = os.path.join(target_dir, f"{table_name}.csv.gz")
    written = 0

    with gzip.open(out_path, mode='wt', encoding='utf-8', newline='') as gz:
        writer = csv.writer(gz)
        writer.writerow(columns)

        has_int_id = hasattr(model, 'id')
        if has_int_id:
            last_id = 0
            while True:
                async with Session() as s:
                    result = await s.execute(
                        select(model)
                        .where(getattr(model, 'id') > last_id)
                        .order_by(getattr(model, 'id').asc())
                        .limit(batch_size)
                    )
                    rows = result.scalars().all()
                if not rows:
                    break
                for row in rows:
                    writer.writerow([getattr(row, col) for col in columns])
                    written += 1
                    last_id = getattr(row, 'id')
                if len(rows) < batch_size:
                    break
        else:
            async with Session() as s:
                result = await s.execute(select(model))
                rows = result.scalars().all()
            for row in rows:
                writer.writerow([getattr(row, col) for col in columns])
                written += 1

    return written


async def backup_all_tables_rotating(backup_dir: str, max_copies: int = 2) -> dict:
    """
    Full DB backup for bot tables with simple rotation.
    Creates:
      - db_backup_current/<table>.csv.gz
      - db_backup_prev/<table>.csv.gz   (if max_copies >= 2)
    """
    backup_root = _resolve_backup_dir(backup_dir)
    os.makedirs(backup_root, exist_ok=True)

    new_dir = os.path.join(backup_root, 'db_backup_new')
    current_dir = os.path.join(backup_root, 'db_backup_current')
    prev_dir = os.path.join(backup_root, 'db_backup_prev')

    if os.path.exists(new_dir):
        shutil.rmtree(new_dir, ignore_errors=True)
    os.makedirs(new_dir, exist_ok=True)

    table_map = {
        'config': Config,
        'pair_history': PairHistory,
        'pairs': Pairs,
        'trades': Trades,
    }

    counts = {}
    for table_name, model in table_map.items():
        counts[table_name] = await _dump_table_csv_gz(table_name, model, new_dir, batch_size=5000)

    if max_copies >= 2:
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir, ignore_errors=True)
        if os.path.exists(current_dir):
            shutil.move(current_dir, prev_dir)
    else:
        if os.path.exists(current_dir):
            shutil.rmtree(current_dir, ignore_errors=True)

    shutil.move(new_dir, current_dir)

    total_rows = sum(counts.values())
    return {
        'backup_root': backup_root,
        'current_dir': current_dir,
        'prev_dir': prev_dir if max_copies >= 2 and os.path.exists(prev_dir) else '',
        'counts': counts,
        'total_rows': total_rows,
    }
