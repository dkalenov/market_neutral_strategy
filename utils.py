import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import math

CAPITAL = 1_000_000.0
MAX_NOTIONAL_PER_PAIR = 0.1
VOL_LOOKBACK = 60

# Timeframe to candles per day mapping
CANDLES_PER_DAY = {
    '1m':  1440,
    '5m':  288,
    '15m': 96,
    '30m': 48,
    '1h':  24,
    '4h':  6,
    '1d':  1,
}

def get_half_life_limits(timeframe: str, hl_min_days: float = 2.0, hl_max_days: float = 5.0) -> tuple:
    """
    Get (min_hl, max_hl) in candles for given timeframe.
    Converts configurable days to candle count.
    """
    candles = CANDLES_PER_DAY.get(timeframe, 24)  # Default to 1h
    min_hl = int(hl_min_days * candles)
    max_hl = int(hl_max_days * candles)
    return (max(1, min_hl), max(min_hl + 1, max_hl))  # Ensure valid range

def safe_get_slope(params):
    """Return slope param (index 1) robustly for numpy.ndarray or pandas.Series."""
    try:
        if isinstance(params, np.ndarray):
            return float(params[1])
        else:
            return float(params.iloc[1])
    except Exception:
        try:
            return float(params[1])
        except Exception:
            return np.nan

def calculate_half_life(spread):
    """
    spread: numpy array or pandas Series (log-spread).
    Returns half-life in bars or np.nan if not mean-reverting or fail.
    """
    try:
        s = pd.Series(spread).dropna()
        if len(s) < 10:
            return np.nan
        spread_lag = s.shift(1).iloc[1:]
        delta = (s - s.shift(1)).iloc[1:]
        X = sm.add_constant(spread_lag)
        res = sm.OLS(delta, X).fit()
        b = safe_get_slope(res.params)
        if np.isnan(b):
            return np.nan
        phi = 1.0 + b
        if phi <= 0 or phi >= 1:
            return np.nan
        hl = -np.log(2) / np.log(phi)
        return float(round(hl, 2))
    except Exception:
        return np.nan

from numpy.linalg import LinAlgError

def calculate_cointegration(log1, log2, p_value_threshold: float = 0.05, strict_hl: bool = True):
    """
    log1, log2: numpy arrays of log(prices) (length WINDOW)
    p_value_threshold: Maximum p-value for valid cointegration (default 0.05)
    strict_hl: If True (discovery mode), reject pairs with bad half-life (NaN/<=0/>200).
               If False (monitoring mode), only check p-value + t-stat, return hl as-is.
    Returns:
        flag (0/1), hedge_ratio (beta), half_life, p_value
    """
    safe_p_value = np.nan
    try:
        coint_t, p_value, crit_vals = coint(log1, log2)
        safe_p_value = float(p_value)
        X = sm.add_constant(log2)
        model = sm.OLS(log1, X).fit()
        hedge = safe_get_slope(model.params)
        if np.isnan(hedge):
            return 0, np.nan, np.nan, safe_p_value

        spread = log1 - hedge * log2
        hl = calculate_half_life(spread)

        try:
            crit5 = crit_vals[1]
            t_check = coint_t < crit5
        except Exception:
            t_check = True

        if strict_hl and (np.isnan(hl) or hl <= 0 or hl > 200):
            return 0, hedge, np.nan, safe_p_value

        flag = 1 if (safe_p_value < p_value_threshold and t_check) else 0
        return flag, hedge, hl, safe_p_value
    except (Exception, LinAlgError):
        return 0, np.nan, np.nan, safe_p_value

def calculate_z_last(spread):
    s = pd.Series(spread)
    m = s.mean()
    sd = s.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float((s.iloc[-1] - m) / sd)

def calculate_pair_beta(pair_r, market_r):
    if pair_r is None or market_r is None:
        return np.nan
    pair_r = np.array(pair_r, dtype=float)
    market_r = np.array(market_r, dtype=float)
    if len(pair_r) != len(market_r) or len(pair_r) < 5:
        return np.nan
    cov = np.cov(pair_r, market_r)[0, 1]
    var_m = np.var(market_r)
    if var_m == 0:
        return np.nan
    return float(cov / var_m)

def batch_process_pairs(pairs_chunk, data_dict, min_data_points, timeframe='1h', hl_min_days=2.0, hl_max_days=5.0):
    """
    Worker function for parallel processing.
    pairs_chunk: list of tuples (s1, s2)
    data_dict: dict {symbol: np.array(log_prices)}
    min_data_points: int
    timeframe: str (for half-life limits)
    hl_min_days: float (configurable min half-life in days)
    hl_max_days: float (configurable max half-life in days)
    """
    min_hl, max_hl = get_half_life_limits(timeframe, hl_min_days, hl_max_days)
    results = []
    for s1, s2 in pairs_chunk:
        try:
            # Data is already pre-processed (log prices) in data_dict
            log1 = data_dict.get(s1)
            log2 = data_dict.get(s2)
            
            if log1 is None or log2 is None:
                continue

            # Ensure alignment (simple truncation to min length if mismatch, though manager should handle this)
            min_len = min(len(log1), len(log2))
            if min_len < min_data_points:
                continue
                
            l1 = log1[-min_len:]
            l2 = log2[-min_len:]
            
            flag, hedge, hl, pval = calculate_cointegration(l1, l2)
            
            # Filter by timeframe-specific half-life limits
            if flag == 1 and min_hl <= hl <= max_hl:
                results.append((s1, s2, hedge, hl, pval))
        except Exception:
            continue
    return results

def vol_parity_notional(log1, log2, hedge, capital=CAPITAL, max_notional_per_pair=MAX_NOTIONAL_PER_PAIR, lookback=VOL_LOOKBACK):
    """
    Hedge-ratio balanced position sizing for pairs trading.
    
    Spread: S = log(P1) - hedge * log(P2)
    PnL = D1 * Δlog(P1) - D2 * Δlog(P2)
        = (D1*hedge - D2) * Δlog(P2)  +  D1 * Δε
    
    For market neutrality (PnL independent of common moves):
        D2 = hedge * D1
    
    Solving  D1 + D2 = cap  and  D2 = |hedge| * D1:
        D1 = cap / (1 + |hedge|)
        D2 = cap * |hedge| / (1 + |hedge|)
    
    Example: hedge=0.94, cap=$23 → D1=$11.86, D2=$11.14 (≈50/50)
    Example: hedge=0.50, cap=$23 → D1=$15.33, D2=$7.67 (intentional)
    """
    cap_pair_usd = capital * max_notional_per_pair
    abs_h = abs(hedge) if abs(hedge) > 0.01 else 1.0
    dollar1 = cap_pair_usd / (1.0 + abs_h)
    dollar2 = cap_pair_usd * abs_h / (1.0 + abs_h)
    return float(dollar1), float(dollar2)

def calculate_qty(dollar1, dollar2, price1, price2, capital=CAPITAL, max_notional_per_pair=MAX_NOTIONAL_PER_PAIR):
    max_notional = capital * max_notional_per_pair
    tot = abs(dollar1) + abs(dollar2)
    if tot > max_notional and tot > 0:
        scale = max_notional / tot
        dollar1 *= scale
        dollar2 *= scale
    qty1 = dollar1 / price1 if price1 > 0 else 0.0
    qty2 = dollar2 / price2 if price2 > 0 else 0.0
    return float(qty1), float(qty2)

# Helper to calculate precision from step size
def get_precision(step_size):
    """Converts a step_size like 0.001 to precision like 3."""
    if step_size == 0 or step_size >= 1:
        return 0
    return int(round(-math.log10(step_size), 0))

# функция для округления вверх до шага (step_size)
def round_up(num, step_size=1.0):
    if step_size < 1e-9 or step_size >= 1:
        return math.ceil(num)
    precision = get_precision(step_size)
    multiplier = 10 ** precision
    return math.ceil(num * multiplier) / multiplier

# функция для округления вниз до шага (step_size)
def round_down(num, step_size=1.0):
    if step_size < 1e-9 or step_size >= 1:
        return math.floor(num)
    precision = get_precision(step_size)
    multiplier = 10 ** precision
    return math.floor(num * multiplier) / multiplier


def calculate_atr(high: list, low: list, close: list, period: int = 14) -> float:
    """
    Calculate Average True Range for volatility-based stop placement.
    
    Args:
        high: List of high prices
        low: List of low prices  
        close: List of close prices
        period: ATR period (default 14)
    
    Returns:
        ATR value as float
    """
    if len(close) < 2:
        return 0.0
    
    tr_list = []
    for i in range(1, len(close)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        tr_list.append(tr)
    
    if len(tr_list) < period:
        return sum(tr_list) / len(tr_list) if tr_list else 0.0
    
    return sum(tr_list[-period:]) / period


def calculate_hardware_stops(entry_price: float, side: str, atr: float, config) -> tuple:
    """
    Calculate hardware SL and TP prices based on ATR and config parameters.
    
    Args:
        entry_price: Entry price of the position
        side: 'LONG' or 'SHORT'
        atr: Average True Range value
        config: Config object with sl_*/tp_* parameters
    
    Returns:
        (stop_loss_price, take_profit_price, sl_pct, tp_pct)
    """
    # Get config values with defaults
    sl_atr_mult = getattr(config, 'sl_atr_mult', 2.5) or 2.5
    sl_min_pct = getattr(config, 'sl_min_pct', 0.10) or 0.10
    sl_max_pct = getattr(config, 'sl_max_pct', 0.30) or 0.30
    tp_atr_mult = getattr(config, 'tp_atr_mult', 4.0) or 4.0
    tp_min_pct = getattr(config, 'tp_min_pct', 0.15) or 0.15
    tp_max_pct = getattr(config, 'tp_max_pct', 0.50) or 0.50
    
    # Calculate ATR-based percentages
    if entry_price > 0 and atr > 0:
        atr_sl = (atr / entry_price) * sl_atr_mult
        atr_tp = (atr / entry_price) * tp_atr_mult
    else:
        atr_sl = sl_min_pct
        atr_tp = tp_min_pct
    
    # Apply min/max bounds
    sl_pct = max(min(atr_sl, sl_max_pct), sl_min_pct)
    tp_pct = max(min(atr_tp, tp_max_pct), tp_min_pct)
    
    # Calculate prices based on side
    if side == 'LONG':
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)
    else:  # SHORT
        sl_price = entry_price * (1 + sl_pct)
        tp_price = entry_price * (1 - tp_pct)
    
    return sl_price, tp_price, sl_pct, tp_pct


def should_skip_trade(min_notional: float, calculated_notional: float, min_order_bump: float = 1.5) -> bool:
    """
    Check if trade should be skipped due to minimum order requirements.
    
    Args:
        min_notional: Minimum notional required by exchange
        calculated_notional: Notional calculated by strategy
        min_order_bump: Maximum allowed increase ratio (default 1.5x)
    
    Returns:
        True if trade should be skipped, False otherwise
    """
    if calculated_notional >= min_notional:
        return False  # No adjustment needed
    
    if calculated_notional <= 0:
        return True  # Invalid notional
    
    bump_ratio = min_notional / calculated_notional
    if bump_ratio > min_order_bump:
        print(f"SKIP: Order bump {bump_ratio:.2f}x exceeds threshold {min_order_bump}x")
        return True
    
    return False  # Small adjustment is acceptable
