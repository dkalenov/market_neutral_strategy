import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import math
import warnings

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

def calculate_half_life(spread) -> float:
    """
    spread: numpy array of log-spread.
    Returns half-life in bars or np.nan if not mean-reverting.
    Pure numpy OLS — no pandas, no statsmodels (3× faster than the original).
    """
    try:
        s = np.asarray(spread, dtype=float)
        # drop NaNs inline
        mask = np.isfinite(s)
        s = s[mask]
        n = len(s)
        if n < 10:
            return np.nan
        lag = s[:-1]          # s_{t-1}
        delta = s[1:] - lag   # Δs_t = s_t - s_{t-1}
        # OLS: delta = a + b*lag  →  b = Cov(lag,delta)/Var(lag)
        lag_m = lag.mean()
        delta_m = delta.mean()
        var_lag = ((lag - lag_m) ** 2).mean()
        if var_lag < 1e-20:
            return np.nan
        b = ((lag - lag_m) * (delta - delta_m)).mean() / var_lag
        if np.isnan(b):
            return np.nan
        phi = 1.0 + b
        if phi <= 0.0 or phi >= 1.0:
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
        arr1 = np.asarray(log1, dtype=float)
        arr2 = np.asarray(log2, dtype=float)
        if len(arr1) < 10 or len(arr2) < 10:
            return 0, np.nan, np.nan, safe_p_value
        if not np.all(np.isfinite(arr1)) or not np.all(np.isfinite(arr2)):
            return 0, np.nan, np.nan, safe_p_value
        # Flat inputs cause unstable regression stats and noisy warnings.
        if arr1.std() < 1e-12 or arr2.std() < 1e-12:
            return 0, np.nan, np.nan, safe_p_value

        # ── ADF cointegration test (statsmodels, most expensive call) ──────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coint_t, p_value, crit_vals = coint(arr1, arr2)

        safe_p_value = float(p_value)

        # ⚡ Early exit (discovery/scan mode only): skip OLS + half-life when the pair
        # clearly fails cointegration. ~80-90% of pairs exit here, saving 2 expensive calls.
        # Not applied in monitoring mode (strict_hl=False) where we always need fresh hedge.
        if strict_hl and safe_p_value > p_value_threshold * 2.0:
            return 0, np.nan, np.nan, safe_p_value

        # ── OLS hedge ratio (only for near-cointegrated pairs) ─────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = sm.add_constant(arr2)
            model = sm.OLS(arr1, X).fit()

        hedge = safe_get_slope(model.params)
        if np.isnan(hedge):
            return 0, np.nan, np.nan, safe_p_value

        spread = arr1 - hedge * arr2
        hl = calculate_half_life(spread)   # pure-numpy, no statsmodels

        try:
            crit5 = crit_vals[1]
            t_check = bool(coint_t < crit5)
        except Exception:
            t_check = True

        if strict_hl and (np.isnan(hl) or hl <= 0 or hl > 200):
            return 0, hedge, np.nan, safe_p_value

        flag = 1 if (safe_p_value < p_value_threshold and t_check) else 0
        return flag, hedge, hl, safe_p_value
    except (Exception, LinAlgError):
        return 0, np.nan, np.nan, safe_p_value

def calculate_z_last(spread):
    """Z-score of the last element. Pure numpy — no pandas overhead."""
    s = np.asarray(spread, dtype=float)
    if len(s) < 2:
        return np.nan
    m = s.mean()
    sd = s.std()
    if sd < 1e-20 or not np.isfinite(sd):
        return np.nan
    return float((s[-1] - m) / sd)

def calculate_pair_beta(pair_r, market_r):
    if pair_r is None or market_r is None:
        return np.nan
    pair_r = np.array(pair_r, dtype=float)
    market_r = np.array(market_r, dtype=float)
    if len(pair_r) != len(market_r) or len(pair_r) < 5:
        return np.nan
    # CRITICAL: Use consistent ddof=1 for both cov and var.
    # np.cov defaults to ddof=1 (sample), but np.var defaults to ddof=0 (population).
    # Mismatch causes beta spikes when BTC variance is low.
    cov = np.cov(pair_r, market_r)[0, 1]  # ddof=1 (sample)
    var_m = np.var(market_r, ddof=1)       # ddof=1 (sample) — MUST match cov
    if var_m < 1e-20:  # Avoid division by near-zero
        return np.nan
    return float(cov / var_m)

def batch_process_pairs(
    pairs_chunk,
    data_dict,
    min_data_points,
    timeframe='1h',
    hl_min_days=2.0,
    hl_max_days=5.0,
    hedge_min=0.3,
    hedge_max=3.0,
    p_value_threshold=0.05,
):
    """
    Worker function for parallel processing.
    pairs_chunk: list of tuples (s1, s2)
    data_dict: dict {symbol: np.array(log_prices)}
    min_data_points: int
    timeframe: str (for half-life limits)
    hl_min_days: float (configurable min half-life in days)
    hl_max_days: float (configurable max half-life in days)
    hedge_min: float (minimum |hedge_ratio| for market neutrality, default 0.3)
    hedge_max: float (maximum |hedge_ratio| for market neutrality, default 3.0)
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
            
            flag, hedge, hl, pval = calculate_cointegration(l1, l2, p_value_threshold=p_value_threshold)
            
            # Filter by timeframe-specific half-life limits
            if flag == 1 and min_hl <= hl <= max_hl:
                # Filter by hedge ratio bounds (market neutrality)
                # |hedge| too small (e.g. 0.06) → positions wildly unbalanced ($5 vs $92)
                # |hedge| too large (e.g. 5.0) → same problem in reverse
                abs_hedge = abs(hedge)
                if abs_hedge < hedge_min or abs_hedge > hedge_max:
                    continue
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
    """
    Convert quantity step metadata to decimal precision.
    Supports both:
    - raw step values (e.g. 0.001 -> 3)
    - precision digits from local Symbol parser (e.g. 3 -> 3)
    """
    if step_size is None:
        return 0
    try:
        step = float(step_size)
    except Exception:
        return 0

    if step < 0:
        return 0

    # Local parser stores *_size as precision digits (int-like >= 0).
    if step >= 1 and abs(step - int(step)) < 1e-9:
        return int(step)

    if step == 0:
        return 0

    return int(round(-math.log10(step), 0))

# функция для округления вверх до шага (step_size)
def round_up(num, step_size=1.0):
    precision = get_precision(step_size)
    multiplier = 10 ** precision
    return math.ceil(num * multiplier) / multiplier

# функция для округления вниз до шага (step_size)
def round_down(num, step_size=1.0):
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
    bars = min(len(high), len(low), len(close))
    if bars < 2:
        return 0.0
    
    tr_list = []
    for i in range(1, bars):
        try:
            hi = float(high[i])
            lo = float(low[i])
            prev_close = float(close[i - 1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(hi) and math.isfinite(lo) and math.isfinite(prev_close)):
            continue
        tr = max(
            hi - lo,
            abs(hi - prev_close),
            abs(lo - prev_close)
        )
        if math.isfinite(tr) and tr >= 0:
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

    try:
        entry_price = float(entry_price)
    except (TypeError, ValueError):
        entry_price = 0.0
    try:
        atr = float(atr)
    except (TypeError, ValueError):
        atr = 0.0
    if not math.isfinite(entry_price) or entry_price <= 0:
        return 0.0, 0.0, sl_min_pct, tp_min_pct
    if not math.isfinite(atr) or atr <= 0:
        atr = 0.0
    
    # Calculate ATR-based percentages
    if entry_price > 0 and atr > 0:
        atr_sl = (atr / entry_price) * sl_atr_mult
        atr_tp = (atr / entry_price) * tp_atr_mult
    else:
        atr_sl = sl_min_pct
        atr_tp = tp_min_pct
    
    # Apply min/max bounds
    if not math.isfinite(atr_sl):
        atr_sl = sl_min_pct
    if not math.isfinite(atr_tp):
        atr_tp = tp_min_pct
    sl_pct = max(min(atr_sl, sl_max_pct), sl_min_pct)
    tp_pct = max(min(atr_tp, tp_max_pct), tp_min_pct)
    
    # SAFETY: Clamp to max 95% to prevent negative prices
    # (e.g., SHORT TP = entry*(1-tp_pct) goes negative if tp_pct >= 1.0)
    sl_pct = min(sl_pct, 0.95)
    tp_pct = min(tp_pct, 0.95)
    
    # Calculate prices based on side
    if side == 'LONG':
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)
    else:  # SHORT
        sl_price = entry_price * (1 + sl_pct)
        tp_price = entry_price * (1 - tp_pct)

    if not (math.isfinite(sl_price) and math.isfinite(tp_price)):
        return 0.0, 0.0, sl_pct, tp_pct
    
    return sl_price, tp_price, sl_pct, tp_pct


def should_skip_trade(min_notional: float, calculated_notional: float, max_order_bump: float = 1.5) -> bool:
    """
    Check if trade should be skipped due to minimum order requirements.
    
    Args:
        min_notional: Minimum notional required by exchange
        calculated_notional: Notional calculated by strategy
        max_order_bump: Maximum allowed increase ratio (default 1.5x)
    
    Returns:
        True if trade should be skipped, False otherwise
    """
    if calculated_notional >= min_notional:
        return False  # No adjustment needed
    
    if calculated_notional <= 0:
        return True  # Invalid notional
    
    bump_ratio = min_notional / calculated_notional
    if bump_ratio > max_order_bump:
        print(f"SKIP: Order bump {bump_ratio:.2f}x exceeds threshold {max_order_bump}x")
        return True

    return False  # Small adjustment is acceptable


def expected_reversion_bars(abs_z_now: float, abs_z_target: float, half_life_bars: float) -> float:
    """
    Estimate bars needed for |Z| to decay to target under OU-style half-life dynamics.

    Uses: |Z_t| ~= |Z_0| * 0.5^(t / HL)  ->  t = HL * log2(|Z_0| / |Z_target|).
    Returns 0 when already at/below target or invalid inputs.
    """
    try:
        z0 = float(abs_z_now)
        zt = float(abs_z_target)
        hl = float(half_life_bars)
    except Exception:
        return 0.0

    if z0 <= 0 or zt <= 0 or hl <= 0 or z0 <= zt:
        return 0.0

    try:
        return float(hl * (math.log(z0 / zt, 2)))
    except Exception:
        return 0.0
