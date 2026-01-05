import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import math

# ---------------- PARAMETERS
CAPITAL = 1_000_000.0
MAX_NOTIONAL_PER_PAIR = 0.1
VOL_LOOKBACK = 60

# ------------------ Utility / stats helpers ------------------
def safe_get_slope(params):
    """Return slope param (index 1) robustly for numpy.ndarray or pandas.Series."""
    try:
        # Ensure we have at least two parameters (e.g., const and slope)
        if len(params) < 2:
            return np.nan

        # Use .iloc for positional access on pandas objects (Series)
        if isinstance(params, pd.Series):
            slope = params.iloc[1]
        # Use standard indexing for numpy.ndarray
        else:
            slope = params[1]

        return float(slope)
    except (IndexError, TypeError, ValueError):
        # Catch errors from indexing, or if conversion to float fails
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

def calculate_cointegration(log1, log2):
    """
    log1, log2: numpy arrays of log(prices) (length WINDOW)
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

        if np.isnan(hl) or hl <= 0 or hl > 200:
            return 0, hedge, np.nan, safe_p_value

        flag = 1 if (safe_p_value < 0.05 and t_check) else 0
        return flag, hedge, hl, safe_p_value
    except Exception:
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

def vol_parity_notional(log1, log2, hedge, capital=CAPITAL, max_notional_per_pair=MAX_NOTIONAL_PER_PAIR, lookback=VOL_LOOKBACK):
    cap_pair_usd = capital * max_notional_per_pair
    r1 = np.diff(log1[-lookback:]) if len(log1) >= lookback else np.diff(log1)
    r2 = np.diff(log2[-lookback:]) if len(log2) >= lookback else np.diff(log2)
    sigma1 = np.std(r1) if len(r1) > 0 else 0.0
    sigma2 = np.std(r2) if len(r2) > 0 else 0.0
    w1_raw = 1.0 / sigma1 if sigma1 > 0 else 0.0
    w2_raw = abs(hedge) / sigma2 if sigma2 > 0 else 0.0
    W = w1_raw + w2_raw
    if W <= 0:
        return 0.0, 0.0
    w1 = w1_raw / W
    w2 = w2_raw / W
    return float(cap_pair_usd * w1), float(cap_pair_usd * w2)

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