import os
import sys
import time
import math
import traceback
from multiprocessing import Process
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


CSV_FILE_LARGE = "klines_data_1h_clean_2024.05.24_2025.10.24.csv"
CSV_FILE_100 = "klines_data_1h_clean_100symbols.csv"
CSV_SIGNALS = "cointegrated_pairs_signals_1h_100symbols.csv"
NPZ_CACHE = "log_data_cache.npz"
PARTS_DIR = "parts"
OUT_ALL = "signals_all.csv"

CAPITAL = 1_000_000.0
MAX_NOTIONAL_PER_PAIR = 0.1
VOL_LOOKBACK = 60



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

#  Data preparation 
def prepare_log_cache(csv_path, npz_path=NPZ_CACHE):
    """
    Loads CSV, pivots, computes log(prices) matrix and shifts, saves compressed npz:
    - symbols: array of symbol names
    - dates: array of ISO datetime strings
    - logmat: float array shape (T, n_symbols)
    - shifts: float array shape (n_symbols,)
    """
    print("Preparing log cache from", csv_path)
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns or "Symbol" not in df.columns or "Close" not in df.columns:
        raise ValueError("CSV must contain Date, Symbol, Close columns")

    # Convert Date to datetime and pivot
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"])
    pivot = df.pivot(index="Date", columns="Symbol", values="Close")

    symbols = list(pivot.columns)
    dates = np.array([str(x) for x in pivot.index])

    # calculate shifts and logs
    log_columns = []
    shifts = []
    for sym in symbols:
        col = pivot[sym].values.astype(float)
        minv = np.nanmin(col)
        shift = 0.0
        if minv <= 0 or np.isnan(minv):
            shift = abs(minv) + 1e-6
        with np.errstate(invalid="ignore", divide="ignore"):
            logcol = np.log(col + shift)
        log_columns.append(logcol)
        shifts.append(shift)

    logmat = np.column_stack(log_columns)  # shape (T, n_symbols)
    np.savez_compressed(npz_path, symbols=np.array(symbols), dates=dates, logmat=logmat, shifts=np.array(shifts))
    print("Saved cache to", npz_path)
    return npz_path

#  Worker (top-level for pickling) 
def worker_process(
    proc_id,
    pairs_list,
    npz_path,
    out_part_csv,
    WINDOW=200,
    STEP=50,
    MAX_HALF_LIFE=400,
    BETA_THRESHOLD=0.11,
    Z_ENTRY=2.0,
    CAPITAL=CAPITAL,
    MAX_NOTIONAL_PER_PAIR=MAX_NOTIONAL_PER_PAIR,
    VOL_LOOKBACK=VOL_LOOKBACK,
    TEST_MODE=False,
    TEST_MAX_WINDOWS_PER_PAIR=30,
):
    import time
    import os
    import traceback

    t_proc_start = time.time()

    done_file = out_part_csv.replace(".csv", ".done.txt")

    # CHECKPOINT LOAD 
    done_pairs = set()
    if os.path.exists(done_file):
        with open(done_file, "r") as f:
            for line in f:
                done_pairs.add(line.strip())

    print(
        f"[P{proc_id}] start | total pairs={len(pairs_list)} | "
        f"already done={len(done_pairs)}"
    )

    # LOAD CACHE 
    npz = np.load(npz_path, allow_pickle=True)
    symbols = [s.decode() if isinstance(s, bytes) else s for s in npz["symbols"]]
    dates = npz["dates"]
    logmat = npz["logmat"]
    shifts = npz["shifts"]
    sym2idx = {sym: i for i, sym in enumerate(symbols)}

    results_rows = []
    processed_pairs = 0
    total_pair_time = 0.0
    windows_done = 0

    for (a, b) in pairs_list:
        pairname = f"{a}-{b}"

        if pairname in done_pairs:
            continue
        if a not in sym2idx or b not in sym2idx or "BTCUSDT" not in sym2idx:
            continue

        t_pair_start = time.time()

        try:
            ia = sym2idx[a]
            ib = sym2idx[b]
            ibtc = sym2idx["BTCUSDT"]
            T = logmat.shape[0]
            max_start = T - WINDOW
            if max_start < 0:
                continue

            windows_for_pair = 0

            for start in range(0, max_start + 1, STEP):
                if TEST_MODE and windows_for_pair >= TEST_MAX_WINDOWS_PER_PAIR:
                    break

                end = start + WINDOW
                log1 = logmat[start:end, ia]
                log2 = logmat[start:end, ib]
                logbtc = logmat[start:end, ibtc]

                windows_done += 1
                windows_for_pair += 1

                if (
                    np.isnan(log1).any()
                    or np.isnan(log2).any()
                    or np.isnan(logbtc).any()
                ):
                    continue
                if np.allclose(log1, log1[0]) or np.allclose(log2, log2[0]):
                    continue

                flag, hedge, hl, pval = calculate_cointegration(log1, log2)
                if flag != 1 or np.isnan(hl) or hl <= 0 or hl > MAX_HALF_LIFE:
                    continue

                pr = np.diff(log1) - hedge * np.diff(log2)
                btr = np.diff(logbtc)
                beta_btc = calculate_pair_beta(pr, btr)
                if np.isnan(beta_btc) or abs(beta_btc) >= BETA_THRESHOLD:
                    continue

                spread = log1 - hedge * log2
                z = calculate_z_last(spread)
                if np.isnan(z):
                    continue

                if z >= Z_ENTRY:
                    signal = -1
                elif z <= -Z_ENTRY:
                    signal = 1
                else:
                    continue

                dollar1, dollar2 = vol_parity_notional(
                    log1, log2, hedge,
                    capital=CAPITAL,
                    max_notional_per_pair=MAX_NOTIONAL_PER_PAIR,
                    lookback=VOL_LOOKBACK,
                )

                price1 = max(math.exp(log1[-1]) - shifts[ia], 1e-9)
                price2 = max(math.exp(log2[-1]) - shifts[ib], 1e-9)

                qty1, qty2 = calculate_qty(
                    dollar1, dollar2, price1, price2,
                    capital=CAPITAL,
                )

                results_rows.append({
                    "pair": pairname,
                    "start_index": start,
                    "end_index": end,
                    "start_date": str(dates[start]),
                    "end_date": str(dates[end - 1]),
                    "hedge_ratio": hedge,
                    "half_life": hl,
                    "p_value": pval,
                    "beta_btc": beta_btc,
                    "z": z,
                    "signal": signal,
                    "qty1": qty1,
                    "qty2": qty2,
                    "dollar1": round(dollar1, 2),
                    "dollar2": round(dollar2, 2),
                })

        except Exception:
            traceback.print_exc()

        finally:
            # ---------- TIMING ----------
            pair_time = time.time() - t_pair_start
            processed_pairs += 1
            total_pair_time += pair_time
            avg_pair_time = total_pair_time / processed_pairs

            remaining = len(pairs_list) - processed_pairs - len(done_pairs)
            eta_sec = remaining * avg_pair_time
            eta_min = eta_sec / 60
            eta_hr = eta_min / 60

            if processed_pairs % 10 == 0:
                print(
                    f"[P{proc_id}] done={processed_pairs} | "
                    f"avg={avg_pair_time:.2f}s/pair | "
                    f"ETA ≈ {eta_hr:.2f}h"
                )

            # CHECKPOINT SAVE 
            with open(done_file, "a") as f:
                f.write(pairname + "\n")

   
    df_part = pd.DataFrame(results_rows)
    df_part.to_csv(out_part_csv, index=False)

    print(
        f"[P{proc_id}] FINISHED | "
        f"pairs={processed_pairs} | "
        f"windows={windows_done} | "
        f"time={time.time() - t_proc_start:.1f}s"
    )



# Orchestrator 
def run_parallel_scan(
    csv_path,
    npz_cache,
    n_processes=4,
    TEST_MODE=False,
    TEST_MAX_PAIRS=30,
    EXCLUDED_FROM_PAIRS=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
    WINDOW=200,
    STEP=50,
    MAX_HALF_LIFE=200,
    BETA_THRESHOLD=0.11,
    Z_ENTRY=2.0,
    CAPITAL=CAPITAL,
    MAX_NOTIONAL_PER_PAIR=MAX_NOTIONAL_PER_PAIR,
    VOL_LOOKBACK=VOL_LOOKBACK,
    TEST_MAX_WINDOWS_PER_PAIR=2,
):
    if not os.path.exists(npz_cache):
        prepare_log_cache(csv_path, npz_cache)

    npz = np.load(npz_cache, allow_pickle=True)
    symbols = [s.decode() if isinstance(s, bytes) else s for s in npz["symbols"]]
    candidates = [s for s in symbols if s not in EXCLUDED_FROM_PAIRS]
    all_pairs = list(itertools.combinations(candidates, 2))
    if TEST_MODE:
        all_pairs = all_pairs[:TEST_MAX_PAIRS]

    chunks = np.array_split(np.array(all_pairs, dtype=object), n_processes)
    os.makedirs(PARTS_DIR, exist_ok=True)

    procs = []
    part_files = []

    for i, chunk in enumerate(chunks):
        chunk_list = chunk.tolist() if len(chunk) > 0 else []
        out_part = os.path.join(PARTS_DIR, f"part_{i+1:02d}.csv")
        part_files.append(out_part)

        p = Process(
            target=worker_process,
            args=(
                i + 1,
                chunk_list,
                npz_cache,
                out_part,
                WINDOW,
                STEP,
                MAX_HALF_LIFE,
                BETA_THRESHOLD,
                Z_ENTRY,
                CAPITAL,
                MAX_NOTIONAL_PER_PAIR,
                VOL_LOOKBACK,
                TEST_MODE,
                TEST_MAX_WINDOWS_PER_PAIR,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Combine results into a single DataFrame
    dfs = []
    for pf in part_files:
        if os.path.exists(pf):
            try:
                d = pd.read_csv(pf)
                if not d.empty:
                    dfs.append(d)
            except Exception:
                pass

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.to_csv(OUT_ALL, index=False)
        print(f"Wrote combined results to {OUT_ALL} ({len(df_all)} rows)")
        return df_all
    else:
        print("No results found in parts.")
        return pd.DataFrame()


# def ensure_downloads():
#     # If files are absent, attempt to download via gdown (assuming gdown installed)
#     if not os.path.exists(CSV_FILE_100):
#         print("Downloading CSV files with gdown (if available)...")
#         os.system(f"gdown --id 1ikr1Cq4qW5Dkn3NN2twxjs-1jaTSCtZy -O {CSV_FILE_LARGE}")
#         os.system(f"gdown --id 1FA2QRoQ9vuO-Z9EgPHgLXtWinmBlovHm -O {CSV_FILE_100}")
#         os.system(f"gdown --id 1kmX9ActaGZqDT66CA4VJ60bYvXd7ej2M -O {CSV_SIGNALS}")

if __name__ == "__main__":
    # Put imports and top-level operations under main guard for Windows multiprocessing
    # ensure_downloads()

    # Configuration
    csv_file = "klines_data_1h_clean_2024.05.24_2025.10.24.csv"
    npz_cache = NPZ_CACHE
    n_processes = 16
    TEST_MODE = False
    TEST_MAX_PAIRS = 50
    TEST_MAX_WINDOWS_PER_PAIR = 2

    # sanity checks
    # if not os.path.exists(csv_file):
    #     print("CSV file not found:", csv_file)
    #     sys.exit(1)

    start = time.time()
    print("Starting parallel scan...")
    df_signals = run_parallel_scan(
        csv_path=csv_file,
        npz_cache=npz_cache,
        n_processes=n_processes,
        TEST_MODE=TEST_MODE,
        TEST_MAX_PAIRS=TEST_MAX_PAIRS,
        WINDOW=200,
        STEP=50,
        MAX_HALF_LIFE=400,
        BETA_THRESHOLD=0.11,
        Z_ENTRY=2.0,
        CAPITAL=CAPITAL,
        MAX_NOTIONAL_PER_PAIR=MAX_NOTIONAL_PER_PAIR,
        VOL_LOOKBACK=VOL_LOOKBACK,
        TEST_MAX_WINDOWS_PER_PAIR=TEST_MAX_WINDOWS_PER_PAIR,
    )
    print("Done. time elapsed: {:.1f}s".format(time.time() - start))
    print("Total signals:", len(df_signals))
    if not df_signals.empty:
        print(df_signals.head())
