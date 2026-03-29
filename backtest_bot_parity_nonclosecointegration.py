from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import itertools
import json
import math
import os
import random
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import utils

try:
    from statsmodels.tools.sm_exceptions import CollinearityWarning

    warnings.filterwarnings("ignore", category=CollinearityWarning)
except Exception:
    pass


# Strict mode by default: do not forward-fill OHLC at all.
# Missing candles remain NaN and are naturally rejected by execution/window checks.
MAX_FFILL_GAP_BARS = 0


CLOSE_REASONS = {
    "z_tp": "Z-Score Take Profit",
    "z_sl": "Z-Score Stop Loss",
    "hardware_tp": "Hardware Take Profit",
    "hardware_sl": "Hardware Stop Loss",
    "circuit": "Circuit Breaker",
    "broken_coint": "Broken Correlation",
    "beta_drift": "Beta Drift",
    "beta_critical": "Beta Critical",
    "time_exit": "Time Exit",
    "force_close": "Force Close",
}


def timeframe_seconds(timeframe: str) -> int:
    tf = str(timeframe or "1h").strip().lower()
    try:
        if tf.endswith("m"):
            return max(60, int(tf[:-1]) * 60)
        if tf.endswith("h"):
            return max(3600, int(tf[:-1]) * 3600)
        if tf.endswith("d"):
            return max(86400, int(tf[:-1]) * 86400)
    except Exception:
        pass
    return 3600


def timeframe_to_periods_per_year(timeframe: str) -> float:
    sec = max(60, timeframe_seconds(timeframe))
    return float((365 * 24 * 3600) / sec)


def auto_window_size(timeframe: str) -> int:
    tf = str(timeframe or "1h").strip().lower()
    if tf == "1m":
        return 720
    if tf == "5m":
        return 576
    if tf == "15m":
        return 480
    if tf == "1h":
        return 336
    if tf == "4h":
        return 180
    if tf == "1d":
        return 90
    return 336


def to_canonical_pair(a: str, b: str) -> tuple[str, str]:
    a_u = str(a).strip().upper()
    b_u = str(b).strip().upper()
    return (a_u, b_u) if a_u <= b_u else (b_u, a_u)


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def parse_timestamp_utc(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid UTC timestamp/date: {value}")
    return ts


def infer_timeframe_from_index(index: pd.DatetimeIndex) -> str:
    if len(index) < 3:
        return "1h"
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    if diffs.empty:
        return "1h"
    med = float(diffs.median())
    mapping = {
        60: "1m",
        300: "5m",
        900: "15m",
        1800: "30m",
        3600: "1h",
        14400: "4h",
        86400: "1d",
    }
    best = min(mapping.keys(), key=lambda x: abs(x - med))
    return mapping.get(best, "1h")


def _filter_df_by_utc_window(
    df: pd.DataFrame,
    column: str,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    if df is None or df.empty or column not in df.columns:
        return df.copy() if df is not None else pd.DataFrame()
    out = df.copy()
    out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    mask = out[column].notna()
    if start_ts is not None:
        mask &= out[column] >= start_ts
    if end_ts is not None:
        mask &= out[column] < end_ts
    return out.loc[mask].reset_index(drop=True)


@dataclass
class BacktestParams:
    timeframe: str = "1h"
    window_size: int = 0
    capital: float = 100.0
    leverage: int = 20  # informational only, not used in sizing
    max_notional_pct: float = 0.40
    z_entry: float = 1.8
    z_entry_max: float = 2.5
    z_exit: float = 0.05
    z_stop: float = 4.0
    commission_rate: float = 0.0004
    slippage_rate: float = 0.0005
    hardware_sltp_mode: str = "off"  # off | monitor | exit
    hardware_sl_enabled: bool = True
    hardware_tp_enabled: bool = True
    sl_atr_mult: float = 2.5
    sl_min_pct: float = 0.10
    sl_max_pct: float = 0.30
    tp_atr_mult: float = 4.0
    tp_min_pct: float = 0.15
    tp_max_pct: float = 0.50
    p_value_threshold: float = 0.05
    hedge_min: float = 0.3
    hedge_max: float = 3.0
    beta_threshold: float = 0.11
    beta_alert_threshold: float = 0.30
    beta_critical: float = 1.0
    circuit_breaker_pct: float = 0.50
    signal_confirm_sec: int = 10
    coint_stability_min_bars: int = 2
    coint_broken_grace_bars: int = 0  # bars to wait before closing on broken coint (0=immediate)
    entry_et_target_abs_z: float = 0.5
    max_active_pairs: int = 3
    max_idle_pairs: int = 150
    hold_multiplier: float = 3.0
    max_hold_days: float = 30.0
    sl_reentry_cooldown_sec: int = 0
    close_retry_cooldown_sec: int = 30
    discovery_every_bars: int = 1
    discovery_shards: int = 4
    discovery_max_pairs_per_cycle: int = 12000
    max_symbols: int = 300
    top_pairs_limit: int = 300
    funding_csv: str = ""
    progress_every_bars: int = 100
    hl_min_days: float = 0.25
    hl_max_days: float = 2.0

    def resolved_window_size(self) -> int:
        if int(self.window_size or 0) > 0:
            return int(self.window_size)
        return auto_window_size(self.timeframe)

    def timeframe_sec(self) -> int:
        return timeframe_seconds(self.timeframe)

    def confirm_bars(self) -> int:
        sec = max(1, self.timeframe_sec())
        # Parity with runtime behavior:
        # if confirm_sec is smaller than one candle, signal can confirm within
        # the same candle period (no extra bar delay in backtest logic).
        raw = float(self.signal_confirm_sec or 0)
        if raw <= 0:
            return 0
        return max(0, int(math.floor(raw / sec)))

    def cooldown_bars(self, reason: str) -> int:
        if reason in {"z_sl"}:
            raw = int(self.sl_reentry_cooldown_sec or 0)
            if raw <= 0:
                raw = int(self.close_retry_cooldown_sec or 30)
            return max(1, int(math.ceil(raw / max(1, self.timeframe_sec()))))
        return 0


@dataclass
class CandidatePair:
    symbol1: str
    symbol2: str
    score: float = 0.0
    source: str = "best_pairs"

    @property
    def key(self) -> tuple[str, str]:
        return to_canonical_pair(self.symbol1, self.symbol2)


@dataclass
class PairSnapshot:
    valid: bool
    flag: int = 0
    hedge: float = np.nan
    half_life: float = np.nan
    pvalue: float = np.nan
    zscore: float = np.nan
    beta: float = np.nan


@dataclass
class PairState:
    symbol1: str
    symbol2: str
    discovered_idx: int = 0
    position_status: int = 0
    hedge_ratio: float = 0.0
    half_life: float = 0.0
    beta_btc: float = 0.0
    last_pvalue: float = 0.0
    last_z_score: float = 0.0
    coint_streak_bars: int = 0
    coint_run_len: int = 0
    coint_broken_count: int = 0  # consecutive bars with broken coint while in position
    pending_signal: float | None = None
    pending_since_idx: int | None = None
    pending_source: str = ""
    quality_score: float = 0.0
    quality_updated_idx: int = -1
    recent_fail_penalty: float = 0.0
    close_cooldown_until_idx: int = -1
    reentry_block_idx: int = -1
    last_eval_idx: int = -1
    coint_last_full_idx: int = -1  # bar index of last full cointegration run

    @property
    def key(self) -> tuple[str, str]:
        return to_canonical_pair(self.symbol1, self.symbol2)


@dataclass
class Position:
    trade_id: int
    pair_key: tuple[str, str]
    symbol1: str
    symbol2: str
    direction: int
    qty1_signed: float
    qty2_signed: float
    entry_price1: float
    entry_price2: float
    entry_idx: int
    entry_time: pd.Timestamp
    entry_fee: float
    entry_notional: float
    hedge_ratio: float
    entry_z: float
    entry_beta: float
    entry_pvalue: float
    entry_half_life: float
    coint_streak_at_entry: int
    expected_reversion_hours: float
    hardware_sl1: float = np.nan
    hardware_tp1: float = np.nan
    hardware_sl2: float = np.nan
    hardware_tp2: float = np.nan
    hardware_sl_touched: bool = False
    hardware_tp_touched: bool = False
    hardware_first_touch_reason: str = ""
    hardware_first_touch_symbol: str = ""
    hardware_first_touch_note: str = ""
    hardware_first_touch_idx: int = -1
    funding_leg1: float = 0.0
    funding_leg2: float = 0.0
    funding_total: float = 0.0
    funding_events: int = 0
    min_unrealized_pnl: float = 0.0
    max_unrealized_pnl: float = 0.0


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    symbols: list[str]
    symbol_to_idx: dict[str, int]
    open_arr: np.ndarray
    high_arr: np.ndarray
    low_arr: np.ndarray
    close_arr: np.ndarray
    volume_arr: np.ndarray


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict[str, Any]
    coint_phases: pd.DataFrame
    ledger: pd.DataFrame
    params: dict[str, Any]


@dataclass
class FundingData:
    rate_arr: np.ndarray
    source_path: str
    loaded_records: int
    matched_records: int
    skipped_records: int


def load_klines_market_data(csv_path: str, max_symbols: int) -> MarketData:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Klines file not found: {csv_path}")

    df = pd.read_csv(path)
    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume", "Symbol"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Symbol", "Open", "High", "Low", "Close"])
    df = df[(df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)]
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # Some data sources can contain repeated candles for the same Date+Symbol.
    # Collapse duplicates before pivot to avoid "Index contains duplicate entries".
    dup_mask = df.duplicated(subset=["Date", "Symbol"], keep=False)
    if bool(dup_mask.any()):
        dup_rows = int(dup_mask.sum())
        dup_keys = int(df.loc[dup_mask, ["Date", "Symbol"]].drop_duplicates().shape[0])
        df = (
            df.groupby(["Date", "Symbol"], as_index=False, sort=False)
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .sort_values(["Symbol", "Date"])
            .reset_index(drop=True)
        )
        print(
            f"[WARN] Collapsed duplicate candles: rows={dup_rows}, keys={dup_keys} "
            f"in {path.name}"
        )

    vol_rank = (
        df.groupby("Symbol")["Volume"]
        .median()
        .sort_values(ascending=False)
    )
    selected_symbols = list(vol_rank.head(int(max(2, max_symbols))).index)
    if "BTCUSDT" in df["Symbol"].values and "BTCUSDT" not in selected_symbols:
        selected_symbols.append("BTCUSDT")
    df = df[df["Symbol"].isin(selected_symbols)].copy()

    pivots = {}
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        piv = df.pivot(index="Date", columns="Symbol", values=col).sort_index()
        pivots[col] = piv

    close_df = pivots["Close"].copy()
    valid_cols = close_df.columns[close_df.notna().sum() >= 50]
    close_df = close_df[valid_cols]

    for col in list(pivots.keys()):
        piv = pivots[col][close_df.columns]
        # IMPORTANT: no backfill from future candles (avoid look-ahead leakage).
        # Keep long/short gaps as NaN by default (strict no-synthetic-candle mode).
        if int(MAX_FFILL_GAP_BARS) > 0:
            piv = piv.ffill(limit=int(MAX_FFILL_GAP_BARS))
        pivots[col] = piv

    symbols = list(close_df.columns)
    if len(symbols) < 3:
        raise ValueError("Not enough symbols in klines after filtering.")

    symbol_to_idx = {s: i for i, s in enumerate(symbols)}
    return MarketData(
        dates=pivots["Close"].index,
        symbols=symbols,
        symbol_to_idx=symbol_to_idx,
        open_arr=pivots["Open"].to_numpy(dtype=float),
        high_arr=pivots["High"].to_numpy(dtype=float),
        low_arr=pivots["Low"].to_numpy(dtype=float),
        close_arr=pivots["Close"].to_numpy(dtype=float),
        volume_arr=pivots["Volume"].to_numpy(dtype=float),
    )


def load_funding_data(csv_path: str | None, market: MarketData) -> FundingData | None:
    path_text = str(csv_path or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Funding CSV not found: {csv_path}")

    df = pd.read_csv(path)
    required_cols = {"Symbol", "fundingTime", "fundingRate"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in funding CSV {csv_path}: {sorted(missing)}")

    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    # Binance funding timestamps can carry tiny millisecond offsets.
    # Normalize them to second precision so they align with bar timestamps.
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], utc=True, errors="coerce").dt.round("s")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.dropna(subset=["Symbol", "fundingTime", "fundingRate"]).copy()
    if df.empty:
        raise ValueError(f"Funding CSV has no valid rows: {csv_path}")

    date_to_idx = {pd.Timestamp(ts).round("s"): i for i, ts in enumerate(market.dates)}
    rate_arr = np.zeros((len(market.dates), len(market.symbols)), dtype=np.float64)
    matched = 0
    skipped = 0

    for row in df.itertuples(index=False):
        sym = str(row.Symbol).upper().strip()
        ts = pd.Timestamp(row.fundingTime)
        rate = float(row.fundingRate)
        sym_idx = market.symbol_to_idx.get(sym)
        bar_idx = date_to_idx.get(ts)
        if sym_idx is None or bar_idx is None:
            skipped += 1
            continue
        rate_arr[bar_idx, sym_idx] += rate
        matched += 1

    return FundingData(
        rate_arr=rate_arr,
        source_path=str(path.resolve()),
        loaded_records=int(len(df)),
        matched_records=int(matched),
        skipped_records=int(skipped),
    )


def _extract_pair_entries(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        pairs = raw.get("pairs")
        if isinstance(pairs, list):
            return pairs
    return []


def _parse_pair_text(entry: Any) -> tuple[str, str] | None:
    pair_str = ""
    if isinstance(entry, str):
        pair_str = entry.strip().upper()
    elif isinstance(entry, dict):
        pair_str = str(entry.get("pair", "")).strip().upper()
        if not pair_str:
            s1 = str(entry.get("symbol1", "")).strip().upper()
            s2 = str(entry.get("symbol2", "")).strip().upper()
            if s1 and s2:
                pair_str = f"{s1}-{s2}"
    if "-" not in pair_str:
        return None
    a, b = [x.strip().upper() for x in pair_str.split("-", 1)]
    if not a or not b or a == b:
        return None
    return to_canonical_pair(a, b)


def load_pair_blacklist(pair_blacklist_path: str | None) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not pair_blacklist_path:
        return keys

    path = Path(str(pair_blacklist_path))
    if not path.exists():
        return keys

    try:
        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for row in _extract_pair_entries(raw):
                key = _parse_pair_text(row)
                if key is not None:
                    keys.add(key)
        elif suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
            if not df.empty:
                col = "pair" if "pair" in df.columns else str(df.columns[0])
                for val in df[col].dropna().astype(str):
                    key = _parse_pair_text(val)
                    if key is not None:
                        keys.add(key)
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    key = _parse_pair_text(line)
                    if key is not None:
                        keys.add(key)
    except Exception as exc:
        print(f"[WARN] Could not load pair blacklist from {path}: {exc}")

    return keys


def load_candidate_pairs(
    best_pairs_path: str,
    symbols: list[str],
    limit: int,
    supplemental_pairs_limit: int = 0,
    supplemental_symbols: int = 40,
    blacklist_keys: set[tuple[str, str]] | None = None,
) -> list[CandidatePair]:
    symbol_set = set(symbols)
    path = Path(best_pairs_path)
    blacklist = blacklist_keys or set()
    primaries: list[CandidatePair] = []

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            entries = _extract_pair_entries(raw)
            for row in entries:
                key = _parse_pair_text(row)
                if key is None:
                    continue
                a, b = key
                if a not in symbol_set or b not in symbol_set:
                    continue
                if key in blacklist:
                    continue
                score = 0.0
                source = "best_pairs"
                if isinstance(row, dict):
                    score = float(row.get("score", 0.0) or 0.0)
                    source = str(row.get("source", "best_pairs"))
                primaries.append(CandidatePair(symbol1=a, symbol2=b, score=score, source=source))
        except Exception as exc:
            print(f"[WARN] Failed to parse {best_pairs_path}: {exc}")
    else:
        print(f"[WARN] best_pairs file not found: {best_pairs_path}")

    # Deduplicate primary pairs by canonical key and keep the best score.
    by_key: dict[tuple[str, str], CandidatePair] = {}
    for cp in primaries:
        cur = by_key.get(cp.key)
        if cur is None or cp.score > cur.score:
            by_key[cp.key] = cp
    primary_unique = list(by_key.values())
    primary_unique.sort(key=lambda x: (-float(x.score), x.symbol1, x.symbol2))
    if limit > 0:
        primary_unique = primary_unique[:limit]
    if not primary_unique:
        print("[WARN] No valid primary pairs loaded from best_pairs; supplemental universe will be used.")

    # Supplemental universe: scanned only when there is no primary signal.
    # If best_pairs is empty, keep old fallback behavior.
    supp_limit = int(supplemental_pairs_limit or 0)
    if not primary_unique and supp_limit <= 0:
        supp_limit = max(1, int(limit or 300))

    supplemental: list[CandidatePair] = []
    if supp_limit > 0:
        max_syms = max(2, int(supplemental_symbols or 40))
        non_btc = [s for s in symbols if s != "BTCUSDT"][: min(max_syms, len(symbols))]
        existing_keys = {cp.key for cp in primary_unique}
        for a, b in itertools.combinations(non_btc, 2):
            key = to_canonical_pair(a, b)
            if key in existing_keys or key in blacklist:
                continue
            supplemental.append(
                CandidatePair(symbol1=key[0], symbol2=key[1], score=-1.0, source="supplemental_combo")
            )
            existing_keys.add(key)
            if len(supplemental) >= supp_limit:
                break

    return primary_unique + supplemental


def _compute_snapshot(args: tuple) -> tuple:
    """Process-safe worker for parallel cointegration computation.

    Accepts small data slices (not the whole market) to minimise pickle overhead.
    Returns (key, result_tuple_or_None).
    """
    key, c1, c2, cb, p_value_threshold = args
    try:
        if len(c1) < 30 or len(c2) < 30:
            return (key, None)
        if not (np.isfinite(c1).all() and np.isfinite(c2).all()):
            return (key, None)
        if (c1 <= 0).any() or (c2 <= 0).any():
            return (key, None)

        log1 = np.log(c1)
        log2 = np.log(c2)
        flag, hedge, hl, pval = utils.calculate_cointegration(
            log1, log2, p_value_threshold=float(p_value_threshold), strict_hl=False,
        )
        if np.isnan(hedge):
            return (key, None)

        spread = log1 - hedge * log2
        z = utils.calculate_z_last(spread)
        beta = np.nan
        if cb is not None and len(cb) == len(c1) and (cb > 0).all() and np.isfinite(cb).all():
            logb = np.log(cb)
            sr = np.diff(log1) - hedge * np.diff(log2)
            br = np.diff(logb)
            beta = utils.calculate_pair_beta(sr, br)

        return (key, (
            int(flag),
            float(hedge),
            float(hl) if not np.isnan(hl) else float("nan"),
            float(pval) if not np.isnan(pval) else float("nan"),
            float(z) if z is not None and not np.isnan(z) else float("nan"),
            float(beta) if not np.isnan(beta) else float("nan"),
        ))
    except Exception:
        return (key, None)


# ── Worker-local market data (populated once per process via initializer) ──────
_W_CLOSE: np.ndarray | None = None
_W_BTX_IDX: int | None = None


def _worker_init(close_arr_bytes: bytes, shape: tuple, btc_idx: int | None) -> None:
    """Called once per worker process when pool starts.
    Reconstructs close_arr from raw bytes — no repeated pickling overhead per task.
    """
    global _W_CLOSE, _W_BTX_IDX
    arr = np.frombuffer(close_arr_bytes, dtype=np.float64).reshape(shape).copy()
    _W_CLOSE = arr
    _W_BTX_IDX = btc_idx


def _compute_snapshot_indexed(args: tuple) -> tuple:
    """Fast worker: uses pre-loaded close_arr in worker memory.
    Args per call: (key, ia, ib, idx, min_dp, p_thresh)  — only 6 small values, no arrays.
    """
    key, ia, ib, idx, min_dp, p_thresh = args
    close_arr = _W_CLOSE
    btc_idx = _W_BTX_IDX
    try:
        start = idx - min_dp + 1
        if start < 0 or close_arr is None:
            return (key, None)
        c1 = close_arr[start:idx + 1, ia]
        c2 = close_arr[start:idx + 1, ib]
        if len(c1) < min_dp or len(c2) < min_dp:
            return (key, None)
        if not (np.isfinite(c1).all() and np.isfinite(c2).all()):
            return (key, None)
        if (c1 <= 0).any() or (c2 <= 0).any():
            return (key, None)

        log1 = np.log(c1)
        log2 = np.log(c2)
        flag, hedge, hl, pval = utils.calculate_cointegration(
            log1, log2, p_value_threshold=float(p_thresh), strict_hl=False,
        )
        if np.isnan(hedge):
            return (key, None)

        spread = log1 - hedge * log2
        z = utils.calculate_z_last(spread)
        beta = np.nan
        if btc_idx is not None:
            cb = close_arr[start:idx + 1, btc_idx]
            if len(cb) == len(c1) and (cb > 0).all() and np.isfinite(cb).all():
                logb = np.log(cb)
                sr = np.diff(log1) - hedge * np.diff(log2)
                br = np.diff(logb)
                beta = utils.calculate_pair_beta(sr, br)

        return (key, (
            int(flag),
            float(hedge),
            float(hl) if not np.isnan(hl) else float("nan"),
            float(pval) if not np.isnan(pval) else float("nan"),
            float(z) if z is not None and not np.isnan(z) else float("nan"),
            float(beta) if not np.isnan(beta) else float("nan"),
        ))
    except Exception:
        return (key, None)


def _compute_snapshot_with_arrays(
    close_arr: np.ndarray,
    btc_idx: int | None,
    key: tuple[str, str],
    ia: int,
    ib: int,
    idx: int,
    min_dp: int,
    p_thresh: float,
) -> tuple[tuple[str, str], tuple | None]:
    """Same logic as _compute_snapshot_indexed, but with explicit close_arr.
    Used for sequential fallback (n_workers=1) to avoid dependency on worker globals.
    """
    try:
        start = idx - min_dp + 1
        if start < 0 or close_arr is None:
            return (key, None)
        c1 = close_arr[start:idx + 1, ia]
        c2 = close_arr[start:idx + 1, ib]
        if len(c1) < min_dp or len(c2) < min_dp:
            return (key, None)
        if not (np.isfinite(c1).all() and np.isfinite(c2).all()):
            return (key, None)
        if (c1 <= 0).any() or (c2 <= 0).any():
            return (key, None)

        log1 = np.log(c1)
        log2 = np.log(c2)
        flag, hedge, hl, pval = utils.calculate_cointegration(
            log1, log2, p_value_threshold=float(p_thresh), strict_hl=False,
        )
        if np.isnan(hedge):
            return (key, None)

        spread = log1 - hedge * log2
        z = utils.calculate_z_last(spread)
        beta = np.nan
        if btc_idx is not None:
            cb = close_arr[start:idx + 1, btc_idx]
            if len(cb) == len(c1) and (cb > 0).all() and np.isfinite(cb).all():
                logb = np.log(cb)
                sr = np.diff(log1) - hedge * np.diff(log2)
                br = np.diff(logb)
                beta = utils.calculate_pair_beta(sr, br)

        return (key, (
            int(flag),
            float(hedge),
            float(hl) if not np.isnan(hl) else float("nan"),
            float(pval) if not np.isnan(pval) else float("nan"),
            float(z) if z is not None and not np.isnan(z) else float("nan"),
            float(beta) if not np.isnan(beta) else float("nan"),
        ))
    except Exception:
        return (key, None)


class BotParityBacktester:
    def __init__(self, market: MarketData, params: BacktestParams, candidates: list[CandidatePair],
                 n_workers: int = 0, idle_eval_every: int = 1,
                 coint_recompute_every: int = 1,
                 exact_mode: bool = True,
                 discovery_recheck_bars: int = 0,
                 supplemental_when_no_primary_signal: bool = True):
        self.market = market
        self.params = params
        self.candidates = candidates
        self.n_bars = len(market.dates)
        self.min_data_points = max(30, int(params.resolved_window_size()))
        self.timeframe_sec = params.timeframe_sec()
        self.periods_per_year = timeframe_to_periods_per_year(params.timeframe)
        self.confirm_bars = params.confirm_bars()
        self.max_hold_bars_days = max(1, int(round(params.max_hold_days * utils.CANDLES_PER_DAY.get(params.timeframe, 24))))
        self.hardware_sltp_mode = str(getattr(params, "hardware_sltp_mode", "off") or "off").strip().lower()
        if self.hardware_sltp_mode not in {"off", "monitor", "exit"}:
            self.hardware_sltp_mode = "off"
        self.hardware_sl_enabled = bool(getattr(params, "hardware_sl_enabled", True))
        self.hardware_tp_enabled = bool(getattr(params, "hardware_tp_enabled", True))
        self.hardware_sltp_active = (
            self.hardware_sltp_mode != "off"
            and (self.hardware_sl_enabled or self.hardware_tp_enabled)
        )

        self.btc_idx = market.symbol_to_idx.get("BTCUSDT", None)
        self.primary_candidate_keys = [cp.key for cp in candidates if str(cp.source) != "supplemental_combo"]
        self.supplemental_candidate_keys = [cp.key for cp in candidates if str(cp.source) == "supplemental_combo"]
        self.candidate_keys = self.primary_candidate_keys + self.supplemental_candidate_keys
        self.primary_candidate_set = set(self.primary_candidate_keys)
        self.supplemental_candidate_set = set(self.supplemental_candidate_keys)
        self.supplemental_when_no_primary_signal = bool(supplemental_when_no_primary_signal)
        if int(n_workers or 0) > 0:
            self.n_workers = max(1, int(n_workers))
        else:
            cpu = os.cpu_count() or 2
            # Windows spawn+IPC overhead grows quickly after ~12 workers for this workload.
            # Keep a balanced default; explicit --n-workers still overrides this.
            self.n_workers = max(1, min(12, cpu - 2))
        self.exact_mode = bool(exact_mode)
        if self.exact_mode:
            # Strict parity profile: no evaluation throttling, no deferred coint refresh,
            # and no discovery retry cooldowns that can skip valid opportunities.
            self.idle_eval_every = 1
            self.coint_recompute_every = 1
            self.discovery_recheck_bars = 0
        else:
            self.idle_eval_every = max(1, idle_eval_every)
            self.coint_recompute_every = max(1, coint_recompute_every)
            self.discovery_recheck_bars = max(0, int(discovery_recheck_bars))
        self.near_entry_z_trigger = float(max(0.0, float(self.params.z_entry) * 0.8))
        self._p_thresh = float(params.p_value_threshold)
        self._min_dp = self.min_data_points

        self.pair_states: dict[tuple[str, str], PairState] = {}
        self.positions: dict[tuple[str, str], Position] = {}
        self.symbol_owner: dict[str, tuple[str, str]] = {}
        self.discovery_retry_until_idx: dict[tuple[str, str], int] = {}
        self.key_to_indices: dict[tuple[str, str], tuple[int, int]] = {}

        self.pending_entries: dict[tuple[str, str], dict[str, Any]] = {}
        self.pending_exits: dict[tuple[str, str], dict[str, Any]] = {}
        self.discovery_round: int = 0

        self.cash = float(params.capital)
        self.trade_seq = 0
        self.trades: list[dict[str, Any]] = []
        self.trade_row_idx: dict[int, int] = {}
        self.ledger: list[dict[str, Any]] = []
        self.equity_rows: list[dict[str, Any]] = []
        self.coint_phase_rows: list[dict[str, Any]] = []
        self.discovered_keys_seen: set[tuple[str, str]] = set()
        self.funding_data = load_funding_data(getattr(params, "funding_csv", ""), market)
        self.funding_enabled = self.funding_data is not None
        self.total_funding_cash = 0.0

        # Build pool with market data baked into each worker at startup.
        # Workers receive close_arr ONCE (as raw bytes) — then per-task args
        # are just 6 integers/floats, eliminating per-bar IPC array pickling.
        self._pool: concurrent.futures.ProcessPoolExecutor | None = None
        if self.n_workers > 1:
            close_bytes = market.close_arr.astype(np.float64).tobytes()
            close_shape = market.close_arr.shape
            self._pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=_worker_init,
                initargs=(close_bytes, close_shape, self.btc_idx),
            )

    def _get_pair_indices(self, key: tuple[str, str]) -> tuple[int, int] | tuple[None, None]:
        cached = self.key_to_indices.get(key)
        if cached is not None:
            return cached
        a, b = key
        ia = self.market.symbol_to_idx.get(a)
        ib = self.market.symbol_to_idx.get(b)
        if ia is None or ib is None:
            return (None, None)
        self.key_to_indices[key] = (ia, ib)
        return (ia, ib)

    def _compute_atr_for_symbol(self, sym_idx: int, end_idx: int, lookback: int = 100) -> float:
        end = max(0, int(end_idx))
        if end <= 1:
            return 0.0
        start = max(0, end - max(20, int(lookback)))
        high = self.market.high_arr[start:end, sym_idx]
        low = self.market.low_arr[start:end, sym_idx]
        close = self.market.close_arr[start:end, sym_idx]
        mask = (
            np.isfinite(high)
            & np.isfinite(low)
            & np.isfinite(close)
            & (high > 0)
            & (low > 0)
            & (close > 0)
        )
        if int(mask.sum()) < 2:
            return 0.0
        return float(
            utils.calculate_atr(
                high[mask].astype(float).tolist(),
                low[mask].astype(float).tolist(),
                close[mask].astype(float).tolist(),
            )
        )

    def _funding_price_for_bar(self, idx: int, sym_idx: int) -> float:
        p = float(self.market.open_arr[idx, sym_idx])
        if np.isfinite(p) and p > 0:
            return p
        p = float(self.market.close_arr[idx, sym_idx])
        if np.isfinite(p) and p > 0:
            return p
        return 0.0

    def _apply_funding_for_bar(self, idx: int) -> None:
        if not self.funding_enabled or self.funding_data is None or not self.positions:
            return
        rate_row = self.funding_data.rate_arr[idx]
        if rate_row is None or not np.any(rate_row):
            return

        ts = self.market.dates[idx]
        for key, pos in list(self.positions.items()):
            i1 = self.market.symbol_to_idx.get(pos.symbol1)
            i2 = self.market.symbol_to_idx.get(pos.symbol2)
            if i1 is None or i2 is None:
                continue

            rate1 = float(rate_row[i1])
            rate2 = float(rate_row[i2])
            if rate1 == 0.0 and rate2 == 0.0:
                continue

            price1 = self._funding_price_for_bar(idx, i1)
            price2 = self._funding_price_for_bar(idx, i2)
            notional1 = abs(float(pos.qty1_signed)) * price1 if price1 > 0 else 0.0
            notional2 = abs(float(pos.qty2_signed)) * price2 if price2 > 0 else 0.0
            sign1 = 1.0 if float(pos.qty1_signed) > 0 else (-1.0 if float(pos.qty1_signed) < 0 else 0.0)
            sign2 = 1.0 if float(pos.qty2_signed) > 0 else (-1.0 if float(pos.qty2_signed) < 0 else 0.0)
            funding1 = -sign1 * notional1 * rate1
            funding2 = -sign2 * notional2 * rate2
            funding_total = float(funding1 + funding2)
            if funding_total == 0.0:
                continue

            pos.funding_leg1 += float(funding1)
            pos.funding_leg2 += float(funding2)
            pos.funding_total += funding_total
            pos.funding_events += 1
            self.cash += funding_total
            self.total_funding_cash += funding_total
            self.ledger.append(
                {
                    "time": ts,
                    "pair": f"{pos.symbol1}-{pos.symbol2}",
                    "type": "funding",
                    "cash_change": funding_total,
                    "note": (
                        f"trade_id={pos.trade_id} "
                        f"{pos.symbol1}:{rate1:+.8f} {funding1:+.6f} | "
                        f"{pos.symbol2}:{rate2:+.8f} {funding2:+.6f}"
                    ),
                }
            )

    def _detect_leg_hardware_hits(
        self,
        qty_signed: float,
        low_price: float,
        high_price: float,
        sl_price: float,
        tp_price: float,
    ) -> tuple[bool, bool]:
        if not np.isfinite(low_price) or not np.isfinite(high_price) or low_price <= 0 or high_price <= 0:
            return (False, False)
        is_long = float(qty_signed) > 0
        sl_hit = False
        tp_hit = False
        if self.hardware_sl_enabled and np.isfinite(sl_price) and sl_price > 0:
            sl_hit = (low_price <= sl_price) if is_long else (high_price >= sl_price)
        if self.hardware_tp_enabled and np.isfinite(tp_price) and tp_price > 0:
            tp_hit = (high_price >= tp_price) if is_long else (low_price <= tp_price)
        return (bool(sl_hit), bool(tp_hit))

    def _check_hardware_sltp_for_bar(
        self,
        st: PairState,
        pos: Position,
        idx: int,
    ) -> dict[str, Any] | None:
        if not self.hardware_sltp_active:
            return None

        i1 = self.market.symbol_to_idx.get(pos.symbol1)
        i2 = self.market.symbol_to_idx.get(pos.symbol2)
        if i1 is None or i2 is None:
            return None

        low1 = float(self.market.low_arr[idx, i1])
        high1 = float(self.market.high_arr[idx, i1])
        low2 = float(self.market.low_arr[idx, i2])
        high2 = float(self.market.high_arr[idx, i2])
        close1 = float(self.market.close_arr[idx, i1])
        close2 = float(self.market.close_arr[idx, i2])

        sl1_hit, tp1_hit = self._detect_leg_hardware_hits(
            pos.qty1_signed, low1, high1, float(pos.hardware_sl1), float(pos.hardware_tp1)
        )
        sl2_hit, tp2_hit = self._detect_leg_hardware_hits(
            pos.qty2_signed, low2, high2, float(pos.hardware_sl2), float(pos.hardware_tp2)
        )

        if sl1_hit or sl2_hit:
            pos.hardware_sl_touched = True
        if tp1_hit or tp2_hit:
            pos.hardware_tp_touched = True

        any_sl = sl1_hit or sl2_hit
        any_tp = tp1_hit or tp2_hit
        if not any_sl and not any_tp:
            return None

        reason = "hardware_sl" if any_sl else "hardware_tp"
        note = "ambiguous_same_bar" if any_sl and any_tp else ""
        trigger_symbol = ""
        if reason == "hardware_sl":
            trigger_symbol = pos.symbol1 if sl1_hit else pos.symbol2
        else:
            trigger_symbol = pos.symbol1 if tp1_hit else pos.symbol2

        if not pos.hardware_first_touch_reason:
            pos.hardware_first_touch_reason = reason
            pos.hardware_first_touch_symbol = trigger_symbol
            pos.hardware_first_touch_note = note
            pos.hardware_first_touch_idx = int(idx)

        if self.hardware_sltp_mode != "exit":
            return None

        price1_base = close1
        price2_base = close2
        if reason == "hardware_sl":
            if sl1_hit and np.isfinite(pos.hardware_sl1):
                price1_base = float(pos.hardware_sl1)
            if sl2_hit and np.isfinite(pos.hardware_sl2):
                price2_base = float(pos.hardware_sl2)
        else:
            if tp1_hit and np.isfinite(pos.hardware_tp1):
                price1_base = float(pos.hardware_tp1)
            if tp2_hit and np.isfinite(pos.hardware_tp2):
                price2_base = float(pos.hardware_tp2)

        if not np.isfinite(price1_base) or price1_base <= 0:
            price1_base = close1
        if not np.isfinite(price2_base) or price2_base <= 0:
            price2_base = close2

        return {
            "reason": reason,
            "zscore": st.last_z_score,
            "beta": st.beta_btc,
            "pvalue": st.last_pvalue,
            "price1_base": float(price1_base),
            "price2_base": float(price2_base),
            "hardware_trigger_symbol": trigger_symbol,
            "hardware_note": note,
        }

    def _shutdown_pool(self) -> None:
        pool = self._pool
        if pool is None:
            return
        self._pool = None
        try:
            pool.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass

    def _prepare_snapshot_task(self, key: tuple[str, str], idx: int):
        """Prepare a small data slice for _compute_snapshot worker."""
        a, b = key
        ia = self.market.symbol_to_idx.get(a)
        ib = self.market.symbol_to_idx.get(b)
        if ia is None or ib is None:
            return None
        start = idx - self.min_data_points + 1
        if start < 0:
            return None
        c1 = self.market.close_arr[start:idx + 1, ia].copy()
        c2 = self.market.close_arr[start:idx + 1, ib].copy()
        cb = None
        if self.btc_idx is not None:
            cb = self.market.close_arr[start:idx + 1, self.btc_idx].copy()
        return (key, c1, c2, cb, float(self.params.p_value_threshold))

    def _batch_snapshots(self, keys: list[tuple[str, str]], idx: int) -> dict[tuple[str, str], PairSnapshot]:
        """Compute pair snapshots in parallel.
        Workers have close_arr pre-loaded; we only send tiny index tuples per call.
        """
        if not keys:
            return {}

        # Build lightweight tasks: (key, ia, ib, idx, min_dp, p_thresh)
        tasks: list[tuple] = []
        for key in keys:
            ia, ib = self._get_pair_indices(key)
            if ia is None or ib is None:
                continue
            tasks.append((key, ia, ib, idx, self._min_dp, self._p_thresh))

        if not tasks:
            return {}

        # Parallel or sequential
        raw_results: list[tuple] = []
        parallel_threshold = max(8, self.n_workers)
        if self._pool is not None and len(tasks) >= parallel_threshold:
            chk = max(1, len(tasks) // self.n_workers)
            raw_results = list(self._pool.map(_compute_snapshot_indexed, tasks, chunksize=chk))
        else:
            # Sequential path must NOT use worker globals (_W_CLOSE),
            # otherwise n_workers=1 returns empty snapshots.
            raw_results = [
                _compute_snapshot_with_arrays(
                    self.market.close_arr,
                    self.btc_idx,
                    t[0],  # key
                    t[1],  # ia
                    t[2],  # ib
                    t[3],  # idx
                    t[4],  # min_dp
                    t[5],  # p_thresh
                )
                for t in tasks
            ]

        # Convert to PairSnapshot dict
        results: dict[tuple[str, str], PairSnapshot] = {}
        for key, data in raw_results:
            if data is None:
                results[key] = PairSnapshot(valid=False)
            else:
                flag, hedge, hl, pval, z, beta = data
                results[key] = PairSnapshot(
                    valid=True, flag=flag, hedge=hedge,
                    half_life=hl, pvalue=pval, zscore=z, beta=beta,
                )
        return results

    def _fast_zscore_snapshot(self, key: tuple[str, str], idx: int, cached_hedge: float) -> float | None:
        """Ultra-fast z-score using cached hedge ratio — pure numpy, no statsmodels.
        Returns z-score float or None if data invalid.
        """
        ia, ib = self._get_pair_indices(key)
        if ia is None or ib is None:
            return None
        start = idx - self.min_data_points + 1
        if start < 0:
            return None
        c1 = self.market.close_arr[start:idx + 1, ia]
        c2 = self.market.close_arr[start:idx + 1, ib]
        if len(c1) < self.min_data_points:
            return None
        if not (np.isfinite(c1).all() and np.isfinite(c2).all()):
            return None
        if (c1 <= 0).any() or (c2 <= 0).any():
            return None
        spread = np.log(c1) - cached_hedge * np.log(c2)
        z = utils.calculate_z_last(spread)
        if z is None or not np.isfinite(z):
            return None
        return float(z)

    def _pair_snapshot(self, key: tuple[str, str], idx: int) -> PairSnapshot:
        """Single-pair snapshot (sequential fallback)."""
        ia, ib = self._get_pair_indices(key)
        if ia is None or ib is None:
            return PairSnapshot(valid=False)
        _, data = _compute_snapshot_with_arrays(
            self.market.close_arr,
            self.btc_idx,
            key,
            ia,
            ib,
            idx,
            self._min_dp,
            self._p_thresh,
        )
        if data is None:
            return PairSnapshot(valid=False)
        flag, hedge, hl, pval, z, beta = data
        return PairSnapshot(
            valid=True, flag=flag, hedge=hedge,
            half_life=hl, pvalue=pval, zscore=z, beta=beta,
        )

    @staticmethod
    def _clamp01(x: float) -> float:
        if x < 0:
            return 0.0
        if x > 1:
            return 1.0
        return float(x)

    def _update_quality_score(self, st: PairState, idx: int) -> None:
        pval = float(st.last_pvalue or 0.0)
        beta = abs(float(st.beta_btc or 0.0))
        hedge = abs(float(st.hedge_ratio or 0.0))
        hl = float(st.half_life or 0.0)

        p_quality = 1.0 - self._clamp01(pval / 0.05) if pval > 0 else 0.0
        beta_quality = 1.0 - self._clamp01(beta / 0.15)
        hedge_quality = 1.0 - self._clamp01(abs(hedge - 1.0) / 1.0)
        if hl <= 0:
            hl_quality = 0.0
        elif hl < 6:
            hl_quality = self._clamp01(hl / 6.0)
        elif hl > 72:
            hl_quality = 1.0 - self._clamp01((hl - 72.0) / 120.0)
        else:
            hl_quality = 1.0

        score = (
            0.38 * p_quality +
            0.32 * beta_quality +
            0.20 * hedge_quality +
            0.10 * hl_quality
        )
        score = max(0.0, score - 0.25 * self._clamp01(st.recent_fail_penalty))
        st.quality_score = float(score)
        st.quality_updated_idx = int(idx)

    @staticmethod
    def _update_quality_penalty_on_close(st: PairState, close_reason: str) -> None:
        reason = str(close_reason or "").strip().lower()
        penalty = float(st.recent_fail_penalty or 0.0)
        if reason in {"z_sl", "broken_coint", "beta_critical", "circuit"}:
            penalty = min(1.0, penalty + 0.35)
        elif reason in {"z_tp"}:
            penalty = max(0.0, penalty - 0.20)
        else:
            penalty = max(0.0, penalty - 0.05)
        st.recent_fail_penalty = penalty

    def _has_primary_signal(self) -> bool:
        for key in self.positions:
            if key in self.primary_candidate_set:
                return True
        for key in self.pending_entries:
            if key in self.primary_candidate_set:
                return True
        for key, st in self.pair_states.items():
            if key in self.primary_candidate_set and st.pending_signal is not None:
                return True
        return False

    def _allow_supplemental_discovery(self) -> bool:
        if not self.supplemental_candidate_keys:
            return False
        if not self.supplemental_when_no_primary_signal:
            return True
        return not self._has_primary_signal()

    def _prune_idle_supplemental_states(self) -> None:
        if not self.supplemental_candidate_set:
            return
        stale: list[tuple[str, str]] = []
        for key in list(self.pair_states.keys()):
            if key not in self.supplemental_candidate_set:
                continue
            if key in self.positions or key in self.pending_entries:
                continue
            stale.append(key)
        for key in stale:
            self.pair_states.pop(key, None)
            self.discovery_retry_until_idx.pop(key, None)

    def _can_open_pair(self, st: PairState, idx: int) -> bool:
        if st.key in self.positions:
            return False
        if len(self.positions) >= int(self.params.max_active_pairs):
            return False
        if st.reentry_block_idx >= 0 and idx <= st.reentry_block_idx:
            return False
        if st.close_cooldown_until_idx >= 0 and idx <= st.close_cooldown_until_idx:
            return False
        if st.symbol1 in self.symbol_owner or st.symbol2 in self.symbol_owner:
            return False
        if st.key in self.pending_entries:
            return False
        return True

    def _idle_eval_slot(self, key: tuple[str, str]) -> int:
        """Deterministic round-robin slot for idle pair evaluation throttling."""
        if self.idle_eval_every <= 1:
            return 0
        a, b = key
        ia = self.market.symbol_to_idx.get(a, 0)
        ib = self.market.symbol_to_idx.get(b, 0)
        return int((ia * 131 + ib * 17) % self.idle_eval_every)

    def _unrealized_for_position(self, pos: Position, idx: int, update_extremes: bool) -> float:
        i1 = self.market.symbol_to_idx[pos.symbol1]
        i2 = self.market.symbol_to_idx[pos.symbol2]
        p1 = float(self.market.close_arr[idx, i1])
        p2 = float(self.market.close_arr[idx, i2])
        pnl1 = (p1 - pos.entry_price1) * pos.qty1_signed
        pnl2 = (p2 - pos.entry_price2) * pos.qty2_signed
        total = float(pnl1 + pnl2)
        if update_extremes:
            pos.min_unrealized_pnl = min(pos.min_unrealized_pnl, total)
            pos.max_unrealized_pnl = max(pos.max_unrealized_pnl, total)
        return total

    def _schedule_exit(self, key: tuple[str, str], idx: int, reason: str, snapshot: PairSnapshot) -> None:
        if key not in self.positions:
            return
        if key in self.pending_exits:
            return
        exec_idx = idx + 1
        if exec_idx >= self.n_bars:
            return
        self.pending_exits[key] = {
            "exec_idx": exec_idx,
            "reason": reason,
            "zscore": snapshot.zscore,
            "beta": snapshot.beta,
            "pvalue": snapshot.pvalue,
        }

    def _schedule_entry(
        self,
        key: tuple[str, str],
        idx: int,
        direction: int,
        zscore: float,
        expected_hours: float,
    ) -> None:
        if key in self.pending_entries:
            return
        exec_idx = idx + 1
        if exec_idx >= self.n_bars:
            return
        self.pending_entries[key] = {
            "exec_idx": exec_idx,
            "direction": int(direction),
            "entry_z": float(zscore),
            "expected_hours": float(expected_hours),
        }

    def _execute_entry(self, key: tuple[str, str], order: dict[str, Any], idx: int) -> None:
        st = self.pair_states.get(key)
        if st is None:
            return
        if not self._can_open_pair(st, idx):
            return

        i1 = self.market.symbol_to_idx[st.symbol1]
        i2 = self.market.symbol_to_idx[st.symbol2]
        o1 = float(self.market.open_arr[idx, i1])
        o2 = float(self.market.open_arr[idx, i2])
        if not np.isfinite(o1) or not np.isfinite(o2) or o1 <= 0 or o2 <= 0:
            return

        direction = int(order["direction"])
        hedge = float(st.hedge_ratio or 1.0)
        dummy = np.array([1.0, 1.1], dtype=float)
        d1, d2 = utils.vol_parity_notional(
            dummy,
            dummy,
            hedge,
            capital=float(self.params.capital),
            max_notional_per_pair=float(self.params.max_notional_pct),
        )

        if d1 <= 0 or d2 <= 0:
            return

        slip = float(self.params.slippage_rate)
        # direction=1: long spread => buy s1 / sell s2
        if direction == 1:
            p1_exec = o1 * (1.0 + slip)
            p2_exec = o2 * (1.0 - slip)
            q1_signed = (d1 / p1_exec)
            q2_signed = -(d2 / p2_exec)
        else:
            p1_exec = o1 * (1.0 - slip)
            p2_exec = o2 * (1.0 + slip)
            q1_signed = -(d1 / p1_exec)
            q2_signed = (d2 / p2_exec)

        entry_notional = abs(q1_signed * p1_exec) + abs(q2_signed * p2_exec)
        entry_fee = entry_notional * float(self.params.commission_rate)
        if self.cash - entry_fee <= 0:
            self.ledger.append(
                {
                    "time": self.market.dates[idx],
                    "pair": f"{st.symbol1}-{st.symbol2}",
                    "type": "skip_entry",
                    "cash_change": 0.0,
                    "note": "no_cash_for_fees",
                }
            )
            return

        self.trade_seq += 1
        trade_id = self.trade_seq
        hardware_sl1 = np.nan
        hardware_tp1 = np.nan
        hardware_sl2 = np.nan
        hardware_tp2 = np.nan
        if self.hardware_sltp_active:
            atr1 = self._compute_atr_for_symbol(i1, idx)
            atr2 = self._compute_atr_for_symbol(i2, idx)
            leg1_side = "LONG" if direction == 1 else "SHORT"
            leg2_side = "SHORT" if direction == 1 else "LONG"
            hardware_sl1, hardware_tp1, _, _ = utils.calculate_hardware_stops(
                float(p1_exec), leg1_side, float(atr1), self.params
            )
            hardware_sl2, hardware_tp2, _, _ = utils.calculate_hardware_stops(
                float(p2_exec), leg2_side, float(atr2), self.params
            )
        pos = Position(
            trade_id=trade_id,
            pair_key=key,
            symbol1=st.symbol1,
            symbol2=st.symbol2,
            direction=direction,
            qty1_signed=float(q1_signed),
            qty2_signed=float(q2_signed),
            entry_price1=float(p1_exec),
            entry_price2=float(p2_exec),
            entry_idx=idx,
            entry_time=self.market.dates[idx],
            entry_fee=float(entry_fee),
            entry_notional=float(entry_notional),
            hedge_ratio=float(st.hedge_ratio),
            entry_z=float(order["entry_z"]),
            entry_beta=float(st.beta_btc),
            entry_pvalue=float(st.last_pvalue),
            entry_half_life=float(st.half_life),
            coint_streak_at_entry=int(st.coint_streak_bars),
            expected_reversion_hours=float(order["expected_hours"]),
            hardware_sl1=float(hardware_sl1) if np.isfinite(hardware_sl1) else np.nan,
            hardware_tp1=float(hardware_tp1) if np.isfinite(hardware_tp1) else np.nan,
            hardware_sl2=float(hardware_sl2) if np.isfinite(hardware_sl2) else np.nan,
            hardware_tp2=float(hardware_tp2) if np.isfinite(hardware_tp2) else np.nan,
        )
        self.positions[key] = pos
        self.symbol_owner[st.symbol1] = key
        self.symbol_owner[st.symbol2] = key
        st.position_status = direction

        self.cash -= entry_fee
        self.ledger.append(
            {
                "time": self.market.dates[idx],
                "pair": f"{st.symbol1}-{st.symbol2}",
                "type": "entry_fee",
                "cash_change": -float(entry_fee),
                "note": f"trade_id={trade_id}",
            }
        )

        row = {
            "trade_id": trade_id,
            "pair": f"{st.symbol1}-{st.symbol2}",
            "symbol1": st.symbol1,
            "symbol2": st.symbol2,
            "direction": direction,
            "entry_idx": idx,
            "entry_time": self.market.dates[idx],
            "entry_price1": float(p1_exec),
            "entry_price2": float(p2_exec),
            "qty1_signed": float(q1_signed),
            "qty2_signed": float(q2_signed),
            "entry_notional": float(entry_notional),
            "entry_fee": float(entry_fee),
            "entry_z": float(order["entry_z"]),
            "entry_beta": float(st.beta_btc),
            "entry_pvalue": float(st.last_pvalue),
            "entry_half_life": float(st.half_life),
            "entry_hedge_ratio": float(st.hedge_ratio),
            "entry_coint_streak_bars": int(st.coint_streak_bars),
            "expected_reversion_hours": float(order["expected_hours"]),
            "hardware_sltp_mode": self.hardware_sltp_mode,
            "hardware_sl1": float(pos.hardware_sl1) if np.isfinite(pos.hardware_sl1) else np.nan,
            "hardware_tp1": float(pos.hardware_tp1) if np.isfinite(pos.hardware_tp1) else np.nan,
            "hardware_sl2": float(pos.hardware_sl2) if np.isfinite(pos.hardware_sl2) else np.nan,
            "hardware_tp2": float(pos.hardware_tp2) if np.isfinite(pos.hardware_tp2) else np.nan,
            "hardware_sl_touched": False,
            "hardware_tp_touched": False,
            "hardware_first_touch_reason": None,
            "hardware_first_touch_symbol": None,
            "hardware_first_touch_note": None,
            "hardware_first_touch_idx": None,
            "hardware_first_touch_time": None,
            "exit_idx": None,
            "exit_time": None,
            "exit_price1": None,
            "exit_price2": None,
            "exit_reason": None,
            "exit_reason_text": None,
            "exit_fee": None,
            "gross_pnl": None,
            "funding_leg1": 0.0,
            "funding_leg2": 0.0,
            "funding_total": 0.0,
            "funding_events": 0,
            "net_pnl_before_funding": None,
            "net_pnl": None,
            "hold_bars": None,
            "hold_hours": None,
            "mae_usdt": None,
            "mfe_usdt": None,
            "mae_pct_notional": None,
            "mfe_pct_notional": None,
            "risk_per_trade_usdt": float(entry_notional * self.params.circuit_breaker_pct),
            "risk_per_trade_pct_notional": float(self.params.circuit_breaker_pct),
        }
        self.trade_row_idx[trade_id] = len(self.trades)
        self.trades.append(row)

    def _execute_exit(self, key: tuple[str, str], order: dict[str, Any], idx: int) -> None:
        pos = self.positions.get(key)
        st = self.pair_states.get(key)
        if pos is None or st is None:
            return

        if "price1_base" in order and "price2_base" in order:
            p1_base = float(order["price1_base"])
            p2_base = float(order["price2_base"])
        else:
            i1 = self.market.symbol_to_idx[pos.symbol1]
            i2 = self.market.symbol_to_idx[pos.symbol2]
            p1_base = float(self.market.open_arr[idx, i1])
            p2_base = float(self.market.open_arr[idx, i2])
        if not np.isfinite(p1_base) or not np.isfinite(p2_base) or p1_base <= 0 or p2_base <= 0:
            return

        slip = float(self.params.slippage_rate)
        # Close long qty by selling (price worse by -slip), close short qty by buying (+slip)
        p1_exec = p1_base * (1.0 - slip) if pos.qty1_signed > 0 else p1_base * (1.0 + slip)
        p2_exec = p2_base * (1.0 - slip) if pos.qty2_signed > 0 else p2_base * (1.0 + slip)

        pnl1 = (p1_exec - pos.entry_price1) * pos.qty1_signed
        pnl2 = (p2_exec - pos.entry_price2) * pos.qty2_signed
        gross = float(pnl1 + pnl2)
        exit_notional = abs(pos.qty1_signed * p1_exec) + abs(pos.qty2_signed * p2_exec)
        exit_fee = float(exit_notional * self.params.commission_rate)
        net_before_funding = float(gross - pos.entry_fee - exit_fee)
        net = float(net_before_funding + float(pos.funding_total))

        self.cash += gross - exit_fee
        reason = str(order["reason"])

        # Update MAE/MFE with exit-price PnL before recording
        exit_pnl = float(pnl1 + pnl2)
        pos.min_unrealized_pnl = min(pos.min_unrealized_pnl, exit_pnl)
        pos.max_unrealized_pnl = max(pos.max_unrealized_pnl, exit_pnl)

        hold_bars = int(idx - pos.entry_idx)
        hold_hours = float(hold_bars * self.timeframe_sec / 3600.0)
        mae = float(min(0.0, pos.min_unrealized_pnl))
        mfe = float(max(0.0, pos.max_unrealized_pnl))
        entry_notional = max(1e-12, float(pos.entry_notional))

        tr_idx = self.trade_row_idx.get(pos.trade_id)
        if tr_idx is not None and 0 <= tr_idx < len(self.trades):
            self.trades[tr_idx].update(
                {
                    "exit_idx": idx,
                    "exit_time": self.market.dates[idx],
                    "exit_price1": float(p1_exec),
                    "exit_price2": float(p2_exec),
                    "exit_reason": reason,
                    "exit_reason_text": CLOSE_REASONS.get(reason, reason),
                    "exit_fee": float(exit_fee),
                    "gross_pnl": float(gross),
                    "funding_leg1": float(pos.funding_leg1),
                    "funding_leg2": float(pos.funding_leg2),
                    "funding_total": float(pos.funding_total),
                    "funding_events": int(pos.funding_events),
                    "net_pnl_before_funding": float(net_before_funding),
                    "net_pnl": float(net),
                    "hold_bars": hold_bars,
                    "hold_hours": hold_hours,
                    "mae_usdt": mae,
                    "mfe_usdt": mfe,
                    "mae_pct_notional": float(mae / entry_notional),
                    "mfe_pct_notional": float(mfe / entry_notional),
                    "close_z": float(order.get("zscore", np.nan)),
                    "close_beta": float(order.get("beta", np.nan)),
                    "close_pvalue": float(order.get("pvalue", np.nan)),
                    "pnl_leg1": float(pnl1),
                    "pnl_leg2": float(pnl2),
                    "hardware_sl_touched": bool(pos.hardware_sl_touched),
                    "hardware_tp_touched": bool(pos.hardware_tp_touched),
                    "hardware_first_touch_reason": pos.hardware_first_touch_reason or None,
                    "hardware_first_touch_symbol": pos.hardware_first_touch_symbol or None,
                    "hardware_first_touch_note": pos.hardware_first_touch_note or None,
                    "hardware_first_touch_idx": (
                        int(pos.hardware_first_touch_idx)
                        if int(pos.hardware_first_touch_idx) >= 0
                        else None
                    ),
                    "hardware_first_touch_time": (
                        self.market.dates[int(pos.hardware_first_touch_idx)]
                        if int(pos.hardware_first_touch_idx) >= 0
                        else None
                    ),
                }
            )

        self.ledger.append(
            {
                "time": self.market.dates[idx],
                "pair": f"{pos.symbol1}-{pos.symbol2}",
                "type": f"exit_{reason}",
                "cash_change": float(gross - exit_fee),
                "note": f"trade_id={pos.trade_id}",
            }
        )

        self.positions.pop(key, None)
        self.symbol_owner.pop(pos.symbol1, None)
        self.symbol_owner.pop(pos.symbol2, None)

        st.position_status = 0
        st.pending_signal = None
        st.pending_since_idx = None
        st.pending_source = ""
        st.reentry_block_idx = idx
        st.close_cooldown_until_idx = idx + self.params.cooldown_bars(reason)
        self._update_quality_penalty_on_close(st, reason)
        self._update_quality_score(st, idx)

        if st.coint_run_len > 0 and reason in {"broken_coint"}:
            self.coint_phase_rows.append(
                {
                    "pair": f"{st.symbol1}-{st.symbol2}",
                    "duration_bars": int(st.coint_run_len),
                    "duration_hours": float(st.coint_run_len * self.timeframe_sec / 3600.0),
                    "ended_idx": idx,
                    "ended_time": self.market.dates[idx],
                }
            )
            st.coint_run_len = 0

        if reason == "broken_coint":
            self.pair_states.pop(key, None)

    def _force_close_remaining(self, idx: int) -> None:
        # Clean any stale pending exits first
        self.pending_exits.clear()
        keys = list(self.positions.keys())
        for key in keys:
            pos = self.positions.get(key)
            st = self.pair_states.get(key)
            if pos is None or st is None:
                continue
            i1 = self.market.symbol_to_idx[pos.symbol1]
            i2 = self.market.symbol_to_idx[pos.symbol2]
            c1 = float(self.market.close_arr[idx, i1])
            c2 = float(self.market.close_arr[idx, i2])
            if not np.isfinite(c1) or not np.isfinite(c2) or c1 <= 0 or c2 <= 0:
                continue
            order = {
                "reason": "force_close",
                "zscore": np.nan,
                "beta": np.nan,
                "pvalue": np.nan,
            }
            # Save originals, temporarily set close as open, then restore
            orig_o1 = self.market.open_arr[idx, i1]
            orig_o2 = self.market.open_arr[idx, i2]
            self.market.open_arr[idx, i1] = c1
            self.market.open_arr[idx, i2] = c2
            self._execute_exit(key, order, idx)
            self.market.open_arr[idx, i1] = orig_o1
            self.market.open_arr[idx, i2] = orig_o2

    def _discover_new_pairs(self, idx: int) -> None:
        if idx < self.min_data_points - 1:
            return
        if len(self.pair_states) >= int(self.params.max_idle_pairs):
            return
        if int(self.params.discovery_every_bars) > 1:
            if idx % int(self.params.discovery_every_bars) != 0:
                return

        allow_supplemental = self._allow_supplemental_discovery()
        if not allow_supplemental and self.supplemental_when_no_primary_signal:
            self._prune_idle_supplemental_states()

        scan_keys = self.primary_candidate_keys
        if allow_supplemental:
            scan_keys = self.candidate_keys

        shards = max(1, int(self.params.discovery_shards))
        shard_idx = int(self.discovery_round % shards)
        self.discovery_round += 1

        # Collect keys eligible for snapshot computation
        keys_to_check: list[tuple[str, str]] = []
        total_scanned = 0
        cap = max(1, int(self.params.discovery_max_pairs_per_cycle))
        for i, key in enumerate(scan_keys):
            total_scanned += 1
            if total_scanned > cap:
                break
            if shards > 1 and (i % shards) != shard_idx:
                continue
            if self.discovery_recheck_bars > 0:
                retry_until = int(self.discovery_retry_until_idx.get(key, -1))
                if retry_until >= 0 and idx < retry_until:
                    continue
            if key in self.pair_states:
                continue
            if key in self.positions:
                continue
            a, b = key
            if a not in self.market.symbol_to_idx or b not in self.market.symbol_to_idx:
                continue
            if a in self.symbol_owner or b in self.symbol_owner:
                continue
            keys_to_check.append(key)

        # Batch-compute all snapshots in parallel
        snapshots = self._batch_snapshots(keys_to_check, idx)

        added = 0
        for key in keys_to_check:
            snap = snapshots.get(key)
            if snap is None or not snap.valid or snap.flag != 1:
                if self.discovery_recheck_bars > 0:
                    self.discovery_retry_until_idx[key] = idx + self.discovery_recheck_bars
                continue
            if np.isnan(snap.hedge):
                if self.discovery_recheck_bars > 0:
                    self.discovery_retry_until_idx[key] = idx + self.discovery_recheck_bars
                continue
            abs_h = abs(float(snap.hedge))
            if abs_h < float(self.params.hedge_min) or abs_h > float(self.params.hedge_max):
                if self.discovery_recheck_bars > 0:
                    self.discovery_retry_until_idx[key] = idx + self.discovery_recheck_bars
                continue
            # Half-life range filter (parity with bot's discovery filter)
            hl = float(snap.half_life) if not np.isnan(snap.half_life) else 0.0
            cpd = utils.CANDLES_PER_DAY.get(self.params.timeframe, 24)
            hl_min_candles = float(self.params.hl_min_days) * cpd
            hl_max_candles = float(self.params.hl_max_days) * cpd
            if hl <= 0 or hl < hl_min_candles or hl > hl_max_candles:
                if self.discovery_recheck_bars > 0:
                    self.discovery_retry_until_idx[key] = idx + self.discovery_recheck_bars
                continue
            if not np.isnan(snap.beta) and abs(float(snap.beta)) >= float(self.params.beta_threshold):
                if self.discovery_recheck_bars > 0:
                    self.discovery_retry_until_idx[key] = idx + self.discovery_recheck_bars
                continue

            a, b = key
            st = PairState(symbol1=a, symbol2=b, discovered_idx=idx)
            st.hedge_ratio = float(snap.hedge)
            st.half_life = float(snap.half_life) if not np.isnan(snap.half_life) else 0.0
            st.beta_btc = float(snap.beta) if not np.isnan(snap.beta) else 0.0
            st.last_pvalue = float(snap.pvalue) if not np.isnan(snap.pvalue) else 0.0
            st.last_z_score = float(snap.zscore) if not np.isnan(snap.zscore) else 0.0
            st.coint_streak_bars = 1
            st.coint_run_len = 1
            self._update_quality_score(st, idx)
            self.pair_states[key] = st
            self.discovered_keys_seen.add(key)
            if self.discovery_recheck_bars > 0:
                self.discovery_retry_until_idx.pop(key, None)
            added += 1
            if len(self.pair_states) >= int(self.params.max_idle_pairs):
                break

    def _evaluate_pair_states(self, idx: int) -> None:
        ready_candidates: list[tuple[float, int, PairState, float, float]] = []
        keys = list(self.pair_states.keys())

        # Two tiers:
        # 1. full_keys  → full cointegration (active positions + periodic idle refresh)
        # 2. fast_keys  → fast z-score only using cached hedge (idle pairs between refreshes)
        full_keys: list[tuple[str, str]] = []
        fast_snapshots: dict[tuple[str, str], PairSnapshot] = {}

        for key in keys:
            st = self.pair_states.get(key)
            if st is None or st.last_eval_idx == idx:
                continue
            has_position = key in self.positions
            if has_position:
                # Active positions: ALWAYS full coint (needed for exit decisions)
                full_keys.append(key)
            elif st.pending_signal is not None:
                # Pending signal confirming: full coint (need pvalue/flag)
                full_keys.append(key)
            elif self.idle_eval_every > 1 and (idx % self.idle_eval_every) != self._idle_eval_slot(key):
                # Optional speed-up: evaluate only a deterministic idle subset each bar.
                # Active and pending pairs are never throttled.
                continue
            elif self.exact_mode:
                # Strict mode: full cointegration refresh on every eligible bar.
                full_keys.append(key)
            else:
                if st.hedge_ratio == 0.0:
                    full_keys.append(key)
                    continue

                z_fast = self._fast_zscore_snapshot(key, idx, float(st.hedge_ratio))
                if z_fast is None:
                    full_keys.append(key)
                    continue

                st.last_eval_idx = idx
                st.last_z_score = float(z_fast)

                due_full = (
                    st.coint_last_full_idx < 0
                    or (idx - st.coint_last_full_idx) >= self.coint_recompute_every
                )
                near_entry = abs(float(z_fast)) >= self.near_entry_z_trigger
                
                # If we are aggressively scanning (recompute > 1), don't let near_entry force
                # an OLS recompute on every single bar if the pair lingers near its trigger.
                if self.coint_recompute_every > 1:
                    needs_refresh = due_full
                else:
                    needs_refresh = due_full or near_entry

                if needs_refresh:
                    full_keys.append(key)
                    continue

                fast_snapshots[key] = PairSnapshot(
                    valid=True,
                    flag=1 if st.coint_streak_bars > 0 else 0,
                    hedge=float(st.hedge_ratio),
                    half_life=float(st.half_life),
                    pvalue=float(st.last_pvalue),
                    zscore=float(st.last_z_score),
                    beta=float(st.beta_btc),
                )

        # Run full cointegration snapshots in parallel
        snapshots = self._batch_snapshots(full_keys, idx)
        for key in full_keys:
            st = self.pair_states.get(key)
            if st is None:
                continue
            if key not in self.positions:
                st.coint_last_full_idx = idx  # mark refresh timestamp
            st.last_eval_idx = idx

        # Merge fast-path idle snapshots (already computed above).
        snapshots.update(fast_snapshots)

        for key in list(full_keys) + list(fast_snapshots.keys()):
            st = self.pair_states.get(key)
            if st is None:
                continue

            snap = snapshots.get(key)
            if snap is None or not snap.valid:
                continue

            if snap.flag == 1:
                st.coint_streak_bars = int(st.coint_streak_bars) + 1
                st.coint_run_len = int(st.coint_run_len) + 1
            else:
                if st.coint_run_len > 0:
                    self.coint_phase_rows.append(
                        {
                            "pair": f"{st.symbol1}-{st.symbol2}",
                            "duration_bars": int(st.coint_run_len),
                            "duration_hours": float(st.coint_run_len * self.timeframe_sec / 3600.0),
                            "ended_idx": idx,
                            "ended_time": self.market.dates[idx],
                        }
                    )
                st.coint_run_len = 0
                st.coint_streak_bars = 0

            st.last_z_score = float(snap.zscore) if not np.isnan(snap.zscore) else st.last_z_score
            st.last_pvalue = float(snap.pvalue) if not np.isnan(snap.pvalue) else st.last_pvalue
            st.beta_btc = float(snap.beta) if not np.isnan(snap.beta) else st.beta_btc
            st.hedge_ratio = float(snap.hedge) if not np.isnan(snap.hedge) else st.hedge_ratio
            if not np.isnan(snap.half_life):
                st.half_life = float(snap.half_life)
            self._update_quality_score(st, idx)

            if key in self.positions:
                pos = self.positions.get(key)
                if pos is not None:
                    hardware_order = self._check_hardware_sltp_for_bar(st, pos, idx)
                    if hardware_order is not None and key in self.positions:
                        self._execute_exit(key, hardware_order, idx)
                if key not in self.positions:
                    continue
                self._check_open_position_exit(st, snap, idx)
                continue

            if snap.flag != 1:
                st.pending_signal = None
                st.pending_since_idx = None
                st.pending_source = ""
                self.pair_states.pop(key, None)
                continue

            abs_h = abs(float(st.hedge_ratio or 0.0))
            if abs_h < float(self.params.hedge_min) or abs_h > float(self.params.hedge_max):
                st.pending_signal = None
                st.pending_since_idx = None
                st.pending_source = ""
                continue
            if not np.isnan(st.beta_btc) and abs(float(st.beta_btc)) >= float(self.params.beta_threshold):
                st.pending_signal = None
                st.pending_since_idx = None
                st.pending_source = ""
                continue

            z = float(st.last_z_score)
            if np.isnan(z):
                st.pending_signal = None
                st.pending_since_idx = None
                st.pending_source = ""
                continue

            if abs(z) >= float(self.params.z_entry) and abs(z) < float(self.params.z_entry_max):
                if st.pending_signal is None:
                    st.pending_signal = float(z)
                    st.pending_since_idx = idx
                    st.pending_source = "candle"
            elif abs(z) >= float(self.params.z_entry_max):
                st.pending_signal = None
                st.pending_since_idx = None
                st.pending_source = ""

            if st.pending_signal is None or st.pending_since_idx is None:
                continue
            if idx - int(st.pending_since_idx) < self.confirm_bars:
                continue

            pending = float(st.pending_signal)
            st.pending_signal = None
            st.pending_since_idx = None
            st.pending_source = ""

            cond_dir = (z * pending) > 0
            cond_window = abs(z) >= float(self.params.z_entry) and abs(z) < float(self.params.z_entry_max)
            
            # If we are aggressively recomputing (e.g. recompute_bars=6), coint_streak_bars 
            # won't tick up every single candle if it's cached. 
            # So if coint_recompute_every > 1, we bypass the strict streak check to prevent missing valid signals
            if self.coint_recompute_every > 1:
                cond_coint = True
            else:
                cond_coint = int(st.coint_streak_bars) >= int(self.params.coint_stability_min_bars)

            if not (cond_dir and cond_window and cond_coint):
                continue

            expected_bars = utils.expected_reversion_bars(
                abs_z_now=abs(float(z)),
                abs_z_target=float(self.params.entry_et_target_abs_z),
                half_life_bars=float(st.half_life or 0.0),
            )
            expected_hours = float(expected_bars * self.timeframe_sec / 3600.0)
            ready_candidates.append(
                (
                    float(st.quality_score),
                    int(st.quality_updated_idx),
                    st,
                    float(z),
                    expected_hours,
                )
            )

        if not ready_candidates:
            return

        ready_candidates.sort(key=lambda x: (-x[0], -x[1], -abs(x[3])))
        free_slots = max(
            0,
            int(self.params.max_active_pairs) - len(self.positions) - len(self.pending_entries),
        )
        if free_slots <= 0:
            return

        for _, _, st, z, expected_hours in ready_candidates:
            if free_slots <= 0:
                break
            if not self._can_open_pair(st, idx):
                continue
            direction = 1 if z < 0 else -1
            self._schedule_entry(st.key, idx, direction, z, expected_hours)
            free_slots -= 1

    def _check_open_position_exit(self, st: PairState, snap: PairSnapshot, idx: int) -> None:
        key = st.key
        pos = self.positions.get(key)
        if pos is None:
            return

        pnl_now = self._unrealized_for_position(pos, idx, update_extremes=True)
        notional = max(1e-12, float(pos.entry_notional))
        roi_notional = pnl_now / notional
        z = float(st.last_z_score)
        beta = float(st.beta_btc)

        reason = ""
        # Grace period for broken cointegration
        if snap.flag == 0:
            st.coint_broken_count += 1
            grace = int(self.params.coint_broken_grace_bars)
            if grace <= 0 or st.coint_broken_count > grace:
                reason = "broken_coint"
        else:
            st.coint_broken_count = 0  # coint restored, reset counter

        if not reason and roi_notional < -float(self.params.circuit_breaker_pct):
            reason = "circuit"
        if not reason and not np.isnan(beta) and abs(beta) >= float(self.params.beta_critical):
            reason = "beta_critical"
        if not reason:
            if pos.direction == 1:
                if z >= float(self.params.z_exit):
                    reason = "z_tp"
                elif z <= -float(self.params.z_stop):
                    reason = "z_sl"
            elif pos.direction == -1:
                if z <= -float(self.params.z_exit):
                    reason = "z_tp"
                elif z >= float(self.params.z_stop):
                    reason = "z_sl"

            if not reason and not np.isnan(beta):
                if abs(beta) >= float(self.params.beta_alert_threshold) and pnl_now > 0:
                    reason = "beta_drift"

            if not reason:
                hold_bars = idx - pos.entry_idx
                hl = float(pos.entry_half_life or 0.0)
                if hl > 0:
                    hl_limit = max(1, int(round(hl * float(self.params.hold_multiplier))))
                    hold_limit = min(hl_limit, self.max_hold_bars_days)
                else:
                    hold_limit = self.max_hold_bars_days
                if hold_bars >= hold_limit:
                    reason = "time_exit"

        if reason:
            self._schedule_exit(key, idx, reason, snap)

    def _execute_orders_for_bar(self, idx: int) -> None:
        # Exits first, then entries.
        exit_keys = [k for k, v in self.pending_exits.items() if int(v["exec_idx"]) == idx]
        for key in exit_keys:
            order = self.pending_exits.pop(key, None)
            if order is not None:
                self._execute_exit(key, order, idx)

        entry_keys = [k for k, v in self.pending_entries.items() if int(v["exec_idx"]) == idx]
        for key in entry_keys:
            order = self.pending_entries.pop(key, None)
            if order is not None:
                self._execute_entry(key, order, idx)

    def _mark_equity(self, idx: int, replace_last_same_ts: bool = False) -> None:
        unrealized = 0.0
        exposure_notional = 0.0
        for pos in self.positions.values():
            unrealized += self._unrealized_for_position(pos, idx, update_extremes=False)
            exposure_notional += float(pos.entry_notional)
        equity = float(self.cash + unrealized)
        exposure_pct = float(exposure_notional / equity) if equity > 0 else np.nan
        row = {
            "Date": self.market.dates[idx],
            "equity": equity,
            "cash": float(self.cash),
            "unrealized_pnl": float(unrealized),
            "open_positions": len(self.positions),
            "exposure_notional": float(exposure_notional),
            "exposure_pct": exposure_pct,
        }
        if (
            replace_last_same_ts
            and self.equity_rows
            and self.equity_rows[-1].get("Date") == row["Date"]
        ):
            self.equity_rows[-1] = row
        else:
            self.equity_rows.append(row)

    def run(self, start_idx: int | None = None, end_idx: int | None = None) -> BacktestResult:
        if self.n_bars <= self.min_data_points + 2:
            raise ValueError("Not enough bars for backtest with current window size.")

        i0 = self.min_data_points - 1 if start_idx is None else max(self.min_data_points - 1, int(start_idx))
        i1 = self.n_bars - 1 if end_idx is None else min(self.n_bars - 1, int(end_idx))
        if i0 >= i1:
            raise ValueError("Invalid backtest range.")

        total = i1 - i0 + 1
        progress_step = max(0, int(self.params.progress_every_bars or 0))
        t0 = time.perf_counter()
        if progress_step > 0:
            print(
                f"[BT] runtime workers={self.n_workers} "
                f"pool={'on' if self._pool is not None else 'off'} "
                f"exact_mode={self.exact_mode} "
                f"idle_eval_every={self.idle_eval_every} "
                f"coint_recompute_every={self.coint_recompute_every} "
                f"discovery_recheck_bars={self.discovery_recheck_bars}"
            )

        try:
            for idx in range(i0, i1 + 1):
                self._apply_funding_for_bar(idx)
                self._execute_orders_for_bar(idx)
                self._discover_new_pairs(idx)
                self._evaluate_pair_states(idx)
                self._mark_equity(idx)

                if progress_step > 0:
                    done = idx - i0 + 1
                    if done == 1 or done % progress_step == 0 or idx == i1:
                        elapsed = time.perf_counter() - t0
                        pct = (done / total) * 100.0 if total > 0 else 100.0
                        print(
                            f"[BT] {done}/{total} ({pct:.1f}%) "
                            f"elapsed={elapsed:.1f}s "
                            f"open={len(self.positions)} idle_pairs={len(self.pair_states)} trades={len(self.trades)}"
                        )

            if self.positions:
                self._force_close_remaining(i1)
                self._mark_equity(i1, replace_last_same_ts=True)

            return self._build_result()
        finally:
            self._shutdown_pool()

    def _build_result(self) -> BacktestResult:
        trades_df = pd.DataFrame(self.trades)
        eq_df = pd.DataFrame(self.equity_rows)
        if not eq_df.empty:
            eq_df = eq_df.sort_values("Date").reset_index(drop=True)
        coint_df = pd.DataFrame(self.coint_phase_rows)
        ledger_df = pd.DataFrame(self.ledger)
        metrics = self._build_metrics(trades_df, eq_df, coint_df)
        return BacktestResult(
            trades=trades_df,
            equity=eq_df,
            metrics=metrics,
            coint_phases=coint_df,
            ledger=ledger_df,
            params=dataclasses.asdict(self.params),
        )

    def _build_metrics(
        self,
        trades_df: pd.DataFrame,
        equity_df: pd.DataFrame,
        coint_df: pd.DataFrame,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if equity_df.empty:
            out["error"] = "empty_equity"
            return out

        eq = equity_df["equity"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(eq) < 2:
            out["error"] = "insufficient_equity_points"
            return out

        start_equity = float(eq.iloc[0])
        end_equity = float(eq.iloc[-1])
        total_return = (end_equity / start_equity - 1.0) if start_equity > 0 else np.nan

        sec_span = (equity_df["Date"].iloc[-1] - equity_df["Date"].iloc[0]).total_seconds()
        years = sec_span / (365.0 * 24.0 * 3600.0) if sec_span > 0 else np.nan
        cagr = ((end_equity / start_equity) ** (1.0 / years) - 1.0) if years and years > 0 and start_equity > 0 else np.nan

        roll_max = eq.cummax()
        drawdown = eq / roll_max - 1.0
        max_dd = float(drawdown.min()) if len(drawdown) else np.nan

        mean_ret = float(np.nanmean(returns)) if len(returns) else np.nan
        std_ret = float(np.nanstd(returns, ddof=1)) if len(returns) > 1 else np.nan
        downside = returns[returns < 0]
        std_down = float(np.nanstd(downside, ddof=1)) if len(downside) > 1 else np.nan

        sharpe = (
            (mean_ret / std_ret) * math.sqrt(self.periods_per_year)
            if std_ret and std_ret > 0
            else np.nan
        )
        sortino = (
            (mean_ret / std_down) * math.sqrt(self.periods_per_year)
            if std_down and std_down > 0
            else np.nan
        )
        calmar = (cagr / abs(max_dd)) if max_dd < 0 and not np.isnan(cagr) else np.nan

        closed = trades_df[trades_df["net_pnl"].notna()].copy() if "net_pnl" in trades_df.columns else pd.DataFrame()
        total_trades = int(len(closed))
        wins = int((closed["net_pnl"] > 0).sum()) if total_trades > 0 else 0
        losses = int((closed["net_pnl"] <= 0).sum()) if total_trades > 0 else 0
        win_rate = float(wins / total_trades) if total_trades > 0 else np.nan

        total_net = float(closed["net_pnl"].sum()) if total_trades > 0 else 0.0
        total_net_before_funding = float(closed["net_pnl_before_funding"].sum()) if total_trades > 0 and "net_pnl_before_funding" in closed.columns else total_net
        total_gross = float(closed["gross_pnl"].sum()) if total_trades > 0 else 0.0
        total_fees = float(closed["entry_fee"].sum() + closed["exit_fee"].sum()) if total_trades > 0 else 0.0
        total_funding = float(closed["funding_total"].sum()) if total_trades > 0 and "funding_total" in closed.columns else 0.0
        avg_trade = float(closed["net_pnl"].mean()) if total_trades > 0 else np.nan
        med_trade = float(closed["net_pnl"].median()) if total_trades > 0 else np.nan
        best_trade = float(closed["net_pnl"].max()) if total_trades > 0 else np.nan
        worst_trade = float(closed["net_pnl"].min()) if total_trades > 0 else np.nan
        avg_hold_bars = float(closed["hold_bars"].mean()) if total_trades > 0 else np.nan
        avg_hold_hours = float(closed["hold_hours"].mean()) if total_trades > 0 else np.nan

        sum_pos = float(closed.loc[closed["net_pnl"] > 0, "net_pnl"].sum()) if total_trades > 0 else 0.0
        sum_neg = float(closed.loc[closed["net_pnl"] < 0, "net_pnl"].sum()) if total_trades > 0 else 0.0
        profit_factor = (sum_pos / abs(sum_neg)) if sum_neg < 0 else np.nan

        avg_mae = float(closed["mae_usdt"].mean()) if total_trades > 0 else np.nan
        avg_mfe = float(closed["mfe_usdt"].mean()) if total_trades > 0 else np.nan
        avg_mae_pct = float(closed["mae_pct_notional"].mean()) if total_trades > 0 else np.nan
        avg_mfe_pct = float(closed["mfe_pct_notional"].mean()) if total_trades > 0 else np.nan
        avg_risk_trade = float(closed["risk_per_trade_usdt"].mean()) if total_trades > 0 else np.nan

        close_reason_breakdown = {}
        if total_trades > 0 and "exit_reason" in closed.columns:
            cnt = Counter(closed["exit_reason"].astype(str).tolist())
            close_reason_breakdown = dict(sorted(cnt.items(), key=lambda x: (-x[1], x[0])))

        hardware_touch_trades = 0
        hardware_sl_touch_trades = 0
        hardware_tp_touch_trades = 0
        hardware_first_touch_breakdown: dict[str, int] = {}
        if total_trades > 0 and "hardware_sl_touched" in closed.columns:
            sl_touched = closed["hardware_sl_touched"].fillna(False).astype(bool)
            tp_touched = closed["hardware_tp_touched"].fillna(False).astype(bool) if "hardware_tp_touched" in closed.columns else pd.Series(False, index=closed.index)
            hardware_sl_touch_trades = int(sl_touched.sum())
            hardware_tp_touch_trades = int(tp_touched.sum())
            hardware_touch_trades = int((sl_touched | tp_touched).sum())
            if "hardware_first_touch_reason" in closed.columns:
                first_touch = closed["hardware_first_touch_reason"].dropna().astype(str)
                if not first_touch.empty:
                    hardware_first_touch_breakdown = dict(
                        sorted(Counter(first_touch.tolist()).items(), key=lambda x: (-x[1], x[0]))
                    )

        if coint_df is not None and not coint_df.empty:
            avg_coint_bars = float(coint_df["duration_bars"].mean())
            avg_coint_hours = float(coint_df["duration_hours"].mean())
            median_coint_hours = float(coint_df["duration_hours"].median())
        else:
            avg_coint_bars = np.nan
            avg_coint_hours = np.nan
            median_coint_hours = np.nan

        avg_exposure_pct = float(equity_df["exposure_pct"].replace([np.inf, -np.inf], np.nan).dropna().mean())
        unique_pairs_traded = int(closed["pair"].nunique()) if total_trades > 0 and "pair" in closed.columns else 0

        out.update(
            {
                "timeframe": self.params.timeframe,
                "window_size": int(self.min_data_points),
                "start_time": str(equity_df["Date"].iloc[0]),
                "end_time": str(equity_df["Date"].iloc[-1]),
                "bars": int(len(equity_df)),
                "start_equity": start_equity,
                "final_equity": end_equity,
                "total_return_pct": float(total_return * 100.0) if not np.isnan(total_return) else np.nan,
                "cagr": float(cagr) if not np.isnan(cagr) else np.nan,
                "max_drawdown": max_dd,
                "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
                "sortino": float(sortino) if not np.isnan(sortino) else np.nan,
                "calmar": float(calmar) if not np.isnan(calmar) else np.nan,
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "total_net_pnl": total_net,
                "total_net_pnl_before_funding": total_net_before_funding,
                "total_gross_pnl": total_gross,
                "total_fees": total_fees,
                "total_funding": total_funding,
                "avg_trade_pnl": avg_trade,
                "median_trade_pnl": med_trade,
                "best_trade_pnl": best_trade,
                "worst_trade_pnl": worst_trade,
                "profit_factor": profit_factor,
                "avg_trade_hold_bars": avg_hold_bars,
                "avg_trade_hold_hours": avg_hold_hours,
                "avg_cointegration_duration_bars": avg_coint_bars,
                "avg_cointegration_duration_hours": avg_coint_hours,
                "median_cointegration_duration_hours": median_coint_hours,
                "avg_risk_per_trade_usdt": avg_risk_trade,
                "avg_mae_usdt": avg_mae,
                "avg_mfe_usdt": avg_mfe,
                "avg_mae_pct_notional": avg_mae_pct,
                "avg_mfe_pct_notional": avg_mfe_pct,
                "avg_exposure_pct": avg_exposure_pct,
                "close_reason_breakdown": close_reason_breakdown,
                "hardware_sltp_mode": self.hardware_sltp_mode,
                "hardware_sl_enabled": bool(self.hardware_sl_enabled),
                "hardware_tp_enabled": bool(self.hardware_tp_enabled),
                "hardware_touch_trades": hardware_touch_trades,
                "hardware_sl_touch_trades": hardware_sl_touch_trades,
                "hardware_tp_touch_trades": hardware_tp_touch_trades,
                "hardware_first_touch_breakdown": hardware_first_touch_breakdown,
                "active_pairs_final": int(len(self.pair_states)),
                "candidate_pairs_total": int(len(self.candidates)),
                "candidate_pairs_primary": int(len(self.primary_candidate_keys)),
                "candidate_pairs_supplemental": int(len(self.supplemental_candidate_keys)),
                "discovered_pairs_total": int(len(self.discovered_keys_seen)),
                "unique_pairs_traded": unique_pairs_traded,
                "universe_symbols_total": int(len(self.market.symbols)),
                "funding_enabled": bool(self.funding_enabled),
                "funding_records_loaded": int(self.funding_data.loaded_records) if self.funding_data is not None else 0,
                "funding_records_matched": int(self.funding_data.matched_records) if self.funding_data is not None else 0,
                "funding_records_skipped": int(self.funding_data.skipped_records) if self.funding_data is not None else 0,
                "funding_cash_total": float(self.total_funding_cash),
                "funding_source_path": self.funding_data.source_path if self.funding_data is not None else "",
            }
        )

        # ── Per-pair breakdown ────────────────────────────────────────────────
        if total_trades > 0 and "pair" in closed.columns:
            pair_rows = []
            for pair_name, grp in closed.groupby("pair"):
                n = len(grp)
                w = int((grp["net_pnl"] > 0).sum())
                l = n - w
                wr = w / n if n > 0 else np.nan
                net = float(grp["net_pnl"].sum())
                net_before_funding = float(grp["net_pnl_before_funding"].sum()) if "net_pnl_before_funding" in grp.columns else net
                gross = float(grp["gross_pnl"].sum())
                fees = float(grp["entry_fee"].sum() + grp["exit_fee"].sum())
                funding = float(grp["funding_total"].sum()) if "funding_total" in grp.columns else 0.0
                avg_pnl = float(grp["net_pnl"].mean())
                std_pnl = float(grp["net_pnl"].std(ddof=1)) if n > 1 else np.nan
                pair_sharpe = (avg_pnl / std_pnl) if std_pnl and std_pnl > 0 else np.nan
                sum_w = float(grp.loc[grp["net_pnl"] > 0, "net_pnl"].sum())
                sum_l = float(grp.loc[grp["net_pnl"] < 0, "net_pnl"].sum())
                pf = (sum_w / abs(sum_l)) if sum_l < 0 else np.nan
                avg_hold_h = float(grp["hold_hours"].mean()) if "hold_hours" in grp.columns else np.nan
                avg_entry_z = float(grp["entry_z"].mean()) if "entry_z" in grp.columns else np.nan
                avg_hedge = float(grp["entry_hedge_ratio"].mean()) if "entry_hedge_ratio" in grp.columns else np.nan
                avg_hl = float(grp["entry_half_life"].mean()) if "entry_half_life" in grp.columns else np.nan
                hw_touch = int(
                    (
                        grp.get("hardware_sl_touched", pd.Series(False, index=grp.index)).fillna(False).astype(bool)
                        | grp.get("hardware_tp_touched", pd.Series(False, index=grp.index)).fillna(False).astype(bool)
                    ).sum()
                )
                reasons = dict(Counter(grp["exit_reason"].astype(str).tolist())) if "exit_reason" in grp.columns else {}
                pair_rows.append({
                    "pair": pair_name,
                    "trades": n, "wins": w, "losses": l, "win_rate": wr,
                    "net_pnl": net, "net_pnl_before_funding": net_before_funding,
                    "gross_pnl": gross, "total_fees": fees, "total_funding": funding,
                    "avg_pnl": avg_pnl, "std_pnl": std_pnl,
                    "pair_sharpe": pair_sharpe, "profit_factor": pf,
                    "avg_hold_hours": avg_hold_h,
                    "avg_entry_z": avg_entry_z, "avg_hedge_ratio": avg_hedge,
                    "avg_half_life": avg_hl,
                    "hardware_touch_trades": hw_touch,
                    "close_reasons": str(reasons),
                })
            pair_rows.sort(key=lambda x: -x["net_pnl"])
            out["per_pair_stats"] = pair_rows
        else:
            out["per_pair_stats"] = []

        return out


def metrics_to_objective(metrics: dict[str, Any]) -> float:
    sharpe = float(metrics.get("sharpe", np.nan))
    ret = float(metrics.get("total_return_pct", np.nan))
    mdd = abs(float(metrics.get("max_drawdown", np.nan)))
    trades = int(metrics.get("total_trades", 0) or 0)

    score = 0.0
    if not np.isnan(sharpe):
        score += sharpe
    if not np.isnan(ret):
        score += ret / 100.0
    if not np.isnan(mdd):
        score -= 1.5 * mdd
    score += min(trades, 40) / 80.0
    if trades < 5:
        score -= 1.0
    return float(score)


# ── Hyperopt worker globals (loaded once per worker process via initializer) ──
_HO_MARKET_DICT: dict | None = None
_HO_CANDIDATE_DICTS: list | None = None


def _hyperopt_worker_init(market_dict: dict, candidate_dicts: list) -> None:
    """Called once per worker process — stores market data in process-local globals.
    This avoids pickling large numpy arrays on every trial submission.
    """
    global _HO_MARKET_DICT, _HO_CANDIDATE_DICTS
    _HO_MARKET_DICT = market_dict
    _HO_CANDIDATE_DICTS = candidate_dicts


def _run_hyperopt_trial_worker(args: tuple) -> dict[str, Any]:
    """Worker for hyperopt — receives only small params dict, uses global market data."""
    params_dict, end_idx = args
    market_dict = _HO_MARKET_DICT
    candidate_dicts = _HO_CANDIDATE_DICTS
    if market_dict is None or candidate_dicts is None:
        return {"_error": "Worker not initialized"}
    try:
        market = MarketData(
            dates=market_dict["dates"],
            symbols=market_dict["symbols"],
            symbol_to_idx=market_dict["symbol_to_idx"],
            open_arr=market_dict["open_arr"],
            high_arr=market_dict["high_arr"],
            low_arr=market_dict["low_arr"],
            close_arr=market_dict["close_arr"],
            volume_arr=market_dict["volume_arr"],
        )
        candidates = [CandidatePair(**cd) for cd in candidate_dicts]
        trial_params = BacktestParams(**params_dict)
        bt = BotParityBacktester(
            market=market, params=trial_params, candidates=candidates, n_workers=1
        )
        res = bt.run(end_idx=end_idx)
        return dict(res.metrics)
    except Exception as e:
        return {"_error": str(e)}


def run_hyperopt(
    market: MarketData,
    candidates: list[CandidatePair],
    base_params: BacktestParams,
    trials: int,
    seed: int,
    end_idx: int | None = None,
    max_workers: int = 0,
    n_startup: int = 3,
    quick_bars: int = 0,
) -> tuple[BacktestParams, list[dict[str, Any]]]:
    """Bayesian hyperopt using optuna TPE — sequential or parallel.

    If max_workers <= 1: sequential (stable on all OS).
    If max_workers >= 2: parallel batches via ProcessPoolExecutor.
      - Market data loaded ONCE per worker via initializer (not per trial).
      - Each batch = n_workers trials, all run simultaneously.
      - After each batch optuna TPE model updates before next batch.

    Example with workers=2, trials=10, n_startup=3:
      Batch 1: trials 1-2  (random exploration)
      Batch 2: trials 3-4  (1 random + 1 Bayesian)
      Batch 3: trials 5-6  (Bayesian)
      Batch 4: trials 7-8  (Bayesian)
      Batch 5: trials 9-10 (Bayesian)
      Total: ~5 batches × trial_time → ~2.5× faster than sequential
    """
    import optuna
    import concurrent.futures

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_workers = max_workers if max_workers > 0 else 1
    batch = n_workers  # batch size = workers count (saturate all cores)

    # Slice market to quick_bars for faster trials (optional)
    if quick_bars > 0 and quick_bars < len(market.dates):
        q_start = max(0, len(market.dates) - quick_bars)
        trial_market = MarketData(
            dates=market.dates[q_start:],
            symbols=market.symbols,
            symbol_to_idx=market.symbol_to_idx,
            open_arr=market.open_arr[q_start:, :].copy(),
            high_arr=market.high_arr[q_start:, :].copy(),
            low_arr=market.low_arr[q_start:, :].copy(),
            close_arr=market.close_arr[q_start:, :].copy(),
            volume_arr=market.volume_arr[q_start:, :].copy(),
        )
        trial_end_idx: int | None = None
        if end_idx is not None:
            mapped = end_idx - q_start
            if mapped >= int(base_params.resolved_window_size()):
                trial_end_idx = mapped
        bars_label = f"{len(trial_market.dates)} bars (quick)"
    else:
        trial_market = market
        trial_end_idx = end_idx
        bars_label = f"{len(market.dates)} bars"

    mode = f"{n_workers} workers, batch={batch}" if n_workers > 1 else "sequential"
    print(
        f"  [hyperopt] Optuna TPE [{mode}]: {trials} trials on {bars_label}, "
        f"first {n_startup} random → then Bayesian"
    )

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    history: list[dict[str, Any]] = []

    def _suggest(trial: optuna.Trial) -> BacktestParams:
        p = dataclasses.replace(base_params)
        # Z-score thresholds — step=0.1 (12 choices for z_entry)
        p.z_entry     = trial.suggest_float("z_entry",     1.4, 2.5,  step=0.1)
        p.z_entry_max = trial.suggest_float("z_entry_max", p.z_entry + 0.2, 3.5, step=0.1)
        p.z_exit      = trial.suggest_float("z_exit",      0.0, 0.35, step=0.05)   # 8 choices
        p.z_stop      = trial.suggest_float("z_stop",      2.5, 5.0,  step=0.5)    # 6 choices
        # Position sizing — step=0.05
        p.max_notional_pct    = trial.suggest_float("max_notional_pct",    0.10, 0.55, step=0.05)
        p.circuit_breaker_pct = trial.suggest_float("circuit_breaker_pct", 0.15, 0.70, step=0.05)
        # Beta thresholds — step=0.05
        p.beta_threshold      = trial.suggest_float("beta_threshold",      0.05, 0.25, step=0.05)
        p.beta_alert_threshold = trial.suggest_float(
            "beta_alert_threshold", p.beta_threshold + 0.05, 0.50, step=0.05
        )
        p.beta_critical = trial.suggest_float("beta_critical", 0.5, 1.5, step=0.25)  # 5 choices
        # Cointegration quality — step=0.01 (fine-grained p-value)
        p.p_value_threshold     = trial.suggest_float("p_value_threshold", 0.01, 0.08, step=0.01)
        p.coint_stability_min_bars = trial.suggest_int("coint_stability_min_bars", 1, 8)
        # Portfolio capacity
        p.max_active_pairs = trial.suggest_int("max_active_pairs", 2, 8)
        # Hold duration — step=0.5 for multiplier, step=1.0 for days
        p.hold_multiplier = trial.suggest_float("hold_multiplier", 1.5, 6.0, step=0.5)
        p.max_hold_days   = trial.suggest_float("max_hold_days",   5.0, 45.0, step=1.0)
        # Hedge ratio bounds — step=0.05 / 0.5
        p.hedge_min = trial.suggest_float("hedge_min", 0.10, 0.60, step=0.05)
        p.hedge_max = trial.suggest_float("hedge_max", 1.5,  5.0,  step=0.5)
        # Half-life constraints — step=0.1 / 0.5
        p.hl_min_days = trial.suggest_float("hl_min_days", 0.1, 1.0, step=0.1)
        p.hl_max_days = trial.suggest_float("hl_max_days", p.hl_min_days + 0.5, 5.0, step=0.5)
        return p

    def _log_row(idx: int, p: BacktestParams, metrics: dict, score: float) -> dict:
        row = {
            "trial": idx, "score": score,
            "sharpe": metrics.get("sharpe", np.nan),
            "return_pct": metrics.get("total_return_pct", np.nan),
            "max_drawdown": metrics.get("max_drawdown", np.nan),
            "trades": metrics.get("total_trades", 0),
            "win_rate": metrics.get("win_rate", np.nan),
            "profit_factor": metrics.get("profit_factor", np.nan),
            "z_entry": p.z_entry, "z_entry_max": p.z_entry_max,
            "z_exit": p.z_exit, "z_stop": p.z_stop,
            "max_notional_pct": p.max_notional_pct,
            "circuit_breaker_pct": p.circuit_breaker_pct,
            "beta_threshold": p.beta_threshold,
            "coint_stability_min_bars": p.coint_stability_min_bars,
            "max_active_pairs": p.max_active_pairs,
            "p_value_threshold": p.p_value_threshold,
            "hold_multiplier": p.hold_multiplier,
            "hedge_min": p.hedge_min, "hedge_max": p.hedge_max,
            "max_hold_days": p.max_hold_days,
            "hl_min_days": p.hl_min_days, "hl_max_days": p.hl_max_days,
        }
        print(
            f"  [hyperopt] {idx}/{trials}  score={score:.4f}  "
            f"sharpe={metrics.get('sharpe', 0):.3f}  "
            f"ret={metrics.get('total_return_pct', 0):.2f}%  "
            f"trades={metrics.get('total_trades', 0)}"
        )
        return row

    # ── Sequential mode (workers <= 1) ───────────────────────────────────────
    if n_workers <= 1:
        def objective(trial: optuna.Trial) -> float:
            p = _suggest(trial)
            idx = trial.number + 1
            try:
                bt = BotParityBacktester(
                    market=trial_market, params=p, candidates=candidates, n_workers=1
                )
                res = bt.run(end_idx=trial_end_idx)
                score = metrics_to_objective(res.metrics)
                history.append(_log_row(idx, p, res.metrics, score))
                return score
            except Exception as e:
                history.append({"trial": idx, "score": -1e18, "error": str(e)})
                print(f"  [hyperopt] {idx}/{trials}  ERROR: {e}")
                return -1e18

        study.optimize(objective, n_trials=trials, show_progress_bar=False)

    # ── Parallel batch mode (workers >= 2) ────────────────────────────────────
    else:
        market_dict = {
            "dates": trial_market.dates, "symbols": trial_market.symbols,
            "symbol_to_idx": trial_market.symbol_to_idx,
            "open_arr": trial_market.open_arr, "high_arr": trial_market.high_arr,
            "low_arr": trial_market.low_arr, "close_arr": trial_market.close_arr,
            "volume_arr": trial_market.volume_arr,
        }
        cand_dicts = [dataclasses.asdict(cp) for cp in candidates]

        completed = 0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_hyperopt_worker_init,
            initargs=(market_dict, cand_dicts),
        ) as pool:
            while completed < trials:
                this_batch = min(batch, trials - completed)

                # Ask optuna for this batch
                batch_ot: list[optuna.Trial] = []
                batch_params: list[BacktestParams] = []
                for _ in range(this_batch):
                    ot = study.ask()
                    batch_ot.append(ot)
                    batch_params.append(_suggest(ot))

                # Run in parallel (only tiny params dicts sent, market in worker globals)
                futs = {
                    pool.submit(_run_hyperopt_trial_worker, (dataclasses.asdict(p), trial_end_idx)): i
                    for i, p in enumerate(batch_params)
                }
                batch_metrics: list[dict] = [{}] * this_batch
                for fut, i in futs.items():
                    try:
                        batch_metrics[i] = fut.result()
                    except Exception as e:
                        batch_metrics[i] = {"_error": str(e)}

                # Report back → TPE updates model
                for i, (ot, p, metrics) in enumerate(zip(batch_ot, batch_params, batch_metrics)):
                    completed += 1
                    if "_error" in metrics:
                        study.tell(ot, state=optuna.trial.TrialState.FAIL)
                        history.append({"trial": completed, "score": -1e18, "error": metrics["_error"]})
                        print(f"  [hyperopt] {completed}/{trials}  ERROR: {metrics['_error']}")
                        continue
                    score = metrics_to_objective(metrics)
                    study.tell(ot, score)
                    history.append(_log_row(completed, p, metrics, score))

    # ── Best params ───────────────────────────────────────────────────────────
    best_row = max(history, key=lambda r: r.get("score", -1e18))
    print(
        f"\n  [hyperopt] ✓ Best trial #{best_row['trial']}  score={best_row['score']:.4f}  "
        f"sharpe={best_row.get('sharpe', 0):.3f}  ret={best_row.get('return_pct', 0):.2f}%  "
        f"trades={best_row.get('trades', 0)}"
    )
    best_params = dataclasses.replace(base_params)
    for field in ["z_entry", "z_entry_max", "z_exit", "z_stop", "max_notional_pct",
                  "circuit_breaker_pct", "beta_threshold", "coint_stability_min_bars",
                  "max_active_pairs", "p_value_threshold", "hold_multiplier",
                  "hedge_min", "hedge_max", "max_hold_days", "hl_min_days", "hl_max_days"]:
        if field in best_row:
            setattr(best_params, field, best_row[field])
    best_params.beta_alert_threshold = min(0.50, best_params.beta_threshold + 0.19)

    history.sort(key=lambda x: x.get("trial", 0))
    return best_params, history


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    All defaults are None so that DEFAULT_PARAMS takes precedence when a flag
    is not explicitly passed on the command line.  Only explicitly supplied
    flags override DEFAULT_PARAMS values in main().
    """
    parser = argparse.ArgumentParser(
        description="Market-neutral bot-parity backtest (dynamic window + cointegration + risk exits)."
    )
    # ── Data paths ────────────────────────────────────────────────────────────
    parser.add_argument("--input", default=None, help="Path to klines CSV (Date,Open,High,Low,Close,Volume,Symbol).")
    parser.add_argument("--best-pairs", default=None, help="Path to best_pairs.json.")
    parser.add_argument("--pair-blacklist", default=None, help="Path to pair blacklist (.json/.csv/.txt).")
    parser.add_argument("--funding-csv", default=None, help="Optional funding rates CSV from download_funding_rates.py.")
    parser.add_argument(
        "--disable-pair-blacklist",
        action="store_true",
        help="Disable the default pair blacklist unless an explicit --pair-blacklist path is passed.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for output files.")
    parser.add_argument("--report-start", default=None, help="Optional UTC report start, e.g. 2026-03-09 or 2026-03-09T00:00:00Z.")
    parser.add_argument("--report-end", default=None, help="Optional UTC report end (exclusive).")
    # ── Timeframe ─────────────────────────────────────────────────────────────
    parser.add_argument("--timeframe", default=None, help='Force timeframe label, e.g. "1h", "4h", "30m".')
    parser.add_argument("--window-size", type=int, default=None, help="Rolling window; 0=auto.")
    # ── Capital & sizing ──────────────────────────────────────────────────────
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--leverage", type=int, default=None)
    parser.add_argument("--max-notional-pct", type=float, default=None)
    # ── Costs ─────────────────────────────────────────────────────────────────
    parser.add_argument("--commission", type=float, default=None)
    parser.add_argument("--slippage", type=float, default=None)
    parser.add_argument("--hardware-sltp-mode", default=None, help="off/monitor/exit: disabled, count touches, or simulate exits.")
    parser.add_argument("--hardware-sl-enabled", default=None, help="true/false: include hardware SL levels in check.")
    parser.add_argument("--hardware-tp-enabled", default=None, help="true/false: include hardware TP levels in check.")
    parser.add_argument("--sl-atr-mult", type=float, default=None)
    parser.add_argument("--sl-min-pct", type=float, default=None)
    parser.add_argument("--sl-max-pct", type=float, default=None)
    parser.add_argument("--tp-atr-mult", type=float, default=None)
    parser.add_argument("--tp-min-pct", type=float, default=None)
    parser.add_argument("--tp-max-pct", type=float, default=None)
    # ── Z-score thresholds ────────────────────────────────────────────────────
    parser.add_argument("--z-entry", type=float, default=None)
    parser.add_argument("--z-entry-max", type=float, default=None)
    parser.add_argument("--z-exit", type=float, default=None)
    parser.add_argument("--z-stop", type=float, default=None)
    # ── Cointegration & beta ──────────────────────────────────────────────────
    parser.add_argument("--p-value-threshold", type=float, default=None)
    parser.add_argument("--beta-threshold", type=float, default=None)
    parser.add_argument("--beta-alert-threshold", type=float, default=None)
    parser.add_argument("--beta-critical", type=float, default=None)
    # ── Risk ──────────────────────────────────────────────────────────────────
    parser.add_argument("--circuit-breaker-pct", type=float, default=None)
    parser.add_argument("--hedge-min", type=float, default=None)
    parser.add_argument("--hedge-max", type=float, default=None)
    parser.add_argument("--hl-min-days", type=float, default=None)
    parser.add_argument("--hl-max-days", type=float, default=None)
    # ── Capacity ──────────────────────────────────────────────────────────────
    parser.add_argument("--max-active-pairs", type=int, default=None)
    parser.add_argument("--max-idle-pairs", type=int, default=None)
    # ── Entry confirmation & hold ─────────────────────────────────────────────
    parser.add_argument("--coint-stability-min-bars", type=int, default=None)
    parser.add_argument("--coint-broken-grace-bars", type=int, default=None)
    parser.add_argument("--signal-confirm-sec", type=int, default=None)
    parser.add_argument("--entry-et-target-abs-z", type=float, default=None)
    parser.add_argument("--hold-multiplier", type=float, default=None)
    parser.add_argument("--max-hold-days", type=float, default=None)
    # ── Cooldowns ─────────────────────────────────────────────────────────────
    parser.add_argument("--sl-reentry-cooldown-sec", type=int, default=None)
    parser.add_argument("--close-retry-cooldown-sec", type=int, default=None)
    # ── Discovery ─────────────────────────────────────────────────────────────
    parser.add_argument("--discovery-every-bars", type=int, default=None)
    parser.add_argument("--discovery-shards", type=int, default=None)
    parser.add_argument("--discovery-max-pairs-per-cycle", type=int, default=None)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--top-pairs-limit", type=int, default=None)
    parser.add_argument("--supplemental-pairs-limit", type=int, default=None)
    parser.add_argument("--supplemental-symbols", type=int, default=None)
    parser.add_argument(
        "--supplemental-when-no-primary-signal",
        default=None,
        help="true/false: scan supplemental pairs only when no primary signal is active.",
    )
    # ── Runtime ───────────────────────────────────────────────────────────────
    parser.add_argument("--progress-every-bars", type=int, default=None)
    parser.add_argument("--quick-bars", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--exact-mode", default=None, help="true/false: strict accurate mode.")
    parser.add_argument("--idle-eval-every", type=int, default=None)
    parser.add_argument("--coint-recompute-every", type=int, default=None)
    parser.add_argument("--discovery-recheck-bars", type=int, default=None)
    # ── Hyperopt ──────────────────────────────────────────────────────────────
    parser.add_argument("--hyperopt-trials", type=int, default=None)
    parser.add_argument("--hyperopt-train-frac", type=float, default=None)
    parser.add_argument("--hyperopt-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()



def save_result(result: BacktestResult, out_dir: str, prefix: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trades_path = out / f"{prefix}_trades.csv"
    equity_path = out / f"{prefix}_equity.csv"
    metrics_path = out / f"{prefix}_metrics.json"
    coint_path = out / f"{prefix}_cointegration_phases.csv"
    ledger_path = out / f"{prefix}_ledger.csv"
    params_path = out / f"{prefix}_params.json"
    pair_stats_path = out / f"{prefix}_per_pair_stats.csv"

    result.trades.to_csv(trades_path, index=False)
    result.equity.to_csv(equity_path, index=False)
    result.coint_phases.to_csv(coint_path, index=False)
    result.ledger.to_csv(ledger_path, index=False)

    # Per-pair stats CSV
    per_pair = result.metrics.get("per_pair_stats", [])
    if per_pair:
        pd.DataFrame(per_pair).to_csv(pair_stats_path, index=False)

    # Save metrics without per_pair_stats (it goes to separate CSV)
    metrics_clean = {k: v for k, v in result.metrics.items() if k != "per_pair_stats"}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_clean, f, indent=2, ensure_ascii=False, default=str)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(result.params, f, indent=2, ensure_ascii=False, default=str)


def build_report_window_result(
    backtester: "BotParityBacktester",
    result: BacktestResult,
    report_start: pd.Timestamp | None,
    report_end: pd.Timestamp | None,
) -> BacktestResult | None:
    if report_start is None and report_end is None:
        return None

    trades_df = _filter_df_by_utc_window(result.trades, "entry_time", report_start, report_end)
    equity_df = _filter_df_by_utc_window(result.equity, "Date", report_start, report_end)
    coint_df = _filter_df_by_utc_window(result.coint_phases, "ended_time", report_start, report_end)
    ledger_col = "Date" if "Date" in result.ledger.columns else None
    ledger_df = (
        _filter_df_by_utc_window(result.ledger, ledger_col, report_start, report_end)
        if ledger_col
        else result.ledger.copy()
    )

    if equity_df.empty:
        return None

    metrics = backtester._build_metrics(trades_df, equity_df, coint_df)
    params = dict(result.params)
    params["report_start"] = str(report_start) if report_start is not None else ""
    params["report_end"] = str(report_end) if report_end is not None else ""

    return BacktestResult(
        trades=trades_df,
        equity=equity_df,
        metrics=metrics,
        coint_phases=coint_df,
        ledger=ledger_df,
        params=params,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — all tunable parameters are collected here for easy editing / testing.
#  Modify the DEFAULT_PARAMS dict below, or override via CLI flags.
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS: dict[str, Any] = {
    # ── Data ──────────────────────────────────────────────────────────────────
    # NOTE: "input" is NOT set here — it is auto-resolved from "timeframe" in main().
    #       You can still override it via --input CLI flag.
    "best_pairs":      os.path.join("market_neutral", "best_pairs.json"),
    "pair_blacklist":  os.path.join("market_neutral", "pair_blacklist.json"),
    "funding_csv":     "",
    "output_dir":      os.path.join("market_neutral", "backtest_results"),

    # ── Timeframe & window ────────────────────────────────────────────────────
    # Change ONLY this line to switch data file + all time-related parameters:
    "timeframe":       "4h",     # options: "1h", "4h", "30m"
    "window_size":     0,        # 0 = auto by timeframe

    # ── Capital & sizing ──────────────────────────────────────────────────────
    "capital":         1000000.0,    # USD capital base
    "leverage":        20,       # info only
    "max_notional_pct": 0.30,    # fraction of capital per pair leg (was 0.40)

    # ── Z-Score thresholds ────────────────────────────────────────────────────
    "z_entry":         1.8,      # min |Z| to trigger entry signal (was 1.9)
    "z_entry_max":     2.8,      # max |Z| — skip if above (was 2.2)
    "z_exit":          0.25,     # |Z| at which Z-TP fires (was 0.05)
    "z_stop":          4.0,      # |Z| at which Z-SL fires

    # ── Costs ─────────────────────────────────────────────────────────────────
    "commission_rate":  0.0004,
    "slippage_rate":    0.0005,
    "hardware_sltp_mode": "off",
    "hardware_sl_enabled": True,
    "hardware_tp_enabled": True,
    "sl_atr_mult": 2.5,
    "sl_min_pct": 0.10,
    "sl_max_pct": 0.30,
    "tp_atr_mult": 4.0,
    "tp_min_pct": 0.15,
    "tp_max_pct": 0.50,

    # ── Cointegration & beta ──────────────────────────────────────────────────
    "p_value_threshold":     0.03,
    "hedge_min":             0.3,
    "hedge_max":             3.0,
    "hl_min_days":           0.25,   # min half-life in days (6h for 1h TF)
    "hl_max_days":           2.0,    # max half-life in days (48h for 1h TF)
    "beta_threshold":        0.11,   # reject pair if |β| >= this
    "beta_alert_threshold":  0.30,   # beta_drift exit if |β| >= this and PnL > 0
    "beta_critical":         1.0,    # instant close if |β| >= this

    # ── Risk management ───────────────────────────────────────────────────────
    "circuit_breaker_pct":   0.30,   # max loss per pair (was 0.50)

    # ── Entry confirmation & hold ─────────────────────────────────────────────
    "signal_confirm_sec":    10,
    "coint_stability_min_bars": 1,    # was 2 — require 5 bars of cointegration
    "coint_broken_grace_bars": 2,     # wait N bars of broken coint before forced close
    "entry_et_target_abs_z": 0.5,
    "hold_multiplier":       3.0,    # max_hold = HL * this
    "max_hold_days":         12.0,   # hard cap on hold time (was 30)

    # ── Capacity ──────────────────────────────────────────────────────────────
    "max_active_pairs":  8,      # was 5
    "max_idle_pairs":    200,

    # ── Cooldowns ─────────────────────────────────────────────────────────────
    "sl_reentry_cooldown_sec":  0,
    "close_retry_cooldown_sec": 30,

    # ── Discovery ─────────────────────────────────────────────────────────────
    "discovery_every_bars":         1,
    "discovery_shards":             4,
    "discovery_max_pairs_per_cycle": 12000,
    "max_symbols":                  450,
    "top_pairs_limit":              300,
    "supplemental_pairs_limit":     400,
    "supplemental_symbols":         60,
    "supplemental_when_no_primary_signal": True,

    # ── Runtime ───────────────────────────────────────────────────────────────
    "progress_every_bars": 100,  # heartbeat interval; 0 disables
    "quick_bars":          0,    # use only last N bars; 0 = all
    "n_workers":           1,    # 1 = sequential (stable); 0 = auto (may crash on Windows)
    "exact_mode":          True, # strict accurate mode (recommended for validation)
    "idle_eval_every":     1,    # used when exact_mode=false
    "coint_recompute_every": 1,  # used when exact_mode=false (idle full-coint refresh interval)
    "discovery_recheck_bars": 0, # used when exact_mode=false; 0 disables retry cooldown
    "hyperopt_trials":      0,    # total trials (3 random + 7 Bayesian)
    "hyperopt_train_frac":  0.60,  # 60% of bars for train split
    "hyperopt_quick_bars":  0,     # 0 = full train split (reliable); >0 = fast/less trades
    "hyperopt_workers":     1,     # 2 parallel workers → ~2× faster; 0 or 1 = sequential
    "hyperopt_n_startup":   3,     # first N trials = random exploration before Bayesian
    "seed":                 42,
}


def main() -> None:
    args = parse_args()
    report_start = parse_timestamp_utc(args.report_start)
    report_end = parse_timestamp_utc(args.report_end)
    if report_start is not None and report_end is not None and report_end <= report_start:
        raise ValueError("--report-end must be later than --report-start.")

    # CLI overrides DEFAULT_PARAMS
    # Mapping for CLI arg names that differ from DEFAULT_PARAMS keys
    _cli_aliases = {"commission": "commission_rate", "slippage": "slippage_rate"}
    cfg = dict(DEFAULT_PARAMS)
    cli_dict = vars(args)
    for k, v in cli_dict.items():
        mapped = _cli_aliases.get(k, k.replace("-", "_"))
        if mapped in cfg and v is not None:
            cfg[mapped] = v
    if bool(getattr(args, "disable_pair_blacklist", False)):
        cfg["pair_blacklist"] = ""

    # ── Auto-resolve klines file from timeframe ────────────────────────────────
    # If --input was not explicitly passed on CLI, pick the matching file
    # to guarantee timeframe and data always stay in sync.
    _data_dir = os.path.join("market_neutral")
    _TF_FILES: dict[str, str] = {}
    for _fname in os.listdir(_data_dir):
        if _fname.startswith("klines_data_") and _fname.endswith(".csv"):
            # Extract timeframe token: klines_data_<TF>_clean_*.csv
            _parts = _fname.split("_")
            if len(_parts) >= 3:
                _tf_token = _parts[2]  # e.g. "1h", "4h", "30m"
                _TF_FILES[_tf_token] = os.path.join(_data_dir, _fname)

    _requested_tf = str(cfg.get("timeframe", "")).strip().lower()
    _cli_input = cli_dict.get("input")  # None if not passed on CLI
    if _cli_input:
        cfg["input"] = _cli_input  # explicit CLI override wins
    elif "input" not in cfg or not cfg.get("input"):
        # Auto-pick from discovered files
        if _requested_tf in _TF_FILES:
            cfg["input"] = _TF_FILES[_requested_tf]
        elif _TF_FILES:
            # Fallback: first available file
            cfg["input"] = next(iter(_TF_FILES.values()))
        else:
            raise FileNotFoundError(
                f"No klines CSV found in '{_data_dir}' for timeframe '{_requested_tf}'. "
                f"Available: {list(_TF_FILES.keys())}"
            )

    print(f"[INFO] timeframe={_requested_tf}  data={cfg['input']}")

    market = load_klines_market_data(cfg["input"], max_symbols=int(cfg["max_symbols"]))
    inferred_tf = infer_timeframe_from_index(market.dates)
    tf = str(cfg["timeframe"]).strip().lower() or inferred_tf

    # Sanity check: warn if data timeframe doesn't match requested
    if inferred_tf and _requested_tf and inferred_tf != _requested_tf:
        tf_cli_explicit = cli_dict.get("timeframe") is not None
        input_cli_explicit = cli_dict.get("input") is not None
        if input_cli_explicit and not tf_cli_explicit:
            # If user provided only --input, trust file cadence over DEFAULT timeframe.
            tf = inferred_tf
            print(
                f"[WARN] Requested timeframe '{_requested_tf}' but data appears to be '{inferred_tf}'. "
                f"Using inferred timeframe '{inferred_tf}' (because --timeframe was not explicitly passed)."
            )
        else:
            print(
                f"[WARN] Requested timeframe '{_requested_tf}' but data appears to be '{inferred_tf}'. "
                f"Using '{_requested_tf}' as declared."
            )

    params = BacktestParams(
        timeframe=tf,
        window_size=int(cfg["window_size"]),
        capital=float(cfg["capital"]),
        leverage=int(cfg["leverage"]),
        max_notional_pct=float(cfg["max_notional_pct"]),
        funding_csv=str(cfg.get("funding_csv", "") or ""),
        z_entry=float(cfg["z_entry"]),
        z_entry_max=float(cfg["z_entry_max"]),
        z_exit=float(cfg["z_exit"]),
        z_stop=float(cfg["z_stop"]),
        commission_rate=float(cfg["commission_rate"]),
        slippage_rate=float(cfg["slippage_rate"]),
        hardware_sltp_mode=str(cfg["hardware_sltp_mode"]).strip().lower(),
        hardware_sl_enabled=parse_bool(cfg["hardware_sl_enabled"]),
        hardware_tp_enabled=parse_bool(cfg["hardware_tp_enabled"]),
        sl_atr_mult=float(cfg["sl_atr_mult"]),
        sl_min_pct=float(cfg["sl_min_pct"]),
        sl_max_pct=float(cfg["sl_max_pct"]),
        tp_atr_mult=float(cfg["tp_atr_mult"]),
        tp_min_pct=float(cfg["tp_min_pct"]),
        tp_max_pct=float(cfg["tp_max_pct"]),
        p_value_threshold=float(cfg["p_value_threshold"]),
        hedge_min=float(cfg["hedge_min"]),
        hedge_max=float(cfg["hedge_max"]),
        beta_threshold=float(cfg["beta_threshold"]),
        beta_alert_threshold=float(cfg["beta_alert_threshold"]),
        beta_critical=float(cfg["beta_critical"]),
        circuit_breaker_pct=float(cfg["circuit_breaker_pct"]),
        signal_confirm_sec=int(cfg["signal_confirm_sec"]),
        coint_stability_min_bars=int(cfg["coint_stability_min_bars"]),
        coint_broken_grace_bars=int(cfg["coint_broken_grace_bars"]),
        entry_et_target_abs_z=float(cfg["entry_et_target_abs_z"]),
        max_active_pairs=int(cfg["max_active_pairs"]),
        max_idle_pairs=int(cfg["max_idle_pairs"]),
        hold_multiplier=float(cfg["hold_multiplier"]),
        max_hold_days=float(cfg["max_hold_days"]),
        sl_reentry_cooldown_sec=int(cfg["sl_reentry_cooldown_sec"]),
        close_retry_cooldown_sec=int(cfg["close_retry_cooldown_sec"]),
        discovery_every_bars=int(cfg["discovery_every_bars"]),
        discovery_shards=int(cfg["discovery_shards"]),
        discovery_max_pairs_per_cycle=int(cfg["discovery_max_pairs_per_cycle"]),
        max_symbols=int(cfg["max_symbols"]),
        top_pairs_limit=int(cfg["top_pairs_limit"]),
        progress_every_bars=int(cfg["progress_every_bars"]),
        hl_min_days=float(cfg["hl_min_days"]),
        hl_max_days=float(cfg["hl_max_days"]),
    )

    blacklist_keys = load_pair_blacklist(cfg.get("pair_blacklist"))
    if blacklist_keys:
        print(f"[INFO] Pair blacklist loaded: {len(blacklist_keys)} pairs")

    candidates = load_candidate_pairs(
        best_pairs_path=cfg["best_pairs"],
        symbols=market.symbols,
        limit=int(params.top_pairs_limit),
        supplemental_pairs_limit=int(cfg.get("supplemental_pairs_limit", 0)),
        supplemental_symbols=int(cfg.get("supplemental_symbols", 40)),
        blacklist_keys=blacklist_keys,
    )
    if not candidates:
        raise RuntimeError("No candidate pairs found for current klines universe.")
    n_primary = sum(1 for cp in candidates if str(cp.source) != "supplemental_combo")
    n_supp = len(candidates) - n_primary
    print(f"[INFO] Candidate universe: primary={n_primary}, supplemental={n_supp}, total={len(candidates)}")

    quick_bars = int(cfg["quick_bars"])
    if quick_bars > 0 and quick_bars < len(market.dates):
        start = len(market.dates) - quick_bars
        market = MarketData(
            dates=market.dates[start:],
            symbols=market.symbols,
            symbol_to_idx=market.symbol_to_idx,
            open_arr=market.open_arr[start:, :].copy(),
            high_arr=market.high_arr[start:, :].copy(),
            low_arr=market.low_arr[start:, :].copy(),
            close_arr=market.close_arr[start:, :].copy(),
            volume_arr=market.volume_arr[start:, :].copy(),
        )

    hyperopt_trials = int(cfg["hyperopt_trials"])
    hyperopt_history: list[dict[str, Any]] = []
    if hyperopt_trials > 0:
        frac = min(1.0, max(0.1, float(cfg["hyperopt_train_frac"])))
        train_end = max(1, int(len(market.dates) * frac)) - 1
        # Ensure hyperopt train split has enough bars for current rolling window:
        # run() requires end_idx >= min_data_points.
        min_train_end = max(1, int(params.resolved_window_size()))
        max_end = len(market.dates) - 1
        if max_end < min_train_end:
            print(
                f"[WARN] Hyperopt skipped: not enough bars for window={params.resolved_window_size()} "
                f"(bars={len(market.dates)})."
            )
        else:
            train_end = min(max(train_end, min_train_end), max_end)
            best_params, hyperopt_history = run_hyperopt(
                market=market,
                candidates=candidates,
                base_params=params,
                trials=hyperopt_trials,
                seed=int(cfg["seed"]),
                end_idx=train_end,
                max_workers=int(cfg.get("hyperopt_workers", 0)),
                n_startup=int(cfg.get("hyperopt_n_startup", 3)),
                quick_bars=int(cfg.get("hyperopt_quick_bars", 0)),
            )
            params = best_params

    t0 = time.perf_counter()
    backtester = BotParityBacktester(market=market, params=params, candidates=candidates,
                                     n_workers=int(cfg["n_workers"]),
                                     exact_mode=parse_bool(cfg["exact_mode"]),
                                     idle_eval_every=int(cfg["idle_eval_every"]),
                                     coint_recompute_every=int(cfg["coint_recompute_every"]),
                                     discovery_recheck_bars=int(cfg["discovery_recheck_bars"]),
                                     supplemental_when_no_primary_signal=parse_bool(
                                         cfg.get("supplemental_when_no_primary_signal", True)
                                     ))
    if backtester.funding_enabled and backtester.funding_data is not None:
        print(
            f"[INFO] Funding loaded: matched={backtester.funding_data.matched_records}/"
            f"{backtester.funding_data.loaded_records} skipped={backtester.funding_data.skipped_records}"
        )
    result = backtester.run()
    elapsed = time.perf_counter() - t0

    prefix = f"bot_parity_{params.timeframe}"
    save_result(result=result, out_dir=cfg["output_dir"], prefix=prefix)
    report_result = build_report_window_result(
        backtester=backtester,
        result=result,
        report_start=report_start,
        report_end=report_end,
    )
    if report_result is not None:
        save_result(result=report_result, out_dir=cfg["output_dir"], prefix=f"{prefix}_report_window")

    if hyperopt_history:
        ho_path = Path(cfg["output_dir"]) / f"{prefix}_hyperopt_trials.csv"
        pd.DataFrame(hyperopt_history).to_csv(ho_path, index=False)

    print("=" * 60)
    print("  BACKTEST COMPLETE")
    print("=" * 60)
    print(f"  Timeframe       : {params.timeframe}")
    print(f"  Window size     : {params.resolved_window_size()}")
    print(f"  Candidates      : {len(candidates)} (primary={n_primary}, supplemental={n_supp})")
    print(f"  Blacklist pairs : {len(blacklist_keys)}")
    print(f"  Bars processed  : {len(backtester.equity_rows)}")
    print(f"  Elapsed         : {elapsed:.1f} sec")
    print(f"  Total trades    : {result.metrics.get('total_trades')}")
    print(f"  Win rate        : {result.metrics.get('win_rate')}")
    print(f"  Total net PnL   : {result.metrics.get('total_net_pnl')}")
    print(f"  Funding total   : {result.metrics.get('total_funding')}")
    print(f"  Sharpe          : {result.metrics.get('sharpe')}")
    print(f"  Sortino         : {result.metrics.get('sortino')}")
    print(f"  Max drawdown    : {result.metrics.get('max_drawdown')}")
    print(f"  Profit factor   : {result.metrics.get('profit_factor')}")
    print(f"  Avg trade PnL   : {result.metrics.get('avg_trade_pnl')}")
    print(f"  Avg hold hours  : {result.metrics.get('avg_trade_hold_hours')}")
    print(f"  Avg coint hours : {result.metrics.get('avg_cointegration_duration_hours')}")
    print(f"  Avg risk/trade  : {result.metrics.get('avg_risk_per_trade_usdt')}")
    print(f"  Final equity    : {result.metrics.get('final_equity')}")
    print(f"  Total return %  : {result.metrics.get('total_return_pct')}")
    print(f"  Close reasons   : {result.metrics.get('close_reason_breakdown')}")
    if str(result.metrics.get("hardware_sltp_mode", "off")) != "off":
        print(
            f"  HW SL/TP        : mode={result.metrics.get('hardware_sltp_mode')} "
            f"touches={result.metrics.get('hardware_touch_trades')} "
            f"(SL={result.metrics.get('hardware_sl_touch_trades')}, TP={result.metrics.get('hardware_tp_touch_trades')})"
        )
    print(f"  Unique pairs    : {result.metrics.get('unique_pairs_traded')}")
    print(f"  Output dir      : {cfg['output_dir']}")
    print("=" * 60)
    if report_result is not None:
        print("")
        print("=" * 60)
        print("  REPORT WINDOW")
        print("=" * 60)
        print(f"  Start           : {report_start}")
        print(f"  End             : {report_end}")
        print(f"  Total trades    : {report_result.metrics.get('total_trades')}")
        print(f"  Total net PnL   : {report_result.metrics.get('total_net_pnl')}")
        print(f"  Funding total   : {report_result.metrics.get('total_funding')}")
        print(f"  Final equity    : {report_result.metrics.get('final_equity')}")
        print(f"  Total return %  : {report_result.metrics.get('total_return_pct')}")
        print(f"  Close reasons   : {report_result.metrics.get('close_reason_breakdown')}")
        print("=" * 60)

    # Per-pair summary (top 15 + bottom 5)
    per_pair = result.metrics.get("per_pair_stats", [])
    if per_pair:
        print("\n" + "=" * 80)
        print("  TOP 15 PROFITABLE PAIRS")
        print("=" * 80)
        print(f"  {'Pair':<30} {'Trades':>6} {'WR':>6} {'NetPnL':>12} {'Sharpe':>8} {'AvgHold':>8} {'PF':>7}")
        print("-" * 80)
        for row in per_pair[:15]:
            wr = f"{row['win_rate']*100:.0f}%" if not np.isnan(row.get('win_rate', np.nan)) else 'N/A'
            sh = f"{row['pair_sharpe']:.2f}" if not np.isnan(row.get('pair_sharpe', np.nan)) else 'N/A'
            pf = f"{row['profit_factor']:.2f}" if not np.isnan(row.get('profit_factor', np.nan)) else 'N/A'
            ah = f"{row['avg_hold_hours']:.1f}h" if not np.isnan(row.get('avg_hold_hours', np.nan)) else 'N/A'
            print(f"  {row['pair']:<30} {row['trades']:>6} {wr:>6} {row['net_pnl']:>12.2f} {sh:>8} {ah:>8} {pf:>7}")

        losers = [r for r in per_pair if r['net_pnl'] < 0]
        if losers:
            print("\n  BOTTOM 5 LOSING PAIRS")
            print("-" * 80)
            for row in losers[-5:]:
                wr = f"{row['win_rate']*100:.0f}%" if not np.isnan(row.get('win_rate', np.nan)) else 'N/A'
                print(f"  {row['pair']:<30} {row['trades']:>6} {wr:>6} {row['net_pnl']:>12.2f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
