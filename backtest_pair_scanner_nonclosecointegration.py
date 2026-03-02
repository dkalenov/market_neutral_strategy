#!/usr/bin/env python3
"""
backtest_pair_scanner.py — Full-fidelity per-pair scanner using the REAL engine.

Uses the exact same BotParityBacktester from backtest_bot_parity.py to test each
candidate pair in isolation.  Every pair gets its own backtester instance with
max_active_pairs=1, max_idle_pairs=1, and no symbol contention.

This means the scanner uses:
  - Real OLS cointegration (statsmodels) with full rolling window
  - Real z-score via spread residuals
  - Real entry conditions (z_entry, z_entry_max, coint_stability, beta, hedge, p-value, half-life)
  - Real exit conditions (z_tp, z_sl, broken_coint, beta_drift, beta_critical, circuit, time_exit)
  - Real position sizing (vol_parity_notional) with slippage & commission
  - Real signal confirmation delay (confirm_bars)
  - Real cooldown logic

OUTPUT:
  - pair_scanner_ranking.csv      — full ranking of ALL qualifying pairs
  - best_pairs_backtest.json      — top-N pairs with exact parameters used

USAGE:
  # Scan ALL possible pair combinations (default):
  python market_neutral/backtest_pair_scanner.py

  # Use fewer workers on weak machines:
  python market_neutral/backtest_pair_scanner.py --scan-workers 2

  # Run with hyperopt first, then scan:
  python market_neutral/backtest_pair_scanner.py --hyperopt-trials 10

  # Override strategy params:
  python market_neutral/backtest_pair_scanner.py --z-entry 2.0 --z-stop 3.5
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import dataclasses
import json
import math
import sys
import time
import traceback
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Prevent UnicodeEncodeError on Windows cp1252 terminals.
try:
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
except Exception:
    pass

# Import the real engine from backtest_bot_parity (GRACE PERIOD version)
from backtest_bot_parity_nonclosecointegration import (
    BacktestParams,
    BacktestResult,
    BotParityBacktester,
    CandidatePair,
    MarketData,
    load_candidate_pairs,
    load_klines_market_data,
    metrics_to_objective,
)


# ══════════════════════════════════════════════════════════════════════════════
# 🎛️  DEFAULT_PARAMS — все настройки здесь. Меняй и нажимай F5 ▶️
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_PARAMS: dict[str, Any] = {
    # ── Paths ─────────────────────────────────────────────────────────────────
    "best_pairs":      os.path.join("market_neutral", "best_pairs.json"),
    "output_dir":      os.path.join("market_neutral", "backtest_results"),
    "timeframe":       "4h",

    # ── Capital (per-pair: large but not infinite, still uses real sizing) ─────
    "capital":         1_000_000.0,
    "leverage":        20,
    "max_notional_pct": 0.30,

    # ── Z-Score thresholds ────────────────────────────────────────────────────
    "z_entry":         1.8,
    "z_entry_max":     2.5,
    "z_exit":          0.05,
    "z_stop":          4.0,

    # ── Costs ─────────────────────────────────────────────────────────────────
    "commission_rate":  0.0004,
    "slippage_rate":    0.0005,

    # ── Cointegration quality ─────────────────────────────────────────────────
    "p_value_threshold":  0.05,
    "hedge_min":          0.3,
    "hedge_max":          3.0,

    # ── Beta (BTC neutrality) ─────────────────────────────────────────────────
    "beta_threshold":       0.11,
    "beta_alert_threshold": 0.30,
    "beta_critical":        1.0,

    # ── Circuit breaker ───────────────────────────────────────────────────────
    "circuit_breaker_pct":  0.50,

    # ── Entry confirmation & hold ─────────────────────────────────────────────
    "signal_confirm_sec":    10,
    "coint_stability_min_bars": 1,
    "entry_et_target_abs_z": 0.5,
    "hold_multiplier":       3.0,
    "max_hold_days":         12.0,

    # ── Cooldowns ─────────────────────────────────────────────────────────────
    "sl_reentry_cooldown_sec":  0,
    "close_retry_cooldown_sec": 30,

    # ── Half-life bounds ──────────────────────────────────────────────────────
    "hl_min_days": 0.5,
    "hl_max_days": 4.0,

    # ── Window ────────────────────────────────────────────────────────────────
    "window_size":         180,
    "max_symbols":         450,

    # ── Scanner-specific ──────────────────────────────────────────────────────
    #
    # ⭐⭐ МЕНЯЙ ПЕРЕД КАЖДЫМ ЗАПУСКОМ ⭐⭐
    #
    # Диапазон пар: None=весь рынок (0..53301),  или строка "27500-53301"
    # Твой ПК (40%): "27500-53301"  |  MacBook M2 (60%): None
    "pair_range":          None,  # None = scan all pairs from --from-pairs JSON (set range only for brute all-combos scan)

    # Воркеры: Windows=12, MacBook M2=8, Colab=2
    "scan_workers":        1,
    "max_inflight":       0,   # 0=auto (workers*3)
    "worker_recycle":     80,  # max tasks per child (0=disable)

    # Точность: 1=максимальная (медленно), 6=быстрый фильтр (~90% точности)
    "recompute":           1,

    # Шаг 2 (точный скан по лучшим парам из грубого фильтра):
    # None = сканировать все пары,  или путь к best_pairs_backtest.json
    "from_pairs":          None,
    # Пример: "market_neutral/backtest_results/best_pairs_backtest.json"

    # CSV котировок: None = автопоиск в market_neutral/
    "input_file":          None,

    "min_trades":          3,     # min trades to qualify
    "top_n":               9000,  # how many pairs to export

    # ── Hyperopt ──────────────────────────────────────────────────────────────
    "hyperopt_trials":     0,
    "hyperopt_n_startup":  3,
    "seed":                42,
}


# ══════════════════════════════════════════════════════════════════════════════
# Helper: cfg → BacktestParams dict
# ══════════════════════════════════════════════════════════════════════════════

def _cfg_to_params_dict(cfg: dict) -> dict:
    """Extract BacktestParams fields from flat cfg dict."""
    return {
        "timeframe":                  cfg.get("timeframe", "4h"),
        "window_size":                int(cfg.get("window_size", 180)),
        "capital":                    float(cfg.get("capital", 1_000_000)),
        "leverage":                   int(cfg.get("leverage", 20)),
        "max_notional_pct":           float(cfg.get("max_notional_pct", 0.30)),
        "z_entry":                    float(cfg.get("z_entry", 1.8)),
        "z_entry_max":                float(cfg.get("z_entry_max", 2.8)),
        "z_exit":                     float(cfg.get("z_exit", 0.25)),
        "z_stop":                     float(cfg.get("z_stop", 4.0)),
        "commission_rate":            float(cfg.get("commission_rate", 0.0004)),
        "slippage_rate":              float(cfg.get("slippage_rate", 0.0005)),
        "p_value_threshold":          float(cfg.get("p_value_threshold", 0.05)),
        "hedge_min":                  float(cfg.get("hedge_min", 0.3)),
        "hedge_max":                  float(cfg.get("hedge_max", 3.0)),
        "beta_threshold":             float(cfg.get("beta_threshold", 0.11)),
        "beta_alert_threshold":       float(cfg.get("beta_alert_threshold", 0.30)),
        "beta_critical":              float(cfg.get("beta_critical", 1.0)),
        "circuit_breaker_pct":        float(cfg.get("circuit_breaker_pct", 0.50)),
        "signal_confirm_sec":         int(cfg.get("signal_confirm_sec", 10)),
        "coint_stability_min_bars":   int(cfg.get("coint_stability_min_bars", 1)),
        "coint_broken_grace_bars":    int(cfg.get("coint_broken_grace_bars", 0)),
        "entry_et_target_abs_z":      float(cfg.get("entry_et_target_abs_z", 0.5)),
        "hold_multiplier":            float(cfg.get("hold_multiplier", 3.0)),
        "max_hold_days":              float(cfg.get("max_hold_days", 12.0)),
        "sl_reentry_cooldown_sec":    int(cfg.get("sl_reentry_cooldown_sec", 0)),
        "close_retry_cooldown_sec":   int(cfg.get("close_retry_cooldown_sec", 30)),
        "max_active_pairs":           1,   # single-pair mode
        "max_idle_pairs":             1,
        "discovery_every_bars":       1,
        "discovery_shards":           1,
        "discovery_max_pairs_per_cycle": 1,
        "max_symbols":                int(cfg.get("max_symbols", 450)),
        "top_pairs_limit":            1,
        "progress_every_bars":        0,   # silent
        "hl_min_days":                float(cfg.get("hl_min_days", 0.25)),
        "hl_max_days":                float(cfg.get("hl_max_days", 2.0)),
        # Scanner-only key (not BacktestParams): consumed by _scan_one_pair via pop().
        "recompute_bars":             int(cfg.get("recompute_bars", cfg.get("recompute", 1))),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Per-pair scan — worker globals & init (Windows-safe serialization)
# ══════════════════════════════════════════════════════════════════════════════

# Worker-local market data (populated once via initializer)
_SCAN_MARKET: MarketData | None = None


def _scan_worker_init(
    dates_bytes: bytes,
    dates_len: int,
    symbols: list[str],
    sym_to_idx: dict[str, int],
    ohlcv_bytes: bytes,
    shape: tuple[int, int],
) -> None:
    """Called once per worker process.  Reconstructs MarketData from raw bytes.
    We serialize numpy arrays as bytes to avoid pickling issues on Windows.
    """
    global _SCAN_MARKET
    n_bars, n_syms = shape
    # Reconstruct dates from int64 nanosecond timestamps
    dates_arr = np.frombuffer(dates_bytes, dtype=np.int64)[:dates_len].copy()
    dates_idx = pd.DatetimeIndex(dates_arr, tz="UTC")
    # Reconstruct OHLCV arrays (stacked: open|high|low|close|volume = 5 × n_bars × n_syms)
    all_data = np.frombuffer(ohlcv_bytes, dtype=np.float64).reshape(5, n_bars, n_syms).copy()
    _SCAN_MARKET = MarketData(
        dates=dates_idx,
        symbols=symbols,
        symbol_to_idx=sym_to_idx,
        open_arr=all_data[0],
        high_arr=all_data[1],
        low_arr=all_data[2],
        close_arr=all_data[3],
        volume_arr=all_data[4],
    )


def _prepare_market_init_args(market: MarketData) -> tuple:
    """Serialize MarketData into bytes for worker init (Windows-safe)."""
    # Convert DatetimeIndex → int64 nanoseconds
    dates_int = market.dates.asi8.copy()
    dates_bytes = dates_int.tobytes()
    dates_len = len(dates_int)
    # Stack OHLCV into single buffer
    stacked = np.stack([
        market.open_arr, market.high_arr, market.low_arr,
        market.close_arr, market.volume_arr,
    ], axis=0).astype(np.float64)
    ohlcv_bytes = stacked.tobytes()
    shape = (market.open_arr.shape[0], market.open_arr.shape[1])
    return (dates_bytes, dates_len, market.symbols, market.symbol_to_idx, ohlcv_bytes, shape)


def _scan_one_pair(args: tuple) -> dict | None:
    """Worker: run full BotParityBacktester for a single pair.
    Returns per-pair metrics dict or None if insufficient data/trades.
    """
    sym_a, sym_b, params_dict = args
    market = _SCAN_MARKET
    if market is None:
        return None
    if sym_a not in market.symbol_to_idx or sym_b not in market.symbol_to_idx:
        return None
    try:
        params_dict = dict(params_dict)  # copy so we don't mutate the original
        recompute_bars = params_dict.pop("recompute_bars", 1)
        params = BacktestParams(**params_dict)

        candidate = CandidatePair(symbol1=sym_a, symbol2=sym_b, score=1.0, source="scanner")
        bt = BotParityBacktester(
            market=market,
            params=params,
            candidates=[candidate],
            n_workers=1,
            exact_mode=False,
            coint_recompute_every=recompute_bars,
        )
        result = bt.run()
        m = result.metrics

        total_trades = int(m.get("total_trades", 0))
        if total_trades < 1:
            return None

        return {
            "pair":             f"{sym_a}-{sym_b}",
            "sym_a":            sym_a,
            "sym_b":            sym_b,
            "trades":           total_trades,
            "wins":             int(m.get("wins", 0)),
            "losses":           int(m.get("losses", 0)),
            "win_rate":         float(m.get("win_rate", 0) or 0),
            "total_net_pnl":    float(m.get("total_net_pnl", 0) or 0),
            "total_gross_pnl":  float(m.get("total_gross_pnl", 0) or 0),
            "total_fees":       float(m.get("total_fees", 0) or 0),
            "avg_pnl":          float(m.get("avg_trade_pnl", 0) or 0),
            "median_pnl":       float(m.get("median_trade_pnl", 0) or 0),
            "best_trade":       float(m.get("best_trade_pnl", 0) or 0),
            "worst_trade":      float(m.get("worst_trade_pnl", 0) or 0),
            "sharpe":           float(m.get("sharpe", 0) or 0),
            "sortino":          float(m.get("sortino", 0) or 0),
            "max_dd":           float(m.get("max_drawdown", 0) or 0),
            "pf":               float(m.get("profit_factor", 0) or 0),
            "avg_hold_h":       float(m.get("avg_trade_hold_hours", 0) or 0),
            "avg_hold_bars":    float(m.get("avg_trade_hold_bars", 0) or 0),
            "total_return_pct": float(m.get("total_return_pct", 0) or 0),
            "avg_mae_pct":      float(m.get("avg_mae_pct_notional", 0) or 0),
            "avg_mfe_pct":      float(m.get("avg_mfe_pct_notional", 0) or 0),
            "close_reasons":    m.get("close_reason_breakdown", {}),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Direct single-pair scan (no subprocess — for hyperopt)
# ══════════════════════════════════════════════════════════════════════════════

def _scan_one_pair_direct(
    market: MarketData, sym_a: str, sym_b: str, params_dict: dict,
    recompute_bars: int = 4,
) -> dict | None:
    """Run a single pair backtest directly (no subprocess).

    recompute_bars controls cointegration recalculation cadence:
      1 = every bar (slowest, most accurate)
      4 = every 4 bars  ← default for hyperopt (4x faster, ranks params equally well)
      6 = every 6 bars  ← fast filter mode
    """
    if sym_a not in market.symbol_to_idx or sym_b not in market.symbol_to_idx:
        return None
    try:
        params_dict = dict(params_dict)  # copy so we don't mutate the original
        params_dict.pop("recompute_bars", None)  # not a BacktestParams field
        params = BacktestParams(**params_dict)
        candidate = CandidatePair(symbol1=sym_a, symbol2=sym_b, score=1.0, source="scanner")
        bt = BotParityBacktester(
            market=market, params=params, candidates=[candidate], n_workers=1,
            exact_mode=False,              # hyperopt: use recompute cadence, not every-bar
            coint_recompute_every=max(1, int(recompute_bars)),
        )
        result = bt.run()
        m = result.metrics
        total_trades = int(m.get("total_trades", 0))
        if total_trades < 1:
            return None
        return {
            "pair": f"{sym_a}-{sym_b}", "sym_a": sym_a, "sym_b": sym_b,
            "trades": total_trades,
            "win_rate": float(m.get("win_rate", 0) or 0),
            "total_net_pnl": float(m.get("total_net_pnl", 0) or 0),
            "avg_pnl": float(m.get("avg_trade_pnl", 0) or 0),
            "sharpe": float(m.get("sharpe", 0) or 0),
            "sortino": float(m.get("sortino", 0) or 0),
            "max_dd": float(m.get("max_drawdown", 0) or 0),
            "pf": float(m.get("profit_factor", 0) or 0),
            "avg_hold_h": float(m.get("avg_trade_hold_hours", 0) or 0),
            "total_return_pct": float(m.get("total_return_pct", 0) or 0),
            "close_reasons": m.get("close_reason_breakdown", {}),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Score computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_pair_score(row: dict, min_trades: int = 3) -> float:
    """Composite quality score for ranking pairs.

    Components:
      - Sharpe ratio (risk-adjusted return)
      - Win rate
      - Trade count (log-scaled — more trades = more reliable)
      - Profit factor
      - TP ratio (fraction of z_tp exits — actual mean reversion)
    Penalties:
      - Drawdown severity
      - Too many time_exit (signal never reverted)
    """
    trades = row.get("trades", 0)
    if trades < min_trades:
        return -1.0

    sharpe = row.get("sharpe", 0) or 0
    wr     = row.get("win_rate", 0) or 0
    pf     = row.get("pf", 0) or 0
    mdd    = abs(row.get("max_dd", 0) or 0)

    reasons = row.get("close_reasons", {})
    total_exits = sum(reasons.values()) if reasons else max(1, trades)
    tp_ratio   = reasons.get("z_tp", 0) / total_exits
    time_ratio = reasons.get("time_exit", 0) / total_exits

    score = (
        0.35 * max(-1, min(3, sharpe))
        + 0.20 * wr
        + 0.15 * min(1.0, math.log1p(trades) / math.log1p(30))
        + 0.10 * min(1.0, pf / 3.0)
        + 0.10 * tp_ratio
        - 0.05 * min(1.0, mdd * 10)
        - 0.05 * time_ratio
    )
    return round(score, 6)


# ══════════════════════════════════════════════════════════════════════════════
# Hyperopt (optimizes params on a sample of pairs)
# ══════════════════════════════════════════════════════════════════════════════

_HYPEROPT_PARAM_KEYS = [
    "z_entry", "z_entry_max", "z_exit", "z_stop",
    "max_notional_pct", "circuit_breaker_pct", "beta_threshold",
    "beta_alert_threshold", "beta_critical",
    "coint_stability_min_bars", "coint_broken_grace_bars",
    "hold_multiplier", "max_hold_days",
    "hedge_min", "hedge_max", "hl_min_days", "hl_max_days",
    "p_value_threshold",
]


def _hyperopt_eval_batch(
    init_args: tuple,
    sample: list[tuple[str, str]],
    params_dict: dict,
    n_workers: int,
    recompute_bars: int = 4,
) -> list[dict]:
    """Evaluate a batch of pairs in parallel using existing worker pool."""
    params_dict = dict(params_dict)
    params_dict["recompute_bars"] = recompute_bars
    tasks = [(a, b, params_dict) for a, b in sample]

    if n_workers <= 1:
        # Fallback: sequential (for debugging or single-core machines)
        results = [_scan_one_pair(t) for t in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_scan_worker_init,
            initargs=init_args,
        ) as pool:
            results = list(pool.map(_scan_one_pair, tasks, chunksize=max(1, len(tasks) // n_workers)))

    return [r for r in results if r is not None and r.get("trades", 0) >= 1]


def _save_hyperopt_best_params(history: list[dict], seed: int, out_dir: Path) -> None:
    """Save JSON with the best trial's parameters (overwritten when a new best appears)."""
    best_row = max(history, key=lambda r: r.get("objective", -99))
    best_params_out = {
        "seed": seed,
        "best_trial":   best_row["trial"],
        "objective":    best_row["objective"],
        "avg_score":    best_row["avg_score"],
        "active_pairs": best_row["active_pairs"],
        "total_trades": best_row["total_trades"],
        "avg_pnl":      best_row.get("avg_pnl", 0),
        "avg_win_rate": best_row.get("avg_win_rate", 0),
        "avg_sharpe":   best_row.get("avg_sharpe", 0),
        "avg_pf":       best_row.get("avg_pf", 0),
        "params": {k: best_row[k] for k in _HYPEROPT_PARAM_KEYS if k in best_row},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    bp_path = out_dir / f"best_params_seed{seed}.json"
    with open(bp_path, "w") as f:
        json.dump(best_params_out, f, indent=2)


def _print_hyperopt_summary(history: list[dict], seed: int, elapsed_sec: float) -> None:
    """Print a rich summary after all hyperopt trials finish."""
    if not history:
        print("  [hyperopt] No valid trials completed.")
        return

    sorted_h = sorted(history, key=lambda r: r.get("objective", -99), reverse=True)
    best = sorted_h[0]
    top_n = sorted_h[:min(5, len(sorted_h))]

    print(f"\n  {'═' * 70}")
    print(f"  ║  HYPEROPT SUMMARY  (seed={seed}, {len(history)} trials, {elapsed_sec/60:.1f} min)")
    print(f"  {'═' * 70}")
    print(f"  ║  Best trial: #{best['trial']}  obj={best['objective']:+.4f}")
    bp = {k: best.get(k) for k in _HYPEROPT_PARAM_KEYS if k in best}
    print(f"  ║  Params:  z={bp.get('z_entry','?')}-{bp.get('z_entry_max','?')}→{bp.get('z_exit','?')}"
          f"  stop={bp.get('z_stop','?')}  coint={bp.get('coint_stability_min_bars','?')}b"
          f"  hl={bp.get('hl_min_days','?')}-{bp.get('hl_max_days','?')}d"
          f"  hold_mult={bp.get('hold_multiplier','?')}  max_hold={bp.get('max_hold_days','?')}d")
    print(f"  {'─' * 70}")
    print(f"  ║  Top-{len(top_n)} trials:")
    for r in top_n:
        marker = "★" if r["trial"] == best["trial"] else " "
        print(
            f"  ║  {marker} #{r['trial']:>2}"
            f"  obj={r['objective']:+.4f}"
            f"  wr={r.get('avg_win_rate', 0):.1%}"
            f"  sharpe={r.get('avg_sharpe', 0):+.2f}"
            f"  pf={r.get('avg_pf', 0):.2f}"
            f"  pnl/pair={r.get('avg_pnl', 0):+,.0f}$"
            f"  active={r.get('active_pairs', 0)}"
            f"  trades={r.get('total_trades', 0)}"
        )

    # Parameter stability across top trials
    if len(top_n) >= 3:
        print(f"  {'─' * 70}")
        print(f"  ║  Parameter stability (top-{len(top_n)}: min / median / max):")
        for k in ["z_entry", "z_exit", "z_stop", "coint_stability_min_bars",
                   "hl_min_days", "hl_max_days", "hold_multiplier", "p_value_threshold"]:
            vals = [r.get(k) for r in top_n if r.get(k) is not None]
            if vals:
                vals_sorted = sorted(vals)
                med = vals_sorted[len(vals_sorted) // 2]
                print(f"  ║    {k:<28}  {min(vals):>8.3f} / {med:>8.3f} / {max(vals):>8.3f}")

    print(f"  {'═' * 70}\n")


def run_scanner_hyperopt(
    market: MarketData,
    pairs: list[tuple[str, str]],
    base_cfg: dict,
    trials: int = 10,
    n_startup: int = 3,
    seed: int = 42,
    sample_pairs: int = 30,
) -> tuple[dict, list[dict]]:
    """Run hyperopt on a sample of pairs to find best parameters.

    For each trial: run sampled pairs with candidate params in parallel,
    objective = avg score.  Saves CSV + JSON incrementally after each trial.
    Returns (best_cfg, history).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rng = np.random.RandomState(seed)
    if len(pairs) > sample_pairs:
        idxs = rng.choice(len(pairs), size=sample_pairs, replace=False)
        sample = [pairs[i] for i in idxs]
    else:
        sample = list(pairs)

    # ── Prepare parallel worker pool data ──────────────────────────────────
    n_workers = max(1, int(base_cfg.get("scan_workers", 0)) or (os.cpu_count() or 4) - 2)
    init_args = _prepare_market_init_args(market)
    hyperopt_recompute = max(1, int(base_cfg.get("hyperopt_recompute", 4)))

    # ── Pool warmup: test with 1 pair before committing ───────────────────
    pool_works = False
    if n_workers > 1:
        test_pair = sample[0]
        test_params = _cfg_to_params_dict(base_cfg)
        test_params["recompute_bars"] = hyperopt_recompute
        test_task = (test_pair[0], test_pair[1], test_params)
        try:
            with ProcessPoolExecutor(
                max_workers=1,
                initializer=_scan_worker_init,
                initargs=init_args,
            ) as test_pool:
                test_results = list(test_pool.map(_scan_one_pair, [test_task]))
                r = test_results[0] if test_results else None
                if r is not None:
                    print(f"  [hyperopt] ✓ Pool warmup OK: {test_pair[0]}-{test_pair[1]} → {r.get('trades', 0)} trades", flush=True)
                    pool_works = True
                else:
                    # Try direct call to see if the pair itself is bad
                    direct_r = _scan_one_pair_direct(market, test_pair[0], test_pair[1], test_params, recompute_bars=hyperopt_recompute)
                    if direct_r is not None:
                        print(f"  [hyperopt] ⚠ Pool returns None but direct call works → pool init broken, using sequential", flush=True)
                    else:
                        print(f"  [hyperopt] ⚠ Both pool and direct return None for test pair (pair has 0 trades, this is normal)", flush=True)
                        pool_works = True  # Pool probably works, pair just has no trades
        except Exception as e:
            print(f"  [hyperopt] ⚠ Pool warmup FAILED: {e}", flush=True)
            traceback.print_exc()
            print(f"  [hyperopt] ⚠ Falling back to sequential mode", flush=True)

    use_pool = pool_works and n_workers > 1

    # ── Output paths for incremental saves ─────────────────────────────────
    out_dir = Path(base_cfg.get("output_dir", "market_neutral/backtest_results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ho_csv_path = out_dir / f"pair_scanner_hyperopt_seed{seed}.csv"
    # Remove stale file from previous failed run
    if ho_csv_path.exists():
        ho_csv_path.unlink()

    print(f"  [hyperopt] {trials} trials on {len(sample)} sampled pairs, "
          f"first {n_startup} random → then Bayesian")
    print(f"  [hyperopt] workers={n_workers}  hyperopt_recompute={hyperopt_recompute}")

    history: list[dict] = []
    best_obj_so_far = -999.0
    t_start = time.time()
    _pool_broken = False  # fallback flag if pool crashes

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_obj_so_far, _pool_broken
        cfg = dict(base_cfg)
        # ── NARROWED search space (based on 9 prior runs convergence) ──
        # Core entry/exit
        cfg["z_entry"]     = trial.suggest_float("z_entry",     1.5, 2.2,  step=0.1)    # was 1.4-2.5
        cfg["z_entry_max"] = trial.suggest_float("z_entry_max", cfg["z_entry"] + 0.2, 3.5, step=0.1)
        cfg["z_exit"]      = trial.suggest_float("z_exit",      0.0, 0.30, step=0.05)   # was 0-0.35
        cfg["z_stop"]      = trial.suggest_float("z_stop",      3.5, 5.0,  step=0.5)    # was 2.5-5.0
        # Cointegration & beta filters
        cfg["beta_threshold"]      = trial.suggest_float("beta_threshold",    0.15, 0.25, step=0.05)  # was 0.05-0.25
        cfg["p_value_threshold"]   = trial.suggest_float("p_value_threshold", 0.03, 0.06, step=0.01)  # was 0.01-0.08
        cfg["coint_stability_min_bars"] = trial.suggest_int("coint_stability_min_bars", 2, 5)          # was 1-5
        # Holding period
        cfg["hold_multiplier"] = trial.suggest_float("hold_multiplier", 4.0, 8.0, step=0.5)  # was 2.0-8.0
        cfg["max_hold_days"]   = trial.suggest_float("max_hold_days",   14.0, 40.0, step=1.0) # was 7-60
        # Half-life filter
        cfg["hl_min_days"] = trial.suggest_float("hl_min_days", 0.5, 1.0, step=0.25)     # was 0.5-3.0
        cfg["hl_max_days"] = trial.suggest_float("hl_max_days", cfg["hl_min_days"] + 1.0, 13.0, step=0.5) # was +1-15
        # Grace period (narrowed: 0-4 instead of 0-12)
        cfg["coint_broken_grace_bars"] = trial.suggest_int("coint_broken_grace_bars", 0, 4) # was 0-12
        # Fixed (not optimized): max_notional_pct, circuit_breaker_pct,
        # beta_alert_threshold (Telegram only), beta_critical, hedge_min/max

        params_dict = _cfg_to_params_dict(cfg)
        params_dict["recompute_bars"] = hyperopt_recompute

        # ── Parallel pair evaluation ──────────────────────────────────────
        t_trial = time.time()
        tasks = [(a, b, params_dict) for a, b in sample]

        raw_results = None
        if use_pool and not _pool_broken:
            try:
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_scan_worker_init,
                    initargs=init_args,
                ) as pool:
                    raw_results = list(pool.map(
                        _scan_one_pair, tasks,
                        chunksize=max(1, len(tasks) // n_workers),
                    ))
            except Exception as e:
                print(f"  [hyperopt] ⚠ Pool error: {e}", flush=True)
                print(f"  [hyperopt] ⚠ Falling back to sequential mode", flush=True)
                _pool_broken = True
                raw_results = None

        if raw_results is None:
            # Sequential fallback (or single-core mode)
            raw_results = []
            for t in tasks:
                raw_results.append(_scan_one_pair_direct(
                    market, t[0], t[1], params_dict,
                    recompute_bars=hyperopt_recompute,
                ))

        # ── Aggregate metrics ─────────────────────────────────────────────
        scores = []
        total_trades = 0
        total_pnl = 0.0
        win_rates = []
        sharpes = []
        pfs = []
        for res in raw_results:
            if res and res.get("trades", 0) >= 1:
                s = compute_pair_score(res, min_trades=1)
                scores.append(s)
                total_trades += res["trades"]
                total_pnl    += res.get("total_net_pnl", 0) or 0
                win_rates.append(res.get("win_rate", 0) or 0)
                sharpes.append(res.get("sharpe", 0) or 0)
                pfs.append(res.get("pf", 0) or 0)

        if not scores:
            return -10.0

        avg_score  = float(np.mean(scores))
        avg_wr     = float(np.mean(win_rates))   if win_rates else 0.0
        avg_sharpe = float(np.mean(sharpes))     if sharpes   else 0.0
        avg_pf     = float(np.mean(pfs))         if pfs       else 0.0
        avg_pnl    = total_pnl / len(scores)     if scores    else 0.0
        active_pct = len(scores) / len(sample)

        # ── Objective: balance quality vs coverage ────────────────────────
        # Reject degenerate solutions: need enough pairs & trades for stats
        min_active_pct = 0.10   # at least 10% of sampled pairs must trade
        min_total_trades = 30   # at least 30 trades total
        if active_pct < min_active_pct or total_trades < min_total_trades:
            obj = -5.0 + active_pct  # heavy penalty but still differentiable
        else:
            # Weighted: 50% quality + 30% coverage + 20% trade volume
            coverage_bonus = min(1.0, active_pct / 0.5)   # saturates at 50% active
            trade_bonus = min(1.0, total_trades / 500)     # saturates at 500 trades
            obj = 0.50 * avg_score + 0.30 * coverage_bonus + 0.20 * trade_bonus
        trial_sec  = time.time() - t_trial

        idx = trial.number + 1
        row = {
            "seed": seed,
            "trial": idx, "objective": round(obj, 4),
            "avg_score": round(avg_score, 4),
            "active_pairs": len(scores),
            "total_trades": total_trades,
            "avg_pnl":    round(avg_pnl, 2),
            "avg_win_rate": round(avg_wr, 4),
            "avg_sharpe":  round(avg_sharpe, 4),
            "avg_pf":      round(avg_pf, 4),
            "trial_sec":   round(trial_sec, 1),
        }
        for k in _HYPEROPT_PARAM_KEYS:
            row[k] = cfg[k]
        history.append(row)

        # ── Incremental CSV save (append mode) ────────────────────────────
        pd.DataFrame([row]).to_csv(
            ho_csv_path, mode="a", header=not ho_csv_path.exists() or idx == 1,
            index=False,
        )

        # ── Update best_params JSON if new best ──────────────────────────
        is_best = obj > best_obj_so_far
        if is_best:
            best_obj_so_far = obj
            _save_hyperopt_best_params(history, seed, out_dir)

        # ── Colab: copy to Google Drive after each trial (survives disconnect) ─
        gdrive_dir = Path("/content/drive/MyDrive/hyperopt_results")
        if gdrive_dir.exists():
            import shutil
            shutil.copy2(ho_csv_path, gdrive_dir / ho_csv_path.name)
            bp_src = out_dir / f"best_params_seed{seed}.json"
            if bp_src.exists():
                shutil.copy2(bp_src, gdrive_dir / bp_src.name)

        # ── Rich console line ─────────────────────────────────────────────
        marker = "★" if is_best else " "
        elapsed = time.time() - t_start
        eta = (elapsed / idx) * (trials - idx) if idx > 0 else 0
        print(
            f"  [hyperopt {marker}] {idx:>2}/{trials}"
            f"  obj={obj:+.4f}  score={avg_score:.3f}"
            f"  wr={avg_wr:.1%}  sharpe={avg_sharpe:+.2f}  pf={'inf' if (avg_pf != avg_pf or avg_pf == 0) else f'{avg_pf:.2f}'}"
            f"  pnl/pair={avg_pnl:+,.0f}$"
            f"  active={len(scores)}/{len(sample)}  trades={total_trades}"
            f"  |  z={cfg['z_entry']:.1f}-{cfg['z_entry_max']:.1f}→{cfg['z_exit']:.2f}"
            f"  coint={cfg['coint_stability_min_bars']}b  grace={cfg.get('coint_broken_grace_bars', 0)}b"
            f"  hl={cfg['hl_min_days']:.1f}-{cfg['hl_max_days']:.0f}d"
            f"  [{trial_sec:.0f}s  ETA {eta/60:.0f}m]",
            flush=True,
        )
        return obj

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    _print_hyperopt_summary(history, seed, elapsed)

    # ── Save final files ──────────────────────────────────────────────────
    if history:
        # Overwrite CSV with clean version (append may have duplicated headers)
        pd.DataFrame(history).to_csv(ho_csv_path, index=False)
        print(f"  [hyperopt] History CSV  → {ho_csv_path}")

        _save_hyperopt_best_params(history, seed, out_dir)
        bp_path = out_dir / f"best_params_seed{seed}.json"
        print(f"  [hyperopt] Best params  → {bp_path}")

        # Colab auto-download
        try:
            from google.colab import files as colab_files
            colab_files.download(str(ho_csv_path))
            colab_files.download(str(bp_path))
            print(f"  [hyperopt] Colab download triggered ✓")
        except ImportError:
            pass

        # Apply best params to cfg
        best = max(history, key=lambda r: r.get("objective", -99))
        for k in _HYPEROPT_PARAM_KEYS:
            if k in best:
                base_cfg[k] = best[k]

    return base_cfg, history


# ══════════════════════════════════════════════════════════════════════════════
# Main scanner
# ══════════════════════════════════════════════════════════════════════════════

def run_scanner(
    market: MarketData,
    pairs: list[tuple[str, str]],
    cfg: dict,
    scan_workers: int = 4,
    min_trades: int = 3,
) -> list[dict]:
    """Scan all pairs with SLIDING-WINDOW parallelism (Windows-friendly).

    Architecture:
      - Maintain a bounded pool of pending futures (auto: workers * 3)
      - As each future completes → collect result → immediately submit next pair
      - Workers are NEVER idle (unlike sequential batches where the slowest pair
        blocks the entire batch)
      - Uses concurrent.futures.wait(FIRST_COMPLETED) for zero-waste scheduling

    Memory safety:
      - Never more than `max_inflight` futures in memory (not 53K)
      - gc.collect() every 500 completed pairs
      - Intermediate CSV save every 2000 pairs (crash-safe)
    """
    import gc
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

    t0 = time.perf_counter()
    n = len(pairs)
    params_dict = _cfg_to_params_dict(cfg)
    init_args = _prepare_market_init_args(market)

    max_inflight_cfg = int(cfg.get("max_inflight", 0) or 0)
    max_inflight = max_inflight_cfg if max_inflight_cfg > 0 else max(scan_workers * 3, 20)
    worker_recycle = int(cfg.get("worker_recycle", 80) or 0)

    print(f"\n  [scanner] Scanning {n:,} pairs with {scan_workers} workers "
          f"(max_inflight={max_inflight}, recycle={worker_recycle})")
    print(f"  [scanner] Params:  z_entry={cfg['z_entry']}  z_exit={cfg['z_exit']}  "
          f"z_stop={cfg['z_stop']}  window={cfg.get('window_size', 180)}  "
          f"hold_mult={cfg['hold_multiplier']}  max_hold={cfg['max_hold_days']}d")

    results: list[dict] = []
    done = 0
    errors = 0
    next_idx = 0          # index into pairs[] for next submission
    last_save = 0
    last_gc = 0
    out_dir = Path(cfg.get("output_dir", "market_neutral/backtest_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(
        max_workers=scan_workers,
        initializer=_scan_worker_init,
        initargs=init_args,
        max_tasks_per_child=(worker_recycle if worker_recycle > 0 else None),
    ) as pool:
        pending: set = set()

        # ── Initial fill: submit up to max_inflight pairs ─────────────────────
        while next_idx < n and len(pending) < max_inflight:
            a, b = pairs[next_idx]
            fut = pool.submit(_scan_one_pair, (a, b, params_dict))
            pending.add(fut)
            next_idx += 1

        # ── Main loop: as each completes, refill immediately ──────────────────
        while pending:
            # Wait for at least one future to finish
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)

            for fut in completed:
                done += 1
                try:
                    res = fut.result()
                except Exception:
                    res = None
                    errors += 1

                if res is not None and res["trades"] >= min_trades:
                    res["score"] = compute_pair_score(res, min_trades)
                    results.append(res)

            # Immediately refill pending up to max_inflight
            while next_idx < n and len(pending) < max_inflight:
                a, b = pairs[next_idx]
                fut = pool.submit(_scan_one_pair, (a, b, params_dict))
                pending.add(fut)
                next_idx += 1

            # Progress report every 200 completed pairs
            if done % 200 == 0 or not pending:
                elapsed = time.perf_counter() - t0
                speed = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / speed if speed > 0 else 0
                print(
                    f"  [scanner] {done:>6}/{n}  found={len(results):<5}  "
                    f"inflight={len(pending):<3}  "
                    f"speed={speed:.1f} p/s  elapsed={elapsed:.0f}s  eta={eta:.0f}s"
                    + (f"  err={errors}" if errors else "")
                )

            # gc.collect() every 500 pairs
            if done - last_gc >= 500:
                last_gc = done
                gc.collect()

            # Intermediate crash-safe save every 2000 pairs
            if done - last_save >= 2000:
                last_save = done
                partial_sorted = sorted(
                    results, key=lambda r: r.get("score", -99), reverse=True
                )
                partial_path = out_dir / "pair_scanner_ranking_partial.csv"
                save_ranking_csv(partial_sorted, partial_path)

    results.sort(key=lambda r: r.get("score", -99), reverse=True)
    elapsed = time.perf_counter() - t0
    speed = n / elapsed if elapsed > 0 else 0
    print(f"\n  [scanner] ✓ Done in {elapsed:.0f}s ({speed:.1f} pairs/s) — "
          f"{len(results)} qualifying pairs (from {n:,} tested)")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_ranking_csv(results: list[dict], out_path: Path) -> None:
    rows = []
    for r in results:
        reasons = r.get("close_reasons", {})
        rows.append({
            "pair":        r["pair"],
            "score":       r.get("score", 0),
            "trades":      r["trades"],
            "wins":        r.get("wins", 0),
            "losses":      r.get("losses", 0),
            "win_rate":    round(r["win_rate"], 4),
            "total_pnl":   round(r.get("total_net_pnl", 0), 2),
            "avg_pnl":     round(r["avg_pnl"], 2),
            "median_pnl":  round(r.get("median_pnl", 0), 2),
            "best_trade":  round(r.get("best_trade", 0), 2),
            "worst_trade": round(r.get("worst_trade", 0), 2),
            "sharpe":      round(r["sharpe"], 4),
            "sortino":     round(r.get("sortino", 0), 4),
            "max_dd":      round(r["max_dd"], 4),
            "pf":          round(r["pf"], 4),
            "avg_hold_h":  round(r["avg_hold_h"], 1),
            "return_pct":  round(r.get("total_return_pct", 0), 2),
            "avg_mae_pct": round(r.get("avg_mae_pct", 0), 4),
            "avg_mfe_pct": round(r.get("avg_mfe_pct", 0), 4),
            "z_tp":        reasons.get("z_tp", 0),
            "z_sl":        reasons.get("z_sl", 0),
            "broken_coint": reasons.get("broken_coint", 0),
            "time_exit":   reasons.get("time_exit", 0),
            "beta_drift":  reasons.get("beta_drift", 0),
            "circuit":     reasons.get("circuit", 0),
            "beta_critical": reasons.get("beta_critical", 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  [scanner] Full ranking saved → {out_path}")


def save_best_pairs_json(
    results: list[dict], top_n: int, out_path: Path, params_used: dict
) -> None:
    """Save top pairs with exact strategy parameters used in the scan."""
    entries = []
    for r in results[:top_n]:
        reasons = r.get("close_reasons", {})
        entries.append({
            "pair":        r["pair"],
            "score":       r.get("score", 0),
            "trade_count": r["trades"],
            "total_pnl":   round(r.get("total_net_pnl", 0), 8),
            "avg_pnl":     round(r["avg_pnl"], 8),
            "std_pnl":     0,
            "win_rate":    round(r["win_rate"], 4),
            "trade_sharpe": round(r["sharpe"], 6),
            "profit_factor": round(r.get("pf", 0), 4),
            "max_drawdown": round(r.get("max_dd", 0), 4),
            "avg_hold_hours": round(r.get("avg_hold_h", 0), 1),
            "close_reasons": reasons,
            "source":      "pair_scanner",
        })

    output = {
        "params_used": {
            "z_entry":               params_used.get("z_entry"),
            "z_entry_max":           params_used.get("z_entry_max"),
            "z_exit":                params_used.get("z_exit"),
            "z_stop":                params_used.get("z_stop"),
            "window_size":           params_used.get("window_size"),
            "max_notional_pct":      params_used.get("max_notional_pct"),
            "capital":               params_used.get("capital"),
            "commission_rate":       params_used.get("commission_rate"),
            "slippage_rate":         params_used.get("slippage_rate"),
            "p_value_threshold":     params_used.get("p_value_threshold"),
            "hedge_min":             params_used.get("hedge_min"),
            "hedge_max":             params_used.get("hedge_max"),
            "beta_threshold":        params_used.get("beta_threshold"),
            "beta_alert_threshold":  params_used.get("beta_alert_threshold"),
            "beta_critical":         params_used.get("beta_critical"),
            "circuit_breaker_pct":   params_used.get("circuit_breaker_pct"),
            "coint_stability_min_bars": params_used.get("coint_stability_min_bars"),
            "coint_broken_grace_bars": params_used.get("coint_broken_grace_bars"),
            "hold_multiplier":       params_used.get("hold_multiplier"),
            "max_hold_days":         params_used.get("max_hold_days"),
            "hl_min_days":           params_used.get("hl_min_days"),
            "hl_max_days":           params_used.get("hl_max_days"),
            "timeframe":             params_used.get("timeframe"),
        },
        "scan_info": {
            "total_pairs_scanned":   params_used.get("_total_scanned", 0),
            "qualifying_pairs":      len(entries),
            "min_trades":            params_used.get("min_trades", 3),
            "scan_date":             time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "pairs": entries,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  [scanner] Top {min(top_n, len(results))} pairs → {out_path}")


def print_summary(results: list[dict], top_n: int = 25) -> None:
    print()
    print("=" * 110)
    print(f"  PAIR SCANNER RESULTS  (total qualifying pairs: {len(results)})")
    print("=" * 110)
    hdr = (f"  {'Pair':<30} {'Score':>7} {'Trades':>6} {'WR':>5} {'Sharpe':>7} "
           f"{'TotalPnL':>11} {'MaxDD':>8} {'PF':>5} {'TP%':>5} {'Hold':>6}")
    print(hdr)
    print("-" * 110)
    for r in results[:top_n]:
        reasons = r.get("close_reasons", {})
        total_exits = sum(reasons.values()) if reasons else 1
        tp_pct = reasons.get("z_tp", 0) / total_exits * 100
        mdd_str = f"{r['max_dd']*100:.1f}%" if abs(r['max_dd']) < 1 else f"{r['max_dd']:.2f}"
        print(
            f"  {r['pair']:<30} {r.get('score', 0):>7.4f} {r['trades']:>6} "
            f"{r['win_rate']:>5.1%} {r['sharpe']:>7.3f} "
            f"{r.get('total_net_pnl', 0):>11,.0f} {mdd_str:>8} {r['pf']:>5.2f} "
            f"{tp_pct:>4.0f}% {r['avg_hold_h']:>5.1f}h"
        )
    print("=" * 110)

    if len(results) > top_n:
        print(f"\n  BOTTOM 5 (worst qualifying pairs):")
        print("-" * 110)
        for r in results[-5:]:
            print(
                f"  {r['pair']:<30} {r.get('score', 0):>7.4f} {r['trades']:>6} "
                f"{r['win_rate']:>5.1%} {r['sharpe']:>7.3f} "
                f"{r.get('total_net_pnl', 0):>11,.0f}"
            )

    all_pnls = [r.get("total_net_pnl", 0) for r in results]
    all_trades = [r["trades"] for r in results]
    profitable = sum(1 for p in all_pnls if p > 0)

    # Exit reason aggregate
    agg_reasons: dict[str, int] = {}
    for r in results:
        for reason, cnt in r.get("close_reasons", {}).items():
            agg_reasons[reason] = agg_reasons.get(reason, 0) + cnt
    total_all_exits = sum(agg_reasons.values()) if agg_reasons else 1

    print(f"\n  Aggregate Statistics:")
    print(f"    Total P&L (all pairs):    ${sum(all_pnls):,.0f}")
    print(f"    Profitable / Total:       {profitable}/{len(results)}")
    print(f"    Avg Sharpe:               {np.mean([r['sharpe'] for r in results]):.3f}")
    print(f"    Avg Win-rate:             {np.mean([r['win_rate'] for r in results]):.1%}")
    print(f"    Avg Trades/pair:          {np.mean(all_trades):.1f}")
    print(f"    Total trades (all):       {sum(all_trades):,}")
    print(f"    Exit reasons (all):       ", end="")
    for reason, cnt in sorted(agg_reasons.items(), key=lambda x: -x[1]):
        print(f"{reason}={cnt}({cnt/total_all_exits*100:.0f}%) ", end="")
    print()
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-fidelity per-pair scanner using real BotParityBacktester engine"
    )
    p.add_argument("--input",          default=None,    help="Klines CSV path (auto-detect if not set)")
    p.add_argument("--from-pairs",     default=None,
                   help="Scan only pairs from this JSON (default: ALL combos)")
    p.add_argument("--top",            type=int, default=None, help="Top N pairs to export")
    p.add_argument("--min-trades",     type=int, default=None, help="Min trades to qualify")
    p.add_argument("--scan-workers",   type=int, default=None, help="Parallel worker processes")
    p.add_argument("--max-inflight",   type=int, default=None, help="Max pending futures (0=auto)")
    p.add_argument("--worker-recycle", type=int, default=None,
                   help="Max tasks per child process before recycle (0=disable)")
    p.add_argument("--output-dir",     default=None)

    # Strategy params (override defaults)
    p.add_argument("--z-entry",        type=float, default=None)
    p.add_argument("--z-exit",         type=float, default=None)
    p.add_argument("--z-stop",         type=float, default=None)
    p.add_argument("--z-entry-max",    type=float, default=None)
    p.add_argument("--window",         type=int,   default=None)
    p.add_argument("--max-hold-days",  type=float, default=None)
    p.add_argument("--capital",        type=float, default=None)
    p.add_argument("--notional-pct",   type=float, default=None)
    p.add_argument("--hold-multiplier", type=float, default=None)
    p.add_argument("--coint-min-bars", type=int,   default=None)
    p.add_argument("--coint-broken-grace-bars", type=int, default=None)
    p.add_argument("--recompute",      type=int,   default=1,
                   help="Recalculate engine cointegration every N bars (e.g. 6 = 24h). Default 1.")

    # Hyperopt
    p.add_argument("--hyperopt-trials",   type=int, default=0, help="0=no hyperopt")
    p.add_argument("--hyperopt-startup",  type=int, default=3)
    p.add_argument("--hyperopt-sample",   type=int, default=30,
                   help="Number of pairs to sample for hyperopt")
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--hyperopt-only",     action="store_true",
                   help="Run hyperopt only, skip full scan. For Colab/distributed hyperopt.")

    # Sharding — split work across multiple machines
    p.add_argument("--shard",             default=None,
                   help="Shard spec: K/N  (e.g. 1/4 = shard 1 of 4)")
    p.add_argument("--pair-range",        default=None,
                   help="Explicit pair index range: START-END  (e.g. 0-13325)")
    p.add_argument("--merge-shards",      action="store_true",
                   help="Merge all shard CSVs from output-dir into final results")

    return p.parse_args()


def merge_shards(out_dir: Path, min_trades: int = 3, top_n: int = 9000) -> None:
    """Merge all shard CSVs from output directory into final results.

    Usage:
      python backtest_pair_scanner.py --merge-shards --output-dir market_neutral/backtest_results

    Reads all pair_scanner_shard_*.csv files, combines, re-scores, ranks,
    and exports final pair_scanner_ranking.csv + best_pairs_backtest.json.
    """
    shard_files = sorted(out_dir.glob("pair_scanner_shard_*.csv"))
    range_files = sorted(out_dir.glob("pair_scanner_range_*.csv"))
    files_to_merge = shard_files + range_files
    if not files_to_merge:
        print(f"[ERROR] No shard/range files found in {out_dir}")
        return

    print(f"\n  [merge] Found {len(files_to_merge)} files (shard={len(shard_files)}, range={len(range_files)}):")
    dfs = []
    for f in files_to_merge:
        df = pd.read_csv(f)
        print(f"    {f.name}: {len(df)} pairs")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    # Remove duplicates (shouldn't exist, but just in case)
    merged = merged.drop_duplicates(subset="pair", keep="first")
    merged = merged[merged["trades"] >= min_trades]
    merged = merged.sort_values("score", ascending=False).reset_index(drop=True)

    print(f"\n  [merge] Total: {len(merged)} qualifying pairs")

    # Save merged ranking
    ranking_path = out_dir / "pair_scanner_ranking.csv"
    merged.to_csv(ranking_path, index=False)
    print(f"  [merge] Full ranking → {ranking_path}")

    # Build best_pairs_backtest.json from merged data
    entries = []
    for _, r in merged.head(top_n).iterrows():
        entries.append({
            "pair":         r["pair"],
            "score":        float(r.get("score", 0)),
            "trade_count":  int(r["trades"]),
            "total_pnl":    round(float(r.get("total_pnl", 0)), 8),
            "avg_pnl":      round(float(r.get("avg_pnl", 0)), 8),
            "std_pnl":      0,
            "win_rate":     round(float(r.get("win_rate", 0)), 4),
            "trade_sharpe": round(float(r.get("sharpe", 0)), 6),
            "profit_factor": round(float(r.get("pf", 0)), 4),
            "max_drawdown": round(float(r.get("max_dd", 0)), 4),
            "avg_hold_hours": round(float(r.get("avg_hold_h", 0)), 1),
            "source":       "pair_scanner",
        })

    output = {
        "scan_info": {
            "merged_shards":     len(shard_files),
            "merged_ranges":     len(range_files),
            "merged_files":      len(files_to_merge),
            "total_qualifying":  len(merged),
            "exported_pairs":    len(entries),
            "min_trades":        min_trades,
            "merge_date":        time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "pairs": entries,
    }
    best_path = out_dir / "best_pairs_backtest.json"
    with open(best_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  [merge] Top {len(entries)} pairs → {best_path}")

    # Print top 20
    print(f"\n  Top 20 merged pairs:")
    print(f"  {'Pair':<30} {'Score':>7} {'Trades':>6} {'WR':>5} {'Sharpe':>7} {'PnL':>11}")
    print("  " + "-" * 75)
    for _, r in merged.head(20).iterrows():
        print(f"  {r['pair']:<30} {r['score']:>7.4f} {int(r['trades']):>6} "
              f"{r['win_rate']:>5.1%} {r['sharpe']:>7.3f} {r.get('total_pnl', 0):>11,.0f}")
    print()


def main() -> None:
    args = parse_args()
    cfg = dict(DEFAULT_PARAMS)

    # ── Apply CLI overrides (всегда перекрывают DEFAULT_PARAMS) ──────────────
    if args.z_entry is not None:        cfg["z_entry"] = args.z_entry
    if args.z_exit is not None:         cfg["z_exit"] = args.z_exit
    if args.z_stop is not None:         cfg["z_stop"] = args.z_stop
    if args.z_entry_max is not None:    cfg["z_entry_max"] = args.z_entry_max
    if args.window is not None:         cfg["window_size"] = args.window
    if args.max_hold_days is not None:  cfg["max_hold_days"] = args.max_hold_days
    if args.capital is not None:        cfg["capital"] = args.capital
    if args.notional_pct is not None:   cfg["max_notional_pct"] = args.notional_pct
    if args.output_dir is not None:     cfg["output_dir"] = args.output_dir
    if args.hold_multiplier is not None: cfg["hold_multiplier"] = args.hold_multiplier
    if args.coint_min_bars is not None: cfg["coint_stability_min_bars"] = args.coint_min_bars
    if args.coint_broken_grace_bars is not None:
        cfg["coint_broken_grace_bars"] = args.coint_broken_grace_bars
    if args.min_trades is not None:     cfg["min_trades"] = args.min_trades
    if args.scan_workers is not None:   cfg["scan_workers"] = args.scan_workers
    if args.max_inflight is not None:   cfg["max_inflight"] = args.max_inflight
    if args.worker_recycle is not None: cfg["worker_recycle"] = args.worker_recycle
    if args.top is not None:            cfg["top_n"] = args.top
    # recompute: CLI --recompute перекрывает DEFAULT_PARAMS["recompute"]
    if args.recompute != 1:
        cfg["recompute_bars"] = args.recompute
    else:
        cfg["recompute_bars"] = int(cfg.get("recompute", 1))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Merge mode ─────────────────────────────────────────────────────────────
    if args.merge_shards:
        merge_shards(out_dir, min_trades=int(cfg["min_trades"]), top_n=int(cfg["top_n"]))
        return

    # ── Auto-detect klines file ────────────────────────────────────────────────
    data_dir = "market_neutral"
    # Priority: CLI --input > DEFAULT_PARAMS input_file > auto-detect
    if args.input:
        klines_path = args.input
    elif cfg.get("input_file"):
        klines_path = cfg["input_file"]
    else:
        tf = cfg.get("timeframe", "4h")
        candidates_f = [f for f in os.listdir(data_dir)
                        if f.startswith("klines_data_") and f"_{tf}_" in f and f.endswith(".csv")]
        if candidates_f:
            klines_path = os.path.join(data_dir, candidates_f[0])
        else:
            klines_path = os.path.join(data_dir, "klines_data_4h_clean_2024.05.24_2025.10.24.csv")

    print("=" * 80)
    print("  PAIR SCANNER — Full-fidelity BotParityBacktester per pair")
    print("=" * 80)

    # ── Load market data ───────────────────────────────────────────────────────
    market = load_klines_market_data(klines_path, max_symbols=int(cfg["max_symbols"]))
    print(f"  Data:     {len(market.symbols)} symbols × {len(market.dates)} bars")
    print(f"  Source:   {klines_path}")

    # ── Build pair list ────────────────────────────────────────────────────────
    # Priority: CLI --from-pairs > DEFAULT_PARAMS from_pairs > all combos
    from_pairs_path = args.from_pairs or cfg.get("from_pairs")
    if from_pairs_path:
        raw_candidates = load_candidate_pairs(from_pairs_path, market.symbols, limit=99999)
        pairs = [(c.symbol1, c.symbol2) for c in raw_candidates]
        print(f"  Mode:     FROM JSON — {len(pairs):,} pairs from {from_pairs_path}")
    else:
        sym_set = [s for s in market.symbols if s != "BTCUSDT"]
        pairs = list(combinations(sym_set, 2))
        print(f"  Mode:     ALL PAIRS — {len(pairs):,} combinations from {len(sym_set)} symbols")

    if not pairs:
        print("[ERROR] No pairs to scan!")
        return

    total_all_pairs = len(pairs)
    all_pairs = pairs  # full list for hyperopt (BEFORE pair-range filter)

    # ── Apply shard / pair-range ────────────────────────────────────────────────
    shard_id = 0
    n_shards = 1
    shard_label = ""  # for filename

    # Priority: CLI --pair-range > DEFAULT_PARAMS pair_range
    effective_pair_range = args.pair_range or cfg.get("pair_range")

    if effective_pair_range:
        try:
            parts = effective_pair_range.split("-")
            start = int(parts[0])
            end = int(parts[1])
            assert 0 <= start < end <= total_all_pairs, \
                f"Range [{start}..{end}) out of bounds [0..{total_all_pairs})"
        except Exception as e:
            print(f"[ERROR] Invalid pair-range '{effective_pair_range}'. Use START-END, e.g. 27500-53301")
            print(f"        {e}")
            return
        pairs = pairs[start:end]
        n_shards = 2  # flag for shard-aware output
        shard_label = f"range_{start}_{end}"
        print(f"  Range:    [{start:,}..{end:,}) = {len(pairs):,} pairs (of {total_all_pairs:,} total)")

    elif args.shard:
        try:
            parts = args.shard.split("/")
            shard_id = int(parts[0]) - 1   # 0-indexed internally
            n_shards = int(parts[1])
            assert 0 <= shard_id < n_shards, f"Shard {shard_id+1} out of range 1..{n_shards}"
        except Exception as e:
            print(f"[ERROR] Invalid --shard format '{args.shard}'. Use K/N, e.g. 1/4")
            print(f"        {e}")
            return

        chunk_size = math.ceil(total_all_pairs / n_shards)
        start = shard_id * chunk_size
        end = min(start + chunk_size, total_all_pairs)
        pairs = pairs[start:end]
        shard_label = f"shard_{shard_id+1}_of_{n_shards}"
        print(f"  Shard:    {shard_id+1}/{n_shards} — pairs [{start:,}..{end:,}) = {len(pairs):,} pairs")

    else:
        # No shard: print pair ranges for user info (useful for Colab planning)
        if total_all_pairs > 5000:
            n_suggested = 4
            chunk = math.ceil(total_all_pairs / n_suggested)
            print(f"\n  💡 To split across {n_suggested} Colabs, use --pair-range:")
            for i in range(n_suggested):
                s = i * chunk
                e = min(s + chunk, total_all_pairs)
                print(f"      Colab {i+1}: --pair-range {s}-{e}    ({e-s:,} pairs)")

    # Print config
    print(f"\n  Strategy parameters:")
    for k in ["z_entry", "z_entry_max", "z_exit", "z_stop", "window_size",
              "max_notional_pct", "capital", "hold_multiplier", "max_hold_days",
              "p_value_threshold", "hedge_min", "hedge_max", "beta_threshold",
              "beta_critical", "coint_stability_min_bars", "coint_broken_grace_bars",
              "hl_min_days", "hl_max_days",
              "commission_rate", "slippage_rate", "circuit_breaker_pct"]:
        print(f"    {k:<30} = {cfg[k]}")
    print()

    # ── Hyperopt (optional) ────────────────────────────────────────────────────
    hyperopt_history: list[dict] = []
    if args.hyperopt_trials > 0:
        print(f"  Running hyperopt: {args.hyperopt_trials} trials …\n")
        cfg, hyperopt_history = run_scanner_hyperopt(
            market=market,
            pairs=all_pairs,   # use FULL pair list, not pair-range filtered
            base_cfg=cfg,
            trials=args.hyperopt_trials,
            n_startup=args.hyperopt_startup,
            seed=args.seed,
            sample_pairs=args.hyperopt_sample,
        )
        print(f"\n  Updated parameters after hyperopt:")
        for k in _HYPEROPT_PARAM_KEYS:
            print(f"    {k:<30} = {cfg[k]}")
        print()

        if args.hyperopt_only:
            print("  [hyperopt-only] Skipping full scan. Done.")
            return

    # ── Run scanner ────────────────────────────────────────────────────────────
    scan_workers = max(1, int(cfg["scan_workers"]))
    min_trades = int(cfg["min_trades"])
    top_n = int(cfg["top_n"])
    results = run_scanner(
        market=market,
        pairs=pairs,
        cfg=cfg,
        scan_workers=scan_workers,
        min_trades=min_trades,
    )

    # ── Save results ───────────────────────────────────────────────────────────
    # Shard-aware filenames
    if shard_label:
        ranking_path = out_dir / f"pair_scanner_{shard_label}.csv"
    else:
        ranking_path = out_dir / "pair_scanner_ranking.csv"
    save_ranking_csv(results, ranking_path)

    # Best pairs JSON (only in non-shard mode)
    if not shard_label:
        cfg["_total_scanned"] = total_all_pairs
        best_path = out_dir / "best_pairs_backtest.json"
        save_best_pairs_json(results, top_n, best_path, params_used=cfg)
    else:
        print(f"\n  [scanner] Shard {shard_id+1}/{n_shards} complete!")
        script_name = Path(__file__).name
        print(f"  [scanner] When ALL shards done, merge with:")
        print(f"    python {script_name} --merge-shards")

    # ── Print summary ──────────────────────────────────────────────────────────
    print_summary(results, top_n=25)


if __name__ == "__main__":
    main()
