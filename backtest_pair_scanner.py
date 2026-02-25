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
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Import the real engine from backtest_bot_parity
from backtest_bot_parity import (
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
    "z_entry":         1.5,
    "z_entry_max":     3.0,
    "z_exit":          0.0,
    "z_stop":          4.0,

    # ── Costs ─────────────────────────────────────────────────────────────────
    "commission_rate":  0.0004,
    "slippage_rate":    0.0005,

    # ── Cointegration quality ─────────────────────────────────────────────────
    "p_value_threshold":  0.08,  # coarse scan: wider coint filter
    "hedge_min":          0.25,
    "hedge_max":          4.0,

    # ── Beta (BTC neutrality) ─────────────────────────────────────────────────
    "beta_threshold":       0.18,
    "beta_alert_threshold": 0.40,
    "beta_critical":        1.2,

    # ── Circuit breaker ───────────────────────────────────────────────────────
    "circuit_breaker_pct":  0.50,

    # ── Entry confirmation & hold ─────────────────────────────────────────────
    "signal_confirm_sec":    10,
    "coint_stability_min_bars": 0,
    "entry_et_target_abs_z": 0.5,
    "hold_multiplier":       3.0,
    "max_hold_days":         12.0,

    # ── Cooldowns ─────────────────────────────────────────────────────────────
    "sl_reentry_cooldown_sec":  0,
    "close_retry_cooldown_sec": 30,

    # ── Half-life bounds ──────────────────────────────────────────────────────
    "hl_min_days": 0.25,
    "hl_max_days": 8.0,

    # ── Window ────────────────────────────────────────────────────────────────
    "window_size":         180,
    "max_symbols":         450,

    # ── Scanner-specific ──────────────────────────────────────────────────────
    #
    # ⭐⭐ МЕНЯЙ ПЕРЕД КАЖДЫМ ЗАПУСКОМ ⭐⭐
    #
    # Диапазон пар: None=весь рынок (0..53301),  или строка "27500-53301"
    # Твой ПК (40%): "27500-53301"  |  MacBook M2 (60%): None
    "pair_range":          "27500-28500", #"27500-53301",

    # Воркеры: Windows=12, MacBook M2=8, Colab=2
    "scan_workers":        8,
    # Keep queue depth moderate to reduce IPC pressure on long scans.
    "scan_inflight_per_worker": 3,
    # Recycle worker process after N pairs to avoid long-run memory bloat (0=off).
    "scan_max_tasks_per_child": 80,

    # Точность: 1=максимальная (медленно), 6=быстрый фильтр (~90% точности)
    "recompute":           6,

    # Шаг 2 (точный скан по лучшим парам из грубого фильтра):
    # None = сканировать все пары,  или путь к best_pairs_backtest.json
    "from_pairs":          None,
    # Пример: "market_neutral/backtest_results/best_pairs_backtest.json"

    # CSV котировок: None = автопоиск в market_neutral/
    "input_file":          None,

    "min_trades":          1,     # min trades to qualify
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
        "z_entry":                    float(cfg.get("z_entry", 1.5)),
        "z_entry_max":                float(cfg.get("z_entry_max", 3.0)),
        "z_exit":                     float(cfg.get("z_exit", 0.0)),
        "z_stop":                     float(cfg.get("z_stop", 4.0)),
        "commission_rate":            float(cfg.get("commission_rate", 0.0004)),
        "slippage_rate":              float(cfg.get("slippage_rate", 0.0005)),
        "p_value_threshold":          float(cfg.get("p_value_threshold", 0.08)),
        "hedge_min":                  float(cfg.get("hedge_min", 0.25)),
        "hedge_max":                  float(cfg.get("hedge_max", 4.0)),
        "beta_threshold":             float(cfg.get("beta_threshold", 0.18)),
        "beta_alert_threshold":       float(cfg.get("beta_alert_threshold", 0.40)),
        "beta_critical":              float(cfg.get("beta_critical", 1.2)),
        "circuit_breaker_pct":        float(cfg.get("circuit_breaker_pct", 0.50)),
        "signal_confirm_sec":         int(cfg.get("signal_confirm_sec", 10)),
        "coint_stability_min_bars":   int(cfg.get("coint_stability_min_bars", 0)),
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
        "hl_max_days":                float(cfg.get("hl_max_days", 8.0)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Per-pair scan — worker globals & init (Windows-safe serialization)
# ══════════════════════════════════════════════════════════════════════════════

# Worker-local market data (populated once via initializer)
_SCAN_MARKET: MarketData | None = None
_SCAN_TASKS_DONE: int = 0


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
    global _SCAN_MARKET, _SCAN_TASKS_DONE
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
    _SCAN_TASKS_DONE = 0


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
    global _SCAN_TASKS_DONE
    sym_a, sym_b, params_dict, recompute_bars = args
    market = _SCAN_MARKET
    if market is None:
        return None
    if sym_a not in market.symbol_to_idx or sym_b not in market.symbol_to_idx:
        return None
    try:
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
        m = dict(result.metrics)
        # Scanner only needs metrics; release heavy tables ASAP.
        del result
        del bt

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
    finally:
        _SCAN_TASKS_DONE += 1
        # Periodic GC inside worker limits long-run RSS growth.
        if _SCAN_TASKS_DONE % 25 == 0:
            import gc
            gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# Direct single-pair scan (no subprocess — for hyperopt)
# ══════════════════════════════════════════════════════════════════════════════

def _scan_one_pair_direct(
    market: MarketData, sym_a: str, sym_b: str, params_dict: dict
) -> dict | None:
    """Run a single pair backtest directly (no subprocess)."""
    if sym_a not in market.symbol_to_idx or sym_b not in market.symbol_to_idx:
        return None
    try:
        params = BacktestParams(**params_dict)
        candidate = CandidatePair(symbol1=sym_a, symbol2=sym_b, score=1.0, source="scanner")
        bt = BotParityBacktester(
            market=market, params=params, candidates=[candidate], n_workers=1, exact_mode=True,
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

    For each trial: run sampled pairs with candidate params, objective = avg score.
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

    print(f"  [hyperopt] {trials} trials on {len(sample)} sampled pairs, "
          f"first {n_startup} random -> then Bayesian")

    history: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        cfg = dict(base_cfg)
        # Discretized search space (same as backtest_bot_parity)
        cfg["z_entry"]     = trial.suggest_float("z_entry",     1.4, 2.5,  step=0.1)
        cfg["z_entry_max"] = trial.suggest_float("z_entry_max", cfg["z_entry"] + 0.2, 3.5, step=0.1)
        cfg["z_exit"]      = trial.suggest_float("z_exit",      0.0, 0.35, step=0.05)
        cfg["z_stop"]      = trial.suggest_float("z_stop",      2.5, 5.0,  step=0.5)
        cfg["max_notional_pct"]    = trial.suggest_float("max_notional_pct",    0.10, 0.55, step=0.05)
        cfg["circuit_breaker_pct"] = trial.suggest_float("circuit_breaker_pct", 0.15, 0.70, step=0.05)
        cfg["beta_threshold"]      = trial.suggest_float("beta_threshold",      0.05, 0.25, step=0.05)
        cfg["beta_alert_threshold"] = trial.suggest_float(
            "beta_alert_threshold", cfg["beta_threshold"] + 0.05, 0.50, step=0.05
        )
        cfg["beta_critical"]    = trial.suggest_float("beta_critical", 0.5, 1.5, step=0.25)
        cfg["p_value_threshold"] = trial.suggest_float("p_value_threshold", 0.01, 0.08, step=0.01)
        cfg["coint_stability_min_bars"] = trial.suggest_int("coint_stability_min_bars", 1, 8)
        cfg["hold_multiplier"] = trial.suggest_float("hold_multiplier", 1.5, 6.0, step=0.5)
        cfg["max_hold_days"]   = trial.suggest_float("max_hold_days",   5.0, 45.0, step=1.0)
        cfg["hedge_min"] = trial.suggest_float("hedge_min", 0.10, 0.60, step=0.05)
        cfg["hedge_max"] = trial.suggest_float("hedge_max", 1.5,  5.0,  step=0.5)
        cfg["hl_min_days"] = trial.suggest_float("hl_min_days", 0.1, 1.0, step=0.1)
        cfg["hl_max_days"] = trial.suggest_float("hl_max_days", cfg["hl_min_days"] + 0.5, 5.0, step=0.5)

        params_dict = _cfg_to_params_dict(cfg)

        scores = []
        total_trades = 0
        for sym_a, sym_b in sample:
            res = _scan_one_pair_direct(market, sym_a, sym_b, params_dict)
            if res and res["trades"] >= 1:
                s = compute_pair_score(res, min_trades=1)
                scores.append(s)
                total_trades += res["trades"]

        if not scores:
            return -10.0

        avg_score = float(np.mean(scores))
        active_pct = len(scores) / len(sample)
        obj = avg_score + 0.2 * active_pct

        idx = trial.number + 1
        row = {
            "trial": idx, "objective": round(obj, 4),
            "avg_score": round(avg_score, 4),
            "active_pairs": len(scores),
            "total_trades": total_trades,
        }
        # Save all tuned params
        for k in ["z_entry", "z_entry_max", "z_exit", "z_stop",
                   "max_notional_pct", "circuit_breaker_pct", "beta_threshold",
                   "beta_alert_threshold", "beta_critical",
                   "coint_stability_min_bars", "hold_multiplier", "max_hold_days",
                   "hedge_min", "hedge_max", "hl_min_days", "hl_max_days",
                   "p_value_threshold"]:
            row[k] = cfg[k]
        history.append(row)
        print(f"  [hyperopt] {idx}/{trials}  obj={obj:.4f}  "
              f"avg_score={avg_score:.3f}  active={len(scores)}/{len(sample)}  "
              f"tot_trades={total_trades}")
        return obj

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    if history:
        best = max(history, key=lambda r: r.get("objective", -99))
        print(f"\n  [hyperopt] Best trial #{best['trial']}  obj={best['objective']:.4f}")
        for k in ["z_entry", "z_entry_max", "z_exit", "z_stop",
                   "max_notional_pct", "circuit_breaker_pct", "beta_threshold",
                   "beta_alert_threshold", "beta_critical",
                   "coint_stability_min_bars", "hold_multiplier", "max_hold_days",
                   "hedge_min", "hedge_max", "hl_min_days", "hl_max_days",
                   "p_value_threshold"]:
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
      - Maintain a bounded pending queue (`workers * inflight_per_worker`)
      - As each future completes → collect result → immediately submit next pair
      - Workers are NEVER idle (unlike sequential batches where the slowest pair
        blocks the entire batch)
      - Uses concurrent.futures.wait(FIRST_COMPLETED) for zero-waste scheduling

    Memory safety:
      - Never more than `max_inflight` futures in memory (not 53K)
      - gc.collect() every 500 completed pairs
      - Intermediate CSV save every 1000 pairs (crash-safe)
    """
    import gc
    import inspect
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

    t0 = time.perf_counter()
    n = len(pairs)
    params_dict = _cfg_to_params_dict(cfg)
    recompute_bars = max(1, int(cfg.get("recompute_bars", cfg.get("recompute", 1))))
    init_args = _prepare_market_init_args(market)

    inflight_per_worker = max(1, int(cfg.get("scan_inflight_per_worker", 4)))
    max_inflight = max(scan_workers * inflight_per_worker, scan_workers + 2)
    max_tasks_per_child_cfg = int(cfg.get("scan_max_tasks_per_child", 0) or 0)
    max_tasks_per_child = max_tasks_per_child_cfg if max_tasks_per_child_cfg > 0 else None

    print(f"\n  [scanner] Scanning {n:,} pairs with {scan_workers} workers "
          f"(max_inflight={max_inflight}, recycle={max_tasks_per_child or 'off'})")
    print(f"  [scanner] Params:  z_entry={cfg['z_entry']}  z_exit={cfg['z_exit']}  "
          f"z_stop={cfg['z_stop']}  window={cfg.get('window_size', 180)}  "
          f"hold_mult={cfg['hold_multiplier']}  max_hold={cfg['max_hold_days']}d  "
          f"recompute={recompute_bars}")

    results: list[dict] = []
    done = 0
    errors = 0
    next_idx = 0          # index into pairs[] for next submission
    last_save = 0
    last_gc = 0
    out_dir = Path(cfg.get("output_dir", "market_neutral/backtest_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_kwargs = {
        "max_workers": scan_workers,
        "initializer": _scan_worker_init,
        "initargs": init_args,
    }
    if "max_tasks_per_child" in inspect.signature(ProcessPoolExecutor).parameters:
        pool_kwargs["max_tasks_per_child"] = max_tasks_per_child
    elif max_tasks_per_child is not None:
        print("  [scanner] NOTE: max_tasks_per_child unsupported in this Python, ignoring recycle.")

    with ProcessPoolExecutor(**pool_kwargs) as pool:
        pending: set = set()

        # ── Initial fill: submit up to max_inflight pairs ─────────────────────
        while next_idx < n and len(pending) < max_inflight:
            a, b = pairs[next_idx]
            fut = pool.submit(_scan_one_pair, (a, b, params_dict, recompute_bars))
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
                fut = pool.submit(_scan_one_pair, (a, b, params_dict, recompute_bars))
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

            # Intermediate crash-safe save every 1000 pairs
            if done - last_save >= 1000:
                last_save = done
                partial_sorted = sorted(
                    results, key=lambda r: r.get("score", -99), reverse=True
                )
                partial_path = out_dir / "pair_scanner_ranking_partial.csv"
                save_ranking_csv(partial_sorted, partial_path)

    results.sort(key=lambda r: r.get("score", -99), reverse=True)
    elapsed = time.perf_counter() - t0
    speed = n / elapsed if elapsed > 0 else 0
    print(f"\n  [scanner] Done in {elapsed:.0f}s ({speed:.1f} pairs/s) - "
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
    print(f"  [scanner] Full ranking saved -> {out_path}")


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
            "hold_multiplier":       params_used.get("hold_multiplier"),
            "max_hold_days":         params_used.get("max_hold_days"),
            "hl_min_days":           params_used.get("hl_min_days"),
            "hl_max_days":           params_used.get("hl_max_days"),
            "timeframe":             params_used.get("timeframe"),
        },
        "scan_info": {
            "total_pairs_scanned":   params_used.get("_total_scanned", 0),
            "qualifying_pairs":      len(entries),
            "min_trades":            params_used.get("min_trades", 1),
            "scan_date":             time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "pairs": entries,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  [scanner] Top {min(top_n, len(results))} pairs -> {out_path}")


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
    p.add_argument("--scan-inflight-per-worker", type=int, default=None,
                   help="Pending tasks per worker (lower = less RAM pressure)")
    p.add_argument("--scan-max-tasks-per-child", type=int, default=None,
                   help="Recycle worker after N pairs (0 disables recycle)")
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
    p.add_argument("--recompute",      type=int,   default=None,
                   help="Recalculate engine cointegration every N bars (e.g. 6 = 24h).")

    # Hyperopt
    p.add_argument("--hyperopt-trials",   type=int, default=0, help="0=no hyperopt")
    p.add_argument("--hyperopt-startup",  type=int, default=3)
    p.add_argument("--hyperopt-sample",   type=int, default=30,
                   help="Number of pairs to sample for hyperopt")
    p.add_argument("--seed",              type=int, default=42)

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
    if not shard_files:
        print(f"[ERROR] No shard files found in {out_dir}")
        return

    print(f"\n  [merge] Found {len(shard_files)} shard files:")
    dfs = []
    for f in shard_files:
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
    print(f"  [merge] Full ranking -> {ranking_path}")

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
    print(f"  [merge] Top {len(entries)} pairs -> {best_path}")

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
    if args.min_trades is not None:     cfg["min_trades"] = args.min_trades
    if args.scan_workers is not None:   cfg["scan_workers"] = args.scan_workers
    if args.scan_inflight_per_worker is not None:
        cfg["scan_inflight_per_worker"] = args.scan_inflight_per_worker
    if args.scan_max_tasks_per_child is not None:
        cfg["scan_max_tasks_per_child"] = args.scan_max_tasks_per_child
    if args.top is not None:            cfg["top_n"] = args.top
    # recompute: explicit CLI value always overrides DEFAULT_PARAMS["recompute"]
    if args.recompute is not None:
        cfg["recompute_bars"] = max(1, int(args.recompute))
    else:
        cfg["recompute_bars"] = max(1, int(cfg.get("recompute", 1)))

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
            print(f"\n  Tip: to split across {n_suggested} Colabs, use --pair-range:")
            for i in range(n_suggested):
                s = i * chunk
                e = min(s + chunk, total_all_pairs)
                print(f"      Colab {i+1}: --pair-range {s}-{e}    ({e-s:,} pairs)")

    # Print config
    print(f"\n  Strategy parameters:")
    print(f"    {'recompute_bars':<30} = {cfg.get('recompute_bars', cfg.get('recompute', 1))}")
    for k in ["z_entry", "z_entry_max", "z_exit", "z_stop", "window_size",
              "max_notional_pct", "capital", "hold_multiplier", "max_hold_days",
              "p_value_threshold", "hedge_min", "hedge_max", "beta_threshold",
              "beta_critical", "coint_stability_min_bars", "hl_min_days", "hl_max_days",
              "commission_rate", "slippage_rate", "circuit_breaker_pct"]:
        print(f"    {k:<30} = {cfg[k]}")
    print(f"    {'scan_workers':<30} = {cfg['scan_workers']}")
    print(f"    {'scan_inflight_per_worker':<30} = {cfg.get('scan_inflight_per_worker', 4)}")
    print(f"    {'scan_max_tasks_per_child':<30} = {cfg.get('scan_max_tasks_per_child', 0)}")
    print()

    # ── Hyperopt (optional) ────────────────────────────────────────────────────
    hyperopt_history: list[dict] = []
    if args.hyperopt_trials > 0:
        print(f"  Running hyperopt: {args.hyperopt_trials} trials …\n")
        cfg, hyperopt_history = run_scanner_hyperopt(
            market=market,
            pairs=pairs,
            base_cfg=cfg,
            trials=args.hyperopt_trials,
            n_startup=args.hyperopt_startup,
            seed=args.seed,
            sample_pairs=args.hyperopt_sample,
        )
        print(f"\n  Updated parameters after hyperopt:")
        for k in ["z_entry", "z_entry_max", "z_exit", "z_stop",
                   "hold_multiplier", "max_hold_days", "p_value_threshold"]:
            print(f"    {k:<30} = {cfg[k]}")
        print()

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

    # Hyperopt history CSV
    if hyperopt_history:
        ho_path = out_dir / "pair_scanner_hyperopt.csv"
        pd.DataFrame(hyperopt_history).to_csv(ho_path, index=False)
        print(f"  [scanner] Hyperopt history -> {ho_path}")

    # Best pairs JSON (only in non-shard mode)
    if not shard_label:
        cfg["_total_scanned"] = total_all_pairs
        best_path = out_dir / "best_pairs_backtest.json"
        save_best_pairs_json(results, top_n, best_path, params_used=cfg)
    else:
        print(f"\n  [scanner] Shard {shard_id+1}/{n_shards} complete!")
        print(f"  [scanner] When ALL shards done, merge with:")
        print(f"    python backtest_pair_scanner.py --merge-shards")

    # ── Print summary ──────────────────────────────────────────────────────────
    print_summary(results, top_n=25)


if __name__ == "__main__":
    main()
