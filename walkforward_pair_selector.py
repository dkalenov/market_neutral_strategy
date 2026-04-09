import argparse
import concurrent.futures
import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_bot_parity_nonclosecointegration import (
    BacktestParams,
    BotParityBacktester,
    CandidatePair,
    FundingData,
    MarketData,
    load_funding_data,
    load_klines_market_data,
)
from backtest_pair_scanner import _prepare_market_init_args


WF_MARKET: MarketData | None = None
WF_FUNDING: FundingData | None = None


@dataclass
class SelectorConfig:
    report_start: pd.Timestamp
    report_end: pd.Timestamp
    tier_a_size: int
    tier_b_size: int
    min_trades_90d: int
    min_trades_180d: int
    min_net_pnl_90d: float
    min_pf_90d: float
    max_mdd_90d: float
    max_time_exit_ratio_90d: float
    exclude_symbols: set[str]
    rebalance_frequency: str


def parse_timestamp_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def canonical_pair(symbol1: str, symbol2: str) -> tuple[str, str]:
    a = str(symbol1).strip().upper()
    b = str(symbol2).strip().upper()
    return tuple(sorted((a, b)))


def pair_text(key: tuple[str, str]) -> str:
    return f"{key[0]}-{key[1]}"


def load_pairs_from_json(path: str) -> list[tuple[str, str]]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    entries: list[Any]
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict) and isinstance(raw.get("pairs"), list):
        entries = raw["pairs"]
    else:
        raise ValueError(f"Unsupported pair file format: {path}")

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in entries:
        pair_str = ""
        if isinstance(row, str):
            pair_str = row.strip().upper()
        elif isinstance(row, dict):
            pair_str = str(row.get("pair", "")).strip().upper()
            if not pair_str:
                s1 = str(row.get("symbol1", "")).strip().upper()
                s2 = str(row.get("symbol2", "")).strip().upper()
                if s1 and s2:
                    pair_str = f"{s1}-{s2}"
        if "-" not in pair_str:
            continue
        a, b = [x.strip().upper() for x in pair_str.split("-", 1)]
        if not a or not b or a == b:
            continue
        key = canonical_pair(a, b)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def _worker_init(
    dates_bytes: bytes,
    dates_len: int,
    symbols: list[str],
    sym_to_idx: dict[str, int],
    ohlcv_bytes: bytes,
    shape: tuple[int, int],
    funding_bytes: bytes | None,
    funding_shape: tuple[int, int] | None,
    funding_meta: tuple[str, int, int, int] | None,
) -> None:
    global WF_MARKET, WF_FUNDING
    n_bars, n_syms = shape
    dates_arr = np.frombuffer(dates_bytes, dtype=np.int64)[:dates_len].copy()
    dates_idx = pd.DatetimeIndex(dates_arr, tz="UTC")
    all_data = np.frombuffer(ohlcv_bytes, dtype=np.float64).reshape(5, n_bars, n_syms).copy()
    WF_MARKET = MarketData(
        dates=dates_idx,
        symbols=symbols,
        symbol_to_idx=sym_to_idx,
        open_arr=all_data[0],
        high_arr=all_data[1],
        low_arr=all_data[2],
        close_arr=all_data[3],
        volume_arr=all_data[4],
    )
    WF_FUNDING = None
    if funding_bytes is not None and funding_shape is not None and funding_meta is not None:
        rate_arr = np.frombuffer(funding_bytes, dtype=np.float64).reshape(funding_shape).copy()
        source_path, loaded_records, matched_records, skipped_records = funding_meta
        WF_FUNDING = FundingData(
            rate_arr=rate_arr,
            source_path=source_path,
            loaded_records=loaded_records,
            matched_records=matched_records,
            skipped_records=skipped_records,
        )


def _serializable_trade_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    keep_cols = [
        "pair",
        "entry_time",
        "exit_time",
        "entry_z",
        "close_z",
        "entry_beta",
        "close_beta",
        "entry_pvalue",
        "close_pvalue",
        "entry_half_life",
        "exit_reason",
        "exit_reason_text",
        "gross_pnl",
        "entry_fee",
        "exit_fee",
        "funding_total",
        "net_pnl_before_funding",
        "net_pnl",
        "hold_hours",
    ]
    out = df.copy()
    for col in ["entry_time", "exit_time"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce").astype(str)
    keep = [c for c in keep_cols if c in out.columns]
    return out[keep].to_dict(orient="records")


def _scan_pair_full_history(args: tuple) -> dict[str, Any]:
    global WF_MARKET, WF_FUNDING
    sym_a, sym_b, params_dict, report_start, report_end = args
    if WF_MARKET is None:
        raise RuntimeError("WF_MARKET is not initialized")

    params = BacktestParams(**dict(params_dict))
    params.funding_csv = ""
    candidate = CandidatePair(symbol1=sym_a, symbol2=sym_b, score=1.0, source="walkforward_selector")
    bt = BotParityBacktester(
        market=WF_MARKET,
        params=params,
        candidates=[candidate],
        n_workers=1,
        exact_mode=True,
        coint_recompute_every=1,
        idle_eval_every=1,
        discovery_recheck_bars=0,
    )
    if WF_FUNDING is not None:
        bt.funding_data = WF_FUNDING
        bt.funding_enabled = True

    result = bt.run()
    trades_df = result.trades.copy()
    if "entry_time" in trades_df.columns:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
        trades_df = trades_df[trades_df["entry_time"] >= report_start]
    if "exit_time" in trades_df.columns:
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True, errors="coerce")
        trades_df = trades_df[trades_df["exit_time"] < report_end]

    pair = pair_text(canonical_pair(sym_a, sym_b))
    trades_df = trades_df.copy()
    trades_df["pair"] = pair
    return {
        "pair": pair,
        "trade_count": int(len(trades_df)),
        "trades": _serializable_trade_rows(trades_df),
    }


def _window_trade_metrics(trades_df: pd.DataFrame, end_ts: pd.Timestamp, days: int) -> dict[str, Any]:
    start_ts = end_ts - pd.Timedelta(days=days)
    if trades_df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "fees": 0.0,
            "funding": 0.0,
            "avg_trade_pnl": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "time_exit_ratio": 0.0,
            "tp_ratio": 0.0,
            "force_close_rate": 0.0,
            "hardware_exit_rate": 0.0,
            "last_exit_time": "",
        }

    mask = (trades_df["exit_time"] > start_ts) & (trades_df["exit_time"] <= end_ts)
    window = trades_df.loc[mask].copy()
    if window.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "fees": 0.0,
            "funding": 0.0,
            "avg_trade_pnl": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "time_exit_ratio": 0.0,
            "tp_ratio": 0.0,
            "force_close_rate": 0.0,
            "hardware_exit_rate": 0.0,
            "last_exit_time": "",
        }

    window = window.sort_values("exit_time")
    trades = int(len(window))
    wins = int((window["net_pnl"] > 0).sum())
    losses = int((window["net_pnl"] < 0).sum())
    sum_pos = float(window.loc[window["net_pnl"] > 0, "net_pnl"].sum())
    sum_neg_abs = float(abs(window.loc[window["net_pnl"] < 0, "net_pnl"].sum()))
    equity = 100.0 + window["net_pnl"].cumsum()
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    exit_counts = Counter(window["exit_reason"].astype(str))
    tp_count = int(exit_counts.get("z_tp", 0) + exit_counts.get("hardware_tp", 0))
    hardware_count = int(exit_counts.get("hardware_tp", 0) + exit_counts.get("hardware_sl", 0))
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / trades) if trades > 0 else 0.0,
        "net_pnl": float(window["net_pnl"].sum()),
        "gross_pnl": float(window["gross_pnl"].sum()),
        "fees": float(window["entry_fee"].sum() + window["exit_fee"].sum()),
        "funding": float(window["funding_total"].sum()),
        "avg_trade_pnl": float(window["net_pnl"].mean()),
        "profit_factor": float(sum_pos / sum_neg_abs) if sum_neg_abs > 0 else float(sum_pos > 0),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "time_exit_ratio": float(exit_counts.get("time_exit", 0) / trades) if trades > 0 else 0.0,
        "tp_ratio": float(tp_count / trades) if trades > 0 else 0.0,
        "force_close_rate": float(exit_counts.get("force_close", 0) / trades) if trades > 0 else 0.0,
        "hardware_exit_rate": float(hardware_count / trades) if trades > 0 else 0.0,
        "last_exit_time": str(window["exit_time"].max()),
    }


def _bounded_tanh(value: float, scale: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(math.tanh(float(value) / max(1e-9, float(scale))))


def _window_score(metrics: dict[str, Any]) -> float:
    trades = int(metrics.get("trades", 0) or 0)
    if trades <= 0:
        return -1.0
    net_score = _bounded_tanh(float(metrics.get("net_pnl", 0.0) or 0.0), 3.0)
    pf = float(metrics.get("profit_factor", 0.0) or 0.0)
    pf_score = max(0.0, min(1.0, (pf - 1.0) / 1.5))
    win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
    wr_score = max(0.0, min(1.0, (win_rate - 0.50) / 0.25))
    tp_ratio = float(metrics.get("tp_ratio", 0.0) or 0.0)
    trade_density = min(1.0, trades / 12.0)
    return (
        0.45 * net_score
        + 0.20 * pf_score
        + 0.15 * wr_score
        + 0.10 * tp_ratio
        + 0.10 * trade_density
    )


def _pair_passes_filters(symbols: tuple[str, str], windows: dict[str, dict[str, Any]], cfg: SelectorConfig) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if symbols[0] in cfg.exclude_symbols or symbols[1] in cfg.exclude_symbols:
        reasons.append("excluded_symbol")
    w90 = windows["90d"]
    w180 = windows["180d"]
    if not (
        int(w90["trades"]) >= int(cfg.min_trades_90d)
        or int(w180["trades"]) >= int(cfg.min_trades_180d)
    ):
        reasons.append("insufficient_recent_trades")
    if float(w90["net_pnl"]) <= float(cfg.min_net_pnl_90d):
        reasons.append("net_pnl_90d_non_positive")
    if float(w90["profit_factor"]) < float(cfg.min_pf_90d):
        reasons.append("pf_90d_below_threshold")
    if abs(float(w90["max_drawdown"])) > float(cfg.max_mdd_90d):
        reasons.append("mdd_90d_too_deep")
    if float(w90["time_exit_ratio"]) > float(cfg.max_time_exit_ratio_90d):
        reasons.append("time_exit_ratio_90d_too_high")
    if float(w90["force_close_rate"]) > 0.0:
        reasons.append("force_close_present")
    return (len(reasons) == 0, reasons)


def _pair_selector_score(windows: dict[str, dict[str, Any]]) -> float:
    score_60 = _window_score(windows["60d"])
    score_90 = _window_score(windows["90d"])
    score_180 = _window_score(windows["180d"])
    score_30 = _window_score(windows["30d"])
    positive_count = sum(1 for key in ["60d", "90d", "180d"] if float(windows[key]["net_pnl"]) > 0)
    stability_bonus = {0: -0.20, 1: 0.0, 2: 0.15, 3: 0.30}.get(positive_count, 0.0)
    if float(windows["30d"]["net_pnl"]) <= 0:
        stability_bonus -= 0.10
    trade_density_bonus = 0.10 * min(1.0, float(windows["90d"]["trades"]) / 12.0)
    risk_penalty = (
        0.25 * max(0.0, (abs(float(windows["90d"]["max_drawdown"])) - 0.04) / 0.04)
        + 0.15 * max(0.0, (float(windows["90d"]["time_exit_ratio"]) - 0.20) / 0.20)
        + 0.10 * float(windows["90d"]["force_close_rate"])
        + 0.05 * float(windows["90d"]["hardware_exit_rate"])
    )
    score = (
        0.40 * score_90
        + 0.25 * score_60
        + 0.15 * score_180
        + 0.10 * score_30
        + stability_bonus
        + trade_density_bonus
        - risk_penalty
    )
    return float(score)


def build_rebalance_points(
    market_dates: pd.DatetimeIndex,
    report_start: pd.Timestamp,
    report_end: pd.Timestamp,
    frequency: str,
) -> list[pd.Timestamp]:
    if frequency == "weekly":
        targets = list(pd.date_range(report_start, report_end, freq="7D", inclusive="left", tz="UTC"))
    elif frequency == "monthly":
        targets = list(pd.date_range(report_start, report_end, freq="MS", inclusive="left", tz="UTC"))
        if not targets or targets[0] != report_start:
            targets = [report_start] + [t for t in targets if t > report_start]
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")

    dates = pd.DatetimeIndex(market_dates, tz="UTC")
    points: list[pd.Timestamp] = []
    for target in targets:
        idx = dates.searchsorted(target)
        if idx >= len(dates):
            break
        points.append(pd.Timestamp(dates[idx]).tz_convert("UTC"))
    deduped: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for ts in points:
        if ts not in seen and ts < report_end:
            seen.add(ts)
            deduped.append(ts)
    return deduped


def build_schedule(
    trades_df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    rebalance_points: list[pd.Timestamp],
    cfg: SelectorConfig,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    if trades_df.empty or "pair" not in trades_df.columns:
        grouped: dict[str, pd.DataFrame] = {}
    else:
        grouped = {pair: g.sort_values("exit_time").copy() for pair, g in trades_df.groupby("pair")}
    schedule: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    selection_counts: Counter[str] = Counter()

    for rebalance_ts in rebalance_points:
        ranked_rows: list[dict[str, Any]] = []
        filter_counts: Counter[str] = Counter()
        for key in pairs:
            pair = pair_text(key)
            df_pair = grouped.get(pair)
            if df_pair is None:
                df_pair = pd.DataFrame(
                    columns=["pair", "entry_time", "exit_time", "exit_reason", "gross_pnl", "entry_fee", "exit_fee", "funding_total", "net_pnl"]
                )
            windows = {
                "30d": _window_trade_metrics(df_pair, rebalance_ts, 30),
                "60d": _window_trade_metrics(df_pair, rebalance_ts, 60),
                "90d": _window_trade_metrics(df_pair, rebalance_ts, 90),
                "180d": _window_trade_metrics(df_pair, rebalance_ts, 180),
            }
            passed, reasons = _pair_passes_filters(key, windows, cfg)
            for reason in reasons:
                filter_counts[reason] += 1
            score = _pair_selector_score(windows) if passed else -999.0
            row = {
                "rebalance_time": str(rebalance_ts),
                "pair": pair,
                "score": float(score),
                "passed_filters": bool(passed),
                "filter_reasons": "|".join(reasons),
            }
            for window_key, metrics in windows.items():
                prefix = window_key
                for metric_name, metric_value in metrics.items():
                    row[f"{prefix}_{metric_name}"] = metric_value
            rows.append(row)
            if passed:
                ranked_rows.append(row)

        ranked_rows.sort(
            key=lambda x: (
                -float(x["score"]),
                -float(x["90d_net_pnl"]),
                -int(x["90d_trades"]),
                x["pair"],
            )
        )
        tier_a = ranked_rows[: cfg.tier_a_size]
        tier_b = ranked_rows[cfg.tier_a_size : cfg.tier_a_size + cfg.tier_b_size]
        for item in tier_a:
            selection_counts[item["pair"]] += 1
        schedule.append(
            {
                "rebalance_time": str(rebalance_ts),
                "eligible_count": int(len(ranked_rows)),
                "filter_counts": dict(filter_counts),
                "tier_a": tier_a,
                "tier_b": tier_b,
            }
        )

    detail_df = pd.DataFrame(rows)
    freq_rows = []
    total_rebalances = max(1, len(rebalance_points))
    for pair, count in selection_counts.most_common():
        freq_rows.append(
            {
                "pair": pair,
                "tier_a_selections": int(count),
                "tier_a_selection_rate": float(count / total_rebalances),
            }
        )
    selection_df = pd.DataFrame(freq_rows)
    return schedule, detail_df, selection_df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rolling walk-forward pair selector")
    p.add_argument("--input", required=True, help="Combined klines CSV with Date,Symbol,OHLCV.")
    p.add_argument("--best-pairs", required=True, help="Universe pair JSON to evaluate.")
    p.add_argument("--funding-csv", default="", help="Optional funding CSV.")
    p.add_argument("--output-dir", required=True, help="Output directory.")
    p.add_argument("--report-start", required=True, help="UTC report start.")
    p.add_argument("--report-end", required=True, help="UTC report end (exclusive).")
    p.add_argument("--rebalance-frequency", choices=["weekly", "monthly"], default="weekly")
    p.add_argument("--tier-a-size", type=int, default=10)
    p.add_argument("--tier-b-size", type=int, default=15)
    p.add_argument("--scan-workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--scan-pairs-limit", type=int, default=0, help="Optional limit for debugging.")
    p.add_argument("--exclude-symbols", nargs="*", default=["SPELLUSDT", "NEIROUSDT"])
    p.add_argument("--min-trades-90d", type=int, default=6)
    p.add_argument("--min-trades-180d", type=int, default=8)
    p.add_argument("--min-net-pnl-90d", type=float, default=0.0)
    p.add_argument("--min-pf-90d", type=float, default=1.05)
    p.add_argument("--max-mdd-90d", type=float, default=0.08)
    p.add_argument("--max-time-exit-ratio-90d", type=float, default=0.35)
    p.add_argument("--capital", type=float, default=100.0)
    p.add_argument("--leverage", type=int, default=20)
    p.add_argument("--max-notional-pct", type=float, default=0.175)
    p.add_argument("--z-entry", type=float, default=1.6)
    p.add_argument("--z-entry-max", type=float, default=2.7)
    p.add_argument("--z-exit", type=float, default=0.10)
    p.add_argument("--z-stop", type=float, default=4.0)
    p.add_argument("--beta-threshold", type=float, default=0.20)
    p.add_argument("--beta-alert-threshold", type=float, default=0.50)
    p.add_argument("--beta-critical", type=float, default=1.0)
    p.add_argument("--p-value-threshold", type=float, default=0.05)
    p.add_argument("--coint-stability-min-bars", type=int, default=4)
    p.add_argument("--coint-broken-grace-bars", type=int, default=2)
    p.add_argument("--signal-confirm-sec", type=int, default=10)
    p.add_argument("--hold-multiplier", type=float, default=6.5)
    p.add_argument("--max-hold-days", type=float, default=22)
    p.add_argument("--hl-min-days", type=float, default=0.5)
    p.add_argument("--hl-max-days", type=float, default=4.5)
    p.add_argument("--sl-atr-mult", type=float, default=5.0)
    p.add_argument("--sl-min-pct", type=float, default=0.5)
    p.add_argument("--sl-max-pct", type=float, default=0.75)
    p.add_argument("--tp-atr-mult", type=float, default=8.0)
    p.add_argument("--tp-min-pct", type=float, default=0.5)
    p.add_argument("--tp-max-pct", type=float, default=0.75)
    p.add_argument("--hardware-sltp-mode", default="exit")
    p.add_argument("--hardware-sl-enabled", default="true")
    p.add_argument("--hardware-tp-enabled", default="true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_start = parse_timestamp_utc(args.report_start)
    report_end = parse_timestamp_utc(args.report_end)
    if report_end <= report_start:
        raise ValueError("report_end must be greater than report_start")

    pairs = load_pairs_from_json(args.best_pairs)
    if args.scan_pairs_limit and int(args.scan_pairs_limit) > 0:
        pairs = pairs[: int(args.scan_pairs_limit)]
    universe_symbols = sorted({s for key in pairs for s in key})
    print(f"[INFO] universe pairs={len(pairs)} symbols={len(universe_symbols)}")

    market = load_klines_market_data(args.input, max_symbols=len(universe_symbols))
    print(
        f"[INFO] market bars={len(market.dates)} symbols={len(market.symbols)} "
        f"start={market.dates.min()} end={market.dates.max()}"
    )

    funding_data = None
    if args.funding_csv:
        funding_data = load_funding_data(args.funding_csv, market)
        if funding_data is not None:
            print(
                f"[INFO] funding matched={funding_data.matched_records}/{funding_data.loaded_records} "
                f"skipped={funding_data.skipped_records}"
            )

    market_init = _prepare_market_init_args(market)
    funding_bytes = None
    funding_shape = None
    funding_meta = None
    if funding_data is not None:
        funding_bytes = funding_data.rate_arr.astype(np.float64).tobytes()
        funding_shape = funding_data.rate_arr.shape
        funding_meta = (
            funding_data.source_path,
            int(funding_data.loaded_records),
            int(funding_data.matched_records),
            int(funding_data.skipped_records),
        )

    params = BacktestParams(
        timeframe="4h",
        capital=float(args.capital),
        leverage=int(args.leverage),
        max_notional_pct=float(args.max_notional_pct),
        z_entry=float(args.z_entry),
        z_entry_max=float(args.z_entry_max),
        z_exit=float(args.z_exit),
        z_stop=float(args.z_stop),
        beta_threshold=float(args.beta_threshold),
        beta_alert_threshold=float(args.beta_alert_threshold),
        beta_critical=float(args.beta_critical),
        p_value_threshold=float(args.p_value_threshold),
        coint_stability_min_bars=int(args.coint_stability_min_bars),
        coint_broken_grace_bars=int(args.coint_broken_grace_bars),
        signal_confirm_sec=int(args.signal_confirm_sec),
        hold_multiplier=float(args.hold_multiplier),
        max_hold_days=float(args.max_hold_days),
        hl_min_days=float(args.hl_min_days),
        hl_max_days=float(args.hl_max_days),
        sl_atr_mult=float(args.sl_atr_mult),
        sl_min_pct=float(args.sl_min_pct),
        sl_max_pct=float(args.sl_max_pct),
        tp_atr_mult=float(args.tp_atr_mult),
        tp_min_pct=float(args.tp_min_pct),
        tp_max_pct=float(args.tp_max_pct),
        hardware_sltp_mode=str(args.hardware_sltp_mode),
        hardware_sl_enabled=str(args.hardware_sl_enabled).lower() == "true",
        hardware_tp_enabled=str(args.hardware_tp_enabled).lower() == "true",
        funding_csv="",
        top_pairs_limit=0,
        max_active_pairs=4,
    )
    params_dict = asdict(params)

    tasks = [(a, b, params_dict, report_start, report_end) for a, b in pairs]
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    scan_workers = max(1, int(args.scan_workers))
    if scan_workers <= 1:
        _worker_init(*market_init, funding_bytes, funding_shape, funding_meta)
        for idx, task in enumerate(tasks, start=1):
            results.append(_scan_pair_full_history(task))
            if idx % 50 == 0 or idx == len(tasks):
                elapsed = time.perf_counter() - t0
                print(f"[scan] {idx}/{len(tasks)} elapsed={elapsed:.1f}s")
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=scan_workers,
            initializer=_worker_init,
            initargs=(*market_init, funding_bytes, funding_shape, funding_meta),
        ) as pool:
            for idx, res in enumerate(pool.map(_scan_pair_full_history, tasks, chunksize=1), start=1):
                results.append(res)
                if idx % 50 == 0 or idx == len(tasks):
                    elapsed = time.perf_counter() - t0
                    print(f"[scan] {idx}/{len(tasks)} elapsed={elapsed:.1f}s")

    trade_rows: list[dict[str, Any]] = []
    pair_scan_rows: list[dict[str, Any]] = []
    for item in results:
        pair_scan_rows.append({"pair": item["pair"], "trade_count": int(item["trade_count"])})
        trade_rows.extend(item["trades"])

    trades_df = pd.DataFrame(trade_rows)
    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=[
                "pair",
                "entry_time",
                "exit_time",
                "entry_z",
                "close_z",
                "entry_beta",
                "close_beta",
                "entry_pvalue",
                "close_pvalue",
                "entry_half_life",
                "exit_reason",
                "exit_reason_text",
                "gross_pnl",
                "entry_fee",
                "exit_fee",
                "funding_total",
                "net_pnl_before_funding",
                "net_pnl",
                "hold_hours",
            ]
        )
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True, errors="coerce")
    trades_path = out_dir / "pair_trade_tape.csv"
    trades_df.to_csv(trades_path, index=False)
    pd.DataFrame(pair_scan_rows).sort_values("pair").to_csv(out_dir / "pair_scan_summary.csv", index=False)

    cfg = SelectorConfig(
        report_start=report_start,
        report_end=report_end,
        tier_a_size=int(args.tier_a_size),
        tier_b_size=int(args.tier_b_size),
        min_trades_90d=int(args.min_trades_90d),
        min_trades_180d=int(args.min_trades_180d),
        min_net_pnl_90d=float(args.min_net_pnl_90d),
        min_pf_90d=float(args.min_pf_90d),
        max_mdd_90d=float(args.max_mdd_90d),
        max_time_exit_ratio_90d=float(args.max_time_exit_ratio_90d),
        exclude_symbols={str(x).strip().upper() for x in args.exclude_symbols},
        rebalance_frequency=str(args.rebalance_frequency),
    )

    rebalance_points = build_rebalance_points(market.dates, report_start, report_end, cfg.rebalance_frequency)
    schedule, detail_df, selection_df = build_schedule(trades_df, pairs, rebalance_points, cfg)

    (out_dir / "rebalance_schedule.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    detail_df.to_csv(out_dir / "rebalance_detail.csv", index=False)
    selection_df.to_csv(out_dir / "selection_frequency.csv", index=False)

    latest_tier_a = schedule[-1]["tier_a"] if schedule else []
    latest_tier_b = schedule[-1]["tier_b"] if schedule else []
    (out_dir / "latest_tier_a_pairs.json").write_text(json.dumps(latest_tier_a, indent=2), encoding="utf-8")
    (out_dir / "latest_tier_b_pairs.json").write_text(json.dumps(latest_tier_b, indent=2), encoding="utf-8")
    selector_cfg_out = asdict(cfg)
    selector_cfg_out["report_start"] = str(cfg.report_start)
    selector_cfg_out["report_end"] = str(cfg.report_end)
    selector_cfg_out["exclude_symbols"] = sorted(cfg.exclude_symbols)
    params_out = {
        "selector_config": selector_cfg_out,
        "backtest_params": params_dict,
        "scan_workers": scan_workers,
        "universe_pairs": len(pairs),
        "rebalance_count": len(rebalance_points),
    }
    (out_dir / "selector_params.json").write_text(json.dumps(params_out, indent=2), encoding="utf-8")

    print(f"[DONE] trade tape -> {trades_path}")
    print(f"[DONE] rebalance schedule -> {out_dir / 'rebalance_schedule.json'}")
    print(f"[DONE] latest Tier A -> {len(latest_tier_a)} pairs")
    if latest_tier_a:
        preview = ", ".join(x["pair"] for x in latest_tier_a[: min(8, len(latest_tier_a))])
        print(f"[DONE] Tier A preview: {preview}")


if __name__ == "__main__":
    main()
