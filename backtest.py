import numpy as np
import pandas as pd
from backtesting import Strategy, Backtest
import math



CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0004
SLIPPAGE_RATE = 0.0005
LEVERAGE = 1.0

WINDOW = 200
VOL_LOOKBACK = 60
RISK_PCT_PER_PAIR = 0.1
MAX_NOTIONAL_PER_PAIR = 0.1

Z_ENTRY = 2.0
Z_EXIT_FULL = 0.0
Z_STOP_HARD = 4.0
HOLD_MULTIPLIER = 3
MAX_HOLD_DAYS = 30

PERIODS_PER_YEAR = 24 * 365  # hourly periods


def vol_parity_notional_from_series(s1, s2, hedge, cap_pair_usd, vol_lookback=VOL_LOOKBACK):
    """Compute volatility-parity allocations for both legs."""
    if s1 is None or s2 is None or len(s1) < 2 or len(s2) < 2:
        return 0.0, 0.0
    r1 = np.diff(np.log(s1[-vol_lookback:])) if len(s1) >= vol_lookback else np.diff(np.log(s1))
    r2 = np.diff(np.log(s2[-vol_lookback:])) if len(s2) >= vol_lookback else np.diff(np.log(s2))
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

def calculate_portfolio_beta(portfolio_returns, market_returns):
    """Compute beta of portfolio vs market."""
    if len(portfolio_returns) < 2 or len(market_returns) < 2:
        return np.nan
    common = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
    if len(common) < 2:
        return np.nan
    cov = np.cov(common.iloc[:,0], common.iloc[:,1])[0,1]
    var_m = np.var(common.iloc[:,1])
    if var_m == 0:
        return np.nan
    return cov / var_m






def compute_historical_mae(d1, d2, r1, r2, hold):
    """Estimate 95% worst-case adverse PnL over holding period."""
    r1 = np.array(r1, dtype=float)
    r2 = np.array(r2, dtype=float)
    N = len(r1)
    if N < hold or hold <= 0:
        return np.inf
    adverse = []
    for start in range(0, N - hold + 1):
        future_r1 = r1[start:start+hold]
        future_r2 = r2[start:start+hold]
        cum_pnl = np.cumsum(d1 * future_r1 - d2 * future_r2)
        worst = np.min(cum_pnl)
        adverse.append(-worst if worst < 0 else 0.0)
    if len(adverse) == 0:
        return np.inf
    return float(np.percentile(adverse, 95))


def compute_cagr(equity_series, dates):
    """Compute compound annual growth rate."""
    if len(dates) < 2 or len(equity_series) < 2:
        return np.nan
    total_seconds = (dates.iloc[-1] - dates.iloc[0]).total_seconds()
    if total_seconds <= 0:
        return np.nan
    years = total_seconds / (365.0 * 24 * 3600)
    try:
        return (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1.0 / years) - 1.0
    except Exception:
        return np.nan


def max_drawdown(equity):
    """Return max drawdown and drawdown series."""
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    return float(drawdown.min()), drawdown



class PairsStrategy(Strategy):
    """Minimal wrapper for running custom backtest engine inside backtesting.py."""
    def init(self):
        pass

    def next(self):
        if hasattr(self, "_engine_step"):
            idx_local = len(self.data.Close) - 1
            self._engine_step(idx_local)
        else:
            return


def run_pairs_backtest_with_backtestingpy(input_csv, signals_csv):
    """Run pair-trading backtest using backtesting.py as a timeline driver."""
    # Load price data
    df = pd.read_csv(input_csv)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"])
    has_open = "Open" in df.columns

    pivot_close = df.pivot(index="Date", columns="Symbol", values="Close").sort_index()
    pivot_open = df.pivot(index="Date", columns="Symbol", values="Open").sort_index() if has_open else None
    close = pivot_close.ffill().bfill()
    openp = pivot_open.ffill().bfill() if has_open else None
    symbols = list(close.columns)
    dates = close.index
    N = len(dates)

    if N == 0 or len(symbols) == 0:
        raise ValueError("Empty price data or no symbols found.")

    dummy_symbol = symbols[0]
    dtf = pd.DataFrame({
        "Open": openp[dummy_symbol] if openp is not None else close[dummy_symbol],
        "High": close[dummy_symbol],
        "Low": close[dummy_symbol],
        "Close": close[dummy_symbol]
    }, index=dates)

    # Load signals
    sigs = pd.read_csv(signals_csv)
    if "start_date" in sigs.columns:
        sigs["signal_time"] = pd.to_datetime(sigs["start_date"])
    elif "end_date" in sigs.columns:
        sigs["signal_time"] = pd.to_datetime(sigs["end_date"])
    else:
        raise ValueError("Signals must contain start_date or end_date column.")

    sigs = sigs.sort_values("signal_time").reset_index(drop=True)

    # Map signals to indices
    mapped_signals = []
    for _, row in sigs.iterrows():
        ts = pd.Timestamp(row["signal_time"])
        pos = dates.searchsorted(ts)
        if pos >= N:
            continue
        idx = pos if dates[pos] == ts else max(0, pos - 1)
        next_idx = idx + 1
        if next_idx < N:
            mapped_signals.append((idx, next_idx, row))

    sigs_by_idx = {}
    for idx, next_idx, row in mapped_signals:
        sigs_by_idx.setdefault(idx, []).append((next_idx, row))

    # State variables
    cash = float(CAPITAL)
    reserved_cash = 0.0
    invested_capital = 0.0
    positions = []
    trades = []
    ledger = []
    equity_hist = []
    date_hist = []
    scheduled_entries_global = []

    def ledger_append(time, pair, typ, cash_change, note=""):
        ledger.append({"time": time, "pair": pair, "type": typ,
                       "cash_change": float(cash_change), "note": note})

    # Core per-bar engine
    def engine_step(self, idx_bar):
        nonlocal cash, reserved_cash, invested_capital
        nonlocal positions, trades, ledger, equity_hist, date_hist, scheduled_entries_global

        ts = dates[idx_bar]
        next_idx = idx_bar + 1

        # 1) Schedule new entries
        todays = sigs_by_idx.get(idx_bar, [])
        for next_idx_sig, sigrow in todays:
            pair = sigrow.get("pair")
            if not isinstance(pair, str) or "-" not in pair:
                continue
            a, b = pair.split("-")
            if a not in symbols or b not in symbols:
                continue

            hedge = float(sigrow.get("hedge_ratio", 1.0))
            entry_z = float(sigrow.get("z", sigrow.get("z_entry", np.nan)))

            raw_signal = sigrow.get("signal", sigrow.get("side", sigrow.get("sig", np.nan)))
            side = None
            if not pd.isna(raw_signal):
                s = str(raw_signal).strip().lower()
                if s in ("1", "1.0", "+1", "long", "buy"):
                    side = 1
                elif s in ("-1", "-1.0", "short", "sell"):
                    side = -1
                else:
                    try:
                        side_val = float(s)
                        side = 1 if side_val > 0 else -1
                    except Exception:
                        side = None
            if side is None:
                side = -1 if entry_z > 0 else 1

            caps = CAPITAL * MAX_NOTIONAL_PER_PAIR

            # Allocation logic
            if not (pd.isna(sigrow.get("dollar1")) or pd.isna(sigrow.get("dollar2"))):
                alloc_a = float(sigrow["dollar1"])
                alloc_b = float(sigrow["dollar2"])
            elif not (pd.isna(sigrow.get("alloc_A_usd")) or pd.isna(sigrow.get("alloc_B_usd"))):
                alloc_a = float(sigrow["alloc_A_usd"])
                alloc_b = float(sigrow["alloc_B_usd"])
            else:
                start_idx_v = max(0, idx_bar - VOL_LOOKBACK + 1)
                s1_hist = close[a].iloc[start_idx_v:idx_bar+1].values
                s2_hist = close[b].iloc[start_idx_v:idx_bar+1].values
                alloc_a, alloc_b = vol_parity_notional_from_series(s1_hist, s2_hist, hedge, cap_pair_usd=caps)

            if (abs(alloc_a) + abs(alloc_b)) <= 0:
                ledger_append(ts, pair, "skip", 0.0, note="zero_alloc")
                continue

            # MAE scaling
            hl = float(sigrow.get("half_life", np.nan)) if not pd.isna(sigrow.get("half_life")) else 3.0
            hold = int(max(1, round(hl * HOLD_MULTIPLIER)))
            start_idx_mae = max(0, idx_bar - VOL_LOOKBACK - 10)
            if idx_bar - start_idx_mae >= 2:
                series_a = close[a].iloc[start_idx_mae:idx_bar+1].values
                series_b = close[b].iloc[start_idx_mae:idx_bar+1].values
                r1_hist = np.diff(np.log(series_a)) if len(series_a) >= 2 else np.array([])
                r2_hist = np.diff(np.log(series_b)) if len(series_b) >= 2 else np.array([])
            else:
                r1_hist = np.array([]); r2_hist = np.array([])

            mae95 = compute_historical_mae(alloc_a, alloc_b, r1_hist, r2_hist, hold)
            max_loss_allowed = CAPITAL * RISK_PCT_PER_PAIR
            if mae95 != np.inf and mae95 > 0 and mae95 > max_loss_allowed:
                scale = max_loss_allowed / mae95
                alloc_a *= scale; alloc_b *= scale

            est_entry_comm = (abs(alloc_a) + abs(alloc_b)) * COMMISSION_RATE
            available = cash - reserved_cash
            if est_entry_comm > available + 1e-12:
                ledger_append(ts, pair, "skip", 0.0, note="no_cash")
                continue

            reserved_cash += est_entry_comm
            scheduled_entries_global.append({
                "exec_idx": next_idx_sig, "pair": pair, "a": a, "b": b,
                "alloc_a": alloc_a, "alloc_b": alloc_b, "hedge": hedge,
                "entry_z": entry_z, "side": side, "est_entry_comm": est_entry_comm
            })

        # 2) Execute entries
        to_run = [s for s in scheduled_entries_global if s["exec_idx"] == idx_bar]
        for s in list(to_run):
            pair, a, b = s["pair"], s["a"], s["b"]
            alloc_a, alloc_b = s["alloc_a"], s["alloc_b"]
            hedge, entry_z, side = s["hedge"], s["entry_z"], s["side"]
            est_entry_comm = s.get("est_entry_comm", 0.0)

            pa = float(openp[a].iloc[idx_bar]) if openp is not None else float(close[a].iloc[idx_bar])
            pb = float(openp[b].iloc[idx_bar]) if openp is not None else float(close[b].iloc[idx_bar])
            pa_exec = pa * (1 + SLIPPAGE_RATE) if side == 1 else pa * (1 - SLIPPAGE_RATE)
            pb_exec = pb * (1 - SLIPPAGE_RATE) if side == 1 else pb * (1 + SLIPPAGE_RATE)

            qty_a = alloc_a / pa_exec if pa_exec > 0 else 0.0
            qty_b = alloc_b / pb_exec if pb_exec > 0 else 0.0

            sign_a = 1 if side == 1 else -1
            sign_b = -sign_a
            qty_a_signed, qty_b_signed = qty_a * sign_a, qty_b * sign_b

            entry_comm = (abs(pa_exec * qty_a) + abs(pb_exec * qty_b)) * COMMISSION_RATE
            reserved_cash -= est_entry_comm
            reserved_cash = max(reserved_cash, 0.0)
            cash -= entry_comm

            ledger_append(dates[idx_bar], pair, "entry_comm", -float(entry_comm),
                          note=f"entry pa={pa_exec:.6f}, pb={pb_exec:.6f}")

            invested_capital += (abs(alloc_a) + abs(alloc_b))
            positions.append({
                "pair": pair, "a": a, "b": b,
                "qty_a": qty_a_signed, "qty_b": qty_b_signed,
                "entry_price_a": pa_exec, "entry_price_b": pb_exec,
                "entry_alloc_a": alloc_a, "entry_alloc_b": alloc_b,
                "entry_comm": entry_comm, "entry_idx": idx_bar,
                "entry_time": dates[idx_bar], "hedge": hedge,
                "entry_z": entry_z, "side": side
            })

            trades.append({
                "pair": pair, "entry_idx": idx_bar, "entry_time": dates[idx_bar],
                "entry_price_a": pa_exec, "entry_price_b": pb_exec,
                "qty_a": qty_a_signed, "qty_b": qty_b_signed,
                "alloc_a": alloc_a, "alloc_b": alloc_b,
                "entry_comm": float(entry_comm),
                "exit_idx": None, "exit_time": None,
                "exit_price_a": None, "exit_price_b": None,
                "pnl_gross": None, "pnl_net": None, "exit_comm": None
            })

            scheduled_entries_global.remove(s)

        # 3) Mark-to-market
        unrealized_pnl = 0.0
        for p in positions:
            ca = float(close[p["a"]].iloc[idx_bar])
            cb = float(close[p["b"]].iloc[idx_bar])
            pnl_a = (ca - p["entry_price_a"]) * p["qty_a"]
            pnl_b = (cb - p["entry_price_b"]) * p["qty_b"]
            unrealized_pnl += pnl_a + pnl_b

        equity_value = cash + unrealized_pnl
        equity_hist.append(equity_value)
        date_hist.append(dates[idx_bar])

        # 4) Check exit conditions
        for p in positions.copy():
            a, b = p["a"], p["b"]
            start_idx = max(0, idx_bar - WINDOW + 1)
            if (idx_bar - start_idx + 1) < 10:
                continue
            log_a = np.log(close[a].iloc[start_idx:idx_bar+1].values)
            log_b = np.log(close[b].iloc[start_idx:idx_bar+1].values)
            spread = pd.Series(log_a - p["hedge"] * log_b)
            if spread.std() == 0 or np.isnan(spread.std()):
                continue
            z = (spread.iloc[-1] - spread.mean()) / spread.std()
            entry_z = p["entry_z"]
            need_exit = (
                (not np.isnan(z)) and
                ((entry_z * z < 0) or (abs(z) <= Z_EXIT_FULL) or (abs(z) >= Z_STOP_HARD and z * entry_z > 0))
            )
            if not need_exit:
                hold_bars = idx_bar - p["entry_idx"]
                if hold_bars / 24.0 >= MAX_HOLD_DAYS:
                    need_exit = True
            if not need_exit:
                continue

            exec_idx = min(idx_bar + 1, N - 1)
            pa = float(openp[a].iloc[exec_idx]) if openp is not None else float(close[a].iloc[exec_idx])
            pb = float(openp[b].iloc[exec_idx]) if openp is not None else float(close[b].iloc[exec_idx])
            pa_exec = pa * (1 - SLIPPAGE_RATE) if p["qty_a"] > 0 else pa * (1 + SLIPPAGE_RATE)
            pb_exec = pb * (1 - SLIPPAGE_RATE) if p["qty_b"] > 0 else pb * (1 + SLIPPAGE_RATE)

            pnl_a = (pa_exec - p["entry_price_a"]) * p["qty_a"]
            pnl_b = (pb_exec - p["entry_price_b"]) * p["qty_b"]
            total_pnl = pnl_a + pnl_b

            exit_comm = (abs(pa_exec * p["qty_a"]) + abs(pb_exec * p["qty_b"])) * COMMISSION_RATE
            cash += total_pnl - exit_comm
            invested_capital = max(0.0, invested_capital - (abs(p["entry_alloc_a"]) + abs(p["entry_alloc_b"])))
            ledger_append(dates[exec_idx], p["pair"], "exit_pnl_comm",
                          float(total_pnl - exit_comm),
                          note=f"exit pa={pa_exec:.6f}, pb={pb_exec:.6f}")

            for t in reversed(trades):
                if t.get("entry_idx") == p["entry_idx"] and t.get("pair") == p["pair"] and t.get("pnl_gross") is None:
                    t["exit_idx"] = exec_idx
                    t["exit_time"] = dates[exec_idx]
                    t["exit_price_a"] = pa_exec
                    t["exit_price_b"] = pb_exec
                    t["pnl_gross"] = float(total_pnl)
                    net_pnl = float(total_pnl - p.get("entry_comm", 0.0) - exit_comm)
                    t["pnl_net"] = net_pnl
                    t["exit_comm"] = float(exit_comm)
                    break

            positions.remove(p)

    # Attach to Strategy and run
    PairsStrategy._engine_step = engine_step
    bt = Backtest(dtf, PairsStrategy, cash=CAPITAL, commission=0.0, trade_on_close=False, exclusive_orders=True)
    _ = bt.run()

    # Create result DataFrames
    eq_series = pd.Series(equity_hist, index=pd.DatetimeIndex(date_hist))

    # Compute market returns as benchmark (e.g. equal-weighted average of all symbols)
    market_index = close.mean(axis=1)
    market_returns = np.log(market_index).diff().dropna()
    portfolio_returns = np.log(eq_series).diff().dropna()

    beta = calculate_portfolio_beta(portfolio_returns, market_returns)


    trades_df = pd.DataFrame(trades)
    ledger_df = pd.DataFrame(ledger).sort_values("time").reset_index(drop=True)

    # Metrics
    closed_trades = trades_df[trades_df["pnl_net"].notna()].copy()
    total_trades = len(closed_trades)
    win_rate = (closed_trades["pnl_net"] > 0).sum() / total_trades if total_trades > 0 else np.nan
    cagr = compute_cagr(eq_series, pd.Series(date_hist))
    mdd, dd = max_drawdown(eq_series)
    returns = eq_series.pct_change().dropna()
    sharpe = (np.nanmean(returns) * PERIODS_PER_YEAR) / (np.nanstd(returns) * math.sqrt(PERIODS_PER_YEAR)) if np.nanstd(returns) > 0 else np.nan

    metrics = {
        "CAGR": cagr,
        "MDD": mdd,
        "CAGR/MDD": (cagr / abs(mdd)) if (mdd < 0 and not np.isnan(cagr)) else np.nan,
        "Sharpe": sharpe,
        "Total Trades": total_trades,
        "Win Rate": win_rate,
        "Final Equity": float(eq_series.iloc[-1]) if len(eq_series) > 0 else CAPITAL,
        "Beta vs Market": beta

    }

    return {
        "equity": eq_series,
        "trades": trades_df,
        "metrics": metrics,
        "drawdown": dd,
        "ledger": ledger_df
    }


INPUT_FILE = "klines_data_1h_clean_2024.05.24_2025.10.24.csv"
SIGNALS_FILE = "signals_all.csv"

OUTPUT_EQUITY = "equity_curve_1h_100_backtestpy.csv"
OUTPUT_TRADES = "backtest_trades_backtestpy.csv"
OUTPUT_METRICS = "backtest_metrics_backtestpy.csv"
OUTPUT_DRAW = "backtest_drawdown_backtestpy.csv"
OUTPUT_LEDGER = "ledger_backtestpy.csv"





res = run_pairs_backtest_with_backtestingpy(INPUT_FILE, SIGNALS_FILE)
trades = res["trades"]
metrics = res["metrics"]
equity = res["equity"]
drawdown = res["drawdown"]
ledger = res["ledger"]

trades.to_csv(OUTPUT_TRADES, index=False)
pd.DataFrame.from_dict(metrics, orient='index').to_csv(OUTPUT_METRICS, index=True)
equity.to_csv(OUTPUT_EQUITY, header=["equity"], index_label="Date")
drawdown.to_csv(OUTPUT_DRAW, header=["drawdown"], index_label="Date")
ledger.to_csv(OUTPUT_LEDGER, index=False)

print("Backtest completed.")
print("Trades saved to:", OUTPUT_TRADES)
print("\nMetrics:")
for k, v in metrics.items():
    try:
        print(f"{k}: {v:.6f}")
    except:
        print(f"{k}: {v}")
