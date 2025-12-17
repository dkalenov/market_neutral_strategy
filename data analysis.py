import pandas as pd
import numpy as np

TRADES_FILE = "backtest_trades_backtestpy.csv"


df = pd.read_csv(TRADES_FILE, parse_dates=["entry_time", "exit_time"])


df["win"] = df["pnl_net"] > 0
df["commission_ratio"] = np.where(
    df["pnl_gross"] != 0,
    df["commission_total"] / df["pnl_gross"].abs(),
    np.inf
)


summary = {
    "Total trades": len(df),
    "Winrate %": df["win"].mean() * 100,
    "Total PnL net": df["pnl_net"].sum(),
    "Total commissions": df["commission_total"].sum(),
    "Expectancy per trade": df["pnl_net"].mean(),
    "Median PnL": df["pnl_net"].median(),
    "Profit factor": (
        df.loc[df.pnl_net > 0, "pnl_net"].sum()
        / abs(df.loc[df.pnl_net < 0, "pnl_net"].sum())
        if (df.pnl_net < 0).any() else np.nan
    ),
}

print("\n=== OVERALL ===")
for k, v in summary.items():
    print(f"{k:25s}: {v:.4f}")

# ---------------- BY EXIT REASON
reason_stats = df.groupby("exit_reason").agg(
    trades=("pnl_net", "count"),
    winrate=("win", "mean"),
    avg_pnl=("pnl_net", "mean"),
    median_pnl=("pnl_net", "median"),
    total_pnl=("pnl_net", "sum"),
    avg_hold=("hold_hours", "mean"),
    avg_commission=("commission_total", "mean"),
)

reason_stats["winrate"] *= 100

print("\n=== BY EXIT REASON ===")
print(reason_stats.sort_values("total_pnl"))


commission_analysis = {
    "Trades with pnl_gross ~ 0 (%)":
        (df["pnl_gross"].abs() < 1).mean() * 100,
    "Trades where commission > pnl_gross (%)":
        (df["commission_ratio"] > 1).mean() * 100,
    "Median commission / gross":
        df.replace([np.inf], np.nan)["commission_ratio"].median(),
}

print("\nCOMMISSION ANALYSIS")
for k, v in commission_analysis.items():
    print(f"{k:40s}: {v:.2f}")


hold_corr = df["hold_hours"].corr(df["pnl_net"])

print("\nHOLDING")
print(f"Correlation hold_hours ↔ pnl_net: {hold_corr:.3f}")


print("\nWORST 10 TRADES")
print(
    df.sort_values("pnl_net")
      .head(10)[
          ["pair", "exit_reason", "hold_hours", "pnl_net", "commission_total"]
      ]
)
