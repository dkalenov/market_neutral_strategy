import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def canonical_pair(symbol1: str, symbol2: str) -> tuple[str, str]:
    return tuple(sorted((str(symbol1).strip().upper(), str(symbol2).strip().upper())))


def load_pairs_from_json(path: str) -> list[tuple[str, str]]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        entries: list[Any] = raw
    elif isinstance(raw, dict) and isinstance(raw.get("pairs"), list):
        entries = raw["pairs"]
    else:
        raise ValueError(f"Unsupported pair file format: {path}")

    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
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


def normalize_klines_csv(path: Path, symbols: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required.difference(cols)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    date_col = cols["date"]
    symbol_col = cols["symbol"]
    out = df.loc[df[symbol_col].astype(str).str.upper().isin(symbols), [
        date_col,
        symbol_col,
        cols["open"],
        cols["high"],
        cols["low"],
        cols["close"],
        cols["volume"],
    ]].copy()
    out = out.rename(
        columns={
            date_col: "Date",
            symbol_col: "Symbol",
            cols["open"]: "Open",
            cols["high"]: "High",
            cols["low"]: "Low",
            cols["close"]: "Close",
            cols["volume"]: "Volume",
        }
    )
    out["Symbol"] = out["Symbol"].astype(str).str.upper().str.strip()
    out["Date"] = pd.to_datetime(out["Date"], utc=True, errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Date", "Symbol", "Open", "High", "Low", "Close"])
    out = out[(out["Open"] > 0) & (out["High"] > 0) & (out["Low"] > 0) & (out["Close"] > 0)]
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare merged klines input for walk-forward selector.")
    p.add_argument("--best-pairs", required=True, help="Pair universe JSON.")
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more klines CSV files to merge.",
    )
    p.add_argument("--output-dir", required=True, help="Directory for merged artifacts.")
    p.add_argument("--output-prefix", default="walkforward_input", help="Prefix for output files.")
    p.add_argument("--start", required=True, help="UTC-inclusive start date, e.g. 2024-05-24.")
    p.add_argument("--end", required=True, help="UTC-exclusive end date, e.g. 2026-04-09.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs_from_json(args.best_pairs)
    symbols = sorted({s for key in pairs for s in key})
    symbol_set = set(symbols)

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    if end_ts <= start_ts:
        raise ValueError("end must be greater than start")

    frames = []
    for input_path in args.inputs:
        p = Path(input_path)
        frame = normalize_klines_csv(p, symbol_set)
        frames.append(frame)
        print(f"[INFO] loaded {p.name}: rows={len(frame)}")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["Symbol", "Date"]).drop_duplicates(["Symbol", "Date"], keep="last")
    merged = merged[(merged["Date"] >= start_ts) & (merged["Date"] < end_ts)].copy()
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d %H:%M:%S+0000")

    csv_path = out_dir / f"{args.output_prefix}_klines.csv"
    symbols_path = out_dir / f"{args.output_prefix}_symbols.txt"
    meta_path = out_dir / f"{args.output_prefix}_meta.json"

    merged.to_csv(csv_path, index=False)
    symbols_path.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    meta = {
        "best_pairs_path": str(Path(args.best_pairs).resolve()),
        "inputs": [str(Path(x).resolve()) for x in args.inputs],
        "symbol_count": len(symbols),
        "pair_count": len(pairs),
        "row_count": int(len(merged)),
        "start": str(start_ts),
        "end": str(end_ts),
        "date_min": str(merged["Date"].min()) if not merged.empty else "",
        "date_max": str(merged["Date"].max()) if not merged.empty else "",
        "csv_path": str(csv_path.resolve()),
        "symbols_path": str(symbols_path.resolve()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] merged klines -> {csv_path}")
    print(f"[DONE] symbols list  -> {symbols_path}")
    print(f"[DONE] meta          -> {meta_path}")


if __name__ == "__main__":
    main()
