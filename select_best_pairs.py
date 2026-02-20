import argparse
import csv
import json
import os
import math
from typing import List, Dict, Tuple


def _parse_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _parse_int(v, default=0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _valid_pair(pair: str) -> bool:
    if not isinstance(pair, str) or "-" not in pair:
        return False
    a, b = [x.strip().upper() for x in pair.split("-", 1)]
    return bool(a) and bool(b)


def _canonical_pair(pair: str) -> Tuple[str, str]:
    a, b = [x.strip().upper() for x in pair.split("-", 1)]
    return (a, b) if a <= b else (b, a)


def load_stats(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair = (row.get("pair") or "").strip().upper()
            if not _valid_pair(pair):
                continue
            rows.append(
                {
                    "pair": pair,
                    "trade_count": _parse_int(row.get("trade_count"), 0),
                    "total_pnl": _parse_float(row.get("total_pnl"), 0.0),
                    "avg_pnl": _parse_float(row.get("avg_pnl"), 0.0),
                    "std_pnl": _parse_float(row.get("std_pnl"), 0.0),
                    "win_rate": _parse_float(row.get("win_rate"), 0.0),
                    "trade_sharpe": _parse_float(row.get("trade_sharpe"), 0.0),
                }
            )
    return rows


def build_pairs(
    rows: List[Dict],
    min_trades: int,
    min_positive_trades: int,
    min_total_pnl: float,
    min_win_rate: float,
) -> List[str]:
    filtered: List[Dict] = []
    for r in rows:
        if r["trade_count"] < min_trades:
            continue
        if (float(r["trade_count"]) * float(r["win_rate"])) < float(min_positive_trades):
            continue
        if r["total_pnl"] < min_total_pnl:
            continue
        if r["win_rate"] < min_win_rate:
            continue
        filtered.append(r)

    filtered.sort(
        key=lambda x: (
            x["total_pnl"],
            x["trade_count"],
            x["win_rate"],
            x["trade_sharpe"],
            x["avg_pnl"],
        ),
        reverse=True,
    )

    out: List[str] = []
    seen = set()
    for r in filtered:
        pair = r["pair"]
        ck = _canonical_pair(pair)
        if ck in seen:
            continue
        seen.add(ck)
        out.append(pair)
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _score_row(r: Dict) -> float:
    trade_count = max(0, int(r["trade_count"]))
    total_pnl = float(r["total_pnl"])
    win_rate = _clamp(float(r["win_rate"]), 0.0, 1.0)
    sharpe = float(r["trade_sharpe"])

    trade_score = _clamp(trade_count / 12.0, 0.0, 1.0)
    pnl_score = _clamp(math.tanh(total_pnl / 1000.0), 0.0, 1.0)
    sharpe_score = _clamp(sharpe / 2.0, 0.0, 1.0)

    score = (
        0.40 * win_rate
        + 0.25 * trade_score
        + 0.20 * sharpe_score
        + 0.15 * pnl_score
    )
    return float(score)


def build_pairs_rich(
    rows: List[Dict],
    min_trades: int,
    min_positive_trades: int,
    min_total_pnl: float,
    min_win_rate: float,
) -> List[Dict]:
    filtered: List[Dict] = []
    for r in rows:
        if r["trade_count"] < min_trades:
            continue
        if (float(r["trade_count"]) * float(r["win_rate"])) < float(min_positive_trades):
            continue
        if r["total_pnl"] < min_total_pnl:
            continue
        if r["win_rate"] < min_win_rate:
            continue
        item = dict(r)
        item["score"] = _score_row(item)
        filtered.append(item)

    filtered.sort(
        key=lambda x: (
            x["score"],
            x["total_pnl"],
            x["trade_count"],
            x["win_rate"],
            x["trade_sharpe"],
            x["avg_pnl"],
        ),
        reverse=True,
    )

    out: List[Dict] = []
    seen = set()
    for r in filtered:
        pair = str(r["pair"]).upper()
        ck = _canonical_pair(pair)
        if ck in seen:
            continue
        seen.add(ck)
        out.append(
            {
                "pair": pair,
                "score": round(float(r["score"]), 6),
                "trade_count": int(r["trade_count"]),
                "total_pnl": round(float(r["total_pnl"]), 8),
                "avg_pnl": round(float(r["avg_pnl"]), 8),
                "std_pnl": round(float(r.get("std_pnl", 0.0)), 8),
                "win_rate": round(float(r["win_rate"]), 6),
                "trade_sharpe": round(float(r["trade_sharpe"]), 6),
                "source": "analyzed_pairs_stats",
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild best_pairs.json from analyzed_pairs_stats.csv"
    )
    parser.add_argument(
        "--input",
        default=os.path.join("market_neutral", "analyzed_pairs_stats.csv"),
        help="Path to analyzed_pairs_stats.csv",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("market_neutral", "best_pairs.json"),
        help="Path to output best_pairs.json",
    )
    parser.add_argument("--min-trades", type=int, default=3, help="Minimum trade_count (min=3)")
    parser.add_argument(
        "--min-positive-trades",
        type=int,
        default=3,
        help="Minimum count of profitable trades estimate: trade_count * win_rate (min=3)",
    )
    parser.add_argument(
        "--min-total-pnl",
        type=float,
        default=0.0,
        help="Minimum total_pnl to keep pair",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.5,
        help="Minimum win_rate [0..1] to keep pair",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create output backup (.bak) before overwrite",
    )
    parser.add_argument(
        "--rich",
        action="store_true",
        help="Write rich objects with metrics/score instead of plain pair strings",
    )
    args = parser.parse_args()

    args.min_trades = max(3, int(args.min_trades or 0))
    args.min_positive_trades = max(3, int(args.min_positive_trades or 0))

    rows = load_stats(args.input)
    pairs = build_pairs(
        rows=rows,
        min_trades=args.min_trades,
        min_positive_trades=args.min_positive_trades,
        min_total_pnl=args.min_total_pnl,
        min_win_rate=args.min_win_rate,
    )
    payload = (
        build_pairs_rich(
            rows=rows,
            min_trades=args.min_trades,
            min_positive_trades=args.min_positive_trades,
            min_total_pnl=args.min_total_pnl,
            min_win_rate=args.min_win_rate,
        )
        if args.rich
        else pairs
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.backup and os.path.exists(args.output):
        backup_path = f"{args.output}.bak"
        with open(args.output, "r", encoding="utf-8") as src, open(
            backup_path, "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(
        f"Rebuilt {args.output}: {len(payload)} entries "
        f"(input rows={len(rows)}, min_trades={args.min_trades}, "
        f"min_positive_trades={args.min_positive_trades}, "
        f"min_total_pnl={args.min_total_pnl}, min_win_rate={args.min_win_rate}, "
        f"rich={args.rich})"
    )


if __name__ == "__main__":
    main()
