"""
Traffic Replay Engine for Hybrid NIDS
=====================================
Streams real CICIDS2017 flows to the running NIDS API in real time, so the
dashboard fills with live alerts during a demo.

Because the flows carry ground-truth labels, the replay also reports live
detection accuracy — what the model got right, what it missed, and what it
falsely flagged.

Usage
-----
    python src/replay.py --scenario demo
    python src/replay.py --scenario ddos --rate 10 --count 50
    python src/replay.py --list

Run the API first (run.bat / run.sh), then run this in a second terminal.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

try:
    import pandas as pd
    import requests
except ImportError as e:
    sys.exit(f"Missing dependency: {e.name}. Run: pip install pandas pyarrow requests")


# ── Config ──────────────────────────────────────────────────────
DEFAULT_API = "http://localhost:8000"
DEFAULT_DATA = "data/processed/test.parquet"
LABEL_COL = "Label"

# CICIDS2017 ships some Web Attack labels with mojibake; normalise for display.
def clean_label(lbl: str) -> str:
    return (
        str(lbl)
        .replace("�", "-")
        .replace("ï¿½", "-")
        .replace("--", "-")
        .strip()
    )


# ── Scenarios ───────────────────────────────────────────────────
# Each scenario is a list of (label_pattern, count) phases played in order.
# "BENIGN" phases provide the quiet baseline that makes attacks stand out.
SCENARIOS: dict[str, dict] = {
    "demo": {
        "desc": "Full story: calm -> DDoS burst -> port scan -> calm. Best for a live demo.",
        "phases": [
            ("BENIGN", 12),
            ("DDoS", 15),
            ("BENIGN", 5),
            ("PortScan", 12),
            ("BENIGN", 6),
            ("DoS Hulk", 10),
            ("BENIGN", 8),
        ],
    },
    "ddos": {
        "desc": "Sustained DDoS flood against a benign baseline.",
        "phases": [("BENIGN", 8), ("DDoS", 30), ("BENIGN", 5)],
    },
    "portscan": {
        "desc": "Reconnaissance sweep — many short probe flows.",
        "phases": [("BENIGN", 8), ("PortScan", 30), ("BENIGN", 5)],
    },
    "bruteforce": {
        "desc": "Credential attacks: FTP-Patator and SSH-Patator.",
        "phases": [("BENIGN", 8), ("FTP-Patator", 15), ("SSH-Patator", 15), ("BENIGN", 5)],
    },
    "stealth": {
        "desc": "Hard cases — the rare classes the model struggles with.",
        "phases": [("BENIGN", 10), ("Bot", 10), ("Web Attack", 10), ("Infiltration", 5)],
    },
    "benign": {
        "desc": "Clean traffic only — shows the false-positive rate.",
        "phases": [("BENIGN", 40)],
    },
    "mixed": {
        "desc": "Realistic blend, roughly the natural class distribution.",
        "phases": [("*", 60)],
    },
}


# ── Data loading ────────────────────────────────────────────────
def load_flows(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        sys.exit(
            f"Data file not found: {p}\n"
            f"Expected the processed CICIDS2017 test set. "
            f"Pass a different path with --data."
        )
    print(f"  Loading {p} ...", end=" ", flush=True)
    df = pd.read_parquet(p)
    print(f"{len(df):,} flows, {df.shape[1] - 1} features")
    if LABEL_COL not in df.columns:
        sys.exit(f"Expected a '{LABEL_COL}' column; found {list(df.columns[-3:])}")
    return df


def pick_rows(df: pd.DataFrame, pattern: str, n: int, rng: random.Random) -> list:
    """Pick n random rows whose label matches pattern ('*' = any)."""
    if pattern == "*":
        pool = df
    else:
        mask = df[LABEL_COL].astype(str).str.contains(pattern, case=False, regex=False)
        pool = df[mask]
        if pool.empty:
            print(f"  [!] No flows matching '{pattern}' — skipping this phase.")
            return []
    take = min(n, len(pool))
    idx = rng.sample(range(len(pool)), take)
    return [pool.iloc[i] for i in idx]


# ── Replay ──────────────────────────────────────────────────────
def replay(args) -> int:
    scenario = SCENARIOS[args.scenario]
    df = load_flows(args.data)
    rng = random.Random(args.seed)

    # Build the flow sequence from the scenario's phases.
    sequence = []
    for pattern, count in scenario["phases"]:
        scaled = max(1, round(count * args.count / 60)) if args.count else count
        sequence.extend(pick_rows(df, pattern, scaled, rng))

    if not sequence:
        sys.exit("No flows selected — nothing to replay.")

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    delay = 1.0 / args.rate if args.rate > 0 else 0.0

    if args.dry_run:
        print()
        print(f"  DRY RUN — {len(sequence)} flows selected, {len(feature_cols)} features each.")
        print("  No API calls will be made.\n")
        breakdown = Counter(clean_label(r[LABEL_COL]) for r in sequence)
        for lbl, n in breakdown.most_common():
            print(f"    {lbl:<30} {n:>4}")
        print()
        return 0

    print()
    print("  " + "=" * 74)
    print(f"   SCENARIO: {args.scenario}  —  {scenario['desc']}")
    print(f"   {len(sequence)} flows at {args.rate}/s  ->  {args.api}")
    print("  " + "=" * 74)
    print()
    print(f"  {'#':>4}  {'ACTUAL':<22} {'PREDICTED':<22} {'CONF':>6} {'ANOM':>5}  {'':<3}")
    print("  " + "-" * 74)

    stats = Counter()
    confusion: list[tuple[str, str]] = []
    session = requests.Session()
    t0 = time.time()

    for i, row in enumerate(sequence, 1):
        truth = clean_label(row[LABEL_COL])
        features = [float(row[c]) for c in feature_cols]

        try:
            r = session.post(f"{args.api}/predict", json={"features": features}, timeout=10)
        except requests.exceptions.ConnectionError:
            print()
            print(f"  [X] Cannot reach the API at {args.api}")
            print("      Start it first with run.bat (Windows) or ./run.sh (Linux/WSL).")
            return 1
        except requests.exceptions.Timeout:
            stats["timeout"] += 1
            continue

        if r.status_code != 200:
            print(f"  [X] API error {r.status_code}: {r.text[:120]}")
            return 1

        d = r.json()
        pred = clean_label(d["prediction"])
        conf = d["confidence"]
        anom = d["is_anomaly"]

        truth_attack = truth.upper() != "BENIGN"
        pred_attack = pred.upper() != "BENIGN"

        if truth_attack and pred_attack:
            mark, key = "OK", "true_positive"
        elif not truth_attack and not pred_attack:
            mark, key = "OK", "true_negative"
        elif truth_attack and not pred_attack:
            mark, key = "MISS", "false_negative"
        else:
            mark, key = "FP", "false_positive"

        stats[key] += 1
        if key in ("false_negative", "false_positive"):
            confusion.append((truth, pred))

        print(
            f"  {i:>4}  {truth[:22]:<22} {pred[:22]:<22} "
            f"{conf * 100:>5.1f}% {'Y' if anom else 'n':>5}  {mark:<4}"
        )

        if delay:
            time.sleep(delay)

    elapsed = time.time() - t0
    _summary(stats, confusion, elapsed, len(sequence))
    return 0


def _summary(stats, confusion, elapsed, total):
    tp, tn = stats["true_positive"], stats["true_negative"]
    fn, fp = stats["false_negative"], stats["false_positive"]

    attacks = tp + fn
    benign = tn + fp
    recall = tp / attacks if attacks else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / benign if benign else 0.0

    print("  " + "-" * 74)
    print()
    print("  " + "=" * 74)
    print("   REPLAY SUMMARY")
    print("  " + "=" * 74)
    print(f"   Flows replayed       {total}  in {elapsed:.1f}s  ({total / elapsed:.1f}/s)")
    print()
    print(f"   Attacks detected     {tp} / {attacks}        (recall    {recall:6.1%})")
    print(f"   Attacks missed       {fn}")
    print(f"   False alarms         {fp} / {benign}        (FP rate   {fpr:6.1%})")
    print(f"   Precision                              {precision:6.1%}")
    print(f"   F1 (attack vs benign)                  {f1:6.1%}")
    print("  " + "=" * 74)

    if confusion:
        print()
        print("   Errors worth explaining:")
        for truth, pred in Counter(confusion).most_common(6):
            n = Counter(confusion)[(truth, pred)]
            print(f"     {truth[:28]:<28} -> {pred[:24]:<24} x{n}")
    print()
    print("   Open http://localhost:8000 to see these alerts on the dashboard.")
    print()


# ── CLI ─────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Replay real CICIDS2017 traffic through the Hybrid NIDS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:  python src/replay.py --scenario demo --rate 3",
    )
    ap.add_argument("--scenario", default="demo", choices=list(SCENARIOS),
                    help="which traffic pattern to replay (default: demo)")
    ap.add_argument("--rate", type=float, default=3.0,
                    help="flows per second (default: 3; use 0 for no delay)")
    ap.add_argument("--count", type=int, default=0,
                    help="approximate total flows; 0 keeps the scenario's own size")
    ap.add_argument("--data", default=DEFAULT_DATA, help=f"flow source (default: {DEFAULT_DATA})")
    ap.add_argument("--api", default=DEFAULT_API, help=f"API base URL (default: {DEFAULT_API})")
    ap.add_argument("--seed", type=int, default=None, help="random seed for a repeatable run")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be replayed without calling the API")
    args = ap.parse_args()

    if args.list:
        print("\n  Available scenarios:\n")
        for name, s in SCENARIOS.items():
            print(f"    {name:<12} {s['desc']}")
        print()
        return 0

    print()
    print("  Hybrid NIDS — Traffic Replay")
    try:
        return replay(args)
    except KeyboardInterrupt:
        print("\n\n  Stopped by user.\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
