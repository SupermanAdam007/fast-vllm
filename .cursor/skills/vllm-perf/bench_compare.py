#!/usr/bin/env python3
"""
bench_compare.py — vLLM CPU performance comparison harness.

Two modes:

  --wrap RAW_JSON
      Enrich a raw `vllm bench latency --output-json` file with metadata
      (batch_size, output_len, label) and compute tokens_per_sec.
      Saves a self-describing full JSON.

  --baseline A.json --candidate B.json
      Compare two full JSONs, print an ANSI table, exit 0 if accepted
      (tokens_per_sec improved >= ACCEPT_THRESHOLD and no metric regressed
      > REJECT_THRESHOLD), else exit 1.

Correct formula
---------------
  tokens_per_sec = batch_size * output_len / avg_latency_s

The raw JSON from vllm bench latency only stores avg_latency (seconds) and
percentiles (also seconds). batch_size and output_len must be supplied via
--batch-size and --output-len when wrapping.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Thresholds
ACCEPT_THRESHOLD = 0.03   # tokens/sec must improve by at least 3%
REJECT_THRESHOLD = 0.05   # no other metric may degrade by more than 5%

# ANSI colours
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ---------------------------------------------------------------------------
# Wrap mode
# ---------------------------------------------------------------------------

def cmd_wrap(args: argparse.Namespace) -> None:
    raw_path = Path(args.wrap)
    if not raw_path.exists():
        sys.exit(f"ERROR: raw JSON not found: {raw_path}")

    with raw_path.open() as f:
        raw: dict[str, Any] = json.load(f)

    avg_latency_s: float = raw["avg_latency"]
    percentiles: dict[str, float] = raw["percentiles"]  # keys: "10","25","50","75","90","99"

    tokens_per_sec = (args.batch_size * args.output_len) / avg_latency_s

    full: dict[str, Any] = {
        "label": args.label,
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "config_flags": args.config_flags or "",
        "avg_latency_s": avg_latency_s,
        "avg_latency_ms": avg_latency_s * 1000,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "p50_ms": percentiles.get("50", percentiles.get(50, 0)) * 1000,
        "p90_ms": percentiles.get("90", percentiles.get(90, 0)) * 1000,
        "p99_ms": percentiles.get("99", percentiles.get(99, 0)) * 1000,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "_raw": raw,
    }

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w") as f:
        json.dump(full, f, indent=2)

    print(f"{BOLD}Wrapped:{RESET} {save_path}")
    print(f"  label          : {full['label']}")
    print(f"  tokens_per_sec : {BOLD}{GREEN}{full['tokens_per_sec']:.1f} tok/s{RESET}")
    print(f"  avg_latency    : {full['avg_latency_ms']:.1f} ms/batch")
    print(f"  p50 / p90 / p99: {full['p50_ms']:.0f} / {full['p90_ms']:.0f} / {full['p99_ms']:.0f} ms")

    if args.append_log:
        _append_log(args.append_log, full)


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> None:
    baseline = _load_full(args.baseline)
    candidate = _load_full(args.candidate)

    # When batch_size differs, latency metrics are NOT comparable:
    # doubling batch_size always doubles per-batch latency regardless of efficiency.
    # In that case, only tokens_per_sec is the valid comparison metric.
    same_batch = baseline.get("batch_size") == candidate.get("batch_size")
    if not same_batch:
        print(
            f"\n{YELLOW}Note: batch_size changed "
            f"({baseline.get('batch_size')} → {candidate.get('batch_size')}). "
            f"Latency metrics are not comparable — evaluating tokens_per_sec only.{RESET}"
        )

    # Warn when sequence lengths differ — tokens_per_sec is incomparable across seq lengths
    # because longer sequences are more memory-bandwidth-bound (lower tok/s per token generated).
    for dim in ("input_len", "output_len"):
        b_val = baseline.get(dim)
        c_val = candidate.get(dim)
        if b_val is not None and c_val is not None and b_val != c_val:
            print(
                f"\n{RED}WARNING: {dim} changed ({b_val} → {c_val}). "
                f"tokens_per_sec is NOT comparable across different sequence lengths. "
                f"Re-run baseline with matching {dim} before drawing conclusions.{RESET}"
            )

    all_metrics = [
        ("tokens_per_sec", "tok/s", True),   # higher is better — always compared
        ("avg_latency_ms", "ms",   False),   # lower is better — only when same batch
        ("p50_ms",         "ms",   False),
        ("p90_ms",         "ms",   False),
        ("p99_ms",         "ms",   False),
    ]
    # Skip latency metrics when batch sizes differ
    metrics = [m for m in all_metrics if same_batch or m[0] == "tokens_per_sec"]

    col_w = [22, 12, 12, 10]
    header = (
        f"{'Metric':<{col_w[0]}} {'Baseline':>{col_w[1]}} {'Candidate':>{col_w[2]}} {'Delta':>{col_w[3]}}"
    )
    sep = "─" * sum(col_w + [3 * 1])

    print(f"\n{BOLD}{header}{RESET}")
    print(sep)

    accepted = True
    primary_ok = False
    rows: list[str] = []

    for key, unit, higher_is_better in metrics:
        base_val = baseline.get(key, 0.0)
        cand_val = candidate.get(key, 0.0)

        if base_val == 0:
            delta_pct = 0.0
        else:
            delta_pct = (cand_val - base_val) / abs(base_val)

        improved = (delta_pct > 0) if higher_is_better else (delta_pct < 0)
        regressed = (not improved) and abs(delta_pct) > 0.001

        sign = "+" if delta_pct >= 0 else ""
        delta_str = f"{sign}{delta_pct * 100:.1f}%"

        if key == "tokens_per_sec":
            if delta_pct >= ACCEPT_THRESHOLD:
                primary_ok = True
                marker = f"{GREEN}✓{RESET}"
            else:
                accepted = False
                marker = f"{RED}✗{RESET}"
        else:
            # Latency metrics — only checked when same_batch is True
            if regressed and abs(delta_pct) > REJECT_THRESHOLD:
                accepted = False
                marker = f"{RED}✗{RESET}"
            elif improved:
                marker = f"{GREEN}✓{RESET}"
            else:
                marker = f"{YELLOW}~{RESET}"

        colour = GREEN if improved else (RED if regressed else RESET)
        base_str = f"{base_val:.1f} {unit}"
        cand_str = f"{cand_val:.1f} {unit}"

        row = (
            f"{key:<{col_w[0]}} {base_str:>{col_w[1]}} {cand_str:>{col_w[2]}} "
            f"{colour}{delta_str:>{col_w[3]}}{RESET} {marker}"
        )
        rows.append(row)

    for row in rows:
        print(row)

    print(sep)

    accepted = accepted and primary_ok

    if accepted:
        verdict = f"{GREEN}{BOLD}ACCEPTED{RESET} — tokens/sec improved ≥ {ACCEPT_THRESHOLD*100:.0f}%"
    else:
        verdict = f"{RED}{BOLD}REJECTED{RESET} — improvement below threshold or regression detected"

    print(f"\nVerdict: {verdict}")
    print(f"  Baseline : {baseline['label']}")
    print(f"  Candidate: {candidate['label']}")

    if args.append_log:
        _append_log(args.append_log, candidate, baseline=baseline, accepted=accepted)

    sys.exit(0 if accepted else 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_full(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"ERROR: JSON not found: {p}")
    with p.open() as f:
        return json.load(f)


def _append_log(log_path: str, record: dict[str, Any], *,
                baseline: dict[str, Any] | None = None,
                accepted: bool | None = None) -> None:
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": record.get("label"),
        "tokens_per_sec": record.get("tokens_per_sec"),
        "avg_latency_ms": record.get("avg_latency_ms"),
        "p99_ms": record.get("p99_ms"),
        "config_flags": record.get("config_flags", ""),
    }
    if baseline is not None:
        base_tps = baseline.get("tokens_per_sec", 0) or 1
        entry["delta_pct"] = round(
            (record.get("tokens_per_sec", 0) - base_tps) / abs(base_tps) * 100, 1
        )
    if accepted is not None:
        entry["accepted"] = accepted

    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="vLLM CPU benchmark comparison harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode")

    # ---- wrap ----
    wrap_p = sub.add_parser("wrap", help="Enrich raw latency JSON with metadata")
    wrap_p.add_argument("raw_json", help="Path to raw --output-json file from vllm bench latency")
    wrap_p.add_argument("--batch-size", type=int, required=True)
    wrap_p.add_argument("--output-len", type=int, required=True)
    wrap_p.add_argument("--input-len", type=int, default=32)
    wrap_p.add_argument("--label", type=str, required=True, help="Short description of this run")
    wrap_p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    wrap_p.add_argument("--device", type=str, default="cpu")
    wrap_p.add_argument("--dtype", type=str, default="bfloat16")
    wrap_p.add_argument("--config-flags", type=str, default="", help="Extra flags used in this run")
    wrap_p.add_argument("--save", type=str, required=True, help="Where to save the full JSON")
    wrap_p.add_argument("--append-log", type=str, default=None, metavar="JSONL",
                        help="Append run record to this JSONL file")

    # ---- compare ----
    cmp_p = sub.add_parser("compare", help="Compare baseline vs candidate, exit 0/1")
    cmp_p.add_argument("--baseline", type=str, required=True, help="Full JSON of baseline run")
    cmp_p.add_argument("--candidate", type=str, required=True, help="Full JSON of candidate run")
    cmp_p.add_argument("--append-log", type=str, default=None, metavar="JSONL",
                        help="Append comparison record to this JSONL file")

    # Legacy flat flags (backward compat — map to subcommands transparently)
    parser.add_argument("--wrap", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--baseline", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--candidate", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--output-len", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--input-len", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--label", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default="cpu", help=argparse.SUPPRESS)
    parser.add_argument("--dtype", type=str, default="bfloat16", help=argparse.SUPPRESS)
    parser.add_argument("--config-flags", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--save", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--append-log", type=str, default=None, help=argparse.SUPPRESS)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Handle legacy flat-flag mode (as specified in SKILL.md)
    if args.mode is None:
        if args.wrap:
            args.mode = "wrap"
        elif args.baseline and args.candidate:
            args.mode = "compare"
        else:
            parser.print_help()
            sys.exit(1)

    if args.mode == "wrap":
        cmd_wrap(args)
    elif args.mode == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
