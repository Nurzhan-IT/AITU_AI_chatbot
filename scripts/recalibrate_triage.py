"""Recalibrate the retrieval-grounded triage thresholds — phase C, §3.3 / §3.5 R2.

Reads the phase-B probe score distribution (``scripts/score_distribution.csv``,
produced by ``scripts/score_distribution.py``) and derives the four Stage-3
triage thresholds from REAL percentiles of that distribution — never from round
numbers. The §3.3 illustrative constants (top1 0.80, gap 0.15, top5-mean 0.70,
out-of-scope 0.50) are deliberately NOT usable as defaults: on the project
embedder (``intfloat/multilingual-e5-large``) cosine is compressed into a narrow
~0.70–0.92 band, so those constants make rules A and B fire essentially never.

Why this script exists (§3.5 R2): triage thresholds drift when the embedder or
the corpus changes. When that happens, re-run ``scripts/score_distribution.py``
to refresh the CSV, then re-run this script to regenerate the thresholds.

Thresholds derived (all read by ``rag/dialog/triage.py`` via ``config.Settings``):
  triage_specific_gap_ratio     rule B — (top1-top2)/std must be >= this
  triage_specific_max_entropy   rule B — normalized entropy must be <= this
  triage_ambiguous_min_entropy  rule C — normalized entropy must be >= this
  triage_ambiguous_doc_spread   rule C — distinct docs in top-5 must be >= this

Rule A (out_of_scope) is deliberately NOT given its own threshold: it reuses
``settings.min_chunk_score`` as the shared noise floor (§3.3 point 3). This
script only REPORTS how rule A behaves on the measured top1 distribution — it
never introduces a parallel constant.

The CSV features map straight onto the triage features computed in
``rag/dialog/triage.py`` (same probe, same maths):
    gap_ratio = gap / std_topk        entropy = entropy_norm        doc_spread

Usage:
    python -m scripts.recalibrate_triage                  # report + paste block
    python -m scripts.recalibrate_triage --write          # patch config.py
    python -m scripts.recalibrate_triage --csv x.csv --gap-pct 90 --write
"""

import argparse
import csv
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from config import TZ_UTC5, settings

# Percentiles printed in the diagnostic feature table.
_TABLE_PCTS = [5, 10, 25, 50, 75, 90, 95]

# config.py field lines, used to render the paste-ready block. The --write path
# regex-patches the live file instead, so it tolerates minor formatting drift.
_FIELD_LINES = {
    "triage_specific_gap_ratio":
        "    triage_specific_gap_ratio: float | None = Field(default={v}, ge=0.0)",
    "triage_specific_max_entropy":
        "    triage_specific_max_entropy: float | None = Field(default={v}, ge=0.0, le=1.0)",
    "triage_ambiguous_min_entropy":
        "    triage_ambiguous_min_entropy: float | None = Field(default={v}, ge=0.0, le=1.0)",
    "triage_ambiguous_doc_spread":
        "    triage_ambiguous_doc_spread: int | None = Field(default={v}, ge=1)",
}

_MARKER_RE = re.compile(
    r"^[ \t]*#\ Calibrated\ .*\ by\ scripts/recalibrate_triage\.py\.[ \t]*$",
    re.MULTILINE,
)


# --- CSV loading ----------------------------------------------------------

def _to_float(raw: str | None) -> float | None:
    """Parse a CSV cell to float; empty / non-numeric / NaN → None."""
    if raw is None or raw.strip() == "":
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def load_rows(csv_path: str) -> list[dict]:
    """Load score_distribution.csv into per-query feature records.

    Each record exposes the triage features: ``top1``, ``gap_ratio``,
    ``entropy``, ``doc_spread`` (any of which may be None when the source row
    lacked the inputs to compute it).
    """
    path = Path(csv_path)
    if not path.exists():
        sys.exit(
            f"ERROR: score distribution CSV not found at '{csv_path}'.\n"
            f"Generate it first on the bot host:\n"
            f"    python -m scripts.score_distribution --out {csv_path}"
        )
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"top1", "gap", "std_topk", "entropy_norm", "doc_spread"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(
                f"ERROR: CSV '{csv_path}' is missing columns {sorted(missing)}. "
                f"Regenerate it with the current scripts/score_distribution.py."
            )
        for raw in reader:
            top1 = _to_float(raw.get("top1"))
            gap = _to_float(raw.get("gap"))
            std = _to_float(raw.get("std_topk"))
            entropy = _to_float(raw.get("entropy_norm"))
            doc_spread = _to_float(raw.get("doc_spread"))
            n_results = _to_float(raw.get("n_results"))
            # gap_ratio is the scale-independent rule-B feature (§3.3): the
            # top1-top2 gap measured in std units of the probe score spread.
            gap_ratio = (
                gap / std if (gap is not None and std is not None and std > 1e-9)
                else None
            )
            rows.append({
                "top1": top1,
                "gap_ratio": gap_ratio,
                "entropy": entropy,
                "doc_spread": doc_spread,
                "n_results": n_results,
            })
    return rows


def _column(rows: list[dict], name: str) -> list[float]:
    """Non-None values of one feature column."""
    return [r[name] for r in rows if r[name] is not None]


# --- threshold derivation -------------------------------------------------

def derive_thresholds(rows: list[dict], args: argparse.Namespace) -> dict:
    """Derive the four Stage-3 thresholds from distribution percentiles.

    Rationale (kept conservative on purpose — §3.5 R1):
    - Rule B (SPECIFIC → skip the dialog) must be high-precision, so it fires
      only on the clearly-peaked tail: a HIGH gap_ratio percentile AND a LOW
      entropy percentile.
    - Rule C (AMBIGUOUS → start a dialog) fires only on the clearly-spread
      tail: a HIGH entropy percentile AND a top-quartile document spread.
    - Everything between the tails stays rule D and is sent to the LLM.
    """
    gap_vals = _column(rows, "gap_ratio")
    ent_vals = _column(rows, "entropy")
    spread_vals = _column(rows, "doc_spread")

    for name, vals in (("gap_ratio", gap_vals), ("entropy", ent_vals),
                       ("doc_spread", spread_vals)):
        if not vals:
            sys.exit(f"ERROR: no usable '{name}' values in the CSV — cannot calibrate.")

    gap_ratio = round(float(np.percentile(gap_vals, args.gap_pct)), 4)
    max_entropy = round(float(np.percentile(ent_vals, args.entropy_low_pct)), 4)
    min_entropy = round(float(np.percentile(ent_vals, args.entropy_high_pct)), 4)
    # doc_spread is an integer count of documents — round the percentile up so
    # the threshold is reachable, and never let it drop below 2 (a 1-document
    # top-5 is by definition not "spread").
    doc_spread = max(2, math.ceil(float(np.percentile(spread_vals, args.doc_spread_pct))))

    return {
        "triage_specific_gap_ratio": gap_ratio,
        "triage_specific_max_entropy": max_entropy,
        "triage_ambiguous_min_entropy": min_entropy,
        "triage_ambiguous_doc_spread": doc_spread,
    }


def _simulate_rule(row: dict, thr: dict) -> str:
    """Replay Stage-3 triage for one CSV row — mirrors triage._stage3."""
    top1 = row["top1"]
    if top1 is None:
        return "D"
    if top1 < settings.min_chunk_score:
        return "A"
    gr, ent, ds = row["gap_ratio"], row["entropy"], row["doc_spread"]
    if (gr is not None and ent is not None
            and gr >= thr["triage_specific_gap_ratio"]
            and ent <= thr["triage_specific_max_entropy"]):
        return "B"
    if (ent is not None and ds is not None
            and ent >= thr["triage_ambiguous_min_entropy"]
            and ds >= thr["triage_ambiguous_doc_spread"]):
        return "C"
    return "D"


# --- reporting ------------------------------------------------------------

def _print_feature_table(rows: list[dict]) -> None:
    print("\n--- Feature percentiles ---")
    header = f"{'feature':<12}{'n':>6}{'min':>9}{'max':>9}" + \
        "".join(f"{'p' + str(p):>9}" for p in _TABLE_PCTS)
    print(header)
    print("-" * len(header))
    for name in ("top1", "gap_ratio", "entropy", "doc_spread"):
        vals = _column(rows, name)
        if not vals:
            print(f"{name:<12}{0:>6}{'(no data)':>9}")
            continue
        arr = np.asarray(vals, dtype=float)
        line = f"{name:<12}{len(vals):>6}{arr.min():>9.3f}{arr.max():>9.3f}"
        line += "".join(f"{v:>9.3f}" for v in np.percentile(arr, _TABLE_PCTS))
        print(line)


def _print_rule_a_report(rows: list[dict]) -> None:
    top1 = _column(rows, "top1")
    floor = settings.min_chunk_score
    below = sum(1 for v in top1 if v < floor)
    print("\n--- Rule A (out_of_scope) ---")
    print(f"Reuses settings.min_chunk_score = {floor} as the shared noise floor "
          f"(design-doc 3.3 point 3 - not recalibrated here).")
    if top1:
        pct = 100.0 * below / len(top1)
        print(f"Queries with top1 < {floor}: {below}/{len(top1)} ({pct:.1f}%). "
              f"Observed top1 range: [{min(top1):.3f} .. {max(top1):.3f}].")
        if below == 0:
            print("  -> rule A is a dormant safety net on this corpus, as expected "
                  "for the e5 embedder (3.3). This is fine - do NOT lower the floor "
                  "just to make it fire.")
        if floor > float(np.percentile(top1, 50)):
            print("  WARNING: min_chunk_score sits above the median top1 - it may "
                  "be too aggressive. Review it as a corpus-quality decision.")


def _print_firing_estimate(rows: list[dict], thr: dict) -> None:
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in rows:
        counts[_simulate_rule(r, thr)] += 1
    total = len(rows) or 1
    print("\n--- Rule firing estimate (replaying Stage 3 on this CSV) ---")
    labels = {"A": "out_of_scope", "B": "specific (direct)",
              "C": "ambiguous (dialog)", "D": "borderline (-> LLM)"}
    for rule in ("A", "B", "C", "D"):
        n = counts[rule]
        print(f"  rule {rule}  {labels[rule]:<22} {n:>5}/{total}  ({100.0 * n / total:5.1f}%)")
    for rule in ("B", "C"):
        share = 100.0 * counts[rule] / total
        if share < 1.0:
            print(f"  WARNING: rule {rule} fires on <1% of queries - it is nearly "
                  f"dead. Loosen its percentile (this is the 3.3 failure mode).")
        elif share > 60.0:
            print(f"  WARNING: rule {rule} fires on >60% of queries - it is too "
                  f"greedy. Tighten its percentile.")


def build_config_block(thr: dict, meta: dict) -> str:
    marker = (f"    # Calibrated {meta['date']} from {meta['csv']} "
              f"(N={meta['n']}, k={meta['k']}) by scripts/recalibrate_triage.py.")
    lines = [marker]
    for field, template in _FIELD_LINES.items():
        lines.append(template.format(v=thr[field]))
    return "\n".join(lines)


# --- config.py patching (--write) -----------------------------------------

def write_config(config_path: str, thr: dict, meta: dict) -> None:
    path = Path(config_path)
    if not path.exists():
        sys.exit(f"ERROR: config file '{config_path}' not found.")
    text = original = path.read_text(encoding="utf-8")

    for field, value in thr.items():
        # Replace whatever currently sits in default=... (None or a stale
        # number) so the script is safe to re-run after a drift recalibration.
        pat = re.compile(rf"({re.escape(field)}:[^=\n]+=\s*Field\(default=)[^,)]+")
        text, n = pat.subn(rf"\g<1>{value}", text)
        if n != 1:
            sys.exit(
                f"ERROR: expected exactly one '{field}' Field(...) in "
                f"{config_path}, found {n}. Patch config.py by hand instead."
            )

    marker = (f"    # Calibrated {meta['date']} from {meta['csv']} "
              f"(N={meta['n']}, k={meta['k']}) by scripts/recalibrate_triage.py.")
    if _MARKER_RE.search(text):
        text = _MARKER_RE.sub(lambda _m: marker, text, count=1)
    else:
        anchor = re.search(r"^[ \t]*triage_specific_gap_ratio:", text, re.MULTILINE)
        if not anchor:
            sys.exit("ERROR: could not locate the triage threshold fields to "
                     "anchor the calibration marker.")
        text = text[:anchor.start()] + marker + "\n" + text[anchor.start():]

    if text == original:
        print(f"{config_path} already matches the calibration — nothing to write.")
        return
    path.write_text(text, encoding="utf-8")
    print(f"Patched {config_path}: 4 triage thresholds + calibration marker.")


# --- CLI ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recalibrate triage thresholds from the phase-B score CSV.",
    )
    p.add_argument("--csv", default="scripts/score_distribution.csv",
                   help="score_distribution.csv path (default: %(default)s)")
    p.add_argument("--config", default="config.py",
                   help="config.py path to patch with --write (default: %(default)s)")
    p.add_argument("--write", action="store_true",
                   help="patch config.py in place (default: print only)")
    p.add_argument("--gap-pct", type=float, default=85.0,
                   help="percentile of gap_ratio -> triage_specific_gap_ratio "
                        "(default: %(default)s - clearly-peaked tail)")
    p.add_argument("--entropy-low-pct", type=float, default=15.0,
                   help="percentile of entropy -> triage_specific_max_entropy "
                        "(default: %(default)s - concentrated tail)")
    p.add_argument("--entropy-high-pct", type=float, default=85.0,
                   help="percentile of entropy -> triage_ambiguous_min_entropy "
                        "(default: %(default)s - flat/spread tail)")
    p.add_argument("--doc-spread-pct", type=float, default=75.0,
                   help="percentile of doc_spread -> triage_ambiguous_doc_spread "
                        "(default: %(default)s - top quartile of spread)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rows = load_rows(args.csv)
    if not rows:
        sys.exit(f"ERROR: '{args.csv}' has no data rows.")
    if len(rows) < 200:
        print(f"WARNING: only {len(rows)} queries (< 200) — percentiles will be "
              f"noisy. See phase B3 before trusting this calibration.",
              file=sys.stderr)

    n_results = _column(rows, "n_results")
    k = int(np.median(n_results)) if n_results else 0
    meta = {
        "date": datetime.now(TZ_UTC5).date().isoformat(),
        "csv": args.csv.replace("\\", "/"),
        "n": len(rows),
        "k": k,
    }

    print(f"=== Triage threshold calibration ===")
    print(f"Source CSV : {meta['csv']}")
    print(f"Queries    : {meta['n']} rows (probe k~{k})")
    print(f"Percentiles: gap={args.gap_pct} entropy_low={args.entropy_low_pct} "
          f"entropy_high={args.entropy_high_pct} doc_spread={args.doc_spread_pct}")

    _print_feature_table(rows)
    _print_rule_a_report(rows)

    thr = derive_thresholds(rows, args)
    print("\n--- Derived thresholds ---")
    print(f"  triage_specific_gap_ratio    = {thr['triage_specific_gap_ratio']:<8} "
          f"(P{args.gap_pct:g} of gap_ratio)")
    print(f"  triage_specific_max_entropy  = {thr['triage_specific_max_entropy']:<8} "
          f"(P{args.entropy_low_pct:g} of entropy)")
    print(f"  triage_ambiguous_min_entropy = {thr['triage_ambiguous_min_entropy']:<8} "
          f"(P{args.entropy_high_pct:g} of entropy)")
    print(f"  triage_ambiguous_doc_spread  = {thr['triage_ambiguous_doc_spread']:<8} "
          f"(ceil P{args.doc_spread_pct:g} of doc_spread)")

    _print_firing_estimate(rows, thr)

    print("\n--- config.py block (Settings) ---")
    print(build_config_block(thr, meta))

    if args.write:
        print()
        write_config(args.config, thr, meta)
    else:
        print("\nRun again with --write to patch config.py in place.")


if __name__ == "__main__":
    main()
