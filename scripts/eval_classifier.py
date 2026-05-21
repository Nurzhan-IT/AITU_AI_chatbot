"""Golden-set regression test for the intent classifier — phase B6.

The project's first automated test. Reads the hand-labeled golden set
(data/golden_set.jsonl, produced and labeled in phase B5), runs the live
``classify_intent`` on every labeled query, and scores the result:

  * precision / recall / F1 / accuracy for ``needs_clarification``
    (positive class = "clarification needed");
  * a confusion matrix for ``reason`` (specific / vague_topic / ambiguous);
  * a list of every disagreement (query, expected, got).

This is the regression gate for all later prompt edits (phases C and E):
run it once now to fix a baseline, then after every prompt change re-run with
``--baseline`` to see whether the change improved or regressed the classifier.

Baseline workflow (--baseline FILE):
  * FILE does not exist  -> the current run is SAVED to FILE.
  * FILE exists          -> the current run is COMPARED against it; per-case
                            regressions and metric deltas are printed, and the
                            script exits 1 if any case that used to pass now
                            fails (so it can gate CI).
  * --update-baseline    -> overwrite an existing baseline with the current run
                            (use after an intentional, verified improvement).

Establish the phase-A baseline (acceptance criterion) with:
    python -m scripts.eval_classifier --baseline scripts/eval_baseline_phaseA.json

Usage:
    python -m scripts.eval_classifier
    python -m scripts.eval_classifier --baseline scripts/eval_baseline_phaseA.json
    python -m scripts.eval_classifier --golden data/golden_set.jsonl --limit 20
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from config import TZ_UTC5
from rag.dialog.classifier import classify_intent

_VALID_REASONS = ("specific", "vague_topic", "ambiguous")
_DEFAULT_GOLDEN = "data/golden_set.jsonl"
_PROGRESS_EVERY = 10


def load_golden(path: str) -> list[dict]:
    """Read labeled cases from the golden set. Unlabeled lines are skipped."""
    p = Path(path)
    if not p.exists():
        sys.exit(
            f"ERROR: golden set not found at '{path}'. Create it with "
            f"'python -m scripts.export_for_labeling' and label it (phase B5)."
        )

    cases: list[dict] = []
    skipped_unlabeled = 0
    with p.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [skip] line {lineno}: invalid JSON ({e})", file=sys.stderr)
                continue

            query = obj.get("query")
            needs = obj.get("expected_needs_clarification")
            reason = obj.get("expected_reason")

            if not isinstance(query, str) or not query.strip():
                print(f"  [skip] line {lineno}: missing 'query'", file=sys.stderr)
                continue
            if needs is None or reason is None:
                skipped_unlabeled += 1
                continue
            if not isinstance(needs, bool):
                print(f"  [skip] line {lineno}: expected_needs_clarification "
                      f"must be true/false", file=sys.stderr)
                continue
            if reason not in _VALID_REASONS:
                print(f"  [skip] line {lineno}: invalid expected_reason "
                      f"{reason!r}", file=sys.stderr)
                continue

            # B5 consistency rule: false <=> specific, true <=> vague/ambiguous.
            consistent = (
                (needs and reason in ("vague_topic", "ambiguous"))
                or (not needs and reason == "specific")
            )
            if not consistent:
                print(f"  [warn] line {lineno}: expected_needs_clarification="
                      f"{needs} is inconsistent with expected_reason={reason!r}",
                      file=sys.stderr)

            cases.append({
                "query": query.strip(),
                "expected_needs": needs,
                "expected_reason": reason,
            })

    if skipped_unlabeled:
        print(f"Note: skipped {skipped_unlabeled} unlabeled line(s) "
              f"(expected_* still null).", file=sys.stderr)
    if not cases:
        sys.exit(
            "ERROR: no labeled cases in the golden set. Complete the manual "
            "labeling step (phase B5 step 2) first."
        )
    return cases


async def run_eval(cases: list[dict]) -> list[dict]:
    """Run classify_intent on every case; return per-case result dicts."""
    results: list[dict] = []
    total = len(cases)
    for i, case in enumerate(cases, 1):
        verdict = await classify_intent(case["query"])
        got_needs = bool(verdict["needs_clarification"])
        got_reason = str(verdict["reason"])
        results.append({
            "query": case["query"],
            "expected_needs": case["expected_needs"],
            "expected_reason": case["expected_reason"],
            "got_needs": got_needs,
            "got_reason": got_reason,
            "got_confidence": float(verdict["confidence"]),
            "needs_ok": got_needs == case["expected_needs"],
            "reason_ok": got_reason == case["expected_reason"],
        })
        if i % _PROGRESS_EVERY == 0 or i == total:
            print(f"  classified {i}/{total}...", file=sys.stderr)
    return results


def compute_metrics(results: list[dict]) -> dict:
    """Precision/recall/F1 for needs_clarification + reason confusion matrix."""
    tp = fp = fn = tn = 0
    for r in results:
        exp, got = r["expected_needs"], r["got_needs"]
        if exp and got:
            tp += 1
        elif not exp and got:
            fp += 1
        elif exp and not got:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    accuracy = (tp + tn) / len(results) if results else 0.0

    labels = sorted(
        {r["expected_reason"] for r in results}
        | {r["got_reason"] for r in results}
    )
    matrix = {e: {p: 0 for p in labels} for e in labels}
    for r in results:
        matrix[r["expected_reason"]][r["got_reason"]] += 1
    reason_correct = sum(1 for r in results if r["reason_ok"])

    return {
        "n_cases": len(results),
        "needs_clarification": {
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "reason_accuracy": reason_correct / len(results) if results else 0.0,
        "reason_labels": labels,
        "reason_confusion": matrix,
    }


def print_report(metrics: dict, results: list[dict]) -> None:
    nc = metrics["needs_clarification"]
    print(f"\n=== Classifier evaluation (N={metrics['n_cases']} labeled cases) ===\n")
    print("needs_clarification (positive class = clarification needed):")
    print(f"  precision : {nc['precision']:.3f}")
    print(f"  recall    : {nc['recall']:.3f}")
    print(f"  F1        : {nc['f1']:.3f}")
    print(f"  accuracy  : {nc['accuracy']:.3f}")
    print(f"  TP={nc['tp']}  FP={nc['fp']}  FN={nc['fn']}  TN={nc['tn']}")

    labels = metrics["reason_labels"]
    matrix = metrics["reason_confusion"]
    print(f"\nreason confusion matrix (rows=expected, cols=predicted), "
          f"accuracy={metrics['reason_accuracy']:.3f}:")
    col_w = max([11] + [len(lbl) for lbl in labels]) + 2
    print(" " * 13 + "".join(f"{lbl:>{col_w}}" for lbl in labels))
    for e in labels:
        print(f"{e:<13}" + "".join(f"{matrix[e][p]:>{col_w}}" for p in labels))

    bad = [r for r in results if not (r["needs_ok"] and r["reason_ok"])]
    print(f"\ndisagreements ({len(bad)}):")
    if not bad:
        print("  (none - classifier matches the golden set perfectly)")
    for r in bad:
        q = r["query"] if len(r["query"]) <= 70 else r["query"][:67] + "..."
        print(f"  - {q}")
        print(f"      expected: needs={r['expected_needs']} "
              f"reason={r['expected_reason']}")
        print(f"      got:      needs={r['got_needs']} "
              f"reason={r['got_reason']} (conf={r['got_confidence']:.2f})")


def save_baseline(path: str, metrics: dict, results: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(TZ_UTC5).isoformat(),
        "metrics": metrics,
        "per_case": [
            {
                "query": r["query"],
                "expected_needs": r["expected_needs"],
                "expected_reason": r["expected_reason"],
                "got_needs": r["got_needs"],
                "got_reason": r["got_reason"],
                "needs_ok": r["needs_ok"],
                "reason_ok": r["reason_ok"],
            }
            for r in results
        ],
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nBaseline saved to '{path}' ({metrics['n_cases']} cases).")


def compare_baseline(path: str, metrics: dict, results: list[dict]) -> bool:
    """Compare the current run against a saved baseline.

    Returns True if any case that passed in the baseline now fails (regression).
    """
    with Path(path).open(encoding="utf-8") as f:
        base = json.load(f)

    base_nc = base.get("metrics", {}).get("needs_clarification", {})
    base_cases = {c["query"]: c for c in base.get("per_case", [])}
    cur_cases = {r["query"]: r for r in results}

    print(f"\n=== Comparison vs baseline '{path}' "
          f"(created {base.get('created_at', '?')}) ===")
    cur_nc = metrics["needs_clarification"]
    for key in ("precision", "recall", "f1", "accuracy"):
        b = base_nc.get(key, 0.0)
        c = cur_nc[key]
        print(f"  {key:<10} {b:.3f} -> {c:.3f}  ({c - b:+.3f})")

    common = set(base_cases) & set(cur_cases)
    newly_fail, newly_pass = [], []
    for q in sorted(common):
        b_ok = base_cases[q]["needs_ok"] and base_cases[q]["reason_ok"]
        c_ok = cur_cases[q]["needs_ok"] and cur_cases[q]["reason_ok"]
        if b_ok and not c_ok:
            newly_fail.append(q)
        elif not b_ok and c_ok:
            newly_pass.append(q)

    added = set(cur_cases) - set(base_cases)
    removed = set(base_cases) - set(cur_cases)
    if added or removed:
        print(f"\n  golden set changed since baseline: "
              f"+{len(added)} new case(s), -{len(removed)} removed "
              f"(not counted as regressions)")

    print(f"\n  improvements (newly passing): {len(newly_pass)}")
    for q in newly_pass:
        print(f"    + {q[:70]}")
    print(f"  REGRESSIONS (newly failing): {len(newly_fail)}")
    for q in newly_fail:
        c = cur_cases[q]
        print(f"    - {q[:70]}")
        print(f"        got needs={c['got_needs']} reason={c['got_reason']}")

    if newly_fail:
        print(f"\n  FAIL: {len(newly_fail)} regression(s) vs baseline.")
        return True
    print("\n  OK: no regressions vs baseline.")
    return False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Golden-set regression test for the intent classifier (B6)."
    )
    p.add_argument("--golden", default=_DEFAULT_GOLDEN,
                   help="Labeled golden set JSONL (default: %(default)s)")
    p.add_argument("--baseline", default=None,
                   help="Baseline JSON: saved if missing, compared if it exists.")
    p.add_argument("--update-baseline", action="store_true",
                   help="Overwrite an existing baseline with the current run.")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N cases (quick smoke test).")
    return p.parse_args()


async def main() -> None:
    args = _parse_args()

    cases = load_golden(args.golden)
    if args.limit is not None:
        cases = cases[:args.limit]
    print(f"Evaluating {len(cases)} labeled case(s) from '{args.golden}'...")

    results = await run_eval(cases)
    metrics = compute_metrics(results)
    print_report(metrics, results)

    exit_code = 0
    if args.baseline:
        bp = Path(args.baseline)
        if bp.exists() and not args.update_baseline:
            if compare_baseline(args.baseline, metrics, results):
                exit_code = 1
        else:
            if bp.exists():
                print("\n(--update-baseline) overwriting existing baseline.")
            save_baseline(args.baseline, metrics, results)

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
