"""Golden-set regression test for the intent classifier and triage cascade.

Phase B6 — baseline evaluation of ``classify_intent``.
Phase C5 — extended evaluation that also runs the full triage cascade
           (rag/dialog/triage.py) and compares Phase A vs Phase C metrics.

The project's first automated test. Reads the hand-labeled golden set
(data/golden_set.jsonl, produced and labeled in phase B5), runs the live
classifier (and optionally the triage cascade) on every labeled query, and
scores the result:

  * precision / recall / F1 / accuracy for ``needs_clarification``
    (positive class = "clarification needed");
  * a confusion matrix for ``reason`` (specific / vague_topic / ambiguous);
  * a list of every disagreement (query, expected, got).

Baseline workflow (--baseline FILE):
  * FILE does not exist  -> the current run is SAVED to FILE.
  * FILE exists          -> the current run is COMPARED against it; per-case
                            regressions and metric deltas are printed, and the
                            script exits 1 if any case that used to pass now
                            fails (so it can gate CI).
  * --update-baseline    -> overwrite an existing baseline with the current run
                            (use after an intentional, verified improvement).

Triage evaluation (--triage):
  Requires Qdrant to be running (docker compose up -d). For every case, also
  runs the full Stage 1-4 triage cascade alongside classify_intent. Produces a
  Phase A vs Phase C side-by-side comparison. Use --report to save the result
  to eval_phaseC_report.md.

  NOTE: in the current uncalibrated state (triage_calibrated=False) Stage 3
  always returns rule D, so every query reaches Stage 4 (LLM verdict with
  added corpus context). Running --triage therefore doubles LLM calls per case.

Establish the phase-A baseline (acceptance criterion) with:
    python -m scripts.eval_classifier --baseline scripts/eval_baseline_phaseA.json

Run phase-C evaluation (requires Qdrant):
    python -m scripts.eval_classifier \\
        --triage \\
        --baseline scripts/eval_baseline_phaseA.json \\
        --report scripts/eval_phaseC_report.md

Usage:
    python -m scripts.eval_classifier
    python -m scripts.eval_classifier --baseline scripts/eval_baseline_phaseA.json
    python -m scripts.eval_classifier --golden data/golden_set.jsonl --limit 20
    python -m scripts.eval_classifier --triage --report scripts/eval_phaseC_report.md
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


async def run_eval(cases: list[dict], retriever=None) -> list[dict]:
    """Run classify_intent (and optionally the triage cascade) on every case.

    When ``retriever`` is not None the triage cascade is also run for each
    case. Extra result keys are added: ``triage_needs``, ``triage_reason``,
    ``triage_conf``, ``triage_stage``, ``triage_rule``, ``triage_needs_ok``,
    ``triage_reason_ok``. A triage error sets ``triage_needs=None`` and
    ``triage_stage="error"``.
    """
    results: list[dict] = []
    total = len(cases)
    for i, case in enumerate(cases, 1):
        verdict = await classify_intent(case["query"])
        got_needs = bool(verdict["needs_clarification"])
        got_reason = str(verdict["reason"])
        result: dict = {
            "query": case["query"],
            "expected_needs": case["expected_needs"],
            "expected_reason": case["expected_reason"],
            "got_needs": got_needs,
            "got_reason": got_reason,
            "got_confidence": float(verdict["confidence"]),
            "needs_ok": got_needs == case["expected_needs"],
            "reason_ok": got_reason == case["expected_reason"],
        }

        if retriever is not None:
            try:
                from rag.dialog.triage import triage as _run_triage
                t = await _run_triage(case["query"], retriever)
                t_needs = bool(t["needs_clarification"])
                t_reason = str(t["reason"])
                result.update({
                    "triage_needs": t_needs,
                    "triage_reason": t_reason,
                    "triage_conf": float(t["confidence"]),
                    "triage_stage": t["decided_by"],
                    "triage_rule": t["rule"],
                    "triage_needs_ok": t_needs == case["expected_needs"],
                    "triage_reason_ok": t_reason == case["expected_reason"],
                })
            except Exception as exc:
                print(f"  [triage error] q={case['query'][:60]!r}: {exc}",
                      file=sys.stderr)
                result.update({
                    "triage_needs": None,
                    "triage_reason": "error",
                    "triage_conf": 0.5,
                    "triage_stage": "error",
                    "triage_rule": "",
                    "triage_needs_ok": False,
                    "triage_reason_ok": False,
                })

        if i % _PROGRESS_EVERY == 0 or i == total:
            print(f"  classified {i}/{total}...", file=sys.stderr)
        results.append(result)
    return results


def compute_metrics_for(
    results: list[dict],
    needs_key: str = "got_needs",
    reason_key: str = "got_reason",
) -> dict:
    """Precision/recall/F1 for needs_clarification + reason confusion matrix.

    Parametric on which result keys to read from — use ``needs_key="got_needs"``
    for the classifier and ``needs_key="triage_needs"`` for the triage cascade.
    Rows where the specified ``needs_key`` value is None (triage error) are
    counted as false negatives for recall (conservative: we failed to flag them).
    """
    tp = fp = fn = tn = 0
    for r in results:
        if needs_key not in r:
            continue
        exp = r["expected_needs"]
        got = r[needs_key]
        if got is None:
            if exp:
                fn += 1
            # else: error on a true-negative case — don't count as FP
            continue
        if exp and got:
            tp += 1
        elif not exp and got:
            fp += 1
        elif exp and not got:
            fn += 1
        else:
            tn += 1

    n_valid = sum(
        1 for r in results
        if needs_key in r and r[needs_key] is not None
    )
    n_total = sum(1 for r in results if needs_key in r)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    accuracy = (tp + tn) / n_total if n_total else 0.0

    valid_results = [r for r in results if needs_key in r and r[needs_key] is not None]
    labels = sorted(
        {r["expected_reason"] for r in valid_results}
        | {r[reason_key] for r in valid_results if reason_key in r}
    )
    matrix: dict[str, dict[str, int]] = {e: {p: 0 for p in labels} for e in labels}
    reason_correct = 0
    for r in valid_results:
        got_r = r.get(reason_key, "")
        exp_r = r["expected_reason"]
        if exp_r in matrix and got_r in matrix.get(exp_r, {}):
            matrix[exp_r][got_r] += 1
        if got_r == exp_r:
            reason_correct += 1

    return {
        "n_cases": n_total,
        "n_valid": n_valid,
        "needs_clarification": {
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "reason_accuracy": reason_correct / n_valid if n_valid else 0.0,
        "reason_labels": labels,
        "reason_confusion": matrix,
    }


def compute_metrics(results: list[dict]) -> dict:
    """Compute metrics for the legacy classifier (backward-compatible wrapper)."""
    return compute_metrics_for(results, needs_key="got_needs", reason_key="got_reason")


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


def print_triage_comparison(
    clf_m: dict, tri_m: dict, results: list[dict]
) -> None:
    """Print a side-by-side Phase A (classifier) vs Phase C (triage) comparison."""
    cnc = clf_m["needs_clarification"]
    tnc = tri_m["needs_clarification"]

    print("\n=== Phase A (classify_intent) vs Phase C (triage cascade) ===\n")
    print(f"{'Metric':<13} {'Phase A':>10} {'Phase C':>10} {'Delta':>10}")
    print("-" * 45)
    for key in ("precision", "recall", "f1", "accuracy"):
        a = cnc[key]
        c = tnc[key]
        delta = c - a
        marker = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
        print(f"  {key:<11} {a:>10.3f} {c:>10.3f} {delta:>+10.3f}  {marker}")

    ca = clf_m["reason_accuracy"]
    ta = tri_m["reason_accuracy"]
    delta = ta - ca
    marker = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
    print(f"  {'reason_acc':<11} {ca:>10.3f} {ta:>10.3f} {delta:>+10.3f}  {marker}")

    print(f"\n  Phase A n_cases={clf_m['n_cases']}  "
          f"Phase C n_valid={tri_m['n_valid']} "
          f"(errors={tri_m['n_cases'] - tri_m['n_valid']})")

    # Stage breakdown
    stage_counts: dict[str, int] = {}
    for r in results:
        stage = r.get("triage_stage", "n/a")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    print("\nTriage stage breakdown:")
    for stage in ("stage1", "stage2", "stage3", "stage4", "error", "n/a"):
        count = stage_counts.get(stage, 0)
        if count:
            pct = count / len(results) * 100
            print(f"  {stage:<10} {count:>4} ({pct:.0f}%)")

    # Agreement between classifier and triage
    have_triage = [r for r in results if r.get("triage_needs") is not None]
    agree = sum(1 for r in have_triage if r["got_needs"] == r["triage_needs"])
    if have_triage:
        print(f"\nClassifier/triage agreement: {agree}/{len(have_triage)} "
              f"({agree / len(have_triage):.1%})")

    # Cases that improved (classifier wrong, triage right)
    improved = [r for r in have_triage
                if not r["needs_ok"] and r["triage_needs_ok"]]
    # Cases that regressed (classifier right, triage wrong)
    regressed = [r for r in have_triage
                 if r["needs_ok"] and not r["triage_needs_ok"]]

    print(f"\nImprovements (classifier wrong, triage right): {len(improved)}")
    for r in improved[:5]:
        q = r["query"][:65] if len(r["query"]) <= 65 else r["query"][:62] + "..."
        print(f"  + [{r.get('triage_stage','?')}/{r.get('triage_rule','?')}] {q}")

    print(f"REGRESSIONS (classifier right, triage wrong): {len(regressed)}")
    for r in regressed[:5]:
        q = r["query"][:65] if len(r["query"]) <= 65 else r["query"][:62] + "..."
        print(f"  - [{r.get('triage_stage','?')}/{r.get('triage_rule','?')}] {q}")
        print(f"      expected needs={r['expected_needs']} "
              f"triage_needs={r['triage_needs']} "
              f"triage_reason={r.get('triage_reason','?')}")


def write_report(
    path: str,
    clf_m: dict,
    tri_m: dict,
    results: list[dict],
) -> None:
    """Write the Phase C evaluation report as Markdown to ``path``."""
    now = datetime.now(TZ_UTC5).isoformat()
    n = len(results)
    cnc = clf_m["needs_clarification"]
    tnc = tri_m["needs_clarification"]

    stage_counts: dict[str, int] = {}
    for r in results:
        stage_counts[r.get("triage_stage", "n/a")] = (
            stage_counts.get(r.get("triage_stage", "n/a"), 0) + 1
        )

    have_triage = [r for r in results if r.get("triage_needs") is not None]
    improved = [r for r in have_triage if not r["needs_ok"] and r["triage_needs_ok"]]
    regressed = [r for r in have_triage if r["needs_ok"] and not r["triage_needs_ok"]]

    n_errors = tri_m["n_cases"] - tri_m["n_valid"]
    f1_delta = tnc["f1"] - cnc["f1"]
    recommend_enable = (
        f1_delta >= 0.0
        and len(regressed) <= len(improved)
        and n_errors == 0
    )

    lines: list[str] = [
        "# Phase C Evaluation Report — Triage Cascade vs Legacy Classifier",
        "",
        f"_Generated: {now}_  ",
        f"_Evaluated on {n} labeled cases from `data/golden_set.jsonl`_",
        "",
        "## Summary",
        "",
        "| Metric | Phase A (classifier) | Phase C (triage) | Delta |",
        "|--------|---------------------|-----------------|-------|",
    ]
    for key in ("precision", "recall", "f1", "accuracy"):
        a = cnc[key]
        c = tnc[key]
        d = c - a
        marker = "improvement" if d > 0.001 else ("regression" if d < -0.001 else "no change")
        lines.append(
            f"| needs/{key} | {a:.3f} | {c:.3f} | {d:+.3f} ({marker}) |"
        )
    ca = clf_m["reason_accuracy"]
    ta = tri_m["reason_accuracy"]
    d = ta - ca
    marker = "improvement" if d > 0.001 else ("regression" if d < -0.001 else "no change")
    lines.append(f"| reason accuracy | {ca:.3f} | {ta:.3f} | {d:+.3f} ({marker}) |")

    lines += [
        "",
        f"Triage errors (Qdrant/LLM failures during evaluation): **{n_errors}**  ",
        f"Classifier/triage agreement: "
        f"**{sum(1 for r in have_triage if r['got_needs'] == r['triage_needs'])}"
        f"/{len(have_triage)}** "
        f"({sum(1 for r in have_triage if r['got_needs'] == r['triage_needs']) / len(have_triage):.1%}"
        f" of cases where triage ran)" if have_triage else "",
        "",
        "## Triage Stage Breakdown",
        "",
        "| Stage | Count | % |",
        "|-------|-------|---|",
    ]
    for stage in ("stage1", "stage2", "stage3", "stage4", "error"):
        count = stage_counts.get(stage, 0)
        if count:
            lines.append(f"| {stage} | {count} | {count / n:.0%} |")

    lines += [
        "",
        "**Note:** In the current uncalibrated state (`triage_calibrated=False`) "
        "Stage 3 always returns rule D, so all non-Stage-1 queries reach Stage 4 "
        "(LLM verdict with added corpus context). Rules A/B/C never fire until "
        "`scripts/recalibrate_triage.py` is run with a real phase-B CSV.",
        "",
        "## Regressions (classifier correct → triage wrong)",
        "",
    ]
    if not regressed:
        lines.append("_None — no regressions detected._")
    else:
        lines.append(f"**{len(regressed)} regression(s):**")
        lines.append("")
        for r in regressed:
            q = r["query"]
            lines.append(
                f"- `{q[:80]}` — "
                f"expected needs={r['expected_needs']}, "
                f"triage needs={r['triage_needs']} "
                f"(reason={r.get('triage_reason','?')}, "
                f"stage={r.get('triage_stage','?')}, "
                f"rule={r.get('triage_rule','?')})"
            )

    lines += [
        "",
        "## Improvements (classifier wrong → triage correct)",
        "",
    ]
    if not improved:
        lines.append("_None._")
    else:
        lines.append(f"**{len(improved)} improvement(s):**")
        lines.append("")
        for r in improved:
            q = r["query"]
            lines.append(
                f"- `{q[:80]}` — "
                f"expected needs={r['expected_needs']}, "
                f"triage needs={r['triage_needs']} "
                f"(stage={r.get('triage_stage','?')}, "
                f"rule={r.get('triage_rule','?')})"
            )

    lines += [
        "",
        "## Recommendation: TRIAGE_ENABLED",
        "",
    ]

    if n_errors > 0:
        lines += [
            f"**HOLD** — {n_errors} triage error(s) during evaluation "
            "(Qdrant/LLM failures). Resolve infrastructure issues before "
            "enabling in production.",
            "",
            "Do not set `TRIAGE_ENABLED=True` until a clean run (0 errors) "
            "confirms the cascade is stable.",
        ]
    elif recommend_enable:
        lines += [
            f"**ENABLE** — Phase C F1 ({tnc['f1']:.3f}) >= Phase A F1 "
            f"({cnc['f1']:.3f}), {len(regressed)} regression(s), "
            f"{len(improved)} improvement(s).",
            "",
            "Set `TRIAGE_ENABLED=True` in `.env` to activate the cascade on "
            "the hot path. The shadow-mode log comparison is no longer needed "
            "once the flag is on.",
            "",
            "**Before enabling, verify:**",
            "- Calibration has been run (`triage_calibrated=True`) so Stage 3 "
            "rules A/B/C can fire. Without calibration the cascade only adds "
            "latency (probe retrieval) with no accuracy benefit from rules A–C.",
            "- p99 latency on production traffic is acceptable "
            "(the extra probe embed + Qdrant call adds ~50–150 ms).",
        ]
    else:
        lines += [
            f"**HOLD** — Phase C F1 ({tnc['f1']:.3f}) vs Phase A F1 "
            f"({cnc['f1']:.3f}), {len(regressed)} regression(s) vs "
            f"{len(improved)} improvement(s). Net quality change is negative "
            "or neutral — do not activate the flag yet.",
            "",
            "Investigate regressions above. Common causes:",
            "- Stage 4 LLM verdict with corpus context is occasionally misled "
            "by off-topic top-5 hits — review `CORPUS MATCHES` prompt wording.",
            "- Too-short / stopword-only Stage 1 is too aggressive — check "
            "false-positive Stage 1 cases and adjust `_STOPWORDS`.",
        ]

    lines += [
        "",
        "## How to Update This Report",
        "",
        "Re-run after prompt changes, corpus updates, or calibration:",
        "",
        "```bash",
        "# 1. Establish / refresh Phase A baseline",
        "python -m scripts.eval_classifier \\",
        "    --baseline scripts/eval_baseline_phaseA.json --update-baseline",
        "",
        "# 2. Generate Phase C report (requires Qdrant)",
        "python -m scripts.eval_classifier \\",
        "    --triage \\",
        "    --baseline scripts/eval_baseline_phaseA.json \\",
        "    --report scripts/eval_phaseC_report.md",
        "```",
    ]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nPhase C report saved to '{path}'.")


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
        description="Golden-set regression test for classifier + triage (B6/C5)."
    )
    p.add_argument("--golden", default=_DEFAULT_GOLDEN,
                   help="Labeled golden set JSONL (default: %(default)s)")
    p.add_argument("--baseline", default=None,
                   help="Baseline JSON: saved if missing, compared if it exists.")
    p.add_argument("--update-baseline", action="store_true",
                   help="Overwrite an existing baseline with the current run.")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N cases (quick smoke test).")
    p.add_argument("--triage", action="store_true",
                   help="Also run the full triage cascade on every case "
                        "(requires Qdrant). Prints Phase A vs Phase C comparison.")
    p.add_argument("--report", default=None, metavar="FILE",
                   help="When --triage is set, write the Phase C markdown report "
                        "to FILE (e.g. scripts/eval_phaseC_report.md).")
    return p.parse_args()


async def main() -> None:
    args = _parse_args()

    if args.report and not args.triage:
        print("WARNING: --report has no effect without --triage.", file=sys.stderr)

    cases = load_golden(args.golden)
    if args.limit is not None:
        cases = cases[:args.limit]
    print(f"Evaluating {len(cases)} labeled case(s) from '{args.golden}'...")

    retriever = None
    if args.triage:
        try:
            from rag.retriever import Retriever
            retriever = Retriever()
            print("Triage mode: Retriever created (Qdrant connection is lazy).",
                  file=sys.stderr)
        except Exception as exc:
            sys.exit(f"ERROR: could not create Retriever for --triage: {exc}")

    results = await run_eval(cases, retriever=retriever)
    metrics = compute_metrics(results)
    print_report(metrics, results)

    if args.triage:
        tri_m = compute_metrics_for(results, "triage_needs", "triage_reason")
        print_triage_comparison(metrics, tri_m, results)
        if args.report:
            write_report(args.report, metrics, tri_m, results)

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
