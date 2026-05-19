# Stabilization Report — Clarifying Dialog (Stages 1–3)

**Date:** 2026-05-19
**Branch:** `feature/dialog_and_doc`
**Status:** ⚠ Blocked — feature not yet deployed; no production traffic to analyze.

---

## 1. Why this report cannot give the metrics requested

The stabilization prompt assumes Stages 1–3 are live in prod and have accumulated user traffic. Reality on `data/bot.db` right now:

| Check | Observed |
|---|---|
| `query_logs` row count | **14** |
| Timestamp range | 2026-04-15 → 2026-04-30 |
| Rows after dialog feature merged | **0** (all 14 pre-date this work) |
| `clarification_rounds` column present | **No** — migration runs on next `init_db()`, but the bot has not been restarted since the schema change |
| Rows with `clarification_rounds > 0` | **0** |
| Feedback present | 2 × 👍, 0 × 👎, rest NULL |

The SQL from the prompt would currently fail (`no such column: clarification_rounds`) and, even after the migration, would only yield the `no_clarify` bucket. No A/B comparison is possible.

---

## 2. What we *can* read from the existing corpus

The 14 historical queries are useful as a preview of how the classifier (Stage 1) will likely behave once the bot ships.

Predicted classifier verdict on the 14 existing queries (manual best-guess against `CLASSIFY_SYSTEM_PROMPT`):

| Bucket | Predicted count | Examples |
|---|---|---|
| `specific` (no clarification) | **12** | "Сколько стоит грант", "Когда каникулы зимнего триместра у 2 курса докторантуры", "What are a master's thesis and a master's project?" |
| `vague_topic` / `ambiguous` | **2** | "Академ календарь?" (ambiguous — calendar of what year/group?), "1?" (degenerate) |
| Predicted `no_clarify` share | **~86%** | — |

Qualitative read: AITU students who actually reach the bot today already ask quite specific things (course registration, theses, financing amounts, vacation dates). The dialog feature is unlikely to fire often — closer to ~10–20% of traffic, not the 30–40% the prompt's "60–70% no_clarify" target implies.

That changes the cost-benefit shape: the dialog will be a tail-improvement, not a mainline change. Worth keeping that expectation when reading future metrics.

---

## 3. Implementation review (in lieu of prod metrics)

Things I'd flag for the next pass regardless of traffic:

1. **`/skip` semantics diverged from the prompt text.** Per spec wording, `/skip` was supposed to mark one round skipped and call `_ask_next`; per the readiness criterion and `dialog_feature_description.md`, `/skip` exits the dialog immediately. I went with the criterion. If the criterion was wrong, this is a 3-line fix in `cmd_skip`.
2. **`_proceed_to_search` clears state at the top.** If the LLM hangs mid-search, the user is at least not stuck in `ClarifyDialog.waiting_for_answer`. The prompt sample placed `state.clear()` at the bottom — be aware the order differs.
3. **Dialog-state `F.text` handler catches everything** including `📖 FAQ`, `/start`, `/help`. MVP-acceptable, but it will surprise a user who tries to bail to the FAQ mid-dialog. Cheap fix later: pass commands and known reply-keyboard labels through.
4. **`enriched_query` is logged at DEBUG level only.** That's intentional to keep prod logs quiet, but it means you need `LOG_LEVEL=DEBUG` to verify rewrite quality on real traffic. Worth a one-time bump to INFO during the first week post-deploy.
5. **Docs cache (`_DOCS_TTL = 600`) is process-local.** Restart wipes it; fresh first dialog after restart eats one full Qdrant scroll. Acceptable for a single-instance bot.

---

## 4. Decision

**Do NOT proceed to Stage 4 yet. First fix: deployment + telemetry collection.**

Concrete blockers before Stage 4 (LLM reranker + multi-factor scoring) makes sense:

- [ ] Merge `feature/dialog_and_doc` to `main` and deploy.
- [ ] Verify `init_db()` runs and adds the `clarification_rounds` column on the prod DB (one-line check via `PRAGMA table_info(query_logs);`).
- [ ] Run for at least **2 weeks** or **≥200 query_logs rows**, whichever comes first.
- [ ] Re-run this report's SQL. Targets:
  - share of `with_clarify` rows: **10–25%** (lower than prompt's 60–70% — see §2)
  - `bad_rate` of `with_clarify` should be **≤** `bad_rate` of `no_clarify`. If it's worse, the classifier and/or enricher is doing net harm.
- [ ] Read ~20 sampled `with_clarify` rows by eye — same heuristic the prompt suggested.

Reasons not to rush into Stage 4 without that data:

- Multi-factor scoring's weights (`semantic_sim 0.35`, `user_type_match 0.20`, etc.) are guesses until we see which factor actually correlates with 👍/👎 in *this* corpus.
- The LLM reranker doubles per-query LLM cost in the `with_clarify` path. Justifying that cost needs evidence the current path has a measurable miss problem, not a hypothesis.

If after the 2-week run the metrics look fine and the dialog is rarely firing, **drop Stage 4 entirely** — it would be optimizing a path that handles 10% of traffic.

---

## 5. Open question for the operator

When you do deploy: do you want the `intent` log line in `handle_question` left at INFO (visible by default) or downgraded to DEBUG? Right now it logs the full classification result for every text query — useful during stabilization, noisy long-term.
