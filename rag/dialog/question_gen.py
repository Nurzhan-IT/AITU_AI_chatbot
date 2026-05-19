"""Generation of clarifying questions for the dialog pipeline.

Given the original user query, available documents, and any prior answers,
this module will (in a later stage) ask the LLM to produce the next
clarifying question with 2–4 Telegram-friendly option labels. For now it
is a stub introduced as part of the package scaffolding (Prompt 0).
"""

from rag.generator import _make_llm_client  # noqa: F401 — used by later stages
