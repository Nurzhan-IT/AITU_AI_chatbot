"""Query enrichment and user-profile extraction for the dialog pipeline.

After the user answers clarifying questions, this module will combine the
original query with the collected answers into (a) a richer search string
fed to the retriever, and (b) a structured user profile (topics, user
type, document hints, temporal context) consumed by multi-factor scoring.
This file is a stub introduced with the package scaffolding (Prompt 0).
"""

from rag.generator import _make_llm_client  # noqa: F401 — used by later stages
