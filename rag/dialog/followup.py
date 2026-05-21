"""Follow-up context store and detection (§4.1).

After a completed query (with or without clarification), saves the enriched query
and user profile per user_id with a 5-minute TTL.  On the next incoming message, a
lightweight LLM call decides whether the new message is a follow-up (reuse context)
or a new independent query (classify from scratch).
"""

import json
import logging
from dataclasses import dataclass, field
from time import time
from typing import Optional

from config import settings
from rag.dialog.enricher import UserProfile, _ALLOWED_USER_TYPES, _empty_profile

logger = logging.getLogger(__name__)

_FOLLOWUP_TTL = 300.0  # 5 minutes


@dataclass
class _FollowUpEntry:
    original_query: str
    enriched_query: str
    profile: dict          # plain-dict snapshot of UserProfile
    saved_at: float = field(default_factory=time)


_store: dict[int, _FollowUpEntry] = {}


def save_context(
    user_id: int,
    original_query: str,
    enriched_query: str,
    profile: UserProfile,
) -> None:
    """Save completed-turn context for follow-up detection (TTL = 5 min)."""
    _store[user_id] = _FollowUpEntry(
        original_query=original_query,
        enriched_query=enriched_query,
        profile=dict(profile),
    )
    logger.debug("followup context saved: user_id=%d q=%.60r", user_id, original_query)


def _get_fresh(user_id: int) -> Optional[_FollowUpEntry]:
    entry = _store.get(user_id)
    if entry is None:
        return None
    if time() - entry.saved_at > _FOLLOWUP_TTL:
        del _store[user_id]
        return None
    return entry


def _apply_patch(base: dict, patch: dict) -> UserProfile:
    """Merge profile_patch from the LLM into the stored profile."""
    merged = dict(base)

    if "user_type" in patch:
        ut = str(patch["user_type"] or "").strip().lower()
        merged["user_type"] = ut if ut in _ALLOWED_USER_TYPES else base.get("user_type")

    if "topics" in patch and isinstance(patch["topics"], list):
        existing: list[str] = list(base.get("topics") or [])
        for t in patch["topics"]:
            s = str(t).strip() if t else ""
            if s and s not in existing:
                existing.append(s)
        merged["topics"] = existing[:10]

    if "document_hints" in patch and isinstance(patch["document_hints"], list):
        merged["document_hints"] = [
            str(h).strip() for h in patch["document_hints"] if str(h).strip()
        ][:10]

    if "temporal_context" in patch:
        tc = str(patch["temporal_context"] or "").strip()
        merged["temporal_context"] = tc or base.get("temporal_context")

    return UserProfile(
        topics=merged.get("topics") or [],
        user_type=merged.get("user_type"),
        document_hints=merged.get("document_hints") or [],
        temporal_context=merged.get("temporal_context"),
    )


async def check_followup(
    user_id: int,
    new_message: str,
) -> tuple[bool, str, UserProfile]:
    """Determine if new_message is a follow-up to the last completed turn.

    Returns:
        (is_followup, merged_query, merged_profile)

        merged_query and merged_profile are only meaningful when is_followup=True.
        On any failure the function returns (False, new_message, empty_profile) so
        the caller falls back to the normal classify_intent path.
    """
    entry = _get_fresh(user_id)
    if entry is None:
        return False, new_message, _empty_profile()

    try:
        from rag.dialog.prompts import FOLLOWUP_SYSTEM_PROMPT
        from rag.generator import _make_llm_client

        ctx_parts = [f"original_query: {entry.original_query}"]
        if entry.profile.get("user_type"):
            ctx_parts.append(f"user_type: {entry.profile['user_type']}")
        if entry.profile.get("topics"):
            ctx_parts.append(f"topics: {', '.join(entry.profile['topics'])}")
        if entry.profile.get("temporal_context"):
            ctx_parts.append(f"temporal_context: {entry.profile['temporal_context']}")

        user_content = (
            f"last_turn_context: {chr(10).join(ctx_parts)}\n"
            f"new_message: {new_message}"
        )

        client = _make_llm_client()
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=150,
        )
        content = response.choices[0].message.content or ""

        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON in followup check response: {content!r}")

        data = json.loads(content[start:end])
        is_followup = bool(data.get("is_followup", False))

        if not is_followup:
            return False, new_message, _empty_profile()

        merged_query = str(data.get("merged_query") or "").strip() or entry.enriched_query
        patch = data.get("profile_patch") or {}
        merged_profile = _apply_patch(
            entry.profile, patch if isinstance(patch, dict) else {}
        )

        logger.info(
            "follow-up detected: user_id=%d msg=%.60r merged=%.80r profile=%s",
            user_id, new_message, merged_query, merged_profile,
        )
        return True, merged_query, merged_profile

    except Exception:
        logger.warning(
            "check_followup failed for user_id=%d; treating as new query",
            user_id,
            exc_info=True,
        )
        return False, new_message, _empty_profile()
