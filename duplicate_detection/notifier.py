"""Format and dispatch duplicate/stale warnings to the admin via Telegram."""
import logging

from aiogram import Bot

logger = logging.getLogger(__name__)

_MAX_SNIPPET = 80   # chars to show from chunk text


def _snippet(text: str) -> str:
    text = " ".join(text.split())   # collapse whitespace
    if len(text) > _MAX_SNIPPET:
        return text[:_MAX_SNIPPET] + "…"
    return text


async def send_upload_warnings(
    bot: Bot,
    admin_ids: list[int],
    filename: str,
    warnings: list[dict],
) -> None:
    if not warnings:
        return

    duplicates = [w for w in warnings if w["warning_type"] == "DUPLICATE"]
    stale = [w for w in warnings if w["warning_type"] == "STALE"]

    lines = [
        "⚠️ <b>Обнаружены предупреждения при загрузке</b>",
        f"📄 <code>{filename}</code>",
        "",
    ]

    if duplicates:
        lines.append(f"<b>🔁 ДУБЛИКАТЫ ({len(duplicates)}):</b>")
        for w in duplicates:
            lines.append(
                f"• [#{w['id']}] Совпадает с <code>{w['existing_filename']}</code> "
                f"— {w['similarity']:.0%}"
            )
            lines.append(f"  «{_snippet(w['new_chunk_text'])}»")
        lines.append("")

    if stale:
        lines.append(f"<b>📅 УСТАРЕВШИЕ ({len(stale)}):</b>")
        for w in stale:
            lines.append(
                f"• [#{w['id']}] Похоже, заменяет <code>{w['existing_filename']}</code> "
                f"— {w['similarity']:.0%}"
            )
            if w.get("llm_reason"):
                lines.append(f"  Причина: {w['llm_reason']}")
        lines.append("")

    lines.append("/warnings — просмотр всех  |  /resolve &lt;id&gt; — закрыть")

    text = "\n".join(lines)

    for admin_id in admin_ids:
        try:
            await bot.send_message(admin_id, text, parse_mode="HTML")
        except Exception as exc:
            logger.warning("Failed to send warning notification to admin %d: %s", admin_id, exc)
