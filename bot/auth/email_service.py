import logging

import httpx

from config import settings

logger = logging.getLogger(__name__)

_BREVO_URL = "https://api.brevo.com/v3/smtp/email"


async def send_verification_code(to_email: str, code: str) -> bool:
    payload = {
        "sender": {
            "name": settings.brevo_sender_name,
            "email": settings.brevo_sender_email,
        },
        "to": [{"email": to_email}],
        "subject": "Код подтверждения — AITU Bot",
        "htmlContent": (
            "<div style='font-family:sans-serif;max-width:480px'>"
            "<h2>Подтверждение почты</h2>"
            "<p>Ваш код подтверждения для доступа к AITU Chatbot:</p>"
            f"<h1 style='letter-spacing:8px;color:#1a56db'>{code}</h1>"
            "<p style='color:#6b7280'>Код действителен 24 часа.<br>"
            "Если вы не запрашивали код — просто проигнорируйте это письмо.</p>"
            "</div>"
        ),
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                _BREVO_URL,
                headers={
                    "api-key": settings.brevo_api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if response.status_code in (200, 201):
            logger.info("Verification code sent to %s", to_email)
            return True
        logger.error(
            "Brevo API error %d: %s", response.status_code, response.text
        )
        return False
    except Exception as exc:
        logger.error("Failed to send verification email: %s", exc)
        return False
