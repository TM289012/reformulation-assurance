"""Notification delivery for v0.6.

The database outbox is always written first. SMTP delivery is optional, which
makes invitations and password resets testable without silently pretending an
email was sent.
"""
from __future__ import annotations

from dataclasses import dataclass
from email.message import EmailMessage
import os
import smtplib
from typing import Any


@dataclass(frozen=True)
class SMTPSettings:
    host: str
    port: int
    username: str | None
    password: str | None
    sender: str
    use_tls: bool = True

    @classmethod
    def from_environment(cls) -> "SMTPSettings | None":
        host = os.environ.get("REFORMULATION_SMTP_HOST", "").strip()
        sender = os.environ.get("REFORMULATION_EMAIL_FROM", "").strip()
        if not host or not sender:
            return None
        return cls(
            host=host,
            port=int(os.environ.get("REFORMULATION_SMTP_PORT", "587")),
            username=os.environ.get("REFORMULATION_SMTP_USERNAME") or None,
            password=os.environ.get("REFORMULATION_SMTP_PASSWORD") or None,
            sender=sender,
            use_tls=os.environ.get("REFORMULATION_SMTP_TLS", "true").lower() not in {"0", "false", "no"},
        )


def send_email(recipient: str, subject: str, body: str, settings: SMTPSettings) -> None:
    message = EmailMessage()
    message["From"] = settings.sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP(settings.host, settings.port, timeout=20) as client:
        if settings.use_tls:
            client.starttls()
        if settings.username:
            client.login(settings.username, settings.password or "")
        client.send_message(message)


def deliver_queued_notifications(store: Any, *, limit: int = 25) -> dict[str, int]:
    settings = SMTPSettings.from_environment()
    if settings is None:
        return {"sent": 0, "failed": 0, "queued": int(len(store.list_notifications(status="queued", limit=limit)))}
    sent = failed = 0
    for _, row in store.list_notifications(status="queued", limit=limit).iterrows():
        try:
            send_email(str(row["recipient_email"]), str(row["subject"]), str(row["body"]), settings)
            store.mark_notification(str(row["id"]), "sent")
            sent += 1
        except Exception as exc:  # delivery failures stay visible in the outbox
            store.mark_notification(str(row["id"]), "failed", error=str(exc))
            failed += 1
    return {"sent": sent, "failed": failed, "queued": 0}
