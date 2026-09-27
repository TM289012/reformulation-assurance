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
import socket
import ssl
from typing import Any, Callable

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
CONNECT_TIMEOUT_SECONDS = 20


class DeliveryError(RuntimeError):
    """An SMTP send failed; the message explains what to fix, never the credentials."""


@dataclass(frozen=True)
class SMTPSettings:
    host: str
    port: int
    username: str | None
    password: str | None
    sender: str
    use_tls: bool = True
    use_ssl: bool = False  # implicit TLS from the first byte (port 465), instead of STARTTLS

    @classmethod
    def from_settings(cls, get: Callable[[str], str]) -> "SMTPSettings | None":
        """Build from any settings source: ``get(name)`` returns the value or ``""``.

        The app passes its own reader (environment first, then Streamlit secrets),
        so a hosted deployment works whether or not the platform copies secrets
        into environment variables. Values are trimmed, because secrets pasted
        into a web form often carry a trailing space or newline.
        """
        host = (get("REFORMULATION_SMTP_HOST") or "").strip()
        sender = (get("REFORMULATION_EMAIL_FROM") or "").strip()
        if not host or not sender:
            return None
        port_text = (get("REFORMULATION_SMTP_PORT") or "587").strip() or "587"
        try:
            port = int(port_text)
        except ValueError as exc:
            raise ValueError(f"REFORMULATION_SMTP_PORT must be a number, got {port_text!r}") from exc
        ssl_text = (get("REFORMULATION_SMTP_SSL") or "").strip().lower()
        use_ssl = ssl_text in TRUE_VALUES if ssl_text else port == 465
        return cls(
            host=host,
            port=port,
            username=(get("REFORMULATION_SMTP_USERNAME") or "").strip() or None,
            password=(get("REFORMULATION_SMTP_PASSWORD") or "").strip() or None,
            sender=sender,
            use_tls=(get("REFORMULATION_SMTP_TLS") or "true").strip().lower() not in FALSE_VALUES,
            use_ssl=use_ssl,
        )

    @classmethod
    def from_environment(cls) -> "SMTPSettings | None":
        return cls.from_settings(lambda name: os.environ.get(name, ""))

    def describe(self) -> str:
        """One line for the admin page: transport and sender, never the password."""
        mode = "implicit TLS" if self.use_ssl else ("STARTTLS" if self.use_tls else "no encryption")
        who = f", signing in as {self.username}" if self.username else ", no sign-in"
        return f"{self.host}:{self.port} ({mode}{who}), sender {self.sender}"


def _explain(exc: BaseException, settings: SMTPSettings) -> str:
    """Turn the usual smtplib failures into a sentence an administrator can act on."""
    where = f"{settings.host}:{settings.port}"
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        detail = exc.smtp_error.decode("utf-8", "replace") if isinstance(exc.smtp_error, bytes) else str(exc.smtp_error)
        hint = (
            "The mail server rejected the sign-in. Check REFORMULATION_SMTP_USERNAME (for Gmail and most "
            "providers it is the full email address) and REFORMULATION_SMTP_PASSWORD. Gmail and Outlook need "
            "an app password generated with 2-step verification on, not the normal account password."
        )
        return f"{hint} Server said: {exc.smtp_code} {detail.strip()}"
    if isinstance(exc, smtplib.SMTPSenderRefused):
        detail = exc.smtp_error.decode("utf-8", "replace") if isinstance(exc.smtp_error, bytes) else str(exc.smtp_error)
        if exc.smtp_code == 530 or "auth" in detail.lower():
            return (
                "The server requires a sign-in before it accepts mail, and none was configured: set "
                f"REFORMULATION_SMTP_USERNAME and REFORMULATION_SMTP_PASSWORD. Server said: {exc.smtp_code} {detail.strip()}"
            )
        return f"The server refused the sender address {settings.sender}. Server said: {exc.smtp_code} {detail.strip()}"
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        refused = ", ".join(f"{addr}: {code} {msg.decode('utf-8', 'replace') if isinstance(msg, bytes) else msg}" for addr, (code, msg) in exc.recipients.items())
        return f"The server refused the recipient. {refused}"
    if isinstance(exc, smtplib.SMTPNotSupportedError):
        return (
            f"{where} does not offer STARTTLS. Use port 465 (encrypted from the first byte) or, for a server on a "
            "trusted network only, set REFORMULATION_SMTP_TLS=false."
        )
    if isinstance(exc, ssl.SSLError):
        if settings.use_ssl:
            return (
                f"{where} did not answer with TLS from the first byte, which is what port 465 style delivery expects. "
                "If this is a STARTTLS port (usually 587), set REFORMULATION_SMTP_PORT=587 and REFORMULATION_SMTP_SSL=false. "
                f"Detail: {exc}"
            )
        return f"TLS handshake with {where} failed. Detail: {exc}"
    if isinstance(exc, smtplib.SMTPServerDisconnected):
        return (
            f"{where} closed the connection before the message was accepted. If the port is 465, the server expects "
            "TLS from the first byte (set REFORMULATION_SMTP_SSL=true or use port 465 with the default). Detail: "
            f"{exc}"
        )
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return (
            f"No answer from {where} within {CONNECT_TIMEOUT_SECONDS} seconds. Either the host or port is wrong, or the "
            "platform blocks outgoing connections on this port. Port 465 sometimes works where 587 does not."
        )
    if isinstance(exc, socket.gaierror):
        return f"The host name {settings.host!r} could not be resolved. Check REFORMULATION_SMTP_HOST for typos."
    if isinstance(exc, ConnectionRefusedError):
        return f"{where} refused the connection. Check the port (587 for STARTTLS, 465 for implicit TLS)."
    if isinstance(exc, smtplib.SMTPResponseException):
        detail = exc.smtp_error.decode("utf-8", "replace") if isinstance(exc.smtp_error, bytes) else str(exc.smtp_error)
        return f"The server answered with an error: {exc.smtp_code} {detail.strip()}"
    if isinstance(exc, OSError):
        return f"Could not reach {where}: {exc}"
    return f"{type(exc).__name__}: {exc}"


def _open_connection(settings: SMTPSettings) -> smtplib.SMTP:
    if settings.use_ssl:
        return smtplib.SMTP_SSL(settings.host, settings.port, timeout=CONNECT_TIMEOUT_SECONDS)
    return smtplib.SMTP(settings.host, settings.port, timeout=CONNECT_TIMEOUT_SECONDS)


def send_email(recipient: str, subject: str, body: str, settings: SMTPSettings) -> None:
    """Send one plain-text message. Raises :class:`DeliveryError` with an actionable message."""
    message = EmailMessage()
    message["From"] = settings.sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    try:
        with _open_connection(settings) as client:
            if settings.use_tls and not settings.use_ssl:
                client.starttls()
            if settings.username:
                client.login(settings.username, settings.password or "")
            client.send_message(message)
    except Exception as exc:
        raise DeliveryError(_explain(exc, settings)) from exc


TEST_MESSAGE_KIND = "mail_test"
TEST_MESSAGE_SUBJECT = "Reformulation Assurance: outgoing mail works"


def test_message_body(settings: SMTPSettings) -> str:
    return (
        "This is a test message from your Reformulation Assurance workspace.\n\n"
        f"It was sent through {settings.host}:{settings.port} as {settings.sender}. "
        "Invitations and password-reset links from this instance will arrive the same way.\n"
    )


def send_test_email(recipient: str, settings: SMTPSettings) -> None:
    """A short message an administrator sends to themselves to prove the settings work."""
    send_email(recipient, TEST_MESSAGE_SUBJECT, test_message_body(settings), settings)


def deliver_queued_notifications(
    store: Any,
    *,
    limit: int = 25,
    settings: "SMTPSettings | None" = None,
    transport: Callable[[str, str, str, SMTPSettings], None] | None = None,
) -> dict[str, Any]:
    """Send every queued message once. Failures are recorded on the row and never retried
    automatically; the Team page offers a retry once the settings are fixed.

    Returns counts plus ``errors``, the plain-language reason for each failure in this pass.
    """
    settings = settings or SMTPSettings.from_environment()
    if settings is None:
        return {
            "sent": 0,
            "failed": 0,
            "queued": int(len(store.list_notifications(status="queued", limit=limit))),
            "errors": [],
        }
    send = transport or send_email
    sent = failed = 0
    errors: list[str] = []
    for _, row in store.list_notifications(status="queued", limit=limit).iterrows():
        try:
            send(str(row["recipient_email"]), str(row["subject"]), str(row["body"]), settings)
            store.mark_notification(str(row["id"]), "sent")
            sent += 1
        except Exception as exc:  # delivery failures stay visible in the outbox
            reason = str(exc) if isinstance(exc, DeliveryError) else _explain(exc, settings)
            store.mark_notification(str(row["id"]), "failed", error=reason)
            errors.append(reason)
            failed += 1
    return {"sent": sent, "failed": failed, "queued": 0, "errors": errors}
