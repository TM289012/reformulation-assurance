"""Security primitives for Reformulation Assurance v0.6.

Provides password hashing, single-use token generation, and evidence-safe token
comparison. These controls are suitable for private pilots, but they are not a
replacement for enterprise SSO, MFA, or a validated regulated identity system.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_SCHEME = "pbkdf2_sha256"
_DEFAULT_ITERATIONS = 310_000


@dataclass(frozen=True)
class PasswordPolicy:
    minimum_length: int = 10
    require_letter: bool = True
    require_number: bool = True


def validate_email(email: str) -> str:
    normalized = email.strip().lower()
    if not _EMAIL_RE.match(normalized):
        raise ValueError("enter a valid email address")
    return normalized


def validate_password(password: str, policy: PasswordPolicy | None = None) -> None:
    policy = policy or PasswordPolicy()
    if len(password) < policy.minimum_length:
        raise ValueError(f"password must be at least {policy.minimum_length} characters")
    if policy.require_letter and not any(ch.isalpha() for ch in password):
        raise ValueError("password must contain at least one letter")
    if policy.require_number and not any(ch.isdigit() for ch in password):
        raise ValueError("password must contain at least one number")


def hash_password(password: str, *, iterations: int = _DEFAULT_ITERATIONS) -> str:
    validate_password(password)
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "$".join(
        [
            _SCHEME,
            str(iterations),
            base64.urlsafe_b64encode(salt).decode("ascii"),
            base64.urlsafe_b64encode(derived).decode("ascii"),
        ]
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations_raw, salt_raw, expected_raw = encoded.split("$", 3)
        if scheme != _SCHEME:
            return False
        iterations = int(iterations_raw)
        salt = base64.urlsafe_b64decode(salt_raw.encode("ascii"))
        expected = base64.urlsafe_b64decode(expected_raw.encode("ascii"))
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def generate_single_use_token(nbytes: int = 32) -> str:
    """Return a URL-safe token. Only its digest should be persisted."""
    return secrets.token_urlsafe(nbytes)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_token(token: str, stored_digest: str) -> bool:
    return hmac.compare_digest(token_digest(token), stored_digest)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def expires_at(*, hours: float = 0, minutes: float = 0) -> str:
    return (utc_now() + timedelta(hours=hours, minutes=minutes)).isoformat(timespec="seconds")


def is_expired(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return True
    return parsed <= utc_now()
