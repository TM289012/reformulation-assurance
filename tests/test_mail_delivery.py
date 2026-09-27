"""Outgoing mail: settings, transport selection, plain-language failures, retry from the outbox."""
from __future__ import annotations

from pathlib import Path
import smtplib
import socket
import tempfile
import unittest
from unittest import mock

from notifications import (
    TEST_MESSAGE_KIND,
    TEST_MESSAGE_SUBJECT,
    DeliveryError,
    SMTPSettings,
    deliver_queued_notifications,
    send_email,
    send_test_email,
    test_message_body,
)
from pilot_store import PilotStore

SETTINGS = SMTPSettings(host="smtp.example.com", port=587, username="mailer@example.com",
                        password="app-password", sender="mailer@example.com")


def _reader(values: dict[str, str]):
    return lambda name: values.get(name, "")


class SettingsTest(unittest.TestCase):
    def test_port_465_means_implicit_tls_unless_told_otherwise(self):
        base = {"REFORMULATION_SMTP_HOST": "smtp.example.com", "REFORMULATION_EMAIL_FROM": "m@example.com"}
        self.assertFalse(SMTPSettings.from_settings(_reader(base)).use_ssl)
        self.assertTrue(SMTPSettings.from_settings(_reader({**base, "REFORMULATION_SMTP_PORT": "465"})).use_ssl)
        self.assertFalse(SMTPSettings.from_settings(
            _reader({**base, "REFORMULATION_SMTP_PORT": "465", "REFORMULATION_SMTP_SSL": "false"})).use_ssl)
        self.assertTrue(SMTPSettings.from_settings(
            _reader({**base, "REFORMULATION_SMTP_PORT": "2525", "REFORMULATION_SMTP_SSL": "yes"})).use_ssl)

    def test_pasted_whitespace_is_trimmed_and_a_bad_port_is_a_clear_error(self):
        values = {
            "REFORMULATION_SMTP_HOST": " smtp.example.com \n",
            "REFORMULATION_EMAIL_FROM": "m@example.com ",
            "REFORMULATION_SMTP_USERNAME": " m@example.com",
            "REFORMULATION_SMTP_PASSWORD": "abcd efgh ijkl mnop\n",
            "REFORMULATION_SMTP_PORT": " 587 ",
        }
        settings = SMTPSettings.from_settings(_reader(values))
        self.assertEqual((settings.host, settings.username, settings.port), ("smtp.example.com", "m@example.com", 587))
        self.assertEqual(settings.password, "abcd efgh ijkl mnop", "inner spaces belong to the password")
        with self.assertRaises(ValueError) as caught:
            SMTPSettings.from_settings(_reader({**values, "REFORMULATION_SMTP_PORT": "587,"}))
        self.assertIn("REFORMULATION_SMTP_PORT", str(caught.exception))

    def test_description_never_contains_the_password(self):
        text = SETTINGS.describe()
        self.assertIn("smtp.example.com:587", text)
        self.assertIn("STARTTLS", text)
        self.assertNotIn("app-password", text)
        self.assertIn("implicit TLS", SMTPSettings(**{**SETTINGS.__dict__, "port": 465, "use_ssl": True}).describe())


class _FakeSMTP:
    """Stands in for smtplib.SMTP / SMTP_SSL and records what the sender did."""

    instances: list["_FakeSMTP"] = []
    login_error: BaseException | None = None
    starttls_error: BaseException | None = None

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.calls: list[tuple] = []
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        if _FakeSMTP.starttls_error:
            raise _FakeSMTP.starttls_error
        self.calls.append(("starttls",))

    def login(self, username, password):
        if _FakeSMTP.login_error:
            raise _FakeSMTP.login_error
        self.calls.append(("login", username, password))

    def send_message(self, message):
        self.calls.append(("send", message["To"], message["Subject"]))


class _FakeSMTPSSL(_FakeSMTP):
    pass


class TransportTest(unittest.TestCase):
    def setUp(self):
        _FakeSMTP.instances = []
        _FakeSMTP.login_error = None
        _FakeSMTP.starttls_error = None
        self.patches = [mock.patch("smtplib.SMTP", _FakeSMTP), mock.patch("smtplib.SMTP_SSL", _FakeSMTPSSL)]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()

    def test_starttls_on_587_and_implicit_tls_on_465(self):
        send_email("to@example.com", "Hi", "Body", SETTINGS)
        (client,) = _FakeSMTP.instances
        self.assertIs(type(client), _FakeSMTP)
        self.assertEqual(client.calls, [("starttls",), ("login", "mailer@example.com", "app-password"), ("send", "to@example.com", "Hi")])

        _FakeSMTP.instances = []
        ssl_settings = SMTPSettings(**{**SETTINGS.__dict__, "port": 465, "use_ssl": True})
        send_email("to@example.com", "Hi", "Body", ssl_settings)
        (client,) = _FakeSMTP.instances
        self.assertIs(type(client), _FakeSMTPSSL)
        self.assertEqual([call[0] for call in client.calls], ["login", "send"], "no STARTTLS on an implicit-TLS connection")

    def test_rejected_sign_in_becomes_an_actionable_message(self):
        _FakeSMTP.login_error = smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")
        with self.assertRaises(DeliveryError) as caught:
            send_email("to@example.com", "Hi", "Body", SETTINGS)
        text = str(caught.exception)
        self.assertIn("rejected the sign-in", text)
        self.assertIn("app password", text)
        self.assertIn("535 5.7.8 Username and Password not accepted", text)
        self.assertNotIn("app-password", text.replace("app password", ""), "the configured password never appears")

    def test_missing_starttls_and_timeouts_are_explained(self):
        _FakeSMTP.starttls_error = smtplib.SMTPNotSupportedError("STARTTLS extension not supported by server.")
        with self.assertRaises(DeliveryError) as caught:
            send_email("to@example.com", "Hi", "Body", SETTINGS)
        self.assertIn("does not offer STARTTLS", str(caught.exception))
        self.assertIn("465", str(caught.exception))

        with mock.patch("smtplib.SMTP", side_effect=socket.timeout("timed out")):
            with self.assertRaises(DeliveryError) as caught:
                send_email("to@example.com", "Hi", "Body", SETTINGS)
        self.assertIn("No answer from smtp.example.com:587 within 20 seconds", str(caught.exception))

    def test_test_email_names_the_transport(self):
        send_test_email("me@example.com", SETTINGS)
        (client,) = _FakeSMTP.instances
        self.assertEqual(client.calls[-1], ("send", "me@example.com", TEST_MESSAGE_SUBJECT))
        self.assertIn("smtp.example.com:587", test_message_body(SETTINGS))


class OutboxRetryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = PilotStore(Path(self.tmp.name) / "pilot.db")
        self.owner, self.org = self.store.register_owner(
            email="owner@example.com", display_name="Owner", password="SecurePass123", organization_name="Lab"
        )
        self.scientist, _ = self.store.register_owner(
            email="other@example.com", display_name="Other", password="SecurePass123", organization_name="Other Lab"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _fail(self, *args):
        raise smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")

    def _ok(self, *args):
        return None

    def test_failure_is_recorded_in_plain_language_and_retried_only_on_request(self):
        self.store.create_invitation(self.org, email="student@example.com", role="scientist",
                                     actor_user_id=self.owner, base_url="https://app.example.com")
        first = deliver_queued_notifications(self.store, settings=SETTINGS, transport=self._fail)
        self.assertEqual((first["sent"], first["failed"]), (0, 1))
        self.assertIn("rejected the sign-in", first["errors"][0])
        outbox = self.store.list_outbox(self.org, self.owner)
        self.assertEqual(outbox.iloc[0]["status"], "failed")
        self.assertIn("535", outbox.iloc[0]["error"])

        # A later pass with working settings does not touch failed rows by itself.
        second = deliver_queued_notifications(self.store, settings=SETTINGS, transport=self._ok)
        self.assertEqual((second["sent"], second["failed"]), (0, 0))
        failures = self.store.delivery_failures(self.org, self.owner)
        self.assertEqual(failures["messages"], 1)
        self.assertIn("535", failures["last_error"])

        # The administrator retries once the settings are fixed.
        self.assertEqual(self.store.requeue_failed_notifications(self.org, self.owner), 1)
        third = deliver_queued_notifications(self.store, settings=SETTINGS, transport=self._ok)
        self.assertEqual((third["sent"], third["failed"]), (1, 0))
        outbox = self.store.list_outbox(self.org, self.owner)
        self.assertEqual(outbox.iloc[0]["status"], "sent")
        self.assertEqual(outbox.iloc[0]["error"], "")
        self.assertEqual(self.store.delivery_failures(self.org, self.owner)["messages"], 0)

    def test_password_resets_count_but_stay_private_and_are_not_requeued(self):
        self.store.request_password_reset("owner@example.com", base_url="https://app.example.com")
        deliver_queued_notifications(self.store, settings=SETTINGS, transport=self._fail)
        failures = self.store.delivery_failures(self.org, self.owner)
        self.assertEqual((failures["messages"], failures["password_resets"]), (0, 1))
        self.assertNotIn("owner@example.com", failures["last_error"])
        self.assertTrue(self.store.list_outbox(self.org, self.owner).empty, "reset messages never reach the outbox view")
        self.assertEqual(self.store.requeue_failed_notifications(self.org, self.owner), 0)
        # A successful send afterwards clears the stale warning.
        self.store.queue_notification(recipient_email="owner@example.com", kind=TEST_MESSAGE_KIND,
                                      subject=TEST_MESSAGE_SUBJECT, body="x", organization_id=self.org)
        deliver_queued_notifications(self.store, settings=SETTINGS, transport=self._ok)
        with self.store.connection() as con:  # make the earlier failure strictly older than the success
            con.execute("UPDATE notifications SET created_at = '2000-01-01T00:00:00+00:00' WHERE status = 'failed'")
        failures = self.store.delivery_failures(self.org, self.owner)
        self.assertEqual((failures["messages"], failures["password_resets"]), (0, 0))

    def test_only_administrators_see_or_retry_failures(self):
        with self.assertRaises(PermissionError):
            self.store.delivery_failures(self.org, self.scientist)
        with self.assertRaises(PermissionError):
            self.store.requeue_failed_notifications(self.org, self.scientist)

    def test_without_settings_nothing_is_sent_and_the_queue_is_counted(self):
        self.store.create_invitation(self.org, email="student@example.com", role="scientist",
                                     actor_user_id=self.owner, base_url="https://app.example.com")
        with mock.patch.dict("os.environ", {"REFORMULATION_SMTP_HOST": "", "REFORMULATION_EMAIL_FROM": ""}):
            result = deliver_queued_notifications(self.store, settings=None)
        self.assertEqual(result, {"sent": 0, "failed": 0, "queued": 1, "errors": []})


if __name__ == "__main__":
    unittest.main()
