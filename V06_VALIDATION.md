# v0.6 Validation

## Scope

v0.6 adds pilot collaboration and operational controls without changing the scientific recommendation engine.

## Automated controls

The v0.6 suite verifies:

1. Invitation links are single-use and create the intended organization membership.
2. Invitation and reset messages are written to the durable outbox.
3. Password-reset tokens are single-use and old credentials stop working.
4. Comments and assignments are permission-controlled and audited.
5. Multi-signer policies remain incomplete until every required role signs the same evidence hash.
6. A signer cannot count twice toward the same policy and evidence state.
7. Qualification dossiers are encrypted at rest and recover byte-for-byte.
8. Plaintext and ciphertext checksums are verified during artifact recovery.
9. Backups use a consistent SQLite copy, are encrypted, pass checksum checks, and pass SQLite integrity validation.
10. A verified backup can be restored into a functioning application database.
11. The PostgreSQL migration bundle includes checksummed table exports and a manifest.
12. All v0.4 and v0.5 regression tests continue to pass.

## Commands

```bash
python -m unittest discover -s tests -v
python -m unittest tests.test_v06 -v
python demo_v06.py
```

## Claims deliberately not made

- SMTP configuration proves delivery to a recipient inbox.
- Local password controls equal enterprise SSO or MFA.
- Fernet storage alone constitutes a complete enterprise key-management system.
- Backup creation alone constitutes disaster recovery; restoration must be exercised.
- The PostgreSQL migration loader is equivalent to a production-validated PostgreSQL runtime.
- Multi-signer approvals constitute regulated electronic signatures.
