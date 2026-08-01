# v0.6.2 Build Summary

Implemented:

- Dedicated Process Window Designer with one-factor, corner-plus-center, and three-level grid designs
- Process-window batches created directly in the correct qualification stage
- Server-side rejection of blank, nonfinite, or zero completed measurements
- Persistent success/warning messages after Streamlit reruns
- Replicate-group status, remaining-run counts, and target-total replicate creation
- Duplicate replicate prevention through idempotent target counts
- Duplicate policy and unscoped signature prevention
- Invitation and password-reset links displayed in the administrative outbox
- Run-level and formulation-level calibration tabs and dossier exports
- Separate optimizer nominal, Monte Carlo nominal, and robust success probabilities
- Formal batch closure and bulk cancellation of unused experiments
- Updated qualification dossier with both calibration evidence levels

Validation: **25/25 automated tests passed**.
