# v0.4 Validation

Run:

```bash
python -m unittest discover -s tests -v
```

The suite checks:

1. Bounded mixture repair preserves the required total.
2. Every project receives explicit qualification gates.
3. Recommended mixtures sum to 100% and keep the removed ingredient at zero.
4. Linked triplicates produce valid repeatability evidence.
5. Confirmation gates pass only when replicate count, compliance, and CV rules pass.
6. Calibration uses frozen predictions and real outcomes.
7. Manufacturing-variation probabilities remain valid and robustness evidence persists.
8. Replicate and robustness actions appear in the audit trail.

The separate command-line demonstration exercises the complete workflow against the included coatings dataset.
