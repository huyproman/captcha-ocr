# Limitations

- One dataset, one main training seed and a fixed ten-character task format limit generalization.
- VAL was used for checkpoint selection; its score is not an untouched final evaluation.
- C2's small advantage over S0 has a confidence interval crossing zero.
- C1 and C3 results concern these implementations and schedules, not entire architecture families.
- Exploratory transformed-image measurements are not reported because label preservation under the transforms was not established; no robustness claim is made.
- CPU and GPU runs differed slightly for the class-weighted S0 run and C1; the cause is unknown. Reported comparisons use CPU predictions.
- An augmentation intervention was not run and has no measured effect.
- No external-dataset or cross-generator evaluation was run; this was a scope choice, not a finding of failure.
- TEST was evaluated once, after the pipeline was frozen. Its result is a single held-out estimate; nothing was tuned on it. Later deployment work used development data only.
- `group_id` equals the image hash; the public split protects only against exact duplicates. Near-duplicate and shared-generator dependence are not excluded ([details](split-and-duplicates.md)).
- Intervals reflect example-level uncertainty only, not training-seed variance; no calibration analysis was done.
