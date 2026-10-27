# Methodology

## Evaluation design

The frozen source-level split contains 8,000 TRAIN, 1,000 VAL and 1,000 sealed TEST images. Selection uses VAL only. The primary metric is case-sensitive exact-sequence accuracy; corpus CER divides total Levenshtein edits by total reference characters. Every source, including an invalid or empty output, contributes to scoring.

## Training for the reported results

Seed 42, Adam at learning rate 0.001, cosine scheduling to 0.00001, batch size 32, 25 epochs and gradient clipping at norm 5. Checkpoints were selected by highest VAL exact accuracy, then lowest CER, then earliest epoch. CTC loss ran on CPU for deterministic backward while the model trained on GPU.

The `captcha-ocr train` command saves the final epoch; its default invocation does not reproduce the checkpoint selection above. Existing weights are loaded strictly and were not retrained for publication.

No model uses privileged fixed character crops. Architecture comparisons change multiple modeling assumptions; only the S0 interventions are controlled.
