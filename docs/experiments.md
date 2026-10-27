# Experiments

## What was measured

S0 (CNN-BiGRU, fixed pooling) was the baseline. S0-A replaced fixed pooling with learned positional queries. C1 and C2 use CTC sequence recognition; C3 is a custom causal attention decoder. M0 is a MobileNetV3-Small reference. These are specific implementations under a fixed development schedule, not a universal ranking of architecture families.

CTC training computed the loss on CPU (deterministic backward) with the model on GPU. Model construction here preserves state keys; the original S0 and C2 checkpoints reproduce every development prediction through this code on CPU. No model was retrained for publication.

## Controlled interventions (see [ablations](ablations.md))

Three S0 treatments (learned-query alignment, class-weighted loss, contrast treatment) were compared with a matched S0 control. Class weights are bounded and derived from TRAIN; the contrast treatment applies at training and inference. The training CLI trains the base families and saves the final epoch. It does not replay the interventions or the VAL checkpoint selector. See [methodology](methodology.md) and [ablations](ablations.md).

An augmentation intervention was never trained and was dropped from scope. Its omission has no measured effect.

## Final evaluation

C2 was selected on VAL by exact-sequence accuracy, with CER, latency, complexity and order as tie-breakers. The selected checkpoint was then evaluated on TEST exactly once, on CPU. No comparison, selection, calibration or tuning used TEST. Later CUDA measurements use VAL only.
