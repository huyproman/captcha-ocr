# Checkpoint workflows

| Workflow | Checkpoint saved/used |
|---|---|
| `captcha-ocr train` (this repo) | Final epoch (`cli.py` saves `model.state_dict()` after the last epoch; no VAL evaluation during training). |
| Experiments behind the reported results | Checkpoint chosen on VAL: highest exact count, then lowest CER, then earliest epoch (`methodology.md`; the C2 checkpoint has SHA-256 `56d84df6...f9257`). |

The two are not claimed to be equivalent. Running the CLI with the same seed and epochs is not claimed to reproduce the reported checkpoints: the CLI trains on CPU, performs no VAL selection, and the original training used a GPU model with CPU CTC loss. Byte-identical retraining is not promised. The original training code is not part of this repository. Reported VAL scores come from selected checkpoints and are optimistic; the TEST result used the single frozen selected C2 checkpoint, which is not distributed. `train --help` states the final-epoch behavior.
