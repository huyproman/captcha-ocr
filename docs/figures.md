# Figures

`scripts/render_figures.py` regenerates every PNG here from `assets/figures/results.json` (metric plots) or from fixed architecture facts (diagram). It needs `pip install '.[figures]'` and no dataset or checkpoint: `python scripts/render_figures.py`. No CAPTCHA pixels are included.

| Figure | Content | Qualification |
|---|---|---|
| [Architectures](../assets/figures/architectures.png) | Shared CNN encoder, five heads, separate MobileNetV3-Small reference | Matches the source in `src/captcha_ocr/`; see [architectures](architectures.md) for the config-name mapping |
| [Validation](../assets/figures/validation.png) | Six-model VAL exact-sequence accuracy and corpus CER (CPU) | Single seed, single dataset; model comparisons are not causal ablations |
| [Interventions](../assets/figures/ablations.png) | Three S0 interventions with paired 95% intervals (CPU) | Class-weighted result is the CPU 969, not the GPU 970; augmentation was not run |
| [Held-out](../assets/figures/heldout.png) | One-shot C2 TEST result with Wilson interval | Separate from VAL; interval covers example-level uncertainty only |

Two earlier plots (a baseline error breakdown and a size/latency chart) were removed because their labels were internal run identifiers.
