# Reproducibility

## Environment

Use Python 3.12. Install official CPU PyTorch 2.8.0 and torchvision 0.23.0 first, then `pip install -e .`. Runtime pins include Pillow 12.2.0 and NumPy 2.4.6. No model constructor downloads external weights. Optional figure rendering uses `.[figures]`; optional CUDA installation is described separately.

Run `python -m unittest discover -s tests -v` and `captcha-ocr --help`. The GitHub Actions workflow has never run on hosted infrastructure; the same commands were run locally.

## Exact development reconstruction

Follow [dataset setup](dataset.md). Exact development membership (`configs/development_splits.json`) is excluded pending metadata-rights clearance; accepted internal source/history is unchanged. The retained preparation tool matches an independently authorized local frozen TRAIN/VAL file; it does not generate splits. Without that file, original label-bearing filenames, exact image bytes and external checkpoints, this export cannot reproduce exact reported partitions or artifacts. Preparation prints local manifest hashes; path relocation changes those hashes but must not change IDs, references or fingerprints. Resplitting is not exact reproduction.

## Checkpoints

Weights are not redistributed. Obtain checkpoints separately and store them outside Git, for example in ignored `checkpoints/`. A checkpoint contains a `family` and `model_state` mapping. Loading uses CPU `weights_only=True`, strict parameter keys and family checks before optional device transfer.

Selected C2 checkpoint SHA-256:

```text
56d84df618f80944b6beef1e1fb42d035d3b94036c8b4a6c8e3b2fa5d45f9257
```

Use complete native 282×90 grayscale images, float32 /255, batch 1, one thread and greedy decoding for the CPU replay. No implicit resize, random inference augmentation or label-conditioned inference is used. Missing files, geometry mismatches and protected paths fail.

With the original checkpoints and original development membership, an earlier internal clone reproduced all 1,000 C2 and S0 VAL predictions. Neither asset is distributed here; this is historical evidence, not a public-clone reproduction claim. Training is runnable, but byte-identical retraining is not promised. The CLI saves the final epoch; reported results used the VAL checkpoint selector in [methodology](methodology.md).

TEST was used exactly once. The commands here are development-only.
