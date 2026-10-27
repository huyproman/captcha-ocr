# Split grouping and duplicate handling

## What `group_id` is

In the original internal `configs/development_splits.json` (excluded from publication pending metadata-rights clearance), `group_id` is `group-` plus the first 16 hex characters of the image SHA-256, so it is 1:1 with `source_id` and with the image bytes (historically checked for all 9,000 TRAIN/VAL entries). "No group overlap between splits" is therefore equivalent to "no shared source ID" and is trivially true. Exact membership is not distributed; the public repository neither generates splits nor performs dependency grouping.

## What it protects

- Exact byte-identical duplicates cannot straddle partitions (they share an image SHA and hence a source).

## What it does not protect

- Content-equivalent re-encodings (same pixels, different bytes) in the public checks.
- Near-duplicates, transformed copies, or images from a shared generator or generation session.
- Repeated label strings across images (labels are not used for grouping; identical text alone is not treated as dependence).
- The dataset has no source or author metadata, so independent source groups cannot be constructed. Generator provenance is unknown.

## Evidence from the original experiments

Private records (not in this repository) describe a pre-freeze scan of the 10,000-image archive: no duplicate files; decoded-pixel hashing; a perceptual-hash near-duplicate scan (63-bit DCT pHash, Hamming distance <= 6 candidates, plus pixel-error verification) that reported 373 candidate pairs within distance 6 and 0 verified near-duplicate pairs; and the original split was allocated by dependency groups built from those scans. The scan is described as limited: it can miss shifts, crops, changed backgrounds and redraws, was not manually adjudicated, and cannot establish generator independence. The record states "undetected transformed near-duplicates remain possible". This public repository neither ships that scan nor reproduces it. No leakage has been observed or is claimed; equally, absence of leakage is not demonstrated.

The split itself is frozen and was not altered.
