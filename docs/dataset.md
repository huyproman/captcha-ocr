# Dataset

Acquire [Captcha Images by Aadhav Vignesh](https://www.kaggle.com/datasets/aadhavvignesh/captcha-images) separately. The publisher's Kaggle metadata declares **CC BY-NC-SA 4.0** (dataset ID 807505). This does not grant a source-code or checkpoint license. No images, archives or trained weights are redistributed here.

Expected task: 10,000 full-frame 282×90 images, each with ten case-sensitive characters from `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`.

A development manifest is a JSON list with required `source_id`, `group_id`, `image_path`, `label`, `sha256` and **explicit `split`**. Use `train` or `validation`; the development CLI rejects TEST. Keep the frozen partition membership and source hashes. Do not regenerate the split or infer labels during prediction.

Manifest identities from the original experiments:

| Partition | SHA-256 |
|---|---|
| TRAIN | `90cd4d129a7f8bd85b6263c9cba732c1e7b6306106416ab9a86b0e51858435b8` |
| VAL | `7700a68a5a0934867876bca73fc57337ddda5bd52c46e800fca7c61fbf7edc1f` |

Note: `group_id` is derived from the image SHA-256 (1:1 with the image). See [split and duplicate handling](split-and-duplicates.md) for what this does and does not protect.

## Exact portable TRAIN/VAL membership

Exact membership is not distributed. `configs/development_splits.json`, the original 8,000 TRAIN / 1,000 VAL fingerprint collection, is excluded from this publication export pending metadata-rights clearance; accepted internal source/history remains unchanged. Its absence prevents exact partition/artifact reconstruction from this export alone, even with the archive. The hashes above are identity receipts, not replacement membership. Do not resplit to claim reproduction of reported artifacts.

Verify the acquired original archive SHA-256:

```text
101c494e779d48d797e6568c90bc480332eb337a39095b6eedd78e368f98d271
```

Only with an independently authorized local frozen TRAIN/VAL fingerprint file and authentic label-bearing filenames, run:

```sh
python -m captcha_ocr.prepare --images ACQUIRED_IMAGES --splits AUTHORIZED_LOCAL_SPLITS.json --output LOCAL_MANIFESTS
python -m captcha_ocr.validate --manifest LOCAL_MANIFESTS/validation.json --split validation --expected-sha256 PRINTED_LOCAL_VAL_SHA256 --splits AUTHORIZED_LOCAL_SPLITS.json
```

Preparation matches exact image bytes to frozen development membership. It does not draw new partitions. Unlisted files are never decoded or labeled. The acquired directory may contain other files; only the 9,000 listed sources enter output. Duplicate/missing development sources fail. Renamed files without the original label-bearing names are insufficient; do not invent labels.

Reference labels depend on authentic original filenames from the verified archive. A local JSON hash is a relocation integrity receipt, not independent proof of label correctness. Validation checks exact source/group/fingerprint membership against the supplied trusted split file; it does not infer labels from image pixels.

No row-level development membership or protected TEST labels, filenames, images or prediction records are shipped. A holder of the full archive and omitted fingerprint collection could infer its complement; do not publish that collection without a separate rights/safety decision.

The original manifests contain private paths and are not distributed. Generated local manifest hashes differ with paths; IDs, image hashes, labels and membership must agree. The printed local hash is passed explicitly to validation/evaluation. Dataset access remains governed by the publisher's terms; fingerprints do not authorize image redistribution.
