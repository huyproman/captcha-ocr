# Release provenance and rights

## Evaluated pipeline versus this release

The TEST evaluation used the private pipeline at commit `a7e94caab736767b558902a623f2eede80c5e3d9`, which is not part of this repository's history. This release adds optional device transfer and development-data utilities that were not TEST-evaluated. Model, preprocessing and decoder implementations are unchanged.

The selected checkpoint is external and is identified by SHA-256:

```text
56d84df618f80944b6beef1e1fb42d035d3b94036c8b4a6c8e3b2fa5d45f9257
```

The SHA-256 of the frozen evaluation record (kept outside this repository) is:

```text
d718a7ce8a5c68c698a704f0d126202a72aebf139752ef5cc2a39fcbf6a43ecf
```

The TEST result was not altered. CUDA deployment was measured on VAL only. TEST was used once; rerunning it is not part of reproduction.

## What is distributed

- Standalone architecture implementations, development CLI, tests and configuration descriptions.
- Aggregate scientific results and limitations.
- No raw CAPTCHA images, archives, trained checkpoint binaries or TEST labels.
- No private manifests, credentials or environment directories.
- No exact development membership: `configs/development_splits.json` is excluded pending metadata-rights clearance; accepted internal source/history is unchanged. Preparation/validation require independently authorized local membership and never generate splits. Exact reported partition/artifact reproduction is unavailable from this export alone.

## Rights boundaries

The dataset publisher declares CC BY-NC-SA 4.0. Dataset acquisition and reuse must respect those terms. The dataset's license does not automatically determine the code or trained-weight license.

Nguyen Quang Huy's source-publication authority is confirmed by OWNER_ATTESTATION (2026-10-27), not independent ownership verification. Source remains reserved-rights with no public reuse grant selected, subject to applicable statutory exceptions and hosting-platform terms. PyTorch and torchvision remain separately licensed dependencies; no upstream pretrained weights are embedded.

Checkpoint redistribution needs a separate decision.
