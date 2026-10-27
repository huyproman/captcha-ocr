# CPU and optional CUDA deployment

## Python migration

Update imports from `captcha_ocr.phase3_models` to `captcha_ocr.models` and
replace `PretrainedRecognizer(vocab_size, sequence_length, pretrained)` with
`MobileNetRecognizer(vocab_size, sequence_length)` from `captcha_ocr.mobilenet`.
The constructor does not download pretrained weights. Supported checkpoint
compatibility is `state_dict` loading through the documented loader, not
arbitrary full-object pickles referencing the removed module or class names.

## Runtime and measured deployment

The TEST protocol was fixed: CPU, batch 1, one thread, native grayscale/255, greedy CTC. The later `deployment.py` adapter adds device transfer only and has not been evaluated on TEST.

Prediction and development evaluation accept `--device cpu`, `--device cuda` or `--device auto`. CPU is the default. CUDA errors if unavailable rather than silently falling back. Auto chooses CUDA when available and includes the actual device in output JSON. Training remains CPU in the generic CLI.

Multi-image CLI prediction processes a sequence of batch 1 calls. It does not invoke the batch 8 and 32 throughput measured below; no batching speedup is implied by passing multiple filenames.

On an RTX 4090 with PyTorch 2.8.0+cu128, C2 CUDA predictions matched all 1,000 frozen VAL CPU predictions: 974 exact, 26 edits, zero disagreements. This is development parity for one device/software configuration, not guaranteed bitwise parity on every GPU.

| Device | Batch | Median batch latency | Sources/s |
|---|---:|---:|---:|
| CPU | 1 | 22.87 ms | 43.72 |
| CPU | 8 | 182.80 ms | 43.76 |
| CPU | 32 | 1040.85 ms | 30.74 |
| RTX 4090 | 1 | 1.05 ms | 954.34 |
| RTX 4090 | 8 | 1.07 ms | 7447.75 |
| RTX 4090 | 32 | 2.54 ms | 12604.88 |

In-memory tensor forward plus decoding, one CPU thread, 5 warmups, 20 repetitions, CUDA synchronized. Excludes disk I/O, preprocessing and host-to-device transfer. Repeated development input used for timing; not a production load test. Peak allocated CUDA tensor memory 229,691,392 bytes. Shared-host contention is not isolated; throughput is not an end-to-end service guarantee.

Install CUDA wheels separately from the CPU instructions using the appropriate official PyTorch index. No weights are downloaded by constructors. Checkpoint bytes are unchanged.
