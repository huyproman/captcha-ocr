# Results

## Development validation

| Model | Exact /1,000 | Edits /10,000 | Exact-sequence accuracy | Corpus CER |
|---|---:|---:|---:|---:|
| C2 CNN-Transformer encoder-CTC | 974 | 26 | 97.4% | 0.26% |
| S0 CNN-BiGRU fixed pooling | 968 | 32 | 96.8% | 0.32% |
| S0-A learned-query attention | 945 | 55 | 94.5% | 0.55% |
| C1 CNN-BiGRU-CTC | 79 | 2114 | 7.9% | 21.14% |
| C3 autoregressive decoder | 565 | 518 | 56.5% | 5.18% |
| M0 MobileNetV3-Small reference | 735 | 306 | 73.5% | 3.06% |

C2 versus S0 paired 95% exact-sequence interval: [-0.6,+1.8] percentage points. Superiority is not statistically established.

## Held-out TEST: selected C2 only

One evaluation of the frozen C2 pipeline: **975/1,000 exact; 25/10,000 edits; exact-sequence accuracy 97.5%; corpus CER 0.25%**. No comparison or selection was performed on TEST. The TEST-minus-VAL difference in exact-sequence accuracy is +0.1 percentage points. No post-TEST tuning or second evaluation occurred.

Per-source TEST records are kept outside this repository. No images or TEST labels are included.
