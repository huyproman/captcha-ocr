"""Paired bootstrap over sources. Captures example-level, not training-seed, uncertainty."""
import numpy as np
from .core import distance


def paired(baseline, candidate, seed=42, replicates=2000):
    left = {r['source_id']: r for r in baseline}
    right = {r['source_id']: r for r in candidate}
    if len(left) != len(baseline) or len(right) != len(candidate) or set(left) != set(right):
        raise ValueError('Require identical unique source cohorts')
    ids = sorted(left)
    if not ids or any(left[k]['reference'] != right[k]['reference'] for k in ids):
        raise ValueError('Require nonempty matching references')
    esma = np.array([int(right[k]['reference'] == right[k]['prediction']) -
                     int(left[k]['reference'] == left[k]['prediction']) for k in ids])
    edits = np.array([distance(right[k]['reference'], right[k]['prediction']) -
                     distance(left[k]['reference'], left[k]['prediction']) for k in ids])
    lengths = np.array([len(left[k]['reference']) for k in ids])
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(replicates):
        draw = rng.integers(0, len(ids), len(ids))
        scores.append([esma[draw].mean(), edits[draw].sum() / lengths[draw].sum()])
    return dict(delta_esma=float(esma.mean()), delta_cer=float(edits.sum() / lengths.sum()),
                paired95=np.quantile(scores, [.025, .975], axis=0).tolist(),
                seed=seed, replicates=replicates)
