"""Checkpoint loading, image preprocessing, decoding and sequence metrics."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .models import build_model, ctc_decode, VOCAB
from .mobilenet import MobileNetRecognizer


def build(family):
    return MobileNetRecognizer(62, 10) if family == 'M0' else build_model(family)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_checkpoint(path, family):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    # 'ARC-04' is the family tag stored in existing MobileNetV3 reference checkpoints.
    if checkpoint.get('family') not in (family, 'ARC-04' if family == 'M0' else family):
        raise ValueError('Checkpoint architecture does not match requested family')
    model = build(family)
    state = checkpoint.get('model_state', checkpoint.get('model_state_dict', checkpoint.get('state_dict')))
    if state is None:
        raise ValueError('Checkpoint has no recognized model state')
    model.load_state_dict(state, strict=True)
    return model.eval()


def image_tensor(path):
    if (any('test' in part.lower() for part in Path(path).parts)
            or any('test' in part.lower() for part in Path(path).resolve().parts)):
        raise ValueError('Paths containing "test" are refused')
    with Image.open(path) as image:
        if image.size != (282, 90):
            raise ValueError('Expected a 282x90 image; implicit resizing is disabled')
        array = np.asarray(image.convert('L'), dtype=np.float32).copy() / 255
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def decode(indices, family):
    if family in ('C1', 'C2'):
        return ctc_decode(indices)
    return [''.join(VOCAB[i] for i in row) for row in indices.tolist()]


def predict(model, x, family):
    with torch.inference_mode():
        return model.decode(x) if family == 'C3' else decode(model(x).argmax(-1), family)


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def metrics(records):
    if not records or len({r['source_id'] for r in records}) != len(records):
        raise ValueError('Require nonempty unique source records')
    denominator = sum(len(r['reference']) for r in records)
    if not denominator:
        raise ValueError('Empty reference denominator')
    edits = sum(distance(r['reference'], r['prediction']) for r in records)
    exact = sum(r['reference'] == r['prediction'] for r in records)
    return dict(image_count=len(records), exact_correct=exact, esma=exact / len(records),
                edits=edits, reference_characters=denominator, cer=edits / denominator)
