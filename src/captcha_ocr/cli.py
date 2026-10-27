"""Predict, evaluate (VAL manifests only) and train commands. TEST evaluation is not implemented."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from .core import build, load_checkpoint, image_tensor, predict, metrics, sha
from .models import encode_labels, ctc_loss, VOCAB, SOS, EOS
from .deployment import load, infer


FAMILIES = '''model families (--family):
  S0    CNN + BiGRU, fixed pooling, 10 positional classifiers
  S0-A  CNN + learned-query attention + BiGRU, 10 positional classifiers
  C1    CNN + BiGRU + CTC
  C2    CNN + Transformer encoder + CTC (selected model)
  C3    CNN + autoregressive Transformer decoder
  M0    MobileNetV3-Small reference, random init'''

def development_manifest(path, split):
    # Path guard runs before any file read; substring match, a hygiene check and not a security control.
    if (split not in ('train', 'validation') or any('test' in p.lower() for p in Path(path).parts)
            or any('test' in p.lower() for p in Path(path).resolve().parts)):
        raise ValueError('Only train/validation manifests are accepted; paths containing "test" are refused')
    rows = json.loads(Path(path).read_text())
    if not rows or len({r['source_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest must have unique sources')
    for row in rows:
        if (row.get('split') != split or any('test' in p.lower() for p in Path(row['image_path']).parts)
                or any('test' in p.lower() for p in Path(row['image_path']).resolve().parts)):
            raise ValueError('Split mismatch or path containing "test"')
        if len(row['label']) != 10 or any(c not in VOCAB for c in row['label']):
            raise ValueError('Expected ten case-sensitive vocabulary characters')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=FAMILIES, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    notes = dict(
        predict='Predict from a local checkpoint on CPU (default) or CUDA.',
        evaluate='Evaluate a local checkpoint on a VALIDATION manifest only.',
        train='Train a model on a TRAIN manifest. Saves the FINAL epoch only; no VAL checkpoint selection. '
              'Published results used VAL-selected checkpoints, so this command does not reproduce them.')
    for command in ('predict', 'evaluate', 'train'):
        p = sub.add_parser(command, help=notes[command], description=notes[command])
        p.add_argument('--family', choices=['S0', 'S0-A', 'C1', 'C2', 'C3', 'M0'], required=True)
        p.add_argument('--output', type=Path, required=True)
        if command != 'train':
            p.add_argument('--checkpoint', type=Path, required=True)
            p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
        if command == 'predict':
            p.add_argument('--image', type=Path, nargs='+', required=True)
        else:
            p.add_argument('--manifest', type=Path, required=True)
        if command == 'evaluate':
            p.add_argument('--split', choices=['validation'], default='validation')
            p.add_argument('--expected-manifest-sha256', required=True)
        if command == 'train':
            p.add_argument('--epochs', type=int, default=25)
            p.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'train':
        if args.epochs < 1:
            parser.error('epochs must be positive')
        rows = development_manifest(args.manifest, 'train')
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        model = build(args.family).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.00001)
        for epoch in range(args.epochs):
            order = torch.randperm(len(rows)).tolist()
            for start in range(0, len(rows), 32):
                batch = [rows[i] for i in order[start:start + 32]]
                x = torch.cat([image_tensor(r['image_path']) for r in batch])
                y = encode_labels([r['label'] for r in batch])
                optimizer.zero_grad()
                if args.family in ('C1', 'C2'):
                    loss = ctc_loss(model(x), y)
                elif args.family == 'C3':
                    previous = torch.cat([torch.full((len(y), 1), SOS), y], 1)
                    target = torch.cat([y, torch.full((len(y), 1), EOS)], 1)
                    loss = torch.nn.functional.cross_entropy(model(x, previous).flatten(0, 1), target.flatten())
                else:
                    loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            scheduler.step()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model_state=model.state_dict(), family=args.family, seed=args.seed, epochs=args.epochs), args.output)
        return
    model, device = load(args.checkpoint, args.family, args.device)
    if args.command == 'predict':
        records = [dict(image=str(p), prediction=infer(model, image_tensor(p), args.family, device)[0]) for p in args.image]
        result = dict(records=records, device=str(device))
    else:
        if sha(args.manifest) != args.expected_manifest_sha256:
            raise ValueError('Manifest hash mismatch')
        rows = development_manifest(args.manifest, args.split)
        records = []
        for row in sorted(rows, key=lambda r: r['source_id']):
            if sha(row['image_path']) != row['sha256']:
                raise ValueError('Source bytes changed')
            prediction = infer(model, image_tensor(row['image_path']), args.family, device)[0]
            records.append(dict(source_id=row['source_id'], group_id=row['group_id'], reference=row['label'], prediction=prediction))
        result = dict(metrics=metrics(records), records=records, checkpoint_sha256=sha(args.checkpoint),
                      manifest_sha256=sha(args.manifest), device=str(device), protocol=f'{device} batch1 threads1 raw grayscale/255 no resize')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result.get('metrics', dict(predictions=len(records)))))


if __name__ == '__main__':
    main()
