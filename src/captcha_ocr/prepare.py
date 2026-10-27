"""Match images to authorized local frozen TRAIN/VAL fingerprints (not shipped); never generate splits."""
import argparse
import json
from pathlib import Path
from .core import sha, VOCAB


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    splits = json.loads(args.splits.read_text())
    partitions = splits['partitions']
    wanted = {r['sha256']: (split, r) for split, rows in partitions.items() for r in rows}
    if (set(partitions) != {'train', 'validation'} or len(wanted) != 9000
            or len(partitions['train']) != 8000 or len(partitions['validation']) != 1000
            or len({r['source_id'] for rows in partitions.values() for r in rows}) != 9000):
        raise ValueError('Require exactly 8000 TRAIN / 1000 VAL fingerprints')
    found = {'train': [], 'validation': []}
    seen = set()
    for path in sorted(args.images.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        digest = sha(path)
        if digest not in wanted:
            continue  # unlisted files are never decoded or labeled
        if digest in seen:
            raise ValueError('Duplicate frozen source fingerprint')
        seen.add(digest)
        split, row = wanted[digest]
        label = path.stem
        if len(label) != 10 or any(c not in VOCAB for c in label):
            raise ValueError('Original label-bearing filenames are required (label = filename stem)')
        found[split].append(dict(row, split=split, image_path=str(path.resolve()), label=label))
    if seen != set(wanted) or len(found['train']) != 8000 or len(found['validation']) != 1000:
        raise ValueError('Missing or duplicate fingerprinted images')
    args.output.mkdir(parents=True, exist_ok=False)
    for split, rows in found.items():
        output = args.output / (split + '.json')
        output.write_text(json.dumps(sorted(rows, key=lambda r: r['source_id']), indent=2) + '\n')
        print(split, len(rows), sha(output))


if __name__ == '__main__':
    main()
