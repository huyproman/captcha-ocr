"""Check a manifest against authorized local TRAIN/VAL fingerprints (not shipped) and image bytes/size."""
import argparse
import json
from pathlib import Path
from .cli import development_manifest
from .core import image_tensor, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'validation'], required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--splits', type=Path, required=True)
    args = p.parse_args()
    if sha(args.manifest) != args.expected_sha256:
        raise ValueError('Manifest hash mismatch')
    rows = development_manifest(args.manifest, args.split)
    expected = json.loads(args.splits.read_text())['partitions'][args.split]
    keys = ('source_id', 'group_id', 'sha256')
    if ({tuple(row[k] for k in keys) for row in rows}
            != {tuple(row[k] for k in keys) for row in expected}
            or len(rows) != len(expected)):
        raise ValueError('Manifest differs from frozen development membership')
    for row in rows:
        if sha(row['image_path']) != row['sha256']:
            raise ValueError('Image hash mismatch')
        image_tensor(row['image_path'])
    print(f'Validated {len(rows)} {args.split} sources')


if __name__ == '__main__':
    main()
