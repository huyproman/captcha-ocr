import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from captcha_ocr.prepare import main


class PrepareTests(unittest.TestCase):
    def test_duplicate_image_in_place_of_missing_source_fails_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'; images.mkdir()
            (images / '0123456789.jpg').write_bytes(b'a')
            (images / 'abcdefghij.jpg').write_bytes(b'a')
            partitions = {split: [dict(source_id=f'{split}{i}', group_id=f'g{split}{i}', sha256=f'{split}{i}')
                                  for i in range(count)] for split, count in [('train', 8000), ('validation', 1000)]}
            p = root / 'splits.json'; p.write_text(json.dumps({'partitions': partitions}))
            output = root / 'output'
            with patch('sys.argv', ['prepare', '--images', str(images), '--splits', str(p), '--output', str(output)]), \
                    patch('captcha_ocr.prepare.sha', return_value='train0'):
                with self.assertRaisesRegex(ValueError, 'Duplicate'):
                    main()
            self.assertFalse(output.exists())
