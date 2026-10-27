import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from captcha_ocr.cli import development_manifest
from captcha_ocr.core import build, image_tensor, load_checkpoint, predict, sha


class InterfaceTests(unittest.TestCase):
    def test_train_help_states_final_epoch_only(self):
        import contextlib, io, sys
        from captcha_ocr.cli import main
        out = io.StringIO()
        with patch.object(sys, 'argv', ['captcha-ocr', 'train', '--help']), contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit):
                main()
        self.assertIn('FINAL epoch', out.getvalue())

    def test_missing_split_is_not_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'validation.json'
            p.write_text(json.dumps([dict(source_id='a', image_path='/absent.png', label='0123456789')]))
            with self.assertRaises(ValueError):
                development_manifest(p, 'validation')

    def test_symlink_to_protected_path_is_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / 'protected_test'
            protected.mkdir()
            Image.new('L', (282, 90)).save(protected / 'image.png')
            link = root / 'innocent.png'
            link.symlink_to(protected / 'image.png')
            with self.assertRaises(ValueError):
                image_tensor(link)

    def test_manifest_denies_test_before_read(self):
        with patch.object(Path, 'read_text', side_effect=AssertionError('must not read')):
            with self.assertRaises(ValueError):
                development_manifest('/protected/test.json', 'test')

    def test_manifest_rejects_duplicate_ids_and_wrong_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'validation.json'
            row = dict(source_id='a', image_path='/image.png', label='0123456789')
            p.write_text(json.dumps([row, row]))
            with self.assertRaises(ValueError):
                development_manifest(p, 'validation')
            row['label'] = 'wrong'
            p.write_text(json.dumps([row]))
            with self.assertRaises(ValueError):
                development_manifest(p, 'validation')

    def test_prediction_ignores_label_bearing_filename(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = build('S0').eval()
        with tempfile.TemporaryDirectory() as directory:
            a, b = (Path(directory) / n for n in ('AAAAAAAAAA.png', 'bbbbbbbbbb.png'))
            image = Image.fromarray(np.arange(90 * 282, dtype=np.uint8).reshape(90, 282))
            image.save(a); image.save(b)
            torch.testing.assert_close(image_tensor(a), image_tensor(b), rtol=0, atol=0)
            self.assertEqual(predict(model, image_tensor(a), 'S0'), predict(model, image_tensor(b), 'S0'))

    def test_checkpoint_family_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'checkpoint.pt'
            torch.save(dict(family='C2', model_state=build('C2').state_dict()), p)
            with self.assertRaises(ValueError):
                load_checkpoint(p, 'S0')

    def test_wrong_image_geometry_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'image.png'
            Image.new('L', (10, 10)).save(p)
            with self.assertRaises(ValueError):
                image_tensor(p)
            with self.assertRaises(FileNotFoundError):
                image_tensor(Path(directory) / 'absent.png')


if __name__ == '__main__':
    unittest.main()
