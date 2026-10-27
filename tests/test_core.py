import unittest
import torch
from captcha_ocr.core import build, decode, metrics, load_checkpoint, image_tensor


class CoreTests(unittest.TestCase):
    def test_paths_containing_test_rejected_before_io(self):
        with self.assertRaises(ValueError):
            image_tensor('/nonexistent/protected_test/a.png')

    def test_ctc_collapses_repeats_and_keeps_blank_separated_repeats(self):
        self.assertEqual(decode(torch.tensor([[10, 10, 62, 10, 11]]), 'C2'), ['AAB'])

    def test_metric_denominator_and_case(self):
        result = metrics([{'source_id': 'a', 'reference': 'Aa', 'prediction': 'AA'},
                          {'source_id': 'b', 'reference': 'Bb', 'prediction': ''}])
        self.assertEqual(result['exact_correct'], 0)
        self.assertEqual(result['edits'], 3)
        self.assertEqual(result['cer'], .75)

    def test_all_families_construct_and_output_shapes(self):
        torch.set_num_threads(1)
        for family in ['S0', 'S0-A', 'C1', 'C2', 'C3', 'M0']:
            model = build(family).eval()
            x = torch.zeros(1, 1, 90, 282)
            with torch.inference_mode():
                if family == 'C3':
                    self.assertEqual(len(model.decode(x)), 1)
                else:
                    y = model(x)
                    self.assertEqual(y.shape, (1, 70, 63) if family in ('C1', 'C2') else (1, 10, 62))


if __name__ == '__main__':
    unittest.main()
