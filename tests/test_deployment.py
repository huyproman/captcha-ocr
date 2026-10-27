import unittest
from unittest.mock import patch
from captcha_ocr.deployment import device_for


class DeploymentTests(unittest.TestCase):
    def test_explicit_device_and_unavailable_cuda(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(str(device_for('cpu')), 'cpu')
            self.assertEqual(str(device_for('auto')), 'cpu')
            with self.assertRaises(ValueError):
                device_for('cuda')
        with self.assertRaises(ValueError):
            device_for('tpu')
