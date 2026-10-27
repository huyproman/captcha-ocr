"""Device selection and checkpoint loading. CUDA is optional and never a silent fallback."""
import torch
from .core import load_checkpoint, predict


def device_for(requested='cpu'):
    if requested not in ('cpu', 'cuda', 'auto'):
        raise ValueError('device must be cpu, cuda or auto')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; no silent CPU fallback')
    return torch.device('cuda' if requested == 'cuda' or requested == 'auto' and torch.cuda.is_available() else 'cpu')


def load(path, family, requested='cpu'):
    device = device_for(requested)
    return load_checkpoint(path, family).to(device).eval(), device


def infer(model, tensor, family, device):
    return predict(model, tensor.to(device), family)
