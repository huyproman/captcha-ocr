"""M0: MobileNetV3-Small positional reference. Random init; construction never downloads weights."""
import torch
from torch import nn
from torch.nn import functional as F

class MobileNetRecognizer(nn.Module):
    def __init__(self, vocab_size, sequence_length):
        super().__init__()
        from torchvision.models import mobilenet_v3_small

        backbone = mobilenet_v3_small(weights=None)
        self.encoder = backbone.features
        self.head = nn.Linear(576, vocab_size)
        self.sequence_length = sequence_length
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        features = F.adaptive_avg_pool2d(self.encoder(x), (1, self.sequence_length))
        return self.head(features.squeeze(2).transpose(1, 2))
