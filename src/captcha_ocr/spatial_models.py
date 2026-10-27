"""S0: shared CNN encoder plus fixed-pooling BiGRU positional recognizer."""
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
VOCAB_SIZE = 62


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SpatialOCRModel(nn.Module):
    """Input (B, 1, 90, 282) grayscale. Output (B, 10, 62) logits, one per character position.

    Encoder: five conv blocks (32, 64, 128, 256, 256 channels). Width is halved twice
    (282 -> 141 -> 70) and height is pooled to 1, giving (B, 256, 1, 70): 70 width tokens.
    Tokens are pooled to 10 positions, passed through a 2-layer BiGRU (128 per direction)
    and classified by Linear(256, 62). The other families reuse this encoder.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, sequence_length=10, hidden_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.encoder = nn.Sequential(
            ConvBNReLU(1, 32, stride=1),
            nn.MaxPool2d(2, 2),  # 45, 141
            ConvBNReLU(32, 64, stride=1),
            nn.MaxPool2d(2, 2),  # 22, 70
            ConvBNReLU(64, 128, stride=1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 11, 70
            ConvBNReLU(128, hidden_dim, stride=1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 5, 70
            ConvBNReLU(hidden_dim, hidden_dim, stride=1),
            nn.AdaptiveAvgPool2d((1, 70))  # 1, 70
        )
        self.spatial_pool = nn.AdaptiveAvgPool1d(sequence_length)  # 70 tokens -> 10 positions
        self.rnn = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] != 1 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
        feats = self.encoder(x).squeeze(2)  # (B, 256, 70)
        pos_tokens = self.spatial_pool(feats).transpose(1, 2)  # (B, 10, 256)
        context, _ = self.rnn(pos_tokens)
        return self.head(context)  # (B, 10, 62)
