"""Shared-CNN recognizers for full 282x90 images: S0, S0-A, C1, C2, C3 (see docs/architectures.md)."""
import math
import torch
from torch import nn
from .spatial_models import SpatialOCRModel, VOCAB

SOS, EOS, BLANK = 62, 63, 62


class HeightMean(nn.Module):
    """Mean over height; equivalent to AdaptiveAvgPool2d((1, 70)) for width 70, without its non-deterministic CUDA backward."""
    def forward(self, x):
        if x.ndim != 4 or x.shape[-1] != 70:
            raise ValueError('Height pooling requires width 70')
        return x.mean(dim=2, keepdim=True)


class FixedWidthMean(nn.Module):
    """Mean over 10 bins of 7 tokens; equivalent to AdaptiveAvgPool1d(10) for width 70."""
    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 70:
            raise ValueError('Token pooling requires width 70')
        return x.reshape(*x.shape[:-1], 10, 7).mean(dim=-1)


def encode_labels(labels, device='cpu'):
    if not labels or any(len(s) != 10 or any(c not in VOCAB for c in s) for s in labels):
        raise ValueError('Expected ten case-sensitive vocabulary characters per source')
    return torch.tensor([[VOCAB.index(c) for c in s] for s in labels], dtype=torch.long, device=device)


def ctc_loss(logits, targets):
    targets = targets.to('cpu')
    lengths = torch.full((len(targets),), targets.shape[1], dtype=torch.long)
    needed = targets.shape[1] + (targets[:, 1:] == targets[:, :-1]).sum(1)
    if (needed > logits.shape[1]).any():
        raise ValueError('Insufficient CTC time axis for adjacent repeats')
    # CTC runs on CPU for deterministic backward; the device copy keeps autograd.
    log_probs = logits.float().log_softmax(-1).transpose(0, 1).to('cpu')
    return nn.functional.ctc_loss(log_probs, targets,
        torch.full((len(targets),), logits.shape[1], dtype=torch.long), lengths,
        blank=BLANK, zero_infinity=False).to(logits.device)


def ctc_decode(indices):
    output = []
    for row in indices.tolist():
        chars, previous = [], None
        for token in row:
            if token != BLANK and token != previous:
                chars.append(VOCAB[token])
            previous = token
        output.append(''.join(chars))
    return output


def positional_encoding(length, dim, device, dtype):
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    frequency = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    result = torch.zeros(length, dim, device=device, dtype=dtype)
    result[:, 0::2] = torch.sin(position * frequency)
    result[:, 1::2] = torch.cos(position * frequency)
    return result


class LearnedAlignmentOCR(SpatialOCRModel):
    """S0-A: learned-query attention replaces S0 fixed pooling; BiGRU and head are inherited."""
    def __init__(self):
        super().__init__()
        # Ten ordered queries attend over the 70 tokens; sinusoidal positions are added to the keys only.
        self.queries = nn.Parameter(torch.randn(10, 256) * .02)
        self.alignment = nn.MultiheadAttention(256, 4, batch_first=True, dropout=0)
        del self.spatial_pool
        self.attention_weights = None

    def forward(self, x):
        tokens = self.encoder(x).squeeze(2).transpose(1, 2)
        keys = tokens + positional_encoding(tokens.shape[1], 256, tokens.device, tokens.dtype)
        queries = self.queries.unsqueeze(0).expand(len(x), -1, -1)
        aligned, weights = self.alignment(queries, keys, tokens, need_weights=True)
        self.attention_weights = weights.detach()
        context, _ = self.rnn(aligned)
        return self.head(context)


class TemporalOCR(nn.Module):
    """C1 (BiGRU) and C2 (Transformer encoder): 70 per-token logits over 62 classes + CTC blank (index 62)."""
    def __init__(self, transformer=False):
        super().__init__()
        self.encoder = SpatialOCRModel().encoder
        self.transformer = transformer
        if transformer:
            layer = nn.TransformerEncoderLayer(256, 4, 512, dropout=.1, batch_first=True)
            self.sequence = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        else:
            self.sequence = nn.GRU(256, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.head = nn.Linear(256, 63)

    def forward(self, x):
        tokens = self.encoder(x).squeeze(2).transpose(1, 2)
        if self.transformer:
            tokens = tokens + positional_encoding(tokens.shape[1], 256, tokens.device, tokens.dtype)
            context = self.sequence(tokens)
        else:
            context, _ = self.sequence(tokens)
        return self.head(context)


class AttentionDecoderOCR(nn.Module):
    """C3: autoregressive causal Transformer decoder over the CNN tokens. SOS=62, EOS=63. Not PARSeq/SAR."""
    def __init__(self):
        super().__init__()
        self.encoder = SpatialOCRModel().encoder
        self.embedding = nn.Embedding(64, 256)
        layer = nn.TransformerDecoderLayer(256, 4, 512, dropout=.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.head = nn.Linear(256, 64)

    def memory(self, x):
        tokens = self.encoder(x).squeeze(2).transpose(1, 2)
        return tokens + positional_encoding(tokens.shape[1], 256, tokens.device, tokens.dtype)

    def from_memory(self, memory, previous):
        tokens = self.embedding(previous) * math.sqrt(256)
        tokens = tokens + positional_encoding(tokens.shape[1], 256, tokens.device, tokens.dtype)
        mask = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device).triu(1)
        return self.head(self.decoder(tokens, memory, tgt_mask=mask))

    def forward(self, x, previous):
        return self.from_memory(self.memory(x), previous)

    def decode(self, x):
        memory = self.memory(x)
        previous = torch.full((len(x), 1), SOS, dtype=torch.long, device=x.device)
        results = [''] * len(x)
        ended = [False] * len(x)
        # At most ten symbols plus EOS; an early EOS yields a short output that is scored as wrong.
        for _ in range(11):
            token = self.from_memory(memory, previous)[:, -1].argmax(-1)
            previous = torch.cat((previous, token.unsqueeze(1)), 1)
            for i, value in enumerate(token.tolist()):
                if not ended[i]:
                    if value == EOS:
                        ended[i] = True
                    elif value < len(VOCAB):
                        results[i] += VOCAB[value]
                    else:
                        results[i] += '?'  # SOS emitted as output: keep visible in metrics
            if all(ended):
                break
        return results


def build_model(family):
    if family == 'S0':
        model = SpatialOCRModel()
        model.spatial_pool = FixedWidthMean()
    elif family == 'S0-A':
        model = LearnedAlignmentOCR()
    elif family in ('C1', 'C2'):
        model = TemporalOCR(transformer=family == 'C2')
    elif family == 'C3':
        model = AttentionDecoderOCR()
    else:
        raise ValueError('Unknown model family: ' + str(family))
    model.encoder[-1] = HeightMean()
    return model
