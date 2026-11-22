"""Temporal/contextual encoders, fusion and classifier blocks.

Every model returns (logits, embeddings) where embeddings holds the pooled
output of each block, so the separability of each block can be visualized.
"""
import torch
import torch.nn as nn

N_CLASSES = 7
D = 256


def masked_mean(x, mask):
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


class AttentionPool(nn.Module):
    """Learned weighted average over time: lets the model focus on emotionally salient frames."""

    def __init__(self, d):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))

    def forward(self, x, mask):
        s = self.score(x).squeeze(-1).masked_fill(~mask, -1e4)
        w = torch.softmax(s, dim=1)
        return (w.unsqueeze(-1) * x).sum(1), w


class SpeechEncoder(nn.Module):
    """Temporal modelling block: frame projection -> 2-layer BiGRU -> attention pooling."""

    def __init__(self, in_dim, d=D, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(d, d // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttentionPool(d)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout))

    def forward(self, x, mask):
        h, _ = self.gru(self.proj(x))
        z, _ = self.pool(h, mask)
        return self.out(z)


class TextEncoder(nn.Module):
    """Contextual modelling block: token projection -> 2-layer Transformer encoder -> masked mean."""

    def __init__(self, in_dim, d=D, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout))

    def forward(self, x, mask):
        h = self.enc(self.proj(x), src_key_padding_mask=~mask)
        return self.out(masked_mean(h, mask))


class Classifier(nn.Module):
    def __init__(self, d=D, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, N_CLASSES))

    def forward(self, z):
        return self.net(z)


class UnimodalModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder, self.cls = encoder, Classifier()

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        return self.cls(z), {"encoder": z}


class GatedFusion(nn.Module):
    """Per-dimension gate g in [0,1]: fused = g * speech + (1 - g) * text.

    The gate is computed from both modalities, so the model can learn how much to
    trust each one per example; its mean value is logged to show which modality
    the model relies on.
    """

    def __init__(self, d=D):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.mix = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))

    def forward(self, s, t):
        g = self.gate(torch.cat([s, t], -1))
        return self.mix(g * s + (1 - g) * t), g


class ConcatFusion(nn.Module):
    """Baseline: concatenate and project."""

    def __init__(self, d=D):
        super().__init__()
        self.mix = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d))

    def forward(self, s, t):
        return self.mix(torch.cat([s, t], -1)), None


class FusionModel(nn.Module):
    def __init__(self, speech_dim, text_dim, fusion="gated"):
        super().__init__()
        self.speech, self.text = SpeechEncoder(speech_dim), TextEncoder(text_dim)
        self.fusion = GatedFusion() if fusion == "gated" else ConcatFusion()
        self.cls = Classifier()

    def forward(self, xs, ms, xt, mt):
        s, t = self.speech(xs, ms), self.text(xt, mt)
        f, g = self.fusion(s, t)
        emb = {"speech": s, "text": t, "fusion": f}
        if g is not None:
            emb["gate"] = g.mean(-1, keepdim=True)
        return self.cls(f), emb
