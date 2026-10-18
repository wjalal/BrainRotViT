"""
Stage-2 ablation: Transformer encoder across slices.

Each slice's 768-d ViT embedding is one token, so self-attention models
long-range dependencies between slices, unlike the BiGRU's recurrence.
Everything else matches bigru_lstm.py so that only the sequence encoder
differs: 768 -> 256 input projection, sinusoidal slice position encoding,
2 layers, attention pooling over slices, and the shared regression head.
"""
import torch.nn as nn

from stage2_common import AttentionPooling, PositionalEncoding, RegressionHead, seed_from_env

STAGE2_SEEDABLE = True


class AgePredictionCNN(nn.Module):
    def __init__(self, input_shape, hidden_dim=256, num_layers=2, num_heads=8,
                 ff_dim=512, dropout=0.3, attn_dropout=0.1):
        seed_from_env()
        super().__init__()
        print("AgePredictionCNN (slice Transformer) Shape:", input_shape)

        self.num_slices = input_shape[1]
        self.embedding_dim = input_shape[2]

        self.input_projection = nn.Linear(self.embedding_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=self.num_slices)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=attn_dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(hidden_dim),
                                             enable_nested_tensor=False)
        self.attention_pool = AttentionPooling(hidden_dim)
        self.head = RegressionHead(hidden_dim, dropout=dropout)

    def _encode(self, x):
        x = self.input_projection(x.squeeze(1))  # [B, S, hidden]
        x = self.pos_encoding(x)
        return self.encoder(x)

    def forward(self, x, sex):
        x, _ = self.attention_pool(self._encode(x))
        return self.head(x, sex)

    def get_attention_weights(self, x, sex):
        """Slice-pooling weights for visualisation, [batch, num_slices]."""
        _, weights = self.attention_pool(self._encode(x))
        return weights.detach()
