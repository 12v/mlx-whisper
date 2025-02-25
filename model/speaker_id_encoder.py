import torch.nn as nn

from model.attention import Attention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super().__init__()
        self.self_attention = Attention(
            query_dim=d_model,
            key_value_dim=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x1, self_attention = self.self_attention(x, x)
        x = self.norm1(x + x1)
        x = self.norm2(x + self.feed_forward(x))
        return x, self_attention


class Encoder(nn.Module):
    def __init__(self, d_model, embedding_dim, num_layers, num_heads, dropout_rate):
        super().__init__()
        self.embedder = nn.Linear(embedding_dim, d_model)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dropout_rate) for _ in range(num_layers)]
        )

    def forward(self, x):
        x = self.embedder(x)
        for layer in self.encoder_layers:
            x, self_attention = layer(x)
        return x, self_attention
