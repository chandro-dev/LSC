import torch
import torch.nn as nn

class TransformerSignModel(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len=50,
        num_classes=50,
        dim_model=256,
        num_heads=8,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim_model),
            nn.ReLU(),
        )

        self.positional_encoding = self._positional_encoding(seq_len, dim_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=512,  # más capacidad de aprendizaje
            dropout=dropout,
            batch_first=True,
            activation="gelu"     # mejora el rendimiento sobre ReLU en muchos casos
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def _positional_encoding(self, seq_len, dim_model):
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-torch.log(torch.tensor(10000.0)) / dim_model))
        pe = torch.zeros(seq_len, dim_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding.to(x.device)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)
