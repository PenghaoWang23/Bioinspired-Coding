from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


BASE_TO_INDEX = {"A": 0, "T": 1, "C": 2, "G": 3, "-": 4}
INDEX_TO_BASE = {value: key for key, value in BASE_TO_INDEX.items()}
VOCAB_SIZE = len(BASE_TO_INDEX)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x + self.pe[: x.size(1)].permute(1, 0, 2)
        else:
            x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerDNA(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=BASE_TO_INDEX["-"])
        self.conv_block = nn.Sequential(
            nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm1d(2 * embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(2 * embed_dim, 2 * embed_dim, kernel_size=5, padding=2, groups=2 * embed_dim, bias=False),
            nn.BatchNorm1d(2 * embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(2 * embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
        self.copy_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=1000, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_copies, seq_len = x.shape
        x_embed = self.embedding(x)

        x_conv = x_embed.view(batch_size * num_copies, seq_len, self.embed_dim).permute(0, 2, 1)
        x_conv = self.conv_block(x_conv).permute(0, 2, 1)
        x_conv = x_conv.view(batch_size, num_copies, seq_len, self.embed_dim)

        fused_features: List[torch.Tensor] = []
        for position in range(seq_len):
            features_at_position = x_conv[:, :, position, :]
            attended_features, _ = self.copy_attention(features_at_position, features_at_position, features_at_position)
            fused_features.append(attended_features.mean(dim=1))

        fused_sequence = torch.stack(fused_features, dim=1)
        fused_sequence = self.pos_encoder(fused_sequence)
        return self.fc(self.transformer_encoder(fused_sequence))


def pad_sequence(sequence: Sequence[str], max_len: int, pad_value: str = "-") -> List[str]:
    sequence_list = list(sequence)
    if len(sequence_list) < max_len:
        return sequence_list + [pad_value] * (max_len - len(sequence_list))
    return sequence_list[:max_len]


def preprocess_copies(copies: Iterable[str]) -> torch.Tensor:
    copy_tokens = [list(copy.strip()) for copy in copies]
    max_len = max((len(sequence) for sequence in copy_tokens), default=0)
    copy_indices = []
    for sequence in copy_tokens:
        padded = pad_sequence(sequence, max_len=max_len)
        copy_indices.append([BASE_TO_INDEX.get(base, BASE_TO_INDEX["-"]) for base in padded])
    return torch.tensor([copy_indices], dtype=torch.long)


def reconstruct_sequence(model: TransformerDNA, copies: Sequence[str], device: torch.device | str) -> str:
    if not copies:
        return ""
    copies_tensor = preprocess_copies(copies).to(device)
    with torch.no_grad():
        logits = model(copies_tensor)
        predicted_indices = torch.argmax(logits, dim=2).squeeze(0).cpu().tolist()
    return "".join(
        INDEX_TO_BASE[index]
        for index in predicted_indices
        if index != BASE_TO_INDEX["-"]
    )
