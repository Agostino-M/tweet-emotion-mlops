"""Transformer encoder + small MLP head for sequence classification."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class TweetClassifier(nn.Module):
    """[CLS] embedding -> dropout -> Linear(hidden_dim) -> dropout -> Linear(num_labels).

    Returns raw logits; softmax is applied by the loss during training and by the
    serving layer at inference time.
    """

    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(encoder_name)
        else:  # architecture only, weights loaded later from a checkpoint
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(encoder_name))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.encoder.config.hidden_size, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
        x = self.dropout(self.fc(x))
        return self.classifier(x)
