from __future__ import annotations
import torch
import torch.nn as nn


class ViolationClassifier(nn.Module):
    """Document-level multi-label audit-finding classifier."""

    def __init__(self, hidden_size: int = 768, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self.register_buffer("pos_weight", torch.ones(num_classes), persistent=False)

    def set_pos_weight(self, value: torch.Tensor) -> None:
        if value.shape != self.pos_weight.shape:
            raise ValueError(
                f"pos_weight must have shape {tuple(self.pos_weight.shape)}, "
                f"got {tuple(value.shape)}"
            )
        self.pos_weight.copy_(value.to(self.pos_weight.device, dtype=torch.float))

    def forward(self, document_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the violation classifier.

        Args:
            document_embedding: (batch, hidden_size) - pooled document representation

        Returns:
            logits: (batch, num_classes)
        """
        logits = self.mlp(document_embedding)
        return logits

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute binary cross-entropy for non-exclusive finding labels.

        Args:
            logits: (batch, num_classes)
            labels: float multi-hot tensor shaped (batch, num_classes)

        Returns:
            loss: scalar tensor
        """
        if labels.shape != logits.shape:
            raise ValueError(
                f"finding labels must have shape {tuple(logits.shape)}, got {tuple(labels.shape)}"
            )
        return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(logits, labels.float())
