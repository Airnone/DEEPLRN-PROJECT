from __future__ import annotations
import torch
import torch.nn as nn

class ViolationClassifier(nn.Module):
    """
    Document-level classifier for violation prediction.
    """
    def __init__(self, hidden_size: int = 768, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
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
        Compute cross-entropy loss for classification.
        
        Args:
            logits: (batch, num_classes)
            labels: (batch)
            
        Returns:
            loss: scalar tensor
        """
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return loss
