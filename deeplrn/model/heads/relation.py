from __future__ import annotations
import torch
import torch.nn as nn

class RelationExtractor(nn.Module):
    """
    Relation extraction head using a bilinear scoring layer.
    """
    def __init__(self, hidden_size: int = 768, num_relations: int = 4, dropout: float = 0.1):
        super().__init__()
        self.head_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.tail_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.bilinear = nn.Bilinear(hidden_size // 2, hidden_size // 2, num_relations)
        
    def forward(self, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for relation extraction.
        
        Args:
            head_embeddings: (num_pairs, hidden_size) - representations for head entities
            tail_embeddings: (num_pairs, hidden_size) - representations for tail entities
            
        Returns:
            logits: (num_pairs, num_relations)
        """
        # Project entities
        head = self.head_proj(head_embeddings)
        head = self.act(head)
        head = self.dropout(head)
        
        tail = self.tail_proj(tail_embeddings)
        tail = self.act(tail)
        tail = self.dropout(tail)
        
        # Bilinear scoring
        logits = self.bilinear(head, tail)
        return logits
        
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for relation extraction.
        
        Args:
            logits: (num_pairs, num_relations)
            labels: (num_pairs)
            
        Returns:
            loss: scalar tensor
        """
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return loss
