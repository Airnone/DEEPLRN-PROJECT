from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class NERHead(nn.Module):
    """
    Named Entity Recognition head for sequence labeling.
    """
    def __init__(self, hidden_size: int = 768, num_tags: int = 17, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        
        self.proj = nn.Linear(hidden_size, hidden_size // 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size // 2, num_tags)
        
    def forward(
        self, 
        token_embeddings: torch.Tensor, 
        document_context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass for the NER head.
        
        Args:
            token_embeddings: (batch, seq_len, hidden_size)
            document_context: (batch, hidden_size), optional - enriched chunk CLS from doc transformer
            
        Returns:
            logits: (batch, seq_len, num_tags)
        """
        x = token_embeddings
        
        if document_context is not None:
            # document_context: (batch, hidden_size)
            # Expand to (batch, seq_len, hidden_size)
            batch_size, seq_len, _ = token_embeddings.size()
            doc_context_expanded = document_context.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Concatenate
            x = torch.cat([token_embeddings, doc_context_expanded], dim=-1)
            
            # Fuse back to hidden_size
            x = self.fusion(x)
            
        x = self.proj(x)
        x = self.act(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits
        
    def compute_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for NER.
        
        Args:
            logits: (batch, seq_len, num_tags)
            labels: (batch, seq_len)
            attention_mask: (batch, seq_len), optional
            
        Returns:
            loss: scalar tensor
        """
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Only keep active parts of the loss if attention_mask is provided
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_tags)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_tags), labels.view(-1))
            
        return loss
