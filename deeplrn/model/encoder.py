import torch
import torch.nn as nn
from transformers import AutoModel

class ChunkEncoder(nn.Module):
    """
    Encoder for processing text chunks using a pre-trained RoBERTa model.
    """
    def __init__(self, model_name: str = 'roberta-base', freeze_layers: int = 0):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.roberta.config.hidden_size
        
        # Freeze embedding layer and first N transformer layers if requested
        if freeze_layers > 0:
            for param in self.roberta.embeddings.parameters():
                param.requires_grad = False
                
            for i in range(min(freeze_layers, len(self.roberta.encoder.layer))):
                for param in self.roberta.encoder.layer[i].parameters():
                    param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward pass for the ChunkEncoder.
        
        Args:
            input_ids: (batch, seq_len) tensor of token ids
            attention_mask: (batch, seq_len) tensor for attention mask
            
        Returns:
            token_embeddings: (batch, seq_len, hidden_size) — all token embeddings
            cls_embedding: (batch, hidden_size) — the <s> token embedding
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        token_embeddings = outputs.last_hidden_state
        cls_embedding = token_embeddings[:, 0, :]
        
        return token_embeddings, cls_embedding
