import torch
import torch.nn as nn

class DocumentTransformer(nn.Module):
    """
    A Transformer that operates on chunk-level CLS representations to capture 
    cross-chunk context.
    """
    def __init__(self, hidden_size: int = 768, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1, max_chunks: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Learnable positional embeddings for chunk positions
        self.position_embeddings = nn.Embedding(max_chunks, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, chunk_embeddings: torch.Tensor, chunk_mask: torch.Tensor = None):
        """
        Forward pass for DocumentTransformer.
        
        Args:
            chunk_embeddings: (num_chunks, hidden_size) or (batch, num_chunks, hidden_size)
            chunk_mask: optional boolean mask for padding chunks, shape (batch, num_chunks).
                       True values indicate padded chunks to be ignored.
                       
        Returns:
            enriched_chunks: same shape as input
        """
        is_2d = chunk_embeddings.dim() == 2
        if is_2d:
            # Add batch dimension: (num_chunks, hidden_size) -> (1, num_chunks, hidden_size)
            chunk_embeddings = chunk_embeddings.unsqueeze(0)
            if chunk_mask is not None and chunk_mask.dim() == 1:
                chunk_mask = chunk_mask.unsqueeze(0)
                
        batch_size, num_chunks, _ = chunk_embeddings.size()
        
        # Add positional embeddings
        positions = torch.arange(num_chunks, dtype=torch.long, device=chunk_embeddings.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.position_embeddings(positions)
        
        embeddings = chunk_embeddings + pos_embeddings
        embeddings = self.dropout(embeddings)
        
        # Run through transformer
        # PyTorch convention: src_key_padding_mask True = IGNORE position
        # Our convention: chunk_mask True = REAL chunk, so invert
        padding_mask = ~chunk_mask if chunk_mask is not None else None
        enriched_chunks = self.transformer(
            embeddings,
            src_key_padding_mask=padding_mask
        )
        
        enriched_chunks = self.layer_norm(enriched_chunks)
        
        if is_2d:
            # Remove batch dimension: (1, num_chunks, hidden_size) -> (num_chunks, hidden_size)
            enriched_chunks = enriched_chunks.squeeze(0)
            
        return enriched_chunks
