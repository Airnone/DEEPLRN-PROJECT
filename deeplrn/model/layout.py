"""Learned fusion of page, section, and normalized bounding-box features."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayoutFeatureEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        max_pages: int = 512,
        max_sections: int = 256,
        bbox_bins: int = 1024,
        use_page: bool = True,
        use_section: bool = True,
        use_bbox: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_pages = max_pages
        self.max_sections = max_sections
        self.bbox_bins = bbox_bins
        self.use_page = use_page
        self.use_section = use_section
        self.use_bbox = use_bbox
        self.page_embedding = nn.Embedding(max_pages + 1, hidden_size, padding_idx=0)
        self.section_embedding = nn.Embedding(max_sections + 1, hidden_size, padding_idx=0)
        self.x0_embedding = nn.Embedding(bbox_bins, hidden_size, padding_idx=0)
        self.y0_embedding = nn.Embedding(bbox_bins, hidden_size, padding_idx=0)
        self.x1_embedding = nn.Embedding(bbox_bins, hidden_size, padding_idx=0)
        self.y1_embedding = nn.Embedding(bbox_bins, hidden_size, padding_idx=0)
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        page_ids: torch.Tensor | None = None,
        section_ids: torch.Tensor | None = None,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layout = torch.zeros_like(token_embeddings)
        count = 0
        if self.use_page and page_ids is not None:
            layout = layout + self.page_embedding(page_ids.clamp(0, self.max_pages))
            count += 1
        if self.use_section and section_ids is not None:
            layout = layout + self.section_embedding(section_ids.clamp(0, self.max_sections))
            count += 1
        if self.use_bbox and bboxes is not None:
            coords = bboxes.clamp(0, self.bbox_bins - 1)
            layout = layout + self.x0_embedding(coords[..., 0])
            layout = layout + self.y0_embedding(coords[..., 1])
            layout = layout + self.x1_embedding(coords[..., 2])
            layout = layout + self.y1_embedding(coords[..., 3])
            count += 4
        if count == 0:
            return token_embeddings
        layout = layout / count
        fused = self.fusion(torch.cat([token_embeddings, layout], dim=-1))
        return self.layer_norm(token_embeddings + self.dropout(self.activation(fused)))
