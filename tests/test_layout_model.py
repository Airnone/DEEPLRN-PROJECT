from __future__ import annotations

import torch

from deeplrn.model.layout import LayoutFeatureEncoder


def test_layout_encoder_preserves_shape_and_changes_representation():
    torch.manual_seed(3)
    encoder = LayoutFeatureEncoder(hidden_size=16, max_pages=8, max_sections=8, bbox_bins=32)
    tokens = torch.randn(2, 5, 16)
    result = encoder(
        tokens,
        page_ids=torch.ones(2, 5, dtype=torch.long),
        section_ids=torch.full((2, 5), 2, dtype=torch.long),
        bboxes=torch.full((2, 5, 4), 4, dtype=torch.long),
    )
    assert result.shape == tokens.shape
    assert not torch.equal(result, tokens)


def test_layout_ablation_returns_text_embeddings_unchanged():
    encoder = LayoutFeatureEncoder(
        hidden_size=8,
        use_page=False,
        use_section=False,
        use_bbox=False,
    )
    tokens = torch.randn(1, 4, 8)
    assert torch.equal(encoder(tokens), tokens)


def test_multitask_model_accepts_layout_tensors(monkeypatch):
    import deeplrn.model.encoder as encoder_module
    from deeplrn.model.deeplrn_model import DeepLRNModel, ModelConfig

    class TinyEncoder(torch.nn.Module):
        def __init__(self, model_name="tiny", freeze_layers=0):
            super().__init__()
            self.hidden_size = 16
            self.embedding = torch.nn.Embedding(64, self.hidden_size)
            self.received_bboxes = False

        def forward(self, input_ids, attention_mask, bboxes=None):
            self.received_bboxes = bboxes is not None
            token_embeddings = self.embedding(input_ids)
            return token_embeddings, token_embeddings[:, 0]

    monkeypatch.setattr(encoder_module, "ChunkEncoder", TinyEncoder)
    model = DeepLRNModel(
        ModelConfig(
            encoder_name="tiny",
            encoder_uses_2d_positions=True,
            hidden_size=16,
            doc_num_heads=4,
            max_chunks=4,
            max_pages=8,
            max_sections=8,
            bbox_bins=32,
        )
    )
    input_ids = torch.randint(0, 64, (2, 3, 12))
    attention_mask = torch.ones_like(input_ids)
    output = model(
        input_ids,
        attention_mask,
        chunk_mask=torch.ones(2, 3, dtype=torch.bool),
        page_ids=torch.ones_like(input_ids),
        section_ids=torch.ones_like(input_ids),
        bboxes=torch.ones(2, 3, 12, 4, dtype=torch.long),
    )
    assert output["ner_logits"].shape == (2, 3, 12, 17)
    assert output["cls_logits"].shape == (2, 5)
    assert model.encoder.received_bboxes is True
