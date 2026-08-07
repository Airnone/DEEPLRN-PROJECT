from __future__ import annotations

from deeplrn.config import ChunkConfig
from deeplrn.preprocessing.pdf_extractor import PageData, WordBox
from deeplrn.schema import AnnotatedDocument
from deeplrn.training.builder import TrainingRecordBuilder


class CharacterTokenizer:
    is_fast = True
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True, max_length=None, truncation=False):
        ids = [ord(char) % 251 + 3 for char in text]
        if add_special_tokens:
            ids = [1] + ids + [2]
        return ids[:max_length] if max_length else ids

    def __call__(
        self,
        text,
        max_length,
        truncation,
        padding,
        return_offsets_mapping,
        return_special_tokens_mask,
    ):
        content = list(text[: max_length - 2])
        ids = [1] + [ord(char) % 251 + 3 for char in content] + [2]
        offsets = [(0, 0)] + [(i, i + 1) for i in range(len(content))] + [(0, 0)]
        special = [1] + [0] * len(content) + [1]
        mask = [1] * len(ids)
        pad = max_length - len(ids)
        return {
            "input_ids": ids + [0] * pad,
            "attention_mask": mask + [0] * pad,
            "offset_mapping": offsets + [(0, 0)] * pad,
            "special_tokens_mask": special + [1] * pad,
        }


def test_builder_aligns_annotations_layout_and_relations():
    text = "Unsupported payment to Example Inc."
    pages = [
        PageData(
            page_number=1,
            text=text,
            headers=["Unsupported payment"],
            word_boxes=[
                WordBox("Unsupported", 10, 10, 70, 20),
                WordBox("Example", 80, 10, 120, 20),
            ],
            metadata={"width": 200, "height": 100},
        )
    ]
    annotation = AnnotatedDocument.from_dict(
        {
            "doc_id": "doc-1",
            "source_pdf": "doc-1.pdf",
            "lgu": "Example LGU",
            "year": 2023,
            "finding_label": "unsupported_disbursement",
            "entities": [
                {
                    "entity_id": "f1",
                    "label": "FINDING",
                    "start_char": 0,
                    "end_char": 19,
                    "text": "Unsupported payment",
                    "page_number": 1,
                },
                {
                    "entity_id": "o1",
                    "label": "ORGANIZATION",
                    "start_char": 23,
                    "end_char": 34,
                    "text": "Example Inc",
                    "page_number": 1,
                },
            ],
            "relations": [
                {
                    "relation_id": "r1",
                    "head_entity_id": "o1",
                    "tail_entity_id": "f1",
                    "label": "ASSOCIATED_WITH",
                    "evidence_page_numbers": [1],
                }
            ],
        }
    )
    builder = TrainingRecordBuilder(
        CharacterTokenizer(),
        ChunkConfig(max_tokens=64, overlap_tokens=8, tokenizer_name="fake"),
    )
    record = builder.build(pages, annotation)
    assert record["finding_label_id"] == 3
    assert len(record["chunks"]) == 1
    assert any(value > 0 for bbox in record["chunks"][0]["bboxes"] for value in bbox)
    assert any(triple[-1] == 3 for triple in record["relation_triples"])
    positive_index = next(
        index for index, triple in enumerate(record["relation_triples"]) if triple[-1] == 3
    )
    assert record["relation_candidate_metadata"][positive_index][
        "evidence_page_numbers"
    ] == [1]
