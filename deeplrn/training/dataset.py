from __future__ import annotations
import os
import json
import random
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass 
class DocumentSample:
    """A single training document."""
    doc_id: str
    chunk_input_ids: torch.Tensor      # (num_chunks, max_tokens)
    chunk_attention_masks: torch.Tensor # (num_chunks, max_tokens) 
    ner_labels: torch.Tensor           # (num_chunks, max_tokens) - tag indices, -100 for padding
    violation_label: int               # single label for the document
    # Relation triples: list of (head_chunk_idx, head_start, head_end, tail_chunk_idx, tail_start, tail_end, relation_type_idx)
    relation_triples: List[Tuple[int, int, int, int, int, int, int]]
    num_chunks: int

class DeepLRNDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer_name: str = 'roberta-base', max_tokens: int = 384):
        self.data_dir = data_dir
        self.tokenizer_name = tokenizer_name
        self.max_tokens = max_tokens
        
        self.doc_files = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.json')
        ] if os.path.exists(data_dir) else []
    
    def __getitem__(self, idx: int) -> DocumentSample:
        doc_path = self.doc_files[idx]
        with open(doc_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        doc_id = data.get("doc_id", f"doc_{idx}")
        chunks = data.get("chunks", [])
        
        num_chunks = len(chunks)
        chunk_input_ids = []
        chunk_attention_masks = []
        ner_labels = []
        
        # We assume 1 is the padding token ID for roberta-base
        pad_token_id = 1
        
        for chunk in chunks:
            input_ids = chunk.get("input_ids", [])
            attention_mask = chunk.get("attention_mask", [])
            ner_lbls = chunk.get("ner_labels", [])
            
            # Truncate or pad to max_tokens
            if len(input_ids) > self.max_tokens:
                input_ids = input_ids[:self.max_tokens]
                attention_mask = attention_mask[:self.max_tokens]
                ner_lbls = ner_lbls[:self.max_tokens]
            else:
                pad_len = self.max_tokens - len(input_ids)
                input_ids.extend([pad_token_id] * pad_len)
                attention_mask.extend([0] * pad_len)
                ner_lbls.extend([-100] * pad_len)
                
            chunk_input_ids.append(input_ids)
            chunk_attention_masks.append(attention_mask)
            ner_labels.append(ner_lbls)
            
        if num_chunks == 0:
            # Handle empty document edge-case safely
            chunk_input_ids = [[pad_token_id] * self.max_tokens]
            chunk_attention_masks = [[0] * self.max_tokens]
            ner_labels = [[-100] * self.max_tokens]
            num_chunks = 1
            
        return DocumentSample(
            doc_id=doc_id,
            chunk_input_ids=torch.tensor(chunk_input_ids, dtype=torch.long),
            chunk_attention_masks=torch.tensor(chunk_attention_masks, dtype=torch.long),
            ner_labels=torch.tensor(ner_labels, dtype=torch.long),
            violation_label=data.get("violation_label", 0),
            relation_triples=data.get("relation_triples", []),
            num_chunks=num_chunks
        )
    
    def __len__(self) -> int:
        return len(self.doc_files)

def collate_documents(batch: List[DocumentSample]) -> Dict[str, Any]:
    """Custom collate function that handles variable-length documents.
    
    Since documents have different numbers of chunks, we pad to the max
    number of chunks in the batch and create a chunk_mask.
    """
    batch_size = len(batch)
    max_num_chunks = max((sample.num_chunks for sample in batch), default=1)
    
    # Assume max_tokens is consistent across all samples (enforced by Dataset)
    max_tokens = batch[0].chunk_input_ids.size(1)
    
    # Pad token for RoBERTa is 1
    input_ids = torch.ones((batch_size, max_num_chunks, max_tokens), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_num_chunks, max_tokens), dtype=torch.long)
    chunk_mask = torch.zeros((batch_size, max_num_chunks), dtype=torch.bool)
    ner_labels = torch.full((batch_size, max_num_chunks, max_tokens), -100, dtype=torch.long)
    violation_labels = torch.zeros(batch_size, dtype=torch.long)
    
    relation_triples = []
    doc_ids = []
    
    for i, sample in enumerate(batch):
        c = sample.num_chunks
        input_ids[i, :c, :] = sample.chunk_input_ids
        attention_mask[i, :c, :] = sample.chunk_attention_masks
        chunk_mask[i, :c] = True
        ner_labels[i, :c, :] = sample.ner_labels
        
        violation_labels[i] = sample.violation_label
        relation_triples.append(sample.relation_triples)
        doc_ids.append(sample.doc_id)
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "chunk_mask": chunk_mask,
        "ner_labels": ner_labels,
        "violation_labels": violation_labels,
        "relation_triples": relation_triples,
        "doc_ids": doc_ids
    }

def create_dummy_dataset(output_dir: str, num_docs: int = 5):
    """Generate synthetic annotated documents for pipeline testing.
    
    Creates JSON files with random but structurally valid annotations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_docs):
        num_chunks = random.randint(1, 4)
        chunks = []
        for _ in range(num_chunks):
            seq_len = random.randint(10, 50)
            chunk = {
                "input_ids": [random.randint(2, 1000) for _ in range(seq_len)],
                "attention_mask": [1 for _ in range(seq_len)],
                "ner_labels": [random.choice([-100, 0, 1, 2]) for _ in range(seq_len)]
            }
            chunks.append(chunk)
            
        relations = []
        if random.random() > 0.3:
            num_rels = random.randint(1, 3)
            for _ in range(num_rels):
                # Triple: (head_chunk, head_start, head_end, tail_chunk, tail_start, tail_end, rel_type)
                relations.append([
                    random.randint(0, num_chunks - 1), random.randint(0, 5), random.randint(6, 10),
                    random.randint(0, num_chunks - 1), random.randint(0, 5), random.randint(6, 10),
                    random.randint(0, 2)
                ])
                
        doc_data = {
            "doc_id": f"dummy_doc_{i}",
            "chunks": chunks,
            "violation_label": random.randint(0, 4),
            "relation_triples": relations
        }
        
        with open(os.path.join(output_dir, f"doc_{i}.json"), "w", encoding="utf-8") as f:
            json.dump(doc_data, f, indent=2)
