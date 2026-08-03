from __future__ import annotations
import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, Any, Optional
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .dataset import collate_documents

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    batch_size: int = 2          # small because documents are large
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Multi-task loss weights
    ner_loss_weight: float = 1.0
    cls_loss_weight: float = 0.5
    rel_loss_weight: float = 0.5
    
    # Paths
    output_dir: str = 'checkpoints'
    log_every: int = 10
    eval_every: int = 50
    save_every: int = 100

class Trainer:
    def __init__(self, model: nn.Module, train_dataset, eval_dataset=None, config: Optional[TrainingConfig]=None):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or TrainingConfig()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            collate_fn=collate_documents
        )
        
        if self.eval_dataset:
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_documents
            )
        else:
            self.eval_loader = None
            
        self.optimizer, self.scheduler = self._setup_optimizers()
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
    def _setup_optimizers(self):
        # Separate weight decay for bias/LayerNorm
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        
        total_steps = (len(self.train_loader) // self.config.gradient_accumulation_steps) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler

    def train(self):
        self.model.train()
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_iterator = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
            for step, batch in enumerate(epoch_iterator):
                
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                chunk_mask = batch['chunk_mask'].to(self.device)
                ner_labels = batch['ner_labels'].to(self.device)
                violation_labels = batch['violation_labels'].to(self.device)
                relation_triples = batch['relation_triples'] # List, stays on CPU usually until used
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    chunk_mask=chunk_mask,
                    ner_labels=ner_labels,
                    violation_labels=violation_labels,
                    relation_triples=relation_triples
                )
                
                # Compute weighted multi-task loss if model didn't do it directly
                if 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    loss = 0.0
                    if 'ner_loss' in outputs:
                        loss += self.config.ner_loss_weight * outputs['ner_loss']
                    if 'cls_loss' in outputs:
                        loss += self.config.cls_loss_weight * outputs['cls_loss']
                    if 'rel_loss' in outputs:
                        loss += self.config.rel_loss_weight * outputs['rel_loss']
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    # Optimizer step + scheduler step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % self.config.log_every == 0:
                        logger.info(f"Step {global_step} - Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f}")
                        
                    if self.eval_loader and global_step % self.config.eval_every == 0:
                        metrics = self.evaluate()
                        logger.info(f"Eval metrics at step {global_step}: {metrics}")
                        self.model.train()
                        
                    if global_step % self.config.save_every == 0:
                        self.save_checkpoint(os.path.join(self.config.output_dir, f"checkpoint-{global_step}.pt"))

    def evaluate(self) -> Dict[str, float]:
        """Run eval on eval_dataset and compute per-task metrics."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                chunk_mask = batch['chunk_mask'].to(self.device)
                ner_labels = batch['ner_labels'].to(self.device)
                violation_labels = batch['violation_labels'].to(self.device)
                relation_triples = batch['relation_triples']
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    chunk_mask=chunk_mask,
                    ner_labels=ner_labels,
                    violation_labels=violation_labels,
                    relation_triples=relation_triples
                )
                
                loss = outputs.get('loss')
                if loss is not None:
                    total_loss += loss.item()
                    
        # Dummy metrics here, need to integrate proper precision/recall/F1 with seqeval/sklearn
        metrics = {
            "eval_loss": total_loss / len(self.eval_loader) if len(self.eval_loader) > 0 else 0.0,
            "ner_f1": 0.0, # Placeholder
            "cls_accuracy": 0.0, # Placeholder 
            "rel_f1": 0.0 # Placeholder
        }
        
        return metrics

    def save_checkpoint(self, path: str):
        """Save model state dict + optimizer state + config"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.__dict__
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Checkpoint loaded from {path}")
