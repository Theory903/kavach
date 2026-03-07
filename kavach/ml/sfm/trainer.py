import logging
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from kavach.ml.sfm.model import SecurityFoundationModel
from kavach.ml.sfm.dataset import triplet_loss

logger = logging.getLogger(__name__)

class CurriculumTrainer:
    """
    Curriculum Learning Pipeline for the Security Foundation Model.
    
    Phases:
      1. Binary Attack Classifier & Risk Regressor (Easy Baseline)
      2. Multi-class Intent Classifier (Granular)
      3. Triplet / Contrastive Learning (Adversarial Data & OOD)
    """
    
    def __init__(
        self,
        model: SecurityFoundationModel,
        lr: float = 2e-5,
        device: str = "cpu"
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=lr)
        
        self.binary_criterion = nn.CrossEntropyLoss()
        self.intent_criterion = nn.CrossEntropyLoss()
        self.risk_criterion = nn.MSELoss()
        
    def train_phase_1(self, dataloader: DataLoader, epochs: int = 3):
        """Train basic binary classification and continuous risk."""
        logger.info("Starting Phase 1 Curriculum: Binary Attack & Risk")
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["attack_label"].to(self.device)
                risks = batch["risk_score"].to(self.device)
                
                self.optimizer.zero_grad()
                out = self.model(input_ids, mask)
                
                loss_clf = self.binary_criterion(out["attack_logits"], labels)
                loss_risk = self.risk_criterion(out["risk_score"], risks)
                
                loss = loss_clf + (0.5 * loss_risk)
                loss.backward()
                self.optimizer.step()
                
    def train_phase_2(self, dataloader: DataLoader, epochs: int = 3):
        """Train fine-grained intent classifier."""
        logger.info("Starting Phase 2 Curriculum: Intent Taxonomy")
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                intent_labels = batch["intent_label"].to(self.device)
                
                self.optimizer.zero_grad()
                out = self.model(input_ids, mask)
                
                loss = self.intent_criterion(out["intent_logits"], intent_labels)
                loss.backward()
                self.optimizer.step()
                
    def train_phase_3(self, tri_dataloader: DataLoader, epochs: int = 3):
        """Contrastive training using Triplet Margin Loss on the OOD head."""
        logger.info("Starting Phase 3 Curriculum: Triplet Adversarial Embedding")
        self.model.train()
        for epoch in range(epochs):
            for batch in tri_dataloader:
                # Anchor = Benign
                anc_ids = batch["input_ids"].to(self.device)
                anc_mask = batch["attention_mask"].to(self.device)
                
                # Positive = Another Benign
                pos_ids = batch["pos_input_ids"].to(self.device)
                pos_mask = batch["pos_attention_mask"].to(self.device)
                
                # Negative = Adversarial Payload
                neg_ids = batch["neg_input_ids"].to(self.device)
                neg_mask = batch["neg_attention_mask"].to(self.device)
                
                self.optimizer.zero_grad()
                
                out_anc = self.model(anc_ids, anc_mask)["ood_embeddings"]
                out_pos = self.model(pos_ids, pos_mask)["ood_embeddings"]
                out_neg = self.model(neg_ids, neg_mask)["ood_embeddings"]
                
                # Maximize separation between benign anchors and adversarial variants
                loss = triplet_loss(out_anc, out_pos, out_neg, margin=1.0)
                
                loss.backward()
                self.optimizer.step()
