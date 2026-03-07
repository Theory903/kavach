"""Security Foundation Model (SFM) Architecture.

A multi-head transformer designed to understand attacker intent, 
adversarial language, and AI exploitation techniques.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModel, AutoConfig


class SecurityFoundationModel(nn.Module):
    """
    Security Foundation Model (SFM)

    Combines a pre-trained transformer encoder with 4 task-specific heads:
    - Attack Classification (Binary: Benign vs Attack)
    - Intent Classification (Multi-class: Injection, Jailbreak, Exfiltration, etc.)
    - Risk Regression (Continuous score [0, 1])
    - OOD Distance (Embedding projection for manifold distance calculation)
    """

    def __init__(
        self,
        base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_intent_classes: int = 5,
        hidden_dropout_prob: float = 0.1,
        ood_embedding_dim: int = 128
    ) -> None:
        super().__init__()
        
        # Load the base transformer encoder
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)
        hidden_size = self.config.hidden_size
        
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Head 1: Attack Classifier (Binary: Logits)
        self.attack_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(hidden_size // 2, 2)
        )
        
        # Head 2: Intent Classifier (Multi-class: Logits)
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(hidden_size // 2, num_intent_classes)
        )
        
        # Head 3: Risk Regressor (Predicts a scalar risk score)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Force output between 0 and 1
        )
        
        # Head 4: OOD Embedding Projection
        # Projects the pooler output into a lower-dimensional space optimized 
        # via contrastive learning for distance-based OOD detection
        self.ood_head = nn.Sequential(
            nn.Linear(hidden_size, ood_embedding_dim),
            nn.LayerNorm(ood_embedding_dim)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass producing 4 security signals.
        """
        # Pass through the base transformer
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Use mean pooling over the sequence as the sentence representation
        # (Alternatively, could use [CLS] token if the base model was trained that way)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        pooled_output = self.dropout(pooled_output)
        
        # Compute head outputs
        attack_logits = self.attack_head(pooled_output)
        intent_logits = self.intent_head(pooled_output)
        risk_score = self.risk_head(pooled_output).squeeze(-1)
        ood_embeddings = self.ood_head(pooled_output)
        
        # L2 Normalize the OOD embeddings to place them on a hypersphere 
        # (crucial for Triplet/InfoNCE contrastive cosine distance)
        ood_embeddings = torch.nn.functional.normalize(ood_embeddings, p=2, dim=1)
        
        return {
            "attack_logits": attack_logits,
            "intent_logits": intent_logits,
            "risk_score": risk_score,
            "ood_embeddings": ood_embeddings
        }
