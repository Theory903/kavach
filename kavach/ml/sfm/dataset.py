"""Dataset formulation for SFM Curriculum Training.

Handles multiple types of data inputs:
- Classification data (Benign vs Intent)
- Continuous data (Risk Scores)
- Tuple data for contrastive learning (Anchor, Positive, Negative)
"""

from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class SFMDataset(Dataset):
    """
    Multipurpose Dataset supporting the 4 heads of the SFM.
    
    Data should be passed as a dictionary of tensors or lists 
    containing the necessary labels. Wait to tokenize until `__getitem__`
    or pre-tokenize for speed.
    """
    def __init__(
        self,
        texts: List[str],
        attack_labels: Optional[List[int]] = None,
        intent_labels: Optional[List[int]] = None,
        risk_scores: Optional[List[float]] = None,
        triplet_negatives: Optional[List[str]] = None, # For contrastive learning
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 256
    ):
        self.texts = texts
        self.attack_labels = attack_labels
        self.intent_labels = intent_labels
        self.risk_scores = risk_scores
        self.triplet_negatives = triplet_negatives
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        item: Dict[str, Union[torch.Tensor, str]] = {}
        
        text = self.texts[idx]
        
        if self.tokenizer is not None:
            # Anchor text tokenization
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            item["input_ids"] = encoding["input_ids"].squeeze(0)
            item["attention_mask"] = encoding["attention_mask"].squeeze(0)
            
            if "token_type_ids" in encoding:
                item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
                
            # If doing contrastive triplet loss, we also need the negative pair 
            # (e.g., if anchor is benign, negative is an adversarial mutation)
            if self.triplet_negatives is not None:
                neg_text = self.triplet_negatives[idx]
                neg_encoding = self.tokenizer(
                    neg_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                item["neg_input_ids"] = neg_encoding["input_ids"].squeeze(0)
                item["neg_attention_mask"] = neg_encoding["attention_mask"].squeeze(0)
        else:
            item["text"] = text
            if self.triplet_negatives:
                item["neg_text"] = self.triplet_negatives[idx]

        if self.attack_labels is not None:
            item["attack_label"] = torch.tensor(self.attack_labels[idx], dtype=torch.long)
            
        if self.intent_labels is not None:
            item["intent_label"] = torch.tensor(self.intent_labels[idx], dtype=torch.long)
            
        if self.risk_scores is not None:
            item["risk_score"] = torch.tensor(self.risk_scores[idx], dtype=torch.float)

        return item

def triplet_loss(
    anchor: torch.Tensor, 
    positive: torch.Tensor, 
    negative: torch.Tensor, 
    margin: float = 1.0
) -> torch.Tensor:
    """
    Computes Triplet Margin Loss for OOD head.
    Assumes inputs are L2-normalized embeddings.
    
    Pushes D(anchor, positive) towards 0.
    Pushes D(anchor, negative) to be >= margin.
    """
    distance_pos = torch.norm(anchor - positive, p=2, dim=1)
    distance_neg = torch.norm(anchor - negative, p=2, dim=1)
    losses = torch.relu(distance_pos - distance_neg + margin)
    return losses.mean()
