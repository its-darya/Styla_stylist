"""Ensemble-Level Style Classifier Modelləri (PyTorch MLP).

Tam geyim dəstinin (outfit: top + bottom və ya çoxlu əşyalar) FashionCLIP
embedding-lərini birləşdirərək dəstin aid olduğu stili (məs. casual, formal,
streetwear və s.) təsnif edir.
"""
from __future__ import annotations

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.style import config


class EnsembleStyleMLP(nn.Module):
    """Top və Bottom (və ya digər əşyalar) embedding-lərini birləşdirən Stil Təsnifatçısı."""

    def __init__(
        self,
        num_classes: int = len(config.STYLES),
        emb_dim: int = config.EMB_DIM,
        hidden_dims: list[int] = config.HIDDEN_DIMS,
        dropout: float = config.DROPOUT,
        feature_mode: str = config.INPUT_FEATURE_MODE,
        use_batch_norm: bool = config.USE_BATCH_NORM,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.feature_mode = feature_mode
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        # Xüsusiyyət ölçüsü
        if feature_mode == "concat":
            in_dim = emb_dim * 2
        elif feature_mode == "full_pair":
            in_dim = emb_dim * 4  # [top, bottom, |top-bottom|, top*bottom]
        elif feature_mode == "pooled":
            in_dim = emb_dim * 2  # [mean_pool, max_pool]
        else:
            raise ValueError(f"Dəstəklənməyən feature_mode: {feature_mode!r}")

        self.in_dim = in_dim

        layers: list[nn.Module] = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            current_dim = h_dim

        # Çıxış qatı: sinif sayı qədər logit
        layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def extract_features(
        self, top_emb: torch.Tensor, bottom_emb: torch.Tensor
    ) -> torch.Tensor:
        """Top və Bottom embedding-lərindən birləşdirilmiş xüsusiyyətlər çıxarır."""
        t_norm = F.normalize(top_emb, p=2, dim=-1)
        b_norm = F.normalize(bottom_emb, p=2, dim=-1)

        if self.feature_mode == "concat":
            return torch.cat([t_norm, b_norm], dim=-1)
        elif self.feature_mode == "full_pair":
            diff = torch.abs(t_norm - b_norm)
            prod = t_norm * b_norm
            return torch.cat([t_norm, b_norm, diff, prod], dim=-1)
        elif self.feature_mode == "pooled":
            stacked = torch.stack([t_norm, b_norm], dim=1)  # [B, 2, 512]
            mean_pool = torch.mean(stacked, dim=1)
            max_pool, _ = torch.max(stacked, dim=1)
            return torch.cat([mean_pool, max_pool], dim=-1)
        else:
            raise ValueError(f"Bilinməyən feature_mode: {self.feature_mode}")

    def forward(
        self,
        top_emb: torch.Tensor,
        bottom_emb: torch.Tensor,
        return_prob: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            top_emb: [batch_size, 512] və ya [512]
            bottom_emb: [batch_size, 512] və ya [512]
            return_prob: True olduqda Softmax tətbiq edilir.

        Returns:
            [batch_size, num_classes] logit-lər və ya ehtimallar.
        """
        if top_emb.ndim == 1:
            top_emb = top_emb.unsqueeze(0)
        if bottom_emb.ndim == 1:
            bottom_emb = bottom_emb.unsqueeze(0)

        feats = self.extract_features(top_emb, bottom_emb)
        logits = self.classifier(feats)

        if return_prob:
            return F.softmax(logits, dim=-1)
        return logits

    @torch.no_grad()
    def predict_proba(
        self, top_emb: torch.Tensor, bottom_emb: torch.Tensor
    ) -> torch.Tensor:
        """Inference üçün birbaşa sinif ehtimallarını (Softmax) qaytarır."""
        self.eval()
        return self.forward(top_emb, bottom_emb, return_prob=True)


class MultiItemStyleClassifier(nn.Module):
    """İxtiyari sayda əşyadan ibarət tam outfit-lər üçün Attention-based Stil Təsnifatçısı.

    Giriş: [batch_size, num_items, 512]
    """

    def __init__(
        self,
        num_classes: int = len(config.STYLES),
        emb_dim: int = config.EMB_DIM,
        hidden_dims: list[int] = config.HIDDEN_DIMS,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim

        # Self-attention / Pooling açarı
        self.attention = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        in_dim = emb_dim * 2  # attention weighted sum + max pool
        layers: list[nn.Module] = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, item_embs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            item_embs: [batch_size, num_items, emb_dim]
            mask: [batch_size, num_items] (True = padding)
        """
        if item_embs.ndim == 2:
            item_embs = item_embs.unsqueeze(0)

        norm_embs = F.normalize(item_embs, p=2, dim=-1)
        attn_weights = self.attention(norm_embs)  # [B, N, 1]

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=1)
        weighted_sum = torch.sum(norm_embs * attn_weights, dim=1)  # [B, 512]
        max_pooled, _ = torch.max(norm_embs, dim=1)  # [B, 512]

        outfit_vec = torch.cat([weighted_sum, max_pooled], dim=-1)  # [B, 1024]
        return self.mlp(outfit_vec)
