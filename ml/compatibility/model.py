"""Compatibility (Uyğunluq) Modelləri — PyTorch MLP və Type-Aware Arxitekturalar.

Geyim cütlərinin (məs. top + bottom və ya ayaqqabı + şalvar) FashionCLIP
embedding-ləri üzərində uyğunluq ehtimalını (0 ilə 1 arasında) təxmin edir.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.compatibility import config


class CompatibilityMLP(nn.Module):
    """FashionCLIP embedding cütləri üzərində işləyən iki/üç qatlı MLP.

    Giriş:
        e1: Tensor [batch_size, emb_dim] — 1-ci əşyanın embedding-i
        e2: Tensor [batch_size, emb_dim] — 2-ci əşyanın embedding-i

    Xüsusiyyət reprezentasiyası (mode):
        - "symmetric_full": [|e1-e2|, e1*e2, (e1+e2)/2, cos_sim] (dim = 512*3 + 1 = 1537)
          Sıra-müstəqil (symmetric) xüsusiyyətlər: (e1, e2) və (e2, e1) eyni balı alır.
        - "diff_prod": [|e1-e2|, e1*e2] (dim = 1024)
        - "concat": [e1, e2] (dim = 1024)
    """

    def __init__(
        self,
        emb_dim: int = config.EMB_DIM,
        hidden_dims: list[int] = config.HIDDEN_DIMS,
        dropout: float = config.DROPOUT,
        mode: str = config.INPUT_REPRESENTATION,
        use_batch_norm: bool = config.USE_BATCH_NORM,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.mode = mode
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm

        # Giriş ölçüsünü müəyyən edirik
        if mode == "symmetric_full":
            in_dim = emb_dim * 3 + 1
        elif mode in ("diff_prod", "concat"):
            in_dim = emb_dim * 2
        else:
            raise ValueError(f"Dəstəklənməyən reprezentasiya rejimi: {mode!r}")

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

        # Çıxış qatı: tək skalyar logit
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def extract_features(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """İki embedding-dən birləşdirilmiş xüsusiyyət vektorunu çıxarır."""
        # Vektorları L2-normalize edirik (əgər hələ olunmayıbsa)
        e1_norm = F.normalize(e1, p=2, dim=-1)
        e2_norm = F.normalize(e2, p=2, dim=-1)

        if self.mode == "symmetric_full":
            diff = torch.abs(e1_norm - e2_norm)
            prod = e1_norm * e2_norm
            avg = (e1_norm + e2_norm) * 0.5
            cos_sim = torch.sum(e1_norm * e2_norm, dim=-1, keepdim=True)
            return torch.cat([diff, prod, avg, cos_sim], dim=-1)
        elif self.mode == "diff_prod":
            diff = torch.abs(e1_norm - e2_norm)
            prod = e1_norm * e2_norm
            return torch.cat([diff, prod], dim=-1)
        elif self.mode == "concat":
            return torch.cat([e1_norm, e2_norm], dim=-1)
        else:
            raise ValueError(f"Bilinməyən rejim: {self.mode}")

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        return_prob: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            e1: [batch_size, emb_dim] və ya [emb_dim]
            e2: [batch_size, emb_dim] və ya [emb_dim]
            return_prob: True olduqda Sigmoid tətbiq olunur və [0, 1] ehtimal qaytarılır.

        Returns:
            logits və ya ehtimal tensoru [batch_size, 1]
        """
        # Əgər 1D tensor verilibsə batch ölçüsü əlavə edirik
        if e1.ndim == 1:
            e1 = e1.unsqueeze(0)
        if e2.ndim == 1:
            e2 = e2.unsqueeze(0)

        feats = self.extract_features(e1, e2)
        logits = self.net(feats)

        if return_prob:
            return torch.sigmoid(logits)
        return logits

    @torch.no_grad()
    def predict_proba(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """Inference üçün birbaşa ehtimal (0..1) qaytarır."""
        self.eval()
        return self.forward(e1, e2, return_prob=True)


class TypeAwareCompatibilityModel(nn.Module):
    """Kateqoriya/Tip-məlumatlı (Type-Aware) Uyğunluq Modeli.

    Hər geyim kateqoriyası (məs. top, bottom, shoes) üçün kiçik öyrənilən
    embedding istifadə edir və əşyaların vizual xüsusiyyətlərini kateqoriyaya
    uyğun laylara proyeksiya edir.
    """

    def __init__(
        self,
        emb_dim: int = config.EMB_DIM,
        num_categories: int = config.NUM_CATEGORIES,
        category_emb_dim: int = config.CATEGORY_EMB_DIM,
        hidden_dims: list[int] = config.HIDDEN_DIMS,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.num_categories = num_categories
        self.category_embedding = nn.Embedding(num_categories, category_emb_dim)

        # Vizual embedding + kateqoriya embedding -> Proyeksiya qatı
        item_input_dim = emb_dim + category_emb_dim
        proj_dim = hidden_dims[0] if hidden_dims else 256
        self.item_proj = nn.Sequential(
            nn.Linear(item_input_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Proyeksiya olunmuş vektorlar üzərində uyğunluq MLP-si
        in_dim = proj_dim * 2 + 1  # [z1, z2, z1*z2 dot]
        layers: list[nn.Module] = []
        current_dim = in_dim
        for h_dim in hidden_dims[1:]:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        return_prob: bool = False,
    ) -> torch.Tensor:
        if e1.ndim == 1:
            e1, e2 = e1.unsqueeze(0), e2.unsqueeze(0)
        if c1.ndim == 0:
            c1, c2 = c1.unsqueeze(0), c2.unsqueeze(0)

        e1_norm = F.normalize(e1, p=2, dim=-1)
        e2_norm = F.normalize(e2, p=2, dim=-1)

        cat1_emb = self.category_embedding(c1)
        cat2_emb = self.category_embedding(c2)

        x1 = torch.cat([e1_norm, cat1_emb], dim=-1)
        x2 = torch.cat([e2_norm, cat2_emb], dim=-1)

        z1 = self.item_proj(x1)
        z2 = self.item_proj(x2)

        dot = torch.sum(z1 * z2, dim=-1, keepdim=True)
        feats = torch.cat([torch.abs(z1 - z2), z1 * z2, dot], dim=-1)
        logits = self.classifier(feats)

        if return_prob:
            return torch.sigmoid(logits)
        return logits
