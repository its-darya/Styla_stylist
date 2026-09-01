"""Ensemble-Level Style Classifier Təlim Skripti (PyTorch + Weights & Biases).

D-nin pseudo-label-lənmiş tam outfit-ləri (top+bottom embedding-ləri və stil etiketləri)
üzərində MLP modelini öyrədir və W&B-də qeydə alır.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.style import config
from ml.style.dataset import (
    OutfitStyleDataset,
    StyleEncoder,
    load_style_data_from_npz,
    split_style_dataset,
)
from ml.style.model import EnsembleStyleMLP


def init_wandb(
    experiment_name: Optional[str] = None,
    config_dict: Optional[dict] = None,
    mode: str = config.WANDB_MODE,
) -> Any:
    try:
        import wandb

        if not os.getenv("WANDB_API_KEY") and mode == "online":
            mode = "offline"

        run = wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=experiment_name,
            config=config_dict,
            mode=mode,
            reinit=True,
        )
        return run
    except Exception as exc:
        print(f"[W&B Warning] W&B başladıla bilmədi ({exc}). Təlim W&B olmadan davam edir.")
        return None


def evaluate_style(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Stil modelini qiymətləndirir: Loss, Top-1 Acc, Top-3 Acc, Macro-F1."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []
    top3_correct = 0
    total_samples = 0

    with torch.no_grad():
        for top_embs, bottom_embs, labels in data_loader:
            top_embs = top_embs.to(device)
            bottom_embs = bottom_embs.to(device)
            labels = labels.to(device)

            logits = model(top_embs, bottom_embs, return_prob=False)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)

            # Top-3 Accuracy
            if logits.shape[1] >= 3:
                _, top3_indices = torch.topk(logits, k=3, dim=-1)
                for i in range(len(labels)):
                    if labels[i] in top3_indices[i]:
                        top3_correct += 1
            else:
                top3_correct += int(np.sum(preds == targets))

            total_samples += len(labels)

    if total_samples == 0:
        return {"loss": 0.0, "acc": 0.0, "top3_acc": 0.0, "macro_f1": 0.0}

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    avg_loss = total_loss / total_samples
    acc = float(accuracy_score(y_true, y_pred))
    top3_acc = float(top3_correct / total_samples)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "top3_accuracy": top3_acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "num_samples": float(total_samples),
    }


def train_style_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    num_classes: int = len(config.STYLES),
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    hidden_dims: list[int] = config.HIDDEN_DIMS,
    dropout: float = config.DROPOUT,
    feature_mode: str = config.INPUT_FEATURE_MODE,
    device: Optional[str] = None,
    save_path: Path = config.DEFAULT_MODEL_PATH,
    encoder: Optional[StyleEncoder] = None,
    wandb_run: Any = None,
) -> tuple[EnsembleStyleMLP, dict[str, Any]]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Stil Təlim cihazı: {dev}")

    model = EnsembleStyleMLP(
        num_classes=num_classes,
        emb_dim=config.EMB_DIM,
        hidden_dims=hidden_dims,
        dropout=dropout,
        feature_mode=feature_mode,
        use_batch_norm=config.USE_BATCH_NORM,
    ).to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_f1 = -1.0
    best_epoch = 0
    best_state_dict = None
    history = []

    print(
        f"Stil təlimi başlayır: {epochs} epoch, {num_classes} sinif, rejim: {feature_mode}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for top_embs, bottom_embs, labels in train_loader:
            top_embs = top_embs.to(dev)
            bottom_embs = bottom_embs.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()
            logits = model(top_embs, bottom_embs, return_prob=False)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_samples += len(labels)

        avg_train_loss = train_loss / max(train_samples, 1)

        val_metrics = evaluate_style(model, val_loader, criterion, dev)
        scheduler.step(val_metrics["macro_f1"])

        epoch_stats = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_top3_acc": val_metrics["top3_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_stats)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Top-3: {val_metrics['top3_accuracy']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(epoch_stats)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state_dict = model.state_dict().copy()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_metrics = {}
    if test_loader is not None:
        test_metrics = evaluate_style(model, test_loader, criterion, dev)
        print("\n" + "=" * 50)
        print(f"Stil Test Nəticələri (Best Epoch {best_epoch}):")
        print(f"  Test Loss    : {test_metrics['loss']:.4f}")
        print(f"  Test Top-1   : {test_metrics['accuracy']:.4f}")
        print(f"  Test Top-3   : {test_metrics['top3_accuracy']:.4f}")
        print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f}")
        print("=" * 50)

        if wandb_run is not None:
            wandb_run.log({f"test_{k}": v for k, v in test_metrics.items()})

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "num_classes": num_classes,
            "emb_dim": config.EMB_DIM,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "feature_mode": feature_mode,
            "use_batch_norm": config.USE_BATCH_NORM,
        },
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Stil Modeli qeyd edildi: {save_path}")

    # Sinifləri saxlayırıq
    if encoder is not None:
        encoder.save(config.DEFAULT_CLASSES_PATH)

    report = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history,
    }
    config.DEFAULT_METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return model, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensemble Style Classifier modelinin öyrədilməsi")
    parser.add_argument("--data-path", type=str, default=None, help=".npz stil dataset faylının yolu")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=config.HIDDEN_DIMS)
    parser.add_argument("--dropout", type=float, default=config.DROPOUT)
    parser.add_argument("--feature-mode", type=str, default=config.INPUT_FEATURE_MODE,
                        choices=["concat", "full_pair", "pooled"])
    parser.add_argument("--save-path", type=str, default=str(config.DEFAULT_MODEL_PATH))
    parser.add_argument("--wandb-mode", type=str, default=config.WANDB_MODE,
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    encoder = StyleEncoder(config.STYLES)

    if args.data_path and Path(args.data_path).exists():
        print(f"Dataset yüklənir: {args.data_path}")
        top_embs, bottom_embs, labels, classes = load_style_data_from_npz(args.data_path)
        encoder = StyleEncoder(classes)
        dataset = OutfitStyleDataset(top_embs, bottom_embs, labels, encoder=encoder)
    else:
        print("Data-path verilməyib. Nümunə / pseudo-label stil dataseti simulyasiya edilir...")
        np.random.seed(args.seed)
        num_samples = 600
        num_classes = encoder.num_classes()

        # Hər stil üçün xarakterik klaster mərkəzləri
        style_centers = np.random.randn(num_classes, config.EMB_DIM).astype(np.float32)
        style_centers /= np.linalg.norm(style_centers, axis=1, keepdims=True)

        top_list = []
        bottom_list = []
        label_list = []

        for i in range(num_samples):
            cls_idx = i % num_classes
            center = style_centers[cls_idx]

            top = center + 0.4 * np.random.randn(config.EMB_DIM).astype(np.float32)
            bottom = center + 0.4 * np.random.randn(config.EMB_DIM).astype(np.float32)

            top /= np.linalg.norm(top)
            bottom /= np.linalg.norm(bottom)

            top_list.append(top)
            bottom_list.append(bottom)
            label_list.append(cls_idx)

        perm = np.random.permutation(num_samples)
        top_arr = np.array(top_list, dtype=np.float32)[perm]
        bottom_arr = np.array(bottom_list, dtype=np.float32)[perm]
        label_arr = np.array(label_list)[perm]

        dataset = OutfitStyleDataset(top_arr, bottom_arr, label_arr, encoder=encoder)

    print(f"Dataset nümunə sayı: {len(dataset)}, Sinif sayı: {encoder.num_classes()}")

    train_ds, val_ds, test_ds = split_style_dataset(dataset, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    wandb_run = init_wandb(
        experiment_name=f"style-mlp-{args.feature_mode}",
        config_dict=vars(args),
        mode=args.wandb_mode,
    )

    train_style_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=encoder.num_classes(),
        epochs=args.epochs,
        lr=args.lr,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        feature_mode=args.feature_mode,
        save_path=Path(args.save_path),
        encoder=encoder,
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
