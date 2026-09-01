"""Compatibility Modelinin Öyrədilməsi (PyTorch + Weights & Biases).

Polyvore embedding cütləri (pozitiv və neqativ nümunələr) üzərində
iki-qatlı MLP modelini öyrədir və nəticələri W&B-də qeydə alır.
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.compatibility import config
from ml.compatibility.dataset import (
    PairCompatibilityDataset,
    get_dataloaders,
    load_pairs_from_npz,
    split_dataset,
)
from ml.compatibility.model import CompatibilityMLP


def init_wandb(
    experiment_name: Optional[str] = None,
    config_dict: Optional[dict] = None,
    mode: str = config.WANDB_MODE,
) -> Any:
    """Weights & Biases logger-i başladır (offline və ya online)."""
    try:
        import wandb

        # Əgər API key yoxdursa avtomatik offline rejimə keçirik
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


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Modeli verilən DataLoader üzərində qiymətləndirir və metrikləri hesablayır."""
    model.eval()
    total_loss = 0.0
    all_targets: list[float] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for e1, e2, labels in data_loader:
            e1 = e1.to(device)
            e2 = e2.to(device)
            labels = labels.to(device)

            logits = model(e1, e2, return_prob=False)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets = labels.cpu().numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(targets)

    n_samples = len(all_targets)
    if n_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "auc": 0.0, "f1": 0.0}

    y_true = np.array(all_targets)
    y_scores = np.array(all_probs)
    y_pred = (y_scores >= 0.5).astype(int)

    avg_loss = total_loss / n_samples
    acc = float(accuracy_score(y_true, y_pred))
    
    # AUC hesablama (hər iki sinif mövcud olduqda)
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_scores))
    else:
        auc = 0.5

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "num_samples": float(n_samples),
    }


def train_compatibility_model(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    hidden_dims: list[int] = config.HIDDEN_DIMS,
    dropout: float = config.DROPOUT,
    mode: str = config.INPUT_REPRESENTATION,
    device: Optional[str] = None,
    save_path: Path = config.DEFAULT_MODEL_PATH,
    wandb_run: Any = None,
) -> tuple[CompatibilityMLP, dict[str, Any]]:
    """PyTorch Compatibility MLP modelinin tam təlim dövrü."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Təlim cihazı: {dev}")

    model = CompatibilityMLP(
        emb_dim=config.EMB_DIM,
        hidden_dims=hidden_dims,
        dropout=dropout,
        mode=mode,
        use_batch_norm=config.USE_BATCH_NORM,
    ).to(dev)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_auc = -1.0
    best_epoch = 0
    best_state_dict = None
    history: list[dict[str, Any]] = []

    print(f"Təlim başlayır: {epochs} epoch, model giriş rejimi: {mode}, hidden: {hidden_dims}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for e1, e2, labels in train_loader:
            e1 = e1.to(dev)
            e2 = e2.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()
            logits = model(e1, e2, return_prob=False)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_samples += len(labels)

        avg_train_loss = train_loss / max(train_samples, 1)

        # Qiymətləndirmə
        val_metrics = evaluate(model, val_loader, criterion, dev)
        scheduler.step(val_metrics["auc"])

        epoch_stats = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_f1": val_metrics["f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_stats)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(epoch_stats)

        # Ən yaxşı modeli yadda saxlayırıq (AUC əsasında)
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            best_state_dict = model.state_dict().copy()

    # Ən yaxşı çəkiləri yükləyirik
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Test qiymətləndirməsi
    test_metrics = {}
    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, criterion, dev)
        print("\n" + "=" * 50)
        print(f"Test Nəticələri (Best Epoch {best_epoch}):")
        print(f"  Test Loss: {test_metrics['loss']:.4f}")
        print(f"  Test Acc : {test_metrics['accuracy']:.4f}")
        print(f"  Test AUC : {test_metrics['auc']:.4f} (Hədəf: >= {config.TARGET_AUC})")
        print(f"  Test F1  : {test_metrics['f1']:.4f}")
        print("=" * 50)

        if wandb_run is not None:
            wandb_run.log({f"test_{k}": v for k, v in test_metrics.items()})

    # Modeli diskə yazırıq
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "emb_dim": config.EMB_DIM,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "mode": mode,
            "use_batch_norm": config.USE_BATCH_NORM,
        },
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Model qeyd edildi: {save_path}")

    report = {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_metrics": test_metrics,
        "history": history,
    }

    # Hesabat faylını saxlayırıq
    metrics_path = save_path.with_name("compatibility_metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return model, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compatibility modelinin öyrədilməsi")
    parser.add_argument("--data-path", type=str, default=None, help=".npz dataset faylının yolu")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=config.HIDDEN_DIMS)
    parser.add_argument("--dropout", type=float, default=config.DROPOUT)
    parser.add_argument("--mode", type=str, default=config.INPUT_REPRESENTATION,
                        choices=["symmetric_full", "diff_prod", "concat"])
    parser.add_argument("--save-path", type=str, default=str(config.DEFAULT_MODEL_PATH))
    parser.add_argument("--wandb-mode", type=str, default=config.WANDB_MODE,
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    # Əgər data-path verilməyibsə mövcud polyvore datadan və ya sintetika/demo cütlərdən istifadə edirik
    if args.data_path and Path(args.data_path).exists():
        print(f"Dataset faylı yüklənir: {args.data_path}")
        e1, e2, labels = load_pairs_from_npz(args.data_path)
    else:
        print("Data-path verilməyib və ya fayl yoxdur. Sintetik/Polyvore cütləri hazırlanır...")
        from ml.compatibility.dataset import create_pairs_from_outfits
        
        # Polyvore meta.json və ya mövcud şəkillərdən cütlər yığmaq
        meta_path = config.DATA_DIR / "images" / "meta.json"
        emb_path = config.MODULE_ROOT.parent / "retrieval" / "outputs" / "embeddings.npy"
        ids_path = config.MODULE_ROOT.parent / "retrieval" / "outputs" / "ids.json"

        if meta_path.exists() and emb_path.exists() and ids_path.exists():
            print("Mövcud Polyvore meta.json və FashionCLIP embedding-ləri tapıldı.")
            meta_json = json.loads(meta_path.read_text(encoding="utf-8"))
            embs = np.load(emb_path)
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            item_embeddings = {item_id: embs[idx] for idx, item_id in enumerate(ids)}

            outfit_to_items: dict[str, list[str]] = {}
            for item_id, item_info in meta_json.get("items", {}).items():
                outfit_id = item_info.get("outfit_id", "default")
                outfit_to_items.setdefault(outfit_id, []).append(item_id)

            e1, e2, labels = create_pairs_from_outfits(
                item_embeddings, outfit_to_items, negative_ratio=1.0, seed=args.seed
            )
        else:
            print("Nümunə dataset simulyasiya edilir (500 müsbət və 500 mənfi cüt)...")
            np.random.seed(args.seed)
            num_pairs = 500
            # Simulyasiya: müsbət cütlər yüksək korrelyasiyalı, mənfi cütlər ortoqonaldır
            pos_e1 = np.random.randn(num_pairs, config.EMB_DIM).astype(np.float32)
            pos_e1 /= np.linalg.norm(pos_e1, axis=1, keepdims=True)
            pos_e2 = pos_e1 + 0.3 * np.random.randn(num_pairs, config.EMB_DIM).astype(np.float32)
            pos_e2 /= np.linalg.norm(pos_e2, axis=1, keepdims=True)
            pos_labels = np.ones(num_pairs, dtype=np.float32)

            neg_e1 = np.random.randn(num_pairs, config.EMB_DIM).astype(np.float32)
            neg_e1 /= np.linalg.norm(neg_e1, axis=1, keepdims=True)
            neg_e2 = np.random.randn(num_pairs, config.EMB_DIM).astype(np.float32)
            neg_e2 /= np.linalg.norm(neg_e2, axis=1, keepdims=True)
            neg_labels = np.zeros(num_pairs, dtype=np.float32)

            e1 = np.vstack([pos_e1, neg_e1])
            e2 = np.vstack([pos_e2, neg_e2])
            labels = np.concatenate([pos_labels, neg_labels])

            perm = np.random.permutation(len(labels))
            e1, e2, labels = e1[perm], e2[perm], labels[perm]

    print(f"Toplam cüt sayı: {len(labels)} (Müsbət: {int(np.sum(labels == 1))}, Mənfi: {int(np.sum(labels == 0))})")

    train_ds, val_ds, test_ds = split_dataset(e1, e2, labels, seed=args.seed)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=args.batch_size
    )

    wandb_run = init_wandb(
        experiment_name=f"compat-mlp-{args.mode}",
        config_dict=vars(args),
        mode=args.wandb_mode,
    )

    train_compatibility_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        mode=args.mode,
        save_path=Path(args.save_path),
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
