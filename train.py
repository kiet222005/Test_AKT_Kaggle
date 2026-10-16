import argparse
import json
import os
import random
import time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from src.data import KTTrainDataset
from src.models import build_model


def parse_args():
  parser = argparse.ArgumentParser(description="Knowledge Tracing Training")
  parser.add_argument(
      "--config", type=str, required=True, help="Đường dẫn file .yaml"
  )
  return parser.parse_args()


def set_seed(seed: int):
  random.seed(seed)
  np.random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, device):
  model.train()
  total_loss, total_samples = 0.0, 0
  t_start = time.time()

  for batch in loader:
    b = {
        k: v.to(device)
        for k, v in batch.items()
        if isinstance(v, torch.Tensor)
    }
    optimizer.zero_grad()

    preds, reg_loss = model(b)

    shifted_preds = preds[:, 1:]
    shifted_targets = b["responses"][:, 1:]
    raw_mask = b["select_masks"][:, 1:]

    valid_mask = (raw_mask == 1) & (shifted_targets >= 0)
    valid_preds = shifted_preds[valid_mask]
    valid_targets = shifted_targets[valid_mask].float()

    if valid_targets.numel() == 0:
      continue

    bce_loss = F.binary_cross_entropy(valid_preds, valid_targets)
    if isinstance(reg_loss, torch.Tensor) and reg_loss.dim() > 0:
      reg_loss = reg_loss.mean()

    loss = bce_loss + reg_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    n_items = valid_targets.numel()
    total_loss += loss.item() * n_items
    total_samples += n_items

  return total_loss / max(total_samples, 1), time.time() - t_start


@torch.no_grad()
def eval_one_epoch(model, loader, device):
  model.eval()
  total_loss, total_samples = 0.0, 0
  all_preds, all_labels = [], []

  for batch in loader:
    b = {
        k: v.to(device)
        for k, v in batch.items()
        if isinstance(v, torch.Tensor)
    }
    preds, _ = model(b)

    shifted_preds = preds[:, 1:]
    shifted_targets = b["responses"][:, 1:]
    raw_mask = b["select_masks"][:, 1:]

    valid_mask = (raw_mask == 1) & (shifted_targets >= 0)
    valid_preds = shifted_preds[valid_mask]
    valid_targets = shifted_targets[valid_mask].float()

    if valid_targets.numel() == 0:
      continue

    bce_loss = F.binary_cross_entropy(valid_preds, valid_targets)
    n_items = valid_targets.numel()
    total_loss += bce_loss.item() * n_items
    total_samples += n_items

    all_preds.extend(valid_preds.cpu().tolist())
    all_labels.extend(valid_targets.cpu().tolist())

  preds_arr = np.array(all_preds, dtype=np.float64)
  labels_arr = np.array(all_labels, dtype=np.float64)

  try:
    auc = float(roc_auc_score(labels_arr, preds_arr))
  except ValueError:
    auc = 0.0

  acc = float(accuracy_score(labels_arr, (preds_arr >= 0.5).astype(int)))
  return {"loss": total_loss / max(total_samples, 1), "auc": auc, "acc": acc}


def main():
  args = parse_args()
  with open(args.config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

  set_seed(config["training"]["seed"])
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  paths = config["paths"]
  train_cfg = config["training"]
  model_cfg = config["model"]

  os.makedirs(paths["save_dir"], exist_ok=True)

  n_folds = train_cfg["n_folds"]
  model_tag = model_cfg["name"].lower()

  # Lưu vết AUC qua các Epoch của tất cả Folds để vẽ Learning Curve
  all_folds_history = []

  print(f"{'#'*70}\n  TRAINING EXPERIMENT: {config['experiment']['name']}")
  print(f"  MODEL: {model_cfg['name'].upper()} | DEVICE: {device}")
  print(f"{'#'*70}")

  for fold_idx in range(n_folds):
    train_folds = [f for f in range(n_folds) if f != fold_idx]
    print(f"\n=== FOLD {fold_idx+1}/{n_folds} ===")

    train_ds = KTTrainDataset(
        paths["train_csv"], fold_ids=train_folds, max_len=train_cfg["max_len"]
    )
    valid_ds = KTTrainDataset(
        paths["train_csv"], fold_ids=[fold_idx], max_len=train_cfg["max_len"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(config, device)
    optimizer = optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))

    best_auc = 0.0
    patience_counter = 0
    fold_epoch_aucs = []

    save_path = os.path.join(
        paths["save_dir"], f"best_{model_tag}_fold{fold_idx}.pt"
    )

    for epoch in range(1, train_cfg["epochs"] + 1):
      train_loss, t_epoch = train_one_epoch(
          model, train_loader, optimizer, device
      )
      val_m = eval_one_epoch(model, valid_loader, device)

      fold_epoch_aucs.append(val_m["auc"])

      print(
          f"  Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss:"
          f" {val_m['loss']:.4f} | Val AUC: {val_m['auc']:.4f} | Val ACC:"
          f" {val_m['acc']:.4f}"
      )

      if val_m["auc"] > best_auc:
        best_auc = val_m["auc"]
        checkpoint = {
            "epoch": epoch,
            "fold": fold_idx,
            "best_auc": best_auc,
            "metrics": dict(val_m),
            "model_state": model.state_dict(),
            "config": config,
        }
        torch.save(checkpoint, save_path)
        patience_counter = 0
      else:
        patience_counter += 1
        if patience_counter >= train_cfg["patience"]:
          print(f"  --> Early stopping tại epoch {epoch}")
          break

    all_folds_history.append(fold_epoch_aucs)

  # Lưu file history JSON để dùng cho script vẽ biểu đồ
  history_file = os.path.join(
      paths["save_dir"], f"history_{model_tag}_val_auc.json"
  )
  with open(history_file, "w") as f:
    json.dump(all_folds_history, f)

  print(f"\n✅ Đã hoàn tất Training! Checkpoint & History lưu tại: {paths['save_dir']}")


if __name__ == "__main__":
  main()