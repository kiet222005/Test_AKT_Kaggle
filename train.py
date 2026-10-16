import argparse
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
from src.registry_logger import ResultsRegistry


def parse_args():
  parser = argparse.ArgumentParser(description="Knowledge Tracing Training")
  parser.add_argument(
      "--config",
      type=str,
      required=True,
      help="Đường dẫn file .yaml (vd: configs/akt_xes3g5m.yaml)",
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


def train_one_epoch(model, loader, optimizer, device, log_every=200):
  model.train()
  total_loss, total_samples = 0.0, 0
  t_start = time.time()

  for batch_idx, batch in enumerate(loader):
    b = {
        k: v.to(device)
        for k, v in batch.items()
        if isinstance(v, torch.Tensor)
    }
    optimizer.zero_grad()

    # Model tự dùng b['is_repeat'] để đính kèm Group-Causal Mask
    preds, reg_loss = model(b)

    shifted_preds = preds[:, 1:]
    shifted_targets = b["responses"][:, 1:]
    raw_mask = b["select_masks"][:, 1:]

    # Lọc mask hợp lệ
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

    if log_every > 0 and (batch_idx + 1) % log_every == 0:
      print(
          f"        Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}"
      )

  return total_loss / max(total_samples, 1), time.time() - t_start


@torch.no_grad()
def eval_one_epoch(model, loader, device):
  model.eval()
  total_loss, total_samples = 0.0, 0
  all_preds, all_labels = [], []
  t_start = time.time()

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
  return {
      "loss": total_loss / max(total_samples, 1),
      "auc": auc,
      "acc": acc,
      "time": time.time() - t_start,
  }


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
  os.makedirs(paths["raw_pred_dir"], exist_ok=True)

  registry = ResultsRegistry()
  n_folds = train_cfg["n_folds"]
  fold_results = []

  print(f"{'#'*70}\n  TRAINING EXPERIMENT: {config['experiment']['name']}")
  print(f"  MODEL: {model_cfg['name'].upper()} | DEVICE: {device}")
  print(f"{'#'*70}")

  for fold_idx in range(n_folds):
    fold_start = time.time()
    train_folds = [f for f in range(n_folds) if f != fold_idx]

    print(
        f"\n=== FOLD {fold_idx+1}/{n_folds} (Train: {train_folds} | Val:"
        f" [{fold_idx}]) ==="
    )

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
    best_metrics = None
    patience_counter = 0

    model_tag = model_cfg["name"].lower()
    save_path = os.path.join(
        paths["save_dir"], f"best_{model_tag}_fold{fold_idx}.pt"
    )

    for epoch in range(1, train_cfg["epochs"] + 1):
      train_loss, train_time = train_one_epoch(
          model, train_loader, optimizer, device, train_cfg.get("log_every", 200)
      )
      val_m = eval_one_epoch(model, valid_loader, device)

      print(
          f"  Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss:"
          f" {val_m['loss']:.4f} | Val AUC: {val_m['auc']:.4f} | Val ACC:"
          f" {val_m['acc']:.4f}"
      )

      if val_m["auc"] > best_auc:
        best_auc = val_m["auc"]
        best_metrics = dict(val_m)
        best_metrics["best_epoch"] = epoch

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

    fold_time = time.time() - fold_start
    stopped_epoch = best_metrics["best_epoch"]

    # Ghi sổ Registry
    run_config = dict(config)
    run_config["config_path"] = args.config
    run_config["training"]["fold"] = fold_idx
    run_config["training"]["gpu"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )

    registry.log_run(
        config=run_config,
        metrics={
            "val_auc": best_metrics["auc"],
            "val_acc": best_metrics["acc"],
            "test_auc": 0.0,
            "test_acc": 0.0,
            "test_rmse": 0.0,
        },
        stopped_epoch=stopped_epoch,
        time_per_epoch=fold_time / max(stopped_epoch, 1),
        raw_pred_path="N/A",
        author=config["experiment"].get("author", "KietVA_AnhNB"),
    )

    fold_results.append(best_metrics)
    del model, optimizer
    if device.type == "cuda":
      torch.cuda.empty_cache()

  val_aucs = [f["auc"] for f in fold_results]
  print(
      f"\n🏆 5-Fold Mean Val AUC:"
      f" {np.mean(val_aucs):.4f} ± {np.std(val_aucs, ddof=1):.4f}"
  )


if __name__ == "__main__":
  main()