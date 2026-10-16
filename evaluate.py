import argparse
import csv
from datetime import datetime
import os
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.data import KTTrainDataset
from src.metrics_bench import (
    compute_all_metrics,
    get_model_size_info,
    measure_inference_speed,
)
from src.models import build_model


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True)
  return parser.parse_args()


def get_git_commit():
  try:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("utf-8")
        .strip()
    )
  except Exception:
    return "unknown"


def log_to_registry(registry_path: str, row_data: dict):
  file_exists = os.path.exists(registry_path)
  os.makedirs(os.path.dirname(registry_path), exist_ok=True)

  with open(registry_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
    if not file_exists:
      writer.writeheader()
    writer.writerow(row_data)


def main():
  args = parse_args()
  with open(args.config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  paths = config["paths"]
  train_cfg = config["training"]
  model_cfg = config["model"]

  # 1. Đo thời gian Tiền xử lý / Nạp dữ liệu CSV
  t0_data = time.perf_counter()
  if device.type == "cuda":
    torch.cuda.synchronize(device)

  # Load Dataset để kiểm tra thời gian build
  dummy_ds = KTTrainDataset(
      paths["train_csv"], fold_ids=[0], max_len=train_cfg["max_len"]
  )

  if device.type == "cuda":
    torch.cuda.synchronize(device)
  data_prep_time_sec = time.perf_counter() - t0_data

  n_folds = train_cfg["n_folds"]
  model_tag = model_cfg["name"].lower()
  registry_csv = os.path.join("results", "registry.csv")
  os.makedirs(paths["raw_pred_dir"], exist_ok=True)

  print(f"{'='*80}\n  BENCHMARK & EVALUATION RUNNER")
  print(f"{'='*80}")

  for fold_idx in range(n_folds):
    ckpt_path = os.path.join(
        paths["save_dir"], f"best_{model_tag}_fold{fold_idx}.pt"
    )
    if not os.path.exists(ckpt_path):
      print(
          f"❌ Không tìm thấy Checkpoint tại: {ckpt_path}. Bỏ qua Fold"
          f" {fold_idx}."
      )
      continue

    checkpoint = torch.load(ckpt_path, map_location=device)

    valid_ds = KTTrainDataset(
        paths["train_csv"], fold_ids=[fold_idx], max_len=train_cfg["max_len"]
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
    )

    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state"])

    # 2. Đo Kích thước mô hình
    size_info = get_model_size_info(model, ckpt_path)

    # 3. Đo Tốc độ suy luận & Bộ nhớ GPU
    speed_info = measure_inference_speed(model, valid_loader, device)

    # 4. Thu thập toàn bộ Dự đoán thô (Raw Predictions & Labels)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
      for batch in valid_loader:
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

        if valid_targets.numel() > 0:
          all_preds.extend(valid_preds.cpu().tolist())
          all_labels.extend(valid_targets.cpu().tolist())

    y_pred = np.array(all_preds, dtype=np.float64)
    y_true = np.array(all_labels, dtype=np.float64)

    # 5. Lưu Ma trận dự đoán thô + Nhãn thực tế vào file .npz
    raw_pred_file = os.path.join(
        paths["raw_pred_dir"], f"preds_{model_tag}_fold{fold_idx}.npz"
    )
    np.savez_compressed(raw_pred_file, y_true=y_true, y_pred=y_pred)

    # 6. Tính toán Trọn bộ Metrics
    metrics = compute_all_metrics(y_true, y_pred)

    # 7. Tổng hợp ghi vào Registry CSV
    run_id = f"{model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_f{fold_idx}"
    row_data = {
        # Đã nâng cấp đầy đủ các trường môi trường & benchmark
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "author": config["experiment"].get("author", "KietVA_AnhNB"),
        "git_commit": get_git_commit(),
        "config_path": args.config,
        "model": model_tag,
        "backbone": model_cfg.get("backbone", "N/A"),
        "head": model_cfg.get("head", "N/A"),
        "fold": fold_idx,
        "seed": train_cfg["seed"],
        "stopped_epoch": checkpoint.get("epoch", 0),
        "time_per_epoch": checkpoint.get("time_per_epoch", 0.0),
        "data_prep_time_sec": round(data_prep_time_sec, 3),
        # Môi trường cứng
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
        "batch_size": train_cfg["batch_size"],
        "amp_enabled": train_cfg.get("use_amp", False),
        "num_workers": train_cfg.get("num_workers", 0),
        # Kích thước mô hình
        "trainable_params": size_info["trainable_params"],
        "ckpt_size_mb": size_info["ckpt_size_mb"],
        # Thời gian & Bộ nhớ suy luận
        "inference_latency_sec": speed_info["inference_latency_sec"],
        "throughput_samples_per_sec": speed_info["throughput_samples_per_sec"],
        "eval_peak_gpu_mem_mb": speed_info["eval_peak_gpu_mem_mb"],
        # Metrics dự đoán chính
        "val_auc": metrics["auc"],
        "val_acc": metrics["acc"],
        "val_rmse": metrics["rmse"],
        "val_brier": metrics["brier"],
        "val_ece": metrics["ece"],
        "val_auprc_class0": metrics["auprc_class0"],
        "val_f1_class0": metrics["f1_class0"],
        # Ma trận nhầm lẫn
        "cm_tn": metrics["cm_tn"],
        "cm_fp": metrics["cm_fp"],
        "cm_fn": metrics["cm_fn"],
        "cm_tp": metrics["cm_tp"],
        "raw_pred_path": raw_pred_file,
    }

    log_to_registry(registry_csv, row_data)

    print(f"✅ Fold {fold_idx} Benchmark Done:")
    print(
        f"   - AUC: {metrics['auc']:.4f} | ECE: {metrics['ece']:.4f} | Brier:"
        f" {metrics['brier']:.4f} | AUPRC (Sai): {metrics['auprc_class0']:.4f}"
    )
    print(
        f"   - CM (TN/FP/FN/TP):"
        f" [{metrics['cm_tn']}/{metrics['cm_fp']}/{metrics['cm_fn']}/{metrics['cm_tp']}]"
    )
    print(
        f"   - Size: {size_info['trainable_params']:,} params"
        f" ({size_info['ckpt_size_mb']} MB) | Speed:"
        f" {speed_info['throughput_samples_per_sec']} seq/s"
    )
    print(f"   - Saved predictions to: {raw_pred_file}\n")


if __name__ == "__main__":
  main()