import os
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)
import torch


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
  """Tính Expected Calibration Error (ECE)."""
  bin_boundaries = np.linspace(0, 1, n_bins + 1)
  ece = 0.0
  for i in range(n_bins):
    bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
    in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
    prop_in_bin = np.mean(in_bin)
    if prop_in_bin > 0:
      accuracy_in_bin = np.mean(y_true[in_bin])
      avg_confidence_in_bin = np.mean(y_pred[in_bin])
      ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
  return float(ece)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
  """Tính trọn bộ metrics dự đoán theo đúng danh mục yêu cầu."""
  y_pred_bin = (y_pred >= 0.5).astype(int)

  # Lớp 0 (Lớp "Sai")
  y_true_0 = 1 - y_true
  y_pred_0 = 1 - y_pred
  y_pred_bin_0 = 1 - y_pred_bin

  auc = (
      float(roc_auc_score(y_true, y_pred))
      if len(np.unique(y_true)) > 1
      else 0.0
  )
  acc = float(accuracy_score(y_true, y_pred_bin))
  rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
  brier = float(np.mean((y_pred - y_true) ** 2))
  ece = compute_ece(y_true, y_pred)

  # AUPRC & F1 cho lớp "Sai" (Lớp 0)
  auprc_class0 = float(average_precision_score(y_true_0, y_pred_0))
  f1_class0 = float(f1_score(y_true_0, y_pred_bin_0, zero_division=0))

  # Confusion Matrix
  cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
  tn, fp, fn, tp = cm.ravel()

  return {
      "auc": auc,
      "acc": acc,
      "rmse": rmse,
      "brier": brier,
      "ece": ece,
      "auprc_class0": auprc_class0,
      "f1_class0": f1_class0,
      "cm_tn": int(tn),
      "cm_fp": int(fp),
      "cm_fn": int(fn),
      "cm_tp": int(tp),
  }


def get_model_size_info(model: torch.nn.Module, ckpt_path: str) -> dict:
  """Đo số tham số huấn luyện được và dung lượng checkpoint (MB)."""
  trainable_params = sum(
      p.numel() for p in model.parameters() if p.requires_grad
  )
  total_params = sum(p.numel() for p in model.parameters())
  ckpt_size_mb = (
      os.path.getsize(ckpt_path) / (1024 * 1024)
      if os.path.exists(ckpt_path)
      else 0.0
  )
  return {
      "trainable_params": trainable_params,
      "total_params": total_params,
      "ckpt_size_mb": round(ckpt_size_mb, 2),
  }


def measure_inference_speed(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> dict:
  """Đo thời gian suy luận: Warmup 1 lần, đo 3 lần lấy Trung vị (Median), kèm GPU Peak Memory."""
  model.eval()

  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

  # 1. Warmup (Làm nóng GPU)
  with torch.no_grad():
    for batch in loader:
      b = {
          k: v.to(device)
          for k, v in batch.items()
          if isinstance(v, torch.Tensor)
      }
      _ = model(b)
      break
    if device.type == "cuda":
      torch.cuda.synchronize(device)

  # 2. Đo 3 lần thực tế
  latencies = []
  total_samples = 0

  with torch.no_grad():
    for _ in range(3):
      samples_count = 0
      t_start = time.perf_counter()

      for batch in loader:
        b = {
            k: v.to(device)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }
        _ = model(b)
        samples_count += b["responses"].shape[0]  # Số sequence/học sinh

      if device.type == "cuda":
        torch.cuda.synchronize(device)

      t_end = time.perf_counter()
      latencies.append(t_end - t_start)
      total_samples = samples_count

  median_latency = float(np.median(latencies))
  samples_per_sec = (
      total_samples / median_latency if median_latency > 0 else 0.0
  )

  # 3. Đo GPU Peak Memory khi suy luận
  eval_gpu_mem_mb = (
      torch.cuda.max_memory_allocated(device) / (1024 * 1024)
      if device.type == "cuda"
      else 0.0
  )

  return {
      "inference_latency_sec": round(median_latency, 4),
      "throughput_samples_per_sec": round(samples_per_sec, 2),
      "eval_peak_gpu_mem_mb": round(eval_gpu_mem_mb, 2),
  }