import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_report_table(
    registry_csv="results/registry.csv", source_tag="Tự chạy"
):
  """Đọc sổ Registry và in bảng tổng hợp Mean +- Std chuẩn báo cáo."""
  if not os.path.exists(registry_csv):
    print("❌ Không tìm thấy file registry.csv!")
    return

  df = pd.read_csv(registry_csv)

  print("\n" + "=" * 80)
  print("📊 BẢNG TỔNG HỢP KẾT QUẢ NGHIÊN CỨU (MEAN ± STD CỦA CÁC FOLDS)")
  print("=" * 80)

  grouped = df.groupby("model")
  report_rows = []

  for model_name, group in grouped:
    mean_auc, std_auc = group["val_auc"].mean(), group["val_auc"].std()
    mean_acc, std_acc = group["val_acc"].mean(), group["val_acc"].std()
    mean_rmse, std_rmse = group["test_rmse"].mean(), group["test_rmse"].std()
    n_folds = len(group)

    report_rows.append({
        "Mô hình": model_name.upper(),
        "Số Fold": n_folds,
        "Val AUC": f"{mean_auc:.4f} ± {std_auc:.4f}",
        "Val ACC": f"{mean_acc:.4f} ± {std_acc:.4f}",
        "Test RMSE": f"{mean_rmse:.4f} ± {std_rmse:.4f}",
        "Nguồn": source_tag,
    })

  df_report = pd.DataFrame(report_rows)
  print(df_report.to_string(index=False))
  print("-" * 80)
  print(
      "📝 Chú thích: Giá trị thể hiện Mean ± Std qua các Fold. Dữ liệu chạy thực"
      " nghiệm trên XES3G5M."
  )
  print("=" * 80 + "\n")


def plot_learning_curve_with_error_bar(
    history_json="saved_models/history_akt_val_auc.json",
    save_path="results/learning_curve.png",
):
  """Vẽ biểu đồ Learning Curve AUC qua từng Epoch kèm dải đệm Error Bar (Std)."""
  if not os.path.exists(history_json):
    print(f"❌ Không tìm thấy file {history_json}")
    return

  with open(history_json, "r") as f:
    history = json.load(f)  # List các list AUC từng fold

  # Chuyển thành numpy array (Pad bằng nan nếu các fold dừng ở epoch khác nhau)
  max_epochs = max(len(h) for h in history)
  arr = np.full((len(history), max_epochs), np.nan)
  for i, h in enumerate(history):
    arr[i, : len(h)] = h

  mean_auc = np.nanmean(arr, axis=0)
  std_auc = np.nanstd(arr, axis=0)
  epochs = np.arange(1, max_epochs + 1)

  plt.figure(figsize=(8, 5))
  plt.plot(
      epochs,
      mean_auc,
      label="Validation AUC (Mean)",
      color="#1f77b4",
      linewidth=2,
  )
  plt.fill_between(
      epochs,
      mean_auc - std_auc,
      mean_auc + std_auc,
      color="#1f77b4",
      alpha=0.25,
      label="±1 Std (Độ lệch giữa 5 Folds)",
  )

  plt.xlabel("Epoch", fontsize=12)
  plt.ylabel("AUC Score", fontsize=12)
  plt.title("Biểu đồ Learning Curve qua 5 Folds (Có Error Bar)", fontsize=14)
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend(fontsize=11)

  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.close()
  print(f"✅ Đã xuất biểu đồ Learning Curve tại: {save_path}")


def plot_mastery_trajectory_example(save_path="results/mastery_trajectory.png"):
  """Vẽ biểu đồ ví dụ định tính về đường cong năng lực (Mastery Trajectory) của học sinh."""
  t_steps = np.arange(1, 11)
  # Giả lập 1 chuỗi học sinh: ban đầu yếu, trả lời đúng liên tục -> mastery tăng
  responses = np.array([0, 1, 0, 1, 1, 1, 1, 1, 0, 1])
  p_mastery = np.array(
      [0.35, 0.42, 0.38, 0.55, 0.68, 0.81, 0.89, 0.94, 0.87, 0.93]
  )

  plt.figure(figsize=(9, 4.5))
  plt.plot(
      t_steps,
      p_mastery,
      color="#2ca02c",
      linewidth=2,
      linestyle="-",
      label="P(Khả năng làm đúng)",
  )

  # Đánh dấu điểm Đúng (Xanh) và Sai (Đỏ)
  for t, r, p in zip(t_steps, responses, p_mastery):
    color = "green" if r == 1 else "red"
    marker = "o" if r == 1 else "x"
    label_text = "Đúng" if r == 1 else "Sai"
    plt.scatter(t, p, color=color, s=80, zorder=5)
    plt.annotate(
        f"y={r}",
        (t, p),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
    )

  plt.xlabel("Bước tương tác (Chuỗi bài tập t)", fontsize=11)
  plt.ylabel("Xác suất dự đoán P(y_t = 1)", fontsize=11)
  plt.title(
      "Phân tích Định tính: Quỹ đạo Nắm bắt Trí thức (Mastery Trajectory) của"
      " Học sinh",
      fontsize=13,
  )
  plt.ylim(0.0, 1.05)
  plt.grid(True, linestyle="--", alpha=0.5)

  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.close()
  print(f"✅ Đã xuất biểu đồ Mastery Trajectory tại: {save_path}")


if __name__ == "__main__":
  generate_report_table()
  plot_learning_curve_with_error_bar()
  plot_mastery_trajectory_example()