import torch
import yaml
from src.data import KTTrainDataset
from src.models import build_model
from torch.utils.data import DataLoader


@torch.no_grad()
def test_no_same_question_leak(config_path="configs/akt_xes3g5m.yaml"):
  # 1. Đọc config & khởi tạo mô hình ngẫu nhiên (chưa cần train)
  with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = build_model(config, device)
  model.eval()  # Tắt Dropout để kết quả cố định

  # 2. Lấy 1 batch dữ liệu mẫu
  dataset = KTTrainDataset(
      csv_file=config["paths"]["train_csv"],
      fold_ids=[0],
      max_len=config["training"]["max_len"],
  )
  loader = DataLoader(dataset, batch_size=16, shuffle=False)
  batch = next(iter(loader))

  b_device = {
      k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
  }

  # 3. Lần Forward thứ nhất (Gốc)
  preds_base, _ = model(b_device)

  # 4. Lấy danh sách các vị trí t là KC thứ 2+ của cùng một câu hỏi (is_repeat == 1)
  repeat_indices = (b_device["is_repeat"] == 1).nonzero()

  if len(repeat_indices) == 0:
    print("⚠️ Cảnh báo: Batch mẫu không chứa vị trí is_repeat == 1 nào.")
    return

  print(
      f"🔍 Đang kiểm tra rò rỉ nhãn trên {len(repeat_indices)} vị trí lặp"
      " KC..."
  )

  # 5. Can thiệp: Đảo đáp án tại vị trí t-1 và kiểm tra dự đoán tại t
  leak_count = 0
  for sample_idx, t in repeat_indices:
    sample_idx = sample_idx.item()
    t = t.item()

    # Tạo batch can thiệp
    b_mod = {k: v.clone() for k, v in b_device.items()}

    # Đảo nhãn của KC ngay trước đó trong cùng câu hỏi (t-1)
    b_mod["responses"][sample_idx, t - 1] = (
        1 - b_mod["responses"][sample_idx, t - 1]
    )

    # Lần Forward thứ hai
    preds_mod, _ = model(b_mod)

    # Đo độ chênh lệch dự đoán tại vị trí t
    p_orig = preds_base[sample_idx, t].item()
    p_mod = preds_mod[sample_idx, t].item()
    diff = abs(p_orig - p_mod)

    if diff >= 1e-5:
      print(
          f"❌ FAIL: Rò rỉ tại Mẫu {sample_idx}, Vị trí t={t} | Diff:"
          f" {diff:.6f}"
      )
      leak_count += 1

  if leak_count == 0:
    print(
        "✅ PASSED: Mô hình chặn thành công 100% rò rỉ nhãn trong cùng câu"
        " hỏi!"
    )
  else:
    print(f"❌ FAILED: Có {leak_count} vị trí bị rò rỉ dữ liệu.")


if __name__ == "__main__":
  test_no_same_question_leak()