import os
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.data import KTTrainDataset
from src.models import build_model
from src.masks import question_groups


@torch.no_grad()
def test_same_question_leak(config_path: str = "configs/akt_xes3g5m.yaml", max_display: int = 5):
    """
    Unit Test: Kiểm tra rò rỉ nhãn giữa các KC trong cùng một câu hỏi.
    Có xuất file log CSV và hiển thị trực quan các vị trí can thiệp nhãn.
    """
    print("=" * 85)
    print("🧪 UNIT TEST: KIỂM TRA MẶT NẠ GROUP-CAUSAL MASK (CHỐNG RÒ RỈ NHÃN)")
    print("=" * 85)

    # 1. Kiểm tra & Nạp cấu hình từ file YAML
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file cấu hình tại: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Mô hình: {config['model']['name'].upper()} | Device: {device}")

    # 2. Khởi tạo mô hình ở chế độ Eval
    model = build_model(config, device)
    model.eval()

    # 3. Nạp 1 Batch dữ liệu mẫu
    dataset = KTTrainDataset(
        csv_file=config["paths"]["train_csv"],
        fold_ids=[0],
        max_len=config["training"]["max_len"],
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch_goc = next(iter(loader))

    batch_device = {
        k: v.to(device) for k, v in batch_goc.items() if isinstance(v, torch.Tensor)
    }

    # Tính ID nhóm câu hỏi
    q_groups = question_groups(batch_device["is_repeat"])

    # 4. Chạy Lần 1: Dự đoán trên chuỗi gốc
    preds_base, _ = model(batch_device)

    # 5. Dò các vị trí (b, t) thuộc câu hỏi đa KC (is_repeat == 1)
    repeat_indices = (batch_device["is_repeat"] == 1).nonzero()

    if len(repeat_indices) == 0:
        print("⚠️ Cảnh báo: Batch mẫu không chứa câu hỏi đa KC nào.")
        return

    print(f"🔍 Tìm thấy {len(repeat_indices)} vị trí KC lặp thuộc cùng câu hỏi.")
    print("⏳ Đang nhân bản chuỗi, đảo nhãn t-1 và theo dõi biến động xác suất...\n")

    logs = []
    leak_count = 0

    # 6. Lặp qua từng vị trí câu hỏi đa KC
    for idx, (sample_idx, t) in enumerate(repeat_indices):
        sample_idx = sample_idx.item()
        t = t.item()

        # Step 6a: Lưu vết - Tạo bản sao độc lập của batch
        batch_moi = {k: v.clone() for k, v in batch_device.items()}

        # Lấy thông tin KC & Nhóm câu hỏi
        q_group_id = q_groups[sample_idx, t].item()
        concept_prev = batch_device["concepts"][sample_idx, t - 1].item()
        concept_curr = batch_device["concepts"][sample_idx, t].item()

        orig_resp_prev = batch_moi["responses"][sample_idx, t - 1].item()
        
        # Đảm bảo chỉ đảo giữa 0 và 1 binary
        flipped_resp_prev = 0 if orig_resp_prev == 1 else 1

        # Step 6b: Đảo đáp án tại vị trí t-1
        batch_moi["responses"][sample_idx, t - 1] = flipped_resp_prev

        # Step 6c: Chạy model Lần 2 trên chuỗi đã đảo nhãn
        preds_mod, _ = model(batch_moi)

        # Step 6d: Lấy xác suất dự đoán tại vị trí t
        p_orig = preds_base[sample_idx, t].item()
        p_mod = preds_mod[sample_idx, t].item()
        diff = abs(p_orig - p_mod)
        is_leak = diff >= 1e-5

        if is_leak:
            leak_count += 1

        # Lưu thông tin chi tiết vào danh sách log
        logs.append({
            "sample_idx": sample_idx,
            "q_group": q_group_id,
            "target_t": t,
            "flipped_t": t - 1,
            "concept_flipped": concept_prev,
            "concept_target": concept_curr,
            "resp_orig_prev": orig_resp_prev,
            "resp_flipped_prev": flipped_resp_prev,
            "pred_before": p_orig,
            "pred_after": p_mod,
            "diff_delta": diff,
            "status": "❌ LEAK" if is_leak else "✅ PASS"
        })

    # Convert danh sách log thành DataFrame
    df_results = pd.DataFrame(logs)

    # 7. HIỂN THỊ TRỰC QUAN MỘT SỐ CA MẪU TRÊN TERMINAL
    print("=" * 85)
    print(f"📌 CHI TIẾT SƠ ĐỒ ĐẢO NHÃN TRÊN {min(max_display, len(logs))} VỊ TRÍ MẪU ĐẦU TIÊN:")
    print("=" * 85)

    for i in range(min(max_display, len(logs))):
        row = df_results.iloc[i]
        print(f"👉 Ca thử nghiệm #{i+1} [Mẫu dataset #{int(row['sample_idx'])}, Nhóm Q_{int(row['q_group'])}]:")
        print(f"   ├─ [Vị trí t-1 = {int(row['flipped_t'])}] KC_{int(row['concept_flipped'])} (ĐẢO NHÃN): "
              f"Đáp án gốc = {int(row['resp_orig_prev'])} ➔ Đáp án mới = {int(row['resp_flipped_prev'])}")
        print(f"   ├─ [Vị trí t   = {int(row['target_t'])}] KC_{int(row['concept_target'])} (QUAN SÁT DỰ ĐOÁN):")
        print(f"   │    • Trạng thái gốc   (Pred Before): {row['pred_before']:.6f}")
        print(f"   │    • Trạng thái mới   (Pred After ): {row['pred_after']:.6f}")
        print(f"   │    • Chênh lệch      (Delta Δ   ): {row['diff_delta']:.8f}")
        print(f"   └─ Kết quả kiểm định: {row['status']}\n")

    # 8. XUẤT FILE LOG CSV KẾT QUẢ DỰ ĐOÁN
    os.makedirs("results", exist_ok=True)
    csv_out_path = "results/test_dao_kc_results.csv"
    df_results.to_csv(csv_out_path, index=False, encoding="utf-8-sig")

    # 9. IN BẢNG TỔNG HỢP VÀ THỐNG KÊ
    print("=" * 85)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ KIỂM THỬ (XUẤT RA CSV):")
    print("=" * 85)
    display_cols = ["sample_idx", "target_t", "concept_target", "resp_orig_prev", "pred_before", "pred_after", "diff_delta", "status"]
    print(df_results[display_cols].head(10).to_string(index=False))
    
    print("-" * 85)
    if leak_count == 0:
        print(f"✅ PASSED 100%: Tất cả {len(df_results)}/{len(df_results)} vị trí đa KC đều KHÔNG bị thay đổi dự đoán!")
        print(f"🎉 Group-Causal Mask hoạt động hoàn hảo. Kết quả dự đoán chi tiết đã lưu tại: {csv_out_path}")
    else:
        print(f"❌ FAILED: Có {leak_count}/{len(df_results)} vị trí bị rò rỉ nhãn!")
    print("=" * 85)


if __name__ == "__main__":
    test_same_question_leak()