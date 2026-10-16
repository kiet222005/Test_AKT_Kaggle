import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = (
    BASE_DIR
    / "data"
    / "raw"
    / "XES3G5M"
    / "XES3G5M"
    / "kc_level"
    / "test_question_window_sequences.csv"
)
df = pd.read_csv(data_path)

n_scored = 0
n_leak = 0
n_scored_repeat = 0
n_pos0 = 0

for sm_s, qi_s, rep_s in zip(df["selectmasks"], df["qidxs"], df["is_repeat"]):
    # Xử lý an toàn trường hợp NaN hoặc khoảng trắng
    if pd.isna(sm_s) or pd.isna(qi_s) or pd.isna(rep_s):
        continue

    sm = [int(x.strip()) for x in str(sm_s).split(",") if x.strip() != ""]
    qi = [int(x.strip()) for x in str(qi_s).split(",") if x.strip() != ""]
    rep = [int(x.strip()) for x in str(rep_s).split(",") if x.strip() != ""]

    if not sm:
        continue

    # 1. Kiểm tra vị trí đầu tiên
    if sm[0] == 1:
        n_pos0 += 1

    # 2. Kiểm tra leak và repeat trên các vị trí được chấm (selectmask == 1)
    seen_qis = set()
    for t, m in enumerate(sm):
        current_qi = qi[t]

        if m == 1:
            n_scored += 1

            # Rò rỉ: câu hỏi tại thời điểm t đã xuất hiện trong lịch sử qi[:t]
            if current_qi in seen_qis:
                n_leak += 1

            # Vị trí chấm là câu hỏi lặp lại
            if rep[t] == 1:
                n_scored_repeat += 1

        # Cập nhật lịch sử các câu hỏi đã thấy tính đến thời điểm t
        seen_qis.add(current_qi)

# In kết quả kiểm tra
print(f"Tổng số vị trí được chấm (n_scored): {n_scored}")
print(f"Số vị trí bị rò rỉ dữ liệu (n_leak): {n_leak}")
print(f"Số vị trí chấm là câu lặp (n_scored_repeat): {n_scored_repeat}")
print(f"Số vị trí t=0 bị chấm lỗi (n_pos0): {n_pos0}")

# Đánh giá theo kỳ vọng
print("\n--- KẾT QUẢ KIỂM TRA DỮ LIỆU ---")
assert n_pos0 == 0, f"LỖI: Có {n_pos0} dòng bị chấm điểm tại vị trí t=0!"
assert n_leak == 0, f"LỖI: Rò rỉ dữ liệu test! Có {n_leak} vị trí chấm điểm xuất hiện câu hỏi đã có trong lịch sử."
assert n_scored_repeat > 0, "CẢNH BÁO: Không tìm thấy vị trí M1 nào (n_scored_repeat = 0)!"

print("PASSED: Dữ liệu test hợp lệ, không bị rò rỉ!")