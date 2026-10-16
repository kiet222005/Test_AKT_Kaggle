import argparse
import os
from pathlib import Path
import sys
import pandas as pd

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kiểm tra tính duy nhất của cột qidxs trong tập dữ liệu KT."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Đường dẫn file CSV cần kiểm tra (mặc định ưu tiên file raw, fallback sang sample).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Chỉ kiểm tra trên file sample_200 để chạy nhanh.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Số dòng tối đa cần đọc (mặc định đọc hết).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=2000,
        help="Số dòng mỗi chunk khi đọc file lớn để tối ưu RAM (mặc định 2000).",
    )
    return parser.parse_args()


def resolve_data_path(custom_path=None, use_sample=False):
    base_dir = Path(__file__).resolve().parent.parent

    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Không tìm thấy file chỉ định: {custom_path}")

    sample_path = (
        base_dir
        / "data"
        / "xes3g5m_samples_200"
        / "kc_level"
        / "test_question_window_sequences_sample_200.csv"
    )

    raw_path = (
        base_dir
        / "data"
        / "raw"
        / "XES3G5M"
        / "XES3G5M"
        / "kc_level"
        / "test_question_window_sequences.csv"
    )

    if use_sample:
        if sample_path.exists():
            return sample_path
        raise FileNotFoundError(f"Không tìm thấy file sample: {sample_path}")

    if raw_path.exists():
        return raw_path
    elif sample_path.exists():
        print(f"[Thông báo] Không thấy file raw ({raw_path}), chuyển sang dùng sample ({sample_path}).")
        return sample_path
    else:
        raise FileNotFoundError(
            f"Không tìm thấy cả file raw ({raw_path}) và file sample ({sample_path})!"
        )


def parse_int_sequence(seq_str):
    if pd.isna(seq_str):
        return []
    items = str(seq_str).split(",")
    return [int(x.strip()) for x in items if x.strip() != ""]


def check_qidxs_uniqueness(data_path, max_rows=None, chunksize=2000):
    print(f"=== BẮT ĐẦU KIỂM TRA TÍNH DUY NHẤT CỦA CỘT qidxs ===")
    print(f"Đường dẫn file: {data_path}")
    file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
    print(f"Dung lượng file: {file_size_mb:.2f} MB")
    if max_rows:
        print(f"Giới hạn số dòng kiểm tra: {max_rows:,}")

    # Cấu trúc lưu trữ thông tin của từng qidx
    # qidx -> {"uid": uid, "question": question, "timestamp": timestamp}
    qidx_info = {}

    # Lưu trữ các trường hợp vi phạm để báo cáo chi tiết
    violations_uid = {}        # qidx -> set of uids
    violations_question = {}   # qidx -> set of questions
    violations_timestamp = {}  # qidx -> set of timestamps

    # Kiểm tra chiều ngược lại:
    # 1. Multi-KC: cùng (uid, question, timestamp) có luôn cùng qidx không?
    event_to_qidx = {}         # (uid, question, timestamp) -> set of qidxs

    # 2. Repeated question: cùng (uid, question) nhưng khác timestamp -> qidx có khác nhau không?
    # (uid, question) -> dict(timestamp -> set of qidxs)
    repeat_interactions = {}

    total_rows = 0
    total_tokens = 0
    total_valid_tokens = 0

    usecols = ["uid", "questions", "timestamps", "qidxs", "selectmasks"]

    reader = pd.read_csv(data_path, usecols=usecols, chunksize=chunksize)

    # Thống kê cho các vị trí được chấm điểm (selectmask == 1)
    scored_qidx_counts = {}   # qidx -> số lần được chấm (selectmask == 1)
    scored_qidx_info = {}     # qidx -> (uid, q_id, ts)

    for chunk_idx, chunk in enumerate(reader):
        for _, row in chunk.iterrows():
            total_rows += 1
            if max_rows and total_rows > max_rows:
                break

            uid = row["uid"]
            questions = parse_int_sequence(row["questions"])
            timestamps = parse_int_sequence(row["timestamps"])
            qidxs = parse_int_sequence(row["qidxs"])
            selectmasks = parse_int_sequence(row["selectmasks"])

            seq_len = len(qidxs)
            total_tokens += seq_len

            # Kiểm tra độ dài đồng bộ giữa các cột trong dòng
            min_len = min(len(questions), len(timestamps), len(qidxs), len(selectmasks))

            for t in range(min_len):
                qi = qidxs[t]
                # Bỏ qua giá trị padding (-1)
                if qi == -1:
                    continue

                q_id = questions[t]
                ts = timestamps[t]
                sm = selectmasks[t]
                total_valid_tokens += 1

                # Theo dõi riêng các vị trí được chấm điểm (selectmask == 1)
                if sm == 1:
                    scored_qidx_counts[qi] = scored_qidx_counts.get(qi, 0) + 1
                    if qi not in scored_qidx_info:
                        scored_qidx_info[qi] = (uid, q_id, ts)

                # 1. Kiểm tra tính duy nhất của qidx: 1 qidx chỉ ứng với 1 uid, 1 question, 1 timestamp
                if qi not in qidx_info:
                    qidx_info[qi] = (uid, q_id, ts)
                else:
                    orig_uid, orig_qid, orig_ts = qidx_info[qi]

                    if orig_uid != uid:
                        if qi not in violations_uid:
                            violations_uid[qi] = {orig_uid, uid}
                        else:
                            violations_uid[qi].add(uid)

                    if orig_qid != q_id:
                        if qi not in violations_question:
                            violations_question[qi] = {orig_qid, q_id}
                        else:
                            violations_question[qi].add(q_id)

                    if orig_ts != ts:
                        if qi not in violations_timestamp:
                            violations_timestamp[qi] = {orig_ts, ts}
                        else:
                            violations_timestamp[qi].add(ts)

                # 2. Kiểm tra tính nhất quán:
                # Cùng (uid, question, timestamp) -> có cùng qidx không?
                event_key = (uid, q_id, ts)
                if event_key not in event_to_qidx:
                    event_to_qidx[event_key] = {qi}
                else:
                    event_to_qidx[event_key].add(qi)

                # Cùng (uid, question) nhưng khác timestamp -> theo dõi các qidx tương ứng
                uq_key = (uid, q_id)
                if uq_key not in repeat_interactions:
                    repeat_interactions[uq_key] = {ts: {qi}}
                else:
                    if ts not in repeat_interactions[uq_key]:
                        repeat_interactions[uq_key][ts] = {qi}
                    else:
                        repeat_interactions[uq_key][ts].add(qi)

        if max_rows and total_rows >= max_rows:
            break

        if (chunk_idx + 1) % 10 == 0:
            print(f"  --> Đã quét {total_rows:,} dòng ({len(qidx_info):,} qidx duy nhất)...")

    # Kiểm tra phân biệt giữa các lần làm lại (repeated questions)
    repeat_same_qidx_violations = 0
    repeat_sample_violations = []

    for (u, q), ts_dict in repeat_interactions.items():
        if len(ts_dict) > 1:
            # Học sinh này làm câu hỏi q tại ít nhất 2 thời điểm khác nhau
            all_timestamps = list(ts_dict.keys())
            for i in range(len(all_timestamps)):
                for j in range(i + 1, len(all_timestamps)):
                    t1, t2 = all_timestamps[i], all_timestamps[j]
                    overlap = ts_dict[t1].intersection(ts_dict[t2])
                    if overlap:
                        repeat_same_qidx_violations += len(overlap)
                        if len(repeat_sample_violations) < 3:
                            repeat_sample_violations.append({
                                "uid": u,
                                "question": q,
                                "timestamp_1": t1,
                                "timestamp_2": t2,
                                "overlapping_qidx": list(overlap),
                            })

    # Kiểm tra tính nhất quán 1 event -> 1 qidx
    inconsistent_events = {
        k: v for k, v in event_to_qidx.items() if len(v) > 1
    }

    # Báo cáo tổng kết
    num_unique_qidxs = len(qidx_info)
    print("\n" + "=" * 60)
    print("           KẾT QUẢ KIỂM TRA TOÀN DIỆN CỘT qidxs")
    print("=" * 60)
    print(f"1. Tổng số dòng đã quét      : {total_rows:,}")
    print(f"2. Tổng số vị trí tương tác   : {total_valid_tokens:,} (bỏ qua padding -1)")
    print(f"3. Tổng số qidx duy nhất      : {num_unique_qidxs:,}")
    print("-" * 60)

    print(f"[Check 1] Tính duy nhất của 1 qidx:")
    print(f"  - Số qidx bị gán cho nhiều học sinh (UID): {len(violations_uid):,}")
    print(f"  - Số qidx bị gán cho nhiều câu hỏi       : {len(violations_question):,}")
    print(f"  - Số qidx bị gán cho nhiều thời điểm (ts): {len(violations_timestamp):,}")

    print(f"\n[Check 2] Tính nhất quán của tương tác:")
    print(f"  - Số tương tác (uid, q, ts) có > 1 qidx  : {len(inconsistent_events):,}")
    print(f"  - Số câu lặp lại khác thời điểm trùng qidx: {repeat_same_qidx_violations:,}")

    # Chi tiết vi phạm nếu có
    has_violation = (
        len(violations_uid) > 0
        or len(violations_question) > 0
        or len(violations_timestamp) > 0
    )

    if has_violation:
        print("\n" + "!" * 60)
        print("PHÁT HIỆN VI PHẠM:")
        if violations_uid:
            print(f"\n❌ [Lỗi UID] Có {len(violations_uid)} qidx thuộc về nhiều học sinh khác nhau!")
            for qi in list(violations_uid.keys())[:3]:
                print(f"    * qidx = {qi} xuất hiện ở các uid: {violations_uid[qi]}")

        if violations_question:
            print(f"\n❌ [Lỗi Question] Có {len(violations_question)} qidx ứng với nhiều câu hỏi khác nhau!")
            for qi in list(violations_question.keys())[:3]:
                print(f"    * qidx = {qi} xuất hiện ở các question: {violations_question[qi]}")

        if violations_timestamp:
            print(f"\n❌ [Lỗi Timestamp] Có {len(violations_timestamp)} qidx ứng với nhiều thời điểm khác nhau!")
            for qi in list(violations_timestamp.keys())[:3]:
                print(f"    * qidx = {qi} xuất hiện ở các timestamps: {violations_timestamp[qi]}")
        print("!" * 60)
    else:
        print("\n" + "✅" * 30)
        print("KẾT LUẬN: ĐẠT YÊU CẦU (PASSED)!")
        print("  -> Mỗi qidx HOÀN TOÀN DUY NHẤT:")
        print("     Chỉ ứng với chính xác 1 học sinh, 1 câu hỏi, tại đúng 1 thời điểm.")
        if repeat_same_qidx_violations == 0 and len(inconsistent_events) == 0:
            print("  -> Quan hệ 2 chiều hoàn hảo:")
            print("     - Mỗi lần trả lời một câu hỏi tại một thời điểm nhận đúng 1 qidx.")
            print("     - Nếu câu hỏi đó được làm lại ở thời điểm khác, nó nhận 1 qidx mới.")
        print("✅" * 30)

    # Đưa ra các assertion để dùng trong automated test / CI
    assert len(violations_uid) == 0, (
        f"LỖI: Có {len(violations_uid)} qidx bị gán cho nhiều học sinh khác nhau!"
    )
    assert len(violations_question) == 0, (
        f"LỖI: Có {len(violations_question)} qidx bị gán cho nhiều câu hỏi khác nhau!"
    )
    assert len(violations_timestamp) == 0, (
        f"LỖI: Có {len(violations_timestamp)} qidx bị gán cho nhiều thời điểm khác nhau!"
    )

    return {
        "total_rows": total_rows,
        "total_valid_tokens": total_valid_tokens,
        "unique_qidxs": num_unique_qidxs,
        "violations_uid": len(violations_uid),
        "violations_question": len(violations_question),
        "violations_timestamp": len(violations_timestamp),
        "inconsistent_events": len(inconsistent_events),
        "repeat_same_qidx_violations": repeat_same_qidx_violations,
    }


def main():
    args = parse_args()
    data_path = resolve_data_path(
        custom_path=args.data_path,
        use_sample=args.sample,
    )
    check_qidxs_uniqueness(
        data_path=data_path,
        max_rows=args.max_rows,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
