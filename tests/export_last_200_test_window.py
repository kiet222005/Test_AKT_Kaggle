import argparse
import os
from pathlib import Path
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Xuất N dòng cuối cùng của file test_question_window_sequences.csv mà không cần nạp toàn bộ file vào RAM."
    )
    parser.add_argument(
        "--n_lines",
        type=int,
        default=200,
        help="Số dòng cuối cùng cần xuất (mặc định 200).",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Đường dẫn file đầu vào (mặc định tìm file raw trong data/raw/...).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Đường dẫn file kết quả xuất ra.",
    )
    return parser.parse_args()


def resolve_paths(custom_input=None, custom_output=None):
    base_dir = Path(__file__).resolve().parent.parent

    if custom_input:
        in_path = Path(custom_input)
    else:
        in_path = (
            base_dir
            / "data"
            / "raw"
            / "XES3G5M"
            / "XES3G5M"
            / "kc_level"
            / "test_question_window_sequences.csv"
        )

    if not in_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {in_path}")

    if custom_output:
        out_path = Path(custom_output)
    else:
        out_path = (
            base_dir
            / "data"
            / "xes3g5m_samples_200"
            / "kc_level"
            / "test_question_window_sequences_last_200.csv"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return in_path, out_path


def export_last_lines(input_path, output_path, n_lines=200, buffer_size=4 * 1024 * 1024):
    file_size = os.path.getsize(input_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"File nguồn: {input_path}")
    print(f"Dung lượng: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    print(f"Đang trích xuất {n_lines} dòng cuối cùng...")

    with open(input_path, "rb") as f:
        # 1. Đọc header từ đầu file
        header = f.readline()
        if not header.endswith(b"\n"):
            header += b"\n"

        # 2. Quét ngược từ cuối file
        lines = []
        pos = file_size
        remaining_bytes = b""

        while pos > 0 and len(lines) <= n_lines:
            read_size = min(buffer_size, pos)
            pos -= read_size
            f.seek(pos, os.SEEK_SET)
            chunk = f.read(read_size) + remaining_bytes

            chunk_lines = chunk.split(b"\n")
            # Đoạn đầu tiên chưa chắc là 1 dòng trọn vẹn nếu pos > 0
            remaining_bytes = chunk_lines[0]
            complete_lines = chunk_lines[1:]

            lines = complete_lines + lines

        if pos == 0 and remaining_bytes:
            lines = [remaining_bytes] + lines

    # Lọc bỏ các dòng trắng rỗng
    clean_lines = [l for l in lines if l.strip()]
    last_n = clean_lines[-n_lines:]

    # 3. Ghi ra file đích kèm header
    with open(output_path, "wb") as f_out:
        f_out.write(header)
        for line in last_n:
            f_out.write(line)
            if not line.endswith(b"\n"):
                f_out.write(b"\n")

    print(f" Xuất thành công {len(last_n)} dòng ra: {output_path}")

    # 4. Phân tích nhanh dữ liệu vừa xuất
    analyze_exported_lines(header, last_n)


def analyze_exported_lines(header_bytes, lines_bytes):
    header = header_bytes.decode("utf-8-sig", errors="ignore").strip().split(",")
    print("\n=== THỐNG KÊ NHANH 200 DÒNG CUỐI VỪA XUẤT ===")
    print(f"Các cột: {header}")

    uids = set()
    qidxs_seen = []

    uid_idx = header.index("uid") if "uid" in header else 1
    qidxs_idx = header.index("qidxs") if "qidxs" in header else -1

    for l in lines_bytes:
        line_str = l.decode("utf-8-sig", errors="ignore").strip()
        parts = []
        in_quotes = False
        current_token = []

        # Tách CSV hỗ trợ chuỗi có dấu ngoặc kép ""
        for char in line_str:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                parts.append("".join(current_token))
                current_token = []
                continue
            current_token.append(char)
        parts.append("".join(current_token))

        if len(parts) > uid_idx:
            uids.add(parts[uid_idx])

        if qidxs_idx != -1 and len(parts) > qidxs_idx:
            raw_q = parts[qidxs_idx].strip('"')
            parsed = [int(x.strip()) for x in raw_q.split(",") if x.strip() and x.strip() != "-1"]
            if parsed:
                qidxs_seen.extend(parsed)

    print(f"- Số học sinh (uid) xuất hiện trong 200 dòng cuối: {len(uids)} học sinh (UIDs: {list(uids)[:5]}...)")
    if qidxs_seen:
        print(f"- Khoảng giá trị qidx xuất hiện: từ {min(qidxs_seen)} đến {max(qidxs_seen)}")
        print(f"- Tổng số qidx hợp lệ (không tính -1): {len(qidxs_seen):,}")
        print(f"- Số qidx duy nhất (unique): {len(set(qidxs_seen)):,}")


def main():
    args = parse_args()
    in_path, out_path = resolve_paths(args.input_path, args.output_path)
    export_last_lines(in_path, out_path, n_lines=args.n_lines)


if __name__ == "__main__":
    main()
