import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class KTTrainDataset(Dataset):

  def __init__(self, csv_file, fold_ids, max_len=200):
    self.max_len = max_len
    self.samples = []

    df_all = pd.read_csv(csv_file, encoding="utf-8-sig")
    df_filtered = df_all[df_all["fold"].isin(fold_ids)]

    print(
        f"--> [Train/Valid Data] Nạp {len(df_filtered):,} chuỗi từ Folds"
        f" {fold_ids}..."
    )

    for _, row in df_filtered.iterrows():
      self.samples.append({
          "concepts": self.parse_seq(row["concepts"]),
          "questions": self.parse_seq(row["questions"]),
          "responses": self.parse_seq(row["responses"]),
          "select_masks": self.parse_seq(row["selectmasks"]),
          "is_repeat": self.parse_seq(row["is_repeat"]),
      })

  def __len__(self):
    return len(self.samples)

  def parse_seq(self, seq_str):
    arr = np.fromstring(str(seq_str), sep=",", dtype=np.int64)
    return torch.from_numpy(arr)

  def __getitem__(self, idx):
    return self.samples[idx]