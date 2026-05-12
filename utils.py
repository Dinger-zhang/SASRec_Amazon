import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


def NDCG_at_k(ranked_list, true_item, k=10):
    if true_item in ranked_list[:k]:
        rank = ranked_list[:k].index(true_item)
        return np.reciprocal(np.log2(rank + 2))
    return 0.0


def Hit_at_k(ranked_list, true_item, k=10):
    return 1.0 if true_item in ranked_list[:k] else 0.0


def parse_history(hist_str):
    """将history字符串解析为物品ID列表"""
    if isinstance(hist_str, float) and np.isnan(hist_str):
        return []
    if hist_str.startswith('['):
        hist_str = hist_str.strip("[]").replace("'", "").replace('"', '')
        items = [x.strip() for x in hist_str.split(',') if x.strip()]
    else:
        items = hist_str.split()
    return items


class SequenceDataset(Dataset):
    def __init__(self, csv_path, item2id, max_len):
        self.data = []
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            user = row['user_id']
            target_id = item2id.get(row['parent_asin'])
            if target_id is None:
                continue
            hist_str = row.get('history', '')
            hist_items = parse_history(hist_str)
            hist_ids = [item2id[i] for i in hist_items if i in item2id]
            if len(hist_ids) == 0:
                continue
            self.data.append((hist_ids, target_id))
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hist, target = self.data[idx]
        seq = hist[-self.max_len:]
        seq = [0] * (self.max_len - len(seq)) + seq
        return torch.tensor(seq, dtype=torch.long), target