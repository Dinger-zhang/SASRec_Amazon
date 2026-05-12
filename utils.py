import numpy as np

def NDCG_at_k(ranked_list, true_item, k=10):
    """计算单个用户的NDCG@k"""
    if true_item in ranked_list[:k]:
        rank = ranked_list[:k].index(true_item)
        return np.reciprocal(np.log2(rank + 2))  # 位置从0开始，对应DCG为1/log2(rank+2)
    return 0.0

def Hit_at_k(ranked_list, true_item, k=10):
    return 1.0 if true_item in ranked_list[:k] else 0.0

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, item2id, max_len):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                user, hist, target = line.strip().split('\t')
                hist_ids = [item2id[i] for i in hist.split() if i in item2id]
                target_id = item2id.get(target, None)
                if target_id is not None and len(hist_ids) > 0:
                    self.data.append((hist_ids, target_id))
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        hist, target = self.data[idx]
        # 截断或填充到max_len
        seq = hist[-self.max_len:]
        seq = [0]*(self.max_len - len(seq)) + seq
        return torch.tensor(seq, dtype=torch.long), target