import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SASRec
from utils import SequenceDataset, NDCG_at_k, Hit_at_k, parse_history
import pickle
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    category = args.category
    data_dir = args.data_dir
    max_len = args.max_len
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.epochs
    dropout = args.dropout
    num_blocks = args.num_blocks

    # 加载物品映射
    with open(f'processed/{category}_item_map.pkl', 'rb') as f:
        item2id, id2item = pickle.load(f)
    item_num = len(item2id)
    print(f"Item num: {item_num}")

    # 数据集
    train_csv = os.path.join(data_dir, f'{category}.train.csv')
    val_csv = os.path.join(data_dir, f'{category}.valid.csv')
    test_csv = os.path.join(data_dir, f'{category}.test.csv')

    train_dataset = SequenceDataset(train_csv, item2id, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 模型
    model = SASRec(item_num, max_len, embed_dim, num_blocks, dropout, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_ndcg = 0.0
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for seq, target in tqdm(train_loader, desc=f'Epoch {epoch}'):
            seq = seq.to(device)
            target = target.to(device)

            # 输入去掉最后一个位置，预测最后一步
            input_seq = seq[:, :-1]  # (batch, max_len-1)
            logits = model(input_seq)  # (batch, max_len-1, item_num+1)
            last_logits = logits[:, -1, :]  # (batch, item_num+1)

            loss = criterion(last_logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Loss: {avg_loss:.4f}")

        # 验证
        model.eval()
        ndcg, hit = evaluate_batch(model, val_csv, item2id, id2item, max_len, device)
        print(f"Validation - NDCG@10: {ndcg:.4f}, Hit@10: {hit:.4f}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            torch.save(model.state_dict(), f'saved_models/{category}_best.pth')

    # 测试
    model.load_state_dict(torch.load(f'saved_models/{category}_best.pth'))
    model.eval()
    ndcg, hit = evaluate_batch(model, test_csv, item2id, id2item, max_len, device)
    print(f"Test - NDCG@10: {ndcg:.4f}, Hit@10: {hit:.4f}")

    with open(f'results/{category}_result.txt', 'w') as f:
        f.write(f'NDCG@10: {ndcg:.4f}\nHit@10: {hit:.4f}\n')


def evaluate_batch(model, csv_path, item2id, id2item, max_len, device, sample_size=100, eval_batch_size=256):
    """批量评估，大幅提升速度"""
    df = pd.read_csv(csv_path)

    # 解析所有有效样本
    all_hist_ids = []
    all_targets = []
    for _, row in df.iterrows():
        target_id = item2id.get(row['parent_asin'])
        if target_id is None:
            continue
        hist_items = parse_history(row.get('history', ''))
        hist_ids = [item2id[i] for i in hist_items if i in item2id]
        if len(hist_ids) == 0:
            continue
        all_hist_ids.append(hist_ids)
        all_targets.append(target_id)

    total_items = list(item2id.values())
    num_samples = len(all_hist_ids)

    ndcg_list = []
    hit_list = []

    # 分批处理
    for start in tqdm(range(0, num_samples, eval_batch_size), desc='Evaluating'):
        end = min(start + eval_batch_size, num_samples)
        batch_hists = all_hist_ids[start:end]
        batch_targets = all_targets[start:end]
        current_batch_size = len(batch_hists)

        # 1. 构建序列张量 (batch, max_len)
        batch_seqs = []
        for hist in batch_hists:
            seq = hist[-max_len:]
            seq = [0] * (max_len - len(seq)) + seq
            batch_seqs.append(seq)
        seq_tensor = torch.tensor(batch_seqs, dtype=torch.long, device=device)

        # 2. 为每个样本采样负样本，组成候选集 (batch, 1+sample_size)
        candidates = []
        for target in batch_targets:
            negs = []
            while len(negs) < sample_size:
                neg = np.random.choice(total_items)
                if neg != target:
                    negs.append(neg)
            candidates.append([target] + negs)
        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)  # (batch, 101)

        # 3. 模型批量前向，只取最后位置的输出
        with torch.no_grad():
            input_seq = seq_tensor[:, :-1]  # (batch, max_len-1)
            logits = model(input_seq)  # (batch, max_len-1, item_num+1)
            last_logits = logits[:, -1, :]  # (batch, item_num+1)

            # 4. 从全量logits中抽取候选物品的分数
            # last_logits: (batch, item_num+1), candidate_tensor: (batch, 101)
            scores = torch.gather(last_logits, 1, candidate_tensor)  # (batch, 101)
            _, indices = torch.sort(scores, descending=True, dim=1)  # (batch, 101)
            ranked = candidate_tensor.gather(1, indices)  # (batch, 101)

        # 5. 逐样本计算指标
        for i in range(current_batch_size):
            ranked_list = ranked[i].cpu().tolist()
            true_item = batch_targets[i]
            ndcg_list.append(NDCG_at_k(ranked_list, true_item, k=10))
            hit_list.append(Hit_at_k(ranked_list, true_item, k=10))

    return np.mean(ndcg_list), np.mean(hit_list)


def evaluate(model, csv_path, item2id, id2item, max_len, device, sample_size=100):
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        target_id = item2id.get(row['parent_asin'])
        if target_id is None:
            continue
        hist_str = row.get('history', '')
        hist_items = parse_history(hist_str)
        hist_ids = [item2id[i] for i in hist_items if i in item2id]
        if len(hist_ids) == 0:
            continue
        data.append((hist_ids, target_id))

    total_items = list(item2id.values())
    ndcg_list = []
    hit_list = []

    for hist_ids, target_id in tqdm(data, desc='Evaluating'):
        seq = hist_ids[-max_len:]
        seq = [0] * (max_len - len(seq)) + seq
        seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)

        # 负采样100个
        neg_samples = []
        while len(neg_samples) < sample_size:
            neg = np.random.choice(total_items)
            if neg != target_id:
                neg_samples.append(neg)
        candidates = [target_id] + neg_samples
        candidate_tensor = torch.tensor(candidates, dtype=torch.long).to(device)

        with torch.no_grad():
            input_seq = seq_tensor[:, :-1]
            logits = model(input_seq)
            last_logits = logits[:, -1, :]
            scores = last_logits[0, candidates]
            ranked_indices = torch.argsort(scores, descending=True).cpu().numpy()
            ranked_items = [candidates[i] for i in ranked_indices]

        ndcg = NDCG_at_k(ranked_items, target_id, k=10)
        hit = Hit_at_k(ranked_items, target_id, k=10)
        ndcg_list.append(ndcg)
        hit_list.append(hit)

    return np.mean(ndcg_list), np.mean(hit_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, required=True,
                        choices=['Industrial_and_Scientific', 'Musical_Instruments', 'CDs_and_Vinyl'])
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=50)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    train(args)
