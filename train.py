import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SASRec
from utils import SequenceDataset, NDCG_at_k, Hit_at_k
import pickle
import os
import argparse
import numpy as np
from tqdm import tqdm

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    category = args.category
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
    train_dataset = SequenceDataset(f'processed/{category}_train.txt', item2id, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 模型
    model = SASRec(item_num, max_len, embed_dim, num_blocks, dropout, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding位置
    
    best_ndcg = 0.0
    
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        for seq, target in tqdm(train_loader, desc=f'Epoch {epoch}'):
            seq = seq.to(device)         # (batch, max_len)
            target = target.to(device)   # (batch,)
            
            # 输入序列去掉最后一个，预测序列向后偏移一位
            input_seq = seq[:, :-1]      # (batch, max_len-1)
            output_seq = seq[:, 1:]      # (batch, max_len-1)
            
            # 将目标位置的item_id替换到output_seq的最后一个位置？
            # 论文中每个时间步预测下一个物品，我们简化：取最后一个有效位置的输出预测target
            # 更符合论文的实现是：对于序列中每个位置都做预测，这里为了效率，只预测最后一步。
            # 但你也可按论文方法实现全部位置预测。我们提供两种模式，此处用最后一步预测。
            logits = model(input_seq)    # (batch, max_len-1, item_num+1)
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
        val_file = f'processed/{category}_val.txt'
        ndcg, hit = evaluate(model, val_file, item2id, id2item, max_len, device)
        print(f"Validation - NDCG@10: {ndcg:.4f}, Hit@10: {hit:.4f}")
        
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            torch.save(model.state_dict(), f'saved_models/{category}_best.pth')
    
    # 测试
    model.load_state_dict(torch.load(f'saved_models/{category}_best.pth'))
    model.eval()
    test_file = f'processed/{category}_test.txt'
    ndcg, hit = evaluate(model, test_file, item2id, id2item, max_len, device)
    print(f"Test - NDCG@10: {ndcg:.4f}, Hit@10: {hit:.4f}")
    
    # 保存结果
    with open(f'results/{category}_result.txt', 'w') as f:
        f.write(f'NDCG@10: {ndcg:.4f}\nHit@10: {hit:.4f}\n')

def evaluate(model, file_path, item2id, id2item, max_len, device, sample_size=100):
    """
    对验证/测试集评估，采用负采样方式（采样100个负样本）
    """
    # 加载数据（用户-序列-目标）
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            user, hist, target = line.strip().split('\t')
            hist_ids = [item2id[i] for i in hist.split() if i in item2id]
            target_id = item2id.get(target, None)
            if target_id and len(hist_ids) > 0:
                data.append((hist_ids, target_id))
    
    ndcg_list = []
    hit_list = []
    
    # 所有物品id列表用于负采样
    total_items = list(item2id.values())
    
    for hist_ids, target_id in tqdm(data, desc='Evaluating'):
        # 构造输入序列
        seq = hist_ids[-max_len:]
        seq = [0]*(max_len - len(seq)) + seq
        seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
        
        # 负采样100个（不含目标物品）
        neg_samples = []
        while len(neg_samples) < sample_size:
            neg = np.random.choice(total_items)
            if neg != target_id:
                neg_samples.append(neg)
        candidates = [target_id] + neg_samples  # 第一个是正样本
        candidate_tensor = torch.tensor(candidates, dtype=torch.long).to(device)
        
        with torch.no_grad():
            # 模型前向
            input_seq = seq_tensor[:, :-1]  # (1, max_len-1)
            logits = model(input_seq)       # (1, max_len-1, item_num+1)
            last_logits = logits[:, -1, :]  # (1, item_num+1)
            scores = last_logits[0, candidates]  # (101,)
            # 按分数降序排列
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
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=50)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    train(args)