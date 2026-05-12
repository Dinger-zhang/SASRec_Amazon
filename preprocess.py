import json
import os
import pandas as pd
from collections import defaultdict

def load_and_process(category, data_dir='data'):
    """
    加载Amazon Reviews 2023的5-core数据，按时间排序，
    划分训练/验证/测试集，并生成序列格式文件。
    """
    # 读取review和meta
    review_file = os.path.join(data_dir, f'{category}_5.json')
    meta_file = os.path.join(data_dir, f'meta_{category}_5.json')
    
    # 加载评论
    reviews = []
    with open(review_file, 'r') as f:
        for line in f:
            reviews.append(json.loads(line))
    df = pd.DataFrame(reviews)
    # 选择必要字段
    df = df[['user_id', 'parent_asin', 'timestamp']]
    # 按用户和时间排序
    df.sort_values(['user_id', 'timestamp'], inplace=True)
    
    # 构建用户序列
    user_seq = defaultdict(list)
    for _, row in df.iterrows():
        user_seq[row['user_id']].append(row['parent_asin'])
    
    # 过滤序列长度<3的用户（因为需要train+val+test）
    user_seq = {u: seq for u, seq in user_seq.items() if len(seq) >= 3}
    
    # 划分：前N-2训练，第N-1验证，第N测试
    train_data = []
    val_data = []
    test_data = []
    
    for user, seq in user_seq.items():
        # 训练集：生成序列
        for i in range(1, len(seq)-1):  # i是当前要预测的位置，从1到len-2
            history = seq[:i]
            target = seq[i]
            train_data.append((user, history, target))
        # 验证集：输入前N-1个，预测第N-1个
        val_data.append((user, seq[:-2], seq[-2]))
        # 测试集：输入前N个，预测第N个
        test_data.append((user, seq[:-1], seq[-1]))
    
    # 保存为tsv格式（便于后续读取）
    os.makedirs('processed', exist_ok=True)
    for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        with open(f'processed/{category}_{name}.txt', 'w') as f:
            for user, hist, target in data:
                f.write(f"{user}\t{' '.join(hist)}\t{target}\n")
    
    # 构建物品映射（所有出现过的parent_asin）
    items = set()
    for seq in user_seq.values():
        items.update(seq)
    item2id = {item: i+1 for i, item in enumerate(items)}  # 0留给padding
    id2item = {i: item for item, i in item2id.items()}
    import pickle
    with open(f'processed/{category}_item_map.pkl', 'wb') as f:
        pickle.dump((item2id, id2item), f)
    
    print(f'{category}: train samples {len(train_data)}, val users {len(val_data)}, test users {len(test_data)}')
    return

if __name__ == '__main__':
    for cat in ['Industrial_and_Scientific', 'Musical_Instruments', 'CDs_and_Vinyl']:
        load_and_process(cat)