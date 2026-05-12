import pandas as pd
import pickle
import os
from collections import defaultdict


def build_item_mapping(data_dir='E:\\program\\recommendation_system\\dataset'):
    categories = ['Industrial_and_Scientific', 'Musical_Instruments', 'CDs_and_Vinyl']
    for cat in categories:
        # 读取所有csv文件收集所有出现的parent_asin
        all_asins = set()
        for split in ['train', 'valid', 'test']:
            df = pd.read_csv(os.path.join(data_dir, f'{cat}.{split}.csv'))
            if 'history' in df.columns:
                # history可能是字符串表示的列表，例如 "['B001','B002']" 或 "B001 B002"
                for h in df['history']:
                    if isinstance(h, str):
                        # 尝试解析不同格式
                        items = parse_history(h)
                        all_asins.update(items)
            if 'parent_asin' in df.columns:
                all_asins.update(df['parent_asin'].dropna().unique())

        # 建立映射，0 留给padding
        item2id = {item: idx + 1 for idx, item in enumerate(sorted(all_asins))}
        id2item = {idx: item for item, idx in item2id.items()}

        os.makedirs('processed', exist_ok=True)
        with open(f'processed/{cat}_item_map.pkl', 'wb') as f:
            pickle.dump((item2id, id2item), f)
        print(f"{cat}: {len(item2id)} unique items mapped.")


def parse_history(hist_str):
    """解析history字段，支持列表字符串或空格分隔的字符串"""
    if hist_str.startswith('['):
        # 形如 "['B01', 'B02']"
        hist_str = hist_str.strip("[]").replace("'", "").replace('"', '')
        items = [x.strip() for x in hist_str.split(',') if x.strip()]
    else:
        items = hist_str.split()
    return items


if __name__ == '__main__':
    build_item_mapping()