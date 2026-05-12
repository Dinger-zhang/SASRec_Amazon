# SASRec 推荐系统大作业

## 环境依赖
- Python 3.8+
- PyTorch >= 1.10
- pandas, numpy, tqdm

安装：
pip install torch pandas numpy tqdm

## 数据准备
1. 从 [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) 下载所需类别的5-core数据，
   包括 review 文件（例如 `Industrial_and_Scientific_5.json`）和 meta 文件（`meta_Industrial_and_Scientific_5.json`）。
2. 将所有json文件放入 `data/` 目录。

## 运行步骤

### 1. 预处理数据
python preprocess.py

该命令会在 `processed/` 下生成训练/验证/测试文件及物品映射。

### 2. 训练并评估模型
python train.py --category Industrial_and_Scientific
python train.py --category Musical_Instruments
python train.py --category CDs_and_Vinyl

可选参数（可按需调整）：
--max_len 50       # 最大序列长度
--embed_dim 50     # 嵌入维度
--num_blocks 2     # 自注意力块数量
--dropout 0.2      # Dropout比率
--batch_size 128
--lr 0.001
--epochs 50

## 输出
- 模型权重保存在 `saved_models/`
- 测试结果（NDCG@10, Hit@10）保存在 `results/`

## 实验报告
最终结果请汇总至实验报告，确保结果可复现。
