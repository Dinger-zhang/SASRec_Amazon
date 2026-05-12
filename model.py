import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SASRec(nn.Module):
    def __init__(self, item_num, max_len, embed_dim, num_blocks, dropout_rate, device):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.device = device
        
        # 共享物品嵌入
        self.item_embed = nn.Embedding(item_num+1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        
        # 自注意力块
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionBlock(embed_dim, dropout_rate) for _ in range(num_blocks)]
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # 预测层共享嵌入权重（论文式6）
        self.W = self.item_embed.weight  # (item_num+1, embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.item_embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0, std=0.02)
        # padding嵌入置零
        with torch.no_grad():
            self.item_embed.weight[0] = 0.0
    
    def forward(self, seq, pos=None):
        # seq: (batch, seq_len)
        batch_size, seq_len = seq.shape
        mask = (seq != 0).unsqueeze(1)  # (batch, 1, seq_len)
        
        # 嵌入
        item_emb = self.item_embed(seq)  # (batch, seq_len, embed_dim)
        if pos is None:
            pos = torch.arange(seq_len, device=self.device).unsqueeze(0)
        pos_emb = self.pos_embed(pos)    # (1, seq_len, embed_dim)
        
        x = item_emb + pos_emb
        x = self.dropout(x)
        
        # 因果注意力掩码（下三角）
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).unsqueeze(0)  # (1, seq_len, seq_len)
        attn_mask = mask.float() * causal_mask  # (batch, seq_len, seq_len)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, -1e9)
        
        for block in self.attention_blocks:
            x = block(x, attn_mask)
        
        x = self.layer_norm(x)  # (batch, seq_len, embed_dim)
        
        # 预测所有物品的得分
        logits = x @ self.W.transpose(0, 1)  # (batch, seq_len, item_num+1)
        return logits

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, dropout_rate):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1, dropout=dropout_rate, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, attn_mask):
        # 自注意力
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        # 前馈网络
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x