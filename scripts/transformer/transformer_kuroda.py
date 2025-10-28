import torch
import torch.nn as nn

# 位置エンコーディング
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# Transformerモデル
class TransformerClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_size, num_heads=4, num_layers=2):
        super(TransformerClassification, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.d_model = hidden_dim
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=num_heads, dim_feedforward=4*self.d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, target_size)
        
    def forward(self, input_):
        x = self.embedding(input_)  # 入力を埋め込み
        x = self.pos_encoder(x)  # 位置エンコーディング
        x = self.transformer_encoder(x)  # Transformerエンコーダ
        x = x[:, -1, :]  # 最後のタイムステップの出力を使用
        scores = self.fc(x)  # 全結合層
        return scores