import ast                      # 用于把CSV里的字符串 "[1, 2]" 转回列表
import numpy as np              # 数学运算
import pandas as pd             # 读取CSV文件
import torch                    # PyTorch深度学习框架核心
import torch.nn as nn           # 神经网络模块 (Neural Network)
import torch.optim as optim     # 优化器 (如Adam)
from torch.utils.data import Dataset, DataLoader # 数据加载器
from sklearn.preprocessing import StandardScaler, LabelEncoder # 数据预处理
from sklearn.model_selection import train_test_split # 拆分训练集/验证集

class ClassicalResNet(nn.Module):
    def __init__(self):
        super(ClassicalResNet, self).__init__()
        # 1. 音符嵌入: 词表14个 -> 32维
        self.emb = nn.Embedding(14, 32)
        
        # 2. 初始卷积
        self.stem = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 3. 残差块 (定义在下面)
        self.layer1 = self._make_res_block(64, 64)
        self.layer2 = self._make_res_block(64, 128)
        self.layer3 = self._make_res_block(128, 128)
        
        # 4. 池化
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 5. 辅助特征嵌入 (Key)
        self.key_emb = nn.Embedding(24, 4)
        
        # 6. 最终评分层
        self.fc = nn.Sequential(
            nn.Linear(128 + 4, 64), # 128(旋律) + 4(Key)
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _make_res_block(self, in_c, out_c):
        # 如果通道数变了，需要用1x1卷积调整residual的尺寸
        downsample = None
        if in_c != out_c:
            downsample = nn.Conv1d(in_c, out_c, 1)
            
        return ResBlock(in_c, out_c, downsample)

    def forward(self, x, k):
        # x: [Batch, 64]
        x = self.emb(x).permute(0, 2, 1) # 卷积需要 [Batch, Channel, Time]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1) # [Batch, 128]
        
        k = self.key_emb(k).flatten(1)
        
        # 拼接所有特征
        combined = torch.cat([x, k], dim=1)
        return self.fc(combined)

# 辅助类：残差块
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity # 核心：残差连接 (跳跃连接)
        out = self.relu(out)
        return out

class ACGAttentionNet(nn.Module):
    def __init__(self):
        super(ACGAttentionNet, self).__init__()
        # 嵌入维度较小 (16) 防止过拟合
        self.emb = nn.Embedding(14, 16) 
        
        # 3层双向LSTM
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=3, 
                            batch_first=True, bidirectional=True, dropout=0.4)
        
        # 注意力层：把LSTM所有时刻的输出压缩成一个权重
        self.attn_fc = nn.Linear(64, 1) # 64 = 32 * 2 (双向)
        
        self.key_emb = nn.Embedding(24, 4)
        
        self.fc = nn.Sequential(
            nn.Linear(64 + 4, 32),
            nn.ReLU(),
            nn.Dropout(0.4), # 高Dropout
            nn.Linear(32, 1)
        )

    def forward(self, x, k):
        x = self.emb(x)
        # lstm_out: [Batch, Seq_Len, Hidden*2]
        lstm_out, _ = self.lstm(x) 
        
        # Attention 计算
        # 1. 计算得分: [Batch, Seq_Len, 1]
        attn_scores = self.attn_fc(lstm_out) 
        # 2. 归一化得分 (Softmax)
        attn_weights = torch.softmax(attn_scores, dim=1)
        # 3. 加权求和: sum([Batch, Seq_Len, Hidden*2] * [Batch, Seq_Len, 1])
        context = torch.sum(lstm_out * attn_weights, dim=1) # [Batch, 64]
        
        k = self.key_emb(k).flatten(1)
        return self.fc(torch.cat([context, k], dim=1))

class PopTransformer(nn.Module):
    def __init__(self):
        super(PopTransformer, self).__init__()
        self.d_model = 64
        self.emb = nn.Embedding(14, self.d_model)
        
        # 可学习的位置编码 (Positional Encoding)
        # 告诉模型哪个音是第一拍，哪个是第四拍
        self.pos_emb = nn.Parameter(torch.randn(1, 64, self.d_model))
        
        # Transformer 层 (6层堆叠)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, 
                                                   dim_feedforward=256, batch_first=True,
                                                   activation="gelu") # 使用 GELU
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.key_emb = nn.Embedding(24, 4)
        
        # 宽全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.d_model + 4, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, k):
        # x: [Batch, 64] -> [Batch, 64, 64]
        x = self.emb(x) + self.pos_emb 
        
        # Transformer 处理
        x = self.transformer(x)
        
        # 平均池化：把64个时间步的特征取平均
        x = x.mean(dim=1) 
        
        k = self.key_emb(k).flatten(1)
        return self.fc(torch.cat([x, k], dim=1))
    

class CombinedHybridNet(nn.Module):
    def __init__(self):
        super(CombinedHybridNet, self).__init__()
        self.emb = nn.Embedding(14, 64) # 高维嵌入
        
        # Stream A: CNN (看局部音型)
        self.cnn_stream = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) # 强行提取最显著特征
        )
        
        # Stream B: RNN (看整体流向)
        self.rnn_stream = nn.GRU(input_size=64, hidden_size=64, num_layers=2, 
                                 batch_first=True, bidirectional=True)
        
        self.key_emb = nn.Embedding(24, 8) # Key特征稍微大一点
        
        # 融合层
        self.fc = nn.Sequential(
            # CNN出128 + RNN出128(64*2) + Key(8)
            nn.Linear(128 + 128 + 8, 256), 
            nn.BatchNorm1d(256), # 全连接层也可以用BN
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, k):
        emb = self.emb(x)
        
        # 流 A: CNN (需要变换维度)
        cnn_in = emb.permute(0, 2, 1)
        cnn_feat = self.cnn_stream(cnn_in).flatten(1) # [Batch, 128]
        
        # 流 B: RNN
        _, h_n = self.rnn_stream(emb)
        # 取 GRU 最后时刻的隐状态 (双向需拼接最后两层)
        rnn_feat = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) # [Batch, 128]
        
        k = self.key_emb(k).flatten(1)
        
        # 融合
        combined = torch.cat([cnn_feat, rnn_feat, k], dim=1)
        return self.fc(combined)