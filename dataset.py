import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class MusicDataset(Dataset):
    def __init__(self, csv_paths, mode='train', save_dir='./checkpoints'):
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        
        df_list = []
        for f in csv_paths:
            if os.path.exists(f):
                # 假设CSV第一行是列名。如果不是，请加 header=None, names=[...]
                df_list.append(pd.read_csv(f))
            else:
                print(f"Warning: File {f} not found!")
        
        if not df_list:
            raise ValueError("No data loaded.")
            
        self.df = pd.concat(df_list, ignore_index=True)
        
        # 解析 Melody Vector
        cleaned_melodies = []
        valid_indices = []
        
        for idx, row in self.df.iterrows():
            try:
                vec = ast.literal_eval(row['melody_vector'])
                # 强制长度检查：64
                if len(vec) > 64:
                    vec = vec[:64]
                elif len(vec) < 64:
                    vec = vec + [0] * (64 - len(vec))
                
                cleaned_melodies.append(vec)
                valid_indices.append(idx)
            except:
                continue

        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        self.melody_data = torch.LongTensor(np.array(cleaned_melodies))
        
        # 只处理 Key，忽略 Segment Start
        self.key_encoder = LabelEncoder()
        
        if mode == 'train':
            self.keys = self.key_encoder.fit_transform(self.df['key'])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            joblib.dump(self.key_encoder, f'{save_dir}/key_encoder.pkl')
        else:
            try:
                self.key_encoder = joblib.load(f'{save_dir}/key_encoder.pkl')
                self.keys = self.key_encoder.transform(self.df['key'])
            except:
                self.keys = self.key_encoder.fit_transform(self.df['key']) # Fallback

        self.keys = torch.LongTensor(self.keys)
        self.labels = torch.FloatTensor(self.df['final_score'].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 现在只返回：旋律, 调号, 分数
        return self.melody_data[idx], self.keys[idx], self.labels[idx]