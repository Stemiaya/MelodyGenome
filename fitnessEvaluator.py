import torch
import joblib
import numpy as np
import os
import sys

from fitnessModel import ClassicalResNet, ACGAttentionNet, PopTransformer, CombinedHybridNet


class FitnessEvaluator:
    def __init__(self, model_type='combined', checkpoint_dir='./checkpoints'):
        """
        Args:
            model_type: 选择使用哪个模型 ('classical', 'acg', 'pop', 'combined')
            checkpoint_dir: 存放 .pth 和 .pkl 文件的目录
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        print(
            f"[FitnessEvaluator] Initializing {model_type} model on {self.device}...")

        # 1. 实例化对应的模型架构
        if model_type == 'classical':
            self.model = ClassicalResNet()
        elif model_type == 'acg':
            self.model = ACGAttentionNet()
        elif model_type == 'pop':
            self.model = PopTransformer()
        elif model_type == 'combined':
            self.model = CombinedHybridNet()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # 2. 加载训练好的权重 (.pth)
        weight_path = os.path.join(checkpoint_dir, f"{model_type}_model.pth")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Model weights not found at {weight_path}. Did you train it?")

        self.model.load_state_dict(torch.load(
            weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # 开启评估模式

        # 3. 加载Key的编码器 (.pkl)
        utils_path = os.path.join(checkpoint_dir, f"{model_type}_model_utils")
        encoder_path = os.path.join(utils_path, 'key_encoder.pkl')

        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"Key encoder not found at {encoder_path}.")

        self.key_encoder = joblib.load(encoder_path)
        print("[FitnessEvaluator] Ready.")

    def get_fitness(self, melody_list, key_str):
        """
        计算旋律的适应度分数。

        Args:
            melody_list (list): 长度为64的整数列表，例如 [1, 13, 0, 5, ...]
            key_str (str): 调号字符串，例如 'C', 'F#', 'Bbm'

        Returns:
            float: 模型预测的分数 (越高越好)
        """
        # --- 1. 数据预处理 ---

        # 长度保护：确保是64维
        if len(melody_list) > 64:
            melody_list = melody_list[:64]
        elif len(melody_list) < 64:
            melody_list = melody_list + [0] * (64 - len(melody_list))

        # 转为 Tensor 并增加 Batch 维度 -> [1, 64]
        melody_tensor = torch.LongTensor(
            melody_list).unsqueeze(0).to(self.device)

        # 处理 Key
        try:
            # transform 接受列表，取第一个结果
            key_int = self.key_encoder.transform([key_str])[0]
        except ValueError:
            # 如果遗传算法生成了乱七八糟的调号（训练集里没有的），默认设为0
            # 或者可以打印一个警告
            key_int = 0

        key_tensor = torch.LongTensor([key_int]).unsqueeze(
            0).to(self.device)  # -> [1, 1]

        # --- 2. 模型预测 ---
        with torch.no_grad():  # 不需要计算梯度，节省内存加速计算
            score = self.model(melody_tensor, key_tensor)

        return score.item()
