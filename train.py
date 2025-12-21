import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 引入学习率调度器
import os
import argparse
import random
import numpy as np

from dataHelper.dataPreprocess import MusicDataset
from fitnessModel import ClassicalResNet, ACGAttentionNet, PopTransformer, CombinedHybridNet

# === 配置区域 ===
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100       # 设置大一点，反正有早停机制
PATIENCE = 10      # 早停忍耐度：如果10轮内验证集Loss没创新低，就停止
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./checkpoints"
SEED = 42          # 固定随机种子

# 定义文件路径映射
DATA_FILES = {
    'classical': './data/dataset_classical_instrumental.csv',
    'acg': './data/dataset_acg_ost.csv',
    'pop': './data/dataset_pop_contemporary.csv',
    'all': './data/dataset_total.csv'
}

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def set_seed(seed):
    """固定所有随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_model(model_class, csv_paths, model_name):
    print(f"\n{'='*20} Training {model_name} {'='*20}")
    print(f"Device: {DEVICE}")

    # 1. 准备数据
    dataset = MusicDataset(csv_paths, mode='train',
                           save_dir=f"{SAVE_DIR}/{model_name}_utils")

    # 划分训练集和验证集 (80% / 20%)
    # 使用 Generator 固定划分方式，确保每次跑验证集的数据是一样的
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # 注意：train_loader 开启 drop_last=True 防止 BN 层报错
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, drop_last=False)

    # 2. 模型初始化
    model = model_class().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # 增加L2正则化防止过拟合

    # 学习率调度器：当 val_loss 不再下降时，降低学习率
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    # 3. 训练循环变量
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0

        for melody, key, label in train_loader:
            melody, key, label = melody.to(
                DEVICE), key.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()
            prediction = model(melody, key)
            loss = criterion(prediction.squeeze(), label.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for melody, key, label in val_loader:
                melody, key, label = melody.to(
                    DEVICE), key.to(DEVICE), label.to(DEVICE)
                prediction = model(melody, key)
                loss = criterion(prediction.squeeze(), label.squeeze())
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 打印日志
        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", end="")

        # --- 核心逻辑：保存最优模型 & 早停 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0  # 重置计数器
            save_path = f"{SAVE_DIR}/{model_name}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  --> Best Model Saved! (Loss: {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"  | Patience: {early_stop_counter}/{PATIENCE}")

        if early_stop_counter >= PATIENCE:
            print(
                f"\n[Early Stopping] Validation loss hasn't improved for {PATIENCE} epochs.")
            print(
                f"Training stopped. Best Validation Loss was: {best_val_loss:.4f}")
            break

    if early_stop_counter < PATIENCE:
        print(f"\nFinished. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    # 设置随机种子
    set_seed(SEED)

    parser = argparse.ArgumentParser(
        description="Train specific music evaluation models.")
    parser.add_argument('--target', type=str, default='combined',
                        choices=['classical', 'acg', 'pop', 'combined', 'all'])
    args = parser.parse_args()

    targets = []
    if args.target == 'all':
        targets = ['classical', 'acg', 'pop', 'combined']
    else:
        targets = [args.target]

    for t in targets:
        try:
            if t == 'classical':
                train_one_model(ClassicalResNet,
                                DATA_FILES['classical'], 'classical_model')
            elif t == 'acg':
                train_one_model(ACGAttentionNet,
                                DATA_FILES['acg'], 'acg_model')
            elif t == 'pop':
                train_one_model(PopTransformer, DATA_FILES['pop'], 'pop_model')

            elif t == 'combined':
                train_one_model(CombinedHybridNet,
                                DATA_FILES['all'], 'combined_model')
        except Exception as e:
            print(f"Error training {t}: {e}")
            import traceback
            traceback.print_exc()
