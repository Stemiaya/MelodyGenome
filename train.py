import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import argparse  # 引入命令行参数解析库

from dataset import MusicDataset
# 确保 models.py 和这个文件在同一个目录下
from fitnessModel import ClassicalResNet, ACGAttentionNet, PopTransformer, CombinedHybridNet

# === 配置区域 ===
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./checkpoints"

# 定义文件路径映射
DATA_FILES = {
    'classical': 'dataset_classical_instrumental.csv',
    'acg': 'dataset_acg_ost.csv',
    'pop': 'dataset_pop_contemporary.csv'
}

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def train_one_model(model_class, csv_paths, model_name):
    """
    model_class: 模型类本身 (例如 ClassicalResNet)，我们在函数内实例化
    csv_paths: 数据路径 (str 或 list)
    model_name: 用于保存文件名的前缀
    """
    print(f"\n{'='*20} Training {model_name} {'='*20}")
    print(f"Device: {DEVICE}")
    print(f"Data source: {csv_paths}")
    
    # 1. 实例化模型
    model = model_class().to(DEVICE)
    
    # 2. 准备数据
    # save_dir 用于保存 scaler 和 encoder，这对于后续使用至关重要
    dataset = MusicDataset(csv_paths, mode='train', save_dir=f"{SAVE_DIR}/{model_name}_utils")
    
    # 划分训练集和验证集 (80% / 20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # 3. 优化器与损失函数
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for melody, key, label in train_loader:
            melody, key, label = melody.to(DEVICE), key.to(DEVICE), label.to(DEVICE)
            
            optimizer.zero_grad()
            prediction = model(melody, key)
            loss = criterion(prediction.squeeze(), label.squeeze())
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for melody, key, label in val_loader:
                melody, key, label = melody.to(DEVICE), key.to(DEVICE), label.to(DEVICE)
                prediction = model(melody, key)
                loss = criterion(prediction.squeeze(), label.squeeze())
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # 打印进度 (每5轮一次，或者第一轮)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{SAVE_DIR}/{model_name}.pth")

    print(f"Finished {model_name}. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Train specific music evaluation models.")
    parser.add_argument(
        '--target', 
        type=str, 
        default='all', 
        choices=['classical', 'acg', 'pop', 'combined', 'all'],
        help="Specify which model to train: 'classical', 'acg', 'pop', 'combined', or 'all'"
    )
    args = parser.parse_args()

    # 根据参数选择执行
    targets = []
    if args.target == 'all':
        targets = ['classical', 'acg', 'pop', 'combined']
    else:
        targets = [args.target]

    print(f"Plan to train: {targets}")

    # 执行训练逻辑
    for t in targets:
        try:
            if t == 'classical':
                train_one_model(ClassicalResNet, DATA_FILES['classical'], 'classical_model')
            
            elif t == 'acg':
                train_one_model(ACGAttentionNet, DATA_FILES['acg'], 'acg_model')
            
            elif t == 'pop':
                train_one_model(PopTransformer, DATA_FILES['pop'], 'pop_model')
            
            elif t == 'combined':
                # combined 需要所有文件的列表
                all_files_list = list(DATA_FILES.values())
                train_one_model(CombinedHybridNet, all_files_list, 'combined_model')
                
        except Exception as e:
            print(f"Error occurred while training {t}: {e}")
            import traceback
            traceback.print_exc()