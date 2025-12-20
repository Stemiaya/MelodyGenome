import pandas as pd


def map_distribution(input_file, output_file):
    df = pd.read_csv(input_file)

    # 目标分布统计量（来自你的Fitness计算结果）
    target_mean = 102.6362
    target_std = 72.3792

    # 当前原始数据的统计量
    current_mean = df['final_score'].mean()
    current_std = df['final_score'].std()

    # 执行线性变换映射
    # 1. 归一化原分数 2. 放大到目标标准差 3. 平移到目标均值
    df['final_score'] = ((df['final_score'] - current_mean) /
                         current_std) * target_std + target_mean

    df.to_csv(output_file, index=False)
    print("分布映射完成")


map_distribution('dataset_total.csv', 'total.csv')
