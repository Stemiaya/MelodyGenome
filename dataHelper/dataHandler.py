import pandas as pd
import ast


def has_consecutive_zeros(vector, n=16):
    """
    检查列表中是否存在至少 n 个连续的 0
    """
    count = 0
    for x in vector:
        if x == 0:
            count += 1
            if count >= n:
                return True
        else:
            count = 0
    return False


def filter_csv(input_file, output_file):
    # 1. 读取 CSV
    df = pd.read_csv(input_file)

    # 2. 将字符串形式的列表转换为真正的列表
    df['melody_vector'] = df['melody_vector'].apply(ast.literal_eval)

    # 3. 定义过滤逻辑：
    mask = df['melody_vector'].apply(
        lambda v: not has_consecutive_zeros(v, 16))

    df_filtered = df[mask]

    # 4. 保存结果
    df_filtered.to_csv(output_file, index=False)

    print(f"处理完成！")
    print(f"原始行数: {len(df)}")
    print(f"过滤后行数: {len(df_filtered)}")
    print(f"删除了 {len(df) - len(df_filtered)} 条包含连续16个0的数据。")


# 使用示例
filter_csv('./data/dataset_pop_contemporary.csv', './data/pop.csv')
