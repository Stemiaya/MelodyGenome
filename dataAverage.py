import pandas as pd
import ast
import numpy as np


class ScoreEvaluator:
    def __init__(self):
        pass

    def calculate_rule_score(self, individual):
        """
        基于你提供的乐理规则评分逻辑
        individual: 64个整数的列表
        """
        score = 0
        major_scale = [1, 3, 5, 6, 8, 10, 12]
        chord_tones = [1, 5, 8]
        genome_len = len(individual)

        # 提取有效音符 (1-12之间)
        notes = [(individual[i], i)
                 for i in range(genome_len) if 1 <= individual[i] <= 12]

        # 如果有效音符太少，直接返回最低分 1
        if len(notes) < 8:
            return 1

        # 1. 节奏约束
        rhythm_penalty = 0
        for i in range(0, genome_len, 2):
            # 如果弱拍位置有新音符起始（非13且非持续音），扣分
            if i+1 < genome_len and individual[i+1] != 13 and individual[i+1] != individual[i] and individual[i+1] != 0:
                rhythm_penalty += 15
        score -= rhythm_penalty

        # 2. 强拍和弦音 (每4个位置为一个强拍)
        strong_beat_reward = 0
        for pos in range(0, genome_len, 4):
            note = individual[pos]
            if note in chord_tones:
                strong_beat_reward += 10
            elif note in major_scale:
                strong_beat_reward += 5
        score += strong_beat_reward

        # 3. 起止音
        if notes[0][0] in chord_tones:
            score += 15
        if notes[-1][0] == 1:
            score += 30
        elif notes[-1][0] in chord_tones:
            score += 15

        # 4. 音程与方向
        interval_score = 0
        direction_changes = 0
        prev_direction = 0
        for i in range(len(notes) - 1):
            pitch1, _ = notes[i]
            pitch2, _ = notes[i+1]
            diff = abs(pitch2 - pitch1)
            if diff == 0:
                interval_score += 2
            elif 1 <= diff <= 2:
                interval_score += 8
            elif 3 <= diff <= 4:
                interval_score += 5
            elif diff > 7:
                interval_score -= 20

            curr_dir = 1 if pitch2 > pitch1 else -1 if pitch2 < pitch1 else 0
            if curr_dir != 0 and prev_direction != 0 and curr_dir != prev_direction:
                direction_changes += 1
            prev_direction = curr_dir

        if direction_changes > len(notes) / 3:
            interval_score -= 15
        score += interval_score

        # 5. 丰富度
        unique_notes = len(set([n[0] for n in notes]))
        if unique_notes < 3:
            score -= 50
        else:
            score += unique_notes * 5

        return max(score, 1)


def process_and_analyze(csv_file):
    # 1. 读取数据
    print(f"正在读取文件: {csv_file}...")
    df = pd.read_csv(csv_file)

    # 2. 解析列表字符串
    # 将 "[1, 2, ...]" 转换为真正的 Python list
    df['melody_vector'] = df['melody_vector'].apply(ast.literal_eval)

    # 3. 计算评分
    evaluator = ScoreEvaluator()
    print("正在计算每条数据的 Fitness 分数...")
    calculated_scores = df['melody_vector'].apply(
        evaluator.calculate_rule_score)

    # 4. 统计计算
    mean_val = np.mean(calculated_scores)
    std_val = np.std(calculated_scores)
    max_val = np.max(calculated_scores)
    min_val = np.min(calculated_scores)
    median_val = np.median(calculated_scores)

    # 5. 输出结果
    print("\n" + "="*30)
    print("评分统计结果 (Fitness Score Analysis)")
    print("="*30)
    print(f"处理样本总数: {len(calculated_scores)}")
    print(f"期望值 (Mean): {mean_val:.4f}")
    print(f"平均值 (Average): {mean_val:.4f}")
    print(f"标准差 (Std Dev): {std_val:.4f}")
    print(f"中位数 (Median): {median_val:.4f}")
    print(f"最高分 (Max): {max_val}")
    print(f"最低分 (Min): {min_val}")
    print("="*30)

    # 如果你想把计算出的分数保存回 CSV
    # df['calculated_fitness'] = calculated_scores
    # df.to_csv("analyzed_results.csv", index=False)


if __name__ == "__main__":
    # 请确保你的文件名正确
    data = ["dataset_acg_ost.csv",
            "dataset_classical_instrumental.csv", "dataset_pop_contemporary.csv", "dataset_total.csv"]
    process_and_analyze(data[3])
