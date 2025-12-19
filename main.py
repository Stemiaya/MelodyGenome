import pandas as pd
import numpy as np
import random
import ast


class MelodyValidator:
    @staticmethod
    def repair_melody(melody):
        """
        强制修复旋律中的不合法逻辑
        """
        # 1. 修复首位禁令：如果第一位是13，强制随机变更为一个音符1-12
        if melody[0] == 13:
            melody[0] = random.randint(1, 12)

        # 2. 修复悬空延长：13之前不能是0
        for i in range(1, len(melody)):
            if melody[i] == 13:
                if melody[i-1] == 0:
                    # 如果休止符后面跟着延长，将延长改为休止符或前一个有效音符的延续
                    melody[i] = 0

        # 3. 强制八分音符约束（实验特殊要求）：
        # 每两个单位（0.25+0.25）必须属于同一个音符逻辑
        for i in range(0, len(melody), 2):
            # 如果偶数位i是音符，奇数位i+1必须是13或与i相同
            if 1 <= melody[i] <= 12:
                if melody[i+1] != 13 and melody[i+1] != melody[i]:
                    melody[i+1] = 13  # 强制延长，确保形成八分音符
            # 如果偶数位是休止，奇数位也必须是休止
            if melody[i] == 0:
                melody[i+1] = 0

        return melody


class MelodyGA:
    def __init__(self, csv_files, pop_size=100, mutation_rate=0.08, transform_rate=0.3, seed=None):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.transform_rate = transform_rate
        self.genome_length = 64
        self.seed = seed
        self.pool = self._load_data(csv_files)

    def _load_data(self, csv_files):
        """合并CSV并解析数据"""
        combined_df = pd.concat([pd.read_csv(f) for f in csv_files])
        # 确保只加载合法的向量数据
        return combined_df['melody_vector'].apply(ast.literal_eval).tolist()

    def init_population(self, mode="random"):
        """
        初始化种群
        mode: "random" - 纯随机选择（每次运行结果不同）
              "seeded" - 固定种子选择（每次运行根据同一个种子选出相同的数据）
        """
        population = []

        if mode == "seeded":
            if self.seed is None:
                raise ValueError("使用seeded模式必须在初始化时提供seed值（如: seed=42）")

            # 创建一个局部的随机发生器，确保采样过程可重复
            local_rng = random.Random(self.seed)
            # 从数据库中选出固定的样本
            sample_vectors = local_rng.sample(self.pool, self.pop_size)

            for vec in sample_vectors:
                # 即使是固定样本，我们也需要修复它以确保符合“禁令”规则
                population.append(MelodyValidator.repair_melody(vec.copy()))

            print(f"--- 初始种群初始化：使用固定种子 [{self.seed}] 采样 ---")

        else:
            # 模式一：纯随机选择（默认模式）
            for _ in range(self.pop_size):
                raw_vec = random.choice(self.pool).copy()
                population.append(MelodyValidator.repair_melody(raw_vec))

            print("--- 初始种群初始化：纯随机采样 ---")

        return population

    # --- 适应度函数 (Fitness Function) ---
    def calculate_fitness(self, individual):
        """
        基于 Özcan & Erçal 论文改进的适应度函数
        评价维度：调性、强拍和弦音、音程平滑度、终止式、节奏约束、丰富度
        """
        score = 0

        # 基础参数定义
        major_scale = [1, 3, 5, 6, 8, 10, 12]  # 大调音阶
        chord_tones = [1, 5, 8]               # 主三和弦音 (C, E, G)
        genome_len = len(individual)

        # 1. 预处理：提取有效音符序列（排除休止符0和延音13）
        # notes 存储格式: (音高, 索引位置)
        notes = [(individual[i], i)
                 for i in range(genome_len) if 1 <= individual[i] <= 12]

        if len(notes) < 8:
            return 1  # 音符太少，视为无效旋律

        # --- 维度 1: 节奏约束 (Rhythm Constraint / f7) ---
        # 实验要求：最短音符为八分音符。即偶数位置不能出现新音符改变（除非是延音或休止）
        rhythm_penalty = 0
        for i in range(0, genome_len, 2):
            # 如果在一个八分音符的时值内(两个0.25单位)，第二个单位变成了另一个不同的音符
            if individual[i+1] != 13 and individual[i+1] != individual[i] and individual[i+1] != 0:
                rhythm_penalty += 15
        score -= rhythm_penalty

        # --- 维度 2: 强拍和弦音 (Chord Note Compatibility / f1) ---
        # 强拍位置：0, 4, 8, 12... (每个四分音符的开始)
        strong_beat_reward = 0
        for pos in range(0, genome_len, 4):
            note = individual[pos]
            if note in chord_tones:
                strong_beat_reward += 10  # 强拍落在和弦音上加分
            elif note in major_scale:
                strong_beat_reward += 5  # 强拍落在音阶内加分
        score += strong_beat_reward

        # --- 维度 3: 起止音约束 (Beginning/Ending / f4, f5) ---
        # 首音奖励
        if notes[0][0] in chord_tones:
            score += 15
        # 末音奖励：强烈建议回到主音1
        if notes[-1][0] == 1:
            score += 30
        elif notes[-1][0] in chord_tones:
            score += 15

        # --- 维度 4: 音程跳跃与平滑度 (Intervals & Relationship / f2, f6) ---
        interval_score = 0
        direction_changes = 0
        prev_direction = 0  # 1 为上行，-1 为下行

        for i in range(len(notes) - 1):
            pitch1, pos1 = notes[i]
            pitch2, pos2 = notes[i+1]
            diff = abs(pitch2 - pitch1)

            # 论文建议：级进(1-2度)和小跳(3-4度)最悦耳
            if diff == 0:
                interval_score += 2   # 适当重复
            elif 1 <= diff <= 2:
                interval_score += 8   # 级进（高分奖励）
            elif 3 <= diff <= 4:
                interval_score += 5   # 小跳
            elif diff > 7:
                interval_score -= 20  # 大于五度的跳进严厉惩罚

            # 维度 5: 方向性控制 (Direction / f3)
            current_direction = 1 if pitch2 > pitch1 else -1 if pitch2 < pitch1 else 0
            if current_direction != 0 and prev_direction != 0 and current_direction != prev_direction:
                direction_changes += 1
            prev_direction = current_direction

        # 惩罚过于频繁的方向改变（锯齿状旋律）
        if direction_changes > len(notes) / 3:
            interval_score -= 15

        score += interval_score

        # --- 维度 6: 丰富度/多样性 (Variety - 针对你之前全1的问题) ---
        unique_notes = len(set([n[0] for n in notes]))
        if unique_notes < 3:
            score -= 50  # 音符太单一（如全是1）大幅扣分
        else:
            score += unique_notes * 5  # 鼓励使用更多不同的音符

        return max(score, 1)

    # --- 遗传操作 (Genetic Operators) ---

    def crossover(self, parent1, parent2):
        """双点交叉"""
        cp1 = random.randint(10, 25)
        cp2 = random.randint(35, 50)
        child1 = parent1[:cp1] + parent2[cp1:cp2] + parent1[cp2:]
        child2 = parent2[:cp1] + parent1[cp1:cp2] + parent2[cp2:]
        return child1, child2

    def mutate(self, individual):
        """常规变异：随机改变某个音符"""
        for i in range(self.genome_length):
            if random.random() < self.mutation_rate:
                # 保持节奏结构，只变异音高
                if 1 <= individual[i] <= 12:
                    individual[i] = random.randint(1, 12)
        return individual

    def musical_transform(self, individual):
        """音乐理论变换"""
        op = random.random()

        # 1. 移调 (Transposition)
        if op < 0.3:
            shift = random.choice([-2, -1, 1, 2])
            return [(n + shift - 1) % 12 + 1 if 1 <= n <= 12 else n for n in individual]

        # 2. 逆行 (Retrograde)
        elif op < 0.5:
            # 简单翻转，但需要注意延音13的处理（简化处理：直接反转）
            return individual[::-1]

        # 3. 倒影 (Inversion)
        elif op < 0.7:
            # 以第一音为轴进行音程翻转
            first_note = next((n for n in individual if 1 <= n <= 12), 6)
            return [max(1, min(12, 2*first_note - n)) if 1 <= n <= 12 else n for n in individual]

        else:
            # --- 第一步：倒影 (Inversion) ---
            # 找到第一个有效音符作为镜像轴
            first_note = next((n for n in individual if 1 <= n <= 12), 6)

            inverted = []
            for n in individual:
                if 1 <= n <= 12:
                    # 镜像计算：new_pitch = 2 * axis - old_pitch
                    # 结果使用 (x-1)%12 + 1 确保落在 1-12 范围内
                    new_pitch = (2 * first_note - n - 1) % 12 + 1
                    inverted.append(new_pitch)
                else:
                    inverted.append(n)  # 0 和 13 保持位置不变

            # --- 第二步：逆行 (Retrograde) ---
            # 将倒影后的序列完全反转
            ri_melody = inverted[::-1]
            return individual

    # --- 进化主程序 ---

    def evolve(self, generations=100):
        # 函数参数选取:
        # "seeded": 使用固定种子产生初始种群, 种子须在运行时输入
        # "random": 使用随机种子产生初始种群
        population = self.init_population("seeded")

        for gen in range(generations):
            # 评分与选择
            scored_pop = [(self.calculate_fitness(ind), ind)
                          for ind in population]
            scored_pop.sort(key=lambda x: x[0], reverse=True)

            # 选择前50%作为精英
            elites = [ind for score, ind in scored_pop[:self.pop_size // 2]]

            next_generation = []
            while len(next_generation) < self.pop_size:
                p1, p2 = random.sample(elites, 2)

                # 交叉
                c1, c2 = self.crossover(p1, p2)

                # 变异
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)

                # 音乐变换
                if random.random() < self.transform_rate:
                    c1 = self.musical_transform(c1)

                next_generation.extend([c1, c2])

            population = next_generation[:self.pop_size]

            if gen % 10 == 0:
                print(f"Generation {gen}: Best Fitness = {scored_pop[0][0]}")

        # 返回最后一代中表现最好的 15 条
        final_scored = [(self.calculate_fitness(ind), ind)
                        for ind in population]
        final_scored.sort(key=lambda x: x[0], reverse=True)
        return [ind for score, ind in final_scored[:30]]


def save_results_to_csv(results, ga_instance, output_filename="generated_melodies.csv"):
    """
    将生成的旋律格式化并保存到CSV文件中
    """
    output_data = []

    for i, melody in enumerate(results):
        # 计算该旋律的最终适应度得分
        possible_keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'Bb', 'Eb']
        fitness_score = ga_instance.calculate_fitness(melody)

        # 构造一行数据，匹配原CSV表头
        row = {
            'score_id': f"gen_{i+1:02d}",      # 生成编号，如 gen_01
            'key': random.choice(possible_keys),  # 默认为主调（1-12 relative）
            'segment_start': 0.0,              # 生成片段默认从0开始
            'melody_vector': str(melody),      # 转换为字符串存储，以便CSV兼容
            'final_score': fitness_score       # 记录适应度分数
        }
        output_data.append(row)

    # 转换为 DataFrame 并导出
    df_output = pd.DataFrame(output_data)
    df_output.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"成功导出 {len(results)} 条旋律至: {output_filename}")


files = ['dataset_acg_ost.csv', 'dataset_classical_instrumental.csv',
         'dataset_pop_contemporary.csv']

# 若此处想要使用固定种子, 须在evolve函数中修改
ga = MelodyGA(files, seed=2024)
results = ga.evolve(generations=150)

save_results_to_csv(results, ga, "generated_music.csv")
