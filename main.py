import pandas as pd
import numpy as np
import random
import ast
from fitnessEvaluator import FitnessEvaluator
import argparse
from seq2midi import vector_to_midi


class MelodyValidator:
    @staticmethod
    def repair_melody(melody):
        # 强制修复旋律中的不合法逻辑
        # 1. 修复首位禁令：如果第一位是13，强制随机变更为一个音符1-12
        if melody[0] == 13:
            melody[0] = random.randint(1, 12)

        # 2. 修复悬空延长：13之前不能是0
        for i in range(1, len(melody)):
            if melody[i] == 13:
                if melody[i-1] == 0:
                    # 如果休止符后面跟着延长，将延长改为休止符或前一个有效音符的延续
                    melody[i] = 0

        return melody


class MelodyGA:
    def __init__(self, csv_files, pop_size=100, mutation_rate=0.08, transform_rate=0.3,
                 seed=42, fitness_mode='rule', ai_model_type='combined'):
        """
        fitness_mode: 
            - 'rule': 仅使用硬编码乐理规则 
            - 'ai': 仅使用神经网络评分
            - 'hybrid': 规则分 + AI分 
        ai_model_type: 'classical', 'acg', 'pop', 'combined'
        """
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.transform_rate = transform_rate
        self.genome_length = 64
        self.seed = seed
        self.pool = self._load_data(csv_files)

        self.fitness_mode = fitness_mode
        self.possible_keys = ['C', 'D', 'E', 'F', 'G',
                              'A', 'B', 'Bb', 'Eb']  # 用于AI评估时的随机Key

        # ✨ 如果模式涉及AI，则初始化评估器
        self.ai_evaluator = None
        if fitness_mode in ['ai', 'hybrid']:
            try:
                self.ai_evaluator = FitnessEvaluator(model_type=ai_model_type)
                print(f"✅ AI模型 ({ai_model_type}) 加载成功。")
            except Exception as e:
                print(f"⚠️ AI模型加载失败: {e}。自动回退到 'rule' 模式。")
                self.fitness_mode = 'rule'

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
                raise ValueError("使用seeded模式必须在初始化时提供seed值")

            # 创建一个局部的随机发生器，确保采样过程可重复
            local_rng = random.Random(self.seed)
            # 从数据库中选出固定的样本
            sample_vectors = local_rng.sample(self.pool, self.pop_size)

            for vec in sample_vectors:
                population.append(MelodyValidator.repair_melody(vec.copy()))

            print(f"--- 初始种群初始化：使用固定种子 [{self.seed}] 采样 ---")

        else:
            # 纯随机选择
            for _ in range(self.pop_size):
                raw_vec = random.choice(self.pool).copy()
                population.append(MelodyValidator.repair_melody(raw_vec))

            print("--- 初始种群初始化：纯随机采样 ---")

        return population

    def _calculate_rule_score(self, individual):
        """原有的基于乐理规则的评分"""
        score = 0
        major_scale = [1, 3, 5, 6, 8, 10, 12]
        chord_tones = [1, 5, 8]
        genome_len = len(individual)
        notes = [(individual[i], i)
                 for i in range(genome_len) if 1 <= individual[i] <= 12]

        if len(notes) < 8:
            return 1

        # 1. 节奏约束
        rhythm_penalty = 0
        for i in range(0, genome_len, 2):
            if i+1 < genome_len and individual[i+1] != 13 and individual[i+1] != individual[i] and individual[i+1] != 0:
                rhythm_penalty += 15
        score -= rhythm_penalty

        # 2. 强拍和弦音
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

    # ✨fitnessModel评分函数
    def _calculate_ai_score(self, individual):
        """调用神经网络模型进行评分"""
        key = random.choice(self.possible_keys)

        try:
            # 模型输出通常在 0-100 之间 (取决于你的训练标签)
            score = self.ai_evaluator.get_fitness(individual, key)
            return score, key
        except Exception as e:
            # 如果出错，返回保底分
            print(f"AI Eval Error: {e}")
            return 0

    # 主适应度函数-根据模式调度
    def calculate_fitness(self, individual, return_key=False):
        score = 0
        used_key = 'C'

        if self.fitness_mode == 'rule':
            score = self._calculate_rule_score(individual)

        elif self.fitness_mode == 'ai':
            s, k = self._calculate_ai_score(individual)
            score = s
            used_key = k

        elif self.fitness_mode == 'hybrid':
            rule_score = self._calculate_rule_score(individual)
            ai_score, k = self._calculate_ai_score(individual)
            used_key = k

            if rule_score <= 10:
                score = 1
            else:
                # 加权求和: 40% 规则 + 60% AI
                final_score = (rule_score * 0.4) + (ai_score * 0.6)
                score = max(final_score, 1)

        if return_key:
            return score, used_key
        return score

    # --- 遗传操作 (Genetic Operators) ---

    def crossover(self, parent1, parent2):
        cp1 = random.randint(10, 25)
        cp2 = random.randint(35, 50)

        raw_new_g1 = parent1[:cp1] + parent2[cp1:cp2] + parent1[cp2:]
        raw_new_g2 = parent2[:cp1] + parent1[cp1:cp2] + parent2[cp2:]

        # 交叉后立即修复
        repaired_g1 = MelodyValidator.repair_melody(raw_new_g1)
        repaired_g2 = MelodyValidator.repair_melody(raw_new_g2)

        return repaired_g1, repaired_g2

    def mutate(self, individual):
        """常规变异：随机改变某个音符"""
        for i in range(self.genome_length):
            if random.random() < self.mutation_rate:
                # 保持节奏结构，只变异音高
                if 1 <= individual[i] <= 12:
                    individual[i] = random.randint(1, 12)

        repaired_individual = MelodyValidator.repair_melody(individual)
        return repaired_individual

    def musical_transform(self, individual):
        """音乐理论变换"""
        op = random.random()

        # 1. 移调 (Transposition)
        if op < 0.25:
            shift = random.choice([-2, -1, 1, 2])
            individual_1 = [(n + shift - 1) % 12 + 1 if 1 <=
                            n <= 12 else n for n in individual]

        # 2. 逆行 (Retrograde)
        elif op < 0.5:
            # 简单翻转，但需要注意延音13的处理（简化处理：直接反转）
            individual_1 = individual[::-1]

        # 3. 倒影 (Inversion)
        elif op < 0.75:
            # 以第一音为轴进行音程翻转
            first_note = next((n for n in individual if 1 <= n <= 12), 6)
            individual_1 = [max(1, min(12, 2*first_note - n))
                            if 1 <= n <= 12 else n for n in individual]

        # 4. 逆行倒影 (RI)
        else:
            # --- 倒影 (Inversion) ---
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

            # --- 逆行 (Retrograde) ---
            individual_1 = inverted[::-1]

        repaired_individual = MelodyValidator.repair_melody(individual_1)

        return repaired_individual

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

            # 选择前50%
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

        final_results = []

        for ind in population:
            # 这里调用带 return_key=True 的计算
            fitness, best_key = self.calculate_fitness(ind, return_key=True)

            final_results.append({
                'genes': ind,
                'fitness': fitness,
                'key': best_key
            })

        # 排序
        final_results.sort(key=lambda x: x['fitness'], reverse=True)

        # 返回前30个字典对象
        return final_results[:30]


def save_results_to_csv(results, ga_instance, output_filename="generated_melodies.csv"):
    """
    将生成的旋律格式化并保存到CSV文件中
    """
    output_data = []

    for i, item in enumerate(results):
        # item 是一个字典: {'genes': [...], 'fitness': 88.5, 'key': 'F#'}
        melody = item['genes']
        fitness_score = item['fitness']
        key = item['key']

        # 构造数据，匹配原CSV表头
        row = {
            'score_id': f"gen_{i+1:02d}",      # 生成编号，如 gen_01
            'key': key,  # 默认为主调（1-12 relative）
            'segment_start': 0.0,              # 生成片段默认从0开始
            'melody_vector': str(melody),      # 转换为字符串存储，以便CSV兼容
            'final_score': fitness_score       # 记录适应度分数
        }
        output_data.append(row)

    # 转换为 DataFrame 并导出
    df_output = pd.DataFrame(output_data)
    df_output.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"成功导出 {len(results)} 条旋律至: {output_filename}")


files = ['./data/dataset_acg_ost.csv', './data/dataset_classical_instrumental.csv',
         './data/dataset_pop_contemporary.csv']


if __name__ == "__main__":
    # 1. 定义命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Melody Generation with Genetic Algorithm & AI")

    # 参数: 进化代数
    parser.add_argument('--gens', type=int, default=100,
                        help='Number of generations (default: 100)')

    # 参数: 种群大小
    parser.add_argument('--pop_size', type=int, default=100,
                        help='Population size (default: 100)')

    # 参数: 适应度模式
    parser.add_argument('--mode', type=str, default='hybrid',
                        choices=['rule', 'ai', 'hybrid'],
                        help="Fitness mode: 'rule' (Music Theory), 'ai' (Neural Net), or 'hybrid' (Both)")

    # 参数: AI模型 (关键参数!)
    parser.add_argument('--model', type=str, default='combined',
                        choices=['classical', 'acg', 'pop', 'combined'],
                        help="Which AI model to use for evaluation")

    # 参数: 输出文件名
    parser.add_argument('--output', type=str, default='generated_music.csv',
                        help="Output CSV filename")

    # 参数: 随机种子
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # 2. 打印当前配置
    print("\n" + "="*40)
    print(f"🎵 Melody Generator Configuration")
    print(f"{'='*40}")
    print(f"  - Generations:  {args.gens}")
    print(f"  - Pop Size:     {args.pop_size}")
    print(f"  - Fitness Mode: {args.mode.upper()}")
    if args.mode in ['ai', 'hybrid']:
        print(f"  - AI Model:     {args.model.upper()}")
    else:
        print(f"  - AI Model:     Disabled (Rule Only)")
    print(f"  - Output File:  {args.output}")
    print(f"  - Seed:         {args.seed if args.seed else 'Random'}")
    print("="*40 + "\n")

    # 3. 初始化遗传算法
    files = ['./data/dataset_acg_ost.csv', './data/dataset_classical_instrumental.csv',
             './data/dataset_pop_contemporary.csv']

    try:
        ga = MelodyGA(
            csv_files=files,
            pop_size=args.pop_size,
            seed=args.seed,
            fitness_mode=args.mode,    # 传入命令行参数
            ai_model_type=args.model   # 传入命令行参数
        )

        # 4. 开始进化
        results = ga.evolve(generations=args.gens)

        # 5. 保存结果
        save_results_to_csv(results, ga, args.output)

        vector_to_midi(args.output)

    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        # 如果是没找到模型文件，给个提示
        if "No such file" in str(e) or "not found" in str(e):
            print("提示: 请先运行 'python train.py' 训练模型，或检查 checkpoints 文件夹。")
