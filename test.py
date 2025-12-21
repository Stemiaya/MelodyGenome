import random
from fitnessEvaluator import FitnessEvaluator


def test_model_manually():
    print("========================================")
    print("      Music Model Inference Test        ")
    print("========================================")

    # 1. 选择要测试的模型类型：'classical', 'acg', 'pop', 'combined'
    TARGET_MODEL = 'pop'

    try:
        # 初始化评估器 (这会加载训练好的文件)
        evaluator = FitnessEvaluator(model_type=TARGET_MODEL)
    except Exception as e:
        print(f"\nCRITICAL ERROR: 初始化失败。\n{e}")
        return

    # 2. 造一些假数据来测试
    print("\n--- Generating Dummy Data ---")

    # 案例 A: 随机乱弹 (分数应该低或者是随机的)
    random_melody = [random.randint(0, 13) for _ in range(64)]

    # 案例 B: 全是休止符 (分数应该很低)
    silent_melody = [1] * 64

    # 案例 C: 简单的爬音阶 (模拟好一点的旋律)
    scale_melody = [1, 3, 5, 6, 8, 10, 12, 13] * 8

    test_cases = [
        ("Random Noise", random_melody, "C"),
        ("Silence", silent_melody, "D#"),
        ("Simple Scale", scale_melody, "Am")  # 测试一个不常见的调号
    ]

    # 3. 运行测试
    print("\n--- Running Inference ---")
    for name, melody, key in test_cases:
        try:
            score = evaluator.get_fitness(melody, key)
            print(f"Name: {name:<15} | Key: {key:<3} | Score: {score:.4f}")
        except Exception as e:
            print(f"Name: {name:<15} | Error: {e}")

    print("\n✅ Test Finished. If you see scores, everything works!")


if __name__ == "__main__":
    test_model_manually()
