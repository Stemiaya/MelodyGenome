# MelodyGenome
运用遗传算法进行机器编曲

## 项目结构
```
MelodyGenome
│  dataset.py
│  dataset_acg_ost.csv
│  dataset_classical_instrumental.csv
│  dataset_pop_contemporary.csv
│  fitnessModel.py
│  fitness_evaluator.py
│  generated_music.csv
│  main.py
│  README.md
│  seq2midi.py
│  test.py
│  train.py
│  
├─checkpoints
│  │  acg_model.pth
│  │  classical_model.pth
│  │  combined_model.pth
│  │  pop_model.pth
│  │  
│  ├─acg_model_utils
│  │      key_encoder.pkl
│  │      
│  ├─classical_model_utils
│  │      key_encoder.pkl
│  │      
│  ├─combined_model_utils
│  │      key_encoder.pkl
│  │      
│  └─pop_model_utils
│          key_encoder.pkl
│          
└──output_midi
```

## 项目文件介绍
**README.md**: 本文件;

**main.py**: 遗传算法主要程序;

**seq2midi.py**: 将生成的序列转化为音频文件的程序;

**dataset.py**: 数据集选择程序;

**fitness_evaluator.py**: fitness函数评估器程序;

**fitnessModel.py**: 训练模型的层次架构程序;

**train.py**: 训练模型的程序;

**generated_music.csv**: 遗传算法程序生成的音乐序列;

**dataset_acg_ost.csv**, **dataset_classical_instrumental.csv**, **dataset_pop_contemporary.csv**: 算法使用的数据库;

**output_midi**: 生成的音频文件文件夹，内含多个**gen_XX.mid**音频文件;

**checkpoints**: 训练的模型文件夹，内含一次训练中四个模型的最佳参数.

## 使用说明
- 在该目录下打开终端，输入`python train.py`进行模型训练，训练好的模型会存放在**checkpoints**文件夹中
- **train.py** 支持输入参数进行特定模型的训练，要进行此操作，只需输入`python train.py --target <type>`:
  - `target`参数支持输入`acg`, `pop`, `classical`, `combined`, `all`, 分别对应数据库选择中的 **acg_ost** , **classical_instrumental** , **pop_contemporary** , **随机所有数据**, 以及 **所有数据** ，无输入默认为`combined`.
- 输入`python main.py`进行乐曲生成，生成好的乐曲会直接转化为 **X.mid** 文件存放在 **output_midi** 文件夹中
- **main.py** 支持输入参数进行特定规则的生成，要进行此操作，只需输入`python main.py --seed <seed_num> --gen <generation_num> --pop_size <population_size> --mode<type> --model <type> --output <output_file>`:
  - `seed`参数支持输入一个整数进行固定种子选取初始数据库;
  - `gen`参数支持输入一个整数作为遗传算法的子代数量，默认值为`100`;
  - `pop_size`参数支持输入一个整数作为种群数量，默认值为`100`;
  - `mode`参数支持输入`rule`, `ai`, `hybrid`作为适应度函数的选择，其分别对应 **基于规则的** , **基于AI训练的** , 以及 **基于两者混用的** ，无输入默认为`hybrid`;
  - `model`参数支持输入`acg`, `pop`, `classical`, `combined`作为训练模型的选择，其分别对应训练时所产生的 **四种** 模型，无输入默认为`combined`;
  - `output`参数支持输入一个文件名作为音乐序列的输出，默认为 **generated_music.csv** .
