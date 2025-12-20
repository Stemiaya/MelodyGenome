import pandas as pd
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import ast


def key_to_base_midi(key_str):
    """将调号映射为 MIDI 基础音高 (第4八度)"""
    key_map = {
        'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63,
        'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68,
        'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71
    }
    return key_map.get(key_str.strip(), 60)


def vector_to_midi(csv_path, output_dir="output_midi"):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取 CSV
    df = pd.read_csv(csv_path)

    # MIDI 刻度设置 (假设 480 ticks 为一拍，0.25单位即120 ticks)
    ticks_per_unit = 120

    for index, row in df.iterrows():
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        # 设置速度 (120 BPM)
        track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))

        # 解析数据
        score_id = row['score_id']
        base_pitch = key_to_base_midi(row['key'])
        # 兼容字符串格式的列表
        melody = ast.literal_eval(row['melody_vector']) if isinstance(
            row['melody_vector'], str) else row['melody_vector']

        i = 0
        while i < len(melody):
            val = melody[i]

            # 1. 处理休止符 (0)
            if val == 0:
                # 寻找休止符持续了多久
                duration = ticks_per_unit
                j = i + 1
                while j < len(melody) and melody[j] == 0:
                    duration += ticks_per_unit
                    j += 1
                # MIDI 中休止符表现为下一个 Note On 的 time 偏移，所以这里只记录位置
                i = j
                continue

            # 2. 处理音符 (1-12)
            if 1 <= val <= 12:
                pitch = base_pitch + (val - 1)

                # 计算音符持续时间（看后面有多少个 13）
                duration = ticks_per_unit
                j = i + 1
                while j < len(melody) and melody[j] == 13:
                    duration += ticks_per_unit
                    j += 1

                # 计算自上一个事件以来的时间偏移 (处理之前的休止符)
                # 简化处理：每个事件紧跟上一个
                time_offset = 0
                if i > 0 and melody[i-1] == 0:
                    # 如果前一个是休止符，计算回溯
                    rest_count = 0
                    k = i - 1
                    while k >= 0 and melody[k] == 0:
                        rest_count += 1
                        k -= 1
                    time_offset = rest_count * ticks_per_unit

                # 写入 MIDI
                track.append(Message('note_on', note=pitch,
                             velocity=80, time=time_offset))
                track.append(Message('note_off', note=pitch,
                             velocity=80, time=duration))

                i = j
            else:
                # 跳过异常值（如开头的13）
                i += 1

        # 保存文件
        filename = f"{output_dir}/{score_id}.mid"
        mid.save(filename)
        print(f"Saved: {filename}")


# --- 执行转换 ---
# 假设你的文件名为 generated_results.csv
vector_to_midi("generated_music.csv")
