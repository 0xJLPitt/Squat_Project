import numpy as np
import os
import argparse
import json

# 函式：讀取骨架數據的 txt 檔案
def read_skeleton_data(filename):
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            frame, joint, x, y = map(int, line.strip().split(','))
            if frame not in data:
                data[frame] = {}
            data[frame][joint] = (x, y)
    return data

# 函式：計算角度與長度
def calculate_angles_and_length(data):
    frames = sorted(data.keys())
    left_knee_angles, left_hip_angles, body_lengths = [], [], []

    for frame in frames:
        joints = data[frame]
        if all(k in joints for k in [12, 14, 16, 6, 10]):
            left_knee_angles.append(
                calculate_angle(joints[12], joints[14], joints[16]))
            left_hip_angles.append(
                calculate_angle(joints[6], joints[12], joints[16]))
            body_lengths.append(calculate_distance(joints[6], joints[12]))
        else:
            left_knee_angles.append(0)
            left_hip_angles.append(0)
            body_lengths.append(0)

    return frames, left_knee_angles, left_hip_angles, body_lengths

# 函式：計算角度
def calculate_angle(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = a - b, c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

# 函式：計算距離
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# 函式：讀取槓端數據
def read_barbell_positions(filename):
    frames, x_coords, y_coords = [], [], []
    with open(filename, 'r') as file:
        for line in file:
            frame, x, y = map(float, line.strip().split(',')[:3])
            frames.append(int(frame))
            x_coords.append(x)
            y_coords.append(y)
    return frames, x_coords, y_coords

def save_to_config(title, y_label, y_data, output_file, skeleton_frames):
    # 确保数据长度一致
    min_length = min(len(skeleton_frames), len(y_data))
    skeleton_frames = skeleton_frames[:min_length]
    y_data = y_data[:min_length]

    # 排除前后 100 帧数据计算上下限
    if min_length > 200:
        trimmed_data = y_data[100:-100]
    else:
        trimmed_data = y_data

    # 计算自动上下限
    y_min = min(trimmed_data) * 0.9
    y_max = max(trimmed_data) * 1.1

    # 构造 JSON 数据
    config_data = {
        "title": title,
        "y_label": y_label,
        "y_min": round(y_min, 2),
        "y_max": round(y_max, 2),
        "frames": skeleton_frames,
        "values": [round(v, 2) for v in y_data]
    }

    # 生成 JSON 配置文件路径
    config_path = output_file

    # 保存 JSON 文件
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)

    print(f"✅ 数据已存入 {config_path}")

def run_data_produce(dir):
    skeleton_file_path = os.path.join(dir,
                                    'interpolated_skeleton_left-front.txt')
    out = os.path.join(dir, 'config')

    skeleton_data = read_skeleton_data(skeleton_file_path)
    skeleton_frames, left_knee_angles, left_hip_angles, body_lengths = calculate_angles_and_length(
        skeleton_data)
    
    # Apply Savgol Filter to all series
    try:
        from ..Benchpress_tool.hampel import run_savgol_on_series
        left_knee_angles = run_savgol_on_series(left_knee_angles, window_length=21)
        left_hip_angles = run_savgol_on_series(left_hip_angles, window_length=21)
        body_lengths = run_savgol_on_series(body_lengths, window_length=21)
    except Exception as e:
        print(f"[DeadliftProduce] Warning: Savgol filtering failed: {e}")

    knee_to_hip_ratios = [
        left_knee / left_hip if left_hip != 0 else 0
        for left_knee, left_hip in zip(left_knee_angles, left_hip_angles)
    ]
    # Filter ratios too
    try:
        from ..Benchpress_tool.hampel import run_savgol_on_series
        knee_to_hip_ratios = run_savgol_on_series(knee_to_hip_ratios, window_length=21)
    except:
        pass



    save_to_config(title='Left Knee Angle Over Time',
                y_label='Angle (degrees)',
                y_data=left_knee_angles,
                output_file=os.path.join(out, 'Knee_Angle.json'),
                skeleton_frames=skeleton_frames)

    save_to_config(title='Left Hip Angle Over Time',
                y_label='Angle (degrees)',
                y_data=left_hip_angles,
                output_file=os.path.join(out, 'Hip_Angle.json'),
                skeleton_frames=skeleton_frames)

    save_to_config(title='Knee-to-Hip Angle Ratio Over Time',
                y_label='Ratio (Knee / Hip)',
                y_data=knee_to_hip_ratios,
                output_file=os.path.join(out, 'Knee_to_Hip.json'),
                skeleton_frames=skeleton_frames)

    save_to_config(title='Body Length Over Time',
                y_label='Length',
                y_data=body_lengths,
                output_file=os.path.join(out, 'Body_Length.json'),
                skeleton_frames=skeleton_frames)
