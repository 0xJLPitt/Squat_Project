import numpy as np
import re, json
import os
import math
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d

def calculate_angles(data):
    frames = sorted(data.keys())
    angles = []

    for frame in frames:
        joints = data[frame]
        if all(k in joints for k in [12, 14, 16]):
            a, b, c = np.array(joints[12]), np.array(joints[14]), np.array(joints[16])
            ba = a - b
            bc = c - b
            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)
            if norm_ba == 0 or norm_bc == 0:
                angle = 0
            else:
                cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
                cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
                angle = np.degrees(np.arccos(cosine_angle))
            angles.append(angle)
        else:
            angles.append(None)

    return frames, angles

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None

def read_skeleton_data(filename):
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            if len(values) < 4:
                continue
            try:
                frame, joint, x, y = map(int, values[:4])
                if frame not in data:
                    data[frame] = {}
                data[frame][joint] = (x, y)
            except ValueError:
                continue
    return data

def find_valleys(smoothed_angles,
                peaks,
                search_range=10,
                min_valley_value=170,
                min_depth=10):
    valleys = []
    valleys1 = []

    for peak in peaks:
        # 在峰值前後各 `search_range` 幀內找最小值
        left_bound = max(0, peak - search_range)
        right_bound = min(len(smoothed_angles) - 1, peak + search_range)

        # 找到左邊和右邊的波谷
        left_min_index = left_bound + np.argmin(
            smoothed_angles[left_bound:peak])
        right_min_index = peak + np.argmin(
            smoothed_angles[peak:right_bound + 1])

        left_min_value = smoothed_angles[left_min_index]
        right_min_value = smoothed_angles[right_min_index]
        peak_value = smoothed_angles[peak]

        # 檢查條件：波谷值小於 170，且谷底夠深 (峰值 - 谷底 >= min_depth)
        if left_min_value < min_valley_value and (peak_value -
                                                left_min_value) >= min_depth:
            valleys.append(left_min_index)

        if right_min_value < min_valley_value and (
                peak_value - right_min_value) >= min_depth:
            valleys1.append(right_min_index)

    return valleys, valleys1

def read_bar_data(bar_file):
    """ 讀取 yolo_coordinates_interpolated.txt，回傳 {frame: x} 字典 """
    yolo_data = {}
    with open(bar_file, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            if len(values) < 3:
                continue
            try:
                frame = int(values[0])
                x_value = float(values[1])
                y_value = float(values[2])
                yolo_data[frame] = np.round((x_value, y_value), 4)
            except ValueError:
                continue
    return yolo_data

def find_valley(left_knee_angles):
    valid_angles = np.array([angle for angle in left_knee_angles if angle is not None])

    # 平滑化角度
    # Safety check: window_length must be less than or equal to size of valid_angles and must be odd
    win_len = 11
    if len(valid_angles) < 3:
        return [], [] # Not enough data to find valleys
        
    if win_len >= len(valid_angles):
        win_len = len(valid_angles) if len(valid_angles) % 2 != 0 else len(valid_angles) - 1
    
    if win_len < 3:
        smoothed_angles = valid_angles
    else:
        smoothed_angles = savgol_filter(valid_angles, window_length=win_len, polyorder=3)

    peaks, _ = find_peaks(smoothed_angles, height=160, distance=55, prominence=5)

    # 波谷檢測
    all_valleys, all_valleys1 = find_valleys(smoothed_angles, peaks, search_range=90, min_valley_value=170, min_depth=10)

    valleys, valleys1 = [], []
    last_start, last_end = -1, -1  # 初始化前一段範圍
    
    for peak in peaks:
        left_valley = next((v for v in reversed(all_valleys) if v < peak), None)
        right_valley = next((v for v in all_valleys1 if v > peak), None)
    
        if left_valley is not None and right_valley is not None:
            new_start, new_end = left_valley, right_valley
    
            # 檢查是否和上一段有超過 50% 重疊
            if last_start != -1 and last_end != -1:
                overlap_start = max(new_start, last_start)
                overlap_end = min(new_end, last_end)
                overlap = max(0, overlap_end - overlap_start)
                len_last = last_end - last_start
                len_new = new_end - new_start
    
                if overlap / min(len_last, len_new) > 0.5:
                    print(f"跳過重疊片段：({new_start}, {new_end}) 和上一段重疊 {overlap} 幀")
                    continue  # 跳過這個片段
    
            valleys.append(new_start)
            valleys1.append(new_end)
            last_start, last_end = new_start, new_end  # 更新上一段範圍
    return valleys, valleys1

def adjust_valleys_with_bar_data(path, bar_data, knee_angles):
    """
    根據 valleys (黃色谷底) 和 valleys1 (綠色谷底) 分割槓鈴數據，
    **只存 Y 座標變化超過 20 的片段**，如果變化 ≤ 20，則不存。
    """
    save_path = os.path.join(path, 'config', 'Split_info.json')
    valleys, valleys1 = find_valley(knee_angles)
    data = {}
    reps = {}
    k = 0
    for i, (start_frame, end_frame) in enumerate(zip(valleys, valleys1)):
        if start_frame >= end_frame:
            print(f"跳過無效範圍: 開始 {start_frame}, 結束 {end_frame}")
            continue

        # 計算該片段內的 Y 座標變化範圍
        y_values = [
            bar_data[f][1] for f in range(start_frame, end_frame + 1)
            if f in bar_data
        ]
        y_range = max(y_values) - min(y_values) if y_values else 0

        # **如果變化小於等於 20，則跳過該片段**
        if y_range <= 50 or np.isnan(y_range):
            print(f"跳過槓鈴數據 {i+1}（Y 變化範圍: {y_range:.2f}，過小）")
            continue
        
        split_info = {}
        split_info['start'] = int(start_frame)
        split_info['end'] = int(end_frame)
        data[str(k)] = split_info
        k += 1

        reps[i] = (int(start_frame), int(end_frame))
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("所有槓鈴數據處理完成！")
    return reps

def split_skeleton_data(skeleton_path, reps):
    """
    根據 valleys (黃色谷底) 和 valleys1 (綠色谷底) 分割骨架數據，
    **只存 Y 座標變化超過 20 的片段**，如果變化 <= 20，則不存。
    """
    skeleton_info = {}
    with open(skeleton_path, 'r') as skel_file:
        lines = skel_file.readlines()
    for i, rep in reps.items():
        for line in lines:
            values = line.strip().split(',')
            if len(values) < 4:
                continue
            if rep[0] <= int(values[0]) <= rep[1]:
                if int(values[0]) not in skeleton_info:
                    skeleton_info[int(values[0])] = {}
                skeleton_info[int(values[0])][int(values[1])] = (float(values[2]), float(values[3]))
    return skeleton_info

def calculate_distance(x1, y1, x2, y2):
    """計算兩點之間的歐氏距離"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calculate_angle1(x1, y1, x2, y2, x3, y3):
    """計算由三點構成的夾角，x2, y2 是頂點"""
    c = calculate_distance(x2, y2, x3, y3)
    a = calculate_distance(x1, y1, x3, y3)
    b = calculate_distance(x1, y1, x2, y2)

    if b == 0 or c == 0:
        return 0.0

    try:
        cos_theta = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle = math.acos(cos_theta)
        return math.degrees(angle)
    except ValueError:
        return 0.0
    
def interpolate_features(features, target_length):
    features = np.array(features)  # shape: (original_len, feature_dim)
    if features.ndim < 2 or features.shape[0] < 2:
        # Return zeros if not enough data
        feature_dim = features.shape[1] if features.ndim == 2 else 1
        return np.zeros((target_length, feature_dim))
        
    original_len, feature_dim = features.shape


    # 原本的時間軸 (ex: 0, 1, 2, ..., original_len-1)
    x_original = np.linspace(0, 1, original_len)
    # 目標的時間軸 (ex: 0, 1, 2, ..., target_len-1)
    x_target = np.linspace(0, 1, target_length)

    interpolated = []

    for i in range(feature_dim):
        f = interp1d(x_original, features[:, i], kind='linear')
        interpolated.append(f(x_target))
    
    filtered_interpolated = remove_outliers(np.stack(interpolated, axis=-1))
    return np.round(filtered_interpolated,4)

def remove_outliers(features):
    # 計算每個特徵維度的 mean 和 std
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)

    # 找出異常值位置（超過 ±3 std）
    outliers = np.abs(features - mean) > 3 * std

    # 建立一個複製來避免改動原始資料（如不需保留原本資料可略過）
    features_clean = features.copy()

    # 用 mean 替換異常值
    features_clean[outliers] = np.tile(mean, (features.shape[0], 1))[outliers]
    return features
