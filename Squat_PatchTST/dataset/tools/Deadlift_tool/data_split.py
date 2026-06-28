import numpy as np
import os
import math
import numpy as np
from .utils import *

def process_skeleton2angle(skeleton_data, point):
    features = {}
    for frame, joints in sorted(skeleton_data.items()):
        if all(idx in joints for idx in point):
            x0, y0 = joints[point[0]]
            x1, y1 = joints[point[1]]
            x2, y2 = joints[point[2]]
            x3, y3 = joints[point[3]]

            hip_angle = calculate_angle1(x0, y0, x1, y1, x2, y2)
            knee_angle = calculate_angle1(x1, y1, x2, y2, x3, y3)
            features[frame] = np.round((hip_angle, knee_angle), 2)

    print(f"✅ 已處理：left-front vision features extraction")
    return features

def process_bar_vision(skeleton_data, bar_data):
    features_bar = {}
    for i in range(min(len(bar_data), len(skeleton_data))):
        # 嘗試對應同一個 frame（假設每一行對應一個 frame）
        frame = list(sorted(skeleton_data.keys()))[i]
        keypoints = skeleton_data[frame]

        if 5 in keypoints and 6 in keypoints and 12 in keypoints:
            shoulder_x = keypoints[5][0]
            bar_x = float(bar_data[frame][0])
            diff = shoulder_x - bar_x

            x6, y6 = keypoints[6]
            x12, y12 = keypoints[12]
            dist_6_12 = math.sqrt((x12 - x6) ** 2 + (y12 - y6) ** 2)

            features_bar[frame] = np.round((diff, dist_6_12), 4)
    print(f"✅ 已處理：bar vision feartures extraction")
    return features_bar


def merge_and_interpolate(reps, features_left_front, bar_data, features_bar, features_left_back, target_length=110):
    """合併 features"""
    # 找出四者都有的檔案
    common_keys = set(features_left_front.keys()) & set(bar_data.keys()) & set(features_bar.keys()) & set(features_left_back.keys())
    split_features = {}
    filtered_interpolated = {}
    all_features = {}
    
    for key in sorted(common_keys):
        all_features[key] = np.concatenate([features_left_front[key], bar_data[key], features_bar[key], features_left_back[key]])
    print(f"✅ 已合併特徵數據，共 {len(all_features)} 個 frame。")
    
    for i, rep in reps.items():
        split_features[i] = [
            all_features[f] for f in range(rep[0], rep[1] + 1)
            if f in all_features
        ]
    for i, feature in split_features.items():
        filtered_interpolated[i] = interpolate_features(feature, target_length)
    print(f"✅ 已分割特徵數據，共 {len(filtered_interpolated)} 個片段。")
    # for i, data in filtered_interpolated.items():
    #     print(i, ":", data)
    return filtered_interpolated

def process_delta(filtered_interpolated):
    delta_feature = {}
    for i, feature in filtered_interpolated.items():
        # 計算每一列與前一列的變化量
        arr = np.array(feature)
        delta = np.diff(arr, axis=0)
        delta_feature[i] = np.vstack([np.zeros(arr.shape[1]), delta])
    return delta_feature

def process_delta_ratio(filtered_interpolated):
    """計算變化量與原始值的比 (B - A) / A"""
    delta_ratio_feature = {}
    for i, data in filtered_interpolated.items():
        epsilon = 1e-6  # 避免 A 為 0
        delta_ratio = (data[1:] - data[:-1]) / (data[:-1] + epsilon)
        delta_ratio = np.vstack([np.zeros((1, data.shape[1])),
                                delta_ratio])  # 第一行補 0
        delta_ratio_feature[i] = delta_ratio
        
    return delta_ratio_feature

def z_score_normalization(df):
    """對 DataFrame 進行 Z-score 標準化，每一列 (特徵) 依據自身均值與標準差計算。"""
    mean = df.mean()
    std = df.std()

    std.replace(0, 1, inplace=True)  # 避免標準差為 0 導致除以 0
    zscore_df = (df - mean) / std
    return zscore_df


def process_zscore(filtered_interpolated):
    zscore = {}
    for i, data in filtered_interpolated.items():
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std == 0] = 1  # 避免除以 0
        zscore[i] = np.round((data - mean) / std, 4)
    return zscore

def normalize_to_neg1_1(arr):
    """
    將 numpy 2D array 正規化到 [-1, 1]。
    以每個欄位（feature）為單位縮放。
    """
    min_vals = np.min(arr, axis=0)
    max_vals = np.max(arr, axis=0)
    
    # 避免除以 0：max == min 時，設為 0
    scale = max_vals - min_vals
    scale[scale == 0] = 1

    normalized = 2 * (arr - min_vals) / scale - 1
    return normalized


def process_normalization(features, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    normalized_features = {}
    for i, feature in features.items():
        normalized_feature = normalize_to_neg1_1(feature)
        normalized_features[i] = normalized_feature
        file = os.path.join(output_folder, f'merged_{i}.txt')
        np.savetxt(file, normalized_feature, fmt='%.6f', delimiter=',')

def run_data_split(path):
    features = {}
    visions = ['bar', 'left-back', 'left-front']
    bar_file_path = os.path.join(path, 'coordinates_interpolated.txt')
    bar_data = read_bar_data(bar_file_path)
    print('Reading skeleton data...')
    
    # 讀取數據
    for vision in visions:
        # 設定檔案路徑
        file = os.path.join(path, f'interpolated_skeleton_{vision}.txt')
        skeleton_data = read_skeleton_data(file)
        frames, left_knee_angles = calculate_angles(skeleton_data)
        features[vision] = {
            'frames': frames,
            'skeleton': skeleton_data,
            'knee_angles': left_knee_angles
        }

        
    print('Find valleys...')
    reps = adjust_valleys_with_bar_data(path, bar_data, features['left-front']['knee_angles'])
    print(reps)

    
    print('splitting data...')
    for vision in visions:
        file = os.path.join(path, f'interpolated_skeleton_{vision}.txt')
        skeleton_data = split_skeleton_data(file, reps)
        if vision == 'bar':
            features_bar = process_bar_vision(skeleton_data, bar_data)
        elif vision == 'left-front':
            features_left_front = process_skeleton2angle(skeleton_data, point=[6, 12, 14, 16])
        elif vision == 'left-back':
            features_left_back = process_skeleton2angle(skeleton_data, point=[5, 11, 13, 15])

    # Example usage
    filtered_feature = merge_and_interpolate(reps, features_left_front, bar_data, features_bar, features_left_back, target_length=110)
    delta_feature = process_delta(filtered_feature)
    delta_square_feature = process_delta(delta_feature)
    zscore_feature = process_zscore(filtered_feature)
    delta_ratio_feature = process_delta_ratio(filtered_feature)
    output_folder = os.path.join(path, 'data_norm2')
    output = {"filtered_norm":filtered_feature, "filtered_delta_norm":delta_feature, "filtered_delta2_norm":delta_ratio_feature, "filtered_zscore_norm":zscore_feature, "filtered_delta_square_norm": delta_square_feature}
    # 處理 delta 和 delta2
    for folder, features in output.items():
        features_path = os.path.join(output_folder, folder)
        process_normalization(features, features_path)

if __name__ == "__main__":
    run_data_split('./recordings/recording_20250328_140412')