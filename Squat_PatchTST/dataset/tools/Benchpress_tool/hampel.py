import os
import numpy as np
from scipy.interpolate import interp1d
# from ..bar_data_produce import run_bar_data_produce

def hampel_filter(series, window_size=7, n_sigmas=3):
    """
    Hampel Filter for outlier detection.
    """
    series = np.array(series)
    n = len(series)
    k = 1.4826
    half_window = window_size // 2
    outlier_mask = np.zeros(n, dtype=bool)

    for i in range(half_window, n - half_window):
        window = series[i - half_window:i + half_window + 1]
        
        # Filter NaNs and check if enough points remain
        window_valid = window[~np.isnan(window)]
        if len(window_valid) < 3:
            continue

        median = np.median(window_valid)
        mad = k * np.median(np.abs(window_valid - median))
        if mad == 0:
            continue
        if np.abs(series[i] - median) > n_sigmas * mad:
            outlier_mask[i] = True

    return outlier_mask

def run_savgol_on_series(values, window_length=21, polyorder=3):
    """
    Apply Savitzky-Golay filter to a series of values, handling Nones and NaNs via interpolation.
    """
    from scipy.signal import savgol_filter
    if not values or len(values) < 3:
        return np.nan_to_num(values).tolist() # Ensure output is list and NaNs are handled
        
    v_arr = np.array([v if v is not None else np.nan for v in values]).astype(float)
    idx = np.arange(len(v_arr))
    valid = ~np.isnan(v_arr)
    
    if valid.sum() < 2:
        return np.nan_to_num(v_arr).tolist()
        
    # Interpolate
    interp_func = interp1d(idx[valid], v_arr[valid], kind='linear', fill_value='extrapolate')
    v_interp = interp_func(idx)
    
    # Apply Savgol
    win_len = window_length
    if win_len >= len(v_interp):
        win_len = len(v_interp) if len(v_interp) % 2 != 0 else len(v_interp) - 1
    if win_len < 3:
        return v_interp.tolist()
        
    v_smooth = savgol_filter(v_interp, window_length=win_len, polyorder=polyorder)
    return v_smooth.tolist()



def run_hampel_bar(folder_path, sport='benchpress'):
    """
    Process bar coordinates with Hampel Filter.
    Equivalent to step0_hampel_bar.py
    """
    input_filename = "coordinates_interpolated.txt"
    input_path = os.path.join(folder_path, input_filename)

    if not os.path.exists(input_path):
        print(f"⛔ [Hampel Bar] 找不到檔案：{input_path}")
        return {}

    try:
        # 初始化數據存儲
        frames = []
        values = []

        # 讀取 YOLO 偵測數據
        with open(input_path, "r") as file:
            for line in file:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                frame_count = int(parts[0])  # 幀數
                frames.append(frame_count)
                values.append([float(parts[1]), float(parts[2])])
        
        values = np.array(values)
        x_outliers = hampel_filter(values[:, 0])
        y_outliers = hampel_filter(values[:, 1])

        outlier_frames = x_outliers | y_outliers
        values_filtered = values.copy().astype(float)
        values_filtered[outlier_frames] = np.nan

        results = {}
        for i, frame in enumerate(frames):
            results[frame] = [values_filtered[i][0], values_filtered[i][1]]
        
        # 進行插值處理
        results = interpolate_hampel_dict(results)
        
        # run_bar_data_produce(folder_path, sport, results)
        
        print(f"✅ [Hampel Bar] 已處理完畢 (回傳資料量：{len(results)})")
        return results

    except Exception as e:
        print(f"❌ [Hampel Bar] 處理失敗：{input_path}, 錯誤：{e}")
        return {}

def process_skeleton_file(input_path, output_path, expected_joints):
    data = {} # frame -> {joint_idx: (x, y)}

    if not os.path.exists(input_path):
        return {}

    with open(input_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            frame_idx = int(parts[0])
            joint_idx = int(parts[1])
            x, y = float(parts[2]), float(parts[3])
            
            if frame_idx not in data:
                data[frame_idx] = {}
            data[frame_idx][joint_idx] = (x, y)

    all_frames = sorted(data.keys())
    all_points = []
    valid_frames = []
    
    for f_idx in all_frames:
        joints = data[f_idx]
        if len(joints) >= expected_joints:
            p_list = []
            for j_idx in range(expected_joints):
                if j_idx in joints:
                    p_list.append(joints[j_idx])
                else:
                    p_list.append((np.nan, np.nan))
            all_points.append(p_list)
            valid_frames.append(f_idx)

    if len(all_points) == 0:
        return {}

    all_points = np.array(all_points) # (num_frames, expected_joints, 2)
    mask = np.zeros_like(all_points[:, :, 0], dtype=bool)

    for joint_idx in range(expected_joints):
        x_series = all_points[:, joint_idx, 0]
        y_series = all_points[:, joint_idx, 1]
        x_outliers = hampel_filter(x_series)
        y_outliers = hampel_filter(y_series)
        mask[:, joint_idx] = x_outliers | y_outliers

    cleaned_points = all_points.copy().astype(float)
    cleaned_points[mask] = np.nan

    results = {}
    for i, frame in enumerate(valid_frames):
        coords_flat = []
        for j in range(expected_joints):
            coords_flat.extend([cleaned_points[i, j, 0], cleaned_points[i, j, 1]])
        results[frame] = coords_flat
    return results

def interpolate_hampel_dict(data):
    """
    Interpolate NaNs generated by Hampel Filter using linear interpolation and extrapolation.
    """
    if not data:
        return data
    
    frames = sorted(data.keys())
    num_coords = len(data[frames[0]])
    matrix = np.array([data[f] for f in frames])
    frames_np = np.array(frames)

    for col in range(num_coords):
        values = matrix[:, col]
        valid = ~np.isnan(values)
        if valid.sum() >= 2:
            interp_func = interp1d(frames_np[valid], values[valid], kind='linear', fill_value='extrapolate')
            matrix[:, col] = interp_func(frames_np)
        else:
            matrix[:, col] = np.nan_to_num(values)

    interpolated_data = {frames[i]: matrix[i].tolist() for i in range(len(frames))}
    return interpolated_data

def run_hampel_yolo_ske_rear(folder_path):
    """
    Process rear view skeleton with Hampel Filter.
    Equivalent to step0_hampel_yolo_ske_rear.py
    """
    input_path = os.path.join(folder_path, "interpolated_skeleton_rear.txt")
    
    data = process_skeleton_file(input_path, None, expected_joints=6)
    if data:
        data = interpolate_hampel_dict(data)
        print(f"✅ [Hampel Rear] 已處理完畢 (回傳資料量：{len(data)})")
    else:
        print(f"⚠️ [Hampel Rear] 無有效資料或找不到檔案：{input_path}")
    return data

def run_hampel_yolo_ske_top(folder_path):
    """
    Process top view skeleton with Hampel Filter.
    Equivalent to step0_hampel_yolo_ske_top.py
    """
    input_path = os.path.join(folder_path, "interpolated_skeleton_top.txt")
    
    data = process_skeleton_file(input_path, None, expected_joints=8)
    if data:
        data = interpolate_hampel_dict(data)
        print(f"✅ [Hampel Top] 已處理完畢 (回傳資料量：{len(data)})")
    else:
        print(f"⚠️ [Hampel Top] 無有效資料或找不到檔案：{input_path}")
    return data

def run_hampel_yolo_ske_left_front(folder_path):
    """
    Process left-front view skeleton with Hampel Filter (for Deadlift).
    """
    input_path = os.path.join(folder_path, "interpolated_skeleton_left-front.txt")
    
    # Deadlift seems to use high joint counts, but let's see standard YOLO pose joints
    # Usually it's 17 joints, checking data_produce.py it uses up to joint 16.
    data = process_skeleton_file(input_path, None, expected_joints=17)
    if data:
        data = interpolate_hampel_dict(data)
        print(f"✅ [Hampel Left-Front] 已處理完畢 (回傳資料量：{len(data)})")
    else:
        print(f"⚠️ [Hampel Left-Front] 無有效資料或找不到檔案：{input_path}")
    return data

