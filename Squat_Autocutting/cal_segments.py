import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
from squat_analysis import SquatFeatureExtractor



# 定義 COCO 與 MediaPipe 關鍵點索引對照表
COCO_KEYPOINT_MAP = {
    5: 'left_shoulder', 6: 'right_shoulder',
    11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee',
    15: 'left_ankle', 16: 'right_ankle'
}

MEDIAPIPE_KEYPOINT_MAP = {
    11: 'left_shoulder', 12: 'right_shoulder',
    23: 'left_hip', 24: 'right_hip',
    25: 'left_knee', 26: 'right_knee',
    27: 'left_ankle', 28: 'right_ankle'
}

def load_pose_data(filepath):
    """讀取 mediapipe_landmarks.txt 或 yolo_skeleton.txt 並轉換為 DataFrame"""
    raw_data = pd.read_csv(filepath, header=None, names=['frame', 'idx', 'x', 'y'])
    raw_data['idx'] = pd.to_numeric(raw_data['idx'], errors='coerce').fillna(-1).astype(int)
    max_frame = int(raw_data['frame'].max())
    
    # 自動判定是 MediaPipe (33點, 含 index 23/25) 還是 COCO/YOLO (17點, 含 index 13/14)
    has_mediapipe_idx = raw_data['idx'].isin([23, 25]).any()
    keypoint_map = MEDIAPIPE_KEYPOINT_MAP if has_mediapipe_idx else COCO_KEYPOINT_MAP
    
    columns = []
    for name in COCO_KEYPOINT_MAP.values():
        columns.extend([f'{name}_x', f'{name}_y', f'{name}_confidence'])
    
    pose_df = pd.DataFrame(0.0, index=range(1, max_frame + 1), columns=columns)
    
    for idx, name in keypoint_map.items():
        subset = raw_data[raw_data['idx'] == idx]
        for _, row in subset.iterrows():
            f = int(row['frame'])
            if f <= max_frame:
                pose_df.at[f, f'{name}_x'] = row['x']
                pose_df.at[f, f'{name}_y'] = row['y']
                pose_df.at[f, f'{name}_confidence'] = 1.0 if (row['x'] != 0 or row['y'] != 0) else 0.0
                
    return pose_df.reset_index(drop=True)

def load_bar_data(filepath, n_frames):
    """讀取 yolo_coordinates.txt 並轉換為 DataFrame"""
    bar_data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2: continue
            
            frame_idx = int(parts[0])
            if 'no detection' in parts[1]:
                bar_data.append([frame_idx, np.nan, np.nan])
            else:
                bar_data.append([frame_idx, float(parts[1]), float(parts[2])])
    
    df = pd.DataFrame(bar_data, columns=['frame', 'bar_x', 'bar_y'])
    df = df.set_index('frame')
    
    full_index = range(1, n_frames + 1)
    df = df.reindex(full_index)
    df = df.interpolate(method='linear').bfill().ffill()
    
    return df.reset_index(drop=True)

def visualize_segmentation_debug_integrated(rec_name, df_bar, reps, output_path):
    """生成整合位置、速度、加速度於單一圖表的檢視圖"""
    plt.figure(figsize=(15, 8))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    
    # 調整第三個軸的位置，避免與第二個軸重疊
    ax3.spines['right'].set_position(('outward', 60))
    
    # 計算數據
    bar_y = df_bar['bar_y'].values
    bar_v_y = np.gradient(bar_y) * 30
    bar_a_y = np.gradient(bar_v_y) * 30
    v_thresh = np.std(bar_v_y) * 0.5
    
    # 1. 繪製位置 (左軸)
    p1, = ax1.plot(df_bar.index, bar_y, label='Position (Pix)', color='blue', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Position')
    ax1.invert_yaxis()
    ax1.yaxis.label.set_color('blue')
    
    # 2. 繪製速度 (右軸 1)
    p2, = ax2.plot(df_bar.index, bar_v_y, label='Velocity (Pix/s)', color='gray', alpha=0.5, linestyle='--')
    ax2.axhline(v_thresh, color='red', linestyle=':', alpha=0.3)
    ax2.axhline(-v_thresh, color='green', linestyle=':', alpha=0.3)
    ax2.set_ylabel('Velocity')
    ax2.yaxis.label.set_color('gray')
    
    # 3. 繪製加速度 (右軸 2)
    p3, = ax3.plot(df_bar.index, bar_a_y, label='Accel (Pix/s²)', color='purple', alpha=0.4, linestyle='-.')
    ax3.set_ylabel('Acceleration')
    ax3.yaxis.label.set_color('purple')
    
    # 4. 標記 Rep 區間
    for i, rep in enumerate(reps):
        ax1.axvspan(rep['start'], rep['end'], color='yellow', alpha=0.1)
        ax1.scatter(rep['start'], bar_y[rep['start']], color='red', s=40, zorder=5)
        ax1.scatter(rep['bottom'], bar_y[rep['bottom']], color='black', marker='v', s=40, zorder=5)
        ax1.scatter(rep['end'], bar_y[rep['end']], color='green', s=40, zorder=5)

    plt.title(f"Integrated Segmentation Debug - {rec_name}")
    ax1.grid(True, alpha=0.2)
    
    # 合併圖例
    lines = [p1, p2, p3]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_segmentation_debug_layered(rec_name, df_bar, reps, output_path):
    """生成三層子圖顯示位置、速度、加速度的檢視圖"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    bar_y = df_bar['bar_y'].values
    bar_v_y = np.gradient(bar_y) * 30
    bar_a_y = np.gradient(bar_v_y) * 30
    v_thresh = np.std(bar_v_y) * 0.5
    
    # 1. 位置
    axes[0].plot(df_bar.index, bar_y, label='Position', color='blue')
    axes[0].invert_yaxis()
    axes[0].set_ylabel('Position')
    
    # 2. 速度
    axes[1].plot(df_bar.index, bar_v_y, label='Velocity', color='gray', linestyle='--')
    axes[1].axhline(v_thresh, color='red', linestyle=':', alpha=0.5)
    axes[1].axhline(-v_thresh, color='green', linestyle=':', alpha=0.5)
    axes[1].set_ylabel('Velocity')
    
    # 3. 加速度
    axes[2].plot(df_bar.index, bar_a_y, label='Acceleration', color='purple', linestyle='-.')
    axes[2].set_ylabel('Acceleration')
    axes[2].set_xlabel('Frame Index')

    for ax in axes:
        for i, rep in enumerate(reps):
            ax.axvspan(rep['start'], rep['end'], color='yellow', alpha=0.1)
            if ax == axes[0]:
                ax.scatter(rep['start'], bar_y[rep['start']], color='red', s=40)
                ax.scatter(rep['bottom'], bar_y[rep['bottom']], color='black', marker='v', s=40)
                ax.scatter(rep['end'], bar_y[rep['end']], color='green', s=40)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Layered Segmentation Debug - {rec_name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_results(rec_name, df_bar, df_timeseries, reps, output_path):
    """生成包含所有特徵參數的視覺化圖表 (多子圖)"""
    if df_timeseries.empty:
        return

    fig, axes = plt.subplots(6, 1, figsize=(15, 30), sharex=True)
    
    # 1. 槓鈴與重心軌跡
    ax = axes[0]
    ax.plot(df_bar.index, df_bar['bar_y'], label='Barbell Y', color='blue', linewidth=2)
    ax.set_ylabel('Y Coordinate (Pixels)')
    ax.set_title(f"Squat Analysis - {rec_name}")
    ax.invert_yaxis()
    
    # 2. 關節角度
    ax = axes[1]
    ax.plot(df_timeseries['frame'], df_timeseries['hip_angle'], label='Hip Angle', color='red')
    ax.plot(df_timeseries['frame'], df_timeseries['knee_angle'], label='Knee Angle', color='green')
    ax.plot(df_timeseries['frame'], df_timeseries['femur_horiz_angle'], label='Femur Horiz Angle', color='orange', linestyle='--')
    ax.set_ylabel('Degrees')
    ax.set_title('Joint Angles')
    
    # 3. 正規化位移與比例
    ax = axes[2]
    ax.plot(df_timeseries['frame'], df_timeseries['norm_knee_fwd'], label='Norm Knee Fwd', color='purple')
    ax.plot(df_timeseries['frame'], df_timeseries['norm_hip_bwd'], label='Norm Hip Bwd', color='brown')
    ax.plot(df_timeseries['frame'], df_timeseries['bar_x_deviation'], label='Bar X Dev', color='black', alpha=0.5)
    ax.plot(df_timeseries['frame'], df_timeseries['torso_comp_ratio'], label='Torso Comp Ratio', color='cyan', linestyle=':')
    ax.set_ylabel('Normalized Ratio')
    ax.set_title('Displacements & Ratios')
    
    # 4. 垂直速度特徵
    ax = axes[3]
    ax.plot(df_timeseries['frame'], df_timeseries['hip_y_velocity'], label='Hip V-Y', color='red', alpha=0.7)
    ax.plot(df_timeseries['frame'], df_timeseries['bar_y_velocity'], label='Bar V-Y', color='blue', alpha=0.7)
    ax.set_ylabel('Velocity (Pixels/s)')
    ax.set_title('Linear Velocities')

    # 5. 角速度特徵
    ax = axes[4]
    ax.plot(df_timeseries['frame'], df_timeseries['hip_angle_velocity'], label='Hip Angle Vel', color='darkred', linestyle='--')
    ax.plot(df_timeseries['frame'], df_timeseries['knee_angle_velocity'], label='Knee Angle Vel', color='darkgreen', linestyle='--')
    ax.set_ylabel('Angular Vel (Deg/s)')
    ax.set_title('Angular Velocities')

    # 6. 加速度特徵
    ax = axes[5]
    ax.plot(df_timeseries['frame'], df_timeseries['hip_y_acceleration'], label='Hip Accel-Y', color='red', alpha=0.5, linestyle='-.')
    ax.plot(df_timeseries['frame'], df_timeseries['bar_y_acceleration'], label='Bar Accel-Y', color='blue', alpha=0.5, linestyle='-.')
    ax.set_ylabel('Accel (Pixels/s²)')
    ax.set_xlabel('Frame Index')
    ax.set_title('Linear Accelerations')

    # 在所有子圖標記 Rep 區間
    for ax in axes:
        for i, rep in enumerate(reps):
            ax.axvspan(rep['start'], rep['end'], color='gray', alpha=0.1)
            ax.axvline(rep['bottom'], color='black', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

from scipy.signal import savgol_filter

def smooth_dataframe(df, window_length=11, polyorder=3, exclude_cols=['frame', 'rep_id']):
    """對 DataFrame 中的特徵序列進行 Savitzky-Golay 平滑化"""
    result = df.copy()
    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            # 確保數據長度足以執行平滑 (必須大於 window_length)
            if len(df) > window_length:
                result[col] = savgol_filter(df[col], window_length, polyorder)
    return result

def normalize_dataframe(df, exclude_cols=['frame', 'rep_id']):
    """對 DataFrame 中的指定欄位進行 Min-Max 正規化"""
    result = df.copy()
    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val > 1e-8:
                result[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                result[col] = 0.0
    return result

def extract_recording_id(path_str):
    """從路徑字串中提取識別碼"""
    import re
    cleaned = str(path_str).replace('\\', '/').strip('/')
    cleaned_lower = cleaned.lower()
    
    match = re.search(r'(s\d+)[/_](session\d+)[/_](recording_\d+_\d+)', cleaned_lower)
    if match:
        full_key = f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
        rec_name = match.group(3)
        return full_key, rec_name
        
    match_rec = re.search(r'(recording_\d+_\d+)', cleaned_lower)
    if match_rec:
        return match_rec.group(1), match_rec.group(1)
        
    parts = [p for p in cleaned_lower.split('/') if p]
    if len(parts) >= 3:
        return "/".join(parts[-3:]), parts[-1]
    elif len(parts) >= 1:
        return parts[-1], parts[-1]
        
    return cleaned_lower, cleaned_lower

def parse_rep_list(reps_obj):
    """解開 GT 與 Pred 內部深蹲片段"""
    import re
    if not reps_obj:
        return []

    if isinstance(reps_obj, list) and len(reps_obj) > 0 and isinstance(reps_obj[0], dict):
        item0 = reps_obj[0]
        if 'result' in item0 and isinstance(item0['result'], dict) and 'clips' in item0['result']:
            return item0['result']['clips']
        if 'clips' in item0 and isinstance(item0['clips'], list):
            return item0['clips']

    if isinstance(reps_obj, dict):
        if 'result' in reps_obj and isinstance(reps_obj['result'], dict) and 'clips' in reps_obj['result']:
            return reps_obj['result']['clips']
        if 'clips' in reps_obj and isinstance(reps_obj['clips'], list):
            return reps_obj['clips']
        for key in ['reps', 'segments', 'repetitions', 'annotations', 'events', 'labels', 'data', 'reps_with_id', 'items', 'clips']:
            if key in reps_obj and isinstance(reps_obj[key], list):
                return reps_obj[key]
                
        keys = list(reps_obj.keys())
        try:
            sorted_keys = sorted(keys, key=lambda k: int(re.search(r'\d+', str(k)).group()) if re.search(r'\d+', str(k)) else k)
            return [reps_obj[k] for k in sorted_keys]
        except Exception:
            return list(reps_obj.values())

    if isinstance(reps_obj, list):
        return reps_obj
        
    return []

def parse_rep_frames(rep_item):
    """解析單一下深蹲的 start 與 end frame"""
    start_val = None
    end_val = None
    
    if isinstance(rep_item, dict):
        for k in ['start_frame', 'start', 'start_idx', 'startFrame', 'begin', 'onset', 'start_time', 's', 'start_pos', 'from']:
            if k in rep_item and rep_item[k] is not None:
                start_val = rep_item[k]
                break
        for k in ['end_frame', 'end', 'end_idx', 'endFrame', 'stop', 'offset', 'end_time', 'e', 'end_pos', 'to']:
            if k in rep_item and rep_item[k] is not None:
                end_val = rep_item[k]
                break
                
        if start_val is None or end_val is None:
            for k in ['segment', 'range', 'interval', 'frames']:
                if k in rep_item and isinstance(rep_item[k], (list, tuple)) and len(rep_item[k]) >= 2:
                    start_val = rep_item[k][0]
                    end_val = rep_item[k][-1]
                    break

    elif isinstance(rep_item, (list, tuple)):
        if len(rep_item) >= 2:
            start_val = rep_item[0]
            end_val = rep_item[-1]
            
    if start_val is not None and end_val is not None:
        try:
            return int(start_val), int(end_val)
        except (ValueError, TypeError):
            return None, None
            
    return None, None

def load_gt_records(gt_json_path):
    """讀取 GT JSON 檔案並轉換為字典對照"""
    if not gt_json_path or not os.path.exists(gt_json_path):
        return {}, {}
        
    try:
        with open(gt_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARNING] 無法讀取 GT JSON ({gt_json_path}): {e}")
        return {}, {}
        
    if isinstance(data, dict):
        for wrapper_key in ['data', 'results', 'annotations', 'records', 'items']:
            if wrapper_key in data and isinstance(data[wrapper_key], (dict, list)):
                data = data[wrapper_key]
                break

    full_dict = {}
    rec_dict = {}
    
    if isinstance(data, dict):
        for k, reps in data.items():
            full_k, rec_name = extract_recording_id(k)
            parsed_reps = parse_rep_list(reps)
            full_dict[full_k] = parsed_reps
            rec_dict[rec_name] = parsed_reps
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                rec_id = (item.get('recording') or item.get('video') or item.get('file') or 
                          item.get('name') or item.get('id') or item.get('file_name') or item.get('path'))
                reps = (item.get('segments') or item.get('reps') or item.get('repetitions') or 
                        item.get('labels') or item.get('annotations') or item.get('results') or item)
                
                if rec_id:
                    full_k, rec_name = extract_recording_id(str(rec_id))
                    parsed_reps = parse_rep_list(reps)
                    full_dict[full_k] = parsed_reps
                    rec_dict[rec_name] = parsed_reps

    return full_dict, rec_dict

def visualize_basic_segmentation(rec_path, df_pose, df_bar, reps, output_path, gt_reps=None):
    """使用 GT (S83_S108.json) 與 yolo_coordinates.txt 畫出基本切割情況"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    bar_y = df_bar['bar_y_smoothed'].values if 'bar_y_smoothed' in df_bar.columns else df_bar['bar_y'].values
    frames = df_bar.index
    
    # 畫出槓鈴 Y 軸軌跡 (反轉 Y 軸以符合影像座標)
    ax1.plot(frames, bar_y, label='Barbell Y (Pixels)', color='#3498db', linewidth=2)
    ax1.invert_yaxis()
    
    # 畫出黃色螢光區間 (優先使用 Ground Truth S83_S108)
    spans_to_draw = []
    if gt_reps:
        for item in gt_reps:
            s_f, e_f = parse_rep_frames(item)
            if s_f is not None and e_f is not None:
                spans_to_draw.append((s_f, e_f))
        span_label = "GT Segment (S83_S108)"
    else:
        for r in reps:
            spans_to_draw.append((r['start'], r['end']))
        span_label = "Predicted Segment"

    for idx, (start, end) in enumerate(spans_to_draw):
        ax1.axvspan(start, end, color='#f1c40f', alpha=0.25, label=span_label if idx == 0 else "")

    # 標記每一下預測的起點、最低點、終點
    for r in reps:
        start = r['start']
        bottom = r['bottom']
        end = r['end']
        if start < len(bar_y):
            ax1.scatter(start, bar_y[start], color='green', marker='o', s=40, label='Start' if r['rep_id'] == 1 else "")
        if bottom < len(bar_y):
            ax1.scatter(bottom, bar_y[bottom], color='red', marker='v', s=60, label='Bottom' if r['rep_id'] == 1 else "")
        if end < len(bar_y):
            ax1.scatter(end, bar_y[end], color='blue', marker='x', s=40, label='End' if r['rep_id'] == 1 else "")
        
    ax1.set_title(f"Basic Segmentation Check - {os.path.basename(rec_path)}")
    ax1.set_ylabel("Barbell Y Position")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper right')
    
    if 'hip_angle_smoothed' in df_pose.columns and 'knee_angle_smoothed' in df_pose.columns:
        hip_angles = df_pose['hip_angle_smoothed'].values
        knee_angles = df_pose['knee_angle_smoothed'].values
        if np.max(hip_angles) > 1.0 or np.max(knee_angles) > 1.0:
            ax2.plot(frames, hip_angles, label='Hip Angle', color='red', linewidth=2)
            ax2.plot(frames, knee_angles, label='Knee Angle', color='green', linewidth=2)
            for idx, (start, end) in enumerate(spans_to_draw):
                ax2.axvspan(start, end, color='#f1c40f', alpha=0.25, label=span_label if idx == 0 else "")
            ax2.set_ylim(0, 185)
    
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Angle (Degrees)")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Squat Feature Extraction & Visualization")
    parser.add_argument("--input", required=True, help="Input directory containing recordings")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--normalize", action="store_true", help="Whether to output normalized CSVs")
    parser.add_argument("--gt", type=str, default=r"C:\squat\squat_dataset2_0706\S83_S108.json", help="Ground Truth JSON Path")
    args = parser.parse_args()

    demo_dir = args.input
    extractor = SquatFeatureExtractor(fps=args.fps)
    
    # 載入 Ground Truth (S83_S108.json)
    gt_path = args.gt
    if not os.path.exists(gt_path):
        candidate = os.path.join(demo_dir, "S83_S108.json")
        if os.path.exists(candidate):
            gt_path = candidate
            
    print(f"[INFO] 載入 Ground Truth: {gt_path}")
    gt_full_dict, gt_rec_dict = load_gt_records(gt_path)

    # 建立全局總結果儲存路徑 (供彙整表使用)
    # global_summary_dir = os.path.join(demo_dir, "analysis_results")
    # os.makedirs(global_summary_dir, exist_ok=True)
    
    all_results = []
    all_segments = {}
    
    # 遍歷目錄，遞迴尋找所有包含資料檔的錄影資料夾
    recordings_paths = []
    
    # 如果輸入路徑本身就是錄影資料夾
    if (os.path.exists(os.path.join(demo_dir, 'yolo_skeleton.txt')) or 
        os.path.exists(os.path.join(demo_dir, 'mediapipe_landmarks.txt'))) and \
       os.path.exists(os.path.join(demo_dir, 'yolo_coordinates.txt')):
        recordings_paths.append(os.path.abspath(demo_dir))
    else:
        for root, dirs, files in os.walk(demo_dir):
            if "analysis_results" in dirs:
                dirs.remove("analysis_results")
                
            has_skeleton = 'yolo_skeleton.txt' in files or 'mediapipe_landmarks.txt' in files
            has_coords = 'yolo_coordinates.txt' in files
            if has_skeleton and has_coords:
                if "棋盤" not in root:
                    recordings_paths.append(os.path.abspath(root))
    
    EXCLUDE_RECORDINGS = [
        "recording_20260512_112434",  # S87
        "recording_20260514_162045",  # S90
        "recording_20260521_140622",  # S94
        "recording_20260601_104441",  # S103
        "recording_20260611_153424",  # S108
    ]

    current_subject = None
    for rec_path in recordings_paths:
        rec = os.path.basename(rec_path)
        # 取得相對路徑作為字典 Key 避免重名覆蓋
        rel_rec_path = os.path.relpath(rec_path, demo_dir)
        rel_key = rel_rec_path.replace('\\', '/')
        
        # 檢查是否為排除之錄影資料夾
        if any(ex in rel_key for ex in EXCLUDE_RECORDINGS):
            print(f"[EXCLUDE] 跳過指定排除的錄影: {rel_key}")
            continue
        
        path_parts = rel_rec_path.split(os.sep)
        subject_name = path_parts[0] if len(path_parts) > 0 else "unknown"
        if current_subject is not None and subject_name != current_subject:
            print(f"\n[Progress Report] Finished processing Subject: {current_subject}\n")
        current_subject = subject_name
        
        print(f"Processing {rel_key}...")
        
        pose_path = os.path.join(rec_path, 'yolo_skeleton.txt')
        if not os.path.exists(pose_path):
            pose_path = os.path.join(rec_path, 'mediapipe_landmarks.txt')
            
        bar_path = os.path.join(rec_path, 'yolo_coordinates.txt')
        
        if not os.path.exists(pose_path) or not os.path.exists(bar_path):
            print(f"  Missing data files ({pose_path} or {bar_path}), skipping.")
            continue
            
        # 建立該錄影專屬的結果資料夾
        # rec_summary_dir = os.path.join(rec_path, "analysis_results")
        # os.makedirs(rec_summary_dir, exist_ok=True)
            
        try:
            # 1. 讀取與前處理資料
            df_pose = load_pose_data(pose_path)
            df_bar = load_bar_data(bar_path, len(df_pose))
            
            # 2. 執行分析
            pose_clean, bar_clean = extractor._preprocess(df_pose, df_bar)
            reps = extractor._segment_reps(pose_clean, bar_clean)
            features = extractor.fit_transform(df_pose, df_bar)
            
            # 擷取逐幀特徵 (Timeseries)
            df_timeseries = extractor.extract_timeseries(df_pose, df_bar)
            
            if not df_timeseries.empty:
                # 執行平滑化 (方案二：特徵層級平滑)
                df_timeseries = smooth_dataframe(df_timeseries, window_length=15, polyorder=3)
                
            if not features.empty:
                features['recording'] = rec
                all_results.append(features)
                
                # 3. 儲存與視覺化
                # 儲存平滑後的逐幀特徵
                # ts_path = os.path.join(rec_summary_dir, f"{rec}_timeseries.csv")
                # df_timeseries.to_csv(ts_path, index=False)
                
                # 執行正規化並儲存
                # df_norm = normalize_dataframe(df_timeseries)
                # norm_ts_path = os.path.join(rec_summary_dir, f"{rec}_timeseries_normalized.csv")
                # df_norm.to_csv(norm_ts_path, index=False)
                
                # viz_path = os.path.join(rec_summary_dir, f"{rec}_plot.png")
                # visualize_results(rec, bar_clean, df_timeseries, reps, viz_path)
                
                # 1. 整合圖 (Single Plot)
                # integrated_debug_path = os.path.join(rec_summary_dir, f"{rec}_segmentation_debug.png")
                # visualize_segmentation_debug_integrated(rec, bar_clean, reps, integrated_debug_path)
                
                # 2. 三層圖 (Three Layer Plot)
                # layered_debug_path = os.path.join(rec_summary_dir, f"{rec}_segmentation_debug_three_layer.png")
                # visualize_segmentation_debug_layered(rec, bar_clean, reps, layered_debug_path)
                
                # 3. 匯出分段 Frame 為 JSON
                json_path = os.path.join(rec_path, "segments.json")
                reps_with_id = [{"rep_id": i + 1, "start": int(r['start']), "bottom": int(r['bottom']), "end": int(r['end'])} for i, r in enumerate(reps)]
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(reps_with_id, f, indent=4)
                
                # 4. 尋找對應之 GT 段落並生成基本切割視覺化圖表
                full_k, rec_name = extract_recording_id(rel_key)
                rec_gt_reps = gt_full_dict.get(full_k) or gt_rec_dict.get(rec_name) or []
                
                segment_plot_path = os.path.join(rec_path, "segmentation_check_v2.png")
                visualize_basic_segmentation(rec_path, pose_clean, bar_clean, reps_with_id, segment_plot_path, gt_reps=rec_gt_reps)
                
                all_segments[rel_key] = reps_with_id
                print(f"  Found {len(features)} reps (GT reps: {len(rec_gt_reps)}). Results saved in {rec_path}")
            else:
                all_segments[rel_key] = []
                print(f"  No reps detected.")
                
        except Exception as e:
            print(f"  Error processing {rel_key}: {e}")
            import traceback
            traceback.print_exc()
            
    if current_subject is not None:
        print(f"\n[Progress Report] Finished processing Subject: {current_subject}\n")
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        cols = ['recording'] + [c for c in final_df.columns if c != 'recording']
        final_df = final_df[cols]
        
        # csv_path = os.path.join(global_summary_dir, "squat_features_summary.csv")
        # final_df.to_csv(csv_path, index=False)
        # print(f"\nSuccessfully processed {len(all_results)} recordings.")
        # print(f"Global summary saved in: {global_summary_dir}")
        pass
    else:
        print("No features extracted.")

    if all_segments:
        summary_json_path = os.path.join(demo_dir, "segments.json")
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_segments, f, indent=4)
        print(f"Global segments summary saved in: {summary_json_path}")

if __name__ == "__main__":
    main()
