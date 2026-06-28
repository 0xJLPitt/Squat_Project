# -*- coding: utf-8 -*-                                                                                     # 檔頭宣告
import os, sys, re, json, math, ast                                                                         # 基本匯入
import numpy as np                                                                                           # 向量/角度計算

# === 點位索引：0左肩,1右肩,2左髖,3右髖,4左肘,5右肘,6左腕,7右腕 ===
L_SHO, R_SHO, L_HIP, R_HIP, L_ELB, R_ELB = 0, 1, 2, 3, 4, 5                                                  # 索引簡寫

def angle_abc(a, b, c):                                                                                      # 計算∠ABC（頂點B）
    a = np.asarray(a, dtype=float)                                                                            # 轉float陣列
    b = np.asarray(b, dtype=float)                                                                            # 轉float陣列
    c = np.asarray(c, dtype=float)                                                                            # 轉float陣列
    v1, v2 = a - b, c - b                                                                                     # 邊向量
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)                                                            # 邊長
    if n1 == 0 or n2 == 0:                                                                                    # 邊長為0保護
        return None                                                                                           # 回None
    cosv = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)                                                     # 夾角cos且夾範圍
    return float(np.degrees(np.arccos(cosv)))                                                                 # 轉度數回傳

def parse_skeleton_txt(txt_path):                                                                             # 解析骨架txt
    frames, kps_all = [], []                                                                                  # 幀序與每幀關鍵點
    pat = re.compile(r"^Frame\s+(\d+):\s*(\[.*\])\s*$")                                                       # 行樣式
    with open(txt_path, "r", encoding="utf-8") as f:                                                          # 開檔
        for line in f:                                                                                        # 逐行
            m = pat.match(line.strip())                                                                       # 比對
            if not m:                                                                                         # 不符合
                continue                                                                                      # 跳過
            fid = int(m.group(1))                                                                             # 幀號
            raw = m.group(2)                                                                                  # 資料字串
            try:
                data = ast.literal_eval(raw)                                                                  # 安全解析
            except Exception:
                data = []                                                                                     # 解析失敗給空
            kp_list = data[0] if isinstance(data, list) and len(data) > 0 else []                             # 取第一人
            frames.append(fid)                                                                                # 存幀號
            kps_all.append(kp_list)                                                                           # 存關鍵點
    # 依幀號排序，確保 frames/points 對齊（若檔案未排序）                                                             # 排序保護
    order = np.argsort(frames)                                                                                # 排序索引
    frames = [frames[i] for i in order]                                                                       # 重排幀
    kps_all = [kps_all[i] for i in order]                                                                     # 重排點
    return frames, kps_all                                                                                    # 回傳

def compute_angles(frames, kps_all):                                                                          # 計算左右角度
    left_vals, right_vals = [], []                                                                            # 左右序列
    for kp in kps_all:                                                                                        # 逐幀
        if not kp or len(kp) < 6:                                                                             # 點不足
            left_vals.append(None); right_vals.append(None)                                                   # 填None
            continue                                                                                          # 下一幀
        try:
            angL = angle_abc(kp[L_ELB], kp[L_SHO], kp[L_HIP])                                                 # 左：4-0-2
            angR = angle_abc(kp[R_ELB], kp[R_SHO], kp[R_HIP])                                                 # 右：5-1-3
        except Exception:
            angL, angR = None, None                                                                           # 任何索引錯誤
        left_vals.append(None if (angL is None or (isinstance(angL, float) and math.isnan(angL))) else angL)  # 收左
        right_vals.append(None if (angR is None or (isinstance(angR, float) and math.isnan(angR))) else angR) # 收右
    return left_vals, right_vals                                                                              # 回傳

def dump_angle_json(out_path, title, frames, values):                                                         # 輸出JSON
    # values can be [v1, v2...] or [[L1, R1], [L2, R2]...]
    # Flatten everything to find real min/max
    all_reals = []
    for v in values:
        if isinstance(v, (list, tuple)):
            all_reals.extend([x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))])
        elif v is not None and not (isinstance(v, float) and np.isnan(v)):
            all_reals.append(v)

    y_min = float(np.min(all_reals)) if all_reals else 0.0                                                           # y最小
    y_max = float(np.max(all_reals)) if all_reals else 180.0                                                           # y最大
    
    # Ensure values is a list (if it was a zip/iterator) and contents are lists (not tuples)
    processed_values = []
    for v in values:
        if isinstance(v, (list, tuple, np.ndarray)):
            processed_values.append(list(v))
        else:
            processed_values.append(v)

    payload = {                                                                                               # 組物件
        "title": title,                                                                                       # 標題
        "y_label": "Angle (degrees)",                                                                         # Y標籤
        "y_min": round(y_min, 2),                                                                                       # y_min
        "y_max": round(y_max, 2),                                                                                       # y_max
        "frames": [int(f) for f in frames],                                                                                     # 幀序列
        "values": [[round(v, 2) for v in val] if isinstance(val, (list, tuple, np.ndarray)) else round(val, 2) for val in processed_values]                                                                                      # 角度序列
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:                                                          # 開檔
        json.dump(payload, f, ensure_ascii=False, indent=4) 
    return payload

def run_torso_angle_produce(folder_path, skeleton_dict=None):                                         # 拋錯
    if skeleton_dict is not None and isinstance(skeleton_dict, dict) and len(skeleton_dict) > 0:
        # Use provided dict from memory
        frames = sorted(skeleton_dict.keys())
        kps_all = []
        for f in frames:
            raw_coords = skeleton_dict[f]
            # Chunking [x1, y1, x2, y2...] into [(x1,y1), (x2, y2)...]
            kp_list = []
            for i in range(0, len(raw_coords), 2):
                kp_list.append((raw_coords[i], raw_coords[i+1]))
            kps_all.append(kp_list)
    else:
        # Fallback to reading from file
        txt_path = os.path.join(folder_path, "interpolated_skeleton_top_hampel.txt")
        if not os.path.exists(txt_path):                                                                          # 檔案檢查
            # We also check the non-hampel one as fallback if needed, but hampel is expected
            raise FileNotFoundError(f"找不到骨架txt：{txt_path}")                                                   # 拋錯
        frames, kps_all = parse_skeleton_txt(txt_path)                                                            # 讀取解析

    if not frames:                                                                                            # 無資料
        raise ValueError("沒有可解析的 Frame 資料")                                                       # 拋錯

    left_vals, right_vals = compute_angles(frames, kps_all)                                                   # 計算角度

    # Apply Savgol filter to angles
    try:
        from .hampel import run_savgol_on_series
        left_vals = run_savgol_on_series(left_vals, window_length=21, polyorder=3)
        right_vals = run_savgol_on_series(right_vals, window_length=21, polyorder=3)
    except Exception as e:
        print(f"[TorsoAngle] Warning: Savgol filtering failed: {e}")



    out_torso = os.path.join(folder_path, "config", "Torso_Angle.json")
    # Using list() to consume zip and convert tuples to lists for JSON
    combined_values = [list(pair) for pair in zip(left_vals, right_vals)]

    payload = dump_angle_json(out_torso, "Elbow–Trunk Angle (L, R)", frames, combined_values)

    print(f"✅ 完成！已儲存於：{out_torso}")
