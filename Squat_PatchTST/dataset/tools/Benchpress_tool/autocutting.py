import numpy as np
import pandas as pd
import os
import json
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

def get_angle(a, b, c):  # 計算 ∠abc 的角度（單位為度）
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    dot = np.dot(ab, cb)
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    if norm_ab == 0 or norm_cb == 0:
        return np.nan
    angle = np.arccos(np.clip(dot / (norm_ab * norm_cb), -1.0, 1.0))
    angle_deg = np.degrees(angle)
    return angle_deg if angle_deg <= 180 else 360 - angle_deg

def apply_uniform_smoothing(series, window_size=7, shift_size=2):
    filled_series = series.ffill().bfill()
    smoothed = uniform_filter1d(filled_series.values, size=window_size)
    shift = shift_size // 2
    shifted = np.roll(smoothed, -shift)
    shifted[-shift:] = np.nan
    return pd.Series(shifted, index=series.index).interpolate(method="linear").bfill().ffill()

def parse_frame_data(bar_dict, rear_ske_dict):
    results = []
    # Joint order in rear_ske_dict: 0:L_SHO, 1:R_SHO, 2:L_ELB, 3:R_ELB, 4:L_WRI, 5:R_WRI
    # flat list frame_data: [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
    for frame_idx, frame_data in rear_ske_dict.items():
        try:
            if len(frame_data) < 12:
                continue
            bar_data = bar_dict.get(frame_idx)
            if bar_data is None:
                continue
            
            bar_y = 480 - bar_data[1] # Reverse Y for visual logic
            
            # Extract [x, y] pairs
            l_sho = [frame_data[0], frame_data[1]]
            r_sho = [frame_data[2], frame_data[3]]
            l_elb = [frame_data[4], frame_data[5]]
            r_elb = [frame_data[6], frame_data[7]]
            l_wri = [frame_data[8], frame_data[9]]
            r_wri = [frame_data[10], frame_data[11]]

            l_elb_ang = get_angle(l_wri, l_elb, l_sho)
            r_elb_ang = get_angle(r_wri, r_elb, r_sho)
            avg_elb_ang = np.nanmean([l_elb_ang, r_elb_ang])

            results.append([frame_idx, bar_y, l_elb_ang, r_elb_ang, avg_elb_ang])
        except Exception as e:
            print(f"錯誤發生在第 {frame_idx} 幀: {e}")
            continue
            
    return pd.DataFrame(results, columns=["Frame", "Bar_Y", "left_elbow_angle", "right_elbow_angle", "avg_elbow_angle"])

# --- Core Segmentation Logic ported from original ---

def find_clear_difference_side(angle, idx, min_peak_diff, max_search=70):
    val = angle[idx]
    left_idx, right_idx = idx - 1, idx + 1
    while left_idx >= max(0, idx - max_search):
        if angle[left_idx] - val >= min_peak_diff: break
        left_idx -= 1
    else: return False
    while right_idx < min(len(angle), idx + max_search):
        if angle[right_idx] - val >= min_peak_diff: break
        right_idx += 1
    else: return False
    return True

def is_clear_valley(angle, idx, window=25, min_prominence=0.05, future_window=18, min_peak_diff=100):
    idx = int(idx)
    val = angle[idx]
    start, end = max(0, idx - window), min(len(angle), idx + window)
    left, right = angle[start:idx], angle[idx+1:end]
    if len(left) == 0 or len(right) == 0: return False
    
    left_max, right_max = np.max(left), np.max(right)
    dynamic_prom = min(min_prominence * val, 20)
    cond_shape = (val < left_max and val < right_max and (left_max - val > dynamic_prom or right_max - val > dynamic_prom))
    cond_no_drop = all(val <= f for f in angle[idx+1 : idx+1+future_window])
    cond_peak_diff = find_clear_difference_side(angle, idx, min_peak_diff)
    
    past_window = 20
    cond_pre_slope = np.mean(np.diff(angle[max(0, idx-past_window):idx])) < 0 if idx > 0 else True
    cond_post_slope = np.mean(np.diff(angle[idx : min(len(angle), idx+past_window+1)])) > 0 if idx < len(angle)-1 else True
    
    return cond_shape and cond_no_drop and cond_peak_diff and cond_pre_slope and cond_post_slope

def filter_close_valleys(angle, valleys, peaks, min_peak_diff=60):
    filtered = []
    i = 0
    while i < len(valleys):
        v1 = valleys[i]
        if i + 1 < len(valleys):
            v2 = valleys[i+1]
            mid_peaks = [p for p in peaks if v1 < p < v2]
            deepest = v1 if angle[v1] < angle[v2] else v2
            if not any(angle[p] - angle[deepest] >= min_peak_diff for p in mid_peaks):
                filtered.append(deepest); i += 2; continue
        filtered.append(v1); i += 1
    return filtered

def analyze_valleys_peaks(angle_series, min_peak_diff=60):
    angle = angle_series.values
    bar_pos_threshold = np.mean(angle_series.dropna().values[-5:]) + 20
    
    raw_valleys, _ = find_peaks(-angle, distance=25, prominence=0.1, width=2)
    valleys = [v for v in raw_valleys if is_clear_valley(angle, v, min_peak_diff=min_peak_diff) and angle[v] < bar_pos_threshold]
    
    raw_peaks, _ = find_peaks(angle, distance=25, prominence=0.1, width=2)
    peaks = [p for p in raw_peaks if valleys and (angle[p] - angle[min(valleys, key=lambda v: abs(v-p))] >= min_peak_diff)]
    
    valleys = filter_close_valleys(angle, valleys, peaks, min_peak_diff=min_peak_diff)
    
    final_valleys = []
    for i, v in enumerate(valleys):
        l_peaks = [p for p in peaks if p < v]; r_peaks = [p for p in peaks if p > v]
        if i == 0:
            if r_peaks and angle[min(r_peaks)] - angle[v] >= min_peak_diff: final_valleys.append(v)
        elif l_peaks and r_peaks:
            if angle[max(l_peaks)] - angle[v] >= min_peak_diff or angle[min(r_peaks)] - angle[v] >= min_peak_diff: final_valleys.append(v)
    return peaks, final_valleys

def find_peaks_between_valleys(data, valleys, bar, min_height_diff=8):
    median_peaks, ext_v = [], [-1] + valleys + [len(data)]
    for i in range(len(ext_v)-1):
        v1, v2 = max(0, ext_v[i]), ext_v[i+1]
        seg = data[v1:v2]
        if len(seg) == 0: continue
        pks, _ = find_peaks(seg)
        valid_pks = []
        for p in pks:
            gi = v1 + p
            if i == 0 or i == len(ext_v)-2:
                bw = bar[max(gi-5, 0):min(gi+5, len(bar))]
                if len(bw) >= 2 and (np.max(bw) - np.min(bw)) < 3: continue
            if (seg[p] - seg[0] >= min_height_diff) and (seg[p] - seg[-1] >= min_height_diff): valid_pks.append(gi)
        
        if valid_pks: median_peaks.append(valid_pks[len(valid_pks)//2])
        else:
            candidates = np.argsort(seg)[::-1]
            fb = None
            for idx in candidates:
                gi = v1 + idx
                if i == 0 or i == len(ext_v)-2:
                    bw = bar[max(gi-5, 0):min(gi+5, len(bar))]
                    if len(bw) >= 2 and (np.max(bw) - np.min(bw)) < 3: continue
                fb = gi; break
            median_peaks.append(fb if fb is not None else v1 + np.argmax(seg))
    return median_peaks

def find_corners(kappa, angle, valleys, peaks, threshold=20):
    starts, ends = [], []
    for i, v in enumerate(valleys):
        lp, rp = peaks[i], peaks[i+1]
        s_seg = np.arange(lp, v); e_seg = np.arange(rp, v, -1)
        valid_s = next((p for p in s_seg[np.argsort(kappa[s_seg])] if angle[p] - angle[v] >= threshold), lp)
        valid_e = next((p for p in e_seg[np.argsort(kappa[e_seg])] if angle[p] - angle[v] >= threshold), rp)
        starts.append(valid_s); ends.append(valid_e)
    return starts, ends

def run_autocutting(video_path, bar_dict, rear_ske_dict):
    df = parse_frame_data(bar_dict, rear_ske_dict)
    if df.empty: return []
    
    df["Bar_Y"] = apply_uniform_smoothing(df["Bar_Y"])
    for col in ["avg_elbow_angle", "left_elbow_angle", "right_elbow_angle"]:
        df[col] = df[col].rolling(window=12, center=True).mean().interpolate().bfill().ffill()
    
    dy = np.gradient(df["left_elbow_angle"].values)
    ddy = np.gradient(dy)
    kappa = (ddy / np.power(1 + dy**2, 1.5)) * 3 # Scaled
    
    peaks_raw, valleys = analyze_valleys_peaks(df["Bar_Y"])
    if not valleys: return []
    
    peaks = find_peaks_between_valleys(df["left_elbow_angle"].values, valleys, df["Bar_Y"].values)
    starts, ends = find_corners(kappa, df["left_elbow_angle"].values, valleys, peaks)
    
    split_info = {str(i): {"start": int(df.loc[s, "Frame"]), "end": int(df.loc[e, "Frame"])} for i, (s, e) in enumerate(zip(starts, ends))}
    
    config_dir = os.path.join(video_path, "config")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "Split_info.json"), "w") as f:
        json.dump(split_info, f, indent=4)
        
    print(f"✅ Autocutting complete. Found {len(valleys)} segments.")
    return split_info