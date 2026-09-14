import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def apply_rolling_smoothing(series, window=12):
    return series.rolling(window=window, center=True).mean()

def apply_hampel_filter(series, window_size=15, n_sigmas=45):
    series = series.copy()
    L = 1.4826
    rolling_median = series.rolling(window_size, center=True).median()
    diff = np.abs(series - rolling_median)
    mad = L * diff.rolling(window_size, center=True).median()
    # 處理 mad 為 0 或 NaN 的情況避免除以 0 警告
    mad = mad.replace(0, 1e-6).fillna(1e-6)
    # 處理 NA，outliers 在 NA 處會是 False
    outliers = (diff > n_sigmas * mad).fillna(False)
    series[outliers] = rolling_median[outliers]
    return series

class SquatFeatureExtractor:
    """
    深蹲特徵擷取器 (SquatFeatureExtractor)
    
    本類別旨在將 YOLO 偵測到的 2D 骨架與槓鈴軌跡原始資料，轉換為生物力學特徵，
    並聚合為可用於機器學習模型的 1D 特徵向量。
    """
    def __init__(self, fps=30, conf_thresh=0.5):
        """
        初始化擷取器
        :param fps: 影片的幀率 (預設為 30)
        :param conf_thresh: 關鍵點信心度閾值，低於此值將進行插值
        """
        self.fps = fps
        self.conf_thresh = conf_thresh
        self.window_length = 11  # Savitzky-Golay 濾波器窗口長度
        self.polyorder = 3       # Savitzky-Golay 濾波器多項式階數

    def _calculate_angle(self, p_top, p_mid, p_bot):
        """計算三點形成的夾角 (角度制)"""
        v1 = np.array([p_top[0] - p_mid[0], p_top[1] - p_mid[1]])
        v2 = np.array([p_bot[0] - p_mid[0], p_bot[1] - p_mid[1]])
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            return 0.0
            
        cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def _calculate_femur_horizontal_angle(self, hip, knee):
        """
        計算髖-膝向量與水平線的夾角 (Femur Horizontal Angle)。
        影像座標系 Y 向下為正。
        - 當膝蓋比髖部低時 (站立階段)，dy > 0，角度為正。
        - 當大腿完全水平時，dy = 0，角度為 0。
        - 當髖部低於膝蓋時 (深蹲深度過水平)，dy < 0，角度為負。
        """
        dx = np.abs(knee[0] - hip[0])
        dy = knee[1] - hip[1]
        return np.degrees(np.arctan2(dy, dx + 1e-6))

    def _preprocess(self, pose_df, bar_df):
        """第一階段：訊號前處理 (過濾、插值、平滑化、視角選擇)"""
        pose_clean = pose_df.copy()
        bar_clean = bar_df.copy()

        # 1. Confidence 過濾與線性插值
        joints = [c.replace('_x', '').replace('_y', '').replace('_confidence', '') 
                  for c in pose_df.columns if '_confidence' in c]
        joints = list(set(joints))

        for joint in joints:
            conf_col = f"{joint}_confidence"
            if conf_col not in pose_clean.columns: continue
            
            mask = pose_clean[conf_col] < self.conf_thresh
            for axis in ['x', 'y']:
                col = f"{joint}_{axis}"
                if col not in pose_clean.columns: continue
                
                if mask.any():
                    indices = np.arange(len(pose_clean))
                    valid_idx = indices[~mask]
                    if len(valid_idx) > 1:
                        f = interp1d(valid_idx, pose_clean.loc[~mask, col], 
                                     kind='linear', fill_value="extrapolate")
                        pose_clean.loc[mask, col] = f(indices[mask])
                    elif len(valid_idx) == 1:
                        pose_clean.loc[mask, col] = pose_clean.loc[valid_idx[0], col]

        # 2. 平滑化 (Smoothing) - 消除偵測抖動
        for col in pose_clean.columns:
            if '_x' in col or '_y' in col:
                pose_clean[col] = savgol_filter(pose_clean[col], self.window_length, self.polyorder)
        
        for col in bar_clean.columns:
            if 'bar_x' in col or 'bar_y' in col:
                bar_clean[col] = savgol_filter(bar_clean[col], self.window_length, self.polyorder)

        # 3. 視角選擇 (自動選擇信心度較高的一側作為分析主斷面)
        left_confs = [f"{j}_confidence" for j in ['left_shoulder', 'left_hip', 'left_knee', 'left_ankle'] 
                      if f"{j}_confidence" in pose_clean.columns]
        right_confs = [f"{j}_confidence" for j in ['right_shoulder', 'right_hip', 'right_knee', 'right_ankle'] 
                       if f"{j}_confidence" in pose_clean.columns]
        
        avg_left = pose_df[left_confs].mean().mean() if left_confs else 0
        avg_right = pose_df[right_confs].mean().mean() if right_confs else 0
        
        side = 'left' if avg_left >= avg_right else 'right'
        
        # 建立統一命名的關鍵點欄位，方便後續計算
        side_map = {
            'shoulder_x': f'{side}_shoulder_x', 'shoulder_y': f'{side}_shoulder_y',
            'hip_x': f'{side}_hip_x', 'hip_y': f'{side}_hip_y',
            'knee_x': f'{side}_knee_x', 'knee_y': f'{side}_knee_y',
            'ankle_x': f'{side}_ankle_x', 'ankle_y': f'{side}_ankle_y'
        }
        
        for key, val in side_map.items():
            if val in pose_clean.columns:
                pose_clean[key] = pose_clean[val]

        return pose_clean, bar_clean

    def _segment_reps(self, pose_df, bar_df):
        """
        第二階段：動作切割 (基於速度換向與穩定性的動態切割)
        不再死守固定的起點高度，而是偵測動作的「開始-換向-穩定」。
        加入髖部與膝蓋角度啟動偵測，避免代償動作導致槓鈴延遲啟動。
        三個數值都先使用 Hampel 與 Rolling 平滑過再計算。
        """
        bar_y_series = bar_df['bar_y'].copy()
        
        # 計算膝蓋與髖部角度
        hip_angles = pd.Series([self._calculate_angle((r.shoulder_x, r.shoulder_y), (r.hip_x, r.hip_y), (r.knee_x, r.knee_y)) 
                      for _, r in pose_df.iterrows()])
        knee_angles = pd.Series([self._calculate_angle((r.hip_x, r.hip_y), (r.knee_x, r.knee_y), (r.ankle_x, r.ankle_y)) 
                       for _, r in pose_df.iterrows()])
        
        # 依序使用 Hampel 平滑與 Rolling 平滑
        bar_y_smooth = apply_rolling_smoothing(apply_hampel_filter(bar_y_series)).bfill().ffill()
        hip_smooth = apply_rolling_smoothing(apply_hampel_filter(hip_angles)).bfill().ffill()
        knee_smooth = apply_rolling_smoothing(apply_hampel_filter(knee_angles)).bfill().ffill()
        
        # 增加: 計算膝蓋/髖關節與槓鈴 X 軸的相對位移 (取絕對值)
        hip_rel_x = (pose_df['hip_x'] - bar_df['bar_x']).abs()
        knee_rel_x = (pose_df['knee_x'] - bar_df['bar_x']).abs()
        hip_rel_x_smooth = apply_rolling_smoothing(apply_hampel_filter(hip_rel_x)).bfill().ffill()
        knee_rel_x_smooth = apply_rolling_smoothing(apply_hampel_filter(knee_rel_x)).bfill().ffill()
        
        # 存回 array
        bar_y = bar_y_smooth.values
        hip_angles_val = hip_smooth.values
        knee_angles_val = knee_smooth.values
        
        # 記錄至 DF 以便畫圖使用
        bar_df['bar_y_smoothed'] = bar_y
        pose_df['hip_angle_smoothed'] = hip_angles_val
        pose_df['knee_angle_smoothed'] = knee_angles_val
        
        # 計算一階導數 (速度/角速度/位移速度)
        bar_v_y = np.gradient(bar_y) * self.fps
        hip_v = np.gradient(hip_angles_val) * self.fps
        knee_v = np.gradient(knee_angles_val) * self.fps
        hip_rel_v = np.gradient(hip_rel_x_smooth.values) * self.fps
        knee_rel_v = np.gradient(knee_rel_x_smooth.values) * self.fps
        
        reps = []
        n_frames = len(bar_y)
        
        # 狀態機變數 (已捨棄，但保留變數)
        state = 0 
        start_idx = 0
        bottom_idx = 0
        
        from scipy.signal import find_peaks

        # 動態閾值 (加入最低雜訊門檻，避免某些訊號標準差太小導致微小雜訊被判定為活躍)
        v_start_thresh = max(np.std(bar_v_y) * 0.5, 3.0)
        hip_start_thresh = max(np.std(hip_v) * 0.5, 3.0)
        knee_start_thresh = max(np.std(knee_v) * 0.5, 3.0)
        hip_rel_start_thresh = max(np.std(hip_rel_v) * 0.5, 3.0)
        knee_rel_start_thresh = max(np.std(knee_rel_v) * 0.5, 3.0)
        
        v_end_thresh = max(np.std(bar_v_y) * 0.5, 3.0)
        hip_end_thresh = max(np.std(hip_v) * 0.5, 3.0)
        knee_end_thresh = max(np.std(knee_v) * 0.5, 3.0)
        hip_rel_end_thresh = max(np.std(hip_rel_v) * 0.5, 3.0)
        knee_rel_end_thresh = max(np.std(knee_rel_v) * 0.5, 3.0)
        
        # 1. 尋找所有深蹲的波谷 (bar_y 的波峰，因為 y 向下為正)
        # prominence=100 確保起槓或收槓時的小碎步被忽略
        bottoms, _ = find_peaks(bar_y, prominence=100)
        
        reps = []
        for i_peak, bottom_idx in enumerate(bottoms):
            is_first = (i_peak == 0)
            is_last = (i_peak == len(bottoms) - 1)
            
            # --- 2. 尋找起始點 ---
            start_idx = -1
            if is_first:
                # 第一下：往回推找 is_idle (因為前面沒有其他波谷，必須仰賴靜止來排除預備碎步)
                highest_point_before = bottom_idx
                in_descent = False
                for i in range(bottom_idx, 0, -1):
                    if bar_y[i] < bar_y[highest_point_before]:
                        highest_point_before = i
                        
                    v = bar_v_y[i]
                    hv = hip_v[i]
                    kv = knee_v[i]
                    hrv = hip_rel_v[i]
                    krv = knee_rel_v[i]
                    
                    is_active = (
                        (v > v_start_thresh) or (hv < -hip_start_thresh) or (kv < -knee_start_thresh) or
                        (abs(hrv) > hip_rel_start_thresh) or (abs(krv) > knee_rel_start_thresh)
                    )
                    is_idle = (
                        (abs(v) < v_start_thresh) and (abs(hv) < hip_start_thresh) and (abs(kv) < knee_start_thresh) and
                        (abs(hrv) < hip_rel_start_thresh) and (abs(krv) < knee_rel_start_thresh)
                    )
                    
                    if not in_descent:
                        if is_active:
                            in_descent = True
                    else:
                        if is_idle and (bar_y[bottom_idx] - bar_y[i] >= 100):
                            start_idx = i
                            break
                            
                if start_idx == -1:
                    start_idx = highest_point_before
            else:
                # 中間：不使用 is_idle。先找到與上一組之間的最高點，然後「往後推」找第一個滿足 0.2 std 的點 (啟動點)
                prev_bottom = bottoms[i_peak - 1]
                highest_point_before = bottom_idx
                for i in range(bottom_idx, prev_bottom, -1):
                    if bar_y[i] < bar_y[highest_point_before]:
                        highest_point_before = i
                        
                start_idx = highest_point_before
                
                v_ready = (bar_v_y[highest_point_before] <= v_start_thresh)
                hv_ready = (hip_v[highest_point_before] >= -hip_start_thresh)
                kv_ready = (knee_v[highest_point_before] >= -knee_start_thresh)
                hrv_ready = (abs(hip_rel_v[highest_point_before]) <= hip_rel_start_thresh)
                krv_ready = (abs(knee_rel_v[highest_point_before]) <= knee_rel_start_thresh)
                
                active_consecutive = 0
                for i in range(highest_point_before, bottom_idx):
                    v = bar_v_y[i]
                    hv = hip_v[i]
                    kv = knee_v[i]
                    hrv = hip_rel_v[i]
                    krv = knee_rel_v[i]
                    
                    if v <= v_start_thresh: v_ready = True
                    if hv >= -hip_start_thresh: hv_ready = True
                    if kv >= -knee_start_thresh: kv_ready = True
                    if abs(hrv) <= hip_rel_start_thresh: hrv_ready = True
                    if abs(krv) <= knee_rel_start_thresh: krv_ready = True
                    
                    is_active_frame = False
                    if v_ready and (v > v_start_thresh): is_active_frame = True
                    if hv_ready and (hv < -hip_start_thresh): is_active_frame = True
                    if kv_ready and (kv < -knee_start_thresh): is_active_frame = True
                    if hrv_ready and (abs(hrv) > hip_rel_start_thresh): is_active_frame = True
                    if krv_ready and (abs(krv) > knee_rel_start_thresh): is_active_frame = True
                    
                    if is_active_frame:
                        active_consecutive += 1
                        if active_consecutive >= 3:
                            start_cand = i - 2
                            if (bar_y[bottom_idx] - bar_y[start_cand] >= 100):
                                start_idx = start_cand
                                break
                    else:
                        active_consecutive = 0
                        
            # --- 3. 尋找結束點 ---
            end_idx = -1
            if is_last:
                # 最後一下：往前推找 is_idle (排除做完走回架上的碎步)
                highest_point_after = bottom_idx
                in_ascent = False
                for i in range(bottom_idx, n_frames):
                    if bar_y[i] < bar_y[highest_point_after]:
                        highest_point_after = i
                        
                    v = bar_v_y[i]
                    hv = hip_v[i]
                    kv = knee_v[i]
                    hrv = hip_rel_v[i]
                    krv = knee_rel_v[i]
                    
                    is_active = (
                        (v < -v_end_thresh) or (hv > hip_end_thresh) or (kv > knee_end_thresh) or
                        (abs(hrv) > hip_rel_end_thresh) or (abs(krv) > knee_rel_end_thresh)
                    )
                    is_idle = (
                        (abs(v) < v_end_thresh) and (abs(hv) < hip_end_thresh) and (abs(kv) < knee_end_thresh) and
                        (abs(hrv) < hip_rel_end_thresh) and (abs(krv) < knee_rel_end_thresh)
                    )
                    
                    if not in_ascent:
                        if is_active:
                            in_ascent = True
                    else:
                        if is_idle and (bar_y[bottom_idx] - bar_y[i] >= 100):
                            end_idx = i
                            break
                            
                if end_idx == -1:
                    end_idx = highest_point_after
            else:
                # 中間：不使用 is_idle。找到與下一組之間的最高點，然後「往回推」找剛好觸發 0.2 std 的點 (結束點)
                next_bottom = bottoms[i_peak + 1]
                highest_point_after = bottom_idx
                for i in range(bottom_idx, next_bottom):
                    if bar_y[i] < bar_y[highest_point_after]:
                        highest_point_after = i
                        
                end_idx = highest_point_after
                
                v_ready = (bar_v_y[highest_point_after] >= -v_end_thresh)
                hv_ready = (hip_v[highest_point_after] <= hip_end_thresh)
                kv_ready = (knee_v[highest_point_after] <= knee_end_thresh)
                hrv_ready = (abs(hip_rel_v[highest_point_after]) <= hip_rel_end_thresh)
                krv_ready = (abs(knee_rel_v[highest_point_after]) <= knee_rel_end_thresh)
                
                active_consecutive = 0
                for i in range(highest_point_after, bottom_idx, -1):
                    v = bar_v_y[i]
                    hv = hip_v[i]
                    kv = knee_v[i]
                    hrv = hip_rel_v[i]
                    krv = knee_rel_v[i]
                    
                    if v >= -v_end_thresh: v_ready = True
                    if hv <= hip_end_thresh: hv_ready = True
                    if kv <= knee_end_thresh: kv_ready = True
                    if abs(hrv) <= hip_rel_end_thresh: hrv_ready = True
                    if abs(krv) <= knee_rel_end_thresh: krv_ready = True
                    
                    is_active_frame = False
                    if v_ready and (v < -v_end_thresh): is_active_frame = True
                    if hv_ready and (hv > hip_end_thresh): is_active_frame = True
                    if kv_ready and (kv > knee_end_thresh): is_active_frame = True
                    if hrv_ready and (abs(hrv) > hip_rel_end_thresh): is_active_frame = True
                    if krv_ready and (abs(krv) > knee_rel_end_thresh): is_active_frame = True
                    
                    if is_active_frame:
                        active_consecutive += 1
                        if active_consecutive >= 3:
                            end_cand = i + 2
                            if (bar_y[bottom_idx] - bar_y[end_cand] >= 100):
                                end_idx = end_cand
                                break
                    else:
                        active_consecutive = 0
                        
            # 確保找到合理的區間且至少有 100 的落差
            if end_idx > start_idx and (end_idx - start_idx > self.fps // 2):
                if (bar_y[bottom_idx] - bar_y[start_idx] >= 100) and (bar_y[bottom_idx] - bar_y[end_idx] >= 100):
                    reps.append({
                        'start': start_idx,
                        'bottom': bottom_idx,
                        'end': end_idx
                    })
                
        return reps

    def extract_timeseries(self, pose_df, bar_df):
        """
        第三階段：逐幀時序特徵擷取
        """
        pose_clean, bar_clean = self._preprocess(pose_df, bar_df)
        rep_intervals = self._segment_reps(pose_clean, bar_clean)
        
        all_timeseries = []
        
        for i, rep in enumerate(rep_intervals):
            s, b, e = rep['start'], rep['bottom'], rep['end']
            rep_id = i + 1
            
            # 1. 計算該 Rep 的動態基準
            ref_start = max(0, s - 15)
            ref_window = pose_clean.iloc[ref_start:s]
            ref_bar_window = bar_clean.iloc[ref_start:s]
            
            ref_bar_x = ref_bar_window['bar_x'].mean()
            ref_torso_len = np.sqrt((ref_window['shoulder_x'] - ref_window['hip_x'])**2 + 
                                    (ref_window['shoulder_y'] - ref_window['hip_y'])**2).mean() + 1e-6
            ref_femur_len = np.sqrt((ref_window['hip_x'] - ref_window['knee_x'])**2 + 
                                    (ref_window['hip_y'] - ref_window['knee_y'])**2).mean() + 1e-6
            
            # 2. 擷取該 Rep 區間的資料
            rep_pose = pose_clean.iloc[s:e+1].copy()
            rep_bar = bar_clean.iloc[s:e+1].copy()
            
            # 3. 計算逐幀特徵
            # A. 角度
            rep_pose['hip_angle'] = [self._calculate_angle((r.shoulder_x, r.shoulder_y), (r.hip_x, r.hip_y), (r.knee_x, r.knee_y)) 
                                    for _, r in rep_pose.iterrows()]
            rep_pose['knee_angle'] = [self._calculate_angle((r.hip_x, r.hip_y), (r.knee_x, r.knee_y), (r.ankle_x, r.ankle_y)) 
                                     for _, r in rep_pose.iterrows()]
            rep_pose['femur_horiz_angle'] = [self._calculate_femur_horizontal_angle((r.hip_x, r.hip_y), (r.knee_x, r.knee_y)) 
                                            for _, r in rep_pose.iterrows()]
            
            # B. 正規化位移
            rep_pose['norm_knee_fwd'] = (rep_pose['knee_x'] - ref_bar_x) / ref_femur_len
            rep_pose['norm_hip_bwd'] = (ref_bar_x - rep_pose['hip_x']) / ref_femur_len
            rep_bar['norm_bar_dev'] = (rep_bar['bar_x'] - ref_bar_x) / ref_torso_len
            
            curr_torso_len = np.sqrt((rep_pose['shoulder_x'] - rep_pose['hip_x'])**2 + 
                                     (rep_pose['shoulder_y'] - rep_pose['hip_y'])**2)
            rep_pose['torso_comp_ratio'] = curr_torso_len / ref_torso_len
            
            # C. 速度與加速度 (一階與二階導數)
            rep_pose['hip_y_velocity'] = np.gradient(rep_pose['hip_y'].values) * self.fps
            rep_bar['bar_y_velocity'] = np.gradient(rep_bar['bar_y'].values) * self.fps
            rep_pose['hip_angle_velocity'] = np.gradient(rep_pose['hip_angle'].values) * self.fps
            rep_pose['knee_angle_velocity'] = np.gradient(rep_pose['knee_angle'].values) * self.fps
            
            rep_pose['hip_y_acceleration'] = np.gradient(rep_pose['hip_y_velocity'].values) * self.fps
            rep_bar['bar_y_acceleration'] = np.gradient(rep_bar['bar_y_velocity'].values) * self.fps
            
            # 合併特徵
            rep_features = pd.DataFrame({
                'frame': range(s, e + 1),
                'rep_id': rep_id,
                'hip_angle': rep_pose['hip_angle'],
                'knee_angle': rep_pose['knee_angle'],
                'femur_horiz_angle': rep_pose['femur_horiz_angle'],
                'norm_knee_fwd': rep_pose['norm_knee_fwd'],
                'norm_hip_bwd': rep_pose['norm_hip_bwd'],
                'bar_x_deviation': rep_bar['norm_bar_dev'],
                'torso_comp_ratio': rep_pose['torso_comp_ratio'],
                'hip_y_velocity': rep_pose['hip_y_velocity'],
                'bar_y_velocity': rep_bar['bar_y_velocity'],
                'hip_angle_velocity': rep_pose['hip_angle_velocity'],
                'knee_angle_velocity': rep_pose['knee_angle_velocity'],
                'hip_y_acceleration': rep_pose['hip_y_acceleration'],
                'bar_y_acceleration': rep_bar['bar_y_acceleration']
            })
            
            all_timeseries.append(rep_features)
            
        return pd.concat(all_timeseries, ignore_index=True) if all_timeseries else pd.DataFrame()

    def fit_transform(self, pose_df, bar_df):
        """
        執行完整分析流程 (Pipeline)
        :param pose_df: YOLOv11 Pose DataFrame
        :param bar_df: YOLOv11 Barbell DataFrame
        :return: (List[Dict] reps_features, reps_timeseries)
        """
        pose_clean, bar_clean = self._preprocess(pose_df, bar_df)

        # 切割動作
        rep_intervals = self._segment_reps(pose_clean, bar_clean)
        
        all_rep_features = []
        
        for rep in rep_intervals:
            s, b, e = rep['start'], rep['bottom'], rep['end']
            
            # 1. 計算該 Rep 的動態基準常數 (Reference Values) - 取下放前 15 幀
            ref_start = max(0, s - 15)
            ref_end = s
            ref_window = pose_clean.iloc[ref_start:ref_end]
            ref_bar_window = bar_clean.iloc[ref_start:ref_end]
            
            ref_bar_x = ref_bar_window['bar_x'].mean()
            
            # 軀幹與股骨長度基準 (用來消除身高差異)
            torso_lens = np.sqrt((ref_window['shoulder_x'] - ref_window['hip_x'])**2 + 
                                 (ref_window['shoulder_y'] - ref_window['hip_y'])**2)
            ref_torso_len = torso_lens.mean() + 1e-6
            
            femur_lens = np.sqrt((ref_window['hip_x'] - ref_window['knee_x'])**2 + 
                                 (ref_window['hip_y'] - ref_window['knee_y'])**2)
            ref_femur_len = femur_lens.mean() + 1e-6
            
            # 第三階段：逐幀時序特徵 (Frame-by-Frame)
            rep_pose = pose_clean.iloc[s:e+1]
            rep_bar = bar_clean.iloc[s:e+1]
            
            # A. 尺度不變角度
            hip_angles = [self._calculate_angle((r.shoulder_x, r.shoulder_y), (r.hip_x, r.hip_y), (r.knee_x, r.knee_y)) 
                          for _, r in rep_pose.iterrows()]
            knee_angles = [self._calculate_angle((r.hip_x, r.hip_y), (r.knee_x, r.knee_y), (r.ankle_x, r.ankle_y)) 
                           for _, r in rep_pose.iterrows()]
            femur_horiz_angles = [self._calculate_femur_horizontal_angle((r.hip_x, r.hip_y), (r.knee_x, r.knee_y)) 
                                  for _, r in rep_pose.iterrows()]
            
            # B. 正規化位移 (除以骨骼長度)
            norm_knee_fwd = (rep_pose['knee_x'].values - ref_bar_x) / ref_femur_len
            norm_hip_bwd = (ref_bar_x - rep_pose['hip_x'].values) / ref_femur_len
            norm_bar_dev = (rep_bar['bar_x'].values - ref_bar_x) / ref_torso_len
            
            curr_torso_len = np.sqrt((rep_pose['shoulder_x'] - rep_pose['hip_x'])**2 + 
                                     (rep_pose['shoulder_y'] - rep_pose['hip_y'])**2)
            torso_comp_ratio = curr_torso_len / ref_torso_len
            
            # C. 速度與角速度
            hip_v_y = np.gradient(rep_pose['hip_y'].values) * self.fps
            bar_v_y = np.gradient(rep_bar['bar_y'].values) * self.fps
            hip_ang_vel = np.gradient(hip_angles) * self.fps
            knee_ang_vel = np.gradient(knee_angles) * self.fps
            
            # 第四階段：聚合特徵 (Aggregated Rep Features)
            bottom_rel = b - s # 相對於 Rep 開始的索引
            
            # 針對錯誤 1: 下蹲深度不足
            f_min_squat_depth = np.min(femur_horiz_angles)
            
            # 針對錯誤 2: 骨盆後傾/屁股眨眼 (最低點前後 10 幀的軀幹長度縮短比例)
            t_start = max(0, bottom_rel - 10)
            t_end = min(len(torso_comp_ratio), bottom_rel + 11)
            f_min_torso_ratio = np.min(torso_comp_ratio[t_start:t_end])
            
            # 針對錯誤 3: 臀部上升過快 (起立階段前 30% 時間內，髖部與槓鈴的垂直速度比)
            ascent_len = len(bar_v_y) - bottom_rel
            ascent_30_idx = bottom_rel + int(ascent_len * 0.3)
            hip_v_mean = np.mean(hip_v_y[bottom_rel:ascent_30_idx+1])
            bar_v_mean = np.mean(bar_v_y[bottom_rel:ascent_30_idx+1])
            f_concentric_v_ratio = hip_v_mean / (bar_v_mean + 1e-6)
            
            # 針對錯誤 4 & 5: 下放階段髖/膝主導 (前 0.5 秒內角速度比)
            ecc_05_idx = int(0.5 * self.fps)
            f_eccentric_init_ang_ratio = np.mean(hip_ang_vel[:ecc_05_idx+1]) / \
                                         (np.mean(knee_ang_vel[:ecc_05_idx+1]) + 1e-6)
            
            f_max_knee_fwd = np.max(norm_knee_fwd[:bottom_rel+1])
            f_max_hip_bwd = np.max(norm_hip_bwd[:bottom_rel+1])
            
            # 系統性穩定度: 槓鈴鉛垂偏差標準差
            f_bar_path_std = np.std(norm_bar_dev)
            
            all_rep_features.append({
                'rep_id': len(all_rep_features) + 1,
                'start_frame': s,
                'bottom_frame': b,
                'end_frame': e,
                'f_min_squat_depth': f_min_squat_depth,
                'f_min_torso_ratio': f_min_torso_ratio,
                'f_concentric_v_ratio': f_concentric_v_ratio,
                'f_eccentric_init_ang_ratio': f_eccentric_init_ang_ratio,
                'f_max_knee_fwd': f_max_knee_fwd,
                'f_max_hip_bwd': f_max_hip_bwd,
                'f_bar_path_std': f_bar_path_std
            })
            
        return pd.DataFrame(all_rep_features)
