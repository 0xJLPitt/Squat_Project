import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

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

    def _segment_reps(self, bar_df):
        """
        第二階段：動作切割 (基於速度換向與穩定性的動態切割)
        不再死守固定的起點高度，而是偵測動作的「開始-換向-穩定」。
        """
        bar_y = bar_df['bar_y'].values
        bar_v_y = np.gradient(bar_y) * self.fps
        
        reps = []
        n_frames = len(bar_y)
        
        # 狀態機變數
        # 0: IDLE (等待下蹲), 1: DESCENDING (下蹲中), 2: ASCENDING (起立中)
        state = 0 
        start_idx = 0
        bottom_idx = 0
        
        # 動態閾值
        v_thresh = np.std(bar_v_y) * 0.5 if np.std(bar_v_y) > 0 else 1.0
        
        for i in range(1, n_frames):
            v = bar_v_y[i]
            
            if state == 0: # IDLE
                if v > v_thresh and i > 10: # 開始下蹲
                    state = 1
                    start_idx = i
                    bottom_idx = i
            
            elif state == 1: # DESCENDING
                if bar_y[i] > bar_y[bottom_idx]:
                    bottom_idx = i
                
                if v < -v_thresh: # 已經開始向上起立
                    state = 2
            
            elif state == 2: # ASCENDING
                # 判斷是否穩定下來 (結束動作)
                if abs(v) < v_thresh and i > bottom_idx + 5:
                    # 確保動作有足夠長度 (至少 0.5 秒)
                    if i - start_idx > self.fps // 2:
                        reps.append({
                            'start': start_idx,
                            'bottom': bottom_idx,
                            'end': i
                        })
                    state = 0 # 回到等待下一組
                    
        return reps

    def extract_timeseries(self, pose_df, bar_df):
        """
        第三階段：逐幀時序特徵擷取
        :return: 包含逐幀特徵的 DataFrame，並標註所屬 Rep ID
        """
        pose_clean, bar_clean = self._preprocess(pose_df, bar_df)
        rep_intervals = self._segment_reps(bar_clean)
        
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
        主處理流程：Raw Data -> Preprocessing -> Segmentation -> Feature Extraction -> Aggregation
        :param pose_df: YOLOv26 Pose DataFrame
        :param bar_df: YOLOv11 Barbell DataFrame
        :return: 每一下深蹲一個 Row 的特徵 DataFrame
        """
        # 第一階段：前處理
        pose_clean, bar_clean = self._preprocess(pose_df, bar_df)
        
        # 第二階段：動作切割
        rep_intervals = self._segment_reps(bar_clean)
        
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
