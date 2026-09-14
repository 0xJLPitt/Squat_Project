#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
Step 2: 臥推關鍵點判斷與動作切片預處理演算法
(Bench Press Keypoints Kinematic Preprocess Algorithm)
================================================================================
專門針對 YOLO 2D 骨架中的「左手手腕 (Left Wrist, 12 點骨架 Index 4 / 舊版 17 點 Index 9)」進行運動學特徵分析，
自動精準擷取每次臥推動作的 5 大關鍵影格 (Keyframes)：
  1. 頂點 (起) [Top Start / Lockout]   : 出槓/推起至最高鎖定點 (Y 座標局部極小值)
  2. 下降中段  [Descent Mid, 50% ROM]  : 下放過程通過 50% 行程垂直高度差影格
  3. 底點      [Bottom / Chest Touch]  : 槓鈴觸胸最低點 (Y 座標局部極大值)
  4. 上升中段  [Ascent Mid, 50% ROM]   : 推升過程通過 50% 行程垂直高度差影格
  5. 頂點 (訖) [Top End / Lockout]     : 推至頂部鎖定點 (Y 座標局部極小值)

--------------------------------------------------------------------------------
📖 使用方法一：命令列 (CLI Terminal) 直接執行
--------------------------------------------------------------------------------
1. 受試者母目錄全自動批次處理 (推薦，自動分析 dataprocess 下所有骨架 txt):
   $ mamba run -n hw1 python step2_benchpress_label_preprocess_alog.py --dir "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2"

2. 指定單一骨架檔測試:
   $ mamba run -n hw1 python step2_benchpress_label_preprocess_alog.py --input "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\dataprocess\yolo_skeleton_error1.txt"

3. 自訂輸出路徑與影格率:
   $ mamba run -n hw1 python step2_benchpress_label_preprocess_alog.py \
       --input "path/to/yolo_skeleton.txt" \
       --output-json "path/to/output.json" \
       --output-csv "path/to/output.csv" \
       --plot "path/to/output.png" \
       --fps 30.0

--------------------------------------------------------------------------------
💻 使用方法二：Python 腳本匯入呼叫 (Python API)
--------------------------------------------------------------------------------
【寫法 A：一鍵高階 API 函式 (推薦)】
    from step2_benchpress_label_preprocess_alog import extract_benchpress_key_frames

    reps = extract_benchpress_key_frames(
        txt_path=r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\dataprocess\yolo_skeleton_normal.txt",
        output_json="benchpress_segments.json",  # 選填，設為 None 則不存
        output_csv="benchpress_segments.csv",    # 選填
        plot_path="benchpress_plot.png",         # 選填
        fps=30.0
    )

    # 查看分析結果 (回傳為 list of dict)
    for rep in reps:
        print(f"Rep {rep['rep']}: Top={rep['top_start_frame']}, Bottom={rep['bottom_frame']}")

【寫法 B：物件導向客製化呼叫 (支援進階調整與取得 DataFrame)】
    from step2_benchpress_label_preprocess_alog import BenchpressWristAnalyzer

    analyzer = BenchpressWristAnalyzer(fps=30.0, min_prominence=35.0)
    analyzer.load_skeleton_file(r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\dataprocess\yolo_skeleton_normal.txt")
    analyzer.preprocess_left_wrist()
    reps = analyzer.extract_keyframes()

    # 取得 Pandas DataFrame 表格
    df_summary = analyzer.get_summary_dataframe()
    print(df_summary)

--------------------------------------------------------------------------------
⚙️ 如何手動關閉 CSV 表格 或 PNG 圖表輸出 (不刪除程式碼)
--------------------------------------------------------------------------------
【方法一：在程式碼開關區手動切換 (推薦)】
  至本程式碼約第 500 行「⚙️ 輸出開關手動控制區」直接修改變數：
    ENABLE_SAVE_CSV  = False   # 設為 False 即可關閉 *_segments.csv 產出
    ENABLE_SAVE_PLOT = False   # 設為 False 即可關閉 *_segmentation.png 產出

【方法二：在命令列執行時加入關閉參數】
  $ mamba run -n hw1 python step2_benchpress_label_preprocess_alog.py --no-csv
  $ mamba run -n hw1 python step2_benchpress_label_preprocess_alog.py --no-plot
  $ mamba run -n hw1 python step2_benchpress_label_preprocess_alog.py --no-csv --no-plot

【方法三：在 Python API 呼叫時關閉】
  reps = extract_benchpress_key_frames(..., save_csv=False, save_plot=False)
================================================================================
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import matplotlib
matplotlib.use('Agg')  # 預設非互動式後端，避免無 GUI 環境報錯
import matplotlib.pyplot as plt


class BenchpressWristAnalyzer:
    """
    臥推左手手腕運動學特徵分析器
    """
    LEFT_WRIST_JOINT_ID = 4  # 12 身體關鍵點定義: 4 代表 left_wrist (舊版 17 點格式為 9)

    def __init__(
        self,
        fps: float = 30.0,
        smooth_window: int = 15,
        polyorder: int = 3,
        min_prominence: float = 35.0,
        min_rep_distance: int = 20,
        y_bottom_max_threshold: float = 450.0,
        x_bench_range: tuple = (600.0, 1050.0),
    ):
        """
        :param fps: 影片幀率 (預設 30 fps)
        :param smooth_window: Savitzky-Golay 濾波器窗口長度 (必須為奇數)
        :param polyorder: Savitzky-Golay 濾波器多項式階數
        :param min_prominence: 尋找波峰 (底點) 的最小突顯度 (像素值)
        :param min_rep_distance: 兩次反覆底點之間的最小影格間隔
        :param y_bottom_max_threshold: 正常臥推底點的 Y 軸上限 (超過代表受試者起身坐起或異常)
        :param x_bench_range: 受試者在臥推椅上的 X 軸合理範圍 (min_x, max_x)
        """
        self.fps = fps
        self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        self.polyorder = polyorder
        self.min_prominence = min_prominence
        self.min_rep_distance = min_rep_distance
        self.y_bottom_max_threshold = y_bottom_max_threshold
        self.x_bench_range = x_bench_range

        # 內部狀態
        self.raw_df = None
        self.wrist_df = None
        self.y_smooth = None
        self.x_smooth = None
        self.v_y = None
        self.reps_info = []

    def load_skeleton_file(self, filepath: str) -> pd.DataFrame:
        """
        讀取 YOLO skeleton 文字檔 (格式: frame, joint, x, y [, confidence])
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到指定的骨架檔案: {filepath}")

        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [float(val) for val in line.split(',')]
                records.append(parts)

        if not records:
            raise ValueError(f"骨架檔案為空: {filepath}")

        num_cols = len(records[0])
        if num_cols == 4:
            col_names = ['frame', 'joint', 'x', 'y']
        elif num_cols >= 5:
            col_names = ['frame', 'joint', 'x', 'y', 'conf'] + [f'extra_{i}' for i in range(num_cols - 5)]
        else:
            raise ValueError(f"未知的格式欄位數: {num_cols} (預期至少為 frame, joint, x, y)")

        df = pd.DataFrame(records, columns=col_names)
        self.raw_df = df
        return df

    def preprocess_left_wrist(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        篩選左手手腕 (12 點格式預設 Joint 4，智慧相容舊版 17 點 Joint 9)，進行異常值濾除、插值補全與平滑化處理
        """
        if df is None:
            df = self.raw_df
        if df is None:
            raise ValueError("請先執行 load_skeleton_file 讀取資料！")

        # 智慧判斷 12 點或 17 點骨架格式
        max_joint = df['joint'].max() if len(df) > 0 else 11
        if max_joint >= 16:
            target_joint_id = 9  # 舊版 COCO 17 點格式
        else:
            target_joint_id = self.LEFT_WRIST_JOINT_ID  # 12 點格式預設 (4)
        self.current_wrist_joint_id = target_joint_id

        # 1. 僅抓取左手手腕
        wrist = df[df['joint'] == target_joint_id].sort_values('frame').copy().reset_index(drop=True)
        if len(wrist) == 0:
            raise ValueError(f"資料中未找到關節索引為 {target_joint_id} 的左手手腕骨架點！")

        # 2. 標記非有效值 (座標為 0 代表偵測遺漏/未入鏡)
        mask_zero = (wrist['x'] <= 0) | (wrist['y'] <= 0)
        wrist.loc[mask_zero, ['x', 'y']] = np.nan

        # 3. 線性插值填補中間遺失影格
        wrist['y'] = wrist['y'].interpolate(method='linear').ffill().bfill()
        wrist['x'] = wrist['x'].interpolate(method='linear').ffill().bfill()

        # 4. 裁切影片結尾全為 0 的無效影格 (如受試者已完全走開或攝影中斷)
        valid_indices = np.where(~mask_zero)[0]
        if len(valid_indices) > 0:
            last_valid_idx = valid_indices[-1]
            wrist = wrist.iloc[:last_valid_idx + 1].copy().reset_index(drop=True)

        # 5. Savitzky-Golay 濾波平滑化
        y_vals = wrist['y'].values
        x_vals = wrist['x'].values
        
        # 確保資料長度大於平滑窗長
        effective_window = min(self.smooth_window, len(y_vals) if len(y_vals) % 2 == 1 else len(y_vals) - 1)
        if effective_window < self.polyorder + 2:
            effective_window = self.polyorder + 2
            if effective_window % 2 == 0:
                effective_window += 1

        self.y_smooth = savgol_filter(y_vals, effective_window, self.polyorder)
        self.x_smooth = savgol_filter(x_vals, effective_window, self.polyorder)
        wrist['y_smooth'] = self.y_smooth
        wrist['x_smooth'] = self.x_smooth

        # 6. 計算 Y 軸速度 (pixel / s)
        # 在像素座標中，Y 軸朝下為正：
        # 下降 (槓鈴下落至胸口): y 遞增，v_y > 0
        # 上推 (槓鈴推離胸口): y 遞減，v_y < 0
        self.v_y = np.gradient(self.y_smooth) * self.fps
        wrist['v_y'] = self.v_y

        self.wrist_df = wrist
        return wrist

    def extract_keyframes(self) -> list:
        """
        核心演算法：精準擷取每次反覆的頂點、底點、與上下兩段中間幀
        :return: 包含每次反覆關鍵幀資訊的字典列表
        """
        if self.wrist_df is None or self.y_smooth is None:
            self.preprocess_left_wrist()

        wrist = self.wrist_df
        frames = wrist['frame'].values.astype(int)
        y_smooth = self.y_smooth
        v_y = self.v_y

        # --- 第一階段：尋找所有反覆的底點 (Bottom Peaks) ---
        # 臥推動作中，手腕下落至胸前時，Y 座標達到最大值（波峰）
        raw_bottoms, props = find_peaks(
            y_smooth,
            prominence=self.min_prominence,
            distance=self.min_rep_distance
        )

        # 初始過濾：排除受試者站立/走動時 (Y > 400 或 X 嚴重偏離臥推椅)
        valid_candidates = []
        for b_idx in raw_bottoms:
            b_y = y_smooth[b_idx]
            b_x = wrist['x_smooth'].iloc[b_idx]
            if b_y <= 380.0 and (self.x_bench_range[0] <= b_x <= self.x_bench_range[1]):
                valid_candidates.append(b_idx)

        # 自適應群聚過濾：臥推動作每次觸胸深度高度一致 (Y 值標準差極小)
        # 若有起槓前或完成後起身等離散峰值 (如坐起)，與主要群體高度差異會超過 40 px，予以排除
        clean_bottoms = []
        if valid_candidates:
            cand_ys = [y_smooth[b] for b in valid_candidates]
            median_y = np.median(cand_ys)
            for b_idx in valid_candidates:
                if abs(y_smooth[b_idx] - median_y) < 35.0:
                    clean_bottoms.append(b_idx)

        if not clean_bottoms:
            print("警告：未偵測到任何符合條件的臥推反覆底點！")
            self.reps_info = []
            return []

        # --- 第二階段：配對每次反覆的起始頂點、結束頂點與中間段 ---
        reps = []
        num_bottoms = len(clean_bottoms)

        for i, b_idx in enumerate(clean_bottoms):
            # 1. 尋找起始頂點 (Top Start)
            if i == 0:
                # 第一下：往回搜尋出槓後的最高鎖定位置
                search_start = max(0, b_idx - int(self.fps * 3.0))  # 往回最多找 3 秒
                local_top_idx = search_start + np.argmin(y_smooth[search_start:b_idx])
                # 微調：精準定位下落剛開始的時刻 (向下速度顯著增加)
                t_start = local_top_idx
                for k in range(local_top_idx, b_idx):
                    if v_y[k] > 15.0 and (y_smooth[k] - y_smooth[local_top_idx]) > 5.0:
                        t_start = max(local_top_idx, k - 2)
                        break
            else:
                # 中間次數：上一動底點與本次底點之間的 Y 最小值（物理最高點）
                prev_b = clean_bottoms[i - 1]
                t_start = prev_b + np.argmin(y_smooth[prev_b:b_idx])

            # 2. 尋找結束頂點 (Top End)
            if i == num_bottoms - 1:
                # 最後一下：往後搜尋推至頂部鎖定且停頓的位置 (避免納入最後放回槓架的軌跡)
                search_end = min(len(y_smooth) - 1, b_idx + int(self.fps * 3.0))
                local_end_idx = b_idx + np.argmin(y_smooth[b_idx:search_end])
                t_end = local_end_idx
                # 尋找向上推停頓瞬間 (速度回升到接近 0)
                for k in range(b_idx, search_end):
                    if v_y[k] >= -5.0 and (y_smooth[b_idx] - y_smooth[k]) > 40.0:
                        t_end = k
                        break
            else:
                # 中間次數：本次底點與下一動底點之間的 Y 最小值
                next_b = clean_bottoms[i + 1]
                t_end = b_idx + np.argmin(y_smooth[b_idx:next_b])

            # 3. 尋找下降中間段 (Descent Midpoint)
            # 標準運動學定義：起始頂點至底點之 50% 行程垂直高度差 (50% ROM)
            y_descent_target = (y_smooth[t_start] + y_smooth[b_idx]) / 2.0
            descent_range = list(range(t_start, b_idx + 1))
            if descent_range:
                d_mid_idx = descent_range[np.argmin(np.abs(y_smooth[descent_range] - y_descent_target))]
                # 同時計算最大下沉速度幀與時間中點幀
                d_max_v_idx = descent_range[np.argmax(v_y[descent_range])]
            else:
                d_mid_idx = t_start
                d_max_v_idx = t_start

            f_descent_time_mid = int(round((frames[t_start] + frames[b_idx]) / 2.0))

            # 4. 尋找上升中間段 (Ascent Midpoint)
            # 標準運動學定義：底點至結束頂點之 50% 行程垂直高度差 (50% ROM)
            y_ascent_target = (y_smooth[b_idx] + y_smooth[t_end]) / 2.0
            ascent_range = list(range(b_idx, t_end + 1))
            if ascent_range:
                a_mid_idx = ascent_range[np.argmin(np.abs(y_smooth[ascent_range] - y_ascent_target))]
                # 同時計算最大推升速度幀 (v_y 最負值)
                a_max_v_idx = ascent_range[np.argmin(v_y[ascent_range])]
            else:
                a_mid_idx = b_idx
                a_max_v_idx = b_idx

            f_ascent_time_mid = int(round((frames[b_idx] + frames[t_end]) / 2.0))

            # 記錄結果
            rep_data = {
                "rep": i + 1,
                "top_start_frame": int(frames[t_start]),
                "descent_mid_frame": int(frames[d_mid_idx]),
                "bottom_frame": int(frames[b_idx]),
                "ascent_mid_frame": int(frames[a_mid_idx]),
                "top_end_frame": int(frames[t_end]),
                "indices": {
                    "top_start": int(t_start),
                    "descent_mid": int(d_mid_idx),
                    "bottom": int(b_idx),
                    "ascent_mid": int(a_mid_idx),
                    "top_end": int(t_end)
                },
                "coordinates": {
                    "top_start": {"y": round(float(y_smooth[t_start]), 1), "x": round(float(wrist['x_smooth'].iloc[t_start]), 1)},
                    "descent_mid": {"y": round(float(y_smooth[d_mid_idx]), 1), "x": round(float(wrist['x_smooth'].iloc[d_mid_idx]), 1)},
                    "bottom": {"y": round(float(y_smooth[b_idx]), 1), "x": round(float(wrist['x_smooth'].iloc[b_idx]), 1)},
                    "ascent_mid": {"y": round(float(y_smooth[a_mid_idx]), 1), "x": round(float(wrist['x_smooth'].iloc[a_mid_idx]), 1)},
                    "top_end": {"y": round(float(y_smooth[t_end]), 1), "x": round(float(wrist['x_smooth'].iloc[t_end]), 1)}
                },
                "supplementary_midpoints": {
                    "descent_mid_time_frame": f_descent_time_mid,
                    "ascent_mid_time_frame": f_ascent_time_mid,
                    "descent_max_v_frame": int(frames[d_max_v_idx]),
                    "ascent_max_v_frame": int(frames[a_max_v_idx])
                }
            }
            reps.append(rep_data)

        self.reps_info = reps
        return reps

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        產出簡明扼要的 DataFrame 格式表格
        """
        if not self.reps_info:
            return pd.DataFrame()

        rows = []
        for r in self.reps_info:
            rows.append({
                "Rep": r["rep"],
                "頂點(起) Frame": r["top_start_frame"],
                "下降中段 Frame": r["descent_mid_frame"],
                "底點 Frame": r["bottom_frame"],
                "上升中段 Frame": r["ascent_mid_frame"],
                "頂點(訖) Frame": r["top_end_frame"],
                "下降耗時(s)": round((r["bottom_frame"] - r["top_start_frame"]) / self.fps, 2),
                "上升耗時(s)": round((r["top_end_frame"] - r["bottom_frame"]) / self.fps, 2),
                "垂直行程(px)": round(r["coordinates"]["bottom"]["y"] - r["coordinates"]["top_start"]["y"], 1)
            })
        return pd.DataFrame(rows)

    def plot_segmentation(self, save_path: str = None, title: str = "Bench Press Left Wrist Trajectory Analysis") -> None:
        """
        繪製高品質波形與關鍵點視覺化標註圖
        """
        if not self.reps_info or self.wrist_df is None:
            print("無分析數據可繪圖。")
            return

        wrist = self.wrist_df
        frames = wrist['frame'].values
        y_raw = wrist['y'].values
        y_smooth = self.y_smooth
        v_y = self.v_y

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.5]})

        # --- Subplot 1: Y 軸軌跡與關鍵幀標記 ---
        ax1.plot(frames, y_raw, color='gray', alpha=0.35, label='Raw Left Wrist Y', linewidth=1)
        ax1.plot(frames, y_smooth, color='#1f77b4', linewidth=2.2, label='Smoothed Y (Savitzky-Golay)')

        # 標記關鍵點
        top_start_f = [r["top_start_frame"] for r in self.reps_info]
        top_start_y = [r["coordinates"]["top_start"]["y"] for r in self.reps_info]

        descent_mid_f = [r["descent_mid_frame"] for r in self.reps_info]
        descent_mid_y = [r["coordinates"]["descent_mid"]["y"] for r in self.reps_info]

        bottom_f = [r["bottom_frame"] for r in self.reps_info]
        bottom_y = [r["coordinates"]["bottom"]["y"] for r in self.reps_info]

        ascent_mid_f = [r["ascent_mid_frame"] for r in self.reps_info]
        ascent_mid_y = [r["coordinates"]["ascent_mid"]["y"] for r in self.reps_info]

        top_end_f = [r["top_end_frame"] for r in self.reps_info]
        top_end_y = [r["coordinates"]["top_end"]["y"] for r in self.reps_info]

        ax1.scatter(top_start_f, top_start_y, color='#2ca02c', s=70, zorder=5, marker='^', label='Top (Lockout)')
        ax1.scatter(descent_mid_f, descent_mid_y, color='#ff7f0e', s=50, zorder=5, marker='o', label='Descent Mid (50% ROM)')
        ax1.scatter(bottom_f, bottom_y, color='#d62728', s=90, zorder=5, marker='v', label='Bottom (Chest Touch)')
        ax1.scatter(ascent_mid_f, ascent_mid_y, color='#9467bd', s=50, zorder=5, marker='s', label='Ascent Mid (50% ROM)')

        # 為每次反覆加上反覆編號文字標註
        for r in self.reps_info:
            bf = r["bottom_frame"]
            by = r["coordinates"]["bottom"]["y"]
            ax1.text(bf, by + 12, f'Rep {r["rep"]}', ha='center', va='top', fontsize=9, fontweight='bold', color='#8c1111')

        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Vertical Position Y (px, downward+)', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper right', framealpha=0.9)

        # --- Subplot 2: 垂直速度曲線與零交越點 ---
        ax2.plot(frames, v_y, color='#333333', linewidth=1.5, label='Vertical Velocity $v_y$ (px/s)')
        ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

        # 底點處垂直畫虛線
        for bf in bottom_f:
            ax2.axvline(bf, color='#d62728', linestyle=':', alpha=0.5)

        ax2.set_ylabel('Velocity (px/s)', fontsize=11)
        ax2.set_xlabel('Frame Number', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='upper right', framealpha=0.9)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=200)
            print(f"視覺化檢查圖已成功儲存至: {save_path}")

        plt.close(fig)

    def export_results(self, json_path: str = None, csv_path: str = None) -> None:
        """
        輸出分析結果至 JSON 與 CSV 格式
        """
        if not self.reps_info:
            print("無分析結果可輸出。")
            return

        if json_path:
            os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
            export_payload = {
                "metadata": {
                    "analyzed_joint": "left_wrist",
                    "joint_id": getattr(self, "current_wrist_joint_id", self.LEFT_WRIST_JOINT_ID),
                    "fps": self.fps,
                    "total_reps": len(self.reps_info)
                },
                "reps": self.reps_info
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(export_payload, f, indent=4, ensure_ascii=False)
            print(f"標註結果已成功匯出 JSON: {json_path}")

        if csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            df = self.get_summary_dataframe()
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"簡明表格已成功匯出 CSV: {csv_path}")


def extract_benchpress_key_frames(
    txt_path: str,
    output_json: str = None,
    output_csv: str = None,
    plot_path: str = None,
    save_json: bool = True,
    save_csv: bool = True,
    save_plot: bool = True,
    fps: float = 30.0
) -> list:
    """
    提供給外部腳本一鍵呼叫的高階 API 函式
    :param save_json: 是否儲存 JSON 檔 (預設 True，若設為 False 則不儲存)
    :param save_csv: 是否儲存 CSV 檔 (預設 True，若設為 False 則不儲存)
    :param save_plot: 是否繪製並儲存 PNG 圖表 (預設 True，若設為 False 則不儲存)
    """
    analyzer = BenchpressWristAnalyzer(fps=fps)
    analyzer.load_skeleton_file(txt_path)
    analyzer.preprocess_left_wrist()
    reps = analyzer.extract_keyframes()

    target_json = output_json if save_json else None
    target_csv = output_csv if save_csv else None
    if target_json or target_csv:
        analyzer.export_results(json_path=target_json, csv_path=target_csv)

    if save_plot and plot_path:
        analyzer.plot_segmentation(save_path=plot_path)

    return reps


# ==============================================================================
# ⚙️ 輸出開關手動控制區 (若不想產生 CSV 表格或 PNG 圖表，直接在此改為 False 即可)
# ==============================================================================
ENABLE_SAVE_JSON = True   # 是否匯出 JSON 標註檔 (*_segments.json)
ENABLE_SAVE_CSV  = False  # 是否匯出 CSV 表格檔 (*_segments.csv) -> 改為 False 即可關閉
ENABLE_SAVE_PLOT = True   # 是否繪製 PNG 圖表檔 (*_segmentation.png) -> 改為 False 即可關閉
# ==============================================================================

DEFAULT_SUBJECT_DIR = r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2"


def process_subject_directory(
    subject_dir: str,
    target_cam_keyword: str = "i17",
    output_dir_name: str = "dataprocess",
    fps: float = 30.0,
    save_json: bool = True,
    save_csv: bool = False,
    save_plot: bool = True,
    exclude_keyword: str = "checkboard"
) -> list:
    """
    全自動批次處理母資料夾 (Subject Directory) 模式：
      1. 自動於母資料夾下尋找包含 target_cam_keyword (預設 'i17') 的相機目錄或 dataprocess
      2. 收集所有 yolo_skeleton_*.txt (排除 checkboard)
      3. 自動在母資料夾下建立 dataprocess/ 目錄
      4. 將切片結果 (*_segments.json, *_segmentation.png, *_segments.csv) 統一儲存於母資料夾的 dataprocess/ 中
    """
    from pathlib import Path
    subj_path = Path(subject_dir)
    if not subj_path.exists():
        print(f"❌ 錯誤：找不到母目錄 {subject_dir}")
        return []

    # 1. 尋找含有骨架 txt 的來源目錄 (優先找 *i17* 目錄，次之找母目錄下的 dataprocess)
    source_dir = None
    for d in subj_path.iterdir():
        if d.is_dir() and target_cam_keyword.lower() in d.name.lower() and not d.name.lower().startswith("dataprocess"):
            source_dir = d
            break

    # 若未找到 i17 子目錄，檢查是否 txt 已經在 dataprocess 中
    dp_cand = subj_path / output_dir_name
    dp_i17_cand = subj_path / f"{output_dir_name}-i17"

    txt_files = []
    # 策略 A: 從 i17 相機目錄尋找
    if source_dir:
        txt_files = sorted(list(source_dir.glob("yolo_skeleton_*.txt")) + list(source_dir.glob("skeleton_*.txt")))

    # 策略 B: 若 i17 目錄沒有 txt，從 dataprocess 尋找
    if not txt_files and dp_cand.exists():
        source_dir = dp_cand
        txt_files = sorted(list(dp_cand.glob("yolo_skeleton_*.txt")) + list(dp_cand.glob("skeleton_*.txt")))

    if not txt_files and dp_i17_cand.exists():
        source_dir = dp_i17_cand
        txt_files = sorted(list(dp_i17_cand.glob("yolo_skeleton_*.txt")) + list(dp_i17_cand.glob("skeleton_*.txt")))

    # 過濾 checkboard
    if exclude_keyword:
        txt_files = [f for f in txt_files if exclude_keyword.lower() not in f.name.lower()]

    if not txt_files:
        print(f"❌ 錯誤：在 {subj_path} 中未找到含有 '{target_cam_keyword}' 的骨架 txt 檔案！")
        return []

    # 2. 確定母目錄層級的輸出目錄
    out_dir = subj_path / output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 臥推左手手腕關鍵幀擷取演算法 - 母目錄批次處理模式")
    print(f" 📂 母目錄 (Subject)  : {subj_path}")
    print(f" 🔍 骨架來源視角目錄  : {source_dir}")
    print(f" 💾 輸出目標目錄      : {out_dir}")
    print(f" 🎥 待處理骨架檔案    : {[f.name for f in txt_files]}")
    print(f" ⚙️ 輸出開關設定      : JSON={'ON' if save_json else 'OFF'}, CSV={'ON' if save_csv else 'OFF'}, PNG={'ON' if save_plot else 'OFF'}")
    print("=" * 70)

    results_summary = []

    for idx, txt_p in enumerate(txt_files, 1):
        stem = txt_p.stem
        out_json = (out_dir / f"{stem}_segments.json") if save_json else None
        out_csv = (out_dir / f"{stem}_segments.csv") if save_csv else None
        out_plot = (out_dir / f"{stem}_segmentation.png") if save_plot else None

        print(f"\n▶ [{idx}/{len(txt_files)}] 正在分析骨架: 【{txt_p.name}】")
        try:
            analyzer = BenchpressWristAnalyzer(fps=fps)
            analyzer.load_skeleton_file(str(txt_p))
            analyzer.preprocess_left_wrist()
            reps = analyzer.extract_keyframes()
            print(f"   ✅ 成功辨識 {len(reps)} 組反覆動作")

            if out_json or out_csv:
                analyzer.export_results(
                    json_path=str(out_json) if out_json else None,
                    csv_path=str(out_csv) if out_csv else None
                )
                if out_json:
                    print(f"   📄 JSON 標註檔: {out_json.name}")

            if out_plot:
                analyzer.plot_segmentation(
                    save_path=str(out_plot),
                    title=f"Bench Press Analysis: {stem}"
                )
                print(f"   📈 PNG 圖表檔 : {out_plot.name}")

            results_summary.append({"file": txt_p.name, "reps": len(reps), "status": "OK"})
        except Exception as e:
            print(f"   ❌ 處理失敗: {e}")
            results_summary.append({"file": txt_p.name, "reps": 0, "status": f"FAIL: {e}"})

    print("\n" + "=" * 70)
    print("🎉 批次關鍵幀分析完成！成果已全數存入母資料夾層級的 dataprocess/ :")
    for r in results_summary:
        print(f"   - {r['file']:<35} : {r['status']} ({r['reps']} reps)")
    print(f" 📁 存放位置: {out_dir}")
    print("=" * 70)

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="臥推左手手腕關鍵幀擷取演算法 (頂點、底點、上下中間段)")
    parser.add_argument("--dir", "-d", type=str, default=None,
                        help=f"母資料夾路徑 (例如: '{DEFAULT_SUBJECT_DIR}')。若給定此參數，將自動尋找 i17 視角並將結果輸出至該母目錄下的 dataprocess/")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="單一 YOLO 骨架 txt 檔案路徑")
    parser.add_argument("--output-json", "-j", type=str, default=None,
                        help="輸出 JSON 檔案路徑 (預設為同目錄下的 <檔名>_segments.json)")
    parser.add_argument("--output-csv", "-c", type=str, default=None,
                        help="輸出 CSV 檔案路徑 (預設為同目錄下的 <檔名>_segments.csv)")
    parser.add_argument("--plot", "-p", type=str, default=None,
                        help="儲存波形分析圖的路徑 (預設為同目錄下的 <檔名>_segmentation.png)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="影片影格率 FPS (預設為 30.0)")

    # 手動開關參數
    parser.add_argument("--no-csv", dest="save_csv", action="store_false", default=ENABLE_SAVE_CSV,
                        help="手動關閉 CSV 表格輸出 (不產生 *_segments.csv)")
    parser.add_argument("--save-csv", dest="save_csv", action="store_true",
                        help="強制開啟 CSV 表格輸出")
    parser.add_argument("--no-plot", dest="save_plot", action="store_false", default=ENABLE_SAVE_PLOT,
                        help="手動關閉 PNG 波形圖表輸出 (不產生 *_segmentation.png)")
    parser.add_argument("--save-plot", dest="save_plot", action="store_true",
                        help="強制開啟 PNG 波形圖表輸出")
    parser.add_argument("--no-json", dest="save_json", action="store_false", default=ENABLE_SAVE_JSON,
                        help="手動關閉 JSON 標註輸出 (不產生 *_segments.json)")
    parser.add_argument("--save-json", dest="save_json", action="store_true",
                        help="強制開啟 JSON 標註輸出")

    args = parser.parse_args()

    # 模式 A: 母目錄批次處理模式 (若指定 --dir 或未帶任何參數時預設執行)
    if args.dir or (args.input is None):
        target_dir = args.dir or DEFAULT_SUBJECT_DIR
        process_subject_directory(
            subject_dir=target_dir,
            target_cam_keyword="i17",
            output_dir_name="dataprocess",
            fps=args.fps,
            save_json=args.save_json,
            save_csv=args.save_csv,
            save_plot=args.save_plot
        )
        return

    # 模式 B: 單一檔案處理模式
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"錯誤：找不到輸入檔案 {input_path}")
        sys.exit(1)

    # 若未指定輸出路徑，自動以輸入檔名為前綴產出於相同資料夾
    base_dir = os.path.dirname(input_path)
    base_stem = os.path.splitext(os.path.basename(input_path))[0]

    out_json = (args.output_json or os.path.join(base_dir, f"{base_stem}_segments.json")) if args.save_json else None
    out_csv = (args.output_csv or os.path.join(base_dir, f"{base_stem}_segments.csv")) if args.save_csv else None
    out_plot = (args.plot or os.path.join(base_dir, f"{base_stem}_segmentation.png")) if args.save_plot else None

    print(f"\n==================================================")
    print(f" 🚀 開始執行臥推左手手腕關鍵幀擷取演算法 (單檔模式)")
    print(f" 輸入檔案: {input_path}")
    print(f" 鎖定關節: 左手手腕 (Left Wrist, 12 點骨架 Index 4 / 舊版 17 點 Index 9)")
    print(f" 影片幀率: {args.fps} FPS")
    print(f" 輸出設定: JSON={'ON' if args.save_json else 'OFF'}, CSV={'ON' if args.save_csv else 'OFF'}, PNG={'ON' if args.save_plot else 'OFF'}")
    print(f"==================================================")

    analyzer = BenchpressWristAnalyzer(fps=args.fps)
    analyzer.load_skeleton_file(input_path)
    analyzer.preprocess_left_wrist()
    reps = analyzer.extract_keyframes()

    print(f"\n✅ 成功偵測到 {len(reps)} 組完整臥推反覆動作！\n")
    df_summary = analyzer.get_summary_dataframe()
    print(df_summary.to_string(index=False))
    print("\n--------------------------------------------------")

    if out_json or out_csv:
        analyzer.export_results(json_path=out_json, csv_path=out_csv)
    if not args.save_csv:
        print("ℹ️ [提示] 已手動關閉 CSV 檔案輸出 (未建立 CSV)")
    if not args.save_json:
        print("ℹ️ [提示] 已手動關閉 JSON 檔案輸出 (未建立 JSON)")

    if out_plot:
        analyzer.plot_segmentation(save_path=out_plot, title=f"Bench Press Analysis: {os.path.basename(input_path)}")
    else:
        print("ℹ️ [提示] 已手動關閉 PNG 圖表繪製輸出 (未建立 PNG)")

    print(f"==================================================\n")


if __name__ == "__main__":
    main()
