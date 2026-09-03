r"""
目的: 比較原始 YOLO 2D、AGFormer 輸入 2D 與 AGFormer 輸出 3D 的深蹲角度變化差異圖。
執行範例:
python plot_compare_all.py --yolo_txt D:\Pitt\Project\squat\demo\recording_20260415_105356\yolo_skeleton.txt --h36m_2d D:\Pitt\Project\squat\demo\recording_20260415_105356\keypoints.npz --h36m_3d D:\Pitt\Project\squat\demo\recording_20260415_105356\keypoints_3d.npz
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def calculate_angle(a, b, c):
    """計算 a, b, c 三點的夾角 (適用於 2D 或 3D 向量)"""
    v1 = a - b
    v2 = c - b
    
    norm_v1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    norm_v2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    
    # 避免除以 0 (遇到 NaN 時會保持 NaN)
    with np.errstate(divide='ignore', invalid='ignore'):
        v1_u = v1 / norm_v1
        v2_u = v2 / norm_v2
        dot_product = np.sum(v1_u * v2_u, axis=-1)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)
        
    return np.degrees(angle)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_txt', type=str, required=True, help='Path to yolo_skeleton.txt')
    parser.add_argument('--h36m_2d', type=str, required=True, help='Path to keypoints.npz')
    parser.add_argument('--h36m_3d', type=str, required=True, help='Path to keypoints_3d.npz')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    print("正在讀取資料...")
    
    # --- 1. 讀取最原始的 yolo_skeleton.txt ---
    yolo_data = np.loadtxt(args.yolo_txt, delimiter=',')
    frames_ids = np.unique(yolo_data[:, 0]).astype(int)
    num_frames_yolo = np.max(frames_ids)
    
    # 建立 COCO 陣列，預設填入 NaN (這樣畫圖時遺失的點會變成斷線，而不是掉到 0)
    k_yolo = np.full((num_frames_yolo, 17, 2), np.nan)
    for row in yolo_data:
        f_idx = int(row[0]) - 1
        j_idx = int(row[1])
        x = row[2]
        y = row[3]
        if x != 0 or y != 0: # 如果不是 (0,0) 才填入
            k_yolo[f_idx, j_idx, 0] = x
            k_yolo[f_idx, j_idx, 1] = y

    # --- 2. 讀取 AGFormer 輸入用的 H36M 2D (已內插補齊) ---
    d_2d = np.load(args.h36m_2d, allow_pickle=True)
    k_2d = d_2d['reconstruction']
    if len(k_2d.shape) == 4 and k_2d.shape[0] == 1:
        k_2d = k_2d[0]
    k_2d = k_2d[..., :2]

    # --- 3. 讀取 AGFormer 輸出的 H36M 3D ---
    d_3d = np.load(args.h36m_3d, allow_pickle=True)
    k_3d = d_3d['reconstruction']
    if len(k_3d.shape) == 4 and k_3d.shape[0] == 1:
        k_3d = k_3d[0]
    k_3d = k_3d[..., :3]

    # 統一幀數長度
    n_frames = min(len(k_yolo), len(k_2d), len(k_3d))
    k_yolo = k_yolo[:n_frames]
    k_2d = k_2d[:n_frames]
    k_3d = k_3d[:n_frames]

    # --- 計算 YOLO 的角度 (依據 COCO 17 關節格式) ---
    # 左肩=5, 右肩=6, 左髖=11, 右髖=12, 左膝=13, 右膝=14, 左踝=15, 右踝=16
    r_knee_yolo = calculate_angle(k_yolo[:, 12], k_yolo[:, 14], k_yolo[:, 16])
    l_knee_yolo = calculate_angle(k_yolo[:, 11], k_yolo[:, 13], k_yolo[:, 15])
    r_hip_yolo = calculate_angle(k_yolo[:, 6], k_yolo[:, 12], k_yolo[:, 14])
    l_hip_yolo = calculate_angle(k_yolo[:, 5], k_yolo[:, 11], k_yolo[:, 13])

    # --- 計算 H36M 2D (AGFormer 預先處理好的輸入) ---
    # 右腳: 髖(Hip)=1, 膝(Knee)=2, 踝(Ankle)=3, 右肩(Shoulder)=14
    # 左腳: 髖(Hip)=4, 膝(Knee)=5, 踝(Ankle)=6, 左肩(Shoulder)=11
    r_knee_2d = calculate_angle(k_2d[:, 1], k_2d[:, 2], k_2d[:, 3])
    l_knee_2d = calculate_angle(k_2d[:, 4], k_2d[:, 5], k_2d[:, 6])
    r_hip_2d = calculate_angle(k_2d[:, 14], k_2d[:, 1], k_2d[:, 2])
    l_hip_2d = calculate_angle(k_2d[:, 11], k_2d[:, 4], k_2d[:, 5])

    # --- 計算 H36M 3D (AGFormer 最終輸出) ---
    r_knee_3d = calculate_angle(k_3d[:, 1], k_3d[:, 2], k_3d[:, 3])
    l_knee_3d = calculate_angle(k_3d[:, 4], k_3d[:, 5], k_3d[:, 6])
    r_hip_3d = calculate_angle(k_3d[:, 14], k_3d[:, 1], k_3d[:, 2])
    l_hip_3d = calculate_angle(k_3d[:, 11], k_3d[:, 4], k_3d[:, 5])

    frames = np.arange(n_frames)

    # === 繪圖 ===
    plt.figure(figsize=(16, 12))

    def plot_subplot(idx, yolo, h2d, h3d, title, ylabel=''):
        plt.subplot(2, 2, idx)
        # 1. 原始 YOLO 點 (灰色帶圓點)，如果有斷裂代表該幀沒偵測到
        plt.plot(frames, yolo, label='1. True YOLO Raw (COCO 2D)', color='gray', alpha=0.6, linestyle=':', marker='.', markersize=2)
        # 2. 轉換並內插補齊後的 2D (藍色虛線)
        plt.plot(frames, h2d, label='2. AGFormer Input (H36M 2D)', color='blue', alpha=0.5, linestyle='--')
        # 3. 最終 3D 輸出 (紅色實線)
        plt.plot(frames, h3d, label='3. AGFormer Output (H36M 3D)', color='red', linewidth=2)
        
        plt.title(title, fontsize=14)
        if ylabel:
            plt.ylabel(ylabel, fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)

    plot_subplot(1, r_knee_yolo, r_knee_2d, r_knee_3d, 'Right Knee Angle', 'Angle (degrees)')
    plot_subplot(2, l_knee_yolo, l_knee_2d, l_knee_3d, 'Left Knee Angle')
    
    plt.subplot(2, 2, 3)
    plt.xlabel('Frames', fontsize=12)
    plot_subplot(3, r_hip_yolo, r_hip_2d, r_hip_3d, 'Right Hip Angle', 'Angle (degrees)')
    
    plt.subplot(2, 2, 4)
    plt.xlabel('Frames', fontsize=12)
    plot_subplot(4, l_hip_yolo, l_hip_2d, l_hip_3d, 'Left Hip Angle')

    plt.tight_layout()
    out_path = args.output if args.output else os.path.join(os.path.dirname(args.h36m_3d), 'compare_true_yolo_vs_2d_vs_3d.png')
    plt.savefig(out_path, dpi=150)
    print(f"✅ 正確的三者比較圖表已儲存至: {out_path}")

if __name__ == "__main__":
    main()
