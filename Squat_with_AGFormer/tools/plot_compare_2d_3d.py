r"""
目的: 比較深蹲時 2D 與 3D 骨架推算出來的關節角度差異圖。
執行範例:
python plot_compare_2d_3d.py --npz_2d D:\Pitt\Project\squat\demo\recording_20260415_105356\keypoints.npz --npz_3d D:\Pitt\Project\squat\demo\recording_20260415_105356\keypoints_3d.npz
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def calculate_angle(a, b, c):
    """
    計算 a, b, c 三點的夾角 (適用於 2D 或 3D 向量)
    回傳角度 (degrees)
    """
    v1 = a - b
    v2 = c - b
    
    # 正規化
    norm_v1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    norm_v2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    
    # 避免除以 0
    norm_v1[norm_v1 == 0] = 1
    norm_v2[norm_v2 == 0] = 1
    
    v1_u = v1 / norm_v1
    v2_u = v2 / norm_v2
    
    # 內積
    dot_product = np.sum(v1_u * v2_u, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_2d', type=str, required=True, help='Path to 2D keypoints.npz')
    parser.add_argument('--npz_3d', type=str, required=True, help='Path to 3D keypoints_3d.npz')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    print("正在讀取資料...")
    # 讀取 2D
    data_2d = np.load(args.npz_2d, allow_pickle=True)
    # keypoints.npz 的形狀通常為 (1, N_frames, 17, 3)，第三維是 [x, y, conf]
    kpts_2d = data_2d['reconstruction'][0, :, :, :2] 
    
    # 讀取 3D
    data_3d = np.load(args.npz_3d, allow_pickle=True)
    # keypoints_3d.npz 的形狀為 (N_frames, 17, 3)
    kpts_3d = data_3d['reconstruction']
    
    # 確保兩者幀數一致
    n_frames = min(len(kpts_2d), len(kpts_3d))
    kpts_2d = kpts_2d[:n_frames]
    kpts_3d = kpts_3d[:n_frames]
    
    # 依據 Human3.6M 格式的關節索引：
    # 右腳: 髖(Hip)=1, 膝(Knee)=2, 踝(Ankle)=3, 右肩(Shoulder)=14
    # 左腳: 髖(Hip)=4, 膝(Knee)=5, 踝(Ankle)=6, 左肩(Shoulder)=11
    
    # 計算 2D 角度 (只用 X, Y)
    r_knee_2d = calculate_angle(kpts_2d[:, 1], kpts_2d[:, 2], kpts_2d[:, 3])
    l_knee_2d = calculate_angle(kpts_2d[:, 4], kpts_2d[:, 5], kpts_2d[:, 6])
    r_hip_2d = calculate_angle(kpts_2d[:, 14], kpts_2d[:, 1], kpts_2d[:, 2])
    l_hip_2d = calculate_angle(kpts_2d[:, 11], kpts_2d[:, 4], kpts_2d[:, 5])
    
    # 計算 3D 角度 (用 X, Y, Z)
    r_knee_3d = calculate_angle(kpts_3d[:, 1], kpts_3d[:, 2], kpts_3d[:, 3])
    l_knee_3d = calculate_angle(kpts_3d[:, 4], kpts_3d[:, 5], kpts_3d[:, 6])
    r_hip_3d = calculate_angle(kpts_3d[:, 14], kpts_3d[:, 1], kpts_3d[:, 2])
    l_hip_3d = calculate_angle(kpts_3d[:, 11], kpts_3d[:, 4], kpts_3d[:, 5])
    
    frames = np.arange(n_frames)
    
    # 繪製圖表 - 左右腳分開畫，方便比較 2D vs 3D
    plt.figure(figsize=(15, 10))
    
    # 右膝 2D vs 3D
    plt.subplot(2, 2, 1)
    plt.plot(frames, r_knee_2d, label='2D Right Knee', color='red', alpha=0.4, linestyle='--')
    plt.plot(frames, r_knee_3d, label='3D Right Knee', color='red', linewidth=2)
    plt.title('Right Knee Angle: 2D vs 3D', fontsize=12)
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 左膝 2D vs 3D
    plt.subplot(2, 2, 2)
    plt.plot(frames, l_knee_2d, label='2D Left Knee', color='blue', alpha=0.4, linestyle='--')
    plt.plot(frames, l_knee_3d, label='3D Left Knee', color='blue', linewidth=2)
    plt.title('Left Knee Angle: 2D vs 3D', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 右髖 2D vs 3D
    plt.subplot(2, 2, 3)
    plt.plot(frames, r_hip_2d, label='2D Right Hip', color='red', alpha=0.4, linestyle='--')
    plt.plot(frames, r_hip_3d, label='3D Right Hip', color='red', linewidth=2)
    plt.title('Right Hip Angle: 2D vs 3D', fontsize=12)
    plt.xlabel('Frames')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 左髖 2D vs 3D
    plt.subplot(2, 2, 4)
    plt.plot(frames, l_hip_2d, label='2D Left Hip', color='blue', alpha=0.4, linestyle='--')
    plt.plot(frames, l_hip_3d, label='3D Left Hip', color='blue', linewidth=2)
    plt.title('Left Hip Angle: 2D vs 3D', fontsize=12)
    plt.xlabel('Frames')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    out_path = args.output if args.output else os.path.join(os.path.dirname(args.npz_3d), 'compare_2d_3d_angles.png')
    plt.savefig(out_path, dpi=150)
    print(f"✅ 2D vs 3D 比較圖表已儲存至: {out_path}")

if __name__ == "__main__":
    main()
