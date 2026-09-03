r"""
目的: 計算並繪製 3D 骨架深蹲時的膝蓋和髖關節角度變化圖。
執行範例:
python plot_angles.py --npz D:\Pitt\Project\squat\demo\recording_20260415_105356\keypoints_3d.npz --show
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def calculate_angle(a, b, c):
    """
    計算 a, b, c 三點在 3D 空間中以 b 為頂點的夾角
    回傳角度 (degrees)
    """
    # 建立向量 ba 與 bc
    v1 = a - b
    v2 = c - b
    
    # 將向量正規化 (Normalize)
    v1_u = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2_u = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)
    
    # 內積
    dot_product = np.sum(v1_u * v2_u, axis=-1)
    
    # 避免浮點數誤差導致 arccos 出錯，將範圍限制在 [-1.0, 1.0]
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 轉換為弧度再轉為角度
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def main():
    parser = argparse.ArgumentParser(description="Calculate and plot joint angles from 3D keypoints")
    parser.add_argument('--npz', type=str, required=True, help='Path to keypoints_3d.npz')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    parser.add_argument('--show', action='store_true', help='Show the plot in a window')
    args = parser.parse_args()

    if not os.path.exists(args.npz):
        print(f"錯誤: 找不到檔案 {args.npz}")
        return

    print(f"正在讀取資料: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)
    # 原始維度通常為 (N_frames, 17, 3)
    kpts = data['reconstruction']
    
    n_frames = len(kpts)
    
    # 依據 Human3.6M 格式的關節索引：
    # 右腳: 髖(Hip)=1, 膝(Knee)=2, 踝(Ankle)=3, 右肩(Shoulder)=14
    # 左腳: 髖(Hip)=4, 膝(Knee)=5, 踝(Ankle)=6, 左肩(Shoulder)=11
    
    # 計算右膝角: 右髖(1) - 右膝(2) - 右踝(3)
    r_knee_angles = calculate_angle(kpts[:, 1], kpts[:, 2], kpts[:, 3])
    # 計算左膝角: 左髖(4) - 左膝(5) - 左踝(6)
    l_knee_angles = calculate_angle(kpts[:, 4], kpts[:, 5], kpts[:, 6])
    
    # 計算右髖角: 這裡我們取 右肩(14) - 右髖(1) - 右膝(2) 作為軀幹與大腿的夾角
    r_hip_angles = calculate_angle(kpts[:, 14], kpts[:, 1], kpts[:, 2])
    # 計算左髖角: 取 左肩(11) - 左髖(4) - 左膝(5)
    l_hip_angles = calculate_angle(kpts[:, 11], kpts[:, 4], kpts[:, 5])
    
    frames = np.arange(n_frames)
    
    # 繪製圖表
    plt.figure(figsize=(12, 8))
    
    # === 繪製膝角 ===
    plt.subplot(2, 1, 1)
    plt.plot(frames, r_knee_angles, label='Right Knee Angle', color='red')
    plt.plot(frames, l_knee_angles, label='Left Knee Angle', color='blue')
    plt.title('Knee Angles During Squat', fontsize=14)
    plt.ylabel('Angle (degrees)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # === 繪製髖角 ===
    plt.subplot(2, 1, 2)
    plt.plot(frames, r_hip_angles, label='Right Hip Angle', color='red')
    plt.plot(frames, l_hip_angles, label='Left Hip Angle', color='blue')
    plt.title('Hip Angles During Squat', fontsize=14)
    plt.xlabel('Frames', fontsize=12)
    plt.ylabel('Angle (degrees)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 儲存與顯示
    out_path = args.output if args.output else args.npz.replace('.npz', '_angles_plot.png')
    plt.savefig(out_path, dpi=150)
    print(f"✅ 圖表已儲存至: {out_path}")
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
