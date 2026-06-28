"""
目的: 計算並繪製原始 YOLO 2D 骨架深蹲時的膝蓋和髖關節角度變化圖。
執行範例:
python plot_yolo_angles.py --txt D:\Pitt\Project\squat\demo\recording_20260415_105356\yolo_skeleton.txt --show
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def interpolate_missing_keypoints(kpts):
    """將缺失的點 (x=0, y=0) 沿著時間軸做線性插值補齊"""
    N_frames, num_joints, _ = kpts.shape
    for j in range(num_joints):
        # 假設 x=0 且 y=0 代表該點遺失
        missing = (kpts[:, j, 0] == 0.0) & (kpts[:, j, 1] == 0.0)
        valid = ~missing
        if missing.all() or valid.all():
            continue
        valid_indices = np.where(valid)[0]
        missing_indices = np.where(missing)[0]
        kpts[missing_indices, j, 0] = np.interp(missing_indices, valid_indices, kpts[valid_indices, j, 0])
        kpts[missing_indices, j, 1] = np.interp(missing_indices, valid_indices, kpts[valid_indices, j, 1])
    return kpts

def calculate_angle_2d(a, b, c):
    """
    計算 a, b, c 三點在 2D 空間中以 b 為頂點的夾角
    回傳角度 (degrees)
    """
    # 建立向量 ba 與 bc
    v1 = a - b
    v2 = c - b
    
    # 將向量正規化 (Normalize)
    # 若長度為 0 則避免除以 0
    norm_v1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    norm_v2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    
    # 處理 norm 為 0 的情況
    norm_v1[norm_v1 == 0] = 1
    norm_v2[norm_v2 == 0] = 1
    
    v1_u = v1 / norm_v1
    v2_u = v2 / norm_v2
    
    # 內積
    dot_product = np.sum(v1_u * v2_u, axis=-1)
    
    # 避免浮點數誤差導致 arccos 出錯，將範圍限制在 [-1.0, 1.0]
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 轉換為弧度再轉為角度
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', type=str, required=True, help='Path to yolo_skeleton.txt')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    args = parser.parse_args()

    if not os.path.exists(args.txt):
        print(f"錯誤: 找不到檔案 {args.txt}")
        return

    print(f"正在讀取資料: {args.txt}")
    data = np.loadtxt(args.txt, delimiter=',')
    
    if len(data) == 0:
        print("檔案為空！")
        return

    # 計算總幀數
    frames_ids = np.unique(data[:, 0]).astype(int)
    num_frames = np.max(frames_ids)
    
    # 建立空的陣列 shape (num_frames, 17, 2)
    coco_kpts = np.zeros((num_frames, 17, 2))
    
    for row in data:
        f_idx = int(row[0]) - 1   # frame id 轉 index
        j_idx = int(row[1])       # 關節 id (0~16)
        x = row[2]
        y = row[3]
        
        coco_kpts[f_idx, j_idx, 0] = x
        coco_kpts[f_idx, j_idx, 1] = y
        
    # 內插補齊遺失的座標 (避免因為偵測不到瞬間變成 0 導致角度亂跳)
    coco_kpts = interpolate_missing_keypoints(coco_kpts)
    
    # COCO 17 關節定義：
    # 左肩=5, 右肩=6
    # 左髖=11, 右髖=12
    # 左膝=13, 右膝=14
    # 左踝=15, 右踝=16
    
    # 計算右膝角: 右髖(12) - 右膝(14) - 右踝(16)
    r_knee_angles = calculate_angle_2d(coco_kpts[:, 12], coco_kpts[:, 14], coco_kpts[:, 16])
    # 計算左膝角: 左髖(11) - 左膝(13) - 左踝(15)
    l_knee_angles = calculate_angle_2d(coco_kpts[:, 11], coco_kpts[:, 13], coco_kpts[:, 15])
    
    # 計算右髖角: 取 右肩(6) - 右髖(12) - 右膝(14)
    r_hip_angles = calculate_angle_2d(coco_kpts[:, 6], coco_kpts[:, 12], coco_kpts[:, 14])
    # 計算左髖角: 取 左肩(5) - 左髖(11) - 左膝(13)
    l_hip_angles = calculate_angle_2d(coco_kpts[:, 5], coco_kpts[:, 11], coco_kpts[:, 13])
    
    frames = np.arange(num_frames)
    
    # 繪製圖表
    plt.figure(figsize=(12, 8))
    
    # === 繪製膝角 ===
    plt.subplot(2, 1, 1)
    plt.plot(frames, r_knee_angles, label='Right Knee Angle (YOLO 2D)', color='red')
    plt.plot(frames, l_knee_angles, label='Left Knee Angle (YOLO 2D)', color='blue')
    plt.title('Knee Angles During Squat (Original 2D YOLO)', fontsize=14)
    plt.ylabel('Angle (degrees)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # === 繪製髖角 ===
    plt.subplot(2, 1, 2)
    plt.plot(frames, r_hip_angles, label='Right Hip Angle (YOLO 2D)', color='red')
    plt.plot(frames, l_hip_angles, label='Left Hip Angle (YOLO 2D)', color='blue')
    plt.title('Hip Angles During Squat (Original 2D YOLO)', fontsize=14)
    plt.xlabel('Frames', fontsize=12)
    plt.ylabel('Angle (degrees)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 儲存
    out_path = args.output if args.output else args.txt.replace('.txt', '_2d_angles_plot.png')
    plt.savefig(out_path, dpi=150)
    print(f"✅ 原始 2D 圖表已儲存至: {out_path}")

if __name__ == "__main__":
    main()
