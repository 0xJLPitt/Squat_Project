"""
目的: 修正 3D 骨架點的長度，使其符合人體運動學特性。
執行範例:
python fix_bone_length.py --npz D:\Pitt\Project\squat\demo\recording_20260415_105356\keypoints_3d.npz
"""
import numpy as np
import argparse
import os

def fix_bone_lengths(kpts_3d):
    """
    透過 Kinematic Tree (由內而外) 強制修正所有的骨骼長度為影片中的中位數長度，
    避免任何關節在某些 frame 發生嚴重的外推或拉長現象。
    """
    # 定義樹狀結構 (Parent, Child)
    tree = [
        (0, 7), (7, 8), (8, 9), (9, 10),  # 軀幹與頭部
        (0, 1), (1, 2), (2, 3),           # 右腿 (Pelvis->Hip->Knee->Ankle)
        (0, 4), (4, 5), (5, 6),           # 左腿 (Pelvis->Hip->Knee->Ankle)
        (8, 11), (11, 12), (12, 13),      # 左手 (Thorax->Shoulder->Elbow->Wrist)
        (8, 14), (14, 15), (15, 16)       # 右手 (Thorax->Shoulder->Elbow->Wrist)
    ]
    
    # 1. 找出最真實的骨骼長度 (使用中位數排除極端值)
    bone_lengths = {}
    for (parent, child) in tree:
        vectors = kpts_3d[:, child, :] - kpts_3d[:, parent, :]
        lengths = np.linalg.norm(vectors, axis=-1)
        true_length = np.median(lengths)
        bone_lengths[(parent, child)] = true_length
        print(f"骨骼段 ({parent}->{child}) 修正長度鎖定為: {true_length:.4f}")
        
    # 2. 依照 Kinematic Tree 由內往外修正
    new_kpts = np.copy(kpts_3d)
    
    for (parent, child) in tree:
        for i in range(len(new_kpts)):
            # 計算該影格中，從 (可能已更新的) parent 指向 child 的向量
            vec = new_kpts[i, child, :] - new_kpts[i, parent, :]
            curr_len = np.linalg.norm(vec)
            
            if curr_len > 1e-5:
                # 保持方向不變，將長度強制縮放回真實長度
                true_len = bone_lengths[(parent, child)]
                vec_normalized = vec / curr_len
                new_kpts[i, child, :] = new_kpts[i, parent, :] + vec_normalized * true_len

    return new_kpts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="強制修復 3D 骨架的長度拉伸問題")
    parser.add_argument('--npz', type=str, required=True, help="輸入的 keypoints_3d.npz 檔案路徑")
    parser.add_argument('--out', type=str, default=None, help="輸出的檔案路徑 (預設為覆蓋原本的 npz 或另存新檔)")
    args = parser.parse_args()
    
    if not os.path.exists(args.npz):
        print(f"找不到檔案: {args.npz}")
        exit(1)
        
    print(f"讀取 3D 資料: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)
    kpts = data['reconstruction']
    
    # 修復長度
    fixed_kpts = fix_bone_lengths(kpts)
    
    # 輸出
    out_path = args.out if args.out else args.npz.replace(".npz", "_fixed.npz")
    np.savez_compressed(out_path, reconstruction=fixed_kpts)
    print(f"修正完成！骨架已存至: {out_path}")
