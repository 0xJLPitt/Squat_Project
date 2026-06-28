"""
目的: 將 YOLO COCO 17 骨架點轉換為 Human3.6M 格式的 2D 骨架，並存為 npz 檔案。
執行範例:
python convert_keypoints.py
"""
import numpy as np
import os
import glob

def interpolate_missing_keypoints(kpts):
    """將缺失的點 (x=0, y=0) 沿著時間軸做線性插值補齊"""
    N_frames, num_joints, _ = kpts.shape
    for j in range(num_joints):
        missing = kpts[:, j, 2] == 0.0
        valid = ~missing
        if missing.all() or valid.all():
            continue
        valid_indices = np.where(valid)[0]
        missing_indices = np.where(missing)[0]
        kpts[missing_indices, j, 0] = np.interp(missing_indices, valid_indices, kpts[valid_indices, j, 0])
        kpts[missing_indices, j, 1] = np.interp(missing_indices, valid_indices, kpts[valid_indices, j, 1])
        kpts[missing_indices, j, 2] = 1.0
    return kpts

def turn_into_h36m(keypoints):
    """將 YOLO 的 COCO 17 格式轉換為 Human3.6M 17 格式"""
    # keypoints shape: (N_frames, 17, 3)
    new_keypoints = np.zeros_like(keypoints)
    
    # 0: 骨盆 (Pelvis) = 左臀與右臀的中點
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    # 1: 右臀 (Right Hip) -> YOLO 12
    new_keypoints[..., 1, :] = keypoints[..., 12, :]
    new_keypoints[..., 2, :] = keypoints[..., 14, :]
    new_keypoints[..., 3, :] = keypoints[..., 16, :]
    
    # 4: 左臀 (Left Hip) -> YOLO 11
    new_keypoints[..., 4, :] = keypoints[..., 11, :]
    new_keypoints[..., 5, :] = keypoints[..., 13, :]
    new_keypoints[..., 6, :] = keypoints[..., 15, :]
    
    # 8: 脖子下端 (Thorax/Neck Base) = 左肩與右肩的中點
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    # 7: 脊椎中段 (Spine) = 骨盆與脖子的中點
    new_keypoints[..., 7, :] = (new_keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    # 9: 鼻子/頭部 (Nose/Head)
    new_keypoints[..., 9, :] = keypoints[..., 0, :]
    
    # 頭頂通常用兩眼的中點來代替，但如果其中一眼完全沒偵測到(0,0)，直接用另一眼
    left_eye_missing = (keypoints[..., 1, 0] == 0) & (keypoints[..., 1, 1] == 0)
    right_eye_missing = (keypoints[..., 2, 0] == 0) & (keypoints[..., 2, 1] == 0)
    
    # 預設為兩眼中點
    new_keypoints[..., 10, :] = (keypoints[..., 1, :] + keypoints[..., 2, :]) * 0.5
    
    # 只有左眼
    only_left = ~left_eye_missing & right_eye_missing
    new_keypoints[only_left, 10, :] = keypoints[only_left, 1, :]
    
    # 只有右眼
    only_right = left_eye_missing & ~right_eye_missing
    new_keypoints[only_right, 10, :] = keypoints[only_right, 2, :]
    
    # 兩眼皆無
    both_missing = left_eye_missing & right_eye_missing
    new_keypoints[both_missing, 10, :] = 0
    
    # 處理完全沒偵測到的頭部與鼻子 (座標為 0,0) -> 改用脖子跟脊椎延伸
    missing_nose = (new_keypoints[..., 9, 0] == 0) & (new_keypoints[..., 9, 1] == 0)
    new_keypoints[missing_nose, 9, :2] = new_keypoints[missing_nose, 8, :2] + (new_keypoints[missing_nose, 8, :2] - new_keypoints[missing_nose, 7, :2]) * 0.8
    
    missing_head = (new_keypoints[..., 10, 0] == 0) & (new_keypoints[..., 10, 1] == 0)
    new_keypoints[missing_head, 10, :2] = new_keypoints[missing_head, 8, :2] + (new_keypoints[missing_head, 8, :2] - new_keypoints[missing_head, 7, :2]) * 1.2
    
    # 手部: 11~13 是左手(Left Arm) -> YOLO 5, 7, 9
    new_keypoints[..., 11, :] = keypoints[..., 5, :]
    new_keypoints[..., 12, :] = keypoints[..., 7, :]
    new_keypoints[..., 13, :] = keypoints[..., 9, :]
    
    # 手部: 14~16 是右手(Right Arm) -> YOLO 6, 8, 10
    new_keypoints[..., 14, :] = keypoints[..., 6, :]
    new_keypoints[..., 15, :] = keypoints[..., 8, :]
    new_keypoints[..., 16, :] = keypoints[..., 10, :]

    return new_keypoints

def convert_txt_to_npz(txt_path, output_npz_path):
    # 1. 讀取資料
    data = np.loadtxt(txt_path, delimiter=',')
    
    if len(data) == 0:
        print(f"檔案 {txt_path} 為空，跳過。")
        return

    # 計算有幾個 frame (假設 frame id 是從 1 開始連續的)
    frames = np.unique(data[:, 0]).astype(int)
    # 取最大 frame index 來決定陣列大小，以防 frame 跳號
    max_frame = np.max(frames)
    num_frames = max_frame
    
    # 建立空的 COCO 陣列: shape (num_frames, 17, 3) 
    # 第三維度是 [x, y, conf]，如果沒有 conf，我們先預設為 1.0 (代表 100% 信心)
    coco_kpts = np.zeros((num_frames, 17, 3))
    
    for row in data:
        f_idx = int(row[0]) - 1   # 將 frame 1 轉為 index 0
        j_idx = int(row[1])       # 關節 id (0~16)
        x = row[2]
        y = row[3]
        
        coco_kpts[f_idx, j_idx, 0] = x
        coco_kpts[f_idx, j_idx, 1] = y
        
        # 由於你的資料沒有信心分數，如果是 0,0 我們視為沒偵測到(信心=0)，否則信心=1
        if x == 0 and y == 0:
            coco_kpts[f_idx, j_idx, 2] = 0.0
        else:
            coco_kpts[f_idx, j_idx, 2] = 1.0
            
    # 先將暫時缺失的點做插值補齊
    coco_kpts = interpolate_missing_keypoints(coco_kpts)
    
    # 2. 將 COCO 格式轉換為 MotionAGFormer 必須的 Human3.6M 格式
    h36m_kpts = turn_into_h36m(coco_kpts)
    
    # 3. 增加 batch 維度，變成 (1, num_frames, 17, 3)
    final_input = np.expand_dims(h36m_kpts, axis=0)
    
    # 4. 存檔
    np.savez_compressed(output_npz_path, reconstruction=final_input)
    print(f"轉換完成！檔案已儲存至 {output_npz_path}")
    print(f"資料形狀：{final_input.shape}")

if __name__ == '__main__':
    demo_dir = r'D:\Pitt\Project\squat\demo'
    print(f"開始處理資料夾: {demo_dir}")
    
    # 掃描所有的 yolo_skeleton.txt
    txt_files = glob.glob(os.path.join(demo_dir, '**', 'yolo_skeleton.txt'), recursive=True)
    
    if not txt_files:
        print("找不到任何 yolo_skeleton.txt 檔案。")
    
    for txt_file in txt_files:
        output_npz = os.path.join(os.path.dirname(txt_file), 'keypoints.npz')
        print(f"\n發現檔案: {txt_file}")
        convert_txt_to_npz(txt_file, output_npz)
