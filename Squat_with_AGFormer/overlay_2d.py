"""
目的: 將 2D Human3.6M 骨架點畫在原始影片上。
執行範例:
python overlay_2d.py --video D:\Pitt\Project\squat\demo\recording_20260415_105356\vision2.avi --npz D:\Pitt\Project\squat\demo\recording_20260415_105356\keypoints.npz --out overlay_output.mp4
"""
import cv2
import numpy as np
import argparse
import os

def overlay_2d(video_path, npz_path, output_path):
    print(f"讀取影片: {video_path}")
    if not os.path.exists(video_path):
        print("找不到影片檔案！")
        return
        
    cap = cv2.VideoCapture(video_path)
    
    print(f"讀取 2D 骨架資料: {npz_path}")
    if not os.path.exists(npz_path):
        print("找不到 npz 檔案！")
        return
        
    data = np.load(npz_path, allow_pickle=True)
    keypoints = data['reconstruction'][0] # shape (N_frames, 17, 3)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30.0, (w, h))
    
    # Human3.6M 骨架連接順序
    I = [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9]
    J = [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10]
    # True 代表左半邊(藍色)，False 代表右半邊(紅色)
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx < len(keypoints):
            kpts = keypoints[frame_idx] # (17, 3)
            
            # 畫骨架連線
            for i in range(len(I)):
                pt1 = kpts[I[i]]
                pt2 = kpts[J[i]]
                
                # 如果座標不為 0 才畫
                if pt1[2] > 0 and pt2[2] > 0 and not (pt1[0]==0 and pt1[1]==0) and not (pt2[0]==0 and pt2[1]==0):
                    # OpenCV 是 BGR 格式
                    color = (255, 0, 0) if LR[i] else (0, 0, 255) # 左邊藍色，右邊紅色
                    p1 = (int(pt1[0]), int(pt1[1]))
                    p2 = (int(pt2[0]), int(pt2[1]))
                    cv2.line(frame, p1, p2, color, 3)
            
            # 畫關節點
            for j in range(17):
                pt = kpts[j]
                if pt[2] > 0 and not (pt[0]==0 and pt[1]==0):
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1) # 綠色點
                    
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"影片處理完成！已儲存至: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="將 2D 骨架點畫在影片上")
    parser.add_argument('--video', type=str, required=True, help="原始影片路徑 (.avi / .mp4)")
    parser.add_argument('--npz', type=str, required=True, help="2D 骨架檔案路徑 (keypoints.npz)")
    parser.add_argument('--out', type=str, default='overlay_output.mp4', help="輸出影片名稱")
    args = parser.parse_args()
    
    overlay_2d(args.video, args.npz, args.out)
