"""
檔案目的: 將 YOLO 辨識出的 2D 骨架座標，重新畫回原影片上以供人工肉眼檢查。
注意: 可選步驟，如果 3D 投影有問題，可以先用此步驟確認是否是 YOLO 本身在 2D 就已經抓錯。
呼叫指令: python step2_draw_2d_on_video.py
"""
import numpy as np
import cv2
import os

def main():
    # Configure paths
    DATA_DIR = r"E:\squat\squat_dataset\S001_Pitt\session02\recording_20260207_163410"
    videos = ["FL", "FR", "RLU", "RR"]
    
    # Define COCO connections
    connections = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
    ]
    
    for vid_name in videos:
        txt_path = os.path.join(DATA_DIR, f"yolo_skeleton_{vid_name}.txt")
        vid_path = os.path.join(DATA_DIR, f"{vid_name}.avi")
        out_path = os.path.join(DATA_DIR, f"{vid_name}_2d_overlay.mp4")
        
        if not os.path.exists(vid_path) or not os.path.exists(txt_path):
            print(f"Skipping {vid_name}: Video or txt not found.")
            continue
            
        print(f"Loading 2D keypoints for {vid_name}...")
        data = np.loadtxt(txt_path, delimiter=',')
        
        # Organize data by frame: {frame_idx (0-based): {joint_idx: (x, y)}}
        frame_data = {}
        if len(data) > 0:
            for row in data:
                f_idx = int(row[0]) - 1
                j_idx = int(row[1])
                x, y = int(row[2]), int(row[3])
                if f_idx not in frame_data:
                    frame_data[f_idx] = {}
                frame_data[f_idx][j_idx] = (x, y)
                
        print(f"Drawing 2D skeleton onto {vid_name}.avi...")
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        
        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            pts = frame_data.get(f_idx, {})
            
            # Draw keypoints
            for j, (x, y) in pts.items():
                if x != 0 and y != 0:
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    
            # Draw connections
            for j1, j2 in connections:
                if j1 in pts and j2 in pts:
                    x1, y1 = pts[j1]
                    x2, y2 = pts[j2]
                    if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        
            out_video.write(frame)
            
            if f_idx % 100 == 0 and f_idx > 0:
                print(f"{vid_name}: {f_idx} frames processed")
                
            f_idx += 1
            
        cap.release()
        out_video.release()
        print(f"Saved {out_path}")

    print("All 2D overlay projections completed!")

if __name__ == "__main__":
    main()
