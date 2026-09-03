r"""
目的: 將 YOLO 骨架點畫在原始影片上。
執行範例:
python overlay_yolo.py --video D:\Pitt\Project\squat\demo\recording_20260415_105356\vision2.avi --txt D:\Pitt\Project\squat\demo\recording_20260415_105356\yolo_skeleton.txt --out yolo_overlay_output.mp4
"""
import cv2
import argparse
import os
import numpy as np

def overlay_yolo(video_path, txt_path, output_path):
    print(f"讀取影片: {video_path}")
    if not os.path.exists(video_path):
        print("找不到影片檔案！")
        return
        
    cap = cv2.VideoCapture(video_path)
    
    print(f"讀取 YOLO 骨架資料: {txt_path}")
    if not os.path.exists(txt_path):
        print("找不到 txt 檔案！")
        return

    # Parse the text file
    # Format: frame_idx, joint_idx, x, y
    keypoints_dict = {}
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                frame_idx = int(parts[0]) - 1 # Convert to 0-indexed
                joint_idx = int(parts[1])
                x = int(float(parts[2]))
                y = int(float(parts[3]))
                
                if frame_idx not in keypoints_dict:
                    keypoints_dict[frame_idx] = {}
                keypoints_dict[frame_idx][joint_idx] = (x, y)
                
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30.0, (w, h))
    
    # COCO Skeleton format for YOLO
    skeleton = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4), (3, 5), (4, 6)
    ]
    
    # Colors for different parts (OpenCV uses BGR)
    color_point = (0, 255, 0) # Green for joints
    color_left = (255, 0, 0)  # Blue for left side
    color_right = (0, 0, 255) # Red for right side
    color_center = (0, 255, 255) # Yellow for center/face
    
    def get_bone_color(p1, p2):
        # Even indices (except 0) are right side (2, 4, 6, 8, 10, 12, 14, 16)
        # Odd indices are left side (1, 3, 5, 7, 9, 11, 13, 15)
        # Center points: 0
        if p1 % 2 != 0 and p2 % 2 != 0:
            return color_left
        elif p1 % 2 == 0 and p2 % 2 == 0 and p1 != 0 and p2 != 0:
            return color_right
        else:
            return color_center

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in keypoints_dict:
            kpts = keypoints_dict[frame_idx]
            
            # Draw bones
            for pt1, pt2 in skeleton:
                if pt1 in kpts and pt2 in kpts:
                    x1, y1 = kpts[pt1]
                    x2, y2 = kpts[pt2]
                    
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        color = get_bone_color(pt1, pt2)
                        cv2.line(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw joints
            for j in range(17):
                if j in kpts:
                    x, y = kpts[j]
                    if x > 0 and y > 0:
                        cv2.circle(frame, (x, y), 4, color_point, -1)
                        
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"影片處理完成！已儲存至: {output_path}")

# Compatibility alias
overlay_yolo_pose = overlay_yolo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="將 YOLO 骨架點畫在影片上")
    parser.add_argument('--video', type=str, required=True, help="原始影片路徑 (.avi / .mp4)")
    parser.add_argument('--txt', type=str, required=True, help="YOLO 骨架檔案路徑 (yolo_skeleton.txt)")
    parser.add_argument('--out', type=str, default='yolo_overlay_output.mp4', help="輸出影片名稱")
    args = parser.parse_args()
    
    overlay_yolo(args.video, args.txt, args.out)
