"""
檔案目的: 執行 YOLO11 Pose 姿態辨識，將影片中的人物轉換為 2D 骨架數據 (.txt 或 .npy)。
注意: 確保 YOLO 模型檔 (.pt) 路徑正確，且會消耗較多 GPU 資源。
呼叫指令: python step1_run_yolo_pose.py
"""
import cv2
import os
from ultralytics import YOLO

def process_video(video_path, output_txt, model):
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    print(f"Processing {video_path}...")
    cap = cv2.VideoCapture(video_path)
    frame_idx = 1
    
    with open(output_txt, 'w') as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model(frame, verbose=False)
            
            # Write keypoints
            if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                # Take the first person detected
                keypoints = results[0].keypoints.xy[0].cpu().numpy()
                for j_idx, (x, y) in enumerate(keypoints):
                    f.write(f"{frame_idx},{j_idx},{int(x)},{int(y)}\n")
            else:
                # Missing points for this frame
                for j_idx in range(17):
                    f.write(f"{frame_idx},{j_idx},0,0\n")
                    
            frame_idx += 1
            
    cap.release()
    print(f"Saved keypoints to {output_txt}")

if __name__ == "__main__":
    model_path = r"E:\squat\recordings_20260507_done\yolo11\yolo11x-pose.pt"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        exit(1)
        
    model = YOLO(model_path)
    
    base_dir = r"E:\squat\squat_dataset\S001_Pitt\session02\recording_20260207_163410"
    
    videos = ["RLU.avi", "FL.avi", "FR.avi", "RR.avi"]
    
    for vid in videos:
        video_path = os.path.join(base_dir, vid)
        output_txt = os.path.join(base_dir, f"yolo_skeleton_{vid.replace('.avi', '')}.txt")
        process_video(video_path, output_txt, model)

