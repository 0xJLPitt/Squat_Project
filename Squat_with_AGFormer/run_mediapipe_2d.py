"""
目的: 使用 MediaPipe 模型從影片中提取 2D 骨架點。
執行範例:
python run_mediapipe_2d.py --video D:\Pitt\Project\squat\demo\recording_20260415_105356\vision2.avi --out mediapipe_keypoints.npz
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import argparse
import os

def process_video_mediapipe(video_path, output_npz):
    if not os.path.exists(video_path):
        print(f"找不到影片: {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"影片解析度: {w}x{h}, 總影格數: {n_frames}")
    
    all_kpts = []
    
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = detector.detect(mp_image)
        
        kpts = np.zeros((17, 3), dtype=np.float32)
        
        if detection_result.pose_landmarks:
            lm = detection_result.pose_landmarks[0]
            
            def get_pt(idx):
                return np.array([lm[idx].x * w, lm[idx].y * h, lm[idx].visibility])

            l_hip = get_pt(23)
            r_hip = get_pt(24)
            l_sh = get_pt(11)
            r_sh = get_pt(12)
            nose = get_pt(0)
            
            kpts[1] = r_hip
            kpts[2] = get_pt(26)
            kpts[3] = get_pt(28)
            
            kpts[4] = l_hip
            kpts[5] = get_pt(25)
            kpts[6] = get_pt(27)
            
            kpts[11] = l_sh
            kpts[12] = get_pt(13)
            kpts[13] = get_pt(15)
            
            kpts[14] = r_sh
            kpts[15] = get_pt(14)
            kpts[16] = get_pt(16)
            
            kpts[0] = (l_hip + r_hip) * 0.5
            kpts[8] = (l_sh + r_sh) * 0.5
            kpts[7] = (kpts[0] + kpts[8]) * 0.5
            kpts[9] = nose
            head_vec = kpts[9][:2] - kpts[8][:2]
            kpts[10][:2] = kpts[9][:2] + head_vec * 0.3
            kpts[10][2] = nose[2]
        
        all_kpts.append(kpts)
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"\rProcessing frame {frame_idx}/{n_frames}", end="")
            
    cap.release()
    print("\n完成 2D MediaPipe 偵測！")
    
    all_kpts = np.array(all_kpts)
    all_kpts = np.expand_dims(all_kpts, axis=0)
    
    np.savez_compressed(output_npz, reconstruction=all_kpts)
    print(f"儲存為 {output_npz}")
    
    np.savez_compressed(output_npz, reconstruction=all_kpts)
    print(f"儲存為 {output_npz}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()
    process_video_mediapipe(args.video, args.out)
