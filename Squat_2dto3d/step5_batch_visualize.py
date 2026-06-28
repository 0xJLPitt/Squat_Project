"""
檔案目的: 批次產生校正對照圖 (產生 2-3, 3-4, 4-5 的角點匹配結果圖)，用於檢查外參校正是否正確對齊。
注意: Windows 系統下若路徑有中文，OpenCV 存圖會失敗，此腳本已使用 imencode 避開此問題。
呼叫指令: python step5_batch_visualize.py
"""
import os
import glob
import cv2
import numpy as np
import shutil
from calibration_visualizer import save_stereo_visualization

def imread_zh(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

folders = [
    "recording_20251205_092753_棋盤",
    "recording_20251218_093931_棋盤格",
    "recording_20260211_棋盤格",
    "recording_20260212_090529_棋盤",
    "recording_20260309_102114_棋盤",
    "recording_20260317_174312_棋盤",
    "recording_20260411_175733_棋盤",
    "recording_20260415_150118_棋盤2代",
    "recording_20260420_102021_棋盤",
    "recording_20260420_131935_棋盤",
    "recording_20260422_110624_棋盤",
    "recording_20260422_153250_棋盤",
    "recording_20260504_115055_棋盤",
    "recording_20260505_111254_棋盤"
]

base_dir = r'E:\squat\recordings_202628_fixed'
w, h = 5, 3

pairs_to_check = [
    (('vision2', 'vision2_jpg'), ('vision3', 'vision3_jpg')),
    (('vision3', 'vision3_jpg'), ('vision4', 'vision4_jpg')),
    (('vision4', 'vision4_jpg'), ('vision5', 'vision5_jpg')),
    (('RR', 'RR_jpg'), ('RLU', 'RLU_jpg')),
    (('RLU', 'RLU_jpg'), ('FL', 'FL_jpg')),
    (('FL', 'FL_jpg'), ('FR', 'FR_jpg'))
]

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        continue
        
    vis_dir = os.path.join(folder_path, "calibration_visualized")
    
    # Empty or create the directory
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir)
    
    print(f"Processing: {folder}")
    
    for (id1, dir1_name), (id2, dir2_name) in pairs_to_check:
        dir1 = os.path.join(folder_path, dir1_name)
        dir2 = os.path.join(folder_path, dir2_name)
        
        if not os.path.exists(dir1) or not os.path.exists(dir2):
            continue
            
        imgs1 = sorted(glob.glob(os.path.join(dir1, '*.jpg')))
        imgs2 = sorted(glob.glob(os.path.join(dir2, '*.jpg')))
        
        if not imgs1 or not imgs2:
            continue
            
        # Find the first valid pair
        for f1, f2 in zip(imgs1, imgs2):
            im1 = imread_zh(f1)
            im2 = imread_zh(f2)
            
            gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            
            ret1, c1 = cv2.findChessboardCornersSB(gray1, (w, h), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            ret2, c2 = cv2.findChessboardCornersSB(gray2, (w, h), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            
            if ret1 and ret2:
                # Apply 180 flip if vision3 -> vision4 or RLU -> FL
                if (id1 == "vision3" and id2 == "vision4") or (id1 == "RLU" and id2 == "FL"):
                    c2 = c2[::-1, :, :]
                    
                filename = os.path.basename(f1)
                save_stereo_visualization(im1, im2, c1, c2, id1, id2, filename, vis_dir, pattern_size=(w, h), is_manual=False)
                print(f"  Generated image for pair {id1}-{id2} using {filename}")
                break # Stop searching for this pair
