r"""
Runner Entry Point: Compare AGFormer 3D Wrist Midpoint Trajectory vs YOLO Barbell Tracking
調度 compare_wrist_bar 模組進行軌跡比對與視覺化診斷。

使用範例:
python run_compare_AGFormer_wrist_bar.py --target_dir "E:\\squat\\squat_dataset\\S001_Pitt\\session02\\recording_20260207_163410"
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse

# Ensure current directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from compare_wrist_bar.compare_wrist_vs_bar import run_comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AGFormer 3D Wrist Midpoint vs YOLO Barbell Trajectory Comparison")
    parser.add_argument("--target_dir", type=str, default=r"E:\squat\squat_dataset\S001_Pitt\session02\recording_20260207_163410", help="Path to recording folder containing yolo_skeleton.txt / yolo_coordinates.txt")
    parser.add_argument("--agformer_ckpt", type=str, default=r"D:\Pitt\Project\tools\MotionAGFormer\checkpoint\motionagformer-b-h36m.pth.tr", help="MotionAGFormer model checkpoint")
    args = parser.parse_args()

    run_comparison(args.target_dir, args.agformer_ckpt)
