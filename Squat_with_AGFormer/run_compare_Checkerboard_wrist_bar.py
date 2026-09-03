r"""
Runner Entry Point: Compare Multi-View Chessboard Calibrated 3D Wrist Trajectory vs YOLO Barbell Tracking
調度 compare_wrist_bar 模組進行多視角棋盤格 3D 校正手腕 vs 2D 槓鈴比對與視覺化診斷。
所有產出圖表與數據均會儲存於目標資料夾下的 Checkerboard3d/ 子資料夾中。

使用範例:
python run_compare_calibrated_wrist_bar.py --target_dir "E:\squat\squat_dataset\S001_Pitt\session02\recording_20260207_163410"
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse

# Ensure current directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from compare_wrist_bar.compare_calibrated_wrist_vs_bar import run_calibrated_comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-View Chessboard Calibrated 3D Wrist vs YOLO Barbell Comparison")
    parser.add_argument("--target_dir", type=str, default=r"E:\squat\squat_dataset\S001_Pitt\session02\recording_20260207_163410", help="Path to recording folder containing multi-view yolo_skeleton_*.txt and yolo_coordinates.txt")
    parser.add_argument("--intrinsic_dir", type=str, default=r"E:\squat\recordings_20260507_0628_fixed棋盤格\intrinsics_vision", help="Path to camera intrinsics folder")
    parser.add_argument("--extrinsic_dir", type=str, default=r"E:\squat\recordings_20260507_0628_fixed棋盤格\recording_20260211_棋盤格", help="Path to camera extrinsics folder")
    args = parser.parse_args()

    run_calibrated_comparison(args.target_dir, args.intrinsic_dir, args.extrinsic_dir)
