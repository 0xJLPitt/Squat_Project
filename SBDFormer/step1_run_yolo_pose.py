#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
Step 1: 臥推影片姿態辨識工具 (Run YOLO Pose on Videos - SBDFormer Adapter)
================================================================================
架構設計 (Thin Wrapper Pattern):
  本腳本遵循 Clean Code 與 DRY (Don't Repeat Yourself) 原則，核心 YOLO11 Pose
  模型載入與逐幀關鍵點推論演算法直接引用自:
    -> Squat_2dto3d/step1_run_yolo_pose.py (Single Source of Truth)

  本腳本專注於 SBDFormer 臥推多視角資料集規格適配:
    1. 支援 --subject 直接定位受試者母目錄 (如 sub2/)，智慧尋找主相機視角 (sub2-i17/)
       並將 2D 骨架數據自動輸出至該受試者母目錄下的 dataprocess/。
    2. 自動過濾排除棋盤格影片 (如 checkboard_external2.MP4)。
    3. 骨架檔案統一命名為 yolo_skeleton_{videoname}.txt。

使用範例:
  1. 針對受試者母目錄全自動批次處理 (推薦):
     $ mamba run -n hw1 python step1_run_yolo_pose.py --subject "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2"

  2. 指定相機視角資料夾與輸出路徑:
     $ mamba run -n hw1 python step1_run_yolo_pose.py \
         -d "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\sub2-i17" \
         -o "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\dataprocess"

  3. 跑單一影片:
     $ mamba run -n hw1 python step1_run_yolo_pose.py \
         -v "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\sub2-i17\error1.MP4" \
         -o "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\dataprocess"

  4. 預設執行 (自動載入預設 sub2 目錄):
     $ mamba run -n hw1 python step1_run_yolo_pose.py
================================================================================
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

# 設定 Windows 控制台 UTF-8 編碼
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ------------------------------------------------------------------------------
# 引用共用模組 (Squat_2dto3d/step1_run_yolo_pose.py)
# ------------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
SQUAT_2DTO3D_DIR = CURRENT_DIR.parent / "Squat_2dto3d"

if str(SQUAT_2DTO3D_DIR) not in sys.path:
    sys.path.insert(0, str(SQUAT_2DTO3D_DIR))

try:
    from step1_run_yolo_pose import (
        get_model,
        load_yolo_model,
        process_single_video,
        process_single_video_entry,
    )
except ImportError as e:
    raise ImportError(
        f"無法從 {SQUAT_2DTO3D_DIR} 載入 step1_run_yolo_pose 核心模組: {e}"
    )

# ------------------------------------------------------------------------------
# 12 身體關鍵點定義 (過濾移除頭部 5 點: 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear)
# ------------------------------------------------------------------------------
# 原 COCO 17 點中保留: 雙肩(5,6)、雙肘(7,8)、雙腕(9,10)、雙髖(11,12)、雙膝(13,14)、雙踝(15,16)
SELECTED_KEYPOINT_INDICES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

BODY_12_KEYPOINT_NAMES = [
    "left_shoulder",   # 0 (原 COCO 5)
    "right_shoulder",  # 1 (原 COCO 6)
    "left_elbow",      # 2 (原 COCO 7)
    "right_elbow",     # 3 (原 COCO 8)
    "left_wrist",      # 4 (原 COCO 9)
    "right_wrist",     # 5 (原 COCO 10)
    "left_hip",        # 6 (原 COCO 11)
    "right_hip",       # 7 (原 COCO 12)
    "left_knee",       # 8 (原 COCO 13)
    "right_knee",      # 9 (原 COCO 14)
    "left_ankle",      # 10 (原 COCO 15)
    "right_ankle"      # 11 (原 COCO 16)
]
NUM_BODY_KEYPOINTS = len(SELECTED_KEYPOINT_INDICES)  # 12


def is_valid_12_keypoints_file(txt_path: Path) -> bool:
    """檢查現有骨架檔案是否已為 12 關鍵點格式 (避免 17 點舊格式被誤判為已完成)"""
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        return False
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for _ in range(30):
                line = f.readline()
                if not line:
                    break
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    joint_idx = int(parts[1])
                    if joint_idx > 11:
                        return False
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------
# SBDFormer 臥推預設路徑配置
# ------------------------------------------------------------------------------
DEFAULT_SUBJECT_DIR = Path(r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2")
DEFAULT_CAM_NAME = "sub2-i17"


def resolve_subject_paths(subject_dir: Path) -> Tuple[Path, Path]:
    """
    自受試者母目錄智慧解析主相機目錄 (i17) 與 dataprocess 目錄。
    """
    subject_dir = Path(subject_dir)
    dataprocess_dir = subject_dir / "dataprocess"

    # 尋找名稱包含 i17 的相機目錄 (作為臥推的主分析視角)
    cam_dir = subject_dir / DEFAULT_CAM_NAME
    if not cam_dir.exists():
        for d in subject_dir.iterdir():
            if d.is_dir() and "i17" in d.name.lower():
                cam_dir = d
                break

    return cam_dir, dataprocess_dir


def process_benchpress_directory(
    cam_dir: Path,
    output_dir: Path,
    model_or_path: Optional[str] = None,
    conf_thresh: float = 0.25,
    overwrite: bool = False,
    ignore_checkboard: bool = True
) -> int:
    """
    批次處理相機視角目錄內的所有臥推影片，自動排除 checkboard 影片。
    """
    cam_dir = Path(cam_dir)
    output_dir = Path(output_dir)

    if not cam_dir.exists():
        print(f"[ERROR] 找不到相機目錄: {cam_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = [f for f in sorted(cam_dir.iterdir()) if f.suffix.lower() in video_extensions]

    if ignore_checkboard:
        video_files = [f for f in video_files if "checkboard" not in f.stem.lower()]

    # 排除已生成的視覺化影片
    exclude_tags = ["_overlay", "_visualize", "_projected", "_2d_skeleton"]
    video_files = [
        f for f in video_files
        if not any(tag in f.stem.lower() for tag in exclude_tags)
    ]

    if not video_files:
        print(f"[WARN] 在 {cam_dir} 中找不到待處理的動作影片。")
        return 0

    print("=" * 70)
    print(f"🎬 執行 Step 1: 臥推影片 YOLO Pose 姿態辨識 (僅輸出身體 12 關鍵點)")
    print(f" 📂 影片來源目錄: {cam_dir}")
    print(f" 📁 骨架輸出目錄: {output_dir}")
    print(f" 🦴 關鍵點模式  : 12 關鍵點 (已移除頭部臉部 5 點，保留肩肘腕髖膝踝)")
    print(f" 🎞️ 待處理影片數: {len(video_files)} 部")
    for idx, vf in enumerate(video_files, 1):
        print(f"   [{idx}] {vf.name}")
    print("=" * 70)

    # 預先檢查是否全部已完成 (必須存在且符合 12 關鍵點格式)
    tasks = []
    for v_file in video_files:
        out_txt = output_dir / f"yolo_skeleton_{v_file.stem}.txt"
        tasks.append((v_file, [out_txt]))

    if not overwrite and all(all(is_valid_12_keypoints_file(p) for p in outs) for _, outs in tasks):
        print("[INFO] 所有影片骨架檔案皆已存在且為 12 點格式，略過模型載入與處理。若要重新產生請加上 --overwrite 參數。")
        return len(tasks)

    # 延遲載入 YOLO 模型
    model = get_model(model_or_path)

    success_count = 0
    for v_file, output_txts in tasks:
        ok = process_single_video(
            video_path=v_file,
            output_txts=output_txts,
            model=model,
            conf_thresh=conf_thresh,
            overwrite=overwrite,
            keypoint_indices=SELECTED_KEYPOINT_INDICES
        )
        if ok:
            success_count += 1

    print(f"🎉 全部處理完畢！成功處理: {success_count} / {len(tasks)} 部影片。")
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: 臥推多視角影片 YOLO Pose 姿態辨識 (12 身體關鍵點輸出 - SBDFormer Adapter)"
    )
    parser.add_argument(
        "--subject", "-s",
        type=str,
        default=None,
        help=f"受試者母目錄路徑 (例如 {DEFAULT_SUBJECT_DIR}，自動定位 i17 視角並輸出至 dataprocess/)"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default=None,
        help="指定包含待處理影片的資料夾路徑 (例如 sub2-i17)"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="指定單一待處理影片檔案完整路徑"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="輸出骨架 txt 目錄或單一檔案完整路徑 (預設為 dataprocess/)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="YOLO 模型權重路徑 (.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO 辨識信心閾值 (預設 0.25)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="若骨架檔案已存在是否強制覆蓋重跑"
    )

    args = parser.parse_args()

    # 模式 A: 指定單一影片
    if args.video:
        process_single_video_entry(
            video_path=args.video,
            model_or_path=args.model,
            output_target=args.output,
            conf_thresh=args.conf,
            overwrite=args.overwrite,
            keypoint_indices=SELECTED_KEYPOINT_INDICES
        )
        return

    # 模式 B: 指定受試者母目錄
    if args.subject:
        subj_dir = Path(args.subject)
        cam_dir, dp_dir = resolve_subject_paths(subj_dir)
        out_dir = Path(args.output) if args.output else dp_dir
        process_benchpress_directory(
            cam_dir=cam_dir,
            output_dir=out_dir,
            model_or_path=args.model,
            conf_thresh=args.conf,
            overwrite=args.overwrite
        )
        return

    # 模式 C: 指定特定相機資料夾
    if args.dir:
        cam_dir = Path(args.dir)
        out_dir = Path(args.output) if args.output else cam_dir.parent / "dataprocess"
        process_benchpress_directory(
            cam_dir=cam_dir,
            output_dir=out_dir,
            model_or_path=args.model,
            conf_thresh=args.conf,
            overwrite=args.overwrite
        )
        return

    # 模式 D: 預設執行 (無參數時自動載入預設 sub2 目錄)
    print(f"[INFO] 未指定參數，自動採用預設受試者母目錄: {DEFAULT_SUBJECT_DIR}")
    cam_dir, dp_dir = resolve_subject_paths(DEFAULT_SUBJECT_DIR)
    process_benchpress_directory(
        cam_dir=cam_dir,
        output_dir=dp_dir,
        model_or_path=args.model,
        conf_thresh=args.conf,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
