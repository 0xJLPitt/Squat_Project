r"""
檔案目的: 執行 YOLO11 Pose 姿態辨識，將影片中的人物轉換為 2D 骨架數據 (.txt)。
支援功能:
  1. 支援批次處理整個資料夾內的影片。
  2. 支援指定完整路徑的單一影片。
  3. 檔名命名規則:
     - 統一輸出 yolo_skeleton 格式: yolo_skeleton_{影片名}.txt (單一檔案，乾淨不重複)。
  4. 支援命令列參數與無參數互動模式。

呼叫範例:
  1. 跑整個資料夾:
     python step1_run_yolo_pose.py -d "D:\Pitt\Project\Squat_Project\video\benchpress_3D\i17\sub2"
  2. 跑單一影片 (完整路徑):
     python step1_run_yolo_pose.py -v "D:\Pitt\Project\Squat_Project\video\benchpress_3D\i17\sub2\error1.MP4"
  3. 直接給路徑 (自動判斷影片或資料夾):
     python step1_run_yolo_pose.py "D:\Pitt\Project\Squat_Project\video\benchpress_3D\i17\sub2"
  4. 互動模式:
     python step1_run_yolo_pose.py
"""

import os
import sys
import glob
import shutil
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 預設 YOLO Pose 模型備選路徑清單
DEFAULT_MODEL_CANDIDATES = [
    r"E:\squat\recordings_20260507_done\yolo11\yolo11x-pose.pt",
    r"E:\squat\recordings_20260507_0628_fixed棋盤格\yolo11\yolo11x-pose.pt",
    "yolo11x-pose.pt"
]

def load_yolo_model(model_path=None):
    """
    載入 YOLO Pose 模型，若未指定路徑則自動自候選路徑尋找。
    """
    if model_path and os.path.exists(model_path):
        target_path = model_path
    else:
        target_path = None
        if model_path:
            print(f"[WARN] 指定的模型路徑不存在: {model_path}，嘗試尋找預設路徑...")
            
        for cand in DEFAULT_MODEL_CANDIDATES:
            if os.path.exists(cand):
                target_path = cand
                break
                
        if target_path is None:
            # 最後備選使用官方預設名稱 (ultralytics 會自動下載)
            target_path = "yolo11x-pose.pt"

    print(f"[INFO] 正在載入 YOLO 模型: {target_path}")
    model = YOLO(target_path)
    print(f"[INFO] YOLO 模型載入成功！\n")
    return model

def get_model(model_or_path):
    """取得或延遲載入模型物件"""
    if isinstance(model_or_path, YOLO):
        return model_or_path
    return load_yolo_model(model_or_path)

def process_single_video(video_path, output_txts, model, conf_thresh=0.25, overwrite=False, keypoint_indices=None):
    """
    執行單一影片的 YOLO 姿態辨識並輸出骨架數據。
    
    :param video_path: 影片路徑
    :param output_txts: 輸出的 txt 檔案路徑列表 (可同時寫入主檔並建立相容檔)
    :param model: 已載入的 YOLO 模型物件
    :param conf_thresh: 辨識信心閾值
    :param overwrite: 若檔案已存在是否覆蓋
    :param keypoint_indices: 指定擷取的關鍵點索引清單 (例如 [5,6,...,16] 僅擷取身體 12 點，預設 None 輸出全部 17 點)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] 找不到影片檔案: {video_path}")
        return False

    if isinstance(output_txts, (str, Path)):
        output_txts = [Path(output_txts)]
    else:
        output_txts = [Path(p) for p in output_txts]

    primary_txt = output_txts[0]
    
    # 檢查是否所有目標檔案皆已存在且不需要覆蓋
    if not overwrite and all(p.exists() and p.stat().st_size > 0 for p in output_txts):
        print(f"[SKIP] 已存在骨架檔案，跳過: {video_path.name}")
        for p in output_txts:
            print(f"       -> {p}")
        return True

    # 確保輸出目錄存在
    primary_txt.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟影片: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[PROCESS] 開始處理: {video_path.name}")
    print(f"   解析度: {w}x{h} | FPS: {fps:.2f} | 總幀數: {total_frames}")

    frame_idx = 1
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=f"   {video_path.stem}", leave=False)

    with open(primary_txt, 'w', encoding='utf-8') as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False, conf=conf_thresh)

            # 選擇最合適的人物 (若有多人則選擇 bounding box 面積最大者)
            best_person_idx = 0
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 1:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_person_idx = int(np.argmax(areas))

            # 寫入關鍵點 (支援自訂關鍵點篩選或預設 COCO 17 keypoints: frame_idx, joint_idx, x, y)
            if (len(results) > 0 and 
                results[0].keypoints is not None and 
                len(results[0].keypoints.xy) > best_person_idx):
                
                keypoints = results[0].keypoints.xy[best_person_idx].cpu().numpy()
                if keypoint_indices is not None:
                    keypoints = keypoints[keypoint_indices]
                for j_idx, (x, y) in enumerate(keypoints):
                    f.write(f"{frame_idx},{j_idx},{int(x)},{int(y)}\n")
            else:
                # 該幀漏抓時補 0
                num_kpts = len(keypoint_indices) if keypoint_indices is not None else 17
                for j_idx in range(num_kpts):
                    f.write(f"{frame_idx},{j_idx},0,0\n")

            frame_idx += 1
            pbar.update(1)

    pbar.close()
    cap.release()

    # 若有設定多個輸出路徑 (例如 skeleton_{name}.txt 與 yolo_skeleton_{name}.txt)，複製一份以保證相容
    for extra_txt in output_txts[1:]:
        extra_txt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(primary_txt, extra_txt)

    print(f"   [SUCCESS] 處理完成！骨架數據已存至:")
    for p in output_txts:
        print(f"      - {p}")
    print()
    return True

def process_directory(directory_path, model_or_path, output_root=None, recursive=False, pattern=None, conf_thresh=0.25, overwrite=False, keypoint_indices=None):
    """
    批次處理指定資料夾內的所有影片。
    
    :param directory_path: 影片資料夾路徑
    :param model_or_path: YOLO 模型物件或權重路徑
    :param output_root: 指定輸出根目錄 (若為 None 則與影片同目錄)
    :param recursive: 是否遞迴搜尋子目錄
    :param pattern: 檔名過濾關鍵字
    :param conf_thresh: 辨識信心閾值
    :param overwrite: 是否強制覆蓋
    :param keypoint_indices: 指定擷取的關鍵點索引清單
    """
    target_dir = Path(directory_path)
    if not target_dir.exists():
        print(f"[ERROR] 找不到目錄: {directory_path}")
        return

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    
    if recursive:
        candidates = [f for f in target_dir.rglob('*') if f.suffix.lower() in video_extensions]
    else:
        candidates = [f for f in target_dir.glob('*') if f.suffix.lower() in video_extensions]

    # 排除已生成的視覺化影片 (避免重複處理)
    exclude_tags = ['_overlay', '_visualize', '_projected', '_2d_skeleton']
    video_files = [
        f for f in candidates 
        if not any(tag in f.stem.lower() for tag in exclude_tags)
    ]

    if pattern:
        video_files = [f for f in video_files if pattern.lower() in f.name.lower()]

    video_files = sorted(video_files)

    if not video_files:
        print(f"[WARN] 在 {directory_path} 中找不到符合條件的影片檔案 (過濾條件: pattern='{pattern}')。")
        return

    is_multiple = len(video_files) > 1
    print(f"[START] 發現 {len(video_files)} 個影片，準備開始進行 YOLO Pose 姿態辨識...")
    for idx, v in enumerate(video_files, 1):
        print(f"   [{idx}] {v.name}")
    print("-" * 60)

    # 檢查是否所有檔案皆已完成且不需覆蓋
    tasks = []
    for video in video_files:
        out_folder = Path(output_root) if output_root else video.parent
        name = video.stem
        # 統一輸出 yolo_skeleton 格式: yolo_skeleton_{name}.txt
        primary_txt = out_folder / f"yolo_skeleton_{name}.txt"
        tasks.append((video, [primary_txt]))

    if not overwrite and all(all(p.exists() and p.stat().st_size > 0 for p in outs) for _, outs in tasks):
        print("[INFO] 所有影片骨架檔案皆已存在，略過模型載入與處理。若要重新產生請加上 --overwrite 參數。")
        return

    # 延遲載入 YOLO 模型
    model = get_model(model_or_path)

    for video, output_txts in tasks:
        process_single_video(
            video_path=video,
            output_txts=output_txts,
            model=model,
            conf_thresh=conf_thresh,
            overwrite=overwrite,
            keypoint_indices=keypoint_indices
        )

def process_single_video_entry(video_path, model_or_path, output_target=None, conf_thresh=0.25, overwrite=False, keypoint_indices=None):
    """
    單一影片處理入口，支援指定輸出檔案名稱或目錄。
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] 找不到影片檔案: {video_path}")
        return

    name = video_path.stem

    if output_target:
        out_target = Path(output_target)
        if out_target.suffix.lower() == ".txt":
            # 指定了確切的輸出 txt 檔案
            output_txts = [out_target]
        else:
            # 指定了輸出目錄
            output_txts = [out_target / f"yolo_skeleton_{name}.txt"]
    else:
        # 未指定輸出，預設存於該影片同目錄下的 yolo_skeleton_{name}.txt
        output_txts = [video_path.parent / f"yolo_skeleton_{name}.txt"]

    # 檢查是否已存在
    if not overwrite and all(p.exists() and p.stat().st_size > 0 for p in output_txts):
        print(f"[SKIP] 已存在骨架檔案，跳過: {video_path.name}")
        for p in output_txts:
            print(f"       -> {p}")
        return

    model = get_model(model_or_path)

    process_single_video(
        video_path=video_path,
        output_txts=output_txts,
        model=model,
        conf_thresh=conf_thresh,
        overwrite=overwrite,
        keypoint_indices=keypoint_indices
    )

def main():
    parser = argparse.ArgumentParser(
        description="執行 YOLO11 Pose 姿態辨識，支援單一影片或批次處理整個資料夾"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="目標路徑 (可為單一影片檔或資料夾路徑)"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="單一影片完整路徑"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default=None,
        help="目標影片資料夾路徑"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="輸出 .txt 檔路徑或輸出目錄"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="YOLO 模型權重路徑 (.pt)"
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default=None,
        help="檔名過濾關鍵字 (例如 REC 或 FL)"
    )
    parser.add_argument(
        "--recursive", "-r",
        dest="recursive",
        action="store_true",
        default=False,
        help="是否遞迴搜尋子目錄 (預設 False)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO 辨識信心度閾值 (預設 0.25)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="若骨架檔案已存在是否強制覆蓋重跑"
    )

    args = parser.parse_args()

    # 決定目標輸入路徑 (優先級: --video > --dir > positional input)
    target_input = None
    is_video_mode = False

    if args.video:
        target_input = args.video
        is_video_mode = True
    elif args.dir:
        target_input = args.dir
        is_video_mode = False
    elif args.input:
        target_input = args.input
        target_path = Path(target_input)
        if target_path.is_file():
            is_video_mode = True
        elif target_path.is_dir():
            is_video_mode = False

    # 若有帶入參數，直接執行
    if target_input is not None:
        target_path = Path(target_input)
        if not target_path.exists():
            print(f"[ERROR] 指定的路徑不存在: {target_input}")
            sys.exit(1)

        if is_video_mode or target_path.is_file():
            process_single_video_entry(
                video_path=target_path,
                model_or_path=args.model,
                output_target=args.output,
                conf_thresh=args.conf,
                overwrite=args.overwrite
            )
        else:
            process_directory(
                directory_path=target_path,
                model_or_path=args.model,
                output_root=args.output,
                recursive=args.recursive,
                pattern=args.pattern,
                conf_thresh=args.conf,
                overwrite=args.overwrite
            )
        return

    # --- 互動模式 ---
    default_dir = r"E:\squat\squat_dataset\S001_Pitt\session02\recording_20260207_163410"
    if not os.path.exists(default_dir):
        default_dir = r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\i15\sub1"

    print("=" * 65)
    print("🎬 YOLO11 Pose 姿態辨識批次 / 單片處理工具")
    print(f"預設處理目錄: {default_dir}")
    print("您可直接輸入「單一影片完整路徑」或「影片資料夾路徑」，按 Enter 使用預設目錄。")
    print("=" * 65)
    
    user_input = input("請輸入影片路徑或資料夾 (直接按 Enter 使用預設目錄): ").strip()
    target_str = user_input if user_input else default_dir
    target_path = Path(target_str)

    if not target_path.exists():
        print(f"[ERROR] 輸入的路徑不存在: {target_str}")
        sys.exit(1)

    if target_path.is_file():
        # 單一影片互動模式
        process_single_video_entry(
            video_path=target_path,
            model_or_path=args.model,
            conf_thresh=args.conf,
            overwrite=args.overwrite
        )
    else:
        # 資料夾批次互動模式
        pattern_str = input("請輸入檔名過濾關鍵字 (直接按 Enter 處理全部影片，例如輸入 REC 或 FL): ").strip()
        pattern = pattern_str if pattern_str else None

        rec_str = input("是否遞迴搜尋子目錄？(y/N，預設否): ").strip().lower()
        recursive = (rec_str == 'y')

        process_directory(
            directory_path=target_path,
            model_or_path=args.model,
            output_root=args.output,
            recursive=recursive,
            pattern=pattern,
            conf_thresh=args.conf,
            overwrite=args.overwrite
        )

if __name__ == "__main__":
    main()
