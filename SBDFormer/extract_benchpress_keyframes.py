#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
臥推關鍵影格多視角批量擷取腳本 (Extract Bench Press Keyframes from Multi-Camera Videos)
================================================================================
功能說明:
  1. 自動掃描指定目錄下所有視角資料夾 (如 sub1-i15, sub2-i16, sub2-i17) 中的影片。
  2. 自動排除檔名含有 "checkboard" 的棋盤格影片 (例如 checkboard_external2.MP4)。
  3. 自動比對 dataprocess/ 中的各影片切片標註檔 (*_segments.json):
     - yolo_skeleton_error1_segments.json
     - yolo_skeleton_error2_segments.json
     - yolo_skeleton_error3_segments.json
     - yolo_skeleton_error4_segments.json
     - yolo_skeleton_normal_segments.json
  4. 預設自動提取 Rep 1 ~ Rep n 的 4 大關鍵動作影格:
     (1) top_start_frame   [頂點/起]
     (2) descent_mid_frame [下降中段]
     (3) bottom_frame      [底點觸胸]
     (4) ascent_mid_frame  [上升中段]
  5. 分別在各相機母資料夾中為每部影片建立資料夾，輸出命名為 videoname_framexxx.jpg。

使用範例:
  1. 預設一鍵批次全跑 (自動抓出全部非 checkboard 影片的所有 Rep 1~n 關鍵影格):
     $ mamba run -n hw1 python D:\Pitt\Project\Squat_Project\SBDFormer\extract_benchpress_keyframes.py

  2. 僅跑單一影片 (例如只抽取 error1 的全部 Rep):
     $ python extract_benchpress_keyframes.py --video error1.MP4

  3. 指定只抽取特定 Rep (例如只要第 1 組):
     $ python extract_benchpress_keyframes.py --rep 1

  4. 自訂資料夾名稱不加後綴 (例如直接建立 error1/ 而非 error1_keyframes/):
     $ python extract_benchpress_keyframes.py --no-suffix
================================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import cv2

# 設定 Windows 控制台 UTF-8 編碼
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# 預設路徑配置
DEFAULT_PARENT_DIR = Path(
    r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2"
)
DEFAULT_DATAPROCESS_DIR = Path(
    r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\dataprocess"
)


def resolve_dataprocess_dir(base_dir: Path, custom_dp: Optional[Path] = None) -> Path:
    """
    智慧解析 dataprocess 資料夾路徑，優先順序:
      1. 自訂或預設存在之路徑 (如 sub2/dataprocess)
      2. sub2/dataprocess-i17
      3. sub2/*i17*/dataprocess
    """
    if custom_dp and custom_dp.exists():
        return custom_dp

    cand1 = base_dir / "dataprocess"
    if cand1.exists():
        return cand1

    cand2 = base_dir / "dataprocess-i17"
    if cand2.exists():
        return cand2

    for d in base_dir.iterdir():
        if d.is_dir() and "i17" in d.name.lower():
            sub_dp = d / "dataprocess"
            if sub_dp.exists():
                return sub_dp

    return cand1


def find_segments_json(dataprocess_dir: Path, video_stem: str) -> Optional[Path]:
    """
    在 dataprocess 目錄下根據影片檔名尋找對應的切片 JSON 檔案。
    例如: error1 -> yolo_skeleton_error1_segments.json
    """
    if not dataprocess_dir.exists():
        return None

    # 精準匹配候選
    exact_candidates = [
        dataprocess_dir / f"yolo_skeleton_{video_stem}_segments.json",
        dataprocess_dir / f"{video_stem}_segments.json",
        dataprocess_dir / f"yolo_skeleton_{video_stem.lower()}_segments.json"
    ]
    for cand in exact_candidates:
        if cand.exists():
            return cand

    # 模糊搜尋包含該影片名與 segments.json 的檔案
    for p in dataprocess_dir.glob("*_segments.json"):
        p_stem = p.stem.lower()
        if video_stem.lower() in p_stem:
            return p

    return None


def parse_keyframes_from_json(
    json_path: Path,
    rep_num: Optional[int] = None,
    all_reps: bool = True
) -> Dict[str, any]:
    """
    從 segments.json 中解析指定 rep 或全體 rep 1~n 的 4 大關鍵影格:
      - top_start_frame
      - descent_mid_frame
      - bottom_frame
      - ascent_mid_frame
    """
    if not json_path.exists():
        raise FileNotFoundError(f"找不到 JSON 切片檔案: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reps = data.get("reps", [])
    if not reps:
        raise ValueError(f"JSON 檔案中未發現 'reps' 資料: {json_path}")

    selected_reps = []
    if rep_num is not None and not all_reps:
        target = next((r for r in reps if r.get("rep") == rep_num), None)
        if target is None:
            raise ValueError(f"未在 JSON 中找到 Rep {rep_num} (檔案中共有 {len(reps)} 組)")
        selected_reps = [target]
    else:
        selected_reps = reps

    rep_details = []
    target_frames_set: Set[int] = set()

    for r in selected_reps:
        r_num = r.get("rep")
        f_dict = {
            "top_start": r.get("top_start_frame"),
            "descent_mid": r.get("descent_mid_frame"),
            "bottom": r.get("bottom_frame"),
            "ascent_mid": r.get("ascent_mid_frame")
        }
        rep_details.append({
            "rep": r_num,
            "frames": f_dict
        })
        for f_val in f_dict.values():
            if f_val is not None:
                target_frames_set.add(int(f_val))

    target_frames = sorted(list(target_frames_set))

    return {
        "rep_details": rep_details,
        "target_frames": target_frames,
        "total_reps": len(reps)
    }


def extract_specific_frames(
    video_path: Path,
    target_frames: List[int],
    output_dir: Path,
    video_name: Optional[str] = None,
    pad_width: int = 0,
    overwrite: bool = True
) -> List[Path]:
    """
    從單一影片中抽取指定幀數清單並儲存為 JPG。
    檔名格式: {video_name}_frame{frame_num}.jpg
    """
    if not video_path.exists():
        raise FileNotFoundError(f"找不到影片檔案: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if video_name is None:
        video_name = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_files = []
    sorted_frames = sorted(set(target_frames))

    for frame_idx in sorted_frames:
        if frame_idx < 1 or (total_frames > 0 and frame_idx > total_frames):
            print(f"      [WARN] 影格 {frame_idx} 超出影片總幀數範圍 (1 ~ {total_frames})，跳過。")
            continue

        if pad_width > 0:
            filename = f"{video_name}_frame{frame_idx:0{pad_width}d}.jpg"
        else:
            filename = f"{video_name}_frame{frame_idx}.jpg"

        file_path = output_dir / filename

        # 檢查是否已存在
        if not overwrite and file_path.exists():
            saved_files.append(file_path)
            continue

        # OpenCV 0-based frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"      [FAIL] 無法讀取第 {frame_idx} 幀: {video_path.name}")
            continue

        # 使用 imencode 支援 Windows 中文路徑
        success, img_encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if success:
            img_encoded.tofile(str(file_path))
            saved_files.append(file_path)
        else:
            print(f"      [FAIL] 圖片編碼失敗: {filename}")

    cap.release()
    return saved_files


def discover_videos_to_process(
    base_dir: Path,
    target_video: Optional[str] = None,
    exclude_keyword: str = "checkboard"
) -> Tuple[List[str], List[Path]]:
    """
    在母資料夾下搜尋所有子資料夾，並收集需要處理的影片檔名 (排除 checkboard)。
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"找不到母目錄: {base_dir}")

    # 取得各相機子資料夾 (如 sub1-i15, sub2-i16, sub2-i17，自動排除 dataprocess 目錄)
    cam_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and not d.name.lower().startswith("dataprocess")])
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    unique_video_stems = set()

    for cam_d in cam_dirs:
        for f in cam_d.glob("*"):
            if f.is_file() and f.suffix.lower() in video_extensions:
                stem = f.stem
                # 排除含有 exclude_keyword (如 checkboard) 的影片
                if exclude_keyword and exclude_keyword.lower() in stem.lower():
                    continue
                # 若有指定單一影片
                if target_video:
                    t_stem = Path(target_video).stem.lower()
                    if stem.lower() != t_stem:
                        continue
                unique_video_stems.add(stem)

    sorted_stems = sorted(list(unique_video_stems))
    return sorted_stems, cam_dirs


def process_all_benchpress_videos(
    base_dir: Path,
    dataprocess_dir: Path,
    target_video: Optional[str] = None,
    rep_num: Optional[int] = None,
    all_reps: bool = True,
    folder_suffix: str = "_keyframes",
    pad_width: int = 0,
    overwrite: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    全自動批次處理所有影片與所有相機視角。
    """
    video_stems, cam_dirs = discover_videos_to_process(base_dir, target_video=target_video)

    if not video_stems:
        print(f"[ERROR] 在 {base_dir} 中未發現符合條件的影片！")
        return {}

    print("=" * 75)
    print("🎬 臥推 3D 多視角關鍵影格全自動批量擷取作業啟動")
    print(f" 📂 母目錄路徑    : {base_dir}")
    print(f" 📑 切片資料目錄  : {dataprocess_dir}")
    print(f" 📷 發現視角資料夾: {[d.name for d in cam_dirs]}")
    print(f" 🎥 待處理影片清單: {video_stems} (已排除含有 'checkboard' 之影片)")
    print(f" 🔁 反覆擷取模式  : {'全部 Rep 1~n' if all_reps else f'僅 Rep {rep_num}'}")
    print(f" 🏷️ 輸出資料夾後綴: '{folder_suffix}' (格式: {{videoname}}{folder_suffix})")
    print("=" * 75)

    summary_stats = {}

    for v_idx, v_stem in enumerate(video_stems, 1):
        print(f"\n▶ [{v_idx}/{len(video_stems)}] 處理動作影片: 【{v_stem}】")

        # 1. 尋找對應的切片 JSON
        json_file = find_segments_json(dataprocess_dir, v_stem)
        if not json_file:
            print(f"   ⚠️ 未在 {dataprocess_dir} 找到 {v_stem} 的切片 JSON 檔案，跳過此影片。")
            continue

        print(f"   📄 對應切片檔案: {json_file.name}")

        try:
            parsed = parse_keyframes_from_json(json_file, rep_num=rep_num, all_reps=all_reps)
        except Exception as e:
            print(f"   ❌ 解析切片 JSON 失敗: {e}，跳過。")
            continue

        target_frames = parsed["target_frames"]
        n_reps = len(parsed["rep_details"])
        print(f"   🎯 成功取得 {n_reps} 組 Rep，共 {len(target_frames)} 個關鍵影格點 (影格範圍: {min(target_frames)} ~ {max(target_frames)})")

        summary_stats[v_stem] = {}

        # 2. 對各相機子目錄抽取影格
        for cam_d in cam_dirs:
            # 尋找該相機目錄下的同名影片 (如 error1.MP4 或 error1.mp4)
            matched_video = None
            for ext in [".MP4", ".mp4", ".mov", ".MOV", ".avi"]:
                cand = cam_d / f"{v_stem}{ext}"
                if cand.exists():
                    matched_video = cand
                    break

            if not matched_video:
                print(f"   📁 [{cam_d.name}] 未找到影片 {v_stem}.*，略過。")
                continue

            folder_name = f"{v_stem}{folder_suffix}" if folder_suffix else v_stem
            out_dir = cam_d / folder_name

            print(f"   📁 視角 [{cam_d.name}] -> 抽取至: {out_dir.name}/", end="", flush=True)

            saved_files = extract_specific_frames(
                video_path=matched_video,
                target_frames=target_frames,
                output_dir=out_dir,
                video_name=v_stem,
                pad_width=pad_width,
                overwrite=overwrite
            )

            print(f" (完成 {len(saved_files)} 張)")
            summary_stats[v_stem][cam_d.name] = len(saved_files)

    # 3. 輸出最終統計報表
    print("\n" + "=" * 75)
    print("🎉 批量關鍵影格擷取作業全部完成！統計總表:")
    print("-" * 75)
    header_cams = [d.name for d in cam_dirs]
    print(f"{'影片名稱 (Video)':<20} | " + " | ".join([f"{c:<12}" for c in header_cams]) + " | 總計張數")
    print("-" * 75)

    grand_total = 0
    for v_stem, cam_data in summary_stats.items():
        row_str = f"{v_stem:<20} | "
        v_total = 0
        for c in header_cams:
            cnt = cam_data.get(c, 0)
            v_total += cnt
            row_str += f"{cnt:<12} | "
        row_str += f"{v_total} 張"
        grand_total += v_total
        print(row_str)

    print("-" * 75)
    print(f"🌟 全部 3 個視角總共產生: {grand_total} 張關鍵影格照片")
    print("=" * 75)

    return summary_stats


def main():
    parser = argparse.ArgumentParser(
        description="從多視角攝影機影片中批量擷取所有影片的 Rep 1~n 關鍵影格 (排除 checkboard)"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default=str(DEFAULT_PARENT_DIR),
        help=f"包含各視角母資料夾的根路徑 (預設: {DEFAULT_PARENT_DIR})"
    )
    parser.add_argument(
        "--dataprocess-dir", "-dp",
        type=str,
        default=str(DEFAULT_DATAPROCESS_DIR),
        help=f"存放切片 JSON 檔案的資料夾路徑 (預設: {DEFAULT_DATAPROCESS_DIR})"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="指定單一影片檔名 (例如 error1.MP4)，若不指定則自動處理全部非 checkboard 影片"
    )
    parser.add_argument(
        "--rep", "-r",
        type=int,
        default=None,
        help="指定抽取第幾組 Rep (預設為 None，代表抽取 Rep 1 ~ Rep n 全部)"
    )
    parser.add_argument(
        "--no-suffix",
        action="store_true",
        default=False,
        help="輸出資料夾名稱不加 '_keyframes'，直接以純影片名建立資料夾 (例如 error1/)"
    )
    parser.add_argument(
        "--pad", "-p",
        type=int,
        default=0,
        help="檔名影格數字補零位數 (預設 0 為 raw 數字例如 frame298.jpg；設為 5 則為 frame00298.jpg)"
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        default=False,
        help="若圖片已存在則略過不覆蓋"
    )
    parser.add_argument(
        "--export-cvat",
        action="store_true",
        default=False,
        help="抽幀完成後自動調用 export_benchpress_cvat_labels.py 打包 CVAT 標註檔與照片 ZIP"
    )

    args = parser.parse_args()

    base_dir = Path(args.dir)
    custom_dp = Path(args.dataprocess_dir) if args.dataprocess_dir and args.dataprocess_dir != str(DEFAULT_DATAPROCESS_DIR) else None
    dp_dir = resolve_dataprocess_dir(base_dir, custom_dp=custom_dp)
    folder_suffix = "" if args.no_suffix else "_keyframes"

    all_reps = (args.rep is None)

    summary = process_all_benchpress_videos(
        base_dir=base_dir,
        dataprocess_dir=dp_dir,
        target_video=args.video,
        rep_num=args.rep,
        all_reps=all_reps,
        folder_suffix=folder_suffix,
        pad_width=args.pad,
        overwrite=not args.no_overwrite
    )

    if args.export_cvat:
        try:
            from export_benchpress_cvat_labels import build_coco_keypoints_dataset
            print("\n🚀 自動串聯執行 CVAT 標註資料集打包...")
            build_coco_keypoints_dataset(
                base_dir=base_dir,
                dataprocess_dir=dp_dir,
                cam_name="sub2-i17",
                target_video=args.video
            )
        except Exception as e:
            print(f"[WARN] 自動匯出 CVAT 標註失敗: {e}")


if __name__ == "__main__":
    main()

