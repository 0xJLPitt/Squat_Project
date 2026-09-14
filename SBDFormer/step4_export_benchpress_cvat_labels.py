#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
Step 4: 臥推照片檔與 2D 骨架數據整理 & CVAT 標註包匯出工具
(Export Bench Press Keyframes with YOLO 2D Skeleton to CVAT COCO Keypoints)
================================================================================
功能說明:
  1. 自動讀取 dataprocess/ 下的切片標註檔 (*_segments.json) 與 YOLO 骨架檔 (*.txt)。
  2. 提取每部動作影片（normal, error1~error4）所有 Rep (1~n) 的 4 大關鍵影格:
     - top_start    [頂點/出槓]
     - descent_mid  [下降 50% ROM]
     - bottom       [最低點/觸胸]
     - ascent_mid   [上升 50% ROM]
  3. 整合對應影格的 12 個身體關節點座標 (x, y，已移除臉部眼睛耳朵 5 點)，並自動計算人物 Bounding Box 與可見度。
  4. 自動將抽出的照片與標籤打包成標準 COCO Keypoints 1.0 格式 (CVAT 原生直接支援)。
  5. 產出:
     - cvat_export/{cam_name}/images/             : 匯集該視角所有關鍵影格照片
     - cvat_export/{cam_name}/images_{cam_name}.zip : 純圖片壓縮包 (供 CVAT 建立任務時拖入)
     - cvat_export/{cam_name}/person_keypoints_default.json : 唯一標準 COCO Keypoints 1.0 標註檔 (供 Upload Annotations 拖入)
     - cvat_export/{cam_name}/README_CVAT_UPLOAD.md: CVAT 快速上傳指引說明檔
  6. 支援單一視角 (預設 sub2-i17) 或批次處理所有相機視角。

使用範例:
  1. 預設執行 (針對 sub2-i17 進行完整匯出與打包):
     $ mamba run -n hw1 python step4_export_benchpress_cvat_labels.py

  2. 指定母資料夾路徑:
     $ mamba run -n hw1 python step4_export_benchpress_cvat_labels.py --dir "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2"

  3. 指定相機視角 (例如 sub2-i17):
     $ mamba run -n hw1 python step4_export_benchpress_cvat_labels.py --cam sub2-i17

  4. 僅處理單一影片 (例如 error1):
     $ mamba run -n hw1 python step4_export_benchpress_cvat_labels.py --video error1
================================================================================
"""

import os
import sys
import json
import zipfile
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2

# 設定 Windows 控制台 UTF-8 編碼
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 預設路徑配置
DEFAULT_PARENT_DIR = Path(r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2")
DEFAULT_DATAPROCESS_DIR = Path(r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\dataprocess")
DEFAULT_CAM = "sub2-i17"

# 12 身體關鍵點定義 (移除頭部 5 點: nose, left_eye, right_eye, left_ear, right_ear)
BODY_12_KEYPOINT_NAMES = [
    "left_shoulder",   # 0
    "right_shoulder",  # 1
    "left_elbow",      # 2
    "right_elbow",     # 3
    "left_wrist",      # 4
    "right_wrist",     # 5
    "left_hip",        # 6
    "right_hip",       # 7
    "left_knee",       # 8
    "right_knee",      # 9
    "left_ankle",      # 10
    "right_ankle"      # 11
]

# 12 身體骨骼連線 (1-based index)
BODY_12_SKELETON_EDGES = [
    [1, 2],    # left_shoulder - right_shoulder
    [1, 3],    # left_shoulder - left_elbow
    [3, 5],    # left_elbow - left_wrist
    [2, 4],    # right_shoulder - right_elbow
    [4, 6],    # right_elbow - right_wrist
    [1, 7],    # left_shoulder - left_hip
    [2, 8],    # right_shoulder - right_hip
    [7, 8],    # left_hip - right_hip
    [7, 9],    # left_hip - left_knee
    [9, 11],   # left_knee - left_ankle
    [8, 10],   # right_hip - right_knee
    [10, 12],  # right_knee - right_ankle
]

# 預設採用 12 身體關鍵點規格
COCO_KEYPOINT_NAMES = BODY_12_KEYPOINT_NAMES
COCO_SKELETON_EDGES = BODY_12_SKELETON_EDGES

# 動作 4 大關鍵影格對應鍵值
KEYFRAME_PHASES = [
    ("top_start", "top_start_frame"),
    ("descent_mid", "descent_mid_frame"),
    ("bottom", "bottom_frame"),
    ("ascent_mid", "ascent_mid_frame")
]


def resolve_dataprocess_dir(base_dir: Path, custom_dp: Optional[Path] = None) -> Path:
    """智慧解析 dataprocess 目錄路徑"""
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


def load_skeleton_txt(txt_path: Path) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """
    載入 YOLO 骨架文字檔，格式: frame_idx, joint_idx, x, y
    回傳字典: {frame_idx: {joint_idx: (x, y)}}
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"找不到骨架檔案: {txt_path}")

    skeleton_data: Dict[int, Dict[int, Tuple[float, float]]] = {}

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split(",")
            if len(parts) < 4:
                continue
            try:
                frame_idx = int(parts[0])
                joint_idx = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
            except ValueError:
                continue

            if frame_idx not in skeleton_data:
                skeleton_data[frame_idx] = {}
            skeleton_data[frame_idx][joint_idx] = (x, y)

    return skeleton_data


def load_segments_json(json_path: Path) -> List[Dict[str, Any]]:
    """
    載入切片 JSON 檔案，取得各 Rep 的關鍵影格編號與分期名稱
    """
    if not json_path.exists():
        raise FileNotFoundError(f"找不到切片標註檔: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reps = data.get("reps", [])
    keyframes_list = []

    for r in reps:
        rep_idx = r.get("rep")
        for phase_name, frame_key in KEYFRAME_PHASES:
            f_val = r.get(frame_key)
            if f_val is not None:
                keyframes_list.append({
                    "rep": rep_idx,
                    "phase": phase_name,
                    "frame_idx": int(f_val)
                })

    return keyframes_list


def find_keyframe_image(
    cam_dir: Path,
    video_stem: str,
    frame_idx: int,
    folder_suffix: str = "_keyframes"
) -> Optional[Path]:
    """
    在指定相機目錄中搜尋該關鍵影格的圖片檔案。
    例如: sub2-i17/error1_keyframes/error1_frame298.jpg
    """
    # 候選資料夾
    folder_cands = [
        cam_dir / f"{video_stem}{folder_suffix}",
        cam_dir / video_stem,
        cam_dir
    ]

    # 候選檔名 (支援 frame298.jpg 與補零如 frame00298.jpg)
    name_cands = [
        f"{video_stem}_frame{frame_idx}.jpg",
        f"{video_stem}_frame{frame_idx:05d}.jpg",
        f"frame{frame_idx}.jpg",
        f"frame{frame_idx:05d}.jpg"
    ]

    for f_dir in folder_cands:
        if not f_dir.exists():
            continue
        for n_cand in name_cands:
            img_p = f_dir / n_cand
            if img_p.exists():
                return img_p

    return None


def calculate_bbox_from_keypoints(
    joints_dict: Dict[int, Tuple[float, float]],
    img_w: int,
    img_h: int,
    padding: float = 20.0
) -> Tuple[List[float], float]:
    """
    從 17 個關節點計算人體邊界框 [x, y, w, h] 與面積 area
    """
    valid_pts = [pt for pt in joints_dict.values() if pt[0] > 0 or pt[1] > 0]
    if not valid_pts:
        return [0.0, 0.0, float(img_w), float(img_h)], float(img_w * img_h)

    xs = [pt[0] for pt in valid_pts]
    ys = [pt[1] for pt in valid_pts]

    min_x = max(0.0, min(xs) - padding)
    min_y = max(0.0, min(ys) - padding)
    max_x = min(float(img_w), max(xs) + padding)
    max_y = min(float(img_h), max(ys) + padding)

    bbox_w = max(1.0, max_x - min_x)
    bbox_h = max(1.0, max_y - min_y)
    area = bbox_w * bbox_h

    return [round(min_x, 1), round(min_y, 1), round(bbox_w, 1), round(bbox_h, 1)], round(area, 1)


def build_coco_keypoints_dataset(
    base_dir: Path,
    dataprocess_dir: Path,
    cam_name: str = DEFAULT_CAM,
    target_video: Optional[str] = None,
    output_dir: Optional[Path] = None,
    create_zip: bool = True
) -> Dict[str, Any]:
    """
    融合指定相機的關鍵影格照片與 YOLO 骨架點，產出單一標準 COCO Keypoints 1.0 JSON 標註檔。
    """
    cam_dir = base_dir / cam_name
    if not cam_dir.exists():
        raise FileNotFoundError(f"找不到相機視角目錄: {cam_dir}")

    if output_dir is None:
        output_dir = base_dir / "cvat_export" / cam_name

    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("🏷️ 臥推關鍵影格與 YOLO 2D 骨架數據融合 ➔ CVAT COCO 格式匯出作業啟動")
    print(f" 📂 母目錄路徑  : {base_dir}")
    print(f" 📑 切片資料目錄: {dataprocess_dir}")
    print(f" 📷 目標相機視角: {cam_name}")
    print(f" 📦 輸出目標路徑: {output_dir}")
    print("=" * 75)

    # 1. 搜尋所有可處理的切片 JSON (排除 checkboard)
    json_files = sorted(dataprocess_dir.glob("yolo_skeleton_*_segments.json"))
    if not json_files:
        json_files = sorted(dataprocess_dir.glob("*_segments.json"))

    valid_json_files = [
        f for f in json_files
        if "checkboard" not in f.stem.lower()
    ]

    if target_video:
        t_stem = Path(target_video).stem.lower().replace("yolo_skeleton_", "").replace("_segments", "")
        valid_json_files = [
            f for f in valid_json_files
            if t_stem in f.stem.lower()
        ]

    if not valid_json_files:
        print(f"[ERROR] 在 {dataprocess_dir} 中未找到符合條件的切片標註檔！")
        return {}

    # COCO 資料集容器
    coco_dataset: Dict[str, Any] = {
        "info": {
            "description": f"Bench Press Keyframes Dataset ({cam_name})",
            "version": "1.0",
            "year": 2026,
            "contributor": "Squat_Project SBDFormer",
            "date_created": "2026-09-10"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Proprietary",
                "url": ""
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": COCO_KEYPOINT_NAMES,
                "skeleton": COCO_SKELETON_EDGES
            }
        ],
        "images": [],
        "annotations": []
    }

    # 供演算法訓練用的高階 Metadata 結構
    meta_summary: Dict[str, Any] = {
        "subject": base_dir.name,
        "camera": cam_name,
        "total_images": 0,
        "videos": {}
    }

    image_id_counter = 1
    annotation_id_counter = 1
    copied_images_count = 0

    # 預設影像尺寸 (若無法讀取影像時使用)
    cached_img_w = 1280
    cached_img_h = 720

    for j_idx, json_file in enumerate(valid_json_files, 1):
        # 推導影片主檔名 (例如 error1)
        v_stem = json_file.stem
        for prefix in ["yolo_skeleton_", "skeleton_"]:
            if v_stem.startswith(prefix):
                v_stem = v_stem[len(prefix):]
        if v_stem.endswith("_segments"):
            v_stem = v_stem[:-len("_segments")]

        print(f"\n▶ [{j_idx}/{len(valid_json_files)}] 處理動作標註: 【{v_stem}】")

        # 尋找對應的 YOLO 骨架 TXT 檔
        txt_candidates = [
            dataprocess_dir / f"yolo_skeleton_{v_stem}.txt",
            dataprocess_dir / f"{v_stem}.txt",
            base_dir / "dataprocess" / f"yolo_skeleton_{v_stem}.txt"
        ]
        txt_path = next((p for p in txt_candidates if p.exists()), None)
        if not txt_path:
            print(f"   ⚠️ 未找到骨架檔案 yolo_skeleton_{v_stem}.txt，跳過此動作。")
            continue

        # 載入切片關鍵影格與骨架點
        keyframes_list = load_segments_json(json_file)
        skeleton_dict = load_skeleton_txt(txt_path)
        print(f"   📑 切片關鍵影格數: {len(keyframes_list)} 個 (來源: {json_file.name})")
        print(f"   🦴 骨架數據幀數  : {len(skeleton_dict)} 幀 (來源: {txt_path.name})")

        meta_summary["videos"][v_stem] = {
            "keyframes_count": len(keyframes_list),
            "reps": {}
        }

        # 逐一處理每個關鍵影格
        matched_for_video = 0
        for item in keyframes_list:
            rep_idx = item["rep"]
            phase_name = item["phase"]
            f_idx = item["frame_idx"]

            # 1. 尋找對應的抽幀圖片
            img_file = find_keyframe_image(cam_dir, v_stem, f_idx)
            if not img_file:
                print(f"   ⚠️ 找不到照片: {v_stem}_frame{f_idx}.jpg，略過。")
                continue

            # 2. 獲取圖片尺寸 (僅需獲取一次或檢查每張)
            dest_img_path = images_out_dir / img_file.name
            if not dest_img_path.exists():
                shutil.copyfile(img_file, dest_img_path)
                copied_images_count += 1

            img_w, img_h = cached_img_w, cached_img_h
            if img_file.exists():
                try:
                    # 使用 cv2 快速讀取影像形狀
                    im = cv2.imread(str(img_file))
                    if im is not None:
                        img_h, img_w = im.shape[:2]
                        cached_img_w, cached_img_h = img_w, img_h
                except Exception:
                    pass

            # 3. 取得 12 個身體關節點數據
            frame_joints = skeleton_dict.get(f_idx, {})

            # 組合 COCO keypoints: [x0, y0, v0, x1, y1, v1, ...]
            # v=0: 未標記, v=1: 被遮擋, v=2: 可見/已標記
            coco_kpts = []
            named_joints = {}
            labeled_count = 0

            for j_i in range(len(COCO_KEYPOINT_NAMES)):
                if j_i in frame_joints:
                    jx, jy = frame_joints[j_i]
                    if jx > 0 or jy > 0:
                        v = 2
                        labeled_count += 1
                    else:
                        v = 0
                    coco_kpts.extend([round(jx, 1), round(jy, 1), v])
                    named_joints[COCO_KEYPOINT_NAMES[j_i]] = [round(jx, 1), round(jy, 1), v]
                else:
                    coco_kpts.extend([0.0, 0.0, 0])
                    named_joints[COCO_KEYPOINT_NAMES[j_i]] = [0.0, 0.0, 0]

            # 4. 計算 BBox
            bbox, area = calculate_bbox_from_keypoints(frame_joints, img_w, img_h)

            # 5. 加入嚴格標準 COCO 結構 (不含自訂非標準欄位，避免 Datumaro 報錯)
            image_record = {
                "id": image_id_counter,
                "file_name": img_file.name,
                "width": img_w,
                "height": img_h
            }
            coco_dataset["images"].append(image_record)

            ann_record = {
                "id": annotation_id_counter,
                "image_id": image_id_counter,
                "category_id": 1,
                "segmentation": [],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "keypoints": coco_kpts,
                "num_keypoints": labeled_count
            }
            coco_dataset["annotations"].append(ann_record)

            # 6. 加入高階 Meta
            if rep_idx not in meta_summary["videos"][v_stem]["reps"]:
                meta_summary["videos"][v_stem]["reps"][rep_idx] = {}

            meta_summary["videos"][v_stem]["reps"][rep_idx][phase_name] = {
                "frame_idx": f_idx,
                "image_file": img_file.name,
                "bbox": bbox,
                "num_keypoints": labeled_count,
                "keypoints": named_joints
            }

            image_id_counter += 1
            annotation_id_counter += 1
            matched_for_video += 1

        print(f"   ✅ 完成匹配與融合: {matched_for_video} 張照片 & 骨架點標籤")

    total_images = len(coco_dataset["images"])
    total_annotations = len(coco_dataset["annotations"])
    meta_summary["total_images"] = total_images

    # 輸出 1: 唯一標準 COCO Keypoints JSON 標註檔 (CVAT 原生完全支援直接上傳此檔)
    person_kpts_file = output_dir / "person_keypoints_default.json"
    with open(person_kpts_file, "w", encoding="utf-8") as f:
        json.dump(coco_dataset, f, indent=2, ensure_ascii=False)

    # 輸出 1b: CVAT Raw 標籤定義 (適用無法在 UI 直接建立 COCO 骨架的 CVAT 網站)
    try:
        from make_cvat_skeleton_labels import build_raw_labels
        raw_labels_file = output_dir / "cvat_raw_labels.json"
        with open(raw_labels_file, "w", encoding="utf-8") as f:
            json.dump(build_raw_labels(), f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"   ⚠️ 產生 cvat_raw_labels.json 略過: {e}")

    print("\n" + "=" * 75)
    print("🎉 資料融合與標註檔產出完畢！")
    print(f" 🖼️ 匯總照片數量: {total_images} 張 (已同步集中至 {images_out_dir.name}/)")
    print(f" 🏷️ 標註物件數量: {total_annotations} 個 (12 點骨架 & BBox)")
    print(f" 📄 唯一 COCO 標註檔案: {person_kpts_file.name}")
    print(f" 🦴 CVAT Raw 標籤檔案: cvat_raw_labels.json (適用 Raw 分頁貼上)")
    print("=" * 75)

    # 輸出 2: 純圖片壓縮包 (供建立任務時在 Select files 一鍵拖入)
    if create_zip and total_images > 0:
        img_zip_path = output_dir / f"images_{cam_name}.zip"
        print(f"\n📦 建立純圖片壓縮包: {img_zip_path.name} (供建立任務上傳) ...", end="", flush=True)
        with zipfile.ZipFile(img_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for img_p in sorted(images_out_dir.glob("*.jpg")):
                zf.write(img_p, arcname=img_p.name)
        print(f" [完成] ({img_zip_path.stat().st_size / (1024*1024):.2f} MB)")
        print(f" 🚀 檔案準備就緒！")

    return coco_dataset


def main():
    parser = argparse.ArgumentParser(
        description="將臥推關鍵影格與 YOLO 2D 骨架數據融合，產出 CVAT COCO Keypoints 1.0 標註包"
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
        help=f"存放切片 JSON 與骨架 TXT 的資料夾路徑 (預設: {DEFAULT_DATAPROCESS_DIR})"
    )
    parser.add_argument(
        "--cam", "-c",
        type=str,
        default=DEFAULT_CAM,
        help=f"指定相機視角資料夾名稱 (預設: {DEFAULT_CAM})"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="指定單一影片檔名 (例如 error1)，若不指定則處理全部"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="自訂輸出路徑 (預設為 {dir}/cvat_export/{cam})"
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        default=False,
        help="不建立 ZIP 封裝壓縮檔"
    )

    args = parser.parse_args()

    base_dir = Path(args.dir)
    custom_dp = Path(args.dataprocess_dir) if args.dataprocess_dir and args.dataprocess_dir != str(DEFAULT_DATAPROCESS_DIR) else None
    dp_dir = resolve_dataprocess_dir(base_dir, custom_dp=custom_dp)
    out_dir = Path(args.output_dir) if args.output_dir else None

    build_coco_keypoints_dataset(
        base_dir=base_dir,
        dataprocess_dir=dp_dir,
        cam_name=args.cam,
        target_video=args.video,
        output_dir=out_dir,
        create_zip=not args.no_zip
    )


if __name__ == "__main__":
    main()
