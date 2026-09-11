#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
臥推關鍵影格與 YOLO 2D 骨架數據融合 & CVAT 標註匯出工具
(Export Bench Press Keyframes with YOLO 2D Skeleton to CVAT COCO Keypoints)
================================================================================
功能說明:
  1. 自動讀取 dataprocess/ 下的切片標註檔 (*_segments.json) 與 YOLO 骨架檔 (*.txt)。
  2. 提取每部動作影片（normal, error1~error4）所有 Rep (1~n) 的 4 大關鍵影格:
     - top_start    [頂點/出槓]
     - descent_mid  [下降 50% ROM]
     - bottom       [最低點/觸胸]
     - ascent_mid   [上升 50% ROM]
  3. 整合對應影格的 17 個 COCO 關節點座標 (x, y)，並自動計算人物 Bounding Box 與可見度。
  4. 自動將抽出的照片與標籤打包成標準 COCO Keypoints 1.0 格式 (CVAT 原生直接支援)。
  5. 產出:
     - cvat_export/{cam_name}/images/             : 匯集該視角所有關鍵影格照片
     - cvat_export/{cam_name}/instances_default.json : CVAT / COCO 標準標註檔
     - cvat_export/{cam_name}/benchpress_meta.json: 包含分期與動作真值的完整結構化 JSON
     - cvat_export/{cam_name}/cvat_upload.zip     : 一鍵打包 ZIP，可直接上傳 CVAT
  6. 支援單一視角 (預設 sub2-i17) 或批次處理所有相機視角。

使用範例:
  1. 預設執行 (針對 sub2-i17 進行完整匯出與打包):
     $ mamba run -n hw1 python export_benchpress_cvat_labels.py

  2. 指定母資料夾路徑:
     $ mamba run -n hw1 python export_benchpress_cvat_labels.py --dir "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2"

  3. 指定相機視角 (例如 sub2-i17):
     $ mamba run -n hw1 python export_benchpress_cvat_labels.py --cam sub2-i17

  4. 僅處理單一影片 (例如 error1):
     $ mamba run -n hw1 python export_benchpress_cvat_labels.py --video error1
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

# COCO 17 關節點定義 (標準順序)
COCO_KEYPOINT_NAMES = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle"      # 16
]

# COCO 標準骨骼連線 (1-based index)
COCO_SKELETON_EDGES = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [7, 9], [9, 11],
    [6, 8], [8, 10], [1, 2], [1, 3], [2, 4],
    [3, 5], [4, 6], [5, 7]
]

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


def export_cvat_xml_format(
    images: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    cam_name: str,
    output_file: Path
) -> None:
    """
    產出 CVAT 原生相容性最高的 'CVAT for images 1.1' XML 格式檔案。
    可在 CVAT 介面直接選擇 'CVAT for images 1.1' 上傳，完全不依賴 COCO 轉換器。
    """
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "1"
    ET.SubElement(task, "name").text = f"benchpress_{cam_name}"
    ET.SubElement(task, "size").text = str(len(images))
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(max(0, len(images) - 1))
    ET.SubElement(task, "frame_filter").text = ""

    labels_el = ET.SubElement(task, "labels")
    label_el = ET.SubElement(labels_el, "label")
    ET.SubElement(label_el, "name").text = "person"
    ET.SubElement(label_el, "color").text = "#ff0000"
    ET.SubElement(label_el, "type").text = "skeleton"

    sublabels_el = ET.SubElement(label_el, "sublabels")
    for name in COCO_KEYPOINT_NAMES:
        sub = ET.SubElement(sublabels_el, "label")
        ET.SubElement(sub, "name").text = name
        ET.SubElement(sub, "color").text = "#00ff00"
        ET.SubElement(sub, "type").text = "points"

    ann_by_image = {ann["image_id"]: ann for ann in annotations}
    for idx, img in enumerate(images):
        img_el = ET.SubElement(root, "image", {
            "id": str(idx),
            "name": img["file_name"],
            "width": str(img["width"]),
            "height": str(img["height"])
        })
        ann = ann_by_image.get(img["id"])
        if ann:
            skel_el = ET.SubElement(img_el, "skeleton", {
                "label": "person",
                "group_id": "0"
            })
            kpts = ann.get("keypoints", [])
            for j_i in range(17):
                jx = kpts[j_i * 3]
                jy = kpts[j_i * 3 + 1]
                jv = kpts[j_i * 3 + 2]
                outside = "1" if (jx == 0 and jy == 0) or jv == 0 else "0"
                ET.SubElement(skel_el, "points", {
                    "label": COCO_KEYPOINT_NAMES[j_i],
                    "points": f"{jx:.1f},{jy:.1f}",
                    "occluded": "0",
                    "outside": outside
                })

    xml_str = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ", encoding="utf-8")
    with open(output_file, "wb") as f:
        f.write(xml_str)


def build_coco_keypoints_dataset(

    base_dir: Path,
    dataprocess_dir: Path,
    cam_name: str = DEFAULT_CAM,
    target_video: Optional[str] = None,
    output_dir: Optional[Path] = None,
    create_zip: bool = True
) -> Dict[str, Any]:
    """
    融合指定相機的關鍵影格照片與 YOLO 骨架點，產出標準 COCO Keypoints 1.0 資料集。
    """
    cam_dir = base_dir / cam_name
    if not cam_dir.exists():
        raise FileNotFoundError(f"找不到相機視角目錄: {cam_dir}")

    if output_dir is None:
        output_dir = base_dir / "cvat_export" / cam_name

    images_out_dir = output_dir / "images"
    annotations_out_dir = output_dir / "annotations"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    annotations_out_dir.mkdir(parents=True, exist_ok=True)

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

            # 3. 取得 17 個關節點數據
            frame_joints = skeleton_dict.get(f_idx, {})

            # 組合 COCO keypoints: [x0, y0, v0, x1, y1, v1, ...]
            # v=0: 未標記, v=1: 被遮擋, v=2: 可見/已標記
            coco_kpts = []
            named_joints = {}
            labeled_count = 0

            for j_i in range(17):
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

    # 輸出 1: 純淨 COCO Keypoints JSON
    cvat_ann_file = annotations_out_dir / "instances_default.json"
    root_ann_file = output_dir / "instances_default.json"
    person_kpts_file = output_dir / "person_keypoints_default.json"

    with open(cvat_ann_file, "w", encoding="utf-8") as f:
        json.dump(coco_dataset, f, indent=2, ensure_ascii=False)
    shutil.copyfile(cvat_ann_file, root_ann_file)
    shutil.copyfile(cvat_ann_file, person_kpts_file)

    # 輸出 2: 原生 CVAT for images 1.1 XML 格式 (CVAT 原生絕對相容)
    xml_file = output_dir / "annotations.xml"
    export_cvat_xml_format(
        images=coco_dataset["images"],
        annotations=coco_dataset["annotations"],
        cam_name=cam_name,
        output_file=xml_file
    )

    # 輸出 3: 高階 Metadata JSON (供演算法直接使用)
    meta_file = output_dir / "benchpress_keyframes_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_summary, f, indent=2, ensure_ascii=False)

    # 輸出 4: 產生 CVAT 上傳教學說明文件
    readme_file = output_dir / "README_CVAT_UPLOAD.md"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(generate_cvat_readme_guide(cam_name, total_images))

    print("\n" + "=" * 75)
    print("🎉 資料融合與標註檔產出完畢！")
    print(f" 🖼️ 匯總照片數量: {total_images} 張 (已同步集中至 {images_out_dir.name}/)")
    print(f" 🏷️ 標註物件數量: {total_annotations} 個 (17 點骨架 & BBox)")
    print(f" 📄 COCO 標註檔案: {root_ann_file.name} / {person_kpts_file.name}")
    print(f" 📑 CVAT 原生 XML: {xml_file.name} (CVAT for images 1.1)")
    print(f" 📊 全局切片索引: {meta_file.name}")
    print("=" * 75)

    # 輸出 5: 打包 ZIP
    if create_zip and total_images > 0:
        # 5.1 純圖片 ZIP (專供建立 Task 時在 Select files 拖入)
        img_zip_path = output_dir / f"images_{cam_name}.zip"
        print(f"\n📦 [1/3] 建立純圖片壓縮包: {img_zip_path.name} (供建立任務上傳) ...", end="", flush=True)
        with zipfile.ZipFile(img_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for img_p in sorted(images_out_dir.glob("*.jpg")):
                zf.write(img_p, arcname=img_p.name)
        print(f" [完成] ({img_zip_path.stat().st_size / (1024*1024):.2f} MB)")

        # 5.2 原生 CVAT XML ZIP (推薦：CVAT for images 1.1)
        xml_zip_path = output_dir / f"annotations_cvat_xml_{cam_name}.zip"
        print(f"📦 [2/3] 建立 CVAT XML 壓縮包: {xml_zip_path.name} (CVAT for images 1.1) ...", end="", flush=True)
        with zipfile.ZipFile(xml_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(xml_file, arcname="annotations.xml")
        print(f" [完成] ({xml_zip_path.stat().st_size / 1024:.2f} KB)")

        # 5.3 COCO 標註 ZIP (COCO Keypoints 1.0)
        ann_zip_path = output_dir / f"annotations_coco_{cam_name}.zip"
        print(f"📦 [3/3] 建立 COCO 標註壓縮包: {ann_zip_path.name} (COCO Keypoints 1.0) ...", end="", flush=True)
        with zipfile.ZipFile(ann_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # 只放一份 person_keypoints json！CVAT 的 COCO Keypoints 匯入器會掃描 zip 內
            # 所有 json，同時放 instances_default.json 或重複放在根目錄與 annotations/，
            # 會被讀成兩份標註，導致每張影格出現重疊的兩個 skeleton。
            zf.write(person_kpts_file, arcname="annotations/person_keypoints_default.json")
        print(f" [完成] ({ann_zip_path.stat().st_size / 1024:.2f} KB)")

        print(f" 🚀 檔案準備就緒！")



    return coco_dataset


def generate_cvat_readme_guide(cam_name: str, total_images: int) -> str:
    """產生 CVAT 標註操作指南 Markdown"""
    return f"""# CVAT 臥推關鍵影格標註操作指引 ({cam_name})

本資料夾已將臥推 5 部影片（`normal`, `error1~4`）的所有 Rep 關鍵動作影格（共 {total_images} 張）
與 YOLO 2D 骨架點（17 個 COCO 關鍵點）完成自動融合，格式為標準 **COCO Keypoints 1.0**。

---

## 🚀 快速上傳至 CVAT (app.cvat.ai) 步驟:

### 步驟 1: 登入並建立 Task (Create a new task)
1. 開啟瀏覽器進入 [https://app.cvat.ai/tasks/create](https://app.cvat.ai/tasks/create)
2. **Task Name (必填)**: 輸入任務名稱，例如 `BenchPress_{cam_name}_Keyframes`
3. **Labels 設定 (必填)**:
   - 切到 **`Raw`** 分頁，把 `cvat_raw_labels.json` 的整份內容貼上 ➔ `Done`
     （此檔由 `make_cvat_skeleton_labels.py` 產生，17 個 sublabel 名稱與本標註檔完全一致）
   - ⚠️ **skeleton 結構建立後就不能改**（CVAT `LabelSerializer` 的 `svg` 是唯讀欄位），
     設錯只能刪掉任務重建，請務必一次做對
   - ⚠️ **不要手改 skeleton 的 SVG**，尤其是:
     `svg` 欄位**不可自帶 `<svg>` 外層**（cvat-core 會自己補，自帶會變雙層巢狀而整個壞掉）；
     `<line>` 的 `data-node-from` / `data-node-to` 必須是**整數 node id**（不能寫關節名稱）；
     `<circle>` 必須有 `data-node-id`、`data-element-id`，
     且一律用 `data-label-name` 綁定（不可自己寫死 `data-label-id`）
4. **上傳檔案 (Select Files - 僅上傳圖片)**:
   - 直接將本目錄下的純圖片包 **`images_{cam_name}.zip`** 拖入，或將 **`images/`** 內的 200 張照片全選拖入
   - ⚠️ 注意：建立任務時千萬不要傳入包含 json 的壓縮檔，建立任務只能吃純圖片！
5. 點選最下方 **`Submit & Open`** 建立任務！

---

### 步驟 2: 匯入預標註骨架點 (Upload Pre-annotations)
1. 進入剛剛建立成功的 Task 頁面，點選右上角的 **`Actions`** 選單（三個點點）
2. 選擇 **`Upload Annotations`**
3. 格式選擇: **`COCO Keypoints 1.0`**
4. 檔案選擇本目錄下的 **`annotations_coco_{cam_name}.zip`** (或 `person_keypoints_default.json`)
   - ⚠️ 一次只傳**一份** json。不要傳同時含 `instances_default.json` 與
     `person_keypoints_default.json` 的壓縮檔，CVAT 會讀兩次，每張影格會出現 2 個重疊骨架
5. 點擊確認上傳，1~2 秒鐘即載入完畢！


---

### 步驟 3: 標註人員微調 (Fine-tuning)
- 打開任何一張影格，17 個骨骼點已經依照 YOLO 偵測的位置排列並繪製骨骼連接線。
- 標註人員僅需檢查：
  1. **手腕 (left_wrist / right_wrist)**：臥推時槓鈴反光或遮擋手腕，請以滑鼠拖動點位對齊手掌根骨關節。
  2. **手肘 (left_elbow / right_elbow)**：確認手肘彎曲最低點位置。
  3. **肩膀與胸口**：確認兩肩與鎖骨位置正確。
- 微調完成後，按 `Ctrl + S` 儲存！

---

### 步驟 4: 匯出修正後真值
- 標註完成後，在 Task 頁面點選 `Export dataset` ➔ 選擇 **`COCO Keypoints 1.0`** 下載修正後的真值 JSON。

---

## 🛠 疑難排解

### 骨架畫不出來、滑到標註上就報錯

**兩種錯誤都是「Task 的 skeleton 標籤 SVG 壞掉」，與本標註檔無關。**
CVAT 前端 `addSkeleton()` 一丟例外，`svgShapes[clientID]` 就沒被賦值，後續便對 `undefined` 取值。

- `Cannot read properties of undefined (reading 'attr')` (at `addSkeleton`)
  ➔ **最常見**：`svg` 欄位自帶 `<svg>` 外層。cvat-core 會再包一層
  (`parseUntrustedSvg(`<svg>${svg}</svg>`)`)，變成 `<svg><svg>...</svg></svg>`，
  過濾 `circle` 後得到空陣列。`svg` 欄位只能放 `<line>` 與 `<circle>`。
  ➔ 次之：`<circle>` 的 `data-label-id` 對不上實際 sublabel id，通常是自己把 id 寫死。
  CVAT `create_labels()` 會先 `del label["id"]` 再自行把 `data-label-name` 換成真 id，
  所以 SVG 一律要用 `data-label-name`。
- `Cannot read properties of undefined (reading 'addClass')` (at `activateShape`)
  ➔ `addSkeleton()` 先前已丟出例外（上面任一原因），或 `<line>` 的
  `data-node-from` / `data-node-to` 寫成關節名稱 ——
  `setupSkeletonEdges()` 有 `Number.isInteger()` 檢查，必須是整數 node id。

**修正方式：刪掉任務重建。** 既有任務的 SVG 改不了
（`LabelSerializer.Meta.read_only_fields` 含 `"svg"`，`update_label()` 只在新建 label 時寫入
`Skeleton.svg`），貼回 `Raw` 分頁伺服器會靜默忽略。

1. 需要的話先 `Export dataset` 備份已微調的標註
2. Task ➔ `Actions` ➔ `Delete`
3. 依步驟 1 重建任務，Labels 用 `Raw` 分頁貼上 `cvat_raw_labels.json`
4. 依步驟 2 重新上傳標註

檢查標籤定義是否合規：
`python make_cvat_skeleton_labels.py --check <raw labels json 路徑>`
"""


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
