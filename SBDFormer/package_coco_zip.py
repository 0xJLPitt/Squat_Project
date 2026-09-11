#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
COCO Keypoints 標註打包工具 (CVAT Upload Annotations 專用)
================================================================================
重點:
  CVAT 的「COCO Keypoints 1.0」匯入器會掃描 zip 內所有 json，
  同一份 json 若同時放在根目錄與 annotations/ 底下會被讀兩次，
  結果每張影格產生 2 個重疊的 skeleton。因此這裡只放 annotations/ 一份。

使用範例:
  $ mamba run -n hw1 python package_coco_zip.py
  $ mamba run -n hw1 python package_coco_zip.py --dir "D:\Pitt\...\cvat_export\sub2-i17"
================================================================================
"""

import sys
import json
import zipfile
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_DIR = Path(
    r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\cvat_export\sub2-i17"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="打包 COCO Keypoints 標註成 CVAT 可上傳的 zip")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="cvat_export 視角資料夾")
    parser.add_argument(
        "--json",
        type=str,
        default="person_keypoints_default.json",
        help="要打包的 COCO Keypoints json 檔名",
    )
    args = parser.parse_args()

    json_src = args.dir / args.json
    if not json_src.exists():
        print(f"❌ 找不到標註檔: {json_src}")
        return 1

    # 打包前先做基本檢查，避免上傳後才發現格式問題
    data = json.loads(json_src.read_text(encoding="utf-8"))
    n_kp = len(data["categories"][0]["keypoints"])
    bad = [a["id"] for a in data["annotations"] if len(a["keypoints"]) != n_kp * 3]
    if bad:
        print(f"❌ 有 {len(bad)} 筆標註的 keypoints 長度不等於 {n_kp * 3}: {bad[:5]}")
        return 1
    print(f"   🔍 檢查通過: {len(data['images'])} 張影像 / {len(data['annotations'])} 筆標註 / {n_kp} 點")

    zip_out = args.dir / "coco_keypoints_annotations.zip"
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
        # 只放一份！放兩份 (root + annotations/) 會讓 CVAT 重複匯入
        z.write(json_src, f"annotations/{json_src.name}")

    print(f"✅ 已產生 {zip_out.name} ({zip_out.stat().st_size} bytes)")
    print("   ➔ CVAT: Task ➔ Actions ➔ Upload annotations ➔ 格式選 COCO Keypoints 1.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
