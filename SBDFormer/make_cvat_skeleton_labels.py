#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
CVAT COCO-17 Skeleton 標籤 (Raw Labels) 產生工具
(Generate a valid CVAT skeleton label definition for COCO 17 keypoints)
================================================================================
背景 / 為什麼需要這支程式:

  A) ⚠️ **label 的 svg 欄位絕對不能自帶 `<svg>` 外層** —— 這是最容易踩、
     而且錯誤訊息完全看不出來的一個坑。
     cvat-core 的 Label constructor 會自己補上外層:

         t.svg = Label.parseUntrustedSvg(`<svg>${t.svg ?? ""}</svg>`)

     若存進去的值本身已經是 `<svg>...</svg>`，就會變成 `<svg><svg>...</svg></svg>`，
     於是 structure.svg 的 children 只有「一個內層 <svg>」而不是 17 個 <circle>。
     前端 canvasView.ts 的 addSkeleton():

         const templateElements = Array.from(SVGElement.children())
             .filter((el) => el.type === 'circle');          // ← 變成空陣列
         const templateElement = templateElements.find(...); // ← undefined
         visibleNodeIDs.add(templateElement.attr('data-node-id'));

     就會噴 TypeError: Cannot read properties of undefined (reading 'attr')。
     反證: Label.toJSON() 匯出時用的是 `t.svg.innerHTML`，
     所以 CVAT 原生存的值本來就是「不含外層」的。

  B) <circle> 一律使用 data-label-name，**絕對不要自己寫死 data-label-id**。
     CVAT 伺服器端 cvat/apps/engine/serializers.py create_labels() 會先把
     送上來的 label id 丟掉 (`if label.get("id"): del label["id"]`)，
     再自行把 svg 裡的 data-label-name 換成真正的 sublabel id:

         svg = svg.replace(f'data-label-name="{db_sublabel.name}"',
                           f'data-label-id="{db_sublabel.id}"')

     若自己寫死 id，建立任務後 svg 內的 id 會對不上實際 sublabel id，
     同樣噴 reading 'attr'。

  C) <line> 的 data-node-from / data-node-to 必須是「整數 node id」，
     不能寫關節名稱；<circle> 必須有 data-node-id / data-element-id。
     cvat-canvas/src/typescript/shared.ts setupSkeletonEdges() 有硬性檢查:

         if (!Number.isInteger(dataNodeFrom) || !Number.isInteger(dataNodeTo)) {
             throw new Error("Edge nodeFrom and nodeTo must be numbers, ...");
         }

     這個例外會讓 svgShapes 沒被賦值，後續 activateShape() 噴
     TypeError: Cannot read properties of undefined (reading 'addClass')。

  D) 只有 DOMPurify 白名單內的屬性會被保留 (Label.parseUntrustedSvg):
     ALLOWED_TAGS = svg, line, circle, desc
     ALLOWED_ATTR = cx, cy, r, x1, y1, x2, y2, data-type, data-element-id,
                    data-label-name, data-label-id, data-node-id,
                    data-node-from, data-node-to, data-description-type
     stroke / fill / stroke-width 會被直接濾掉，寫了也沒用。

  E) ⚠️ 既有任務「無法」改 skeleton 結構。LabelSerializer 的
     read_only_fields 含有 "svg"，且 update_label() 只在「新建 label」時
     才寫入 Skeleton.svg。也就是說把修好的 JSON 貼回既有任務的 Raw 分頁，
     伺服器會靜默忽略。SVG 壞掉的任務只能**刪掉重建**。

產出:
  1. cvat_raw_labels.json     : 建立新 Task / Project 時貼進 Labels -> Raw 分頁
  2. coco_person_skeleton.svg : 可直接用瀏覽器預覽的骨架示意圖

使用範例:
  1. 產生定義檔 (輸出到預設 sub2-i17 資料夾):
     $ mamba run -n hw1 python make_cvat_skeleton_labels.py

  2. 指定輸出資料夾:
     $ mamba run -n hw1 python make_cvat_skeleton_labels.py --out "D:\Pitt\...\cvat_export\sub2-i17"

  3. 檢查手上的 raw labels JSON 是否合規 (例如從 CVAT Raw 分頁複製下來的):
     $ mamba run -n hw1 python make_cvat_skeleton_labels.py --check cvat_raw_labels.json
================================================================================
"""

import re
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

# 設定 Windows 控制台 UTF-8 編碼
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_OUT_DIR = Path(
    r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\cvat_export\sub2-i17"
)

SKELETON_LABEL_NAME = "person"
SKELETON_LABEL_COLOR = "#ff0000"
SUBLABEL_COLOR = "#00ff00"

# COCO 17 關節點定義 (標準順序) 與 skeleton 模板上的示意座標 (viewBox 0 0 100 100)
COCO_KEYPOINTS: List[Dict[str, Any]] = [
    {"name": "nose",           "cx": 50, "cy": 15},
    {"name": "left_eye",       "cx": 46, "cy": 12},
    {"name": "right_eye",      "cx": 54, "cy": 12},
    {"name": "left_ear",       "cx": 42, "cy": 14},
    {"name": "right_ear",      "cx": 58, "cy": 14},
    {"name": "left_shoulder",  "cx": 38, "cy": 28},
    {"name": "right_shoulder", "cx": 62, "cy": 28},
    {"name": "left_elbow",     "cx": 28, "cy": 42},
    {"name": "right_elbow",    "cx": 72, "cy": 42},
    {"name": "left_wrist",     "cx": 20, "cy": 56},
    {"name": "right_wrist",    "cx": 80, "cy": 56},
    {"name": "left_hip",       "cx": 42, "cy": 55},
    {"name": "right_hip",      "cx": 58, "cy": 55},
    {"name": "left_knee",      "cx": 40, "cy": 75},
    {"name": "right_knee",     "cx": 60, "cy": 75},
    {"name": "left_ankle",     "cx": 40, "cy": 95},
    {"name": "right_ankle",    "cx": 60, "cy": 95},
]

# COCO 標準骨骼連線 (1-based index，與 person_keypoints_default.json 的 skeleton 欄位一致)
COCO_SKELETON_EDGES = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [7, 9], [9, 11],
    [6, 8], [8, 10], [1, 2], [1, 3], [2, 4],
    [3, 5], [4, 6], [5, 7],
]

NODE_RADIUS = 1.5
NODE_STROKE_WIDTH = 0.1
EDGE_STROKE_WIDTH = 0.5

# cvat-core Label.parseUntrustedSvg() 的 DOMPurify ALLOWED_ATTR
# (白名單外的屬性會被靜默移除，例如 stroke / fill / stroke-width)
CVAT_ALLOWED_SVG_ATTRS = {
    "cx", "cy", "r", "x1", "y1", "x2", "y2",
    "data-type", "data-element-id", "data-label-name", "data-label-id",
    "data-node-id", "data-node-from", "data-node-to", "data-description-type",
}


def build_label_svg() -> str:
    """組出要存進 CVAT label 的 svg 欄位值。

    ⚠️ **刻意不包外層 `<svg>`**：cvat-core 會自己補上
       (`parseUntrustedSvg(`<svg>${svg}</svg>`)`)，自帶外層會變成雙層巢狀，
       導致 addSkeleton() 找不到 <circle> 而噴 reading 'attr'。詳見檔頭說明 (A)。

    其餘重點:
      - <circle> 一定要有整數的 data-node-id 與 data-element-id
      - <circle> 用 data-label-name 綁定 sublabel，id 交給 CVAT 伺服器自己填
      - <line> 的 data-node-from / data-node-to 一定要引用整數 node id，不能寫關節名稱
      - 只寫 DOMPurify 白名單內的屬性 (stroke / fill / stroke-width 會被濾掉)
    """
    parts: List[str] = []

    # 先寫線 (edge)，讓節點 (node) 疊在線的上面
    for src, dst in COCO_SKELETON_EDGES:
        a = COCO_KEYPOINTS[src - 1]
        b = COCO_KEYPOINTS[dst - 1]
        parts.append(
            f'<line x1="{a["cx"]}" y1="{a["cy"]}" x2="{b["cx"]}" y2="{b["cy"]}"'
            f' data-type="edge" data-node-from="{src}" data-node-to="{dst}"></line>'
        )

    # 再寫節點 (node)；node_id / element_id 皆使用 1-based 序號
    for idx, kp in enumerate(COCO_KEYPOINTS, start=1):
        parts.append(
            f'<circle r="{NODE_RADIUS}" cx="{kp["cx"]}" cy="{kp["cy"]}"'
            f' data-type="element node" data-element-id="{idx}" data-node-id="{idx}"'
            f' data-label-name="{kp["name"]}"></circle>'
        )

    return "".join(parts)


def build_preview_svg() -> str:
    """組出可直接用瀏覽器開啟的獨立 svg 檔 (僅供肉眼確認骨架長相)。

    這個版本**才需要**外層 `<svg>` 與 stroke/fill 等樣式屬性，
    但它不是給 CVAT 用的 —— 不要把這份內容貼到 label 的 svg 欄位。
    """
    parts: List[str] = []
    for src, dst in COCO_SKELETON_EDGES:
        a = COCO_KEYPOINTS[src - 1]
        b = COCO_KEYPOINTS[dst - 1]
        parts.append(
            f'<line x1="{a["cx"]}" y1="{a["cy"]}" x2="{b["cx"]}" y2="{b["cy"]}"'
            f' stroke="black" stroke-width="{EDGE_STROKE_WIDTH}"'
            f' data-type="edge" data-node-from="{src}" data-node-to="{dst}"></line>'
        )
    for idx, kp in enumerate(COCO_KEYPOINTS, start=1):
        parts.append(
            f'<circle r="{NODE_RADIUS}" stroke="black" fill="#b3b3b3"'
            f' cx="{kp["cx"]}" cy="{kp["cy"]}" stroke-width="{NODE_STROKE_WIDTH}"'
            f' data-type="element node" data-element-id="{idx}" data-node-id="{idx}"'
            f' data-label-name="{kp["name"]}"></circle>'
        )
    body = "\n  " + "\n  ".join(parts) + "\n"
    return f'<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">{body}</svg>'


def build_raw_labels() -> List[Dict[str, Any]]:
    """建立新 Task / Project 用的 raw labels。

    不含任何 id：CVAT 建立標籤時會丟掉送上來的 id 並自行編號。
    """
    return [
        {
            "name": SKELETON_LABEL_NAME,
            "color": SKELETON_LABEL_COLOR,
            "type": "skeleton",
            "attributes": [],
            "sublabels": [
                {
                    "name": kp["name"],
                    "color": SUBLABEL_COLOR,
                    "type": "points",
                    "attributes": [],
                }
                for kp in COCO_KEYPOINTS
            ],
            "svg": build_label_svg(),
        }
    ]


def validate_svg(svg: str, sublabel_names: List[str], expect_names: bool = True) -> None:
    """依 CVAT 前後端規則自我檢查，避免再次踩到 activateShape / addSkeleton 的雷。

    expect_names=True  : 檢查「要上傳給 CVAT」的定義 (circle 必須用 data-label-name)
    expect_names=False : 檢查「從 CVAT 抓下來」的定義 (circle 會是 data-label-id)
    """
    # 檢查 0 (最重要): label 的 svg 欄位不能自帶 <svg> 外層
    if re.search(r"<svg\b", svg):
        raise AssertionError(
            "svg 欄位不可包含 <svg> 外層標籤。\n"
            "      cvat-core 會自己補: parseUntrustedSvg(`<svg>${svg}</svg>`)，\n"
            "      自帶外層會變成 <svg><svg>...</svg></svg>，addSkeleton() 過濾 circle 得到空陣列，\n"
            "      最終噴 TypeError: Cannot read properties of undefined (reading 'attr')。\n"
            "      正確做法: svg 欄位只放 <line> 與 <circle>，不要外層。"
        )

    circles = re.findall(r"<circle\b[^>]*>", svg)
    lines = re.findall(r"<line\b[^>]*>", svg)

    def attr(tag: str, key: str) -> Optional[str]:
        m = re.search(rf'{key}="([^"]*)"', tag)
        return m.group(1) if m else None

    # 檢查 0b: 只有 DOMPurify 白名單內的屬性會被保留，寫了別的只是誤導
    for tag in circles + lines:
        stripped = [
            k for k in re.findall(r"([\w-]+)=\"", tag) if k not in CVAT_ALLOWED_SVG_ATTRS
        ]
        if stripped:
            raise AssertionError(
                f"這些屬性不在 CVAT 的 DOMPurify 白名單內，會被靜默移除: {stripped}\n"
                f"      出現於: {tag}"
            )

    if len(circles) != len(sublabel_names):
        raise AssertionError(
            f"circle 數量 {len(circles)} 與 sublabel 數量 {len(sublabel_names)} 不符"
        )

    node_ids = set()
    for tag in circles:
        node_id = attr(tag, "data-node-id")
        element_id = attr(tag, "data-element-id")
        if not (node_id and node_id.isdigit()):
            raise AssertionError(f"circle 缺少整數 data-node-id: {tag}")
        if not (element_id and element_id.isdigit()):
            raise AssertionError(f"circle 缺少整數 data-element-id: {tag}")
        if attr(tag, "data-type") != "element node":
            raise AssertionError(f"circle 的 data-type 必須是 'element node': {tag}")
        if expect_names:
            name = attr(tag, "data-label-name")
            if not name:
                # 寫死 data-label-id 會在建立任務後對不上實際 sublabel id
                raise AssertionError(
                    f"circle 必須用 data-label-name 綁定 (不可寫死 data-label-id): {tag}"
                )
            if name not in sublabel_names:
                raise AssertionError(f"circle 的 data-label-name '{name}' 不在 sublabel 清單中")
        elif not attr(tag, "data-label-id"):
            raise AssertionError(f"circle 缺少 data-label-id: {tag}")
        node_ids.add(int(node_id))

    for tag in lines:
        node_from = attr(tag, "data-node-from")
        node_to = attr(tag, "data-node-to")
        if attr(tag, "data-type") != "edge":
            raise AssertionError(f"line 的 data-type 必須是 'edge': {tag}")
        # CVAT setupSkeletonEdges() 會做 Number.isInteger() 檢查，寫關節名稱一定爆
        if not (node_from and node_from.isdigit()):
            raise AssertionError(
                f"line 的 data-node-from 必須是整數 node id (不能是關節名稱): {tag}"
            )
        if not (node_to and node_to.isdigit()):
            raise AssertionError(
                f"line 的 data-node-to 必須是整數 node id (不能是關節名稱): {tag}"
            )
        # getSkeletonEdgeCoordinates() 需要在同一層找到對應的 node
        if int(node_from) not in node_ids:
            raise AssertionError(f"line 參照到不存在的 node {node_from}: {tag}")
        if int(node_to) not in node_ids:
            raise AssertionError(f"line 參照到不存在的 node {node_to}: {tag}")

    print(f"   ✅ SVG 自我檢查通過 ({len(circles)} 個節點 / {len(lines)} 條骨骼連線)")


def check_file(path: Path) -> int:
    """檢查一份既有的 raw labels JSON 是否合規。"""
    labels = json.loads(path.read_text(encoding="utf-8"))
    skeletons = [lb for lb in labels if lb.get("type") == "skeleton"]
    if not skeletons:
        print(f"❌ {path.name} 中找不到 type=skeleton 的標籤")
        return 1

    rc = 0
    for label in skeletons:
        names = [sub["name"] for sub in label.get("sublabels", [])]
        svg = label.get("svg", "")
        # 從 CVAT 抓下來的定義 circle 會是 data-label-id，上傳用的則是 data-label-name
        from_server = "data-label-id=" in svg
        print(f"🔍 檢查標籤 '{label.get('name')}' ({'來自 CVAT' if from_server else '待上傳'})")
        try:
            validate_svg(svg, names, expect_names=not from_server)
        except AssertionError as exc:
            print(f"   ❌ {exc}")
            rc = 1

        if from_server:
            ids = {sub["id"] for sub in label.get("sublabels", []) if "id" in sub}
            svg_ids = {int(m) for m in re.findall(r'data-label-id="(\d+)"', svg)}
            orphan = svg_ids - ids
            if orphan:
                print(
                    f"   ❌ svg 內的 data-label-id {sorted(orphan)} 不存在於 sublabel id 清單\n"
                    "      ➔ 這會造成 addSkeleton() 的 'Cannot read properties of undefined "
                    "(reading \\'attr\\')'\n"
                    "      ➔ 既有任務的 svg 無法修改 (LabelSerializer 的 svg 是唯讀)，"
                    "請刪掉任務重建"
                )
                rc = 1
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="產生 / 檢查 CVAT 合規的 COCO-17 skeleton 標籤定義檔"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="輸出資料夾 (預設為 sub2-i17 的 cvat_export 目錄)",
    )
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        help="只檢查指定的 raw labels JSON 是否合規，不產生檔案",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        default=False,
        help="額外產生供瀏覽器預覽的 coco_person_skeleton.svg 示意圖檔 (預設不產生)",
    )
    args = parser.parse_args()

    if args.check:
        if not args.check.exists():
            print(f"❌ 找不到檔案: {args.check}")
            return 1
        return check_file(args.check)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [kp["name"] for kp in COCO_KEYPOINTS]

    print("=" * 78)
    print("🦴 產生 CVAT COCO-17 Skeleton 標籤定義")
    print("=" * 78)
    print(f"📂 輸出資料夾: {out_dir}")

    raw_labels = build_raw_labels()
    validate_svg(raw_labels[0]["svg"], names)
    raw_path = out_dir / "cvat_raw_labels.json"
    raw_path.write_text(json.dumps(raw_labels, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"   💾 {raw_path.name}  (建立新 Task 時貼進 Labels -> Raw)")

    if args.svg:
        svg_path = out_dir / "coco_person_skeleton.svg"
        svg_path.write_text(build_preview_svg(), encoding="utf-8")
        print(f"   💾 {svg_path.name}  (示意圖，可直接用瀏覽器開啟確認)")

    print("-" * 78)
    print("✅ 完成！")
    print("⚠️  提醒: 既有任務的 skeleton svg 無法修改 (CVAT LabelSerializer 的 svg 為唯讀)，")
    print("    壞掉的任務請刪除後用本定義檔重建。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
