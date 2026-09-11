#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
CVAT Skeleton 標籤 API 工具 (診斷 / 建立)
================================================================================
為什麼要走 API:

  CVAT 的 skeleton 標籤有兩層陷阱，光看 UI 完全看不出問題:

  A) Raw 分頁「顯示」的內容不是伺服器實際存的內容。
     cvat-ui/src/components/labels-editor/raw-viewer.tsx 的 transformSkeletonSVG()
     會把 svg 裡的 data-label-id 反轉成 data-label-name 才顯示給你看:

         const matches = data.matchAll(/data-label-id=\"([\d]+)\"/g);
         ... data.replace(match[0], `data-label-name=\"${idNameMapping[match[1]]}\"`)

     所以就算伺服器上的 id 是壞的，Raw 分頁看起來也「很正常」。
     要看真相只能打 GET /api/labels?task_id=<id>。

  B) 既有任務的 skeleton svg 改不掉。
     cvat/apps/engine/serializers.py 的 LabelSerializer.Meta.read_only_fields
     含有 "svg"，且 update_label() 只在「新建 label」分支才寫入 Skeleton.svg。

  本工具提供:
    diagnose : 直接讀伺服器上的真實 svg + 標註，指出到底哪裡對不上
    create-project : 用 API 建立帶正確 skeleton 標籤的 Project
                     (純 JSON、不含檔案上傳，最不容易出錯)
                     之後在 UI 裡「把任務建在這個 Project 底下」，
                     任務會直接繼承 Project 的標籤，完全繞開 UI 的標籤編輯器

登入方式 (依序嘗試，任選一種):

  1. 瀏覽器 cookie  ← **Google / SSO 登入者請用這個**
     CVAT 的 /api/auth/login 只吃「帳號 + 密碼」，用 Google 登入的帳號沒有密碼可填，
     所以要改用瀏覽器已登入的 session cookie:

       a. 在已登入 app.cvat.ai 的分頁按 F12 開 DevTools
       b. Application (Chrome) / 儲存空間 (Firefox) ➔ Cookies ➔ https://app.cvat.ai
       c. 複製 `sessionid` 與 `csrftoken` 兩個值
          (sessionid 是 HttpOnly，在 Console 打 document.cookie 看不到，要走這個分頁)
       d. set CVAT_SESSIONID=xxx  &  set CVAT_CSRFTOKEN=yyy
          或用參數 --sessionid xxx --csrftoken yyy
     只做 diagnose (GET) 其實只需要 sessionid；create-project (POST) 才需要 csrftoken。

  2. 環境變數 token : set CVAT_TOKEN=xxxxxxxx
  3. 環境變數帳密   : set CVAT_USERNAME=xxx  &  set CVAT_PASSWORD=xxx (SSO 帳號無效)
  4. 完全離線       : 用瀏覽器把 API 回應存成檔案，再用 --labels-file 診斷 (見下方範例 5)

使用範例:
  1. 診斷現在壞掉的任務 (task id 看網址 /tasks/<id>):
     $ mamba run -n hw1 python cvat_skeleton_api.py diagnose --task 1234567

  2. 連標註一起檢查:
     $ mamba run -n hw1 python cvat_skeleton_api.py diagnose --task 1234567 --annotations

  3. 建立帶正確骨架標籤的 Project:
     $ mamba run -n hw1 python cvat_skeleton_api.py create-project --name BenchPress_sub2

  4. 診斷 Project 的標籤:
     $ mamba run -n hw1 python cvat_skeleton_api.py diagnose --project 89012

  5. 離線診斷 (不需要任何認證設定):
     a. 在已登入的瀏覽器直接開這個網址，把整份 JSON 另存成 labels.json:
        https://app.cvat.ai/api/labels?task_id=1234567&page_size=200&format=json
     b. (選用) 標註也存一份 annotations.json:
        https://app.cvat.ai/api/tasks/1234567/annotations?format=json
     c. $ mamba run -n hw1 python cvat_skeleton_api.py diagnose \
            --labels-file labels.json --annotations-file annotations.json
================================================================================
"""

import os
import re
import sys
import json
import argparse
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from make_cvat_skeleton_labels import (
    COCO_KEYPOINTS,
    CVAT_ALLOWED_SVG_ATTRS,
    build_raw_labels,
)

# 設定 Windows 控制台 UTF-8 編碼
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_HOST = "https://app.cvat.ai"
TIMEOUT = 60

# 由 --show-svg 開啟：印出伺服器上 skeleton svg 的完整內容
SHOW_SVG = False


class CvatClient:
    """極簡 CVAT REST API v2 客戶端 (只做本工具需要的幾個呼叫)。"""

    def __init__(self, host: str):
        self.host = host.rstrip("/")
        self.session = requests.Session()
        # CVAT 用 DRF 的 AcceptHeaderVersioning，送 application/json 會被回 406
        # ("Could not satisfy the request Accept header.")
        self.session.headers.update({"Accept": "application/vnd.cvat+json; version=2.0"})

    def _use_token(self, token: str) -> "CvatClient":
        self.session.headers["Authorization"] = f"Token {token}"
        return self

    def _use_cookies(self, sessionid: str, csrftoken: Optional[str]) -> "CvatClient":
        self.session.cookies.set("sessionid", sessionid, domain=self.host.split("//")[-1])
        if csrftoken:
            self.session.cookies.set("csrftoken", csrftoken, domain=self.host.split("//")[-1])
            # Django 的 CSRF 檢查在 HTTPS 下還會比對 Referer
            self.session.headers.update({"X-CSRFToken": csrftoken, "Referer": self.host})
        return self

    @classmethod
    def login(cls, host: str, args: argparse.Namespace) -> "CvatClient":
        client = cls(host)

        sessionid = getattr(args, "sessionid", None) or os.environ.get("CVAT_SESSIONID")
        csrftoken = getattr(args, "csrftoken", None) or os.environ.get("CVAT_CSRFTOKEN")
        if sessionid:
            print("🔑 使用瀏覽器 session cookie 認證")
            if not csrftoken:
                print("   ⚠️  沒有 csrftoken，只能做 GET (diagnose)；create-project 會被拒絕")
            return client._use_cookies(sessionid, csrftoken)

        token = os.environ.get("CVAT_TOKEN")
        if token:
            print("🔑 使用環境變數 CVAT_TOKEN 認證")
            return client._use_token(token)

        username = os.environ.get("CVAT_USERNAME") or input("CVAT 帳號: ").strip()
        password = os.environ.get("CVAT_PASSWORD") or getpass("CVAT 密碼 (不回顯): ")
        resp = requests.post(
            f"{host.rstrip('/')}/api/auth/login",
            json={"username": username, "password": password},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            raise SystemExit(
                f"❌ 登入失敗 ({resp.status_code}): {resp.text[:300]}\n"
                "   ➔ 如果你是用 Google / SSO 登入 CVAT，這個帳號沒有密碼可用。\n"
                "      請改用瀏覽器 cookie: F12 ➔ Application ➔ Cookies ➔ https://app.cvat.ai\n"
                "      複製 sessionid (與 csrftoken)，再加參數 --sessionid xxx --csrftoken yyy\n"
                "   ➔ 或用完全離線的方式，詳見本檔開頭的『使用範例 5』"
            )
        key = resp.json().get("key")
        if not key:
            raise SystemExit(f"❌ 登入回應中沒有 key: {resp.text[:300]}")
        print(f"🔑 登入成功 (帳號 {username})")
        return client._use_token(key)

    @staticmethod
    def _auth_hint(status: int) -> str:
        if status in (401, 403):
            return (
                "\n   ➔ 認證被拒。cookie 可能過期 (重新登入 app.cvat.ai 後再複製一次 sessionid)，\n"
                "      或是 POST 缺少 csrftoken。"
            )
        return ""

    def get(self, path: str, **params: Any) -> Any:
        resp = self.session.get(f"{self.host}{path}", params=params, timeout=TIMEOUT)
        if resp.status_code != 200:
            raise SystemExit(
                f"❌ GET {path} 失敗 ({resp.status_code}): {resp.text[:500]}"
                f"{self._auth_hint(resp.status_code)}"
            )
        return resp.json()

    def post(self, path: str, payload: Any) -> Any:
        resp = self.session.post(f"{self.host}{path}", json=payload, timeout=TIMEOUT)
        if resp.status_code not in (200, 201):
            raise SystemExit(
                f"❌ POST {path} 失敗 ({resp.status_code}): {resp.text[:500]}"
                f"{self._auth_hint(resp.status_code)}"
            )
        return resp.json()

    def list_labels(self, *, task_id: Optional[int] = None, project_id: Optional[int] = None):
        params: Dict[str, Any] = {"page_size": 200}
        if task_id is not None:
            params["task_id"] = task_id
        if project_id is not None:
            params["project_id"] = project_id
        results: List[Dict[str, Any]] = []
        page = 1
        while True:
            data = self.get("/api/labels", page=page, **params)
            results.extend(data.get("results", []))
            if not data.get("next"):
                break
            page += 1
        return results


def parse_svg(svg: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """把 skeleton svg 拆成 circle / line 的屬性字典清單。"""

    def attrs(tag: str) -> Dict[str, str]:
        return dict(re.findall(r'([\w-]+)="([^"]*)"', tag))

    circles = [attrs(t) for t in re.findall(r"<circle\b[^>]*>", svg)]
    lines = [attrs(t) for t in re.findall(r"<line\b[^>]*>", svg)]
    return circles, lines


def diagnose_label(label: Dict[str, Any], shapes: Optional[List[Dict[str, Any]]]) -> int:
    """檢查一個 skeleton 標籤，回傳問題數量。"""
    problems = 0
    name = label.get("name")
    svg = label.get("svg") or ""
    sublabels = label.get("sublabels") or []
    sub_by_id = {sub["id"]: sub["name"] for sub in sublabels if "id" in sub}

    print(f"\n🦴 skeleton 標籤 '{name}' (label id {label.get('id')})")
    print(f"   sublabel 數量: {len(sublabels)}")

    if not svg:
        print("   ❌ svg 是空的")
        return 1

    # --- 檢查 0 (最常踩): svg 欄位自帶 <svg> 外層 ---
    # cvat-core Label constructor: t.svg = parseUntrustedSvg(`<svg>${t.svg ?? ""}</svg>`)
    # 自帶外層 ➔ <svg><svg>...</svg></svg> ➔ children 只有一個內層 <svg>
    # ➔ addSkeleton() 的 filter(el => el.type === 'circle') 得到空陣列 ➔ find() undefined
    if re.search(r"<svg\b", svg):
        print(
            "   ❌ svg 欄位自帶 <svg> 外層標籤 —— 這就是 reading 'attr' 的元兇\n"
            "      cvat-core 會再包一層: parseUntrustedSvg(`<svg>${svg}</svg>`)，\n"
            "      變成 <svg><svg>...</svg></svg>，structure.svg 的 children 只有一個內層 <svg>，\n"
            "      addSkeleton() 過濾 circle 後得到空陣列，find() 回傳 undefined。\n"
            "      ➔ 既有任務的 svg 改不掉 (唯讀)，必須刪掉任務用不含外層的定義重建"
        )
        problems += 1

    circles, lines = parse_svg(svg)
    print(f"   svg 內容: {len(circles)} 個 <circle> / {len(lines)} 條 <line>")
    stripped = sorted(
        {
            k
            for tag in circles + lines
            for k in tag
            if k not in CVAT_ALLOWED_SVG_ATTRS
        }
    )
    if stripped:
        print(f"   ⚠️  這些屬性不在 CVAT 的 DOMPurify 白名單內，會被移除: {stripped}")
    if SHOW_SVG:
        print(f"   svg 全文 ({len(svg)} 字元):\n{svg}")
    elif circles and lines:
        first_c = re.search(r"<circle\b[^>]*>", svg)
        first_l = re.search(r"<line\b[^>]*>", svg)
        print(f"   範例 circle: {first_c.group(0) if first_c else '-'}")
        print(f"   範例 line  : {first_l.group(0) if first_l else '-'}")

    if len(circles) != len(sublabels):
        print(f"   ❌ <circle> 數量 {len(circles)} 與 sublabel 數量 {len(sublabels)} 不符")
        problems += 1

    # --- 檢查 1: 名稱沒被替換成 id (伺服器端替換失敗) ---
    leftover = re.findall(r'data-label-name="([^"]*)"', svg)
    if leftover:
        print(
            f"   ❌ svg 內仍殘留 data-label-name: {leftover[:5]}{' ...' if len(leftover) > 5 else ''}\n"
            "      ➔ 伺服器的 create_labels() 沒有把名稱換成 id (名稱與 sublabel 對不上)\n"
            "      ➔ 前端 addSkeleton() 會找不到 <circle>，噴 reading 'attr'"
        )
        problems += 1

    # --- 檢查 2: node id / element id 必須是整數 (同類問題彙總，避免 17 行洗版) ---
    node_ids: Set[int] = set()
    no_node_id = no_element_id = 0
    for c in circles:
        nid, eid = c.get("data-node-id"), c.get("data-element-id")
        if nid and nid.isdigit():
            node_ids.add(int(nid))
        else:
            no_node_id += 1
        if not (eid and eid.isdigit()):
            no_element_id += 1
    if no_node_id:
        print(f"   ❌ 有 {no_node_id}/{len(circles)} 個 <circle> 缺少整數 data-node-id")
        problems += 1
    if no_element_id:
        print(f"   ❌ 有 {no_element_id}/{len(circles)} 個 <circle> 缺少整數 data-element-id")
        problems += 1

    # --- 檢查 3: edge 必須引用整數 node id (setupSkeletonEdges 的 Number.isInteger 檢查) ---
    non_int_edges = [
        (ln.get("data-node-from"), ln.get("data-node-to"))
        for ln in lines
        if not ((ln.get("data-node-from") or "").isdigit() and (ln.get("data-node-to") or "").isdigit())
    ]
    if non_int_edges:
        print(
            f"   ❌ 有 {len(non_int_edges)}/{len(lines)} 條 <line> 的 data-node-from/to 不是整數 node id"
            f" (例如 from={non_int_edges[0][0]} to={non_int_edges[0][1]})\n"
            "      ➔ setupSkeletonEdges() 會丟例外，最後噴 reading 'addClass'"
        )
        problems += 1
    dangling = [
        (int(ln["data-node-from"]), int(ln["data-node-to"]))
        for ln in lines
        if (ln.get("data-node-from") or "").isdigit() and (ln.get("data-node-to") or "").isdigit()
        and (int(ln["data-node-from"]) not in node_ids or int(ln["data-node-to"]) not in node_ids)
    ]
    if dangling:
        print(
            f"   ❌ 有 {len(dangling)} 條 <line> 參照到不存在的 node: {dangling[:5]}\n"
            f"      現有 node id: {sorted(node_ids)}"
        )
        problems += 1

    # --- 檢查 4: svg 的 data-label-id 必須都是真實 sublabel id ---
    svg_label_ids = {int(c["data-label-id"]) for c in circles if (c.get("data-label-id") or "").isdigit()}
    orphan = svg_label_ids - set(sub_by_id)
    if orphan:
        print(
            f"   ❌ svg 內的 data-label-id {sorted(orphan)} 不是這個標籤的 sublabel\n"
            f"      實際 sublabel id: {sorted(sub_by_id)}\n"
            "      ➔ addSkeleton() 的 templateElements.find() 回傳 undefined，噴 reading 'attr'"
        )
        problems += 1
    dup_ids = [i for i in set(svg_label_ids) if [
        c.get("data-label-id") for c in circles
    ].count(str(i)) > 1]
    if dup_ids:
        print(f"   ❌ 有多個 <circle> 綁到同一個 sublabel: {sorted(dup_ids)}")
        problems += 1

    non_points = [
        (sub.get("id"), sub.get("name"), sub.get("type"))
        for sub in sublabels
        if sub.get("type") not in (None, "points")
    ]
    if non_points:
        print(
            f"   ❌ 有 sublabel 的 type 不是 'points': {non_points}\n"
            "      ➔ addSkeleton() 只處理 shapeType === 'points' 的 element"
        )
        problems += 1

    unbound = set(sub_by_id) - svg_label_ids
    if unbound:
        preview = sorted((i, sub_by_id[i]) for i in unbound)[:5]
        print(
            f"   ❌ 有 {len(unbound)}/{len(sub_by_id)} 個 sublabel 沒有對應的 <circle>: "
            f"{preview}{' ...' if len(unbound) > 5 else ''}"
        )
        problems += 1

    # --- 檢查 5: sublabel 名稱是否與 COCO 17 點一致 (影響 COCO Keypoints 匯入對位) ---
    expected = [kp["name"] for kp in COCO_KEYPOINTS]
    actual = [sub["name"] for sub in sublabels]
    if actual != expected:
        missing = [n for n in expected if n not in actual]
        extra = [n for n in actual if n not in expected]
        print("   ⚠️  sublabel 名稱與 COCO 17 點標準順序不同")
        if missing:
            print(f"      缺少: {missing}")
        if extra:
            print(f"      多出: {extra}")
        if not missing and not extra:
            print(f"      (只是順序不同，匯入以名稱對應，通常無妨)")

    # --- 檢查 6: 標註的 element label_id 是否都能在 svg 找到 ---
    if shapes is not None:
        skeleton_shapes = [s for s in shapes if s.get("type") == "skeleton"]
        print(f"   標註: 共 {len(skeleton_shapes)} 個 skeleton shape")
        bad_ids: Set[int] = set()
        for shape in skeleton_shapes:
            for el in shape.get("elements", []):
                lid = el.get("label_id")
                if lid not in svg_label_ids:
                    bad_ids.add(lid)
        if bad_ids:
            print(
                f"   ❌ 標註中有 element 的 label_id {sorted(bad_ids)} 在 svg 裡找不到對應 <circle>\n"
                "      ➔ 這正是 addSkeleton() 噴 reading 'attr' 的直接原因"
            )
            problems += 1
        elif skeleton_shapes:
            print("   ✅ 標註的 element label_id 全部都能對應到 svg 的 <circle>")

    if problems == 0:
        print("   ✅ 這個 skeleton 標籤完全合規")
    return problems


def unwrap_labels(data: Any) -> List[Dict[str, Any]]:
    """接受 DRF 分頁物件 {count, results:[...]} 或裸陣列 [...]。"""
    if isinstance(data, dict):
        return data.get("results", [])
    if isinstance(data, list):
        return data
    raise SystemExit("❌ 標籤檔格式無法辨識 (預期為 JSON 陣列或含 results 的物件)")


SAVE_HINT = """
   ➔ 瀏覽器開 API 網址時，Chrome 會把 JSON 直接「顯示」在頁面上，並不會自動下載。
      請在該頁面按 Ctrl + S 存檔（存檔類型選「網頁，僅 HTML」以外的純文字皆可），
      或全選頁面內容 (Ctrl+A, Ctrl+C) 貼到新檔案裡。
   ➔ 存檔後請帶「完整路徑」，例如:
        --labels-file "C:\\Users\\USER\\Downloads\\labels.json"
   ➔ 或者乾脆不要存檔，直接用瀏覽器 cookie 讓程式自己抓:
        F12 ➔ Application ➔ Cookies ➔ https://app.cvat.ai，複製 sessionid
        python cvat_skeleton_api.py --sessionid <值> diagnose --task <TASK_ID> --annotations
"""


def load_json_file(path: Path, what: str) -> Any:
    """讀取使用者從瀏覽器存下來的 API 回應，並針對常見失誤給明確指引。"""
    if not path.exists():
        raise SystemExit(f"❌ 找不到{what}檔案: {path}{SAVE_HINT}")

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise SystemExit(f"❌ {path.name} 是空檔案{SAVE_HINT}")

    if text.startswith("<"):
        raise SystemExit(
            f"❌ {path.name} 存成 HTML 了，不是 JSON。\n"
            "   ➔ 網址結尾要加 &format=json (或 ?format=json)，讓 CVAT 回傳純 JSON，\n"
            "      不然存下來的是 DRF 的網頁版介面。"
            + SAVE_HINT
        )

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"❌ {path.name} 不是合法的 JSON ({exc})\n"
            "   ➔ 如果是從瀏覽器複製貼上的，請確認有複製「完整」內容（含最外層的 { 或 [）。"
            + SAVE_HINT
        ) from exc


def cmd_diagnose(client: Optional[CvatClient], args: argparse.Namespace) -> int:
    print("=" * 78)

    if args.labels_file:
        labels = unwrap_labels(load_json_file(args.labels_file, "標籤"))
        print(f"🔍 離線診斷 {args.labels_file.name}")
        shapes = None
        if args.annotations_file:
            ann = load_json_file(args.annotations_file, "標註")
            shapes = ann.get("shapes", []) if isinstance(ann, dict) else []
            print(f"   已載入標註: {len(shapes)} 個 shape")
    else:
        if client is None:
            print("❌ 需要認證，或改用 --labels-file 做離線診斷")
            return 1
        if args.task is None and args.project is None:
            print("❌ 請指定 --task 或 --project (或用 --labels-file 離線診斷)")
            return 1
        labels = client.list_labels(task_id=args.task, project_id=args.project)
        target = f"task {args.task}" if args.task else f"project {args.project}"
        print(f"🔍 診斷 {target} 的標籤 (資料直接來自伺服器，非 UI 顯示值)")
        shapes = None
        if args.annotations:
            if args.task is None:
                print("⚠️  --annotations 只支援 --task，略過標註檢查")
            else:
                ann = client.get(f"/api/tasks/{args.task}/annotations")
                shapes = ann.get("shapes", [])
                print(f"   已取得標註: {len(shapes)} 個 shape")

    print("=" * 78)
    print(f"共 {len(labels)} 筆 label (含 sublabel)")

    skeletons = [lb for lb in labels if lb.get("type") == "skeleton"]
    if not skeletons:
        print("❌ 找不到 type=skeleton 的標籤")
        return 1

    problems = sum(diagnose_label(lb, shapes) for lb in skeletons)

    print("\n" + "-" * 78)
    if problems:
        print(f"❌ 共發現 {problems} 個問題。")
        print("   既有任務的 skeleton svg 無法修改 (LabelSerializer 的 svg 為唯讀欄位)，")
        print("   請用 create-project 建立乾淨的 Project，再把任務建在該 Project 底下。")
    else:
        print("✅ 沒發現問題。")

    if args.dump:
        out = args.dump
        out.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"💾 伺服器原始標籤已存到 {out}")
    return 1 if problems else 0


def cmd_create_project(client: CvatClient, args: argparse.Namespace) -> int:
    print("=" * 78)
    print(f"🏗  建立 Project '{args.name}' (帶 COCO-17 skeleton 標籤)")
    print("=" * 78)

    labels = build_raw_labels()
    project = client.post("/api/projects", {"name": args.name, "labels": labels})
    project_id = project["id"]
    print(f"✅ Project 建立成功: id={project_id}")
    print(f"   {client.host}/projects/{project_id}")

    # 立刻讀回伺服器實際存的 svg 做驗證 —— 這是唯一可信的檢查
    print("\n🔍 讀回伺服器實際存的標籤做驗證 ...")
    server_labels = client.list_labels(project_id=project_id)
    skeletons = [lb for lb in server_labels if lb.get("type") == "skeleton"]
    problems = sum(diagnose_label(lb, None) for lb in skeletons)

    print("\n" + "-" * 78)
    if problems:
        print(f"❌ 伺服器存下來的標籤有 {problems} 個問題，請把上面的輸出貼回來。")
        return 1

    print("✅ 標籤驗證通過！接下來在 UI 建立任務:")
    print(f"   1. 開 {client.host}/tasks/create")
    print(f"   2. **Project** 欄位選擇剛建立的 '{args.name}'")
    print("      (選了 Project 之後 Labels 區塊會變成唯讀，直接繼承 Project 的標籤，")
    print("       完全繞開 UI 的標籤編輯器)")
    print("   3. Select files 拖入 images_sub2-i17.zip ➔ Submit & Open")
    print("   4. Actions ➔ Upload annotations ➔ COCO Keypoints 1.0 ➔ coco_keypoints_annotations.zip")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CVAT skeleton 標籤診斷 / 建立工具")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"CVAT 主機 (預設 {DEFAULT_HOST})")
    parser.add_argument(
        "--sessionid",
        default=None,
        help="瀏覽器的 sessionid cookie (Google/SSO 登入者用；也可設環境變數 CVAT_SESSIONID)",
    )
    parser.add_argument(
        "--csrftoken",
        default=None,
        help="瀏覽器的 csrftoken cookie (POST 才需要；也可設環境變數 CVAT_CSRFTOKEN)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_diag = sub.add_parser("diagnose", help="讀取伺服器上真實的 skeleton 標籤並檢查")
    p_diag.add_argument("--task", type=int, default=None, help="Task id (網址 /tasks/<id>)")
    p_diag.add_argument("--project", type=int, default=None, help="Project id")
    p_diag.add_argument("--annotations", action="store_true", help="一併檢查標註的 element label_id")
    p_diag.add_argument("--dump", type=Path, default=None, help="把伺服器原始標籤存成檔案")
    p_diag.add_argument("--show-svg", action="store_true", help="印出 skeleton svg 完整內容")
    p_diag.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="離線診斷: 從瀏覽器存下來的 /api/labels 回應檔 (不需認證)",
    )
    p_diag.add_argument(
        "--annotations-file",
        type=Path,
        default=None,
        help="離線診斷: 從瀏覽器存下來的 /api/tasks/<id>/annotations 回應檔",
    )

    p_proj = sub.add_parser("create-project", help="建立帶正確 skeleton 標籤的 Project")
    p_proj.add_argument("--name", default="BenchPress_COCO17", help="Project 名稱")

    args = parser.parse_args()

    global SHOW_SVG
    SHOW_SVG = getattr(args, "show_svg", False)

    # 離線診斷不需要任何認證
    offline = args.command == "diagnose" and args.labels_file is not None
    client = None if offline else CvatClient.login(args.host, args)

    if args.command == "diagnose":
        return cmd_diagnose(client, args)
    if args.command == "create-project":
        assert client is not None
        return cmd_create_project(client, args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
