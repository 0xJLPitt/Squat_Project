# SBDFormer / 臥推 3D 多視角動作切片與前處理工作流

本專案目錄存放針對 **臥推動作（Bench Press）** 進行多視角影片 2D 骨架辨識、關鍵動作切片分析（5 大關鍵幀 / 4 階段分期）以及多視角關鍵影格萃取的工具鏈，為後續 **3D 重建** 與 **SBDFormer / AGFormer 模型微調與訓練** 提供標準化標註與資料集切片。

---

## 📁 多視角資料集巢狀結構 (Directory Hierarchy)

為了確保多攝影機感測資料與全局動作真值（Ground Truth）權責清晰，本專案採用 **「受試者母層共享前處理 (Subject-Level Centralized Preprocessing)」** 結構：

```text
D:\Pitt\Project\Squat_Project\video\benchpress_3D\
├── extrinsics\                             # 多相機外部參數矩陣 (.npz)
├── intrinsics\                             # 各相機內部參數矩陣 (.npz)
│
└── subject\                                # 各受試者實驗資料根目錄
    └── sub2\                               # 【母資料夾 (Subject Root)】
        │
        ├── dataprocess\                    # 【母層：受試者全局切片與分析數據】
        │   │                               # (智慧相容命名: dataprocess 或 dataprocess-i17)
        │   ├── yolo_skeleton_error1.txt    # 來自 i17 視角的 YOLO 2D 骨架座標 (12 身體關鍵點，已排除臉部 5 點)
        │   ├── yolo_skeleton_error1_segments.json     # 動作切片 JSON 標註檔 (含 Rep 1~n 關鍵影格)
        │   ├── yolo_skeleton_error1_segmentation.png  # 手腕運動軌跡與分期波形圖
        │   ├── yolo_skeleton_error1_segments.csv      # (選配) 切片資訊表格檔
        │   ├── yolo_skeleton_error2_segments.json
        │   ├── yolo_skeleton_error3_segments.json
        │   ├── yolo_skeleton_error4_segments.json
        │   └── yolo_skeleton_normal_segments.json
        │
        ├── sub1-i15\                       # 【相機 1 視角 (iPhone 15)】
        │   ├── checkboard_external2.MP4    # (選配) 外參棋盤格影片 (分析時自動排除)
        │   ├── error1.MP4                  # 原始動作影片 (如 error1~4, normal)
        │   ├── error1_keyframes\           # 抽出的 Rep 1~n 關鍵點照片 (40 張)
        │   │   ├── error1_frame298.jpg     # 格式: {videoname}_frame{xxx}.jpg
        │   │   ├── error1_frame317.jpg
        │   │   ├── error1_frame332.jpg
        │   │   └── error1_frame356.jpg
        │   ├── error2_keyframes\
        │   ├── error3_keyframes\
        │   ├── error4_keyframes\
        │   └── normal_keyframes\
        │
        ├── sub2-i16\                       # 【相機 2 視角 (iPhone 16)】
        │   ├── error1.MP4
        │   ├── error1_keyframes\
        │   └── ...
        │
        └── sub2-i17\                       # 【相機 3 視角 (iPhone 17，主要骨架分析視角)】
            ├── error1.MP4
            ├── error1_keyframes\
            └── ...
```

### 💡 設計原則與優勢
1. **視角感測器（Sensors）與受試者真值（Subject Ground Truth）分離**：
   - 臥推的動作分期（第幾幀觸胸、推起）是受試者的客觀運動狀態，三台相機同時同步錄製。
   - `sub2-i17` 僅作為骨架分析的「感測來源」；產出的 `segments.json` 是**所有相機共享的動作標籤**。
   - 將切片集中放在 `sub2\dataprocess\`，讓各相機視角引用標籤時路徑最純粹，不會發生跨目錄混淆。
2. **完美相容 SBDFormer / AGFormer 切片格式 (`plan.md`)**：
   - 未來進行 3D 重建與資料集打包時，打包程式只需傳入母目錄 `sub2`，即可直接在 `dataprocess\` 讀取時序切片，並向 `sub1-i15`、`sub2-i16`、`sub2-i17` 提取滑動窗口樣本。

---

## ⚙️ 核心腳本說明 (Core Processing Pipeline)

本工作流分為四大標準處理階段，各步驟腳本檔名前綴標註為 **`step1 ~ step4`**，循序漸進完成前處理：

| 步驟 | 腳本名稱 | 處理階段 | 主要功能 | 關鍵輸入 | 關鍵輸出 |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Step 1** | **`step1_run_yolo_pose.py`** | **影片處理** | YOLO11 Pose 姿態辨識，抽取 2D 人體 17 關節點數據 | 主視角 `.MP4` (如 i17) | `dataprocess\yolo_skeleton_*.txt` |
| **Step 2** | **`step2_benchpress_label_preprocess_alog.py`** | **關鍵點判斷** | 運動學手腕波形分析，自動偵測 5 大關鍵點與切片分期 | `dataprocess\*.txt` | `dataprocess\*_segments.json`<br>`dataprocess\*_segmentation.png` |
| **Step 3** | **`step3_extract_benchpress_keyframes.py`** | **照片擷取** | 多視角關鍵影格批次抽取，自動過濾棋盤格並依切片抽圖 | `dataprocess\*.json`<br>各相機 `.MP4` | 各視角目錄下的<br>`{video}_keyframes\*.jpg` |
| **Step 4** | **`step4_export_benchpress_cvat_labels.py`** | **照片檔與json整理** | 融合關鍵照片與骨架數據，產出唯一標準 COCO Keypoints JSON 標註與圖片 ZIP | 抽出的關鍵照片<br>`dataprocess\*.json`<br>`dataprocess\*.txt` | `cvat_export/{cam}/`<br>`images_*.zip`<br>`person_keypoints_default.json`<br>`cvat_raw_labels.json` |
| **標籤工具** | **`make_cvat_skeleton_labels.py`** | **CVAT 標籤產生** | 產生 CVAT Raw 格式之 COCO-17 骨架定義 JSON，解決網站無法直接建立骨架的限制 | - | `cvat_raw_labels.json` |

---

## 🚀 完整執行工作流 (Standard SOP)

當錄製完成新的受試者資料（例如 `sub3`、`sub4`）時，標準一條龍作業程序如下：

### 步驟 1：影片處理 ➔ 執行 YOLO Pose 姿態辨識 (`step1_run_yolo_pose.py`)
針對主視角（如 `i17`）跑 YOLO11 Pose，自動輸出骨架數據 `.txt` 至母目錄的 `dataprocess/`：
```powershell
# 母目錄模式 (自動定位 i17 視角並輸出至 dataprocess/)
mamba run -n hw1 python step1_run_yolo_pose.py --subject "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2"

# (選填) 指定相機目錄模式
mamba run -n hw1 python step1_run_yolo_pose.py -d "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\sub2-i17" -o "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\dataprocess"
```

### 步驟 2：關鍵點判斷 ➔ 執行動作切片演算法 (`step2_benchpress_label_preprocess_alog.py`)
輸入受試者母目錄，程式自動尋找 `dataprocess/` 內的骨架資料，並將所有動作的切片 JSON 與視覺化波形圖存入 `dataprocess/`：
```powershell
# 母目錄批次處理模式 (推薦)
mamba run -n hw1 python step2_benchpress_label_preprocess_alog.py --dir "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2"

# (選填) 單檔測試模式
mamba run -n hw1 python step2_benchpress_label_preprocess_alog.py --input "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\dataprocess\yolo_skeleton_error1.txt"
```

### 步驟 3：照片擷取 ➔ 多視角關鍵影格批次抽取 (`step3_extract_benchpress_keyframes.py`)
自動讀取母目錄 `dataprocess/` 中的切片數據，對所有非 `checkboard` 影片的三個相機視角（i15、i16、i17）進行 Rep 1~n 關鍵影格抽取：
```powershell
# 預設全量批次抽取 (自動處理 error1~4, normal 全部 10 組 Rep)
mamba run -n hw1 python step3_extract_benchpress_keyframes.py --dir "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2"

# (選填) 抽幀完成後自動串聯執行 Step 4 打包 CVAT 標註包
mamba run -n hw1 python step3_extract_benchpress_keyframes.py --export-cvat
```

### 步驟 4：照片檔與json整理 ➔ 骨架融合與 CVAT 標註包匯出 (`step4_export_benchpress_cvat_labels.py`)
將抽出的關鍵影格圖片與 YOLO 17 點骨架數據精確對齊，打包成標準 **COCO Keypoints 1.0 JSON** 格式、CVAT Raw 標籤檔與圖片 ZIP：
```powershell
# 預設匯出 sub2-i17 視角 (產出 images_sub2-i17.zip、person_keypoints_default.json 與 cvat_raw_labels.json)
mamba run -n hw1 python step4_export_benchpress_cvat_labels.py

# (選填) 指定受試者母目錄與相機視角
mamba run -n hw1 python step4_export_benchpress_cvat_labels.py --dir "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2" --cam sub2-i17

# (選填) 僅匯出單一動作影片 (例如 error1)
mamba run -n hw1 python step4_export_benchpress_cvat_labels.py --video error1
```

---

### 🏷️ 核心輔助工具：CVAT Raw 骨架標籤產生工具 (`make_cvat_skeleton_labels.py`)

> ⚠️ **保留本程式的原因與 CVAT 網站環境說明**：
> 許多 CVAT 網站版本或自建伺服器**無法在 Web UI 上直接點選建立 COCO Keypoints 骨架**（UI 缺乏 Skeleton 精靈，或精靈無法直接選取 17 點 COCO 人體骨骼模板）。
> 因此本專案**嚴格保留 `make_cvat_skeleton_labels.py`**，專門產出符合 CVAT 規格的 **Raw 格式標籤定義檔 (`cvat_raw_labels.json`)**。使用者在建立 Task 時，只需切換至 **`Labels ➔ Raw`** 分頁，直接貼上此 JSON 內容即可一鍵建立合規的 17 關節點骨架！

#### 📌 本工具解決的 CVAT 前後端相容性問題：
1. **SVG 欄位不可自帶 `<svg>` 外層**：`cvat-core` 的 Label constructor 會自行封裝 `<svg>${t.svg}</svg>`。若原始 JSON 自帶外層，會造成雙層嵌套 `<svg><svg>...</svg></svg>`，導致前端 `addSkeleton()` 找不到 `<circle>` 噴出 `TypeError: Cannot read properties of undefined (reading 'attr')`。
2. **`<circle>` 一律使用 `data-label-name` 綁定**：不可寫死 `data-label-id`，必須由 CVAT 伺服器在建立 sublabel 時動態替換為資料庫分配的 ID。
3. **`<line>` 的 `data-node-from` / `data-node-to` 必須為整數 node id**：不可寫關節名稱字串，否則觸發 `setupSkeletonEdges()` 的整數型別檢查錯誤。
4. **完全符合 DOMPurify 白名單**：僅包含白名單屬性，避免被前端靜默過濾。
5. **既有任務唯讀保護**：CVAT 的 skeleton svg 在建立任務後即為唯讀（`read_only_fields`）；若骨架 SVG 有誤必須刪除任務重新建立。

```powershell
# 手動產生 Raw 標籤定義檔 (預設輸出至 sub2-i17 cvat_export 目錄)
mamba run -n hw1 python make_cvat_skeleton_labels.py

# (選填) 檢查既有 raw labels JSON 是否合規 (避免前後端踩雷)
mamba run -n hw1 python make_cvat_skeleton_labels.py --check "path\to\cvat_raw_labels.json"
```

---

#### 🌐 CVAT 網站標註標準操作 SOP (含 Raw 分頁操作):
1. **建立 Task (設定標籤與上傳照片)**:
   - 進入 CVAT 網站（如 `https://app.cvat.ai/tasks/create` 或自建伺服器）。
   - **Task Name**: 輸入任務名稱，如 `BenchPress_sub2-i17`。
   - **Labels 標籤設定（重要）**：
      - **【方案 A：Raw 分頁貼上（推薦，適用 12 關鍵點客製骨架標籤）】**：
        在 `Labels` 區域右上角切換至 **`Raw`** 分頁，打開同目錄下的 **`cvat_raw_labels.json`**，全選複製內容並貼入文字框中即可（包含 12 個關節點與對應骨骼連線）！
      - **【方案 B：UI 精靈法】**：
        若使用 UI 手動建立，請建立 `person` (skeleton) 標籤，並加入 12 個子點 (left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle)。強烈建議直接使用方案 A 的 Raw 分頁最快速且不會出錯。
   - **上傳圖片包 (Select Files)**：
     - 將匯出的純圖片壓縮包 **`images_sub2-i17.zip`** 拖入檔案區（**僅傳圖片 ZIP，切勿把 JSON 拖入此區**）。
   - 點選最下方 **`Submit & Open`** 建立任務。
2. **匯入預標註 (Upload Pre-annotations)**:
   - 進入已建立成功的 Task 頁面，點選右上角選單 **`Actions` (三個點點)** ➔ **`Upload Annotations`**。
   - 格式選擇 **`COCO Keypoints 1.0`**。
   - 檔案選擇匯出的 **`person_keypoints_default.json`**，點擊確認上傳，1~2 秒鐘即載入完畢！
3. **標註人員微調 (Fine-tuning)**:
   - 打開影格，12 個身體骨架點已依照 YOLO 偵測的位置排列並繪製骨骼連線（頭部眼睛與耳朵 5 點已排除，不再干擾畫面）。
   - 標註人員僅需檢查並微調臥推時被槓鈴反光遮擋的關節點：
     - **手腕 (`left_wrist`, `right_wrist`)**：對齊手掌根骨與槓鈴握法交界處。
     - **手肘 (`left_elbow`, `right_elbow`)**：確認手肘彎曲最低點。
     - **肩膀與胸口**：確認肩峰與鎖骨定位。
   - 微調完成後按 `Ctrl + S` 儲存！
4. **匯出修正後真值 (Export Dataset)**:
   - 標註完成後，在 Task 頁面點選 `Export dataset` ➔ 格式選擇 **`COCO Keypoints 1.0`** 下載已校正好的真值 JSON。


---

## 🎯 臥推 4 大關鍵動作影格定義

切片分析產出的 `*_segments.json` 針對每次反覆記錄以下 4 大關鍵點：
1. **頂點 (起) `top_start_frame`** : 出槓/推至頂部鎖定點（Y 座標波谷，垂直加速度起點）。
2. **下降中段 `descent_mid_frame`** : 下放過程中通過 50% 行程之影格。
3. **底點觸胸 `bottom_frame`** : 槓鈴觸胸最低點（Y 座標波峰，垂直位移最大點）。
4. **上升中段 `ascent_mid_frame`** : 上推過程中通過 50% 行程之影格。

每部影片抽取出的圖片命名格式皆保證為 **`{videoname}_frame{xxx}.jpg`**，方便後續電腦視覺檢視與三維空間重建校正。

