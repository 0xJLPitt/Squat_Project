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
        │   ├── yolo_skeleton_error1.txt    # 來自 i17 視角的 YOLO 2D 骨架座標 (17 關節點)
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

## ⚙️ 核心腳本說明

| 腳本名稱 | 主要功能 | 關鍵輸入 | 關鍵輸出 |
| :--- | :--- | :--- | :--- |
| **`benchpress_label_preprocess_alog.py`** | 運動學特徵分析演算法，自動偵測左手手腕波形並擷取關鍵影格 | `sub2-i17` 視角骨架 `.txt` | `sub2\dataprocess\*.json`<br>`sub2\dataprocess\*.png` |
| **`extract_benchpress_keyframes.py`** | 多視角圖片批次萃取程式，自動過濾棋盤格並依據切片抽圖 | `dataprocess\*.json`<br>各相機 `.MP4` | 各視角目錄下的<br>`{video}_keyframes\*.jpg` |
| **`export_benchpress_cvat_labels.py`** | 融合關鍵影格與 17 點骨架數據，產出 CVAT COCO Keypoints 1.0 標註包 | `dataprocess\*.json`<br>`dataprocess\*.txt`<br>抽出的關鍵影格 `.jpg` | `cvat_export/{cam}/`<br>`instances_default.json`<br>`cvat_{cam}_dataset.zip` |

---

## 🚀 完整執行工作流 (Standard SOP)

當錄製完成新的受試者資料（例如 `sub3`、`sub4`）時，標準作業程序如下：

### 步驟 1：執行 YOLO Pose 姿態辨識 (取得 i17 骨架數據)
```powershell
# 針對 i17 視角資料夾跑 YOLO Pose，輸出骨架 txt 至母目錄的 dataprocess 中
python ..\Squat_2dto3d\step1_run_yolo_pose.py -d "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\sub2-i17" -o "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\dataprocess"
```

### 步驟 2：執行動作切片演算法 (`benchpress_label_preprocess_alog.py`)
輸入受試者母目錄，程式自動尋找 `i17` 骨架資料，並將所有動作的切片 JSON 與視覺化檢查圖存入 `dataprocess/`：
```powershell
# 母目錄批次處理模式 (推薦)
mamba run -n hw1 python benchpress_label_preprocess_alog.py --dir "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2"

# (選填) 單檔測試模式
mamba run -n hw1 python benchpress_label_preprocess_alog.py --input "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2\dataprocess\yolo_skeleton_error1.txt"
```

### 步驟 3：多視角關鍵影格批次抽取 (`extract_benchpress_keyframes.py`)
自動讀取母目錄 `dataprocess/` 中的切片數據，對所有非 `checkboard` 影片的三個相機視角進行 Rep 1~n 關鍵影格抽取：
```powershell
# 預設全量批次抽取 (自動處理 error1~4, normal 全部 10 組 Rep)
mamba run -n hw1 python extract_benchpress_keyframes.py --dir "D:\Pitt\Project\Squat_Project\video\benchpress_3D\subject\sub2"

# (選填) 抽幀並自動串聯打包 CVAT 標註包
mamba run -n hw1 python extract_benchpress_keyframes.py --export-cvat
```

### 步驟 4：骨架融合與 CVAT 標註包匯出 (`export_benchpress_cvat_labels.py`)
將抽出的關鍵影格圖片與 YOLO 17 點骨架數據精確對齊，打包成原生 **CVAT for images 1.1** (XML) 與標準 **COCO Keypoints 1.0** 格式：
```powershell
# 預設匯出 sub2-i17 視角 (產出 images_sub2-i17.zip、annotations.xml 與 person_keypoints_default.json)
mamba run -n hw1 python export_benchpress_cvat_labels.py

# (選填) 僅匯出單一動作影片 (例如 error1)
mamba run -n hw1 python export_benchpress_cvat_labels.py --video error1
```

#### 🌐 CVAT 網站標註 3 步快速操作 SOP:
1. **建立 Task**: 進入 [app.cvat.ai](https://app.cvat.ai/tasks/create)，在 Labels 點選 `Setup Skeleton` 選擇 `COCO Keypoints (17 points)`，標籤名稱設為 `person`。
2. **上傳圖片**: 直接將 `images_sub2-i17.zip` 拖入檔案區（僅傳圖片），點選 `Submit & Open`。
3. **匯入預標註 (推薦首選)**: 
   - 進入任務點選 `Actions` ➔ `Upload Annotations` ➔ 格式選擇 **`CVAT for images 1.1`** ➔ 上傳 **`annotations.xml`** (或 `annotations_cvat_xml_sub2-i17.zip`)。
   - *(備選)* 格式選擇 **`COCO Keypoints 1.0`** ➔ 上傳 **`person_keypoints_default.json`**。
4. **滑鼠微調**: 打開影格，17 個骨架點已完整呈現在人體上，標註人員只需微調被槓鈴遮擋的手腕與手肘關節！


---

## 🎯 臥推 4 大關鍵動作影格定義

切片分析產出的 `*_segments.json` 針對每次反覆記錄以下 4 大關鍵點：
1. **頂點 (起) `top_start_frame`** : 出槓/推至頂部鎖定點（Y 座標波谷，垂直加速度起點）。
2. **下降中段 `descent_mid_frame`** : 下放過程中通過 50% 行程之影格。
3. **底點觸胸 `bottom_frame`** : 槓鈴觸胸最低點（Y 座標波峰，垂直位移最大點）。
4. **上升中段 `ascent_mid_frame`** : 上推過程中通過 50% 行程之影格。

每部影片抽取出的圖片命名格式皆保證為 **`{videoname}_frame{xxx}.jpg`**，方便後續電腦視覺檢視與三維空間重建校正。

