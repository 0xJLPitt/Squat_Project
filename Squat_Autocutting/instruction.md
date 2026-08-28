# Squat Autocutting (深蹲自動切片系統) 說明文件

本資料夾包含了深蹲動作自動化特徵提取與切片的核心程式碼。為了方便未來維護與使用，以下將詳細介紹各個 `.py` 檔案的用途，以及總控台 `autocutting.py` 的使用觀念。

---

## 📂 核心檔案功能介紹

### 1. `squat_analysis.py` (核心演算法引擎)
- **用途**：這是整個系統的**大腦**。裡面實作了 `SquatFeatureExtractor` 類別，專門處理物理訊號運算。
- **功能**：
  - 將 YOLO 或 Mediapipe 輸出的骨架特徵與槓鈴座標進行平滑化（Smoothing）。
  - 計算速度與加速度。
  - 透過「高視覺化閾值 (0.5 std)」與「3 幀防手震 (Debounce)」機制，精準抓出每次深蹲的起始點（Start）、波谷（Bottom）與結束點（End）。
- **注意**：這是一個純演算法模組，**不包含任何路徑或資料夾讀取邏輯**，專門提供給其他腳本呼叫。

### 2. `cal_segments.py` (資料夾爬蟲與切片計算)
- **用途**：負責讀取資料集並產出切片數據。
- **功能**：
  - 遞迴爬梳指定的資料夾，自動尋找包含 `yolo_skeleton.txt` 與 `yolo_coordinates.txt` 的目標資料夾。
  - 自動排除名稱中帶有「棋盤」的校正影片資料夾。
  - 呼叫 `squat_analysis.py` 進行運算，並將計算出來的起訖點匯出為 `segments.json`。
  - 繪製並輸出供肉眼檢查的波形圖 `segmentation_check.png`。
- **注意**：此檔案不包含任何寫死（Hardcoded）的處理路徑，強制透過命令列參數 `--input` 來指定目標。

### 3. `stitch_segments.py` (影片實體剪輯與拼接)
- **用途**：負責將數學切片數據實體化為影片。
- **功能**：
  - 讀取各子資料夾內的 `segments.json` 與原始影片 `RD.avi`。
  - 根據 JSON 中的起訖點（Start to End），使用 OpenCV 將每一次的深蹲精華片段獨立擷取下來。
  - 將所有擷取下來的區間無縫拼接成一支流暢的精華影片 `RD_seg.mp4`，去除了受試者休息或走動的垃圾時間。
- **注意**：此檔案同樣不包含寫死路徑，且會自動跳過「棋盤」資料夾。

### 4. `eval_by_error_type.py` (依動作/錯誤類別獨立評估切片演算法誤差)
- **用途**：讀取 `S83_S108.json` 標註中的動作類別（`正常`、`下蹲深度不足`、`骨盆後傾`、`髖部上升過快(起身時髖部先啟動)`、`下蹲時髖部主導`、`下蹲時膝蓋主導`），分別獨立計算每個動作的切片誤差與總下數。
- **功能**：
  - 統計各動作類別的「影片錄影數量」與「實際評估深蹲總下數」。
  - 計算各動作類別的 Start Frame 平均相差 (MAE)、End Frame 平均相差 (MAE)、秒數誤差與標準差。
  - 自動匯出彙整報表 `eval_error_types_summary.csv`、逐下明細 `eval_error_types_details.csv` 以及視覺化長條圖 `eval_error_types_comparison.png`。
- **執行方式**：
  ```powershell
  mamba run -n hw1 python eval_by_error_type.py
  ```

---

## 🚀 總控台：`autocutting.py` 的使用觀念

為了避免每次都要分別執行計算與拼接腳本，我們設計了 **`autocutting.py`** 作為一鍵執行的總控台（Wrapper）。它負責按照順序呼叫 `cal_segments.py` 與 `stitch_segments.py`，並統一管理目標資料夾。

### 核心觀念：預設路徑 (Default) vs. 優先指定 (Override)

打開 `autocutting.py` 原始碼，你會在最上方看到一個預設路徑變數：
```python
DEFAULT_INPUT_PATH = r"E:\squat_dataset2\be"
```

這個設計提供了兩種靈活的使用方式：

#### 方式一：懶人一鍵執行（使用寫死的預設值）
如果你平常都是處理整批資料集，不需要做任何更改，直接在環境中執行：
```powershell
mamba run -n hw1 python autocutting.py
```
程式會自動拿腳本內寫死的 `DEFAULT_INPUT_PATH` 當作目標，掃描整個 `be` 母資料夾。

#### 方式二：針對特定資料夾執行（參數覆蓋最高順位）
如果今天你剛拿到一位新受試者 (例如 S92) 的資料，或者只想測試單一影片，你可以透過 `--input` 參數從外部傳入路徑。**外部傳入的參數擁有最高順位，會直接覆蓋掉寫死的預設值**：
```powershell
# 針對特定受試者的母資料夾
mamba run -n hw1 python autocutting.py --input "E:\squat_dataset2\be\S92\session01"

# 或者針對單一錄影資料夾
mamba run -n hw1 python autocutting.py --input "E:\squat_dataset2\be\S92\session01\recording_20260515_110654"
```
透過這種方式，所有底層腳本（計算、拼接）都會自動遵循這個新指定的路徑去運作，讓你不需要修改任何一行程式碼就能靈活調度！

### 💡 備註：如何開啟/關閉拼接功能
目前在 `autocutting.py` 原始碼中，步驟二（執行 `stitch_segments.py`）被使用多行註解 `"""` 隱藏起來了。這代表現在執行總控台時，**只會產出圖表與 JSON，不會剪輯影片**。
當你確認 `segmentation_check.png` 上的切點都完美無誤，隨時可以將程式碼中的 `"""` 刪除，就能恢復一鍵計算 + 剪輯拼接的完整全自動化流水線！

