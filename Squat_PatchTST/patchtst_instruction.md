# PatchTST 應用於多標籤錯誤判斷模型之架構解析

本文件整理了將原本用於時間序列預測 (Time-Series Forecasting) 的 **PatchTST** 模型，改編為**多標籤錯誤判斷 (Multi-label Classification) 信心值預測模型**的關鍵修改細節與資料準備方式。

---

## 1. 模型架構的核心修改

在原版 PatchTST 中，模型使用「通道獨立 (Channel Independence)」機制，讓各個維度的特徵獨立提取 Patch 並預測未來。為了轉成判斷整體動作是否錯誤的模型，進行了以下關鍵改動（參考 `models.py` 中的 `PatchTSTClassifier`）：

### A. 打破最終的通道獨立性 (Global Flatten)
經過 Transformer 提取特徵後，程式碼**沒有**讓每個 Channel 獨立輸出，而是把**「所有的特徵維度 (Channel)」加上「所有的 Patch 特徵」全部一次性攤平 (Flatten)**：
```python
# (B*C, num_patches, embed_dim) -> 展平成 (B, 總特徵量)
x = x.view(B, -1) 
```

### B. 新增多標籤分類頭 (Classification Head)
捨棄了原本預測未來的 Linear Head，接上了一個自定義的分類頭，輸出 `num_classes`（錯誤種類數）的 Logits：
```python
self.classifier = nn.Sequential(
    nn.LayerNorm(input_dim * num_patches * embed_dim), 
    nn.Linear(input_dim * num_patches * embed_dim, num_classes)
)
```

---

## 2. 對照原始論文架構圖的替換說明

若要修改原始 PatchTST 論文架構圖以符合本專案，只需替換最上方的兩個區塊，底層的 Patching 與 Transformer 機制維持原樣：

1. **綠色區塊 (原本: Flatten + Linear Head)**
   * 替換為：**`Global Flatten + Classifier`** 
   * 說明：強調它是將「所有 Channel 與 Patch」一起攤平，而非單一 Channel 內部攤平。
2. **最上方淺藍色區塊 (原本: Output Univariate Series)**
   * 替換為：**`Multi-label Confidence Scores`** 
   * 說明：輸出不再是一段未來的時間序列，而是針對多個錯誤種類的信心判斷值。

---

## 3. 為什麼必須「攤平所有數據」？

這是一個**「局部觀察」vs「全局觀察」**的設計：
* 判斷複雜的錯誤動作（例如臥推時「左右手不平衡」）單看一個關節的獨立軌跡是不夠的。
* 攤平所有特徵後，Classification Head 宛如站在上帝視角，同時掌握了**空間資訊**（全身關節同瞬間的相對位置）與**時間資訊**（整個動作週期的變化軌跡）。
* 基於這個包含所有資訊的唯一特徵向量，模型才能同時對多個問題做出判斷（例如：A錯誤機率 80%、B錯誤機率 15%），實現多標籤分類。

---

## 4. 損失函數與信心值判斷 (Training Logic)

在 `PatchTST_train.py` 中，透過以下設定實現多標籤與信心值：
* **損失函數**：使用 `torch.nn.BCEWithLogitsLoss()`。把每一個錯誤標籤當作獨立的二元分類問題，各標籤互不互斥。
* **信心值與閾值**：使用 `Sigmoid` 將輸出壓縮至 `0 ~ 1` 的信心值，並以 `0.5` 為判定門檻：
  ```python
  probs = torch.sigmoid(outputs)  # [B, num_classes] 信心值
  preds = (probs > 0.5).int()     # 信心值 > 0.5 視為有犯錯
  ```

---

## 5. 訓練資料準備格式 (Dataset Structure)

訓練需要提供 CSV 檔案（例如 `benchpress_dataset.csv`），每一列代表「一次完整的訓練動作 (Rep)」，必須包含以下欄位：

1. **`subject`**：受測者 ID。用於確保同一個人的資料不會同時出現在訓練集與測試集，避免資料洩漏。
2. **`label`**：多標籤答案，如 `[0, 1, 0, 1]` 代表發生了第 2 種與第 4 種錯誤。
3. **`features`**：形狀為 `(Time, Channel)` 的 2D 陣列。
   * **Time**：動作被 Normalize 到的固定長度（例如臥推 100 Frames）。
   * **Channel**：追蹤的特徵數量（如各關節角度、座標）。

### `features` 資料型態範例 (Time=100, Channel=5)
假設追蹤 5 個特徵：`[槓鈴Y, 左手肘角度, 右手肘角度, 左肩Y, 右肩Y]`：

```python
[
  # Frame 1 (動作剛開始，槓鈴在最高點)
  [0.85, 175.2, 174.8, 0.45, 0.46],  
  
  # Frame 2 
  [0.83, 170.1, 169.5, 0.45, 0.46],  
  
  ... (下放過程) ...

  # Frame 50 (槓鈴最低點，此處出現左右手角度不對稱特徵)
  [0.30, 85.1, 80.5, 0.48, 0.49],   

  ... (推起過程) ...

  # Frame 100 (動作結束)
  [0.85, 174.5, 173.8, 0.45, 0.46]   
]
```
透過這樣的格式輸入，模型將能在訓練過程中學習如何將整體動作軌跡對應到特定的錯誤標籤。
