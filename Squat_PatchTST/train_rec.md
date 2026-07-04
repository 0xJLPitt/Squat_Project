# 訓練與修改紀錄 (Training Record)

本文件紀錄了為了解決 PatchTST 模型過擬合 (Overfitting) 以及資料特徵不一致的問題，所進行的核心程式碼改動。

## 📍 測試與模型儲存位置
上述所有改動的實驗與模型結果，皆儲存於以下資料夾：
👉 `./models/deadlift/TST_Deadlift/deadlift_set_split_aug_patience40`

---

## 🛠️ 核心改動項目

### 1. 修復 Benchpress (臥推) 缺乏 `[-1, 1]` 正規化的問題
* **涉及檔案**：`dataset/tools/Benchpress_tool/predict.py`, `dataset/processors/benchpress.py`
* **改動內容**：
  在 `predict.py` 中新增了針對 1D array 的 `normalize_to_neg1_1()` 函數，並在 `benchpress.py` 產生資料集與推論時，將速度 (v1)、加速度 (v2)、變化率 (vr)、Z-score (z) 這四個變異特徵統一包裝，強制壓縮到 `[-1, 1]` 區間。
* **目的**：解決了原本臥推特徵數值發散的「Normalization 落差」，讓臥推的特徵範圍跟硬舉 (Deadlift) 完全對齊一致。

### 2. 加長 Early Stopping 的耐心值 (Patience)
* **涉及檔案**：`PatchTST_train.py`
* **改動內容**：
  將 `train_model()` 函數中的 `patience` 參數從原本的 `8` 提高到 **`40`**。
* **目的**：由於時間序列模型加上 Learning Rate Warmup，在前中期的驗證分數 (Val F1) 會劇烈震盪。加長耐心值可避免模型在尚未完全收斂、或正準備進步時就被意外腰斬提早結束。

### 3. 實作訓練期的動態資料擴增 (Data Augmentation)
* **涉及檔案**：`dataset/dataset.py`
* **改動內容**：
  在 `Datasubset` 類別中，針對訓練集 (`transform=True`) 加入了每次 Epoch 取資料時的動態干擾機制：
  1. **Jittering (加入雜訊)**：50% 機率給骨架特徵加上微小的常態分佈雜訊 (標準差 0.02)。
  2. **Scaling (特徵縮放)**：50% 機率將骨架特徵隨機乘上 `0.95 ~ 1.05` 的縮放比例，模擬受測者動作幅度的微小差異。
  3. **Clipping (數值裁切)**：最後將數值鎖定在 `[-1.2, 1.2]` 以內，防止雜訊讓數值爆表。
* **目的**：為了對抗 Set Split 帶來的嚴重過擬合 (Overfitting) 問題。減緩模型死背絕對座標的現象，逼迫模型去學習真正的動作軌跡變化。
