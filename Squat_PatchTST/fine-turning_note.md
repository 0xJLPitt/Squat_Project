# 訓練參數紀錄 (Training Parameters Note)

本文件紀錄了 `fitness_action_recognition` 專案中除了模型架構以外的詳細訓練參數。

## 1. Data Split 比例
- **比例**: 75% 訓練集 (Training) : 15% 驗證集 (Validation) : 10% 測試集 (Testing)
- **切分方式**:
  - 支援 Subject-isolated 切分 (`--subject_isolated`)，確保各受試者的資料不會同時出現在不同集合中，並根據類別盡量分配達到全局 75:15:10 的比例。
  - 若非 Subject-isolated，則採用隨機切分（提供 6 組 Seeds：`[42, 2023, 7, 88, 100, 999]`），按照總資料筆數的 `0.75` 及 `0.90` 切點依序分為 Train/Val/Test。

## 2. Loss Function
- **使用函數**: `BCEWithLogitsLoss` (Binary Cross Entropy with Logits)
- **Class Imbalance 處理**: 針對多標籤的正負樣本不平衡，在各 Fold 訓練前計算訓練集中各類別的正負比例 `pos_weight` (負樣本數量 / 正樣本數量)，並套用於 Loss 函數進行加權平衡。

## 3. Scheduler
- **使用函數**: 自訂的 Cosine Annealing with Linear Warmup (`LambdaLR`)
- **參數設定**:
  - `warmup_epochs`: 5
  - `max_epochs`: 100
  - `min_lr_ratio`: 0.0
- **運作機制**: 訓練前 5 個 Epochs 進行線性 Warmup (從 0 增長至 Base LR)，接著使用 Cosine Decay 在剩下的 Epochs 中平滑遞減至 0。

## 4. Optimizer
- **使用優化器**: `Adam`
- **Learning Rate (Base LR)**: 0.0003
- **Gradient Clipping**: 啟用 `clip_grad_norm_`，設定 `max_norm=1.0` 以限制梯度最大範數，防止梯度爆炸。

## 5. Data Augmentation (資料增強)
- **目前設定**: 無額外資料增強實作。
- **備註**: 儘管在建立 Training Datasubset 時傳入了 `transform=True` 參數，但在目前的 `dataset.py` 中，並未於 `__getitem__` 階段實作即時的數據轉換與增強操作 (如加噪、時間平移等)。

## 6. 其他重要訓練參數與設定
- **Batch Size**: 16
- **Total Epochs**: 150 (由 `train_model` 的 `num_epochs=150` 預設值控制)
- **Early Stopping Patience**: 8 (若連續 8 個 Epoch 的 Validation F1-score 皆未創高，即提早停止訓練)
- **Evaluation Metric**: 主要評估指標為 `Macro F1-score`，同時計算整體 Accuracy 與個別類別的 F1-score。
- **模型儲存策略**: 訓練過程會追蹤每個 Epoch 的 Validation F1-score，並儲存表現最好 (Best Val F1) 時的模型權重。
