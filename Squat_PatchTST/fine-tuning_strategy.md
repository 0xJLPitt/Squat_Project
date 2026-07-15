# Fine-Tuning 策略規劃書

> 適用專案：`fitness_action_recognition` PatchTST 多標籤動作錯誤分類器  
> 資料類型：時間序列骨骼角度特徵 (shape: `[T=110, C=40]`)

---

## 目錄

1. [整體策略概覽](#1-整體策略概覽)
2. [Augmentation 組合策略（最核心）](#2-augmentation-組合策略)
3. [Optimizer 調整方向](#3-optimizer-調整方向)
4. [Scheduler 調整方向](#4-scheduler-調整方向)
5. [Loss Function 調整方向](#5-loss-function-調整方向)
6. [模型超參數調整](#6-模型超參數調整)
7. [資料切分策略調整](#7-資料切分策略調整)
8. [訓練流程超參數](#8-訓練流程超參數)
9. [建議實驗順序](#9-建議實驗順序)
10. [搜尋空間總表](#10-搜尋空間總表)

---

## 1. 整體策略概覽

目前模型的主要潛在問題：
- **Augmentation 完全缺失**：`apply_augmentation()` 為空函數，訓練集沒有任何資料增強。
- **Transformer 較淺**：2 層 Encoder，可能欠擬合複雜時序模式。
- **Classifier 過於簡單**：單層 Linear，缺乏中間非線性投影。
- **pos_weight 未截斷**：稀有類別的 pos_weight 可能過大，導致梯度不穩定。

Fine-Tuning 優先次序（從影響力大到小）：

```
Augmentation > Loss > Optimizer/Scheduler > 模型超參數 > 資料切分
```

---

## 2. Augmentation 組合策略

> **執行位置**：`Datasubset.__getitem__()` 中，`transform=True` 的 Branch 下，對已正規化的 `[T, C]` 張量做即時 On-the-fly 增強。

### 2.1 單一增強方法定義

#### A. Gaussian Jitter（加噪）
對每個時間步加入微小高斯雜訊，模擬感測誤差。
```python
def jitter(x, sigma=0.02):
    return x + torch.randn_like(x) * sigma
```
- `sigma` 搜尋範圍：`[0.01, 0.02, 0.05]`

#### B. Time Shift（時間平移）
隨機沿時間軸平移序列，超出邊界用邊緣值填補。
```python
def time_shift(x, max_shift=10):
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(x, shifts=shift, dims=0)
```
- `max_shift` 搜尋範圍：`[5, 10, 15]`

#### C. Time Warp（時間彎曲）
對時間軸進行非線性拉伸與壓縮，模擬動作速度變化。
```python
def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    T = x.shape[0]
    orig = np.arange(T)
    cx = np.linspace(0, T - 1, knot + 2)
    cy = cx + np.random.randn(len(cx)) * sigma * T
    cy[0], cy[-1] = 0, T - 1
    cs = CubicSpline(cx, cy)
    warped = np.clip(cs(orig), 0, T - 1)
    new_x = np.stack([np.interp(warped, orig, x[:, c].numpy()) for c in range(x.shape[1])], axis=1)
    return torch.tensor(new_x, dtype=x.dtype)
```
- `sigma` 搜尋範圍：`[0.1, 0.2, 0.3]`

#### D. Magnitude Scale（幅度縮放）
對整個序列乘上接近 1.0 的隨機因子。
```python
def magnitude_scale(x, sigma=0.1):
    scale = 1.0 + torch.randn(1) * sigma
    return x * scale
```
- `sigma` 搜尋範圍：`[0.05, 0.1, 0.2]`

#### E. Channel-wise Dropout（通道遮蔽）
隨機將特定特徵通道清零，模擬骨骼點遮蔽。
```python
def channel_dropout(x, p=0.1):
    mask = (torch.rand(x.shape[1]) > p).float()
    return x * mask.unsqueeze(0)
```
- `p` 搜尋範圍：`[0.05, 0.1, 0.2]`
- **適合對象**：Channel Independence 的 PatchTST 特別受益

#### F. Window Crop + Resize（視窗裁切）
隨機裁切一段子序列，再線性插值回原長度 T。
```python
def window_crop(x, crop_ratio=0.9):
    T = x.shape[0]
    crop_len = int(T * crop_ratio)
    start = random.randint(0, T - crop_len)
    cropped = x[start:start + crop_len, :]
    import torch.nn.functional as F
    return F.interpolate(cropped.T.unsqueeze(0), size=T, mode='linear', align_corners=False).squeeze(0).T
```
- `crop_ratio` 搜尋範圍：`[0.8, 0.85, 0.9, 0.95]`

#### G. Mixup（樣本混合）
對兩個訓練樣本做線性混合，Soft Label 可降低過擬合。
```python
def mixup(x1, y1, x2, y2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y
```
- `alpha` 搜尋範圍：`[0.2, 0.4, 0.8]`
- **注意**：需要在 DataLoader 的 collate_fn 或 train loop 中實作

#### H. Time Flip（時間翻轉）
翻轉時間軸，模擬動作的時序鏡像。
```python
def time_flip(x):
    return torch.flip(x, dims=[0])
```
- 使用機率：`p=0.5`（50% 機率觸發）

---

### 2.2 組合型 Augmentation 策略

#### 策略 Combo-A：基礎雜訊組合（低風險，推薦第一個跑）
```
Jitter(σ=0.02) → Magnitude Scale(σ=0.1) → Time Shift(shift=10)
```

#### 策略 Combo-B：時間扭曲組合（中等強度）
增強模型對動作速度不均勻的魯棒性。
```
Time Warp(σ=0.2) → Jitter(σ=0.02) → Channel Dropout(p=0.1)
```

#### 策略 Combo-C：裁切組合（強資料增強）
模擬部分動作缺失，讓模型更依賴關鍵時間段。
```
Window Crop(ratio=0.9) → Jitter(σ=0.01) → Magnitude Scale(σ=0.05)
```

#### 策略 Combo-D：全套強增強（最大多樣性）
每個 Augmentation 以機率 p 獨立觸發。
```
50% Time Warp → 50% Window Crop → 80% Jitter → 30% Channel Dropout → 50% Time Shift
```

#### 策略 Combo-E：Mixup + 基礎雜訊
```
Mixup(α=0.4) → Jitter(σ=0.02) → Magnitude Scale(σ=0.1)
```

#### 推薦實驗順序

| 實驗 | 組合 | 預期效果 |
|------|------|----------|
| Exp-1 | Combo-A | Baseline 增強，驗證增強是否有幫助 |
| Exp-2 | Combo-B | 提升時序魯棒性 |
| Exp-3 | Combo-C | 處理資料量少的問題 |
| Exp-4 | Combo-D | 最大正則化效果 |
| Exp-5 | Combo-E | 嘗試 Soft Label 混合效果 |

---

## 3. Optimizer 調整方向

### 目前設定
```python
optim.Adam(lr=0.0003)
```

### 調整選項

#### 3.1 換用 AdamW（推薦優先嘗試）
AdamW 將 weight decay 從梯度更新中獨立出來，對 Transformer 效果更好。
```python
optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
```
- `lr` 搜尋範圍：`[1e-4, 3e-4, 5e-4, 1e-3]`
- `weight_decay` 搜尋範圍：`[1e-4, 1e-3, 5e-3, 1e-2]`

#### 3.2 換用 SGD + Momentum
```python
optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
```

#### 3.3 Gradient Clipping 調整
目前固定 `max_norm=1.0`，可嘗試：
- `max_norm` 搜尋範圍：`[0.5, 1.0, 2.0, 5.0]`

#### 3.4 Layer-wise Learning Rate Decay（進階）
對 Transformer 層使用較小的 LR，對 Classifier 使用較大的 LR。
```python
optimizer = AdamW([
    {"params": model.transformer.parameters(), "lr": lr * 0.1},
    {"params": model.patch_embed.parameters(), "lr": lr * 0.5},
    {"params": model.classifier.parameters(), "lr": lr},
], weight_decay=wd)
```

---

## 4. Scheduler 調整方向

### 目前設定
```python
# Warmup (5 epochs linear) + Cosine Decay to 0
# max_epochs = 100, min_lr_ratio = 0.0
```

### 調整選項

#### 4.1 調整 Warmup 長度與最小 LR
- `warmup_epochs` 搜尋範圍：`[3, 5, 10, 15]`
- `min_lr_ratio` 搜尋範圍：`[0.0, 0.01, 0.05, 0.1]`（不要衰到 0）

#### 4.2 換用 CosineAnnealingWarmRestarts（周期重啟）
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)
```
- `T_0` 搜尋範圍：`[10, 20, 30]`
- `T_mult` 搜尋範圍：`[1, 2]`

#### 4.3 換用 OneCycleLR（三角形 LR）
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3,
    steps_per_epoch=len(train_loader), epochs=num_epochs,
    pct_start=0.3
)
```
> **注意**：OneCycleLR 需在 batch 級別呼叫 `scheduler.step()`，而非 epoch 級別。

#### 4.4 max_epochs 對齊問題（必修）
目前 Warmup Cosine 的 `max_epochs=100` 但 training 跑 `num_epochs=150`，  
建議統一設定：`max_epochs = num_epochs`

---

## 5. Loss Function 調整方向

### 目前設定
```python
BCEWithLogitsLoss(pos_weight = neg_counts / pos_counts)
```

### 調整選項

#### 5.1 pos_weight 上限截斷（必做修正）
防止稀有類別的 pos_weight 過大導致梯度不穩定。
```python
pos_weight = torch.clamp(neg_counts / pos_counts, min=1.0, max=10.0)
```
- `max_clip` 搜尋範圍：`[5.0, 10.0, 20.0]`

#### 5.2 Focal Loss（應對嚴重不平衡）
對被錯誤分類的困難樣本給予更大的 Loss 權重。
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()
```
- `gamma` 搜尋範圍：`[1.0, 2.0, 3.0]`

#### 5.3 Label Smoothing BCE
緩解模型過度自信。
```python
def bce_label_smooth(pred, target, smooth=0.05):
    target = target * (1 - smooth) + smooth * 0.5
    return F.binary_cross_entropy_with_logits(pred, target)
```
- `smooth` 搜尋範圍：`[0.01, 0.05, 0.1]`

#### 5.4 Asymmetric Loss（ASL，最適合多標籤）
對正負樣本設定不同的 gamma，比 Focal Loss 更適合多標籤問題。
```python
# gamma_neg > gamma_pos：強化對容易分類的負樣本的抑制
ASLLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
```

---

## 6. 模型超參數調整

### 目前設定
```python
PatchTSTClassifier(
    input_dim=40, num_classes=4, input_len=110,
    patch_len=10, embed_dim=256,
    num_heads=4, num_layers=2,
    dropout=0.3, stride=1
)
```

### 6.1 Patch 相關

| 參數 | 目前值 | 搜尋範圍 | 說明 |
|------|--------|----------|------|
| `patch_len` | 10 | `[5, 8, 10, 15, 20]` | 較短→細節，較長→整體模式 |
| `stride` | 1 | `[1, 2, 5]` | stride > 1 可減少 patches 數量 |

### 6.2 Transformer 相關

| 參數 | 目前值 | 搜尋範圍 | 說明 |
|------|--------|----------|------|
| `embed_dim` | 256 | `[64, 128, 256, 512]` | 決定模型容量 |
| `num_heads` | 4 | `[2, 4, 8]` | 須整除 embed_dim |
| `num_layers` | 2 | `[1, 2, 3, 4]` | 層數影響表達能力 |
| `dropout` | 0.3 | `[0.1, 0.2, 0.3, 0.5]` | 正則化強度 |

### 6.3 Readout 方式

| 方式 | 現況 | 說明 |
|------|------|------|
| Mean Pooling | ✅ 目前使用 | 對所有 patch 取平均，穩健 |
| CLS Token | 可嘗試 | 在序列前加可學習 [CLS] token |
| Attention Pooling | 可嘗試 | 用 1 層 attention 加權聚合 |
| Last Patch | code 中有，被 comment 掉 | 原論文做法 |

### 6.4 Classifier Head 增強

```python
# 目前（單層）
nn.LayerNorm → Dropout → Linear(C*E, num_classes)

# 建議：加入中間非線性層
nn.LayerNorm → Dropout → Linear(C*E, 128) → GELU → Dropout → Linear(128, num_classes)
```
- 中間維度搜尋範圍：`[64, 128, 256]`

---

## 7. 資料切分策略調整

### 7.1 Subject-isolated 模式（推薦作為主要評估方法）
`--subject_isolated` 確保不同受試者的資料不會跨集，更符合真實應用場景的評估方式。

### 7.2 資料比例微調

| 設定 | Train | Val | Test |
|------|-------|-----|------|
| 目前 | 75% | 15% | 10% |
| 可選 1 | 70% | 15% | 15% |
| 可選 2 | 80% | 10% | 10% |

---

## 8. 訓練流程超參數

| 參數 | 目前值 | 搜尋範圍 |
|------|--------|----------|
| `batch_size` | 16 | `[8, 16, 32, 64]` |
| `num_epochs` | 150 | `[100, 150, 200]` |
| `patience` | 8 | `[5, 8, 10, 15]` |
| `prediction_threshold` | 0.5 (all class) | per-class tuning |

### 8.1 每類別獨立 Threshold 搜尋（推薦）
目前所有類別統一使用 `threshold=0.5`，建議改為在 Validation Set 上搜尋各類別的最佳 threshold：
```python
best_thresholds = []
for c in range(num_classes):
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.3, 0.7, 0.05):
        preds_c = (probs[:, c] > t).int()
        f = f1_score(y_true[:, c], preds_c)
        if f > best_f:
            best_f, best_t = f, t
    best_thresholds.append(best_t)
```

---

## 9. 建議實驗順序

```
Step 1: pos_weight clamp + 換 AdamW（快速修正現有問題）
    ↓
Step 2: 加入 Augmentation Combo-A（Baseline 增強，驗證效果）
    ↓
Step 3: 換用 Focal Loss 或 ASL（改善 Loss 設計）
    ↓
Step 4: 嘗試 Augmentation Combo-B / C / D（尋找最佳增強策略）
    ↓
Step 5: 調整 Scheduler（Warmup 長度、min_lr_ratio 對齊）
    ↓
Step 6: 調整模型超參數（num_layers、patch_len、Classifier Head）
    ↓
Step 7: Per-class threshold 搜尋（後處理最佳化）
    ↓
Step 8: 整合最佳超參數，跑 subject_isolated 全套實驗
```

---

## 10. 搜尋空間總表

| 類別 | 參數 | 搜尋範圍 | 優先級 |
|------|------|----------|--------|
| Augmentation | 組合策略 | Combo-A ~ E | ⭐⭐⭐⭐⭐ |
| Augmentation | Jitter σ | 0.01, 0.02, 0.05 | ⭐⭐⭐⭐ |
| Augmentation | TimeWarp σ | 0.1, 0.2, 0.3 | ⭐⭐⭐⭐ |
| Augmentation | CropRatio | 0.8, 0.85, 0.9 | ⭐⭐⭐ |
| Augmentation | Mixup α | 0.2, 0.4, 0.8 | ⭐⭐⭐ |
| Loss | pos_weight clamp | 5.0, 10.0, 20.0 | ⭐⭐⭐⭐⭐ |
| Loss | Focal γ | 1.0, 2.0, 3.0 | ⭐⭐⭐⭐ |
| Loss | Label Smooth | 0.01, 0.05, 0.1 | ⭐⭐⭐ |
| Optimizer | Type | Adam, AdamW, SGD | ⭐⭐⭐⭐ |
| Optimizer | LR | 1e-4, 3e-4, 5e-4, 1e-3 | ⭐⭐⭐⭐ |
| Optimizer | Weight Decay | 1e-4, 1e-3, 1e-2 | ⭐⭐⭐ |
| Optimizer | Grad Clip | 0.5, 1.0, 2.0 | ⭐⭐ |
| Scheduler | Warmup epochs | 3, 5, 10 | ⭐⭐⭐ |
| Scheduler | min_lr_ratio | 0.0, 0.01, 0.05 | ⭐⭐⭐ |
| Scheduler | Type | Cosine, CosineWR, OneCycle | ⭐⭐⭐ |
| Model | patch_len | 5, 8, 10, 15 | ⭐⭐⭐⭐ |
| Model | embed_dim | 64, 128, 256, 512 | ⭐⭐⭐ |
| Model | num_layers | 1, 2, 3, 4 | ⭐⭐⭐ |
| Model | dropout | 0.1, 0.2, 0.3, 0.5 | ⭐⭐⭐ |
| Model | Readout | MeanPool, CLSToken, AttPool | ⭐⭐ |
| Training | batch_size | 8, 16, 32 | ⭐⭐⭐ |
| Training | patience | 5, 8, 10, 15 | ⭐⭐ |
| Training | threshold | per-class tuning | ⭐⭐⭐⭐ |
| Data | split 比例 | 75:15:10, 70:15:15 | ⭐⭐ |
| Data | subject_isolated | True / False | ⭐⭐⭐ |

---

## 附錄：Augmentation 實作建議

在 `dataset/dataset.py` 的 `Datasubset.__getitem__` 中加入增強邏輯：

```python
class Datasubset(Dataset):
    def __init__(self, dataset, indices, transform=False, aug_combo='A'):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.aug_combo = aug_combo

    def __getitem__(self, idx):
        x, y, true_idx = self.dataset[self.indices[idx]]
        if self.transform:
            x = self._augment(x)
        return x, y, true_idx

    def _augment(self, x):
        if self.aug_combo == 'A':
            x = jitter(x, sigma=0.02)
            x = magnitude_scale(x, sigma=0.1)
            x = time_shift(x, max_shift=10)
        elif self.aug_combo == 'B':
            x = time_warp(x, sigma=0.2)
            x = jitter(x, sigma=0.02)
            x = channel_dropout(x, p=0.1)
        elif self.aug_combo == 'C':
            x = window_crop(x, crop_ratio=0.9)
            x = jitter(x, sigma=0.01)
            x = magnitude_scale(x, sigma=0.05)
        elif self.aug_combo == 'D':
            if random.random() < 0.5: x = time_warp(x, sigma=0.2)
            if random.random() < 0.5: x = window_crop(x, crop_ratio=0.9)
            if random.random() < 0.8: x = jitter(x, sigma=0.02)
            if random.random() < 0.3: x = channel_dropout(x, p=0.1)
            if random.random() < 0.5: x = time_shift(x, max_shift=10)
        return x
```
