# Paper Reference

本文件整理與 `PatchTST + 骨骼時序特徵 + 多標籤分類` 相關的參考文獻，依照 Fine-Tuning 策略的各個面向分類。

---

## 一、模型架構

### [1] PatchTST（必讀）

- **Title**: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
- **Authors**: Nie et al.
- **Venue**: ICLR 2023
- **Link**: https://arxiv.org/abs/2211.14730
- **對應策略**: 模型架構 — 理解 PatchTST 的 Channel Independence 設計、Patching 的好處
- **摘要重點**:
  - 提出 Channel Independence (CI) 模式，每個 channel 獨立通過 Transformer
  - Patching 將時序切成局部片段，減少 attention 計算量並擷取局部時序模式
  - 相比複雜的多變量模型，CI 模式反而更穩健、泛化能力更強
  - 你目前的實作完全遵循此設計：`x → (B*C, T, 1) → Patch Embedding → Transformer`

---

## 二、Data Augmentation（時序增強）

### [2] 時序 Augmentation 全面調查（必讀）

- **Title**: Data Augmentation of Time Series: A Survey
- **Authors**: Wen et al.
- **Year**: 2021
- **Link**: https://arxiv.org/abs/2002.12478
- **對應策略**: Augmentation — 所有方法的理論總覽
- **摘要重點**:
  - 系統性整理 Jitter、Scaling、Time Warp、Permutation、Mixup 等所有主流增強方法
  - 提供各方法在不同時序任務上的適用性分析
  - 對應策略書中 Combo-A~E 所有增強方法的理論依據

### [3] Time Warp 的原始出處

- **Title**: Data Augmentation Using Synthetic Data for Time Series Classification with Deep Residual Networks
- **Authors**: Le Guennec et al.
- **Venue**: ECML Workshop 2016
- **對應策略**: Augmentation — Time Warp (C)、Window Crop (F)
- **摘要重點**:
  - 提出 Window Slicing（對應 Window Crop）
  - 提出 Window Warping（對應 Time Warp）
  - 證明這兩種方法在時序分類上有顯著效果

### [4] TS2Vec（Contrastive Learning + 時序 Augmentation）

- **Title**: TS2Vec: Towards Universal Representation of Time Series
- **Authors**: Yue et al.
- **Venue**: AAAI 2022
- **Link**: https://arxiv.org/abs/2106.10466
- **對應策略**: Augmentation — Temporal Cropping 設計思路
- **摘要重點**:
  - 以對比學習（Contrastive Learning）方式學習通用時序表徵
  - 利用 Temporal Cropping + Instance-wise Cropping 作為正樣本對
  - Window Crop 增強方法的進階理論基礎

---

## 三、Loss Function

### [5] Focal Loss（必讀）

- **Title**: Focal Loss for Dense Object Detection
- **Authors**: Lin et al.
- **Venue**: ICCV 2017
- **Link**: https://arxiv.org/abs/1708.02002
- **對應策略**: Loss Function — 不平衡分類問題的核心解法
- **摘要重點**:
  - 提出 Focal Loss：`FL(p_t) = -(1 - p_t)^γ * log(p_t)`
  - 對「容易分類的樣本」降低 loss 比重，讓模型專注於難分類樣本
  - `γ=2` 為常用設定，搜尋範圍建議 `[1.0, 2.0, 3.0]`
  - 可與 `pos_weight` 結合使用，同時處理類別不平衡

### [6] Asymmetric Loss / ASL（多標籤場景最推薦）

- **Title**: Asymmetric Loss For Multi-Label Classification
- **Authors**: Ben-Baruch et al.
- **Venue**: ICCV 2021
- **Link**: https://arxiv.org/abs/2009.14119
- **對應策略**: Loss Function — 多標籤分類的專用 Loss
- **摘要重點**:
  - 針對多標籤分類設計，對正負樣本設定**不同的 gamma**
  - `gamma_neg > gamma_pos`：強化對「容易分類的負樣本」的抑制，減少其佔據梯度比重
  - 同時包含 Probability Shift（機率截斷）機制，進一步穩定訓練
  - 比 Focal Loss 更適合多標籤不平衡場景（你的 4-class multi-label 問題）
  - 建議參數：`gamma_neg=4, gamma_pos=1, clip=0.05`

---

## 四、Optimizer / 訓練技巧

### [7] AdamW

- **Title**: Decoupled Weight Decay Regularization
- **Authors**: Loshchilov & Hutter
- **Venue**: ICLR 2019
- **Link**: https://arxiv.org/abs/1711.05101
- **對應策略**: Optimizer — 換用 AdamW 的理論依據
- **摘要重點**:
  - 說明標準 Adam + L2 Regularization ≠ AdamW
  - Adam 的 L2 會被 adaptive LR 放大，效果等同於弱化的 weight decay
  - AdamW 將 weight decay 從梯度更新獨立出來，對 Transformer 類模型效果更好
  - 幾乎所有現代 Transformer（BERT、ViT、PatchTST）訓練都使用 AdamW

### [8] Cosine Annealing Warm Restarts (SGDR)

- **Title**: SGDR: Stochastic Gradient Descent with Warm Restarts
- **Authors**: Loshchilov & Hutter
- **Venue**: ICLR 2017
- **Link**: https://arxiv.org/abs/1608.03983
- **對應策略**: Scheduler — CosineAnnealingWarmRestarts 的原始論文
- **摘要重點**:
  - 提出 Cosine Annealing with Warm Restarts：LR 衰減到底後重新升高
  - 幫助模型跳出局部最佳解，提升泛化能力
  - `T_0`（第一週期長度）與 `T_mult`（週期倍增因子）是關鍵超參數

---

## 五、骨骼序列 / 動作識別

### [9] 骨骼 Graph Neural Network

- **Title**: Skeleton-Based Action Recognition with Directed Graph Neural Networks
- **Authors**: Shi et al.
- **Venue**: CVPR 2019
- **Link**: https://arxiv.org/abs/1904.12659
- **對應策略**: 模型架構參考 — 骨骼關節特徵的建模方式
- **摘要重點**:
  - 以有向圖建模骨骼關節間的空間關係
  - 與你用 joint angle 特徵的邏輯類似（都是以關節空間關係作為核心特徵）
  - 可作為設計關節特徵組合的靈感來源

---

## 六、後處理優化

### [10] 多標籤 Threshold 優化

- **Title**: Threshold Optimisation for Multi-label Classifiers
- **Authors**: Pillai et al.
- **Venue**: Pattern Recognition 2013
- **對應策略**: 後處理 — Per-class threshold 搜尋的理論基礎
- **摘要重點**:
  - 在 Validation Set 上針對每個類別獨立搜尋最佳分類門檻
  - 比固定使用 `threshold=0.5` 可顯著提升多標籤的 F1-score
  - 特別在正負樣本不平衡的類別上效果最明顯

---

## 閱讀優先序推薦

| 優先 | Paper | 對應問題 |
|------|-------|---------|
| 1️⃣ | **[1] PatchTST** | 了解你正在用的模型設計原理 |
| 2️⃣ | **[6] ASL** | 最適合多標籤不平衡問題的 Loss |
| 3️⃣ | **[2] 時序 Aug Survey** | Augmentation 全方位參考 |
| 4️⃣ | **[5] Focal Loss** | 不平衡分類的核心知識 |
| 5️⃣ | **[7] AdamW** | 換 Optimizer 的理論依據 |
| 6️⃣ | **[8] SGDR** | Cosine Scheduler 設計理解 |
| 7️⃣ | **[4] TS2Vec** | 進階 Augmentation 思路 |
| 8️⃣ | **[3] Time Warp** | Window Crop / Time Warp 來源 |
| 9️⃣ | **[9] ST-GCN** | 骨骼特徵設計靈感 |
| 🔟 | **[10] Threshold Opt.** | Per-class Threshold 後處理依據 |
