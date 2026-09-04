# Compare Wrist vs Barbell (手腕中點 vs 槓鈴軌跡比對系統)

本模組專門用於評估 **手腕中點（3D Wrist Midpoint）** 作為深蹲時槓鈴代理軌跡的運動學有效性，並提供：
1. **單目 3D 姿態提升比對（MotionAGFormer 3D Pose Lifting）** $\to$ 輸出儲存於 `AGFormer/`
2. **多視角空間三角測量比對（Chessboard Multi-View Triangulation）** $\to$ 輸出儲存於 `Checkerboard3d/`

---

## 🎯 核心原理與運動學定義

### 1. 手腕中點估計 (3D Wrist Midpoint)
提取 3D 人體骨架之左手腕與右手腕座標取幾何中點：
$$P_{\text{wrist\_mid}}(t) = \frac{P_{\text{L\_Wrist}}(t) + P_{\text{R\_Wrist}}(t)}{2} = \big(X_{\text{wrist}}(t),\, Y_{\text{wrist}}(t),\, Z_{\text{wrist}}(t)\big)$$
* **COCO 格式**：Joint 9 (L_Wrist) 與 Joint 10 (R_Wrist)
* **Human3.6M 格式**：Joint 13 (L_Wrist) 與 Joint 16 (R_Wrist)

### 2. 槓鈴真實位置基準 (Ground Truth Barbell)
由側向攝影機（`RU.avi`）之 YOLO 槓鈴偵測模型輸出像素座標 $(X_{\text{bar}}, Y_{\text{bar}})$，垂直高度以 Inverted $Y$ 表示（向上為正）。

---

## 📐 重要核心概念：相機俯角與世界重力軸校正 (Camera Pitch & World Gravity Alignment)

### 1. 幾何問題背景：為什麼未校正前軌跡會「傾斜」？
在多鏡頭或單鏡頭架設時，腳架上的攝影機通常**不是絕對水平放置**，而是帶有約 $10^\circ \sim 20^\circ$ 的**向下俯角（Pitch Down）** 朝向受試者：
* **相機座標系特性**：
  * 相機的 $+Y_{\text{cam}}$ 是沿感光元件向下（朝向前下方）。
  * 相機的 $+Z_{\text{cam}}$ 是光軸深度（朝向深度前方）。
* **投影分解效應**：
  當深蹲者沿著地球真實重力方向純垂直下蹲 $40\text{ cm}$（$\Delta H_{\text{world}} = -40\text{ cm}$）時，在帶有俯角 $\theta$ 的相機局部坐標系中，會被自動分解為：
  $$\Delta Y_{\text{cam}} = \Delta H_{\text{world}} \times \cos(\theta)$$
  $$\Delta Z_{\text{cam}} = \Delta H_{\text{world}} \times \sin(\theta)$$
  例如在 $\theta = 15^\circ$ 的俯角下，垂直下潛 $40\text{ cm}$ 會在相機坐標系的深度軸分解出高達 **$40 \times \sin(15^\circ) \approx 10.4\text{ cm}$ 的深度位移假象**。這會導致未旋轉的 3D 軌跡圖看似手腕斜向前方移動。

```
       相機光軸 (帶俯角 θ)
         \
          \ 
           \ 
            ● 人體位置
           /|
          / | 
   深度軸 /  | 地球真實重力軸 (垂直下潛 40cm)
   ΔZ假象  | 
          \|
```

### 2. 演算法演進：由「骨骼直立連線」升級至「方案 A：運動學自校正（深蹲下潛主軸）」

#### (1) 為什麼廢棄「肩膀 $\to$ 腳踝連線」直立假設？
在人體生物力學中，背槓站立準備時：
* **足中重心平衡（Mid-foot COM）**：背負槓鈴站立時，為平衡合重心，髖關節微屈，軀幹通常呈現約 $5^\circ \sim 15^\circ$ 的自然前傾角（Torso Incline Angle）。
* **解剖關節非共線**：腳踝關節位於足掌後側，若將「雙腳踝中點 $\to$ 雙肩膀中點」強行扳直為 $90^\circ$ 垂直，會**誤將人體生理前傾當成相機俯角進行補償**，反而引入人為系統性偏差（Over-correction）。

#### (2) 方案 A 運動學自校正（Kinematic Self-Calibration using Squat Descent Principal Axis）
深蹲下潛（Eccentric）與蹬伸（Concentric）的過程受地心引力約束最嚴格，因此**運動軌跡的主位移方向即為最客觀的物理重力垂線**：

1. **提取每次深蹲的下潛位移向量（Descent Vectors）**：
   對於影片中的 $K$ 次深蹲，提取每一下從「站立最高點 $t_{\text{start}}$」至「蹲底最低點 $t_{\text{bottom}}$」的 3D 手腕位移：
   $$\vec{d}_k = P_{\text{wrist}}(t_{\text{start}}^{(k)}) - P_{\text{wrist}}(t_{\text{bottom}}^{(k)})$$
2. **計算平均下潛直立單位向量（Mean Descent Upright Vector）**：
   $$\hat{v}_{\text{descent}} = \frac{\sum_{k=1}^K \frac{\vec{d}_k}{\|\vec{d}_k\|}}{\left\| \sum_{k=1}^K \frac{\vec{d}_k}{\|\vec{d}_k\|} \right\|}$$
   *(若缺乏分段標記，則自適應退化為對活躍區間進行 PCA 主成分分析，取第一主成分軸 PC1)*。
3. **Rodrigues 向量重力對齊矩陣**：
   定義目標世界垂直軸 $\hat{g} = [0, 0, 1]^T$（或相機座標系 $[0, -1, 0]^T$），計算旋轉軸 $\vec{v} = \hat{v}_{\text{descent}} \times \hat{g}$ 與夾角餘弦 $c = \hat{v}_{\text{descent}} \cdot \hat{g}$：
   $$R_{\text{align}} = I + [v]_\times + [v]_\times^2 \left( \frac{1 - c}{\|\vec{v}\|^2} \right)$$
4. **座標旋轉至世界重力系**：
   $$P_{\text{world}}(t) = R_{\text{align}} \cdot P_{\text{cam}}(t)$$
   經由方案 A 自校正後，垂直下潛（$Z_{\text{world}}$）與前後晃動（$X_{\text{world}}$）完全正交解耦，且**完全免疫於受試者背槓站立時的前傾姿勢**！

---

## 🧬 生物力學解析：手腕與槓鈴的微幅相對位移

在深蹲過程中，雖然手握槓鈴未主動移動手腕，但仍存在約 $\pm 1 \sim 2\text{ cm}$ 的自然微動，原因包含：
1. **軀幹前傾角（Torso Incline Angle）**：深蹲下潛至最低點時，軀幹為平衡重心會前傾 $30^\circ \sim 45^\circ$。手腕關節中心與槓鈴軸心存在幾公分的剛性力臂，上半身前傾時會帶動手腕中心繞槓鈴微幅旋轉。
2. **2D 側視投影透視**：側向相機若未完全垂直於槓鈴中央軸線，2D 像素在頂部與底部亦有些許透視微差。

---

## 🚀 執行指令與工作流程

### 1. 單目 AGFormer 3D 手腕 vs 2D 槓鈴比對
```powershell
mamba run -n hw1 python run_compare_AGFormer_wrist_bar.py --target_dir "E:\squat\squat_dataset\S001_Pitt\session02\recording_20260207_163410"
```
* **產出目錄**：儲存於目標資料夾下的 `AGFormer/`

### 2. 多視角棋盤格 3D 手腕 vs 2D 槓鈴比對
```powershell
mamba run -n hw1 python run_compare_Checkerboard_wrist_bar.py --target_dir "E:\squat\squat_dataset\S001_Pitt\session02\recording_20260207_163410"
```
* **產出目錄**：儲存於目標資料夾下的 `Checkerboard3d/`

---

## 📊 產出成果與圖表清單

| 檔案名稱 | 說明 |
| :--- | :--- |
| `keypoints_3d.npz` / `calibrated_keypoints_3d.npz` | 重建之完整 3D 骨架關節點座標序列陣列 |
| `wrist_vs_bar_comparison_report.csv` | 10 次深蹲之相關係數、最低點影格、時間差（ms）與下潛深度（ROM）表 |
| `compare_*_wrist_vs_bar_waveforms.png` | 10 次深蹲之垂直波形與波谷最低點時間對齊圖（0ms 誤差檢定） |
| `compare_*_wrist_vs_bar_reps_1to1_cm.png` | **1:1 真實物理公分比例尺** 之單次深蹲矢狀面路徑圖（下潛 $\sim 40\text{ cm}$ vs 晃動 $\sim 2\text{ cm}$） |
| `compare_*_wrist_vs_bar_all_reps_1to1_cm.png` | 1:1 真實公分比例尺之 10 次深蹲疊合圖 |
| `compare_*_wrist_vs_bar_trajectory.png` | 全時序垂直高度與前後位移波形圖（包含起槓與還槓過程） |
| `compare_*_wrist_vs_bar_reps_sideview.png` | 0~1 正規化側向 2D 軌跡圖 |
