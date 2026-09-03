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

### 2. 數學演算法：Rodrigues 向量重力旋轉對齊
為消除相機俯角造成的座標耦合，模組在計算運動學指標前會自動執行**世界重力軸校正**：

1. **提取人體站立直立向量（Upright Vector）**：
   在深蹲開始前的站立起始幀（Standing Frame），提取雙腳踝中點與雙肩膀中點：
   $$\vec{v}_{\text{ankle}} = \frac{P_{\text{L\_Ankle}} + P_{\text{R\_Ankle}}}{2},\quad \vec{v}_{\text{shoulder}} = \frac{P_{\text{L\_Shoulder}} + P_{\text{R\_Shoulder}}}{2}$$
   $$\vec{v}_{\text{upright}} = \vec{v}_{\text{shoulder}} - \vec{v}_{\text{ankle}}$$
2. **計算對齊旋轉矩陣（Rodrigues' Rotation Formula）**：
   定義單位起始向量 $\hat{u} = \frac{\vec{v}_{\text{upright}}}{\|\vec{v}_{\text{upright}}\|}$ 與目標世界重力垂直單位向量 $\hat{g} = [0, 0, 1]^T$。
   計算旋轉軸 $\vec{v} = \hat{u} \times \hat{g}$ 與餘弦值 $c = \hat{u} \cdot \hat{g}$，建構反對稱矩陣 $[v]_\times$：
   $$R_{\text{gravity}} = I + [v]_\times + [v]_\times^2 \left(\frac{1 - c}{\|\vec{v}\|^2}\right)$$
3. **旋轉轉換至世界重力座標系**：
   $$P_{\text{world}}(t) = R_{\text{gravity}} \cdot P_{\text{cam}}(t)$$
   經由世界重力對齊後，垂直位移（$Z_{\text{world}}$）與矢狀前後位移（$X_{\text{world}}$）完全正交解耦，手腕軌跡與槓鈴軌跡即呈現筆直純垂直重疊！

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
