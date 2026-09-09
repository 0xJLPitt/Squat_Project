# SBDFormer / AGFormer 資料集打包與訓練前處理計畫 (plan.md)

在電腦視覺與 3D 人體姿態估計（3D HPE）中，AGFormer（以及 MotionAGFormer、PoseFormer）最標準、最通用的輸入格式是 **NumPy 壓縮封裝檔（`.npz` 格式）**。

當透過多相機棋盤格三角測量（例如 `step8_triangulate_and_render_3d.py`）算出了 3D 骨架座標後，要餵給 AGFormer / SBDFormer 訓練，有兩種組織資料的方式。

**強烈推薦「方案 A（滑動窗口切片格式）」**，這是微調自訂資料集最直觀、最不容易報錯的作法！

---

## 方案 A（強烈推薦）：滑動窗口切片格式（Dataset Chunk Format）

這是直接配合 PyTorch `Dataset` 的格式，把每部影片切成固定長度（例如 $T = 81$ 或 $243$ 幀）的滑動樣本，打包成單一 `.npz` 檔：

### 1. `.npz` 內部包含的鍵值（Keys）與張量維度（Shapes）

```python
{
    # 1. 2D 輸入（來自單一主相機，如正前方或後斜方）
    "data_2d": np.ndarray,  # Shape: [N_samples, T, 17, 2] (正規化後的 2D 座標)
    
    # 2. 3D 真值（來自多視角三角測量重建的 Ground Truth）
    "data_3d": np.ndarray,  # Shape: [N_samples, T, 17, 3] (以骨盆為原點的 3D 座標)
    
    # 3. 2D 關鍵點信心值 (選配，來自 YOLO 的 conf)
    "confidence": np.ndarray,  # Shape: [N_samples, T, 17, 1]
    
    # 4. 樣本後設資料 (選配，記錄受試者與影片名稱)
    "metadata": list / dict
}
```

* **$N_{\text{samples}}$**：切片後的總樣本數（例如 96 部影片切出 4,000 個樣本）。
* **$T$**：時序窗口大小（AGFormer 預設通常是 **$T = 81$** 或 **$T = 243$**）。
* **$17$**：關節點數量。
* **$2$ / $3$**：2D 座標 $(u, v)$ / 3D 座標 $(X, Y, Z)$。

---

## 方案 B：仿照 Human3.6M 階層字典格式（Standard H36M Style）

如果 AGFormer 程式碼庫是完全原生、未修改 DataLoader 的版本，它通常要求吃兩個檔案：
1. `data_2d_custom.npz`
2. `data_3d_custom.npz`

其內部結構是巢狀字典（Nested Dictionary）：

```python
# data_3d_custom.npz
{
    "positions_3d": {
        "Subject_01": {
            "Bench_Correct_01": np.ndarray,  # Shape: [Total_Frames, 17, 3]
            "Bench_Error1_01":  np.ndarray,
            ...
        },
        "Subject_02": { ... }
    }
}

# data_2d_custom.npz
{
    "positions_2d": {
        "Subject_01": {
            "Bench_Correct_01": [
                np.ndarray  # Shape: [Total_Frames, 17, 2] (單視角 2D 影格)
            ]
        }
    },
    "metadata": {
        "layout_name": "coco",      # 或 "h36m"
        "num_joints": 17,
        "keypoints_symmetry": [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]] # 左右對稱索引
    }
}
```

---

## 🚨 製作資料時「最容易翻車的 4 大前處理細節」（務必遵守！）

把三角測量的 3D 點轉成 AGFormer 格式時，有 4 個幾何陷阱：

### 1. 3D 座標必須「骨盆中心化（Root-Relative / Zero-Centering）」！
* **致命錯誤**：如果直接把三角測量算出來的絕對 3D 座標 $(X, Y, Z)$ 丟給 AGFormer，模型會崩潰。
* **正確作法**：**每一幀，都必須把「骨盆中心點（Root/Pelvis）」設為 $(0, 0, 0)$**！
  $$P_{3D,\text{relative}} = P_{3D} - P_{3D,\text{pelvis}}$$
  *AGFormer 學的是「人體的相對姿態」，不是人在房間裡的絕對平移位置！*

### 2. 2D 座標必須「正規化（Normalization）」！
* **致命錯誤**：YOLO 輸出的 2D 點是像素座標（例如 $x \in [0, 1920], y \in [0, 1080]$）。
* **正確作法**：必須將像素座標轉換到歸一化區間 $[-1, 1]$（以畫面中心為原點，除以寬高的一半）：
  $$u_{\text{norm}} = \frac{u - W/2}{W/2}, \quad v_{\text{norm}} = \frac{v - H/2}{W/2}$$

### 3. 3D 物理單位的統一（公釐 mm vs 公尺 m）
* **Human3.6M 原生標準**：長度單位為 **公釐（mm）**（例如大腿長度約 $400$）。
* 三角測量如果輸出的是公尺（m，例如大腿 $0.4$），**請務必乘以 1000 轉為毫米（mm）**，否則 MPJPE 損失函數算出來的梯度會太小，模型學不動。

### 4. 關節點定義對齊（COCO 17 vs H36M 17）
* YOLO-pose 輸出的是 **COCO 17 個點**（包含眼睛、鼻子，但沒有骨盆中點）。
* 製作 3D 骨架時，通常以 **「左髖（點 11）與右髖（點 12）的中點」** 作為虛擬的第 0 點（Pelvis 骨盆中點）。請確認模型設定是吃 `coco_17` 還是 `h36m_17`。

---

## 隨插即用的資料打包 Python 腳本範例

當算好單部的 2D 點與 3D 點後，用這段腳本一鍵切片並儲存為 `.npz`：

```python
import numpy as np

def create_agformer_dataset(video_2d_list, video_3d_list, window_size=81, stride=27, output_path="sbd_custom_dataset.npz"):
    """
    video_2d_list: list of np.ndarray, 每個元素為單部影片的 2D 像素座標 [Frames, 17, 2]
    video_3d_list: list of np.ndarray, 每個元素為對應的 3D 真值座標 [Frames, 17, 3] (公釐)
    """
    all_2d_chunks = []
    all_3d_chunks = []
    
    W, H = 1920.0, 1080.0  # 影像解析度
    
    for k2d, k3d in zip(video_2d_list, video_3d_list):
        num_frames = len(k2d)
        if num_frames < window_size:
            continue
            
        # 1. 2D 歸一化到 [-1, 1]
        k2d_norm = k2d.copy()
        k2d_norm[:, :, 0] = (k2d_norm[:, :, 0] - W / 2) / (W / 2)
        k2d_norm[:, :, 1] = (k2d_norm[:, :, 1] - H / 2) / (W / 2)
        
        # 2. 3D 骨盆中心化 (以左髖 11 與右髖 12 的中點為原點)
        pelvis = (k3d[:, 11:12, :] + k3d[:, 12:13, :]) / 2.0
        k3d_relative = k3d - pelvis  # [Frames, 17, 3]
        
        # 3. 滑動窗口切片 (Sliding Window)
        for start in range(0, num_frames - window_size + 1, stride):
            end = start + window_size
            all_2d_chunks.append(k2d_norm[start:end])      # [81, 17, 2]
            all_3d_chunks.append(k3d_relative[start:end])  # [81, 17, 3]
            
    data_2d = np.array(all_2d_chunks, dtype=np.float32)  # [N, 81, 17, 2]
    data_3d = np.array(all_3d_chunks, dtype=np.float32)  # [N, 81, 17, 3]
    
    # 4. 存檔為 .npz
    np.savez_compressed(output_path, data_2d=data_2d, data_3d=data_3d)
    print(f"成功打包資料集！樣本總數: {len(data_2d)}, 已儲存至: {output_path}")

# 使用範例：
# create_agformer_dataset(my_2d_videos, my_3d_videos, window_size=81, stride=27)
```
