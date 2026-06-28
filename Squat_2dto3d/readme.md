# Camera and Video Mapping

The following is the mapping between the calibration camera names (vision1 to vision6) and the recording video files:

- `vision1.avi` -> `RU.avi`
- `vision2.avi` -> `RR.avi`
- `vision3.avi` -> `RLU.avi`
- `vision4.avi` -> `FL.avi`
- `vision5.avi` -> `FR.avi`
- `vision6.avi` -> `RD.avi`

We use `vision2`, `vision3`, `vision4`, and `vision5` to perform 3D reconstruction.


## 執行步驟說明與注意事項 (Step 0 ~ Step 11)

以下為 code 資料夾中各個 Python 執行檔的作用與注意事項：

### step0_video_to_jpg.py
- **檔案目的**: 將影片抽出圖片 (Frame extraction)，為了後續 YOLO 辨識或棋盤格校正使用。
- **注意**: 請確認程式內的影片路徑 (如 `TARGET_DIR` 或影片檔名) 是否正確，以免抽錯檔案。

### step1_run_yolo_pose.py
- **檔案目的**: 執行 YOLO11 Pose 姿態辨識，將影片中的人物轉換為 2D 骨架數據 (`.txt` 或 `.npy`)。
- **注意**: 確保 YOLO 模型檔 (`.pt`) 路徑正確，且會消耗較多 GPU 資源。

### step2_draw_2d_on_video.py
- **檔案目的**: 將 YOLO 辨識出的 2D 骨架座標，重新畫回原影片上以供人工肉眼檢查。
- **注意**: 這是可選步驟，如果 3D 投影有問題，可以先用此步驟確認是否是 YOLO 本身在 2D 就已經抓錯。

### step3_extrinsics.py
- **檔案目的**: 執行自動外參校正，透過棋盤格找出各攝影機 (`vision2~5` 或 `RR/RLU/FL/FR`) 之間的相對 3D 位置與旋轉角度。
- **注意**: 程式內已包含 vision3 與 vision4 相機對接時的 180 度翻轉修復機制。請確認目標資料夾的棋盤格圖片是否清晰。

### step4_manual_calibration.py
- **檔案目的**: 執行手動外參校正。當自動校正找不到棋盤格時，可用此腳本手動點擊影像中的角點來計算外參。
- **注意**: 滑鼠點擊的角點順序，必須與自動校正演算法的 1~15 點順序完全一致，否則 3D 空間會嚴重扭曲。

### step5_batch_visualize.py
- **檔案目的**: 批次產生校正對照圖 (產生 2-3, 3-4, 4-5 的角點匹配結果圖)，用於檢查外參校正是否正確對齊。
- **注意**: Windows 系統下若路徑有中文，OpenCV 存圖會失敗，此腳本已使用 `imencode` 避開此問題。

### calibration_visualizer.py
- **檔案目的**: 提供外參校正結果的視覺化繪圖功能，將左右鏡頭的棋盤格特徵點畫在圖片上並水平合併。
- **注意**: 這是一個函式庫檔案，提供 `save_stereo_visualization` 供其他腳本呼叫，不需獨立執行。

### step6_invert_extrinsics.py
- **檔案目的**: 矩陣反轉工具。如果目前有 A 到 B 的外參矩陣，但需要 B 到 A 的矩陣，可使用此工具進行數學反轉。
- **注意**: 這是數學處理工具，請確認輸入與輸出的 `.npz` 檔名路徑。

### step7_visualize_extrinsics.py
- **檔案目的**: 3D 空間相機位置視覺化，將計算出來的相機外參 (位置與朝向) 畫在 3D 座標系中供檢查。
- **注意**: 觀察 matplotlib 顯示的 3D 圖表時，請確認相機的相對擺放位置是否符合真實世界 (例如 RR 和 FL 應該是面對面的)。

### step8_triangulate_and_render_3d.py
- **檔案目的**: 進行所有相機的 3D 三角測量，並套用 Savitzky-Golay 濾波器平滑化，最後產生 `reconstructed_3d.mp4`。
- **注意**: 執行前請確保所有內參與外參矩陣 (`intrinsics`/`extrinsics`) 都已經正確產生，否則算出的 3D 骨架會扭曲。

### step9_triangulate_and_render_3d_4cams.py
- **檔案目的**: 專門只挑選 4 個特定鏡頭進行 3D 三角測量。
- **注意**: 確保指定的 4 個攝影機都有對應的 2D 骨架資料與外參檔案。

### step10_project_3d_to_video.py
- **檔案目的**: 將平滑後的 3D 骨架重新投影回 4 個視角的 2D 影片上，產生 `_projected.mp4` 供驗證。
- **注意**: 會自動呼叫 step8 裡的方法來讀取 3D 點。如果投影線條沒有貼合人體，通常是外參校正有問題。

### step11_project_raw_3d_to_video.py
- **檔案目的**: 將「未平滑」的原始 3D 骨架投影回 2D 影片。
- **注意**: 用於除錯，觀察原始三角測量的誤差。投影出來的骨架可能會抖動，此為正常現象。
