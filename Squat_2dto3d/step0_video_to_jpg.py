"""
檔案目的: 將影片抽出圖片 (Frame extraction)，為了後續 YOLO 辨識或棋盤格校正使用。
注意: 請確認程式內的影片路徑 (如 TARGET_DIR 或影片檔名) 是否正確，以免抽錯檔案。
呼叫指令: python step0_video_to_jpg.py
"""
import cv2
import os
import glob
from pathlib import Path

def extract_frames(video_path, output_root=None, frame_interval=10):
    """
    將單個影片轉換為圖片幀。
    
    :param video_path: 影片檔案路徑
    :param output_root: 儲存圖片的根目錄。如果為 None，則在影片同目錄下建立資料夾。
    :param frame_interval: 每隔幾幀擷取一張。
    """
    video_path = Path(video_path)
    video_name = video_path.stem
    
    # 決定輸出資料夾：video_name_jpg
    if output_root:
        output_folder = Path(output_root) / f"{video_name}_jpg"
    else:
        output_folder = video_path.parent / f"{video_name}_jpg"
        
    # 檢查是否已經處理過 (輸出資料夾存在且含有 jpg 檔案)
    if output_folder.exists() and any(output_folder.glob("*.jpg")):
        print(f"⏭️  已存在擷取圖片，跳過影片: {video_path.name}")
        return
        
    os.makedirs(output_folder, exist_ok=True)
    
    # 使用 OpenCV 開啟影片
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ 無法開啟影片: {video_path}")
        return
        
    frame_count = 0
    saved_count = 0
    
    # 獲取影片總幀數 (僅供顯示進度)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎬 正在處理影片: {video_path.name}")
    print(f"   總幀數: {total_frames}, 擷取間隔: {frame_interval}")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # 每隔 frame_interval 幀儲存一次
        if frame_count % frame_interval == 0:
            # 檔名格式: 影片名_frame"n".jpg
            filename = f"{video_name}_frame{frame_count:05d}.jpg"
            filepath = output_folder / filename
            
            # 使用 cv2.imencode 搭配 tofile 以支援 Windows 中文路徑
            success, img_encoded = cv2.imencode(".jpg", frame)
            if success:
                img_encoded.tofile(str(filepath))
                saved_count += 1
            else:
                print(f"❌ 圖片編碼失敗: {filename}")
            
        frame_count += 1
        
        # 簡易進度顯示
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"\r   进度: {progress:.1f}% ({frame_count}/{total_frames})", end="")
            
    cap.release()
    print(f"\n   ✅ 處理完成！擷取了 {saved_count} 張圖片，存於 '{output_folder.name}'\n")

def process_directory(directory_path, frame_interval=10, recursive=False):
    """
    處理指定資料夾內的所有影片。
    """
    target_dir = Path(directory_path)
    if not target_dir.exists():
        print(f"❌ 找不到目錄: {directory_path}")
        return

    # 定義常見的影片副檔名
    video_extensions = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.flv', '*.wmv']
    
    video_files = []
    for ext in video_extensions:
        if recursive:
            video_files.extend(target_dir.rglob(ext))
        else:
            video_files.extend(target_dir.glob(ext))
    
    if not video_files:
        print(f"⚠️ 在 {directory_path} 中找不到指定的影片檔案。")
        return

    print(f"🚀 發現 {len(video_files)} 個影片，準備開始處理...")
    
    for video in video_files:
        extract_frames(video, frame_interval=frame_interval)

def main():
    # --- 使用者設定 ---
    # 指定母資料夾/子資料夾
    default_path = r"D:\Pitt\Project\2dto3d\v2"
    
    print(f"📂 預設處理目錄: {default_path}")
    print("您可以直接按 Enter 處理預設母資料夾下的所有子資料夾影片，或是輸入特定的子資料夾路徑。")
    input_path = input(f"請輸入路徑 (直接按 Enter 開始處理預設目錄): ").strip()
    
    if not input_path:
        input_path = default_path
    
    interval_str = input("請輸入擷取間隔 (預設為 10): ").strip()
    interval = int(interval_str) if interval_str.isdigit() else 10
    
    # 執行處理 (開啟 recursive=True，以防影片在更深層的子目錄)
    process_directory(input_path, frame_interval=interval, recursive=True)

if __name__ == "__main__":
    main()
