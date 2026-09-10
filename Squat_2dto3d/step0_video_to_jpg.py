"""
檔案目的: 將影片抽出圖片 (Frame extraction)，為了後續 YOLO 辨識或棋盤格校正使用。
注意: 請確認程式內的影片路徑 (如 TARGET_DIR 或影片檔名) 是否正確，以免抽錯檔案。
呼叫指令: python step0_video_to_jpg.py
"""
import cv2
import os
import glob
import sys
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def extract_frames(video_path, output_root=None, target_dir=None, frame_interval=None, target_fps=None, overwrite=False, folder_suffix="_jpg"):
    """
    將單個影片轉換為圖片幀。
    
    :param video_path: 影片檔案路徑
    :param output_root: 儲存圖片的根目錄。如果為 None，則在影片同目錄下建立資料夾。
    :param target_dir: 批次處理時的根目錄 (用於在指定 output_root 時保留相對資料夾結構)。
    :param frame_interval: 每隔幾幀擷取一張 (若未指定且有提供 target_fps，則根據影片 FPS 自動計算)。
    :param target_fps: 每秒擷取張數 (例如 5 代表每秒抽 5 張)。
    :param overwrite: 是否覆蓋已存在的圖片。
    :param folder_suffix: 輸出資料夾名稱的後綴 (預設為 '_jpg'，設為空字串即為純影片名)。
    """
    video_path = Path(video_path)
    video_name = video_path.stem
    folder_name = f"{video_name}{folder_suffix}" if folder_suffix else video_name
    
    # 決定輸出資料夾
    if output_root:
        if target_dir:
            try:
                rel_dir = video_path.parent.relative_to(target_dir)
                output_folder = Path(output_root) / rel_dir / folder_name
            except Exception:
                output_folder = Path(output_root) / folder_name
        else:
            output_folder = Path(output_root) / folder_name
    else:
        output_folder = video_path.parent / folder_name
        
    # 檢查是否已經處理過 (輸出資料夾存在且含有 jpg 檔案)
    if not overwrite and output_folder.exists() and any(output_folder.glob("*.jpg")):
        print(f"[SKIP] 已存在擷取圖片，跳過影片: {video_path.name}")
        return
        
    os.makedirs(output_folder, exist_ok=True)
    
    # 使用 OpenCV 開啟影片
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟影片: {video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 計算擷取間隔 (frame_interval)
    if target_fps is not None and target_fps > 0:
        if fps > 0:
            frame_interval = max(1, round(fps / target_fps))
        else:
            frame_interval = 6
        print(f"[PROCESS] 正在處理影片: {video_path.name}")
        print(f"   總幀數: {total_frames}, 原始FPS: {fps:.2f}, 目標FPS: {target_fps} (間隔: 每 {frame_interval} 幀一張)")
    else:
        if frame_interval is None:
            frame_interval = 10
        print(f"[PROCESS] 正在處理影片: {video_path.name}")
        print(f"   總幀數: {total_frames}, 原始FPS: {fps:.2f}, 擷取間隔: 每 {frame_interval} 幀一張")
    
    frame_count = 0
    saved_count = 0
    
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
                print(f"[FAIL] 圖片編碼失敗: {filename}")
            
        frame_count += 1
        
        # 簡易進度顯示
        if frame_count % 100 == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"\r   進度: {progress:.1f}% ({frame_count}/{total_frames})", end="", flush=True)
            
    cap.release()
    print(f"\n   [SUCCESS] 處理完成！擷取了 {saved_count} 張圖片，存於: {output_folder}\n")

def process_directory(directory_path, frame_interval=None, target_fps=None, recursive=True, pattern=None, output_root=None, overwrite=False, folder_suffix="_jpg"):
    """
    處理指定資料夾內的所有影片。
    
    :param directory_path: 目標資料夾
    :param frame_interval: 每隔幾幀擷取一張
    :param target_fps: 每秒擷取張數 (例如 5)
    :param recursive: 是否遞迴搜尋子目錄
    :param pattern: 檔名過濾關鍵字 (例如 "checkboard" 或 "REC")
    :param output_root: 輸出根目錄 (若為 None 則存於影片同目錄)
    :param overwrite: 是否覆蓋已存在圖片
    :param folder_suffix: 影片資料夾後綴名稱 (預設為 '_jpg')
    """
    target_dir = Path(directory_path)
    if not target_dir.exists():
        print(f"[ERROR] 找不到目錄: {directory_path}")
        return

    # 定義常見的影片副檔名
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv'}
    
    if recursive:
        video_files = [f for f in target_dir.rglob('*') if f.suffix.lower() in video_extensions]
    else:
        video_files = [f for f in target_dir.glob('*') if f.suffix.lower() in video_extensions]
    
    if pattern:
        video_files = [f for f in video_files if pattern.lower() in f.name.lower()]
        
    video_files = sorted(video_files)
    
    if not video_files:
        print(f"[WARN] 在 {directory_path} 中找不到指定的影片檔案 (過濾條件: pattern='{pattern}')。")
        return

    print(f"[START] 發現 {len(video_files)} 個影片，準備開始處理...")
    for v in video_files:
        print(f"   - {v}")
    print("-" * 50)
    
    for video in video_files:
        extract_frames(
            video_path=video,
            output_root=output_root,
            target_dir=target_dir,
            frame_interval=frame_interval,
            target_fps=target_fps,
            overwrite=overwrite,
            folder_suffix=folder_suffix
        )

def main():
    parser = argparse.ArgumentParser(description="將影片抽出圖片 (Frame extraction)")
    parser.add_argument("--dir", "-d", type=str, default=None, help="目標影片資料夾路徑")
    parser.add_argument("--interval", "-i", type=int, default=None, help="擷取間隔 (每 N 幀抓一張，例如 1 代表每幀都抽)")
    parser.add_argument("--fps", "-f", type=float, default=None, help="每秒擷取張數 (例如 5 代表 1 秒抓 5 張)")
    parser.add_argument("--pattern", "-p", type=str, default=None, help="檔名過濾關鍵字 (例如 checkboard 或 REC)")
    parser.add_argument("--output_root", "-o", type=str, default=None, help="輸出圖片根目錄 (預設為各影片所在資料夾)")
    parser.add_argument("--recursive", "-r", dest="recursive", action="store_true", default=True, help="遞迴搜尋子目錄 (預設 True)")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="僅搜尋指定目錄，不搜尋子目錄")
    parser.add_argument("--no_suffix", action="store_true", default=False, help="資料夾名稱不加 '_jpg' 後綴 (直接以純影片名稱命名資料夾)")
    parser.add_argument("--suffix", type=str, default=None, help="自訂資料夾後綴 (預設為 '_jpg'；若指定 --no_suffix 則為空)")
    parser.add_argument("--overwrite", action="store_true", default=False, help="若輸出資料夾已有圖片是否重新擷取覆蓋")

    args = parser.parse_args()

    # 決定資料夾後綴
    if args.suffix is not None:
        folder_suffix = args.suffix
    elif args.no_suffix:
        folder_suffix = ""
    else:
        folder_suffix = "_jpg"

    # 若有帶入命令列參數
    if args.dir is not None:
        process_directory(
            directory_path=args.dir,
            frame_interval=args.interval,
            target_fps=args.fps,
            recursive=args.recursive,
            pattern=args.pattern,
            output_root=args.output_root,
            overwrite=args.overwrite,
            folder_suffix=folder_suffix
        )
        return

    # --- 互動模式 ---
    default_path = r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\i15\sub1"
    
    print(f"[INFO] 預設處理目錄: {default_path}")
    print("您可以直接按 Enter 處理預設目錄下的影片，或是輸入特定的資料夾路徑。")
    input_path = input(f"請輸入路徑 (直接按 Enter 使用預設目錄): ").strip()
    
    if not input_path:
        input_path = default_path
    
    fps_choice = input("是否以每秒指定張數 (FPS) 擷取？(例如輸入 5 代表每秒抓 5 張，若要使用固定幀數間隔請直接按 Enter): ").strip()
    target_fps = None
    interval = None
    if fps_choice and fps_choice.replace('.', '', 1).isdigit():
        target_fps = float(fps_choice)
    else:
        interval_str = input("請輸入擷取間隔幀數 (預設為 10，若每影格都要請輸入 1): ").strip()
        interval = int(interval_str) if interval_str.isdigit() else 10
    
    pattern_str = input("請輸入檔名過濾關鍵字 (直接按 Enter 處理全部影片，例如輸入 REC 或 checkboard): ").strip()
    pattern = pattern_str if pattern_str else None

    suffix_choice = input("資料夾名稱是否加上 '_jpg' 後綴？(直接按 Enter 預設加上，輸入 n 則僅以影片名建立資料夾): ").strip().lower()
    interactive_suffix = "" if suffix_choice == 'n' else "_jpg"
    
    # 執行處理
    process_directory(
        directory_path=input_path,
        frame_interval=interval,
        target_fps=target_fps,
        recursive=True,
        pattern=pattern,
        folder_suffix=interactive_suffix
    )

if __name__ == "__main__":
    main()
