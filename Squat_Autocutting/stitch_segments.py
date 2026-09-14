import os
import json
import cv2
import argparse
from pathlib import Path

def stitch_recording(rec_path):
    """
    讀取資料夾下的 segments.json，並從 RD.avi 中擷取這些區間，
    最後將它們拼接成單一影片 RD_seg.mp4。
    """
    video_path = os.path.join(rec_path, 'RD.avi')
    json_path = os.path.join(rec_path, 'segments.json')
    out_path = os.path.join(rec_path, 'RD_seg.mp4')

    if not os.path.exists(video_path):
        print(f"找不到影片檔: {video_path}")
        return
    if not os.path.exists(json_path):
        print(f"找不到 JSON 檔: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        reps = json.load(f)

    if not reps:
        print(f"JSON 無切割資料: {json_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    import math
    if fps == 0 or math.isnan(fps):
        fps = 30.0  # fallback
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"正在處理: {rec_path}")
    print(f"  發現 {len(reps)} 組深蹲，開始擷取與拼接...")

    for rep in reps:
        start_frame = rep['start']
        end_frame = rep['end']
        rep_id = rep['rep_id']
        
        # 設定讀取指針到開始幀
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 在畫面左上角印上這是第幾下 (可選)
            cv2.putText(frame, f"Rep: {rep_id}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            out.write(frame)

    cap.release()
    out.release()
    print(f"✅ 成功產出: {out_path}\n")


def main():
    parser = argparse.ArgumentParser(description="拼接 RD.avi 的深蹲區間")
    # ==============================================================================
    # 請在下方直接貼上包含 segments.json 的「母資料夾」路徑
    # 程式會讀取該資料夾底下的 segments.json，並在各個子資料夾中產出剪輯影片
    # ==============================================================================
    parser.add_argument("--input", required=True, help="包含各錄影的母資料夾路徑")
    args = parser.parse_args()

    input_dir = Path(args.input)
    
    if not input_dir.exists():
        print(f"錯誤: 找不到路徑 {input_dir}")
        return

    print(f"開始搜尋 {input_dir} 內的資料夾...\n")
    
    # 遍歷底下所有資料夾
    processed_any = False
    for root, dirs, files in os.walk(input_dir):
        if "棋盤" in root:
            continue
        if 'RD.avi' in files and 'segments.json' in files:
            stitch_recording(root)
            processed_any = True
            
    if not processed_any:
        print("沒有找到同時包含 'RD.avi' 和 'segments.json' 的資料夾。")

if __name__ == "__main__":
    main()
