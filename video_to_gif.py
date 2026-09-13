#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video to GIF Converter (影片轉 GIF 工具)
支援格式: .mov, .avi, .mp4 (及其他常見影片格式)
支援單檔轉換、資料夾批次轉換、時間裁剪 (start/end/duration)、解析度縮放 (width/scale)、FPS 調整與色彩優化。
prompt指令範例: python video_to_gif.py C:\Users\USER\Downloads\4camrecording.mp4 -w 320 --fps 10 --colors 128
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Union, List, Tuple

# 設定 Windows 控制台為 UTF-8 輸出
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # 簡易替代方案，若未安裝 tqdm 仍可正常運作
    def tqdm(iterable=None, total=None, desc="", unit="it", **kwargs):
        if iterable is not None:
            for item in iterable:
                yield item
        else:
            class DummyBar:
                def __init__(self, total):
                    self.total = total
                    self.count = 0
                def update(self, n=1):
                    self.count += n
                def close(self):
                    pass
            return DummyBar(total)

# 支援的影片副檔名 (不區分大小寫)
SUPPORTED_EXTENSIONS = {".mov", ".avi", ".mp4", ".mkv", ".flv", ".wmv", ".webm"}


def parse_time_str(time_str: Optional[Union[str, int, float]]) -> Optional[float]:
    """
    解析時間字串為秒數 (支援: 秒數如 '12.5', MM:SS 如 '01:30', HH:MM:SS 如 '00:02:15')
    """
    if time_str is None:
        return None
    if isinstance(time_str, (int, float)):
        return float(time_str)
    
    time_str = str(time_str).strip()
    if not time_str:
        return None
    
    parts = time_str.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:  # MM:SS
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:  # HH:MM:SS
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        else:
            raise ValueError
    except ValueError:
        raise ValueError(
            f"無法解析時間格式: '{time_str}'，請輸入秒數 (例如 10.5) 或 MM:SS (例如 01:30) 或 HH:MM:SS"
        )


def calculate_dimensions(
    orig_w: int,
    orig_h: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[float] = None,
) -> Tuple[int, int]:
    """
    根據寬度、高度或縮放比例計算目標輸出解析度 (保持等比例縮放)
    """
    if scale is not None and scale > 0:
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        return new_w, new_h

    if width is not None and height is not None:
        return max(1, int(width)), max(1, int(height))
    elif width is not None:
        aspect = orig_h / orig_w
        new_w = max(1, int(width))
        new_h = max(1, int(round(new_w * aspect)))
        return new_w, new_h
    elif height is not None:
        aspect = orig_w / orig_h
        new_h = max(1, int(height))
        new_w = max(1, int(round(new_h * aspect)))
        return new_w, new_h

    return orig_w, orig_h


def format_bytes(size_in_bytes: int) -> str:
    """轉換檔案大小為好讀的單位"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"


def convert_video_to_gif(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    fps: Optional[float] = 12.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[float] = None,
    start_time: Optional[Union[str, float]] = None,
    end_time: Optional[Union[str, float]] = None,
    duration: Optional[Union[str, float]] = None,
    loop: int = 0,
    colors: int = 256,
    optimize: bool = True,
    quiet: bool = False,
) -> Path:
    """
    將單一影片轉換為 GIF 動畫。

    :param video_path: 來源影片路徑 (.mov, .avi, .mp4 等)
    :param output_path: 輸出 GIF 路徑 (預設為同檔名 .gif)
    :param fps: GIF 幀率 (預設 12 fps，設為 None 或 <=0 則保留原影片幀率)
    :param width: 目標寬度 (等比例縮放)
    :param height: 目標高度 (等比例縮放)
    :param scale: 縮放比例 (例如 0.5 表示 50%)
    :param start_time: 起始時間 (秒數或 'MM:SS')
    :param end_time: 結束時間 (秒數或 'MM:SS')
    :param duration: 擷取時長 (秒數或 'MM:SS')
    :param loop: 循環次數 (0 表示無限循環)
    :param colors: 調色盤顏色數 (最大 256)
    :param optimize: 是否開啟 GIF 調色盤壓縮優化
    :param quiet: 是否靜音輸出
    :return: 輸出 GIF 的 Path 物件
    """
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到影片檔案: {video_path}")

    # 決定輸出路徑
    if output_path is None:
        output_path = video_path.with_suffix(".gif")
    else:
        output_path = Path(output_path).resolve()
        if output_path.is_dir():
            output_path = output_path / f"{video_path.stem}.gif"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 開啟影片
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片檔案: {video_path}")

    try:
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0 or np.isnan(orig_fps):
            orig_fps = 30.0  # 容錯預設值

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_duration = total_frames / orig_fps if total_frames > 0 else 0.0

        # 解析時間參數
        start_sec = parse_time_str(start_time) or 0.0
        if start_sec < 0:
            start_sec = 0.0

        parsed_end = parse_time_str(end_time)
        parsed_dur = parse_time_str(duration)

        if parsed_dur is not None:
            end_sec = start_sec + parsed_dur
        elif parsed_end is not None:
            end_sec = parsed_end
        else:
            end_sec = video_duration if video_duration > 0 else None

        if end_sec is not None and end_sec <= start_sec:
            raise ValueError(f"結束時間 ({end_sec}s) 必須大於起始時間 ({start_sec}s)")

        # 計算目標輸出尺寸與 FPS
        target_w, target_h = calculate_dimensions(orig_w, orig_h, width, height, scale)
        need_resize = (target_w, target_h) != (orig_w, orig_h)
        interp_method = cv2.INTER_AREA if (target_w <= orig_w and target_h <= orig_h) else cv2.INTER_LINEAR

        effective_fps = float(fps) if (fps and fps > 0) else orig_fps
        clip_duration = (end_sec - start_sec) if end_sec else (video_duration - start_sec)
        est_gif_frames = max(1, int(round(clip_duration * effective_fps))) if clip_duration > 0 else None

        if not quiet:
            print("=" * 60)
            print(f"來源影片: {video_path.name}")
            print(f"原始規格: {orig_w}x{orig_h} @ {orig_fps:.2f} fps (總時長: {video_duration:.2f}s)")
            print(f"擷取區間: {start_sec:.2f}s ~ {'結尾' if end_sec is None else f'{end_sec:.2f}s'}")
            print(f"輸出規格: {target_w}x{target_h} @ {effective_fps:.2f} fps")
            print(f"目標檔案: {output_path}")
            if orig_w >= 1280 and width is None and scale is None:
                print("-" * 60)
                print(f"[體積提示] 來源影片解析度較高 ({orig_w}x{orig_h})，以原尺寸輸出 GIF 體積通常會高達 100MB 以上。")
                print("           若預期體積在 3~5 MB 左右，建議加上: -w 360 (或 -w 480) 與 --colors 128")
            print("=" * 60)

        # 移動到起始影格
        start_frame_idx = int(round(start_sec * orig_fps))
        if start_frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

        frames: List[Image.Image] = []
        current_frame_idx = start_frame_idx
        next_target_time = start_sec

        pbar = None
        if not quiet:
            pbar = tqdm(total=est_gif_frames, desc="處理影格", unit="幀")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = current_frame_idx / orig_fps
            if end_sec is not None and current_time >= end_sec:
                break

            # 取樣檢查 (精確配合目標 fps)
            if current_time >= next_target_time - (0.5 / orig_fps):
                # 調整尺寸
                if need_resize:
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=interp_method)

                # BGR 轉 RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)

                # 調色盤優化 (若指定小於 256 色可大幅減少檔案大小)
                if colors < 256:
                    pil_frame = pil_frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors)

                frames.append(pil_frame)
                if pbar:
                    pbar.update(1)

                next_target_time += 1.0 / effective_fps

            current_frame_idx += 1

        if pbar:
            pbar.close()

        if not frames:
            raise RuntimeError("未擷取到任何影格，請檢查時間設定或影片是否損毀。")

        if not quiet:
            print(f"正在儲存 GIF ({len(frames)} 影格)... 請稍候")

        # 計算每幀毫秒數
        duration_ms = max(10, int(round(1000.0 / effective_fps)))

        # 儲存 GIF
        frames[0].save(
            str(output_path),
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=loop,
            optimize=optimize,
        )

        out_size = output_path.stat().st_size
        if not quiet:
            print(f"轉換完成: {output_path.name}")
            print(f"檔案大小: {format_bytes(out_size)}")
            print("-" * 60)

        return output_path

    finally:
        cap.release()


def batch_convert(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    recursive: bool = False,
    **kwargs,
) -> List[Path]:
    """
    批次轉換資料夾內所有支援的影片 (.mov, .avi, .mp4 等) 為 GIF
    """
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"指定路徑不是資料夾: {input_dir}")

    pattern = "**/*" if recursive else "*"
    video_files = [
        f for f in input_dir.glob(pattern)
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not video_files:
        print(f"在 {input_dir} 中找不到支援的影片檔案 ({', '.join(sorted(SUPPORTED_EXTENSIONS))})")
        return []

    print(f"找到 {len(video_files)} 個影片檔案，開始批次轉換...")
    results: List[Path] = []

    for idx, video_file in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] 處理: {video_file.name}")
        target_out = None
        if output_dir:
            out_base = Path(output_dir).resolve()
            if recursive:
                rel = video_file.relative_to(input_dir)
                target_out = (out_base / rel).with_suffix(".gif")
            else:
                target_out = out_base / f"{video_file.stem}.gif"

        try:
            gif_path = convert_video_to_gif(video_file, output_path=target_out, **kwargs)
            results.append(gif_path)
        except Exception as e:
            print(f"轉換失敗 ({video_file.name}): {e}", file=sys.stderr)

    print(f"\n全部批次轉換完成！成功: {len(results)}/{len(video_files)}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="影片轉 GIF 工具 (支援 .mov, .avi, .mp4 等格式)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "positional_input",
        nargs="?",
        default=None,
        help="來源影片檔案路徑或資料夾路徑 (例如: video.mp4 或 ./my_videos)",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="來源影片檔案路徑或資料夾路徑 (也可直接作為第一個無標籤參數傳入)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="輸出 GIF 檔案路徑或輸出資料夾路徑 (預設為同檔名 .gif)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="GIF 幀率 (預設: 12 fps，設為 0 或負數則保持原影片幀率)",
    )
    parser.add_argument(
        "-w", "--width",
        type=int,
        default=None,
        help="設定目標寬度 (高度會依原比例自動計算，例如: 480)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="設定目標高度 (寬度會依原比例自動計算)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="縮放比例 (例如: 0.5 為 50%% 寬高，優先於 width/height)",
    )
    parser.add_argument(
        "-s", "--start",
        type=str,
        default=None,
        help="剪輯起始時間 (支援秒數如 '10.5'，或 'MM:SS' 如 '01:30'，或 'HH:MM:SS')",
    )
    parser.add_argument(
        "-e", "--end",
        type=str,
        default=None,
        help="剪輯結束時間 (支援秒數或 'MM:SS' 或 'HH:MM:SS')",
    )
    parser.add_argument(
        "-d", "--duration",
        type=str,
        default=None,
        help="剪輯時長 (從起始時間開始算，例如: 5 或 '00:05')",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="循環次數 (預設: 0 為無限循環，1 為播放一次)",
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=256,
        help="調色盤最大顏色數量 (2~256，預設: 256；設為 128 或 64 可進一步減小檔案)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="關閉調色盤與壓縮優化 (加速生成，但檔案會較大)",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="若輸入為資料夾，是否遞迴搜尋子目錄內的所有影片",
    )

    args = parser.parse_args()
    raw_input = args.input or args.positional_input
    if not raw_input:
        parser.print_help()
        print("\n錯誤: 請提供來源影片路徑 (例如: python video_to_gif.py my_video.mp4)", file=sys.stderr)
        sys.exit(1)

    input_path = Path(raw_input)
    if not input_path.exists():
        print(f"錯誤: 輸入路徑不存在: {raw_input}", file=sys.stderr)
        sys.exit(1)

    kwargs = {
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "scale": args.scale,
        "start_time": args.start,
        "end_time": args.end,
        "duration": args.duration,
        "loop": args.loop,
        "colors": args.colors,
        "optimize": not args.no_optimize,
    }

    if input_path.is_dir():
        batch_convert(
            input_dir=input_path,
            output_dir=args.output,
            recursive=args.recursive,
            **kwargs,
        )
    else:
        # 單檔轉換
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"警告: 檔案副檔名 '{input_path.suffix}' 不在常見清單中，將嘗試以 OpenCV 開啟處理。")
        
        convert_video_to_gif(
            video_path=input_path,
            output_path=args.output,
            **kwargs,
        )


if __name__ == "__main__":
    main()
