import os
import argparse
import subprocess
import sys

# ==============================================================================
# 預設處理路徑 (當未提供 --input 參數時，將預設執行此路徑)
# ==============================================================================
DEFAULT_INPUT_PATH = r"E:\squat_dataset2\be"

def main():
    parser = argparse.ArgumentParser(description="自動化深蹲影片切片腳本 (包含特徵計算與影片拼接)")
    
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH, 
                        help="請輸入要處理的資料夾路徑 (支援母資料夾或單一影片資料夾)")
    args = parser.parse_args()

    input_path = args.input

    if not os.path.exists(input_path):
        print(f"錯誤: 找不到指定的路徑 '{input_path}'")
        sys.exit(1)

    print("=" * 60)
    print(f" 開始執行自動化切片流程")
    print(f" 目標資料夾: {input_path}")
    print("=" * 60)

    # 1. 執行 cal_segments.py 來完成特徵分析與區間切片
    # 備註: cal_segments.py 內部已實作「棋盤」資料夾的排除邏輯
    print("\n[步驟 1] 正在計算切割點與生成分析圖表 (cal_segments.py)...")
    cal_cmd = [sys.executable, "cal_segments.py", "--input", input_path]
    
    try:
        subprocess.run(cal_cmd, check=True)
        print("\n>> cal_segments.py 執行成功！")
    except subprocess.CalledProcessError as e:
        print(f"\n執行 cal_segments.py 時發生錯誤: {e}")
        sys.exit(1)

    # 2. 執行 stitch_segments.py 來拼接切割出來的區間
    # 根據需求，此部分暫時使用註解隱藏
    """
    print("\n[步驟 2] 正在擷取影片區間並拼接成最終檔案 (stitch_segments.py)...")
    # 備註: stitch_segments.py 內部也已實作「棋盤」資料夾的排除邏輯
    stitch_cmd = [sys.executable, "stitch_segments.py", "--input", input_path]
    
    try:
        subprocess.run(stitch_cmd, check=True)
        print("\n>> stitch_segments.py 執行成功！")
    except subprocess.CalledProcessError as e:
        print(f"\n執行 stitch_segments.py 時發生錯誤: {e}")
        sys.exit(1)
    """

    print("\n" + "=" * 60)
    print(" 所有流程已順利完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
