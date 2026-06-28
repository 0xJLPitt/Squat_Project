"""
檔案目的: 矩陣反轉工具。如果目前有 A 到 B 的外參矩陣，但需要 B 到 A 的矩陣，可使用此工具進行數學反轉。
注意: 這是數學處理工具，請確認輸入與輸出的 .npz 檔名路徑。
呼叫指令: python step6_invert_extrinsics.py
"""
import numpy as np
import os
import glob
import sys

def invert_extrinsics(R, T):
    """
    計算反向的外參矩陣。
    原公式： P_camB = R * P_camA + T
    反向公式： P_camA = R_inv * P_camB + T_inv
    其中：
      R_inv = R^T
      T_inv = -R^T * T
    """
    R_inv = R.T
    T_inv = -np.dot(R.T, T)
    return R_inv, T_inv

def convert_file(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] 找不到外參檔案: {file_path}")
        return False

    try:
        data = np.load(file_path)
        if "R" not in data or "T" not in data:
            print(f"[ERROR] 檔案格式不正確，找不到 R 或 T 矩陣: {file_path}")
            return False

        R_orig = data["R"]
        T_orig = data["T"]

        # 計算反向矩陣
        R_inv, T_inv = invert_extrinsics(R_orig, T_orig)

        # 決定儲存的新檔名與路徑
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)

        if "vision4_to_vision3" in base_name:
            new_base_name = base_name.replace("vision4_to_vision3", "vision3_to_vision4")
        elif "vision3_to_vision4" in base_name:
            new_base_name = base_name.replace("vision3_to_vision4", "vision4_to_vision3")
        else:
            new_base_name = "inverted_" + base_name

        new_file_path = os.path.join(dir_name, new_base_name)

        # 儲存新外參
        np.savez(new_file_path, R=R_inv, T=T_inv)
        print(f"\n[OK] 成功轉換外參！")
        print(f"  - 原檔案: {file_path}")
        print(f"  - 新檔案: {new_file_path}")
        print(f"  - 原 T: {T_orig.flatten()}")
        print(f"  - 新 T: {T_inv.flatten()}")
        return True
    except Exception as e:
        print(f"[ERROR] 處理檔案 {file_path} 時發生錯誤: {e}")
        return False

def scan_and_convert(target_dir):
    """
    遞迴掃描目錄下所有 extrinsics_vision4_to_vision3.npz 檔案，並自動生成對應的 vision3_to_vision4.npz
    """
    print(f"\n[SCAN] 開始掃描目錄: {target_dir}")
    
    # 遞迴搜尋符合名稱的 npz
    pattern = os.path.join(target_dir, "**", "extrinsics_vision4_to_vision3.npz")
    files = glob.glob(pattern, recursive=True)

    # 若遞迴沒找到，也找當前層
    if not files:
        pattern = os.path.join(target_dir, "extrinsics_vision4_to_vision3.npz")
        files = glob.glob(pattern)

    if not files:
        print("[INFO] 找不到任何 extrinsics_vision4_to_vision3.npz 檔案。")
        return

    print(f"[INFO] 共找到 {len(files)} 個 'vision4_to_vision3' 檔案待轉換。")
    success_count = 0
    for f in files:
        if convert_file(f):
            success_count += 1
            
    print(f"\n==================================================")
    print(f"[DONE] 轉換任務完成！成功轉換: {success_count}/{len(files)} 個檔案。")
    print(f"==================================================")

if __name__ == "__main__":
    # 預設掃描的目錄，若您有不同的錄影資料夾，可以在此處修改或執行時傳入參數
    DEFAULT_DIR = r"E:\squat\recordings_20260507"

    target_path = DEFAULT_DIR
    if len(sys.argv) > 1:
        target_path = sys.argv[1]

    target_path = os.path.abspath(target_path)
    if os.path.isdir(target_path):
        scan_and_convert(target_path)
    elif os.path.isfile(target_path):
        convert_file(target_path)
    else:
        print(f"[ERROR] 輸入的路徑無效（非目錄也非檔案）: {target_path}")
