"""
檔案目的: 執行自動外參校正，透過棋盤格找出各攝影機 (vision2~5 或 RR/RLU/FL/FR) 之間的相對 3D 位置與旋轉角度。
注意: 程式內已包含 vision3 與 vision4 相機對接時的 180 度翻轉修復機制。請確認目標資料夾的棋盤格圖片是否清晰。
呼叫指令: python step3_extrinsics.py
"""
import cv2
import numpy as np
import os
import glob

def calibrate_extrinsics(dir1, dir2, mtx1, dist1, mtx2, dist2, id1, id2, pattern_size=(8, 5), square_size=25.0):
    """
    計算雙相機外參 (Stereo Calibration)
    :param id1: 例如 "vision2"
    :param id2: 例如 "vision3"
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []
    img_points1 = []
    img_points2 = []

    # 取得兩邊的檔案清單
    files1 = {f: f for f in os.listdir(dir1) if f.endswith(".jpg")}
    
    h, w = 0, 0
    common_count = 0

    print(f"\n[SCAN] 正在尋找 {id1} 與 {id2} 的同步棋盤格...")

    for f1 in sorted(files1.keys()):
        # 尋找對應的檔名 (例如 vision2_frame00000.jpg -> vision3_frame00000.jpg)
        f2 = f1.replace(f"{id1}_", f"{id2}_")
        path1 = os.path.join(dir1, f1)
        path2 = os.path.join(dir2, f2)

        if not os.path.exists(path2):
            continue

        img1 = cv2.imdecode(np.fromfile(path1, dtype=np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.fromfile(path2, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img1 is None or img2 is None:
            continue
            
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        h, w = gray1.shape[:2]

        # 使用 SB (Sector Based) 演算法找角點，對無白邊棋盤格更強健
        ret1, corners1 = cv2.findChessboardCornersSB(gray1, pattern_size, None)
        ret2, corners2 = cv2.findChessboardCornersSB(gray2, pattern_size, None)

        if ret1 and ret2:
            print(f"[OK] 發現同步偵測幀: {f1}")
            obj_points.append(objp)
            
            # SB 演算法本身已經包含亞像素精確度，不需要額外跑 cornerSubPix
            
            # 關鍵修復：如果是 vision3 (後方) 匹配 vision4 (前方)，棋盤格的物理方向在 OpenCV 偵測時被旋轉了 180 度！
            # 必須把 vision4 (corners2) 完全反轉 (180度)，才能對應到真實世界的同一個物理角點。
            if (id1 == "RLU" and id2 == "FL") or (id1 == "vision3" and id2 == "vision4"):
                # corners2 shape is (15, 1, 2)
                # 反轉整個陣列，等同於旋轉 180 度 (點 1 變成點 15，點 2 變成點 14...)
                corners2 = corners2[::-1, :, :]
                
                
            img_points1.append(corners1)
            img_points2.append(corners2)
            common_count += 1

            # 儲存左右同步標記點視覺化圖片
            try:
                from calibration_visualizer import save_stereo_visualization
                vis_dir = os.path.join(os.path.dirname(dir1), "calibration_visualized")
                save_stereo_visualization(img1, img2, corners1, corners2, id1, id2, f1, vis_dir, pattern_size=pattern_size, is_manual=False)
            except Exception as vis_err:
                print(f"[WARN] 儲存標記視覺化影像失敗: {vis_err}")

    if common_count < 3:
        print(f"[FAIL] {id1} 與 {id2} 同步成功的照片太少 ({common_count} 張)，無法計算外參。")
        return None

    print(f"[CALIB] 正在進行立體校正... 使用 {common_count} 個同步點")
    
    # 固定內參，只算外參 R, T
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points1, img_points2,
        mtx1, dist1, mtx2, dist2, (w, h), flags=flags
    )

    if ret:
        print(f"[DONE] {id1} -> {id2} 外參計算成功！")
        return {"R": R, "T": T}
    return None

if __name__ == "__main__":
    # ================= 設定路徑 =================
    # 可以是母資料夾 (會自動跑底下所有子錄影資料夾)
    # 也可以是單一子資料夾
    TARGET_PATH = r"E:\squat\recordings_20260507_done\recording_20260211_棋盤格"
    
    # 內參檔案所在的資料夾
    INTRINSIC_DIR = r"E:\squat\recordings_20260507\intrinsics_vision"
    # ============================================

    # 讀取所有內參 (同時支援舊版與新版檔名)
    cam_mapping_new = {
        "vision1": "RU",
        "vision2": "RR",
        "vision3": "RLU",
        "vision4": "FL",
        "vision5": "FR",
        "vision6": "RD",
    }

    intrinsics = {}
    for cam_id in ["vision2", "vision3", "vision4", "vision5"]:
        path_old = os.path.join(INTRINSIC_DIR, f"intrinsics_{cam_id}.npz")
        new_name = cam_mapping_new[cam_id]
        path_new = os.path.join(INTRINSIC_DIR, f"intrinsics_{new_name}.npz")

        path = None
        if os.path.exists(path_old):
            path = path_old
        elif os.path.exists(path_new):
            path = path_new

        if path:
            data = np.load(path)
            intrinsics[cam_id] = (data["mtx"], data["dist"])
            print(f"[INFO] 已載入 {cam_id} 的內參 (使用: {os.path.basename(path)})")
        else:
            print(f"[WARN] 找不到 {cam_id} 的內參檔案: 嘗試了 {os.path.basename(path_old)} 與 {os.path.basename(path_new)}")

    # 要進行校正的相機配對 (鍊式校正: 2-3, 3-4, 4-5)
    camera_pairs = [
        ("vision2", "vision3"),
        ("vision3", "vision4"),
        ("vision4", "vision5")
    ]

    # 判斷 TARGET_PATH 是母資料夾還是單一子資料夾
    subdirs = []
    if os.path.exists(TARGET_PATH):
        # 舊版與新版資料夾名稱對照
        old_cams = ["vision2", "vision3", "vision4", "vision5"]
        new_cams = ["RR", "RLU", "FL", "FR"]

        def is_recording_folder(path):
            if any(os.path.exists(os.path.join(path, f"{cam}_jpg")) for cam in old_cams):
                return True
            if any(os.path.exists(os.path.join(path, f"{cam}_jpg")) for cam in new_cams):
                return True
            return False

        if is_recording_folder(TARGET_PATH):
            subdirs = [TARGET_PATH]
            print(f"[INFO] 偵測為單一子資料夾: {TARGET_PATH}")
        else:
            # 掃描母資料夾底下的所有子資料夾
            for item in sorted(os.listdir(TARGET_PATH)):
                item_path = os.path.join(TARGET_PATH, item)
                if os.path.isdir(item_path):
                    # 排除內參資料夾與視覺化結果資料夾
                    if item == "intrinsics_vision" or "visualized" in item:
                        continue
                    if is_recording_folder(item_path):
                        subdirs.append(item_path)
            print(f"[INFO] 偵測為母資料夾，共找到 {len(subdirs)} 個子資料夾待處理")
    else:
        print(f"[ERROR] 目標路徑不存在: {TARGET_PATH}")

    # 依序處理每個資料夾
    for idx, folder in enumerate(subdirs):
        print(f"\n==================================================")
        print(f"[{idx+1}/{len(subdirs)}] 正在處理: {os.path.basename(folder)}")
        print(f"==================================================")

        # 檢查該資料夾下是否有 vision1~6 的影片或資料夾，若無則表示為新版命名
        has_vision = False
        try:
            for name in os.listdir(folder):
                if any(name.startswith(f"vision{i}") for i in range(1, 7)):
                    has_vision = True
                    break
        except Exception:
            pass

        cam_mapping = {
            "vision1": "vision1",
            "vision2": "vision2",
            "vision3": "vision3",
            "vision4": "vision4",
            "vision5": "vision5",
            "vision6": "vision6",
        }
        if not has_vision:
            print(f"[INFO] 偵測到新版影片命名規則 (RU, RR, RLU, FL, FR, RD)")
            cam_mapping = {
                "vision1": "RU",
                "vision2": "RR",
                "vision3": "RLU",
                "vision4": "FL",
                "vision5": "FR",
                "vision6": "RD",
            }
        else:
            print(f"[INFO] 偵測到舊版影片命名規則 (vision1~6)")

        # 依序計算配對外參
        for cam1_id, cam2_id in camera_pairs:
            if cam1_id not in intrinsics or cam2_id not in intrinsics:
                print(f"[SKIP] 跳過 {cam1_id} 與 {cam2_id}，因為缺少內參。")
                continue

            folder_name1 = cam_mapping[cam1_id]
            folder_name2 = cam_mapping[cam2_id]

            dir1 = os.path.join(folder, f"{folder_name1}_jpg")
            dir2 = os.path.join(folder, f"{folder_name2}_jpg")

            if not os.path.exists(dir1) or not os.path.exists(dir2):
                print(f"[SKIP] 跳過 {folder_name1} 與 {folder_name2}，因為找不到影像資料夾。")
                continue

            # 這裡 pattern_size 使用 (5, 3) 根據影像中的 6x4 格棋盤 (6x4 格 -> 5x3 內角點)
            results = calibrate_extrinsics(dir1, dir2, 
                                         intrinsics[cam1_id][0], intrinsics[cam1_id][1],
                                         intrinsics[cam2_id][0], intrinsics[cam2_id][1],
                                         folder_name1, folder_name2,
                                         pattern_size=(5, 3))

            if results:
                save_name = f"extrinsics_{cam1_id}_to_{cam2_id}.npz"
                save_path = os.path.join(folder, save_name)
                np.savez(save_path, R=results["R"], T=results["T"])
                print(f"[SAVE] 外參已儲存至: {save_path}")
            else:
                print(f"[FAIL] {cam1_id} 到 {cam2_id} 的校正失敗。")
