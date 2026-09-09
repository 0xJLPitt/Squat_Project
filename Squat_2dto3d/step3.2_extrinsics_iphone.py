"""
檔案目的: 執行雙相機外參立體校正 (Stereo Extrinsic Calibration)
透過同步拍攝的棋盤格影像，計算各相機間 (如 i15, i16, i17 或 vision2~5) 的相對旋轉矩陣 (R) 與平移向量 (T)。

過濾與對齊機制:
  1. 棋盤格規格: 6x4 格 (5x3 內部角點)，單格邊長 25.0 mm。
  2. 視角適配:
     - i15: 採用中央區域 ROI 偵測，避免深蹲架雜訊干擾，精確擷取 15 個內部角點。
     - i16: 因視角位於對向，棋盤格物理順序在 OpenCV 偵測時相差 180 度，自動進行角點反轉 (c[::-1]) 完美對齊。
     - i17: 採用 SB 演算法直接高精確度辨識。
  3. 視覺化對位圖: 呼叫 calibration_visualizer 輸出左右相機 1~15 點同步對照圖，供肉眼檢驗空間對齊。

輸出格式: .npz 檔案，包含 'R', 'T', 'rms_error', 'baseline_mm', 'common_frames'。

呼叫指令:
  mamba run -n hw1 python step3.2_extrinsics_iphone.py
"""

import os
import glob
import cv2
import numpy as np

# 載入立體校正視覺化函式庫
try:
    from calibration_visualizer import save_stereo_visualization
except ImportError:
    try:
        from Squat_2dto3d.calibration_visualizer import save_stereo_visualization
    except ImportError:
        save_stereo_visualization = None


# ==============================================================================
# 【對向相機 180 度點序反轉設定】 (Opposite Camera Configuration)
# ==============================================================================
# 當相機為對向拍攝時，OpenCV 偵測點序在空間中相差 180 度 (點 1 <-> 點 15)。
# 臥推 iPhone 視角配置: i15 與 i16 位於同側，i17 位於對向。
# 因此與 i17 配對時 (i15-i17, i16-i17)，i17 需進行 180 度點序翻轉以完美對齊。
#
# 【設定 A】對向相機清單:
OPPOSITE_CAMERAS = ["i17"]

# 【設定 B】特定配對反轉規則:
# 格式: (相機1, 相機2): "需要反轉的相機名稱" (若都不反轉則設為 None)
PAIR_FLIP_RULES = {
    ("i15", "i17"): "i17",  # i15 與 i17 配對時，反轉 i17
    ("i16", "i17"): "i17",  # i16 與 i17 配對時，反轉 i17
    ("i15", "i16"): None,   # i15 與 i16 為同側，皆不反轉
}

def should_flip_camera(cam_name, partner_name=None):
    if not cam_name:
        return False
    if partner_name:
        pair = (cam_name, partner_name)
        pair_inv = (partner_name, cam_name)
        if pair in PAIR_FLIP_RULES:
            return PAIR_FLIP_RULES[pair] == cam_name
        elif pair_inv in PAIR_FLIP_RULES:
            return PAIR_FLIP_RULES[pair_inv] == cam_name
    return cam_name in OPPOSITE_CAMERAS
# ==============================================================================


def detect_corners_for_frame(img_bgr, cam_name, pattern_size=(5, 3), partner_name=None):
    """
    根據各相機視角特徵提取棋盤格角點
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. i15 視角最佳化: 因拍攝視角深蹲架雜訊較多，對中央躺板區進行精確 ROI (260:540, 400:800) 偵測
    if cam_name == "i15":
        roi_y1, roi_y2 = 260, 540
        roi_x1, roi_x2 = 400, 800
        crop = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        ret, corners = cv2.findChessboardCornersSB(crop, pattern_size, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            corners[:, 0, 0] += roi_x1
            corners[:, 0, 1] += roi_y1
            return True, corners

    # 2. 通用與 i16, i17 偵測: 優先 SB 演算法
    ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ret:
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    if ret:
        # 依據對向相機設定進行 180 度翻轉以確保左右鏡頭物理點序 100% 一致
        if should_flip_camera(cam_name, partner_name):
            corners = corners[::-1, :, :].copy()
        return True, corners

    return False, None


def calibrate_stereo_pair(
    dir1, dir2,
    mtx1, dist1, mtx2, dist2,
    id1, id2,
    pattern_size=(5, 3),
    square_size=25.0,
    vis_dir=None
):
    """
    計算一對相機之間的相對外參 (Stereo Calibration)
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    files1 = sorted([f for f in os.listdir(dir1) if f.endswith(".jpg")])
    files2_set = set(os.listdir(dir2))

    print(f"\n==================================================")
    print(f"[STEREO SCAN] 正在搜尋 {id1} 與 {id2} 的同步棋盤格...")
    print(f"==================================================")

    obj_points = []
    img_points1 = []
    img_points2 = []
    synced_filenames = []

    # 尋找同步幀 (依據 frameXXXXX 索引對位)
    for f1 in files1:
        # 提取 frame 編號，例如 frame00000
        import re
        match = re.search(r"(frame\d+)", f1)
        if not match:
            continue
        frame_tag = match.group(1)

        # 在 dir2 中搜尋相同 frame 編號的圖檔
        f2_candidates = [f for f in files2_set if frame_tag in f and f.endswith(".jpg")]
        if not f2_candidates:
            continue
        f2 = f2_candidates[0]

        p1 = os.path.join(dir1, f1)
        p2 = os.path.join(dir2, f2)

        img1 = cv2.imdecode(np.fromfile(p1, dtype=np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.fromfile(p2, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img1 is None or img2 is None:
            continue

        ret1, c1 = detect_corners_for_frame(img1, id1, pattern_size, partner_name=id2)
        ret2, c2 = detect_corners_for_frame(img2, id2, pattern_size, partner_name=id1)

        if ret1 and ret2:
            print(f"[OK] 發現同步特徵影格: {f1} <--> {f2}")
            obj_points.append(objp)
            img_points1.append(c1)
            img_points2.append(c2)
            synced_filenames.append(f1)

            # 儲存左右相機角點對照圖
            if vis_dir and save_stereo_visualization:
                try:
                    save_stereo_visualization(
                        img1, img2, c1, c2,
                        id1, id2, f1, vis_dir,
                        pattern_size=pattern_size,
                        is_manual=False
                    )
                except Exception as e:
                    print(f"[WARN] 儲存對照可視化圖失敗: {e}")

    print(f"[{id1} <-> {id2}] 同步有效影格數: {len(obj_points)} 張")

    if len(obj_points) < 3:
        print(f"[FAIL] 同步影格不足 3 張，無法可靠計算外參。")
        return None

    # 執行雙相機立體標定 (固定內參)
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points1, img_points2,
        mtx1, dist1, mtx2, dist2, (1280, 720), flags=flags
    )

    baseline_mm = float(np.linalg.norm(T))

    print(f"[{id1} -> {id2}] [DONE] 外參校正成功！")
    print(f"       立體重投影誤差 (RMS): {ret:.4f} pixels")
    print(f"       相機基線距離 (Baseline): {baseline_mm:.1f} mm ({baseline_mm/10:.1f} cm)")
    print(f"       旋轉矩陣 R:\n{R}")
    print(f"       平移向量 T (mm):\n{T.ravel()}")

    return {
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "rms_error": ret,
        "baseline_mm": baseline_mm,
        "common_frames": len(obj_points),
        "synced_files": synced_filenames
    }


if __name__ == "__main__":
    # ================= 設定路徑 =================
    INTRINSIC_DIR = r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\intrinsics"
    OUTPUT_EXTRINSIC_DIR = r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\extrinsics"
    VIS_DIR = os.path.join(OUTPUT_EXTRINSIC_DIR, "visualized")

    # 相機影像資料夾
    CAMERA_DIRS = {
        "i15": r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\i15\checkboard_external_jpg",
        "i16": r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\i16\checkboard_external1_jpg",
        "i17": r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\i17\checkboard_external1_jpg",
    }

    # 欲計算外參的配對
    CAMERA_PAIRS = [
        ("i15", "i16"),
        ("i16", "i17"),
        ("i15", "i17"),
    ]

    PATTERN_SIZE = (5, 3)
    SQUARE_SIZE = 25.0
    # ============================================

    os.makedirs(OUTPUT_EXTRINSIC_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    # 讀取內參檔案
    intrinsics = {}
    for cam in ["i15", "i16", "i17"]:
        npz_path = os.path.join(INTRINSIC_DIR, f"intrinsics_{cam}.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            intrinsics[cam] = (data["mtx"], data["dist"])
            print(f"[INFO] 成功載入相機 {cam} 內參: {npz_path}")
        else:
            print(f"[ERROR] 找不到內參檔案: {npz_path}")

    all_extrinsics = {}
    for cam1, cam2 in CAMERA_PAIRS:
        if cam1 not in intrinsics or cam2 not in intrinsics:
            print(f"[SKIP] 跳過 {cam1} -> {cam2}，因缺少內參。")
            continue

        dir1 = CAMERA_DIRS.get(cam1)
        dir2 = CAMERA_DIRS.get(cam2)

        if not dir1 or not dir2 or not os.path.exists(dir1) or not os.path.exists(dir2):
            print(f"[SKIP] 跳過 {cam1} -> {cam2}，找不到影像目錄。")
            continue

        mtx1, dist1 = intrinsics[cam1]
        mtx2, dist2 = intrinsics[cam2]

        res = calibrate_stereo_pair(
            dir1, dir2,
            mtx1, dist1, mtx2, dist2,
            cam1, cam2,
            pattern_size=PATTERN_SIZE,
            square_size=SQUARE_SIZE,
            vis_dir=VIS_DIR
        )

        if res:
            all_extrinsics[f"{cam1}_to_{cam2}"] = res

            # 儲存外參 .npz (存於 extrinsics 集中目錄及 benchpress_3D 根目錄)
            save_name = f"extrinsics_{cam1}_to_{cam2}.npz"
            save_path1 = os.path.join(OUTPUT_EXTRINSIC_DIR, save_name)
            save_path2 = os.path.join(r"D:\Pitt\Project\Squat_Project\video\benchpress_3D", save_name)

            np.savez(
                save_path1,
                R=res["R"],
                T=res["T"],
                E=res["E"],
                F=res["F"],
                rms_error=res["rms_error"],
                baseline_mm=res["baseline_mm"],
                common_frames=res["common_frames"]
            )
            np.savez(
                save_path2,
                R=res["R"],
                T=res["T"],
                E=res["E"],
                F=res["F"],
                rms_error=res["rms_error"],
                baseline_mm=res["baseline_mm"],
                common_frames=res["common_frames"]
            )
            print(f"[SAVE] 外參已儲存至:\n  -> {save_path1}\n  -> {save_path2}")

    print("\n" + "=" * 70)
    print("雙相機外參校正 (Stereo Calibration) 彙總報告:")
    print("=" * 70)
    for pair_name, res in all_extrinsics.items():
        print(f"配對 {pair_name:12s} | 同步影格: {res['common_frames']:2d} 張 | RMS 誤差: {res['rms_error']:.4f} px | 相機基線: {res['baseline_mm']:.1f} mm")
