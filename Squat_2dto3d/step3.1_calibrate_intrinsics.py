"""
檔案目的: 執行單相機內參校正 (Camera Intrinsic Calibration)
透過 6x4 格 (5x3 內部角點) 棋盤格影像計算各攝影機 (i15, i16, i17 或 vision1~6) 的內參矩陣 (mtx) 與畸變係數 (dist)。
過濾機制: 
  1. 邊界過濾: 若角點太靠近或超出影像邊界 (margin < 15px) 則自動捨棄。
  2. 離群值/反向幀剔除: 自動計算每張影格之重投影誤差，剔除 > 0.35px 之異常幀。
  3. 視覺化繪製: 將成功參與校正的每張影格繪製角點標記，並輸出至 visualized 資料夾供人工檢驗。

輸出格式: .npz 檔案，包含 'mtx', 'dist', 'reproj_err', 'img_shape', 'num_frames'，與 step3.2_calibrate_extrinsics.py 完全相容。

呼叫指令:
  mamba run -n hw1 python step3.1_calibrate_intrinsics.py
"""

import os
import glob
import cv2
import numpy as np


def calibrate_single_camera(
    image_dir,
    cam_name,
    pattern_size=(5, 3),
    square_size=25.0,
    output_dir=None,
    vis_output_dir=None,
    edge_margin=15,
    max_reproj_thresh=0.35
):
    """
    計算單相機內參 (Intrinsic Calibration)
    :param image_dir: 存放棋盤格 .jpg 的資料夾路徑
    :param cam_name: 相機名稱 (例如 "i15", "i16", "i17")
    :param pattern_size: 內部角點數量 (寬, 高)，6x4格棋盤格內部點為 (5, 3)
    :param square_size: 棋盤格單格邊長 (mm)
    :param output_dir: 內參 .npz 儲存資料夾
    :param vis_output_dir: 視覺化標記圖片儲存資料夾
    :param edge_margin: 邊界保留像素，角點過度靠近或超出邊界直接捨棄
    :param max_reproj_thresh: 單幀最大容許重投影誤差 (px)，超過則剔除
    """
    files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not files:
        print(f"[{cam_name}] [ERROR] 找不到任何 .jpg 圖片: {image_dir}")
        return None

    # 定義棋盤格世界座標 (Z=0 平面)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    candidate_frames = []
    discarded_edge_count = 0
    img_shape = None

    print(f"\n==================================================")
    print(f"[SCAN] 開始處理相機 [{cam_name}]")
    print(f"       來源資料夾: {image_dir}")
    print(f"       總圖片數: {len(files)} 張 | 棋盤格內部角點規格: {pattern_size}")
    print(f"==================================================")

    for f in files:
        img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if img_shape is None:
            img_shape = (w, h)

        # 優先使用 SB (Sector Based) 演算法，對無白邊棋盤格最為精準
        ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ret:
            # 備援使用標準自適應演算法 + 亞像素精確化
            ret, corners = cv2.findChessboardCorners(
                gray, pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        if ret:
            # 1. 邊界範圍檢查: 若角點過度靠近或超出邊界則捨棄不用
            xs = corners[:, 0, 0]
            ys = corners[:, 0, 1]
            if np.any(xs < edge_margin) or np.any(xs > w - edge_margin) or np.any(ys < edge_margin) or np.any(ys > h - edge_margin):
                discarded_edge_count += 1
                continue

            candidate_frames.append({
                "file": f,
                "corners": corners,
                "img": img
            })

    print(f"[{cam_name}] 候選有效幀: {len(candidate_frames)} 張 (已過濾超出邊界/遮擋 {discarded_edge_count} 張)")

    if len(candidate_frames) < 5:
        print(f"[{cam_name}] [FAIL] 有效影格過少 (< 5 張)，無法進行可靠的內參校正。")
        return None

    # 2. 迭代離群值剔除校正 (排除旋轉反向或幾何扭曲造成的異常幀)
    cur_frames = candidate_frames[:]
    best_ret = 999.0
    best_mtx = None
    best_dist = None
    frame_errors = {}

    for iteration in range(5):
        cur_obj = [objp for _ in cur_frames]
        cur_img = [f["corners"] for f in cur_frames]

        ret_err, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            cur_obj, cur_img, img_shape, None, None
        )

        # 計算每張影格的重投影誤差
        errors = []
        for i, frame in enumerate(cur_frames):
            proj, _ = cv2.projectPoints(cur_obj[i], rvecs[i], tvecs[i], mtx, dist)
            err = float(np.sqrt(np.mean(np.sum((cur_img[i] - proj) ** 2, axis=2))))
            errors.append((err, frame))

        errors.sort(key=lambda x: x[0], reverse=True)
        max_err = errors[0][0]
        mean_err = np.mean([e[0] for e in errors])

        print(f"  [迭代 {iteration+1}] 保留影格: {len(cur_frames)} 張 | RMS: {ret_err:.4f} px | 平均誤差: {mean_err:.4f} px | 最大誤差: {max_err:.4f} px")

        best_ret = ret_err
        best_mtx = mtx
        best_dist = dist
        frame_errors = {f["file"]: err for (err, f) in errors}

        if max_err <= max_reproj_thresh or len(cur_frames) <= 10:
            break

        # 剔除誤差大於閥值的影格
        thresh = max(max_reproj_thresh, np.percentile([e[0] for e in errors], 85))
        filtered = [f for (err, f) in errors if err <= thresh]
        if len(filtered) == len(cur_frames):
            filtered = [f for (err, f) in errors[1:]]

        cur_frames = filtered

    print(f"[{cam_name}] [DONE] 內參校正成功！")
    print(f"       最終採用影格: {len(cur_frames)} / {len(files)} 張")
    print(f"       重投影誤差 (RMS Error): {best_ret:.4f} pixels")
    print(f"       焦距 (fx, fy): ({best_mtx[0, 0]:.2f}, {best_mtx[1, 1]:.2f})")
    print(f"       主點中心 (cx, cy): ({best_mtx[0, 2]:.2f}, {best_mtx[1, 2]:.2f})")
    print(f"       畸變係數 (k1, k2, p1, p2, k3):")
    print(f"       {best_dist.ravel()}")

    # 3. 儲存 .npz 檔案
    save_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"intrinsics_{cam_name}.npz")
        np.savez(
            save_path,
            mtx=best_mtx,
            dist=best_dist,
            reproj_err=best_ret,
            img_shape=np.array(img_shape),
            num_frames=len(cur_frames)
        )
        print(f"[{cam_name}] [SAVE] 已儲存內參矩陣至: {save_path}")

    # 4. 視覺化繪製並儲存處理後的圖片
    if vis_output_dir:
        os.makedirs(vis_output_dir, exist_ok=True)
        print(f"[{cam_name}] [VIS] 正在輸出 {len(cur_frames)} 張角點標記視覺化圖片至: {vis_output_dir}")
        for item in cur_frames:
            f_path = item["file"]
            corners = item["corners"]
            vis_img = item["img"].copy()

            # 繪製角點與格線
            cv2.drawChessboardCorners(vis_img, pattern_size, corners, True)

            # 在圖片上方繪製資訊條
            f_name = os.path.basename(f_path)
            err_val = frame_errors.get(f_path, 0.0)
            overlay = vis_img.copy()
            cv2.rectangle(overlay, (0, 0), (vis_img.shape[1], 42), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, vis_img, 0.35, 0, vis_img)

            info_text = f"Cam: {cam_name} | Frame: {f_name} | Pattern: {pattern_size} | Reproj Error: {err_val:.3f} px"
            cv2.putText(vis_img, info_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            out_img_name = f"vis_{f_name}"
            out_img_path = os.path.join(vis_output_dir, out_img_name)
            cv2.imencode(".jpg", vis_img)[1].tofile(out_img_path)

        print(f"[{cam_name}] [VIS] 視覺化圖片輸出完畢！")

    return {
        "cam_name": cam_name,
        "mtx": best_mtx,
        "dist": best_dist,
        "reproj_err": best_ret,
        "num_frames": len(cur_frames),
        "img_shape": img_shape,
        "save_path": save_path
    }


if __name__ == "__main__":
    # 相機路徑設定
    CAM_CONFIGS = [
        {
            "name": "i15",
            "dir": r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\i15\checkboard_internal_jpg"
        },
        {
            "name": "i16",
            "dir": r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\i16\checkboard_internal16_jpg"
        },
        {
            "name": "i17",
            "dir": r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\i17\checkboard_internal_jpg"
        }
    ]

    # 統一集中儲存的內參資料夾
    OUTPUT_INTRINSIC_DIR = r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\intrinsics"
    
    # 棋盤格內部角點 (6x4 格棋盤 -> 5x3 內部角點)
    PATTERN_SIZE = (5, 3)
    SQUARE_SIZE = 25.0

    all_results = {}
    for cam in CAM_CONFIGS:
        vis_folder = os.path.join(OUTPUT_INTRINSIC_DIR, "visualized", cam["name"])
        res = calibrate_single_camera(
            image_dir=cam["dir"],
            cam_name=cam["name"],
            pattern_size=PATTERN_SIZE,
            square_size=SQUARE_SIZE,
            output_dir=OUTPUT_INTRINSIC_DIR,
            vis_output_dir=vis_folder,
            edge_margin=15,
            max_reproj_thresh=0.35
        )
        if res:
            all_results[cam["name"]] = res

            # 同步儲存一份至各相機獨立目錄
            parent_dir = os.path.dirname(cam["dir"])
            single_save_path = os.path.join(parent_dir, f"intrinsics_{cam['name']}.npz")
            np.savez(
                single_save_path,
                mtx=res["mtx"],
                dist=res["dist"],
                reproj_err=res["reproj_err"],
                img_shape=np.array(res["img_shape"]),
                num_frames=res["num_frames"]
            )
            print(f"[{cam['name']}] [SAVE] 同步儲存至各相機目錄: {single_save_path}")

    print("\n" + "=" * 65)
    print("所有相機內參校正彙總報告 (6x4格 / 5x3內部角點):")
    print("=" * 65)
    for cam_name, res in all_results.items():
        print(f"相機 {cam_name:5s} | 採納影格: {res['num_frames']:2d} 張 | RMS 誤差: {res['reproj_err']:.4f} px | 儲存路徑: {res['save_path']}")
