"""
檔案目的: 3D 空間相機位置視覺化，將計算出來的相機外參 (位置與朝向) 畫在 3D 座標系中供檢查。
注意: 觀察 matplotlib 顯示的 3D 圖表時，請確認相機的相對擺放位置是否符合真實世界 (例如 RR 和 FL 應該是面對面的)。
呼叫指令: python step7_visualize_extrinsics.py
"""
import cv2
import numpy as np
import os
import glob

def draw_outlined_text(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.5, color=(255, 255, 0), thickness=1):
    """Draw text with a black outline for maximum legibility on any background."""
    # Outline (black, thick)
    cv2.putText(img, text, pos, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Text (colored, thin)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)

def visualize_single_pair(folder_path, cam1_id, cam2_id, mtx1, dist1, mtx2, dist2, pattern_size=(5, 3), square_size=25.0):
    """
    載入外參並對同步影像進行重投影，生成一張側並側 (side-by-side) 的視覺化比對圖，並計算重投影誤差。
    """
    folder_name = os.path.basename(folder_path)
    extrinsic_name = f"extrinsics_{cam1_id}_to_{cam2_id}.npz"
    extrinsic_path = os.path.join(folder_path, extrinsic_name)
    
    if not os.path.exists(extrinsic_path):
        print(f"[SKIP] {folder_name}: 找不到外參檔案 {extrinsic_name}")
        return None, None

    # 讀取外參 R, T
    try:
        ext_data = np.load(extrinsic_path)
        R = ext_data["R"]
        T = ext_data["T"]
    except Exception as e:
        print(f"[ERROR] 讀取外參失敗 {extrinsic_path}: {e}")
        return None, None

    dir1 = os.path.join(folder_path, f"{cam1_id}_jpg")
    dir2 = os.path.join(folder_path, f"{cam2_id}_jpg")
    
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        print(f"[SKIP] {folder_name}: 找不到影像資料夾 {cam1_id}_jpg 或 {cam2_id}_jpg")
        return None, None

    # 取得同步影像對
    files1 = sorted([f for f in os.listdir(dir1) if f.endswith(".jpg")])
    
    best_vis = None
    best_error = float('inf')
    
    # 遍歷同步影像，找出一張能夠成功偵測棋盤格且誤差最優 (或第一張) 的影像
    for f1 in files1:
        f2 = f1.replace(f"{cam1_id}_", f"{cam2_id}_")
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
        
        ret1, corners1 = cv2.findChessboardCornersSB(gray1, pattern_size, None)
        ret2, corners2 = cv2.findChessboardCornersSB(gray2, pattern_size, None)
        
        if ret1 and ret2:
            # 準備棋盤格3D座標
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
            objp *= square_size
            
            # 使用 solvePnP 估算棋盤格在 Cam1 坐標系下的 3D 位置
            retval, rvec1, tvec1 = cv2.solvePnP(objp, corners1, mtx1, dist1)
            if not retval:
                continue
                
            R_cam1, _ = cv2.Rodrigues(rvec1)
            pts3d_cam1 = (R_cam1 @ objp.T + tvec1).T  # (N, 3)
            
            # 利用外參 [R|T] 將點投影至 Cam2 坐標系下
            # P_cam2 = R * P_cam1 + T
            pts3d_cam2 = (R @ pts3d_cam1.T + T).T  # (N, 3)
            
            # 投影至 Cam2 影像平面
            projected_pts2, _ = cv2.projectPoints(pts3d_cam2, np.zeros((3, 1)), np.zeros((3, 1)), mtx2, dist2)
            projected_pts2 = projected_pts2.reshape(-1, 2)
            
            # 計算平均投影誤差 (像素距離)
            corners2_pts = corners2.reshape(-1, 2)
            errors = np.linalg.norm(projected_pts2 - corners2_pts, axis=1)
            mean_error = np.mean(errors)
            
            # 繪製視覺化
            vis1 = img1.copy()
            vis2 = img2.copy()
            
            # 繪製 Cam1 棋盤格角點 (綠色圓點 + 編號)
            for idx, pt in enumerate(corners1.reshape(-1, 2)):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(vis1, (x, y), 5, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(vis1, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)
                draw_outlined_text(vis1, str(idx+1), (x + 6, y - 6), scale=0.35, color=(0, 255, 255))
                
            # 繪製 Cam2: 
            # 1. 紅色叉號代表 Cam2 實際偵測的點
            # 2. 綠色圈代表從 Cam1 重投影過來的點
            # 3. 黃色線連接兩者表示誤差
            for idx, (pt_real, pt_proj) in enumerate(zip(corners2_pts, projected_pts2)):
                xr, yr = int(pt_real[0]), int(pt_real[1])
                xp, yp = int(pt_proj[0]), int(pt_proj[1])
                
                # 實測點 (紅色十字)
                cv2.line(vis2, (xr - 6, yr), (xr + 6, yr), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(vis2, (xr, yr - 6), (xr, yr + 6), (0, 0, 255), 1, cv2.LINE_AA)
                
                # 重投影點 (綠色圓圈)
                cv2.circle(vis2, (xp, yp), 5, (0, 255, 0), 1, cv2.LINE_AA)
                
                # 誤差連線
                cv2.line(vis2, (xr, yr), (xp, yp), (0, 255, 255), 1, cv2.LINE_AA)
                
                # 繪製編號
                draw_outlined_text(vis2, str(idx+1), (xp + 6, yp - 6), scale=0.35, color=(255, 255, 255))

            # 建立上方的半透明資訊欄
            banner_h = 40
            for vis, cam_id, info_str in [
                (vis1, cam1_id, f"{cam1_id.upper()} (Source Image)"),
                (vis2, cam2_id, f"{cam2_id.upper()} (Reprojected from {cam1_id.upper()}) | MRE: {mean_error:.3f} px")
            ]:
                overlay = vis.copy()
                cv2.rectangle(overlay, (0, 0), (vis.shape[1], banner_h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
                draw_outlined_text(vis, f"{folder_name} | {info_str}", (15, 25), scale=0.45, color=(255, 255, 255))

            # 拼接
            h1, w1 = vis1.shape[:2]
            h2, w2 = vis2.shape[:2]
            target_h = max(h1, h2)
            if h1 != target_h:
                vis1 = cv2.resize(vis1, (int(w1 * target_h / h1), target_h))
            if h2 != target_h:
                vis2 = cv2.resize(vis2, (int(w2 * target_h / h2), target_h))
                
            combined = np.hstack([vis1, vis2])
            
            # 我們只需要一張好圖，找誤差最小的，或者直接回傳第一張
            if mean_error < best_error:
                best_error = mean_error
                best_vis = combined
                
        # 只要找到一個可用的就先回傳，加快速度
        if best_vis is not None:
            break
            
    return best_vis, best_error

def main():
    ROOT_DIR = r"E:\squat\recordings_20260507\recording_20260325_153154_棋盤_有調整相機"
    INTRINSIC_DIR = r"E:\squat\recordings_20260507\intrinsics_vision"
    OUTPUT_DIR = r"E:\squat\recordings_20260507\visual_resul"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 內參路徑
    intrinsic_paths = {
        "vision2": os.path.join(INTRINSIC_DIR, "intrinsics_vision2.npz"),
        "vision3": os.path.join(INTRINSIC_DIR, "intrinsics_vision3.npz"),
        "vision4": os.path.join(INTRINSIC_DIR, "intrinsics_vision4.npz"),
        "vision5": os.path.join(INTRINSIC_DIR, "intrinsics_vision5.npz"),
    }
    
    # 載入內參
    intrinsics = {}
    for cam_id, path in intrinsic_paths.items():
        if os.path.exists(path):
            data = np.load(path)
            intrinsics[cam_id] = (data["mtx"], data["dist"])
            print(f"[INFO] 已載入 {cam_id} 的內參")
        else:
            print(f"[WARN] 找不到 {cam_id} 的內參檔案: {path}")
            
    camera_pairs = [
        ("vision2", "vision3", "vision2to3"),
        ("vision3", "vision4", "vision3to4"),
        ("vision4", "vision5", "vision4to5")
    ]
    
    # 搜尋所有子資料夾
    subdirs = []
    for item in sorted(os.listdir(ROOT_DIR)):
        item_path = os.path.join(ROOT_DIR, item)
        if os.path.isdir(item_path):
            if item == "intrinsics_vision" or "visual" in item:
                continue
            # 只要子資料夾內包含任一 visionX_jpg 資料夾即視為錄影資料夾
            if any(os.path.exists(os.path.join(item_path, f"{cam}_jpg")) for cam in ["vision2", "vision3", "vision4", "vision5"]):
                subdirs.append(item_path)
                
    print(f"[INFO] 找到待驗證的子資料夾數量: {len(subdirs)}")
    
    for idx, folder in enumerate(subdirs):
        folder_name = os.path.basename(folder)
        print(f"\n[{idx+1}/{len(subdirs)}] 正在處理資料夾: {folder_name}")
        
        for cam1, cam2, pair_suffix in camera_pairs:
            if cam1 not in intrinsics or cam2 not in intrinsics:
                continue
                
            mtx1, dist1 = intrinsics[cam1]
            mtx2, dist2 = intrinsics[cam2]
            
            vis_img, err = visualize_single_pair(
                folder, cam1, cam2, mtx1, dist1, mtx2, dist2,
                pattern_size=(5, 3), square_size=25.0
            )
            
            if vis_img is not None:
                # 輸出格式: [資料夾名稱]_[對應配對名稱].jpg
                out_filename = f"{folder_name}_{pair_suffix}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_filename)
                
                # 寫入檔案
                success, img_encoded = cv2.imencode(".jpg", vis_img)
                if success:
                    img_encoded.tofile(out_path)
                    print(f"  -> [SUCCESS] 輸出視覺化圖至: {out_path} (MRE: {err:.3f} px)")
                else:
                    print(f"  -> [ERROR] 影像編碼失敗")
            else:
                print(f"  -> [FAILED] 無法生成 {cam1} 到 {cam2} 的投影驗證圖 (可能缺少對應外參或同步棋盤格)")

if __name__ == "__main__":
    main()
