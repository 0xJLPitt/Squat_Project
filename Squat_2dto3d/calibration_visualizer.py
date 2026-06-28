"""
檔案目的: 提供外參校正結果的視覺化繪圖功能，將左右鏡頭的棋盤格特徵點畫在圖片上並水平合併。
注意: 這是一個函式庫檔案，提供 `save_stereo_visualization` 供其他腳本呼叫，不需獨立執行。
呼叫指令: (無，供 step3, step4, step5 引入呼叫)
"""
import cv2
import numpy as np
import os

def draw_calibration_points(img, corners, cam_id, pattern_size, is_manual=False):
    # Ensure image is color
    if len(img.shape) == 2:
        img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_vis = img.copy()

    # Draw the pink line for the first row
    w = pattern_size[0]
    if len(corners) >= w:
        pts = corners[:w].reshape(-1, 2).astype(int)
        for i in range(len(pts) - 1):
            cv2.line(img_vis, tuple(pts[i]), tuple(pts[i+1]), (255, 0, 255), 3)

    # Draw points and numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2

    for i, pt in enumerate(corners):
        x, y = int(pt[0][0]), int(pt[0][1])
        # Green circle with black outline
        cv2.circle(img_vis, (x, y), 6, (0, 0, 0), -1)
        cv2.circle(img_vis, (x, y), 4, (0, 255, 0), -1)
        
        # Text with black outline
        text = str(i + 1)
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = x + 10
        text_y = y - 10
        
        cv2.putText(img_vis, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
        cv2.putText(img_vis, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Draw banner
    h, w_img, _ = img_vis.shape
    banner_height = 40
    overlay = img_vis.copy()
    cv2.rectangle(overlay, (0, 0), (w_img, banner_height), (0, 0, 0), -1)
    # Apply transparency
    alpha = 0.6
    img_vis = cv2.addWeighted(overlay, alpha, img_vis, 1 - alpha, 0)
    
    # Banner text
    calib_type = "MANUAL CALIBRATION" if is_manual else "AUTO CALIBRATION (SB)"
    banner_text = f"{cam_id} | {calib_type} | {len(corners)} Points"
    cv2.putText(img_vis, banner_text, (10, 28), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return img_vis

def save_stereo_visualization(img1, img2, corners1, corners2, id1, id2, filepath, vis_dir, pattern_size=(5, 3), is_manual=False):
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        
    vis1 = draw_calibration_points(img1, corners1, id1.upper(), pattern_size, is_manual)
    vis2 = draw_calibration_points(img2, corners2, id2.upper(), pattern_size, is_manual)
    
    # Ensure both images have the same height for concatenation
    h1, w1 = vis1.shape[:2]
    h2, w2 = vis2.shape[:2]
    
    if h1 != h2:
        # Resize vis2 to match vis1 height
        ratio = h1 / float(h2)
        new_w2 = int(w2 * ratio)
        vis2 = cv2.resize(vis2, (new_w2, h1))
        
    combined = np.hstack((vis1, vis2))
    
    filename = os.path.basename(filepath)
    out_name = f"match_{id1}_{id2}_{filename}"
    out_path = os.path.join(vis_dir, out_name)
    
    # Use imencode and tofile to support Chinese paths on Windows
    is_success, im_buf_arr = cv2.imencode(".jpg", combined)
    if is_success:
        im_buf_arr.tofile(out_path)
    
    return out_path
