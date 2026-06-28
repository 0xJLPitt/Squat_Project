"""
檔案目的: 將平滑後的 3D 骨架重新投影回 4 個視角的 2D 影片上，產生 _projected.mp4 供驗證。
注意: 會自動呼叫 step8 裡的方法來讀取 3D 點。如果投影線條沒有貼合人體，通常是外參校正有問題。
呼叫指令: python step10_project_3d_to_video.py
"""
import numpy as np
import cv2
import os
from step8_triangulate_and_render_3d import (
    load_intrinsics, load_extrinsics, get_projection_matrices, load_2d_keypoints,
    triangulate_point, fix_left_right_swaps, remove_collapsed_joints, smooth_3d_keypoints, cams, DATA_DIR
)

def get_smoothed_3d_keypoints(P, kpts_2d):
    max_frames = min([kpts_2d[c].shape[0] for c in cams])
    kpts_3d = np.zeros((max_frames, 17, 3))
    
    print("Triangulating...")
    for f in range(max_frames):
        for j in range(17):
            views = []
            for cam in cams:
                x, y = kpts_2d[cam][f, j]
                if x != 0 and y != 0:
                    views.append((x, y, cam))
            kpts_3d[f, j] = triangulate_point(views, P)
            
    print("Fixing left/right joint swaps...")
    kpts_3d = fix_left_right_swaps(kpts_3d)
    
    print("Filtering out occlusion-collapsed joints...")
    kpts_3d = remove_collapsed_joints(kpts_3d)
    
    print("Applying temporal smoothing...")
    kpts_3d = smooth_3d_keypoints(kpts_3d, window_length=21, polyorder=3)
    return kpts_3d

def project_and_render(kpts_3d, P_matrices):
    connections = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
    ]
    
    for cam, name in cams.items():
        vid_path = os.path.join(DATA_DIR, f"{name}.avi")
        out_path = os.path.join(DATA_DIR, f"{name}_projected.mp4")
        if not os.path.exists(vid_path):
            print(f"Video not found: {vid_path}")
            continue
            
        print(f"Projecting 3D points onto {name}.avi...")
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        
        P = P_matrices[cam]
        num_frames = kpts_3d.shape[0]
        
        for f in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            pts = kpts_3d[f]
            valid_mask = np.sum(pts, axis=-1) != 0
            
            projected_pts = {}
            for j in range(17):
                if valid_mask[j]:
                    X = np.array([pts[j, 0], pts[j, 1], pts[j, 2], 1.0])
                    p = P @ X
                    if p[2] != 0:
                        u, v = int(p[0]/p[2]), int(p[1]/p[2])
                        # Only draw if within frame bounds roughly
                        if -1000 < u < w + 1000 and -1000 < v < h + 1000:
                            projected_pts[j] = (u, v)
                            cv2.circle(frame, (u, v), 6, (0, 0, 255), -1)
                        
            for j1, j2 in connections:
                if j1 in projected_pts and j2 in projected_pts:
                    cv2.line(frame, projected_pts[j1], projected_pts[j2], (255, 0, 0), 3)
                    
            out_video.write(frame)
            if f % 100 == 0:
                print(f"{name}: {f}/{num_frames} frames processed")
                
        cap.release()
        out_video.release()
        print(f"Saved {out_path}")

def main():
    print("Loading parameters...")
    intrinsics = load_intrinsics()
    extrinsics = load_extrinsics()
    P = get_projection_matrices(intrinsics, extrinsics)
    
    print("Loading 2D keypoints...")
    kpts_2d = load_2d_keypoints()
    
    kpts_3d = get_smoothed_3d_keypoints(P, kpts_2d)
    project_and_render(kpts_3d, P)
    
    print("All projections completed!")

if __name__ == "__main__":
    main()
