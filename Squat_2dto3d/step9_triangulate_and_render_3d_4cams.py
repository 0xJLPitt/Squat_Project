"""
檔案目的: 專門只挑選 4 個特定鏡頭進行 3D 三角測量。
注意: 確保指定的 4 個攝影機都有對應的 2D 骨架資料與外參檔案。
呼叫指令: python step9_triangulate_and_render_3d_4cams.py
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Configure paths
INTRINSIC_DIR = r"E:\squat\recordings_20260507\intrinsics_vision"
EXTRINSIC_DIR = r"E:\squat\recordings_20260507_done\recording_20260211_棋盤格"
DATA_DIR = r"E:\squat\squat_dataset\S001_Pitt\session02\recording_20260207_163410"

# Cameras used for reconstruction and their YOLO text files
cams = {
    "vision2": "RR",
    "vision3": "RLU",
    "vision4": "FL",
    "vision5": "FR",
}

def load_intrinsics():
    intrinsics = {}
    for cam, name in cams.items():
        path1 = os.path.join(INTRINSIC_DIR, f"intrinsics_{cam}.npz")
        path2 = os.path.join(INTRINSIC_DIR, f"intrinsics_{name}.npz")
        path = path1 if os.path.exists(path1) else path2
        if not os.path.exists(path):
            raise FileNotFoundError(f"Intrinsic not found for {cam}")
        data = np.load(path)
        intrinsics[cam] = data['mtx']
    return intrinsics

def load_extrinsics():
    extr = {}
    for pair in [("vision2", "vision3"), ("vision3", "vision4"), ("vision4", "vision5")]:
        path = os.path.join(EXTRINSIC_DIR, f"extrinsics_{pair[0]}_to_{pair[1]}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Extrinsic not found: {path}")
        data = np.load(path)
        extr[pair] = {"R": data["R"], "T": data["T"].reshape(3, 1)}
    return extr

def get_projection_matrices(intrinsics, extr):
    # World origin at vision2
    R_global = {"vision2": np.eye(3)}
    T_global = {"vision2": np.zeros((3, 1))}
    
    # Compute global extrinsics recursively
    pairs = [("vision2", "vision3"), ("vision3", "vision4"), ("vision4", "vision5")]
    for c1, c2 in pairs:
        R_rel = extr[(c1, c2)]["R"]
        T_rel = extr[(c1, c2)]["T"]
        R_global[c2] = R_rel @ R_global[c1]
        T_global[c2] = R_rel @ T_global[c1] + T_rel
        
    P = {}
    for cam in cams:
        K = intrinsics[cam]
        Rt = np.hstack((R_global[cam], T_global[cam]))
        P[cam] = K @ Rt
    return P

def load_2d_keypoints():
    kpts_2d = {}
    for cam, name in cams.items():
        txt_path = os.path.join(DATA_DIR, f"yolo_skeleton_{name}.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"2D keypoints not found: {txt_path}")
        
        data = np.loadtxt(txt_path, delimiter=',')
        # data format: frame(1-indexed), joint, x, y
        if len(data) == 0:
            continue
        max_frame = int(np.max(data[:, 0]))
        # We'll make it 0-indexed frame array
        arr = np.zeros((max_frame, 17, 2))
        for row in data:
            f = int(row[0]) - 1
            j = int(row[1])
            x, y = row[2], row[3]
            arr[f, j] = [x, y]
        kpts_2d[cam] = arr
    return kpts_2d

def triangulate_point(views, proj_matrices):
    # views: list of (x, y, cam_id)
    if len(views) < 2:
        return np.array([0.0, 0.0, 0.0])
        
    A = []
    for x, y, cam in views:
        P = proj_matrices[cam]
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if X[3] == 0:
        return np.array([0.0, 0.0, 0.0])
    return X[:3] / X[3]

def render_3d_video(kpts_3d, output_mp4):
    num_frames = kpts_3d.shape[0]
    
    # Define COCO connections
    connections = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
    ]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    valid_pts = kpts_3d[np.sum(kpts_3d, axis=-1) != 0]
    if len(valid_pts) > 0:
        # We remap coordinates for plotting: plot_x = X, plot_y = Z, plot_z = -Y
        # This makes the person stand upright.
        px = valid_pts[:, 0]
        py = valid_pts[:, 2]
        pz = -valid_pts[:, 1]
        
        x_min, x_max = np.min(px), np.max(px)
        y_min, y_max = np.min(py), np.max(py)
        z_min, z_max = np.min(pz), np.max(pz)
        
        # Calculate max range to keep 1:1:1 aspect ratio
        max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
        
        mid_x = (x_max + x_min) / 2.0
        mid_y = (y_max + y_min) / 2.0
        mid_z = (z_max + z_min) / 2.0
    else:
        mid_x, mid_y, mid_z = 0, 0, 0
        max_range = 1000
        
    margin = max_range * 0.2
    max_range += margin
    
    print(f"Rendering {num_frames} frames to {output_mp4} using OpenCV...")
    
    # Prepare cv2 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = None
    
    for f_idx in range(num_frames):
        ax.clear()
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Try to set box aspect if supported by the matplotlib version
        try:
            ax.set_box_aspect([1,1,1])
        except AttributeError:
            pass
            
        ax.set_xlabel('X (Right)')
        ax.set_ylabel('Depth')
        ax.set_zlabel('Y (Up)')
        
        # Set viewing angle for better perspective
        ax.view_init(elev=20., azim=-60)
        
        pts = kpts_3d[f_idx]
        valid_mask = np.sum(pts, axis=-1) != 0
        v_pts = pts[valid_mask]
        
        if len(v_pts) > 0:
            # Remap for scatter
            ax.scatter(v_pts[:, 0], v_pts[:, 2], -v_pts[:, 1], c='r', s=20)
            
        for i, (j1, j2) in enumerate(connections):
            if valid_mask[j1] and valid_mask[j2]:
                xs = [pts[j1, 0], pts[j2, 0]]
                ys = [pts[j1, 2], pts[j2, 2]]
                zs = [-pts[j1, 1], -pts[j2, 1]]
                ax.plot(xs, ys, zs, 'b-')
                
        fig.canvas.draw()
        
        # Convert to numpy array
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if out_video is None:
            h, w = img.shape[:2]
            out_video = cv2.VideoWriter(output_mp4, fourcc, 30.0, (w, h))
            
        out_video.write(img)
        
        if f_idx % 100 == 0:
            print(f"Rendered {f_idx}/{num_frames} frames...")
            
    if out_video is not None:
        out_video.release()
    plt.close(fig)
    print("Done!")

from scipy.signal import savgol_filter

def smooth_3d_keypoints(kpts_3d, window_length=15, polyorder=3):
    num_frames, num_joints, dims = kpts_3d.shape
    smoothed = np.copy(kpts_3d)
    
    for j in range(num_joints):
        for d in range(dims):
            series = kpts_3d[:, j, d]
            missing = (series == 0.0)
            valid = ~missing
            
            # Interpolate missing points if there are any valid points
            if valid.any() and missing.any():
                valid_indices = np.where(valid)[0]
                missing_indices = np.where(missing)[0]
                # Linear interpolation for missing frames
                series[missing_indices] = np.interp(missing_indices, valid_indices, series[valid_indices])
            
            # Apply Savitzky-Golay filter if we have enough frames
            if num_frames > window_length:
                smoothed[:, j, d] = savgol_filter(series, window_length, polyorder)
                
    return smoothed

def fix_left_right_swaps(kpts_3d):
    pairs = [
        (15, 16), # Ankles
        (13, 14), # Knees
        (11, 12), # Hips (If shoulders are valid to fix hips)
        (9, 10),  # Wrists
        (7, 8),   # Elbows
        (5, 6)    # Shoulders (If hips are valid to fix shoulders)
    ]
    
    fixed_kpts = np.copy(kpts_3d)
    num_frames = fixed_kpts.shape[0]
    
    for f in range(num_frames):
        l_hip, r_hip = fixed_kpts[f, 11], fixed_kpts[f, 12]
        l_sh, r_sh = fixed_kpts[f, 5], fixed_kpts[f, 6]
        
        # Define Left-Right axis vector (pointing towards Left)
        axis_vec = np.zeros(3)
        if np.sum(l_hip) != 0 and np.sum(r_hip) != 0:
            axis_vec += (l_hip - r_hip)
        if np.sum(l_sh) != 0 and np.sum(r_sh) != 0:
            axis_vec += (l_sh - r_sh)
            
        if np.sum(axis_vec) == 0:
            continue # Can't determine L/R axis for this frame
            
        for l_idx, r_idx in pairs:
            curr_l = fixed_kpts[f, l_idx]
            curr_r = fixed_kpts[f, r_idx]
            
            if np.sum(curr_l) == 0 or np.sum(curr_r) == 0:
                continue
                
            joint_vec = curr_l - curr_r
            
            # If dot product is negative, the left joint is actually on the right side!
            # Since in a squat, legs and arms don't cross the body centerline, this means they are swapped.
            if np.dot(joint_vec, axis_vec) < 0:
                fixed_kpts[f, l_idx] = curr_r
                fixed_kpts[f, r_idx] = curr_l
                
                # If we just swapped hips or shoulders, we should update the axis_vec to remain accurate
                if (l_idx, r_idx) in [(11, 12), (5, 6)]:
                    # Recompute axis_vec
                    l_hip, r_hip = fixed_kpts[f, 11], fixed_kpts[f, 12]
                    l_sh, r_sh = fixed_kpts[f, 5], fixed_kpts[f, 6]
                    axis_vec = np.zeros(3)
                    if np.sum(l_hip) != 0 and np.sum(r_hip) != 0:
                        axis_vec += (l_hip - r_hip)
                    if np.sum(l_sh) != 0 and np.sum(r_sh) != 0:
                        axis_vec += (l_sh - r_sh)
                        
    return fixed_kpts

def main():
    print("Loading parameters...")
    intrinsics = load_intrinsics()
    extrinsics = load_extrinsics()
    P = get_projection_matrices(intrinsics, extrinsics)
    
    print("Loading 2D keypoints...")
    kpts_2d = load_2d_keypoints()
    
    # Determine number of frames
    max_frames = min([kpts_2d[c].shape[0] for c in cams])
    print(f"Total synchronized frames: {max_frames}")
    
    kpts_3d = np.zeros((max_frames, 17, 3))
    
    print("Triangulating...")
    for f in range(max_frames):
        for j in range(17):
            views = []
            for cam in cams:
                x, y = kpts_2d[cam][f, j]
                
                # Filter out self-occluded joints where YOLO merges Left/Right to the same pixel
                if j in [13, 14, 15, 16]:
                    l_idx = 15 if j >= 15 else 13
                    r_idx = 16 if j >= 15 else 14
                    x_l, y_l = kpts_2d[cam][f, l_idx]
                    x_r, y_r = kpts_2d[cam][f, r_idx]
                    if x_l != 0 and x_r != 0:
                        pixel_dist = np.sqrt((x_l - x_r)**2 + (y_l - y_r)**2)
                        # If left and right joints are within 30 pixels, it's a self-occlusion merging error.
                        # Do not trust this camera for this joint.
                        if pixel_dist < 30.0:
                            continue
                            
                if x != 0 and y != 0:
                    views.append((x, y, cam))
            kpts_3d[f, j] = triangulate_point(views, P)
            
    print("Fixing left/right joint swaps...")
    kpts_3d = fix_left_right_swaps(kpts_3d)
    
    print("Applying temporal smoothing (Savitzky-Golay filter)...")
    kpts_3d = smooth_3d_keypoints(kpts_3d, window_length=15, polyorder=3)
            
    output_mp4 = os.path.join(DATA_DIR, "reconstructed_3d.mp4")
    render_3d_video(kpts_3d, output_mp4)

if __name__ == "__main__":
    main()
