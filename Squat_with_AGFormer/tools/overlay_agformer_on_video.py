r"""
Overlay AGFormer 2D and 3D Reprojected Skeleton onto Video.

Usage:
  mamba run -n hw1 python overlay_agformer_on_video.py \
    --video D:\Pitt\Project\Squat_Project\video\squat_3D\recording_20260207_163410\RR.avi \
    --agformer_dir D:\Pitt\Project\Squat_Project\video\squat_3D\recording_20260207_163410\AGFormer
"""
import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm


# Human3.6M 17-joint definitions:
# 0: Pelvis
# 1: R_Hip, 2: R_Knee, 3: R_Ankle
# 4: L_Hip, 5: L_Knee, 6: L_Ankle
# 7: Spine, 8: Thorax, 9: Nose, 10: HeadTop
# 11: L_Shoulder, 12: L_Elbow, 13: L_Wrist
# 14: R_Shoulder, 15: R_Elbow, 16: R_Wrist

SKELETON_CONNECTIONS = [
    # (parent, child)
    (0, 1),   # Pelvis -> R_Hip
    (0, 4),   # Pelvis -> L_Hip
    (1, 2),   # R_Hip -> R_Knee
    (4, 5),   # L_Hip -> L_Knee
    (2, 3),   # R_Knee -> R_Ankle
    (5, 6),   # L_Knee -> L_Ankle
    (0, 7),   # Pelvis -> Spine
    (7, 8),   # Spine -> Thorax
    (8, 14),  # Thorax -> R_Shoulder
    (8, 11),  # Thorax -> L_Shoulder
    (14, 15), # R_Shoulder -> R_Elbow
    (15, 16), # R_Elbow -> R_Wrist
    (11, 12), # L_Shoulder -> L_Elbow
    (12, 13), # L_Elbow -> L_Wrist
    (8, 9),   # Thorax -> Nose
    (9, 10),  # Nose -> HeadTop
]

# True for Left side (Cyan/Blue), False for Right side (Red), None for Center (Yellow)
BONE_SIDE = [
    False, # 0->1: R_Hip
    True,  # 0->4: L_Hip
    False, # 1->2: R_Knee
    True,  # 4->5: L_Knee
    False, # 2->3: R_Ankle
    True,  # 5->6: L_Ankle
    None,  # 0->7: Spine
    None,  # 7->8: Thorax
    False, # 8->14: R_Shoulder
    True,  # 8->11: L_Shoulder
    False, # 14->15: R_Elbow
    False, # 15->16: R_Wrist
    True,  # 11->12: L_Elbow
    True,  # 12->13: L_Wrist
    None,  # 8->9: Nose
    None,  # 9->10: HeadTop
]


def qrot_np(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q in numpy.
    q: (*, 4) where q = [w, x, y, z]
    v: (*, 3)
    """
    qvec = q[..., 1:]
    w = q[..., :1]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


def calculate_angle(a, b, c):
    """Calculate angle between vector ba and bc in degrees."""
    v1 = a - b
    v2 = c - b
    norm_v1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    norm_v2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    norm_v1[norm_v1 == 0] = 1e-6
    norm_v2[norm_v2 == 0] = 1e-6
    with np.errstate(divide='ignore', invalid='ignore'):
        dot_product = np.sum((v1 / norm_v1) * (v2 / norm_v2), axis=-1)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)
    return float(np.degrees(angle))


def draw_hud(frame, title, frame_idx, total_frames, r_knee, l_knee, r_hip, l_hip, hud_pos=(15, 15)):
    """Draw a modern translucent HUD overlay on the frame."""
    x0, y0 = hud_pos
    hud_w, hud_h = 420, 155
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + hud_w, y0 + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Title & Frame
    cv2.putText(frame, f"{title} | Frame: {frame_idx+1}/{total_frames}", (x0 + 12, y0 + 30),
                font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Angles
    rk_txt = f"R Knee: {r_knee:.1f} deg" if r_knee > 0 else "R Knee: N/A"
    lk_txt = f"L Knee: {l_knee:.1f} deg" if l_knee > 0 else "L Knee: N/A"
    rh_txt = f"R Hip : {r_hip:.1f} deg" if r_hip > 0 else "R Hip : N/A"
    lh_txt = f"L Hip : {l_hip:.1f} deg" if l_hip > 0 else "L Hip : N/A"

    # Color code: Right = Orange/Red (0, 140, 255), Left = Cyan (255, 200, 0)
    cv2.putText(frame, rk_txt, (x0 + 12, y0 + 60), font, 0.58, (0, 100, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, lk_txt, (x0 + 220, y0 + 60), font, 0.58, (255, 200, 50), 2, cv2.LINE_AA)
    cv2.putText(frame, rh_txt, (x0 + 12, y0 + 90), font, 0.58, (0, 180, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, lh_txt, (x0 + 220, y0 + 90), font, 0.58, (255, 240, 100), 2, cv2.LINE_AA)

    # Legend
    cv2.circle(frame, (x0 + 20, y0 + 128), 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, "Right limb", (x0 + 32, y0 + 133), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.circle(frame, (x0 + 140, y0 + 128), 6, (255, 180, 0), -1, cv2.LINE_AA)
    cv2.putText(frame, "Left limb", (x0 + 152, y0 + 133), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.circle(frame, (x0 + 250, y0 + 128), 6, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, "Torso/Spine", (x0 + 262, y0 + 133), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


def draw_skeleton(frame, pts_2d, bone_thickness=3, joint_radius=5):
    """
    Draw 17-joint Human3.6M skeleton onto frame.
    pts_2d: (17, 2) or (17, 3)
    """
    # Draw bones
    for (j1, j2), side in zip(SKELETON_CONNECTIONS, BONE_SIDE):
        p1 = pts_2d[j1][:2]
        p2 = pts_2d[j2][:2]
        if (p1 > 0).all() and (p2 > 0).all():
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            pt2 = (int(round(p2[0])), int(round(p2[1])))
            if side is True:
                color = (255, 180, 0)   # Left = Cyan/Light Blue
            elif side is False:
                color = (0, 0, 255)     # Right = Red
            else:
                color = (0, 240, 255)   # Center/Torso = Yellow
            cv2.line(frame, pt1, pt2, color, bone_thickness, cv2.LINE_AA)

    # Draw joint points
    for j in range(17):
        p = pts_2d[j][:2]
        if (p > 0).all():
            pt = (int(round(p[0])), int(round(p[1])))
            # Black border circle
            cv2.circle(frame, pt, joint_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            # Inner color circle (Green)
            cv2.circle(frame, pt, joint_radius, (0, 255, 0), -1, cv2.LINE_AA)


def overlay_agformer(video_path, agformer_dir, out_dir=None):
    if out_dir is None:
        out_dir = agformer_dir
    os.makedirs(out_dir, exist_ok=True)

    npz_2d_path = os.path.join(agformer_dir, "keypoints.npz")
    npz_3d_path = os.path.join(agformer_dir, "keypoints_3d.npz")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(npz_2d_path):
        raise FileNotFoundError(f"2D keypoints not found: {npz_2d_path}")
    if not os.path.exists(npz_3d_path):
        raise FileNotFoundError(f"3D keypoints not found: {npz_3d_path}")

    print(f"===============================================================")
    print(f"Overlay AGFormer Skeleton onto Video")
    print(f"  Video       : {video_path}")
    print(f"  AGFormer Dir: {agformer_dir}")
    print(f"  Output Dir  : {out_dir}")
    print(f"===============================================================")

    # Load 2D and 3D data
    data_2d = np.load(npz_2d_path, allow_pickle=True)['reconstruction']
    if len(data_2d.shape) == 4 and data_2d.shape[0] == 1:
        data_2d = data_2d[0]  # Shape: (N, 17, 3)

    data_3d = np.load(npz_3d_path, allow_pickle=True)['reconstruction']
    if len(data_3d.shape) == 4 and data_3d.shape[0] == 1:
        data_3d = data_3d[0]  # Shape: (N, 17, 3)

    # Invert world to camera coordinates for 3D reprojection
    # AGFormer ground alignment rotation quaternion
    rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], dtype='float32')
    rot_inv = np.array([rot[0], -rot[1], -rot[2], -rot[3]], dtype='float32')

    cam_3d = qrot_np(np.tile(rot_inv, (*data_3d.shape[:-1], 1)), data_3d)
    cam_3d = cam_3d - cam_3d[:, 0:1, :]  # Pelvis-relative 3D

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 29.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info: {w}x{h} | {fps:.2f} FPS | {total_frames} Frames")

    n_frames = min(total_frames, len(data_2d), len(data_3d))

    # Calculate 3D projected 2D coordinates
    # Pelvis 2D anchor + normalized 3D displacement * (w / 2)
    pelvis_2d = data_2d[:n_frames, 0:1, :2]
    proj_3d_2d = np.zeros((n_frames, 17, 2), dtype=np.float32)
    proj_3d_2d[..., 0] = pelvis_2d[..., 0] + cam_3d[:n_frames, :, 0] * (w / 2.0)
    proj_3d_2d[..., 1] = pelvis_2d[..., 1] + cam_3d[:n_frames, :, 1] * (w / 2.0)

    # Setup video writers
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    out_3d_path = os.path.join(out_dir, f"{video_stem}_3d_overlay.mp4")
    out_2d_path = os.path.join(out_dir, f"{video_stem}_2d_overlay.mp4")
    out_compare_path = os.path.join(out_dir, f"{video_stem}_compare_2d_3d.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_3d = cv2.VideoWriter(out_3d_path, fourcc, fps, (w, h))
    writer_2d = cv2.VideoWriter(out_2d_path, fourcc, fps, (w, h))
    writer_compare = cv2.VideoWriter(out_compare_path, fourcc, fps, (w * 2, h))

    preview_saved = False
    pbar = tqdm(total=n_frames, desc="Rendering Overlays")

    for f_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_3d = frame.copy()
        frame_2d = frame.copy()

        # -------------------------------------------------------------
        # 1. AGFormer 3D Reprojected Overlay
        # -------------------------------------------------------------
        draw_skeleton(frame_3d, proj_3d_2d[f_idx], bone_thickness=3, joint_radius=5)
        r_knee_3d = calculate_angle(data_3d[f_idx, 1], data_3d[f_idx, 2], data_3d[f_idx, 3])
        l_knee_3d = calculate_angle(data_3d[f_idx, 4], data_3d[f_idx, 5], data_3d[f_idx, 6])
        r_hip_3d  = calculate_angle(data_3d[f_idx, 14], data_3d[f_idx, 1], data_3d[f_idx, 2])
        l_hip_3d  = calculate_angle(data_3d[f_idx, 11], data_3d[f_idx, 4], data_3d[f_idx, 5])
        draw_hud(frame_3d, "AGFormer 3D Pose", f_idx, total_frames, r_knee_3d, l_knee_3d, r_hip_3d, l_hip_3d)

        # -------------------------------------------------------------
        # 2. 2D Keypoints Overlay
        # -------------------------------------------------------------
        draw_skeleton(frame_2d, data_2d[f_idx, :, :2], bone_thickness=3, joint_radius=5)
        r_knee_2d = calculate_angle(data_2d[f_idx, 1, :2], data_2d[f_idx, 2, :2], data_2d[f_idx, 3, :2])
        l_knee_2d = calculate_angle(data_2d[f_idx, 4, :2], data_2d[f_idx, 5, :2], data_2d[f_idx, 6, :2])
        r_hip_2d  = calculate_angle(data_2d[f_idx, 14, :2], data_2d[f_idx, 1, :2], data_2d[f_idx, 2, :2])
        l_hip_2d  = calculate_angle(data_2d[f_idx, 11, :2], data_2d[f_idx, 4, :2], data_2d[f_idx, 5, :2])
        draw_hud(frame_2d, "Input 2D Pose (H36M)", f_idx, total_frames, r_knee_2d, l_knee_2d, r_hip_2d, l_hip_2d)

        # -------------------------------------------------------------
        # 3. Side-by-Side Comparison
        # -------------------------------------------------------------
        compare_frame = np.hstack([frame_2d, frame_3d])

        # Write to video files
        writer_3d.write(frame_3d)
        writer_2d.write(frame_2d)
        writer_compare.write(compare_frame)

        # Save preview screenshot around rep 1 (e.g. frame 200)
        if f_idx == 200 or (not preview_saved and f_idx == n_frames // 2):
            preview_path = os.path.join(out_dir, f"{video_stem}_overlay_preview.jpg")
            cv2.imwrite(preview_path, compare_frame)
            preview_saved = True

        pbar.update(1)

    pbar.close()
    cap.release()
    writer_3d.release()
    writer_2d.release()
    writer_compare.release()

    print("\n✅ All overlay videos generated successfully!")
    print(f"  [3D Reprojected Video] {out_3d_path} ({os.path.getsize(out_3d_path)/1e6:.1f} MB)")
    print(f"  [2D Input Video]       {out_2d_path} ({os.path.getsize(out_2d_path)/1e6:.1f} MB)")
    print(f"  [Side-by-Side Video]   {out_compare_path} ({os.path.getsize(out_compare_path)/1e6:.1f} MB)")
    if preview_saved:
        print(f"  [Preview Screenshot]   {preview_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay AGFormer 2D and 3D skeleton onto video")
    parser.add_argument("--video", type=str,
                        default=r"D:\Pitt\Project\Squat_Project\video\squat_3D\recording_20260207_163410\RR.avi",
                        help="Path to input video file")
    parser.add_argument("--agformer_dir", type=str,
                        default=r"D:\Pitt\Project\Squat_Project\video\squat_3D\recording_20260207_163410\AGFormer",
                        help="Path to AGFormer directory containing keypoints.npz and keypoints_3d.npz")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory to save output videos (default: agformer_dir)")

    args = parser.parse_args()
    overlay_agformer(args.video, args.agformer_dir, args.out_dir)
