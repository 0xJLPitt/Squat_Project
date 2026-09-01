"""
Batch Video Processing Pipeline: YOLO 2D Pose -> H36M Keypoints -> MotionAGFormer 3D Pose
支援：
1. YOLO 2D 骨架辨識與 COCO 格式儲存 (yolo_skeleton.txt)
2. 時序內插補齊與 Human3.6M 格式轉換 (keypoints.npz)
3. MotionAGFormer 3D 姿態提升 (keypoints_3d.npz)
4. 2D 骨架誤差與 3D 抖動診斷分析 (compare_2d_3d_angles.png, 2d_jitter_analysis.png, 2d_quality_report.txt)
5. 2D 骨架影片疊圖輸出 (<video_stem>_2d_overlay.mp4)
6. 3D 骨架重新投射回原影片疊圖輸出 (<video_stem>_3d_overlay.mp4)
7. 3D 空間立體骨架動作動畫影片輸出 (<video_stem>_visualize_3d.mp4)
"""
import os
import sys
import glob
import copy
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add MotionAGFormer to sys.path
MOTION_AGFORMER_PATH = r"D:\Pitt\Project\tools\MotionAGFormer"
if MOTION_AGFORMER_PATH not in sys.path:
    sys.path.append(MOTION_AGFORMER_PATH)

try:
    from demo.lib.utils import normalize_screen_coordinates, camera_to_world
    from model.MotionAGFormer import MotionAGFormer
except ImportError as e:
    print(f"[Warning] Failed to import MotionAGFormer modules directly: {e}")

try:
    from overlay_2d import overlay_2d
except ImportError:
    overlay_2d = None


# ==============================================================================
# 1. YOLO 2D Pose Estimation
# ==============================================================================

def run_yolo_pose(video_path, output_txt, model, conf_thresh=0.25):
    """
    Run YOLO Pose on the video and output keypoints to txt file.
    Format: frame_idx, joint_idx, x, y, conf
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  [YOLO] Video: {os.path.basename(video_path)} | Size: {w}x{h} | FPS: {fps:.2f} | Frames: {total_frames}")

    frame_idx = 1
    pbar = tqdm(total=total_frames, desc="  [YOLO Pose]", leave=False)

    with open(output_txt, 'w', encoding='utf-8') as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False, conf=conf_thresh)

            best_person_idx = -1
            max_box_area = -1.0

            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for p_i, box in enumerate(boxes):
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area > max_box_area:
                        max_box_area = area
                        best_person_idx = p_i

            if best_person_idx >= 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > best_person_idx:
                kpts = results[0].keypoints.xy[best_person_idx].cpu().numpy()
                confs = results[0].keypoints.conf[best_person_idx].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(len(kpts))
                for j_idx, ((x, y), conf) in enumerate(zip(kpts, confs)):
                    f.write(f"{frame_idx},{j_idx},{int(x)},{int(y)},{float(conf):.3f}\n")
            else:
                for j_idx in range(17):
                    f.write(f"{frame_idx},{j_idx},0,0,0.000\n")

            frame_idx += 1
            pbar.update(1)

    pbar.close()
    cap.release()
    print(f"  [YOLO] Finished! Saved keypoints to {output_txt}")


# ==============================================================================
# 2. Keypoints Conversion & Temporal Interpolation (COCO 17 -> H36M 17)
# ==============================================================================

def interpolate_missing_keypoints(kpts):
    """Linear interpolation for missing keypoints (x=0, y=0) across frames."""
    N_frames, num_joints, _ = kpts.shape
    for j in range(num_joints):
        missing = kpts[:, j, 2] == 0.0
        valid = ~missing
        if missing.all() or valid.all():
            continue
        valid_indices = np.where(valid)[0]
        missing_indices = np.where(missing)[0]
        kpts[missing_indices, j, 0] = np.interp(missing_indices, valid_indices, kpts[valid_indices, j, 0])
        kpts[missing_indices, j, 1] = np.interp(missing_indices, valid_indices, kpts[valid_indices, j, 1])
        kpts[missing_indices, j, 2] = 0.5
    return kpts


def coco_to_h36m(keypoints):
    """Convert COCO 17 keypoint format to Human3.6M 17 format."""
    new_keypoints = np.zeros_like(keypoints)
    
    # 0: Pelvis = (L_Hip + R_Hip) * 0.5 (YOLO 11 & 12)
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    # 1: Right Hip -> YOLO 12
    new_keypoints[..., 1, :] = keypoints[..., 12, :]
    # 2: Right Knee -> YOLO 14
    new_keypoints[..., 2, :] = keypoints[..., 14, :]
    # 3: Right Ankle -> YOLO 16
    new_keypoints[..., 3, :] = keypoints[..., 16, :]
    
    # 4: Left Hip -> YOLO 11
    new_keypoints[..., 4, :] = keypoints[..., 11, :]
    # 5: Left Knee -> YOLO 13
    new_keypoints[..., 5, :] = keypoints[..., 13, :]
    # 6: Left Ankle -> YOLO 15
    new_keypoints[..., 6, :] = keypoints[..., 15, :]
    
    # 8: Thorax/Neck = (L_Shoulder + R_Shoulder) * 0.5 (YOLO 5 & 6)
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    # 7: Spine = (Pelvis + Thorax) * 0.5
    new_keypoints[..., 7, :] = (new_keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    # 9: Nose/Head -> YOLO 0
    new_keypoints[..., 9, :] = keypoints[..., 0, :]
    
    # Head top interpolation from eyes
    left_eye_missing = (keypoints[..., 1, 0] == 0) & (keypoints[..., 1, 1] == 0)
    right_eye_missing = (keypoints[..., 2, 0] == 0) & (keypoints[..., 2, 1] == 0)
    
    new_keypoints[..., 10, :] = (keypoints[..., 1, :] + keypoints[..., 2, :]) * 0.5
    only_left = ~left_eye_missing & right_eye_missing
    new_keypoints[only_left, 10, :] = keypoints[only_left, 1, :]
    only_right = left_eye_missing & ~right_eye_missing
    new_keypoints[only_right, 10, :] = keypoints[only_right, 2, :]
    both_missing = left_eye_missing & right_eye_missing
    new_keypoints[both_missing, 10, :] = 0
    
    missing_nose = (new_keypoints[..., 9, 0] == 0) & (new_keypoints[..., 9, 1] == 0)
    new_keypoints[missing_nose, 9, :2] = new_keypoints[missing_nose, 8, :2] + (new_keypoints[missing_nose, 8, :2] - new_keypoints[missing_nose, 7, :2]) * 0.8
    
    missing_head = (new_keypoints[..., 10, 0] == 0) & (new_keypoints[..., 10, 1] == 0)
    new_keypoints[missing_head, 10, :2] = new_keypoints[missing_head, 8, :2] + (new_keypoints[missing_head, 8, :2] - new_keypoints[missing_head, 7, :2]) * 1.2
    
    # Left Arm: 11~13 -> YOLO 5, 7, 9
    new_keypoints[..., 11, :] = keypoints[..., 5, :]
    new_keypoints[..., 12, :] = keypoints[..., 7, :]
    new_keypoints[..., 13, :] = keypoints[..., 9, :]
    
    # Right Arm: 14~16 -> YOLO 6, 8, 10
    new_keypoints[..., 14, :] = keypoints[..., 6, :]
    new_keypoints[..., 15, :] = keypoints[..., 8, :]
    new_keypoints[..., 16, :] = keypoints[..., 10, :]

    return new_keypoints


def convert_txt_to_npz(txt_path, output_npz_path):
    """Load YOLO keypoints txt, interpolate, convert to H36M format, and save npz."""
    raw_lines = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                raw_lines.append([float(p) for p in parts[:5]])
    
    data = np.array(raw_lines)
    if len(data) == 0:
        raise ValueError(f"File {txt_path} is empty.")

    frames = np.unique(data[:, 0]).astype(int)
    num_frames = int(np.max(frames))
    
    coco_kpts = np.zeros((num_frames, 17, 3))
    for row in data:
        f_idx = int(row[0]) - 1
        j_idx = int(row[1])
        x = float(row[2])
        y = float(row[3])
        conf = float(row[4]) if len(row) > 4 else 1.0
        coco_kpts[f_idx, j_idx, 0] = x
        coco_kpts[f_idx, j_idx, 1] = y
        coco_kpts[f_idx, j_idx, 2] = 0.0 if (x == 0 and y == 0) else conf

    coco_kpts = interpolate_missing_keypoints(coco_kpts)
    h36m_kpts = coco_to_h36m(coco_kpts)
    final_input = np.expand_dims(h36m_kpts, axis=0) # (1, N_frames, 17, 3)

    np.savez_compressed(output_npz_path, reconstruction=final_input)
    print(f"  [Convert] Converted H36M 2D keypoints saved to {output_npz_path} (Shape: {final_input.shape})")


# ==============================================================================
# 3. MotionAGFormer 3D Pose Inference
# ==============================================================================

def resample(n_frames):
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1]
    downsample = np.arange(243)
    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
                downsample = np.arange(243)
    return clips, downsample


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]
    return flipped_data


def load_agformer_model(checkpoint_path):
    class Args: pass
    args = Args()
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 16, 3, 128, 512, 3
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
    args.use_tcn, args.graph_only = False, False
    args.n_frames = 243

    model = nn.DataParallel(MotionAGFormer(**vars(args))).cuda()
    pre_dict = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(pre_dict['model'], strict=True)
    model.eval()
    return model


def infer_agformer_3d(npz_2d_file, output_3d_file, model, w, h):
    data = np.load(npz_2d_file, allow_pickle=True)
    keypoints = data['reconstruction']
    
    clips, downsample = turn_into_clips(keypoints)
    output_3D_all = []

    for idx, clip in enumerate(clips):
        input_2D = normalize_screen_coordinates(clip, w=w, h=h)
        input_2D_aug = flip_data(input_2D)
        
        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()

        with torch.no_grad():
            output_3D_non_flip = model(input_2D) 
            output_3D_flip = flip_data(model(input_2D_aug))
            output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()
        
        for j in range(len(post_out_all)):
            rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out_all[j] = camera_to_world(post_out_all[j], R=rot, t=0)
            post_out_all[j, :, 2] -= np.min(post_out_all[j, :, 2])
            
        output_3D_all.append(post_out_all)

    final_3D = np.concatenate(output_3D_all, axis=0)
    np.savez_compressed(output_3d_file, reconstruction=final_3D)
    print(f"  [AGFormer 3D] Done! 3D keypoints saved to {output_3d_file} (Shape: {final_3D.shape})")


# ==============================================================================
# 4. Angle Calculation Helper
# ==============================================================================

def calculate_angle(a, b, c):
    """Calculate angle between vector ba and bc (in degrees). Works for 2D or 3D."""
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
    return np.degrees(angle)


# ==============================================================================
# 5. 2D vs 3D Diagnostic Plot & Quality Report
# ==============================================================================

def diagnose_and_plot(txt_path, npz_2d_path, npz_3d_path, out_angle_plot, out_jitter_plot, out_report):
    """
    Generate comprehensive diagnostics for 2D errors vs 3D jitter:
    1. Angle Comparison Plot (Raw 2D YOLO vs Interpolated 2D H36M vs 3D AGFormer)
    2. 2D Joint Jitter / Velocity Plot
    3. Text Quality & Error Report
    """
    raw_lines = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                raw_lines.append([float(p) for p in parts[:5]])
    data = np.array(raw_lines)
    frames_ids = np.unique(data[:, 0]).astype(int)
    num_frames = int(np.max(frames_ids))

    k_yolo = np.full((num_frames, 17, 2), np.nan)
    conf_yolo = np.zeros((num_frames, 17))
    for row in data:
        f_idx = int(row[0]) - 1
        j_idx = int(row[1])
        x, y = float(row[2]), float(row[3])
        conf = float(row[4]) if len(row) > 4 else 1.0
        if x != 0 or y != 0:
            k_yolo[f_idx, j_idx, 0] = x
            k_yolo[f_idx, j_idx, 1] = y
            conf_yolo[f_idx, j_idx] = conf

    d_2d = np.load(npz_2d_path)['reconstruction'][0, ..., :2]
    d_3d = np.load(npz_3d_path)['reconstruction']
    
    n_frames = min(len(k_yolo), len(d_2d), len(d_3d))
    k_yolo = k_yolo[:n_frames]
    conf_yolo = conf_yolo[:n_frames]
    d_2d = d_2d[:n_frames]
    d_3d = d_3d[:n_frames]
    frames = np.arange(n_frames)

    # Calculate Angles
    r_knee_yolo = calculate_angle(k_yolo[:, 12], k_yolo[:, 14], k_yolo[:, 16])
    l_knee_yolo = calculate_angle(k_yolo[:, 11], k_yolo[:, 13], k_yolo[:, 15])
    r_hip_yolo = calculate_angle(k_yolo[:, 6], k_yolo[:, 12], k_yolo[:, 14])
    l_hip_yolo = calculate_angle(k_yolo[:, 5], k_yolo[:, 11], k_yolo[:, 13])

    r_knee_2d = calculate_angle(d_2d[:, 1], d_2d[:, 2], d_2d[:, 3])
    l_knee_2d = calculate_angle(d_2d[:, 4], d_2d[:, 5], d_2d[:, 6])
    r_hip_2d = calculate_angle(d_2d[:, 14], d_2d[:, 1], d_2d[:, 2])
    l_hip_2d = calculate_angle(d_2d[:, 11], d_2d[:, 4], d_2d[:, 5])

    r_knee_3d = calculate_angle(d_3d[:, 1], d_3d[:, 2], d_3d[:, 3])
    l_knee_3d = calculate_angle(d_3d[:, 4], d_3d[:, 5], d_3d[:, 6])
    r_hip_3d = calculate_angle(d_3d[:, 14], d_3d[:, 1], d_3d[:, 2])
    l_hip_3d = calculate_angle(d_3d[:, 11], d_3d[:, 4], d_3d[:, 5])

    # Plot 1: Angle Comparison
    plt.figure(figsize=(16, 12))
    def plot_angle_sub(idx, yolo, h2d, h3d, title):
        plt.subplot(2, 2, idx)
        plt.plot(frames, yolo, label='1. Raw YOLO 2D (COCO)', color='gray', alpha=0.6, linestyle=':', marker='.', markersize=2)
        plt.plot(frames, h2d, label='2. Interpolated H36M 2D', color='blue', alpha=0.5, linestyle='--')
        plt.plot(frames, h3d, label='3. AGFormer Output 3D', color='red', linewidth=2)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Angle (degrees)', fontsize=12)
        plt.xlabel('Frame', fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)

    plot_angle_sub(1, r_knee_yolo, r_knee_2d, r_knee_3d, 'Right Knee Angle Comparison')
    plot_angle_sub(2, l_knee_yolo, l_knee_2d, l_knee_3d, 'Left Knee Angle Comparison')
    plot_angle_sub(3, r_hip_yolo, r_hip_2d, r_hip_3d, 'Right Hip Angle Comparison')
    plot_angle_sub(4, l_hip_yolo, l_hip_2d, l_hip_3d, 'Left Hip Angle Comparison')
    plt.tight_layout()
    plt.savefig(out_angle_plot, dpi=150)
    plt.close()

    # Plot 2: 2D Jitter / Velocity
    vel_2d = np.linalg.norm(np.diff(d_2d, axis=0), axis=-1)
    frames_vel = frames[1:]

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 1, 1)
    plt.plot(frames_vel, vel_2d[:, 1], label='Right Hip Velocity', color='green', alpha=0.7)
    plt.plot(frames_vel, vel_2d[:, 2], label='Right Knee Velocity', color='blue', alpha=0.8)
    plt.plot(frames_vel, vel_2d[:, 3], label='Right Ankle Velocity', color='red', alpha=0.8)
    plt.axhline(y=20, color='orange', linestyle='--', label='Jitter Threshold (>20 px/frame)')
    plt.title('2D Keypoint Velocity & Sudden Jumps (Right Leg)', fontsize=14, fontweight='bold')
    plt.ylabel('Displacement (pixels/frame)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(2, 1, 2)
    plt.plot(frames_vel, vel_2d[:, 4], label='Left Hip Velocity', color='green', alpha=0.7)
    plt.plot(frames_vel, vel_2d[:, 5], label='Left Knee Velocity', color='blue', alpha=0.8)
    plt.plot(frames_vel, vel_2d[:, 6], label='Left Ankle Velocity', color='red', alpha=0.8)
    plt.axhline(y=20, color='orange', linestyle='--', label='Jitter Threshold (>20 px/frame)')
    plt.title('2D Keypoint Velocity & Sudden Jumps (Left Leg)', fontsize=14, fontweight='bold')
    plt.ylabel('Displacement (pixels/frame)', fontsize=12)
    plt.xlabel('Frame', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_jitter_plot, dpi=150)
    plt.close()

    # Report
    r_k_jump_2d = np.abs(np.diff(r_knee_2d))
    l_k_jump_2d = np.abs(np.diff(l_knee_2d))
    r_k_jump_3d = np.abs(np.diff(r_knee_3d))
    l_k_jump_3d = np.abs(np.diff(l_knee_3d))
    severe_jump_frames_2d = np.where((r_k_jump_2d > 15) | (l_k_jump_2d > 15))[0] + 1

    with open(out_report, 'w', encoding='utf-8') as f:
        f.write(f"===============================================================\n")
        f.write(f"2D & 3D Pose Stability Diagnostic Report\n")
        f.write(f"===============================================================\n")
        f.write(f"Total Frames Analyzed: {n_frames}\n\n")
        
        f.write(f"[1] 2D Keypoint Detection Stability:\n")
        joint_names = ['Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 
                       'Spine', 'Neck', 'Head', 'HeadTop', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
                       'R_Shoulder', 'R_Elbow', 'R_Wrist']
        for j_i, j_name in enumerate(joint_names):
            mean_v = np.mean(vel_2d[:, j_i])
            max_v = np.max(vel_2d[:, j_i])
            jumps = np.sum(vel_2d[:, j_i] > 20)
            f.write(f"  - {j_name:<12}: Mean speed = {mean_v:5.2f} px/f | Max jump = {max_v:6.2f} px | Sudden jumps (>20px) = {jumps} frames\n")

        f.write(f"\n[2] Angle Jitter Comparison:\n")
        f.write(f"  - Right Knee Mean Jitter: 2D = {np.mean(r_k_jump_2d):.2f} deg/f | 3D = {np.mean(r_k_jump_3d):.2f} deg/f\n")
        f.write(f"  - Right Knee Max Jump   : 2D = {np.max(r_k_jump_2d):.2f} deg   | 3D = {np.max(r_k_jump_3d):.2f} deg\n")
        f.write(f"  - Left Knee Mean Jitter : 2D = {np.mean(l_k_jump_2d):.2f} deg/f | 3D = {np.mean(l_k_jump_3d):.2f} deg/f\n")
        f.write(f"  - Left Knee Max Jump    : 2D = {np.max(l_k_jump_2d):.2f} deg   | 3D = {np.max(l_k_jump_3d):.2f} deg\n")

        f.write(f"\n[3] 2D Error Diagnosis Summary:\n")
        if len(severe_jump_frames_2d) > 0:
            f.write(f"  [!] Detected {len(severe_jump_frames_2d)} frames with severe 2D angle jumps (>15 deg/frame).\n")
            f.write(f"  Sample Jump Frames: {list(severe_jump_frames_2d[:20])} ...\n")
            f.write(f"  Conclusion: The 3D jitter is primarily caused by rapid jumping/flickering in raw 2D YOLO keypoint detections.\n")
        else:
            f.write(f"  2D keypoint detections are relatively smooth.\n")
        f.write(f"===============================================================\n")

    print(f"  [Diagnostic] Angle comparison plot: {out_angle_plot}")
    print(f"  [Diagnostic] Jitter plot: {out_jitter_plot}")
    print(f"  [Diagnostic] Quality report: {out_report}")


# ==============================================================================
# 6. 2D Overlay Video Generation (<stem>_2d_overlay.mp4)
# ==============================================================================

def create_2d_overlay_video(video_path, txt_path, out_video_path):
    """
    Project 2D skeleton onto raw video with angles HUD and joint indicators.
    Saved as <video_stem>_2d_overlay.mp4.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    keypoints_dict = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                frame_idx = int(parts[0]) - 1
                joint_idx = int(parts[1])
                x = int(float(parts[2]))
                y = int(float(parts[3]))
                conf = float(parts[4]) if len(parts) > 4 else 1.0
                if frame_idx not in keypoints_dict:
                    keypoints_dict[frame_idx] = {}
                keypoints_dict[frame_idx][joint_idx] = (x, y, conf)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    skeleton = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4), (3, 5), (4, 6)
    ]

    frame_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    pbar = tqdm(total=total_frames, desc="  [2D Overlay Video]", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        r_knee_ang, l_knee_ang, r_hip_ang, l_hip_ang = 0.0, 0.0, 0.0, 0.0

        if frame_idx in keypoints_dict:
            kpts = keypoints_dict[frame_idx]
            
            # Draw bones
            for pt1, pt2 in skeleton:
                if pt1 in kpts and pt2 in kpts:
                    x1, y1, c1 = kpts[pt1]
                    x2, y2, c2 = kpts[pt2]
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        line_color = (0, 0, 255) if (pt1 % 2 != 0 and pt2 % 2 != 0) else ((255, 120, 0) if (pt1 % 2 == 0 and pt2 % 2 == 0 and pt1 != 0 and pt2 != 0) else (0, 220, 220))
                        cv2.line(frame, (x1, y1), (x2, y2), line_color, 3, cv2.LINE_AA)

            # Draw joints
            for j in range(17):
                if j in kpts:
                    x, y, conf = kpts[j]
                    if x > 0 and y > 0:
                        dot_color = (0, 255, 0) if conf > 0.4 else (0, 140, 255)
                        cv2.circle(frame, (x, y), 6, (0, 0, 0), -1, cv2.LINE_AA)
                        cv2.circle(frame, (x, y), 4, dot_color, -1, cv2.LINE_AA)

            # Angles
            if 12 in kpts and 14 in kpts and 16 in kpts:
                p_h, p_k, p_a = np.array(kpts[12][:2]), np.array(kpts[14][:2]), np.array(kpts[16][:2])
                if (p_h > 0).all() and (p_k > 0).all() and (p_a > 0).all():
                    r_knee_ang = float(calculate_angle(p_h, p_k, p_a))

            if 11 in kpts and 13 in kpts and 15 in kpts:
                p_h, p_k, p_a = np.array(kpts[11][:2]), np.array(kpts[13][:2]), np.array(kpts[15][:2])
                if (p_h > 0).all() and (p_k > 0).all() and (p_a > 0).all():
                    l_knee_ang = float(calculate_angle(p_h, p_k, p_a))

            if 6 in kpts and 12 in kpts and 14 in kpts:
                p_s, p_h, p_k = np.array(kpts[6][:2]), np.array(kpts[12][:2]), np.array(kpts[14][:2])
                if (p_s > 0).all() and (p_h > 0).all() and (p_k > 0).all():
                    r_hip_ang = float(calculate_angle(p_s, p_h, p_k))

            if 5 in kpts and 11 in kpts and 13 in kpts:
                p_s, p_h, p_k = np.array(kpts[5][:2]), np.array(kpts[11][:2]), np.array(kpts[13][:2])
                if (p_s > 0).all() and (p_h > 0).all() and (p_k > 0).all():
                    l_hip_ang = float(calculate_angle(p_s, p_h, p_k))

        # HUD Overlay
        hud_w, hud_h = 420, 160
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (20 + hud_w, 20 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, f"2D Pose | Frame: {frame_idx+1}/{total_frames}", (35, 50), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        r_txt = f"Right Knee: {r_knee_ang:.1f} deg" if r_knee_ang > 0 else "Right Knee: N/A"
        l_txt = f"Left Knee : {l_knee_ang:.1f} deg" if l_knee_ang > 0 else "Left Knee : N/A"
        r_h_txt = f"Right Hip : {r_hip_ang:.1f} deg" if r_hip_ang > 0 else "Right Hip : N/A"
        l_h_txt = f"Left Hip  : {l_hip_ang:.1f} deg" if l_hip_ang > 0 else "Left Hip  : N/A"

        cv2.putText(frame, r_txt, (35, 78), font, 0.65, (255, 120, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, l_txt, (35, 104), font, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, r_h_txt, (35, 130), font, 0.65, (255, 200, 100), 2, cv2.LINE_AA)
        cv2.putText(frame, l_h_txt, (35, 156), font, 0.65, (100, 100, 255), 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"  [2D Video] Saved 2D overlay to: {out_video_path}")


# ==============================================================================
# 7. 3D Reprojected Overlay Video Generation (<stem>_3d_overlay.mp4)
# ==============================================================================

def create_3d_reprojected_overlay_video(video_path, npz_2d_path, npz_3d_path, out_video_path):
    """
    Project the 3D reconstructed skeleton BACK onto the original video frames.
    Saved as <video_stem>_3d_overlay.mp4.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps): fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    k3d = np.load(npz_3d_path, allow_pickle=True)['reconstruction']
    k2d = np.load(npz_2d_path, allow_pickle=True)['reconstruction'][0]

    # Invert world to camera
    rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], dtype='float32')
    rot_inv = np.array([rot[0], -rot[1], -rot[2], -rot[3]], dtype='float32')
    cam_3d = camera_to_world(k3d, rot_inv, 0)
    cam_3d = cam_3d - cam_3d[:, 0:1, :] # Pelvis relative

    n_frames = min(len(k3d), len(k2d), total_frames)

    # Human3.6M skeleton connections:
    # 0:Pelvis, 1:RHip, 2:RKnee, 3:RAnkle, 4:LHip, 5:LKnee, 6:LAnkle,
    # 7:Spine, 8:Thorax, 9:Nose, 10:HeadTop, 11:LShoulder, 12:LElbow, 13:LWrist, 14:RShoulder, 15:RElbow, 16:RWrist
    I = [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9]
    J = [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10]
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    pbar = tqdm(total=n_frames, desc="  [3D Reprojected Overlay]", leave=False)

    for f_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Project 3D coordinates to 2D pixel space based on Pelvis anchor
        pelvis_2d = k2d[f_idx, 0, :2]
        proj_x = pelvis_2d[0] + cam_3d[f_idx, :, 0] * (w / 2)
        proj_y = pelvis_2d[1] + cam_3d[f_idx, :, 1] * (w / 2)

        # Draw 3D projected skeleton lines
        for i in range(len(I)):
            p1 = (int(proj_x[I[i]]), int(proj_y[I[i]]))
            p2 = (int(proj_x[J[i]]), int(proj_y[J[i]]))
            
            # Left = Blue/Cyan, Right = Red/Magenta, Torso = Yellow
            if LR[i]:
                line_color = (255, 180, 0) # Cyan/Blue for left
            elif not LR[i] and I[i] != 0 and I[i] != 7 and I[i] != 8:
                line_color = (0, 0, 255)   # Red for right
            else:
                line_color = (0, 255, 255) # Yellow for spine/head
                
            cv2.line(frame, p1, p2, line_color, 4, cv2.LINE_AA)

        # Draw 3D joints
        for j in range(17):
            pt = (int(proj_x[j]), int(proj_y[j]))
            cv2.circle(frame, pt, 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)

        # Calculate 3D angles
        r_knee_3d = float(calculate_angle(k3d[f_idx, 1], k3d[f_idx, 2], k3d[f_idx, 3]))
        l_knee_3d = float(calculate_angle(k3d[f_idx, 4], k3d[f_idx, 5], k3d[f_idx, 6]))
        r_hip_3d  = float(calculate_angle(k3d[f_idx, 14], k3d[f_idx, 1], k3d[f_idx, 2]))
        l_hip_3d  = float(calculate_angle(k3d[f_idx, 11], k3d[f_idx, 4], k3d[f_idx, 5]))

        # Draw HUD info overlay
        hud_w, hud_h = 430, 160
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (20 + hud_w, 20 + hud_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, f"3D Projected | Frame: {f_idx+1}/{total_frames}", (35, 50), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"3D R Knee: {r_knee_3d:.1f} deg", (35, 78), font, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"3D L Knee: {l_knee_3d:.1f} deg", (35, 104), font, 0.65, (255, 180, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"3D R Hip : {r_hip_3d:.1f} deg", (35, 130), font, 0.65, (100, 100, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"3D L Hip : {l_hip_3d:.1f} deg", (35, 156), font, 0.65, (255, 220, 100), 2, cv2.LINE_AA)

        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"  [3D Video] Saved 3D reprojected overlay to: {out_video_path}")


# ==============================================================================
# 8. 3D Spatial Animation Video Generation (<stem>_visualize_3d.mp4)
# ==============================================================================

def create_visualize_3d_video(npz_3d_path, out_video_path, fps=30.0):
    """
    Render 3D skeleton keypoints into a 3D animated video in 3D coordinate space.
    Saved as <video_stem>_visualize_3d.mp4.
    """
    data = np.load(npz_3d_path, allow_pickle=True)['reconstruction']
    if len(data.shape) == 4 and data.shape[0] == 1:
        data = data[0]

    max_val = np.max(data)
    if max_val > 0:
        data = data / max_val

    n_frames = len(data)
    
    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15., azim=70)

    lcolor = (0, 0, 1) # Blue
    rcolor = (1, 0, 0) # Red
    
    I = np.array([0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    lines = []
    for i in range(len(I)):
        line, = ax.plot([], [], [], lw=3.5, color=lcolor if LR[i] else rcolor)
        lines.append(line)

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (w_px, h_px))

    pbar = tqdm(total=n_frames, desc="  [3D Spatial Animation]", leave=False)

    for f in range(n_frames):
        vals = data[f]
        for i in range(len(I)):
            x = np.array([vals[I[i], 0], vals[J[i], 0]])
            y = np.array([vals[I[i], 1], vals[J[i], 1]])
            z = np.array([vals[I[i], 2], vals[J[i], 2]])
            lines[i].set_data(x, y)
            lines[i].set_3d_properties(z)

        xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
        ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
        ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
        ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
        ax.set_title(f"3D MotionAGFormer Space | Frame {f+1}/{n_frames}", fontsize=13, fontweight='bold')

        fig.canvas.draw()
        rgba_buf = np.asarray(fig.canvas.buffer_rgba())
        bgr_img = cv2.cvtColor(rgba_buf, cv2.COLOR_RGBA2BGR)
        out.write(bgr_img)
        pbar.update(1)

    pbar.close()
    out.release()
    plt.close(fig)
    print(f"  [3D Video] Saved 3D spatial animation to: {out_video_path}")


# ==============================================================================
# Main Pipeline Runner
# ==============================================================================

def process_all_videos(video_dir, yolo_model_path, agformer_ckpt_path, create_overlay=True):
    print("=" * 80)
    print(f"🎬 SQUAT 2D-to-3D Pipeline & 2D/3D Overlay Runner")
    print(f"Target Directory: {video_dir}")
    print(f"YOLO Model: {yolo_model_path}")
    print(f"MotionAGFormer Weights: {agformer_ckpt_path}")
    print("=" * 80)

    # 1. Find all video files
    video_extensions = ['*.MOV', '*.mov', '*.mp4', '*.MP4', '*.avi', '*.AVI']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, '**', ext), recursive=True))

    video_files = sorted(list(set(video_files)))
    # Exclude overlay/visualize outputs if already generated
    video_files = [v for v in video_files if not '_overlay' in os.path.basename(v).lower() and not '_visualize' in os.path.basename(v).lower() and not '_2d_skeleton' in os.path.basename(v).lower()]

    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        return

    print(f"📹 Found {len(video_files)} videos to process:")
    for i, v in enumerate(video_files, 1):
        print(f"  {i}. {v}")
    print("=" * 80)

    # 2. Load models
    print("\n📦 [1/2] Loading YOLO Pose Model...")
    yolo_model = YOLO(yolo_model_path)

    print("\n📦 [2/2] Loading MotionAGFormer 3D Model...")
    agformer_model = load_agformer_model(agformer_ckpt_path)
    print("✅ All models loaded successfully!\n")

    # 3. Process each video
    for idx, vid_path in enumerate(video_files, 1):
        vid_dir = os.path.dirname(vid_path)
        vid_stem = os.path.splitext(os.path.basename(vid_path))[0]
        
        # Create output directory for this video
        out_dir = os.path.join(vid_dir, vid_stem)
        os.makedirs(out_dir, exist_ok=True)

        txt_out = os.path.join(out_dir, "yolo_skeleton.txt")
        npz_2d_out = os.path.join(out_dir, "keypoints.npz")
        npz_3d_out = os.path.join(out_dir, "keypoints_3d.npz")
        
        # Output video naming rule
        overlay_2d_out = os.path.join(out_dir, f"{vid_stem}_2d_overlay.mp4")
        overlay_3d_out = os.path.join(out_dir, f"{vid_stem}_3d_overlay.mp4")
        visualize_3d_out = os.path.join(out_dir, f"{vid_stem}_visualize_3d.mp4")

        angle_plot_out = os.path.join(out_dir, "compare_2d_3d_angles.png")
        jitter_plot_out = os.path.join(out_dir, "2d_jitter_analysis.png")
        report_out = os.path.join(out_dir, "2d_quality_report.txt")

        print(f"\n=======================================================")
        print(f"[{idx}/{len(video_files)}] Processing: {os.path.basename(vid_path)}")
        print(f"Output Directory: {out_dir}")
        print(f"=======================================================")

        # Step 1: YOLO Pose
        print(f"\n▶ Step 1: YOLO 2D Pose Estimation...")
        run_yolo_pose(vid_path, txt_out, yolo_model)

        # Step 2: Convert to H36M
        print(f"\n▶ Step 2: Converting COCO 17 to Human3.6M 2D Format...")
        convert_txt_to_npz(txt_out, npz_2d_out)

        # Step 3: AGFormer 3D Inference
        print(f"\n▶ Step 3: MotionAGFormer 3D Pose Estimation...")
        cap = cv2.VideoCapture(vid_path)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps): fps = 30.0
        cap.release()
        infer_agformer_3d(npz_2d_out, npz_3d_out, agformer_model, w=w, h=h)

        # Step 4: 2D Quality & 3D Stability Diagnosis
        print(f"\n▶ Step 4: Generating 2D Error & 3D Jitter Diagnostic Charts...")
        diagnose_and_plot(txt_out, npz_2d_out, npz_3d_out, angle_plot_out, jitter_plot_out, report_out)

        # Step 5: 2D & 3D Video Generation
        if create_overlay:
            print(f"\n▶ Step 5-1: Generating 2D Overlay Video ({vid_stem}_2d_overlay.mp4)...")
            create_2d_overlay_video(vid_path, txt_out, overlay_2d_out)
            
            print(f"\n▶ Step 5-2: Generating 3D Reprojected Overlay Video ({vid_stem}_3d_overlay.mp4)...")
            create_3d_reprojected_overlay_video(vid_path, npz_2d_out, npz_3d_out, overlay_3d_out)

            print(f"\n▶ Step 5-3: Generating 3D Spatial Animation Video ({vid_stem}_visualize_3d.mp4)...")
            create_visualize_3d_video(npz_3d_out, visualize_3d_out, fps=fps)

        print(f"\n✨ Video {os.path.basename(vid_path)} completed successfully!")

    print("\n" + "=" * 80)
    print("🎉 ALL VIDEOS PROCESSED & VIDEOS GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process videos with YOLO Pose and MotionAGFormer with 2D/3D Diagnostics & Overlays")
    parser.add_argument("--video_dir", type=str, default=r"D:\Pitt\Project\Squat_Project\video", help="Directory containing input videos")
    parser.add_argument("--yolo_model", type=str, default=r"E:\squat\recordings_20260507_done\yolo11\yolo11x-pose.pt", help="Path to YOLO Pose weights")
    parser.add_argument("--agformer_ckpt", type=str, default=r"D:\Pitt\Project\tools\MotionAGFormer\checkpoint\motionagformer-b-h36m.pth.tr", help="Path to MotionAGFormer weights")
    parser.add_argument("--no_overlay", action="store_true", help="Skip generating 2D & 3D overlay videos")
    
    args = parser.parse_args()

    process_all_videos(
        video_dir=args.video_dir,
        yolo_model_path=args.yolo_model,
        agformer_ckpt_path=args.agformer_ckpt,
        create_overlay=not args.no_overlay
    )
