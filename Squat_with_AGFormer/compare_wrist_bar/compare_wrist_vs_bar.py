r"""
Compare AGFormer 3D Wrist Midpoint Trajectory vs YOLO Barbell Tracking (yolo_coordinates.txt)
支援：
1. 自動讀取或執行 AGFormer 3D 姿態提升 (輸出 keypoints.npz, keypoints_3d.npz)
2. 計算兩手腕中心點 (Wrist Midpoint) 之 3D 空間運動軌跡
3. 比對垂直 (Z軸/高度) 與水平 (矢狀面前後位移) 之波形相似度與皮爾森相關係數 (Pearson Correlation)
4. 依據 segments.json 進行逐次 (Per-Repetition) 深蹲軌跡分析與最低點 (Bottom) 幀數時間對齊
5. 輸出全域波形對照圖、逐次矢狀面 2D 路徑圖、波形對齊圖與 CSV 統計量化報告
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import copy

# Ensure package parent directory and tools are in sys.path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

TOOLS_DIR = os.path.join(PARENT_DIR, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

MOTION_AGFORMER_PATH = r"D:\Pitt\Project\tools\MotionAGFormer"
if MOTION_AGFORMER_PATH not in sys.path:
    sys.path.append(MOTION_AGFORMER_PATH)

try:
    from demo.lib.utils import normalize_screen_coordinates, camera_to_world
    from model.MotionAGFormer import MotionAGFormer
except ImportError as e:
    print(f"[Warning] Failed to import MotionAGFormer: {e}")


def interpolate_missing_keypoints(kpts):
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
    new_keypoints = np.zeros_like(keypoints)
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    new_keypoints[..., 1, :] = keypoints[..., 12, :]
    new_keypoints[..., 2, :] = keypoints[..., 14, :]
    new_keypoints[..., 3, :] = keypoints[..., 16, :]
    new_keypoints[..., 4, :] = keypoints[..., 11, :]
    new_keypoints[..., 5, :] = keypoints[..., 13, :]
    new_keypoints[..., 6, :] = keypoints[..., 15, :]
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    new_keypoints[..., 7, :] = (new_keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    new_keypoints[..., 9, :] = keypoints[..., 0, :]
    
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
    
    new_keypoints[..., 11, :] = keypoints[..., 5, :]
    new_keypoints[..., 12, :] = keypoints[..., 7, :]
    new_keypoints[..., 13, :] = keypoints[..., 9, :]
    new_keypoints[..., 14, :] = keypoints[..., 6, :]
    new_keypoints[..., 15, :] = keypoints[..., 8, :]
    new_keypoints[..., 16, :] = keypoints[..., 10, :]
    return new_keypoints


def convert_txt_to_npz(txt_path, output_npz_path):
    raw_lines = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                raw_lines.append([float(p) for p in parts[:5]])
    data = np.array(raw_lines)
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


def infer_agformer_3d(npz_2d_file, output_3d_file, model, w=480, h=640):
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


def norm_minmax(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)


def run_comparison(target_dir, ckpt_path=None):
    out_dir = os.path.join(target_dir, "AGFormer")
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join(target_dir, "yolo_skeleton.txt")
    npz_2d_path = os.path.join(out_dir, "keypoints.npz")
    npz_3d_path = os.path.join(out_dir, "keypoints_3d.npz")
    bar_txt_path = os.path.join(target_dir, "yolo_coordinates.txt")
    seg_json_path = os.path.join(target_dir, "segments.json")

    # Fallback search in target_dir if already generated there
    if not os.path.exists(npz_3d_path) and os.path.exists(os.path.join(target_dir, "keypoints_3d.npz")):
        npz_3d_path = os.path.join(target_dir, "keypoints_3d.npz")
    if not os.path.exists(npz_2d_path) and os.path.exists(os.path.join(target_dir, "keypoints.npz")):
        npz_2d_path = os.path.join(target_dir, "keypoints.npz")

    # 1. Check / Generate 3D npz
    if not os.path.exists(npz_3d_path):
        print("▶ Generating 3D keypoints from 2D YOLO Pose...")
        if not os.path.exists(npz_2d_path):
            convert_txt_to_npz(txt_path, npz_2d_path)
        if ckpt_path is None:
            ckpt_path = r"D:\Pitt\Project\tools\MotionAGFormer\checkpoint\motionagformer-b-h36m.pth.tr"
        model = load_agformer_model(ckpt_path)
        infer_agformer_3d(npz_2d_path, npz_3d_path, model, w=480, h=640)
        print("✅ keypoints_3d.npz created in AGFormer/!")

    # 2. Load 3D Keypoints & YOLO Barbell
    k3d = np.load(npz_3d_path)['reconstruction']
    
    bar_data = []
    with open(bar_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2: continue
            f_idx = int(parts[0])
            if 'no detection' in parts[1] or 'no detection' in parts[2]:
                bar_data.append([f_idx, np.nan, np.nan])
            else:
                try:
                    bar_data.append([f_idx, float(parts[1]), float(parts[2])])
                except ValueError:
                    bar_data.append([f_idx, np.nan, np.nan])

    df_bar = pd.DataFrame(bar_data, columns=['frame', 'bar_x', 'bar_y']).set_index('frame')
    N_total = len(k3d)
    df_bar = df_bar.reindex(range(1, N_total + 1)).interpolate(method='linear').bfill().ffill()

    bar_x = df_bar['bar_x'].values
    bar_y = df_bar['bar_y'].values
    bar_height = -bar_y

    # Segments
    segments = []
    if os.path.exists(seg_json_path):
        with open(seg_json_path, "r") as f:
            segments = json.load(f)

    N = min(len(k3d), len(df_bar))
    frames = np.arange(1, N + 1)

    # World Gravity Alignment
    standing_f = segments[0]['start'] - 1 if segments else 0
    # H36M Joints: 3 (R_Ankle), 6 (L_Ankle), 11 (L_Shoulder), 14 (R_Shoulder)
    ankle_mid = (k3d[standing_f, 3] + k3d[standing_f, 6]) / 2.0
    sh_mid = (k3d[standing_f, 11] + k3d[standing_f, 14]) / 2.0
    upright_vec = sh_mid - ankle_mid

    v_from = upright_vec / np.linalg.norm(upright_vec)
    v_to = np.array([0.0, 0.0, 1.0])
    v_cross = np.cross(v_from, v_to)
    s = np.linalg.norm(v_cross)
    c = float(np.dot(v_from, v_to))

    if s > 1e-6:
        vx = np.array([
            [0.0, -v_cross[2], v_cross[1]],
            [v_cross[2], 0.0, -v_cross[0]],
            [-v_cross[1], v_cross[0], 0.0]
        ])
        R_align = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))
    else:
        R_align = np.eye(3) if c > 0 else -np.eye(3)

    pitch_deg = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    print(f"-> Camera pitch alignment angle: {pitch_deg:.2f}° (Gravity aligned to +Z)")

    k3d_world = np.zeros_like(k3d)
    for f_idx in range(len(k3d)):
        for j_idx in range(17):
            k3d_world[f_idx, j_idx] = R_align @ k3d[f_idx, j_idx]

    # 3D Wrist Midpoint (H36M: L_Wrist=13, R_Wrist=16)
    lw_3d = k3d_world[:N, 13, :]
    rw_3d = k3d_world[:N, 16, :]
    wrist_mid_3d = (lw_3d + rw_3d) / 2.0

    w_z = wrist_mid_3d[:, 2]
    w_x = wrist_mid_3d[:, 0]

    # Metrics
    if segments:
        active_start = segments[0]['start'] - 1
        active_end = segments[-1]['end'] - 1
    else:
        active_start = 0
        active_end = N - 1

    corr_vert_active, _ = pearsonr(w_z[active_start:active_end+1], bar_height[active_start:active_end+1])
    corr_horiz_active, _ = pearsonr(w_x[active_start:active_end+1], bar_x[active_start:active_end+1])

    print(f"\n" + "=" * 70)
    print(f"📊 COMPARISON SUMMARY: AGFormer 3D Wrist Midpoint vs YOLO Barbell")
    print(f"Target Directory: {target_dir}")
    print(f"Output Directory: {out_dir}")
    print(f"Active Squat Range: Frames {active_start+1} ~ {active_end+1}")
    print(f"Vertical Trajectory Pearson Correlation r = {corr_vert_active:.4f}")
    print(f"Horizontal (Sagittal) Pearson Correlation r = {corr_horiz_active:.4f}")
    print("=" * 70)

    rep_stats = []
    if segments:
        for seg in segments:
            rep_id = seg['rep_id']
            st = seg['start'] - 1
            bot = seg['bottom'] - 1
            ed = seg['end'] - 1
            
            sub_w_z = w_z[st:ed+1]
            sub_w_x = w_x[st:ed+1]
            sub_b_h = bar_height[st:ed+1]
            sub_b_x = bar_x[st:ed+1]
            
            w_bot_f = st + np.argmin(sub_w_z) + 1
            b_bot_f = st + np.argmin(sub_b_h) + 1
            
            r_v, _ = pearsonr(sub_w_z, sub_b_h)
            r_h, _ = pearsonr(sub_w_x, sub_b_x)
            
            rep_stats.append({
                'rep_id': rep_id,
                'start': seg['start'],
                'true_bot': seg['bottom'],
                'wrist_bot': w_bot_f,
                'bar_bot': b_bot_f,
                'bot_diff_frames': w_bot_f - b_bot_f,
                'end': seg['end'],
                'vert_corr': r_v,
                'horiz_corr': r_h,
                'wrist_rom_m': np.max(sub_w_z) - np.min(sub_w_z),
                'bar_rom_px': np.max(sub_b_h) - np.min(sub_b_h)
            })

        df_reps = pd.DataFrame(rep_stats)
        print("\n=== Per-Repetition Breakdown ===")
        for _, r in df_reps.iterrows():
            print(f"Rep {int(r['rep_id']):2d} | Range: [{int(r['start']):4d}-{int(r['end']):4d}] | "
                  f"BarBot: {int(r['bar_bot']):4d}, WristBot: {int(r['wrist_bot']):4d} (Diff: {int(r['bot_diff_frames']):+2d}f) | "
                  f"Vert r: {r['vert_corr']:.4f} | Horiz r: {r['horiz_corr']:.4f} | Wrist ROM: {r['wrist_rom_m']:.3f}m")

        print(f"\nAverage Vertical Correlation across reps: {df_reps['vert_corr'].mean():.4f}")
        print(f"Average Absolute Bottom Timing Offset: {np.abs(df_reps['bot_diff_frames']).mean():.2f} frames ({np.abs(df_reps['bot_diff_frames']).mean()/30.0*1000:.1f} ms)")
        df_reps.to_csv(os.path.join(out_dir, "wrist_vs_bar_comparison_report.csv"), index=False)

    # Plot 1: Full Session Time Series
    cm_per_px = 0.0842

    # Normalize height using active squat range min/max so squat standing is 1.0 and bottom is 0.0
    w_min = np.min(w_z[active_start:active_end+1])
    w_max = np.max(w_z[active_start:active_end+1])
    norm_w_z = (w_z - w_min) / (w_max - w_min + 1e-6)

    b_min = np.min(bar_height[active_start:active_end+1])
    b_max = np.max(bar_height[active_start:active_end+1])
    norm_bar_h = (bar_height - b_min) / (b_max - b_min + 1e-6)

    # Convert to physical displacement in centimeters (cm) relative to standing posture
    w_standing = w_z[active_start]
    w_disp_cm = (w_z - w_standing) * 100.0

    b_standing = bar_height[active_start]
    b_disp_cm = (bar_height - b_standing) * cm_per_px

    plt.figure(figsize=(16, 12), dpi=150)
    plt.subplot(3, 1, 1)
    plt.plot(frames, norm_w_z, label='AGFormer 3D Wrist Midpoint (Z / Height)', color='#1f77b4', lw=2.0)
    plt.plot(frames, norm_bar_h, label='YOLO Detected Barbell (Inverted Y / Height)', color='#d62728', lw=1.8, linestyle='--')
    for seg in segments:
        plt.axvspan(seg['start'], seg['end'], color='yellow', alpha=0.15)
        plt.axvline(seg['bottom'], color='green', linestyle=':', alpha=0.6)
    plt.title(f"Full Session: Vertical Trajectory Comparison (3D Wrist vs YOLO Barbell) | Active Pearson r = {corr_vert_active:.4f}", fontsize=14, fontweight='bold')
    plt.ylabel("Normalized Height (0~1)", fontsize=11)
    plt.ylim([-0.1, 1.25])
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Physical Displacement in Centimeters (cm) on unified scale
    plt.subplot(3, 1, 2)
    plt.plot(frames, w_disp_cm, color='#1f77b4', lw=2.0, label='3D Wrist Height Drop (cm)')
    plt.plot(frames, b_disp_cm, color='#d62728', lw=1.8, linestyle='--', label='YOLO Barbell Height Drop (cm)')
    for seg in segments:
        plt.axvspan(seg['start'], seg['end'], color='yellow', alpha=0.15)
        plt.axvline(seg['bottom'], color='green', linestyle=':', alpha=0.6)
    plt.xlabel("Frame Number", fontsize=11)
    plt.ylabel("Vertical Displacement (cm)", fontsize=11)
    plt.title("Physical Units: Real Displacement in Centimeters (cm) | Standing Baseline = 0 cm", fontsize=14, fontweight='bold')
    plt.ylim([-55, 15])
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Subplot 3: Sagittal Displacement
    w_x_min = np.min(w_x[active_start:active_end+1])
    w_x_max = np.max(w_x[active_start:active_end+1])
    norm_w_x = (w_x - w_x_min) / (w_x_max - w_x_min + 1e-6)

    b_x_min = np.min(bar_x[active_start:active_end+1])
    b_x_max = np.max(bar_x[active_start:active_end+1])
    norm_bar_x = (bar_x - b_x_min) / (b_x_max - b_x_min + 1e-6)

    plt.subplot(3, 1, 3)
    plt.plot(frames, norm_w_x, label='AGFormer 3D Wrist Midpoint (Sagittal / A-P)', color='#2ca02c', lw=2.0)
    plt.plot(frames, norm_bar_x, label='YOLO Barbell Position (Sagittal / A-P)', color='#ff7f0e', lw=1.8, linestyle='--')
    for seg in segments:
        plt.axvspan(seg['start'], seg['end'], color='yellow', alpha=0.15)
    plt.title(f"Full Session: Horizontal / Sagittal Displacement Comparison | Active Pearson r = {corr_horiz_active:.4f}", fontsize=14, fontweight='bold')
    plt.xlabel("Frame Number (30 FPS)", fontsize=11)
    plt.ylabel("Normalized Sagittal Position (0~1)", fontsize=11)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    p1 = os.path.join(out_dir, "compare_wrist_vs_bar_trajectory.png")
    plt.savefig(p1)
    plt.close()
    print(f"   -> Saved: {p1}")

    # Plot 2: Per-rep 2D side path
    if segments:
        fig, axes = plt.subplots(2, 5, figsize=(20, 9), dpi=150)
        axes = axes.flatten()
        for i, seg in enumerate(segments):
            ax = axes[i]
            st = seg['start'] - 1
            ed = seg['end'] - 1
            sub_w_z = norm_minmax(w_z[st:ed+1])
            sub_w_x = norm_minmax(w_x[st:ed+1])
            sub_b_h = norm_minmax(bar_height[st:ed+1])
            sub_b_x = norm_minmax(bar_x[st:ed+1])
            ax.plot(sub_w_x, sub_w_z, color='#1f77b4', lw=2.5, label='3D Wrist Mid', marker='o', markersize=3, markevery=5)
            ax.plot(sub_b_x, sub_b_h, color='#d62728', lw=2.0, linestyle='--', label='YOLO Bar', marker='^', markersize=3, markevery=5)
            ax.scatter(sub_w_x[0], sub_w_z[0], color='green', s=60, zorder=5, label='Start' if i==0 else "")
            bot_idx = seg['bottom'] - seg['start']
            if 0 <= bot_idx < len(sub_w_x):
                ax.scatter(sub_w_x[bot_idx], sub_w_z[bot_idx], color='purple', s=60, zorder=5, label='Bottom' if i==0 else "")
            r_v = df_reps.loc[i, 'vert_corr']
            r_h = df_reps.loc[i, 'horiz_corr']
            ax.set_title(f"Rep {seg['rep_id']} (r_v={r_v:.3f}, r_h={r_h:.3f})", fontsize=11, fontweight='bold')
            ax.set_xlabel("Normalized A-P (X)", fontsize=9)
            ax.set_ylabel("Normalized Height (Z)", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 0: ax.legend(loc='best', fontsize=8)

        plt.suptitle("Side-View (Sagittal Plane) 2D Bar Path Trajectory: 3D Wrist Midpoint vs YOLO Barbell", fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        p2 = os.path.join(out_dir, "compare_wrist_vs_bar_reps_sideview.png")
        plt.savefig(p2)
        plt.close()
        print(f"   -> Saved: {p2}")

        # Plot 3: Per-rep waveforms
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=150)
        axes = axes.flatten()
        for i, enumerate_idx in enumerate(segments):
            seg = enumerate_idx
            ax = axes[i]
            st = seg['start'] - 1
            ed = seg['end'] - 1
            rep_len = ed - st + 1
            rep_frames = np.arange(rep_len)
            sub_w_z = norm_minmax(w_z[st:ed+1])
            sub_b_h = norm_minmax(bar_height[st:ed+1])
            ax.plot(rep_frames, sub_w_z, color='#1f77b4', lw=2.2, label='3D Wrist Z')
            ax.plot(rep_frames, sub_b_h, color='#d62728', lw=1.8, linestyle='--', label='YOLO Bar H')
            true_bot_local = seg['bottom'] - seg['start']
            wrist_bot_local = df_reps.loc[i, 'wrist_bot'] - seg['start']
            ax.axvline(true_bot_local, color='green', linestyle=':', label='Bar Bottom' if i==0 else "")
            ax.axvline(wrist_bot_local, color='blue', linestyle=':', label='Wrist Bottom' if i==0 else "")
            diff_f = df_reps.loc[i, 'bot_diff_frames']
            ax.set_title(f"Rep {seg['rep_id']} | Diff={diff_f:+d}f ({diff_f/30.0*1000:+.0f}ms)", fontsize=11, fontweight='bold')
            ax.set_xlabel("Frames within Rep", fontsize=9)
            ax.set_ylabel("Norm Height", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 0: ax.legend(loc='lower right', fontsize=8)

        plt.suptitle("Per-Repetition Waveform & Bottom Timing Alignment: 3D Wrist vs YOLO Barbell", fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        p3 = os.path.join(out_dir, "compare_wrist_vs_bar_rep_waveforms.png")
        plt.savefig(p3)
        plt.close()
        print(f"   -> Saved: {p3}")

        # Plot 4: 1:1 Real Physical Scale (Centimeters) Sagittal Bar Path (Per-Rep 2x5 Grid)
        # Calibrate pixel to cm scale
        cm_per_px = 0.0842
        w_x_cm = w_x * 100.0
        w_z_cm = w_z * 100.0
        bar_x_cm = bar_x * cm_per_px
        bar_z_cm = bar_height * cm_per_px

        fig, axes = plt.subplots(2, 5, figsize=(22, 11), dpi=150)
        axes = axes.flatten()
        for i, seg in enumerate(segments):
            ax = axes[i]
            st = seg['start'] - 1
            bot = seg['bottom'] - 1
            ed = seg['end'] - 1
            w_dx = w_x_cm[st:ed+1] - w_x_cm[st]
            w_dz = w_z_cm[st:ed+1] - w_z_cm[st]
            b_dx = -(bar_x_cm[st:ed+1] - bar_x_cm[st])
            b_dz = bar_z_cm[st:ed+1] - bar_z_cm[st]

            ax.plot(w_dx, w_dz, color='#1f77b4', lw=2.2, label='3D Wrist Midpoint (cm)', marker='o', markersize=2.5, markevery=4)
            ax.plot(b_dx, b_dz, color='#d62728', lw=2.0, linestyle='--', label='YOLO Barbell (cm)', marker='^', markersize=2.5, markevery=4)
            ax.scatter([0], [0], color='green', s=60, zorder=5, label='Start (0,0)' if i==0 else "")
            bot_idx = bot - st
            if 0 <= bot_idx < len(w_dx):
                ax.scatter(w_dx[bot_idx], w_dz[bot_idx], color='#1f77b4', s=60, edgecolors='black', zorder=5, label='Wrist Bottom' if i==0 else "")
                ax.scatter(b_dx[bot_idx], b_dz[bot_idx], color='#d62728', s=60, edgecolors='black', zorder=5, label='Bar Bottom' if i==0 else "")

            ax.set_aspect('equal')
            ax.set_xlim([-15, 15])
            ax.set_ylim([-50, 8])
            ax.set_title(f"Rep {seg['rep_id']} | Drop: {abs(w_dz[bot_idx]):.1f} cm", fontsize=11, fontweight='bold')
            ax.set_xlabel("A-P Horizontal (cm)", fontsize=9)
            ax.set_ylabel("Vertical Height (cm)", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 0: ax.legend(loc='lower left', fontsize=8, framealpha=0.9)

        plt.suptitle("1:1 Physical Scale (Centimeters) Sagittal Bar Path: 3D Wrist Midpoint vs YOLO Barbell", fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        p4 = os.path.join(out_dir, "compare_wrist_vs_bar_reps_1to1_cm.png")
        plt.savefig(p4)
        plt.close()
        print(f"   -> Saved: {p4}")

        # Plot 5: 1:1 Physical Scale (Centimeters) Superimposed (All 10 Reps)
        plt.figure(figsize=(10, 11), dpi=150)
        ax = plt.subplot(1, 1, 1)
        for i, seg in enumerate(segments):
            st = seg['start'] - 1
            ed = seg['end'] - 1
            w_dx = w_x_cm[st:ed+1] - w_x_cm[st]
            w_dz = w_z_cm[st:ed+1] - w_z_cm[st]
            b_dx = -(bar_x_cm[st:ed+1] - bar_x_cm[st])
            b_dz = bar_z_cm[st:ed+1] - bar_z_cm[st]
            alpha = 0.35 if i > 0 else 0.7
            ax.plot(w_dx, w_dz, color='#1f77b4', lw=1.5, alpha=alpha, label='3D Wrist Mid (Reps 1-10)' if i==0 else "")
            ax.plot(b_dx, b_dz, color='#d62728', lw=1.5, linestyle='--', alpha=alpha, label='YOLO Barbell (Reps 1-10)' if i==0 else "")

        ax.scatter([0], [0], color='green', s=100, zorder=6, label='Start Position (0, 0)')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_aspect('equal')
        ax.set_xlim([-18, 18])
        ax.set_ylim([-52, 8])
        ax.set_title("All 10 Squat Reps Superimposed (1:1 Physical Scale in cm)\nTrue Bar Path: ~40 cm Vertical Drop vs ~2-4 cm Horizontal Shift", fontsize=13, fontweight='bold')
        ax.set_xlabel("Sagittal Anterior-Posterior (A-P) Displacement (cm)", fontsize=11)
        ax.set_ylabel("Vertical Height Displacement (cm)", fontsize=11)
        ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        p5 = os.path.join(out_dir, "compare_wrist_vs_bar_all_reps_1to1_cm.png")
        plt.savefig(p5)
        plt.close()
        print(f"   -> Saved: {p5}")

    print(f"\n🎉 Analysis and visualizations successfully generated in {out_dir}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare AGFormer 3D Wrist Trajectory vs YOLO Barbell Trajectory")
    parser.add_argument("--target_dir", type=str, default=r"E:\squat\squat_dataset\S001_Pitt\session02\recording_20260207_163410", help="Target recording folder")
    parser.add_argument("--agformer_ckpt", type=str, default=r"D:\Pitt\Project\tools\MotionAGFormer\checkpoint\motionagformer-b-h36m.pth.tr", help="MotionAGFormer checkpoint")
    args = parser.parse_args()

    run_comparison(args.target_dir, args.agformer_ckpt)
