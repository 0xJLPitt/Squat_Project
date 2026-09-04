r"""
Multi-View Chessboard Calibrated 3D Wrist Midpoint vs 2D Barbell Trajectory Comparison
進行多視角棋盤格 3D 空間三角測量，提取 3D 手腕中點並與側向 2D 槓鈴軌跡進行精準運動學比對。

使用範例:
python compare_calibrated_wrist_vs_bar.py --target_dir "E:\squat\squat_dataset\S001_Pitt\session02\recording_20260207_163410"
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

# Ensure parent directory and tools are in sys.path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Camera mapping (Squat_2dto3d convention)
CAMS = {
    "vision2": "RR",
    "vision3": "RLU",
    "vision4": "FL",
    "vision5": "FR",
}

def load_intrinsics(intrinsic_dir):
    intrinsics = {}
    for cam, name in CAMS.items():
        path1 = os.path.join(intrinsic_dir, f"intrinsics_{cam}.npz")
        path2 = os.path.join(intrinsic_dir, f"intrinsics_{name}.npz")
        path = path1 if os.path.exists(path1) else path2
        if not os.path.exists(path):
            raise FileNotFoundError(f"Intrinsic not found for {cam}: {path1}")
        data = np.load(path)
        intrinsics[cam] = data['mtx']
    return intrinsics

def load_extrinsics(extrinsic_dir):
    extr = {}
    for pair in [("vision2", "vision3"), ("vision3", "vision4"), ("vision4", "vision5")]:
        path = os.path.join(extrinsic_dir, f"extrinsics_{pair[0]}_to_{pair[1]}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Extrinsic not found: {path}")
        data = np.load(path)
        extr[pair] = {"R": data["R"], "T": data["T"].reshape(3, 1)}
    return extr

def get_projection_matrices(intrinsics, extr):
    # World origin at vision2 (RR)
    R_global = {"vision2": np.eye(3)}
    T_global = {"vision2": np.zeros((3, 1))}
    pairs = [("vision2", "vision3"), ("vision3", "vision4"), ("vision4", "vision5")]
    for c1, c2 in pairs:
        R_rel = extr[(c1, c2)]["R"]
        T_rel = extr[(c1, c2)]["T"]
        R_global[c2] = R_rel @ R_global[c1]
        T_global[c2] = R_rel @ T_global[c1] + T_rel
        
    P = {}
    for cam in CAMS:
        K = intrinsics[cam]
        Rt = np.hstack((R_global[cam], T_global[cam]))
        P[cam] = K @ Rt
    return P

def load_2d_keypoints(target_dir):
    kpts_2d = {}
    for cam, name in CAMS.items():
        txt_path = os.path.join(target_dir, f"yolo_skeleton_{name}.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"2D keypoints not found: {txt_path}")
        data = np.loadtxt(txt_path, delimiter=',')
        if len(data) == 0:
            continue
        max_frame = int(np.max(data[:, 0]))
        arr = np.zeros((max_frame, 17, 2))
        for row in data:
            f = int(row[0]) - 1
            j = int(row[1])
            x, y = row[2], row[3]
            arr[f, j] = [x, y]
        kpts_2d[cam] = arr
    return kpts_2d

def triangulate_point(views, proj_matrices):
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

def smooth_3d_keypoints(kpts_3d, window_length=15, polyorder=3):
    num_frames, num_joints, dims = kpts_3d.shape
    smoothed = np.copy(kpts_3d)
    for j in range(num_joints):
        for d in range(dims):
            series = kpts_3d[:, j, d]
            missing = (series == 0.0)
            valid = ~missing
            if valid.any() and missing.any():
                valid_indices = np.where(valid)[0]
                missing_indices = np.where(missing)[0]
                series[missing_indices] = np.interp(missing_indices, valid_indices, series[valid_indices])
            if num_frames > window_length:
                smoothed[:, j, d] = savgol_filter(series, window_length, polyorder)
    return smoothed

def norm_minmax(arr):
    mn, mx = np.min(arr), np.max(arr)
    if mx - mn == 0:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def run_calibrated_comparison(
    target_dir,
    intrinsic_dir=r"E:\squat\recordings_20260507_0628_fixed棋盤格\intrinsics_vision",
    extrinsic_dir=r"E:\squat\recordings_20260507_0628_fixed棋盤格\recording_20260211_棋盤格"
):
    print("\n" + "="*70)
    print("🚀 Running Multi-View Chessboard Calibrated 3D Wrist vs 2D Barbell Comparison")
    print(f"Target Directory: {target_dir}")
    print(f"Intrinsics Dir  : {intrinsic_dir}")
    print(f"Extrinsics Dir  : {extrinsic_dir}")
    print("="*70 + "\n")

    out_dir = os.path.join(target_dir, "Checkerboard3d")
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Output folder: {out_dir}")

    # 1. Load Calibration
    print("\n[Step 1/5] Loading camera calibration parameters...")
    intrinsics = load_intrinsics(intrinsic_dir)
    extrinsics = load_extrinsics(extrinsic_dir)
    P = get_projection_matrices(intrinsics, extrinsics)

    # 2. Load 2D Keypoints & Triangulate
    print("[Step 2/5] Loading multi-view 2D YOLO keypoints and triangulating...")
    kpts_2d = load_2d_keypoints(target_dir)
    max_frames = min([kpts_2d[c].shape[0] for c in CAMS])
    print(f"-> Total synchronized frames across 4 cameras: {max_frames}")

    kpts_3d_raw = np.zeros((max_frames, 17, 3))
    for f in range(max_frames):
        for j in range(17):
            views = []
            for cam in CAMS:
                x, y = kpts_2d[cam][f, j]
                # Filter self-occlusions
                if j in [13, 14, 15, 16]:
                    l_idx = 15 if j >= 15 else 13
                    r_idx = 16 if j >= 15 else 14
                    x_l, y_l = kpts_2d[cam][f, l_idx]
                    x_r, y_r = kpts_2d[cam][f, r_idx]
                    if x_l != 0 and x_r != 0:
                        if np.sqrt((x_l - x_r)**2 + (y_l - y_r)**2) < 30.0:
                            continue
                if x != 0 and y != 0:
                    views.append((x, y, cam))
            kpts_3d_raw[f, j] = triangulate_point(views, P)

    print("-> Applying temporal smoothing (Savitzky-Golay)...")
    kpts_3d = smooth_3d_keypoints(kpts_3d_raw)

    # Save calibrated 3D keypoints (Camera space)
    npz_out = os.path.join(out_dir, "calibrated_keypoints_3d.npz")
    np.savez_compressed(npz_out, keypoints_3d=kpts_3d, keypoints_3d_raw=kpts_3d_raw)
    print(f"✅ Calibrated 3D keypoints saved to: {npz_out}")

    # Load segments.json if available
    seg_file = os.path.join(target_dir, "segments.json")
    segments = []
    if os.path.exists(seg_file):
        with open(seg_file, "r") as f:
            segments = json.load(f)

    # 3. Extract Wrist Midpoint & Load YOLO Barbell with Option A Kinematic Alignment
    print("[Step 3/5] Extracting 3D wrist midpoint with Option A Kinematic Self-Calibration...")
    # Option A: Compute descent vector from standing to bottom across reps
    raw_wrist_mid = (kpts_3d[:, 9, :] + kpts_3d[:, 10, :]) / 2.0
    if segments:
        drop_vecs = []
        for seg in segments:
            st = seg['start'] - 1
            bot = seg['bottom'] - 1
            vec = raw_wrist_mid[st] - raw_wrist_mid[bot]
            norm_v = np.linalg.norm(vec)
            if norm_v > 1e-4:
                drop_vecs.append(vec / norm_v)
        v_from = np.mean(drop_vecs, axis=0) if drop_vecs else np.array([0.0, -1.0, 0.0])
        v_from /= np.linalg.norm(v_from)
    else:
        v_from = np.array([0.0, -1.0, 0.0])

    # In camera coordinates, descent along Y-axis is downward, so upward vector has negative Y
    theta = np.arctan2(v_from[2], -v_from[1])
    print(f"-> Option A Kinematic Self-Calibration: Descent axis pitch angle = {np.degrees(theta):.2f}° (Gravity aligned to -Y)")

    R_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta),  np.cos(theta)]
    ])

    kpts_world = np.zeros_like(kpts_3d)
    for f_idx in range(len(kpts_3d)):
        for j_idx in range(17):
            kpts_world[f_idx, j_idx] = R_pitch @ kpts_3d[f_idx, j_idx]

    # Extract World 3D Wrist Midpoint
    calib_lw_3d = kpts_world[:, 9, :]
    calib_rw_3d = kpts_world[:, 10, :]
    calib_wrist_mid = (calib_lw_3d + calib_rw_3d) / 2.0

    # In World Coordinate System (aligned with Gravity):
    # World Vertical Height = -Y
    # World Sagittal A-P = +Z (Depth from rear)
    calib_h = -calib_wrist_mid[:, 1]
    calib_ap = calib_wrist_mid[:, 2]

    # Load YOLO Barbell (from RU side camera)
    coord_file = os.path.join(target_dir, "yolo_coordinates.txt")
    if not os.path.exists(coord_file):
        raise FileNotFoundError(f"yolo_coordinates.txt not found in {target_dir}")
    df_coord = pd.read_csv(coord_file, header=None)

    N = min(max_frames, len(df_coord))
    bar_x = pd.to_numeric(df_coord.iloc[:N, 1], errors='coerce').interpolate().bfill().ffill().values
    bar_y = pd.to_numeric(df_coord.iloc[:N, 2], errors='coerce').interpolate().bfill().ffill().values
    bar_h = -bar_y  # Inverted Y (Higher is Up)

    calib_h = calib_h[:N]
    calib_ap = calib_ap[:N]
    frames = np.arange(N)

    # 4. Statistical Evaluation
    print("[Step 4/5] Computing kinematic correlation & timing metrics...")
    st_active = segments[0]['start'] - 1 if segments else 0
    ed_active = segments[-1]['end'] - 1 if segments else N - 1

    r_vert_active, _ = pearsonr(calib_h[st_active:ed_active+1], bar_h[st_active:ed_active+1])
    r_horiz_active, _ = pearsonr(calib_ap[st_active:ed_active+1], bar_x[st_active:ed_active+1])

    rep_metrics = []
    if segments:
        for seg in segments:
            st = seg['start'] - 1
            ed = seg['end'] - 1
            r_v, _ = pearsonr(calib_h[st:ed+1], bar_h[st:ed+1])
            r_h, _ = pearsonr(calib_ap[st:ed+1], bar_x[st:ed+1])
            wrist_bot = st + np.argmin(calib_h[st:ed+1]) + 1
            diff_f = wrist_bot - seg['bottom']
            diff_ms = diff_f / 30.0 * 1000.0
            # Convert 3D units to meters
            rom_3d_units = np.max(calib_h[st:ed+1]) - np.min(calib_h[st:ed+1])
            w_rom = (rom_3d_units * 0.499) / 100.0

            rep_metrics.append({
                "rep_id": seg['rep_id'],
                "start_frame": seg['start'],
                "end_frame": seg['end'],
                "bar_bottom_frame": seg['bottom'],
                "wrist_bottom_frame": wrist_bot,
                "bot_diff_frames": diff_f,
                "bot_diff_ms": round(diff_ms, 1),
                "vert_corr": round(r_v, 4),
                "sagittal_corr": round(r_h, 4),
                "wrist_rom_m": round(w_rom, 4)
            })

    df_reps = pd.DataFrame(rep_metrics)
    csv_report = os.path.join(out_dir, "calib_wrist_vs_bar_comparison_report.csv")
    df_reps.to_csv(csv_report, index=False)

    print("\n" + "="*70)
    print(f"📊 SUMMARY: Chessboard Calibrated 3D Wrist Midpoint vs YOLO Barbell (RU)")
    print(f"Active Squat Range: Frames {st_active+1} ~ {ed_active+1}")
    print(f"Vertical Trajectory Pearson Correlation r = {r_vert_active:.4f}")
    print(f"Sagittal (A-P) Trajectory Pearson Correlation r = {r_horiz_active:.4f}")
    if segments:
        print(f"Average Vertical Correlation across reps: {df_reps['vert_corr'].mean():.4f}")
        print(f"Average Absolute Bottom Timing Offset: {np.abs(df_reps['bot_diff_frames']).mean():.2f} frames ({np.abs(df_reps['bot_diff_frames']).mean()/30.0*1000:.1f} ms)")
    print("="*70 + "\n")

    # 5. Visualizations
    print("[Step 5/5] Generating comprehensive diagnostic charts...")

    # Chart 1: Full Session Time Series
    cm_per_px = 0.0842
    cm_per_3d_unit = 0.499

    # Normalize height using active squat range min/max so squat standing is 1.0 and bottom is 0.0
    w_min = np.min(calib_h[st_active:ed_active+1])
    w_max = np.max(calib_h[st_active:ed_active+1])
    norm_w_h = (calib_h - w_min) / (w_max - w_min + 1e-6)

    b_min = np.min(bar_h[st_active:ed_active+1])
    b_max = np.max(bar_h[st_active:ed_active+1])
    norm_bar_h = (bar_h - b_min) / (b_max - b_min + 1e-6)

    # Convert to physical displacement in centimeters (cm) relative to standing posture
    w_standing = calib_h[st_active]
    w_disp_cm = (calib_h - w_standing) * cm_per_3d_unit

    b_standing = bar_h[st_active]
    b_disp_cm = (bar_h - b_standing) * cm_per_px

    plt.figure(figsize=(16, 12), dpi=150)
    plt.subplot(3, 1, 1)
    plt.plot(frames, norm_w_h, label='Calibrated 3D Wrist Midpoint (Height)', color='#1f77b4', lw=2.0)
    plt.plot(frames, norm_bar_h, label='YOLO Barbell Position (RU Image Height)', color='#d62728', lw=1.8, linestyle='--')
    for seg in segments:
        plt.axvspan(seg['start'], seg['end'], color='yellow', alpha=0.15)
        plt.axvline(seg['bottom'], color='green', linestyle=':', alpha=0.6)
    plt.title(f"Full Session: Vertical Trajectory Comparison (Calibrated 3D Wrist vs YOLO Barbell) | Active Pearson r = {r_vert_active:.4f}", fontsize=13, fontweight='bold')
    plt.ylabel("Normalized Height (0~1)", fontsize=11)
    plt.ylim([-0.1, 1.25])
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Physical Displacement in Centimeters (cm) on unified scale
    plt.subplot(3, 1, 2)
    plt.plot(frames, w_disp_cm, color='#1f77b4', lw=2.0, label='Calib 3D Wrist Height Drop (cm)')
    plt.plot(frames, b_disp_cm, color='#d62728', lw=1.8, linestyle='--', label='YOLO Barbell Height Drop (cm)')
    for seg in segments:
        plt.axvspan(seg['start'], seg['end'], color='yellow', alpha=0.15)
        plt.axvline(seg['bottom'], color='green', linestyle=':', alpha=0.6)
    plt.xlabel("Frame Number", fontsize=11)
    plt.ylabel("Vertical Displacement (cm)", fontsize=11)
    plt.title("Physical Units: Real Displacement in Centimeters (cm) | Standing Baseline = 0 cm", fontsize=13, fontweight='bold')
    plt.ylim([-55, 15])
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Subplot 3: Sagittal Displacement
    w_ap_min = np.min(calib_ap[st_active:ed_active+1])
    w_ap_max = np.max(calib_ap[st_active:ed_active+1])
    norm_calib_ap = (calib_ap - w_ap_min) / (w_ap_max - w_ap_min + 1e-6)

    b_x_min = np.min(bar_x[st_active:ed_active+1])
    b_x_max = np.max(bar_x[st_active:ed_active+1])
    norm_bar_x = (bar_x - b_x_min) / (b_x_max - b_x_min + 1e-6)

    plt.subplot(3, 1, 3)
    plt.plot(frames, norm_calib_ap, label='Calibrated 3D Wrist (Sagittal / A-P Depth)', color='#2ca02c', lw=2.0)
    plt.plot(frames, norm_bar_x, label='YOLO Barbell Position (RU X-axis)', color='#ff7f0e', lw=1.8, linestyle='--')
    for seg in segments:
        plt.axvspan(seg['start'], seg['end'], color='yellow', alpha=0.15)
    plt.title(f"Full Session: Sagittal / Horizontal Displacement Comparison | Active Pearson r = {r_horiz_active:.4f}", fontsize=13, fontweight='bold')
    plt.xlabel("Frame Number (30 FPS)", fontsize=11)
    plt.ylabel("Normalized Sagittal Position (0~1)", fontsize=11)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    p1 = os.path.join(out_dir, "compare_calib_wrist_vs_bar_trajectory.png")
    plt.savefig(p1)
    plt.close()
    print(f"   -> Saved: {p1}")

    if segments:
        # Chart 2: Per-rep waveforms & Bottom Timing Alignment
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=150)
        axes = axes.flatten()
        for i, seg in enumerate(segments):
            ax = axes[i]
            st = seg['start'] - 1
            ed = seg['end'] - 1
            rep_frames = np.arange(ed - st + 1)
            sub_w_z = norm_minmax(calib_h[st:ed+1])
            sub_b_h = norm_minmax(bar_h[st:ed+1])
            ax.plot(rep_frames, sub_w_z, color='#1f77b4', lw=2.2, label='Calib 3D Wrist')
            ax.plot(rep_frames, sub_b_h, color='#d62728', lw=1.8, linestyle='--', label='YOLO Bar')
            true_bot = seg['bottom'] - seg['start']
            wrist_bot = df_reps.loc[i, 'wrist_bottom_frame'] - seg['start']
            ax.axvline(true_bot, color='green', linestyle=':', label='Bar Bottom' if i==0 else "")
            ax.axvline(wrist_bot, color='blue', linestyle=':', label='Wrist Bottom' if i==0 else "")
            diff_f = int(df_reps.loc[i, 'bot_diff_frames'])
            ax.set_title(f"Rep {seg['rep_id']} | Diff={diff_f:+d}f ({diff_f/30.0*1000:+.0f}ms)", fontsize=11, fontweight='bold')
            ax.set_xlabel("Frames within Rep", fontsize=9)
            ax.set_ylabel("Norm Height", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 0: ax.legend(loc='lower right', fontsize=8)

        plt.suptitle("Per-Repetition Waveform & Bottom Timing Alignment: Calibrated 3D Wrist vs YOLO Barbell", fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        p2 = os.path.join(out_dir, "compare_calib_wrist_vs_bar_waveforms.png")
        plt.savefig(p2)
        plt.close()
        print(f"   -> Saved: {p2}")

        # Chart 3 & 4: 1:1 Physical Scale (Centimeters)
        # Barbell pixel to cm scale (1 px ~= 0.0842 cm from Olympic 45cm plate / 41.6cm ROM)
        cm_per_px = 0.0842
        # Calibrate 3D units to cm: each 3D unit corresponds to (rom_px * cm_per_px) / rom_3d_units
        scales_3d_to_cm = []
        for seg in segments:
            st, bot = seg['start'] - 1, seg['bottom'] - 1
            rom_3d = calib_h[st] - calib_h[bot]
            rom_px = bar_h[st] - bar_h[bot]
            if rom_3d > 0 and rom_px > 0:
                scales_3d_to_cm.append((rom_px * cm_per_px) / rom_3d)
        cm_per_3d_unit = np.median(scales_3d_to_cm) if scales_3d_to_cm else 0.50

        w_ap_cm = calib_ap * cm_per_3d_unit
        w_h_cm = calib_h * cm_per_3d_unit
        bar_x_cm = bar_x * cm_per_px
        bar_h_cm = bar_h * cm_per_px

        # Grid 1:1
        fig, axes = plt.subplots(2, 5, figsize=(22, 11), dpi=150)
        axes = axes.flatten()
        for i, seg in enumerate(segments):
            ax = axes[i]
            st = seg['start'] - 1
            bot = seg['bottom'] - 1
            ed = seg['end'] - 1
            w_dx = w_ap_cm[st:ed+1] - w_ap_cm[st]
            w_dz = w_h_cm[st:ed+1] - w_h_cm[st]
            b_dx = bar_x_cm[st:ed+1] - bar_x_cm[st]
            b_dz = bar_h_cm[st:ed+1] - bar_h_cm[st]

            ax.plot(w_dx, w_dz, color='#1f77b4', lw=2.2, label='Calib 3D Wrist (cm)', marker='o', markersize=2.5, markevery=4)
            ax.plot(b_dx, b_dz, color='#d62728', lw=2.0, linestyle='--', label='YOLO Bar (cm)', marker='^', markersize=2.5, markevery=4)
            ax.scatter([0], [0], color='green', s=60, zorder=5, label='Start (0,0)' if i==0 else "")
            bot_idx = bot - st
            if 0 <= bot_idx < len(w_dx):
                ax.scatter(w_dx[bot_idx], w_dz[bot_idx], color='#1f77b4', s=60, edgecolors='black', zorder=5)
                ax.scatter(b_dx[bot_idx], b_dz[bot_idx], color='#d62728', s=60, edgecolors='black', zorder=5)

            ax.set_aspect('equal')
            ax.set_xlim([-15, 15])
            ax.set_ylim([-50, 8])
            ax.set_title(f"Rep {seg['rep_id']} | Drop: {abs(w_dz[bot_idx]):.1f} cm", fontsize=11, fontweight='bold')
            ax.set_xlabel("A-P Sagittal (cm)", fontsize=9)
            ax.set_ylabel("Vertical Height (cm)", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 0: ax.legend(loc='lower left', fontsize=8, framealpha=0.9)

        plt.suptitle("1:1 Physical Scale (Centimeters) Sagittal Bar Path: Calibrated 3D Wrist vs YOLO Barbell", fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        p3 = os.path.join(out_dir, "compare_calib_wrist_vs_bar_reps_1to1_cm.png")
        plt.savefig(p3)
        plt.close()
        print(f"   -> Saved: {p3}")

        # Superimposed 1:1
        plt.figure(figsize=(10, 11), dpi=150)
        ax = plt.subplot(1, 1, 1)
        for i, seg in enumerate(segments):
            st = seg['start'] - 1
            ed = seg['end'] - 1
            w_dx = w_ap_cm[st:ed+1] - w_ap_cm[st]
            w_dz = w_h_cm[st:ed+1] - w_h_cm[st]
            b_dx = bar_x_cm[st:ed+1] - bar_x_cm[st]
            b_dz = bar_h_cm[st:ed+1] - bar_h_cm[st]
            alpha = 0.35 if i > 0 else 0.75
            ax.plot(w_dx, w_dz, color='#1f77b4', lw=1.5, alpha=alpha, label='Calib 3D Wrist (Reps 1-10)' if i==0 else "")
            ax.plot(b_dx, b_dz, color='#d62728', lw=1.5, linestyle='--', alpha=alpha, label='YOLO Barbell (Reps 1-10)' if i==0 else "")

        ax.scatter([0], [0], color='green', s=100, zorder=6, label='Start Position (0, 0)')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_aspect('equal')
        ax.set_xlim([-18, 18])
        ax.set_ylim([-52, 8])
        ax.set_title("Calibrated 3D Wrist vs YOLO Barbell (All 10 Reps, 1:1 cm Scale)\nMulti-View Triangulation: ~40 cm Vertical Drop vs ~2-4 cm Horizontal Shift", fontsize=13, fontweight='bold')
        ax.set_xlabel("Sagittal Anterior-Posterior (A-P) Displacement (cm)", fontsize=11)
        ax.set_ylabel("Vertical Height Displacement (cm)", fontsize=11)
        ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        p4 = os.path.join(out_dir, "compare_calib_wrist_vs_bar_all_reps_1to1_cm.png")
        plt.savefig(p4)
        plt.close()
        print(f"   -> Saved: {p4}")

        # Chart 5: Normalized Side View (0~1)
        fig, axes = plt.subplots(2, 5, figsize=(20, 9), dpi=150)
        axes = axes.flatten()
        for i, seg in enumerate(segments):
            ax = axes[i]
            st = seg['start'] - 1
            ed = seg['end'] - 1
            sub_w_z = norm_minmax(calib_h[st:ed+1])
            sub_w_x = norm_minmax(calib_ap[st:ed+1])
            sub_b_h = norm_minmax(bar_h[st:ed+1])
            sub_b_x = norm_minmax(bar_x[st:ed+1])
            ax.plot(sub_w_x, sub_w_z, color='#1f77b4', lw=2.5, label='Calib 3D Wrist', marker='o', markersize=3, markevery=5)
            ax.plot(sub_b_x, sub_b_h, color='#d62728', lw=2.0, linestyle='--', label='YOLO Bar', marker='^', markersize=3, markevery=5)
            ax.scatter(sub_w_x[0], sub_w_z[0], color='green', s=60, zorder=5, label='Start' if i==0 else "")
            bot_idx = seg['bottom'] - seg['start']
            if 0 <= bot_idx < len(sub_w_x):
                ax.scatter(sub_w_x[bot_idx], sub_w_z[bot_idx], color='purple', s=60, zorder=5, label='Bottom' if i==0 else "")
            r_v = df_reps.loc[i, 'vert_corr']
            r_h = df_reps.loc[i, 'sagittal_corr']
            ax.set_title(f"Rep {seg['rep_id']} (r_v={r_v:.3f}, r_h={r_h:.3f})", fontsize=11, fontweight='bold')
            ax.set_xlabel("Normalized A-P (Depth)", fontsize=9)
            ax.set_ylabel("Normalized Height", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 0: ax.legend(loc='best', fontsize=8)

        plt.suptitle("Side-View (Sagittal Plane) 2D Bar Path: Calibrated 3D Wrist vs YOLO Barbell", fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        p5 = os.path.join(out_dir, "compare_calib_wrist_vs_bar_reps_sideview.png")
        plt.savefig(p5)
        plt.close()
        print(f"   -> Saved: {p5}")

        # Chart 6: Tri-way Comparison (Chessboard 3D vs AGFormer 3D vs YOLO Barbell)
        agf_npz_candidates = [
            os.path.join(target_dir, "AGFormer", "keypoints_3d.npz"),
            os.path.join(target_dir, "keypoints_3d.npz")
        ]
        agf_npz = next((p for p in agf_npz_candidates if os.path.exists(p)), None)
        if agf_npz:
            agf_data = np.load(agf_npz)['reconstruction']
            agf_wrist = (agf_data[:N, 13, :] + agf_data[:N, 16, :]) / 2.0
            agf_z = agf_wrist[:, 2]

            fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=150)
            axes = axes.flatten()
            for i, seg in enumerate(segments):
                ax = axes[i]
                st = seg['start'] - 1
                ed = seg['end'] - 1
                rep_frames = np.arange(ed - st + 1)
                sub_calib = norm_minmax(calib_h[st:ed+1])
                sub_agf = norm_minmax(agf_z[st:ed+1])
                sub_bar = norm_minmax(bar_h[st:ed+1])

                ax.plot(rep_frames, sub_calib, color='#1f77b4', lw=2.2, label='Chessboard 3D Wrist')
                ax.plot(rep_frames, sub_agf, color='#2ca02c', lw=1.8, linestyle='-.', label='AGFormer 3D Wrist')
                ax.plot(rep_frames, sub_bar, color='#d62728', lw=1.8, linestyle='--', label='YOLO 2D Barbell')
                ax.set_title(f"Rep {seg['rep_id']}", fontsize=11, fontweight='bold')
                ax.set_xlabel("Frames", fontsize=9)
                ax.set_ylabel("Norm Height", fontsize=9)
                ax.grid(True, linestyle=':', alpha=0.6)
                if i == 0: ax.legend(loc='lower right', fontsize=8)

            plt.suptitle("Tri-Way Squat Comparison: Chessboard 3D Wrist vs AGFormer 3D Wrist vs YOLO 2D Barbell", fontsize=15, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            p6 = os.path.join(out_dir, "compare_calib_vs_agformer_vs_bar.png")
            plt.savefig(p6)
            plt.close()
            print(f"   -> Saved: {p6}")

    print("\n🎉 Checkerboard3D analysis and visualizations successfully created in Checkerboard3d/!")
    return df_reps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-View Chessboard Calibrated 3D Wrist vs 2D Barbell Comparison")
    parser.add_argument("--target_dir", type=str, default=r"E:\squat\squat_dataset\S001_Pitt\session02\recording_20260207_163410", help="Target recording folder")
    parser.add_argument("--intrinsic_dir", type=str, default=r"E:\squat\recordings_20260507_0628_fixed棋盤格\intrinsics_vision", help="Path to camera intrinsics folder")
    parser.add_argument("--extrinsic_dir", type=str, default=r"E:\squat\recordings_20260507_0628_fixed棋盤格\recording_20260211_棋盤格", help="Path to camera extrinsics folder")
    args = parser.parse_args()

    run_calibrated_comparison(args.target_dir, args.intrinsic_dir, args.extrinsic_dir)
