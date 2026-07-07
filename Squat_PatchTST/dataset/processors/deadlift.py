"""
Data Collection and Feature Engineering for Deadlift Dataset:

Base Features (8 dimensions per frame):
    1. Left Knee Angle
    2. Left Hip Angle
    3. Right Knee Angle
    4. Right Hip Angle
    5. Left Arm-Torso Angle
    6. Right Arm-Torso Angle
    7. Barbell X-Coordinate
    8. Barbell Y-Coordinate

Feature Transformation (5 variations):
    For each base feature, we calculate 5 variations:
    1. Original Interpolated Value (fn)
    2. Motion Velocity (fdn) - Delta with initial 0
    3. Variation Ratio (fd2n) - Velocity divided by previous value
    4. Z-Score (fzn) - Standardized value
    5. Acceleration (fdsn) - Delta of velocity

Total Features per Frame:
    8 base features * 5 variations = 40 features per frame (scaled to [-1, 1]).
"""
import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataset.tools.interpolate import run_interpolation
from dataset.tools.Benchpress_tool.hampel import run_hampel_bar, run_hampel_yolo_ske_left_front
from dataset.tools.Deadlift_tool.data_produce import run_data_produce
from dataset.tools.Deadlift_tool.data_split import run_data_split

def pre_process(video_path: str):
    # Run the standard pipeline
    import time
    memo = {}
    
    def run_step(name, func, args, kwargs={}):
        t0 = time.time()
        res = func(*args, **kwargs)
        memo[name] = res
        print(f"[DeadliftProcessor] {name} time : {time.time() - t0:.2f}s")
        return res

    run_step("Interpolation", run_interpolation, [video_path])
    run_step("Hampel Bar", run_hampel_bar, [video_path], {"sport": 'deadlift'})
    run_step("Hampel Skeleton", run_hampel_yolo_ske_left_front, [video_path])
    run_step("Angle Data", run_data_produce, [video_path])
    res = run_step("Data Split", run_data_split, [video_path])
    return memo, res

# deadliftdim=8
# bar x, y
# knee, hip, torso-arm

# benchpress dim=12
# bar x, y
# shoulder(y, angle), torso-arm, elbow, distance(wrist to shoulder line)

def apply_augmentation(df):
    """
    Placeholder for data augmentation (e.g., jittering, scaling, time-warping).
    """
    # Example: return df + np.random.normal(0, 0.01, df.shape)
    return df

def standardize_features(seq):
    """
    Fixed-scale zero-centering to preserve physical amplitude and handle camera shifts.
    seq: numpy array of shape (N, 8)
    """
    import numpy as np
    out = np.zeros_like(seq, dtype=np.float32)
    # 0-5 are Angles (Knee, Hip, Torso-arm) -> center at 90, scale by 90
    out[:, 0:6] = (seq[:, 0:6] - 90.0) / 90.0
    # 6 is Bar X -> subtract first frame, scale by 320 (half width)
    out[:, 6] = (seq[:, 6] - seq[0, 6]) / 320.0
    # 7 is Bar Y -> subtract first frame, scale by 240 (half height)
    out[:, 7] = (seq[:, 7] - seq[0, 7]) / 240.0
    # Clip to avoid extreme outliers
    return np.clip(out, -1.0, 1.0)

def generate_csv(dataset_dir, output_csv):
    import os
    import pandas as pd
    import numpy as np
    import json
    
    # Load multi-error mapping
    multi_error_path = os.path.join(dataset_dir, "multierror.json")
    multi_labels_map = {} # (subject, set, clip) -> set of errors
    if os.path.exists(multi_error_path):
        with open(multi_error_path, 'r') as f:
            me_data = json.load(f)
            for subject, mistake_groups in me_data.items():
                for group in mistake_groups:
                    for error_info in group:
                        err_name = error_info["error"]
                        set_name = error_info["set"]
                        for clip in error_info["clips"]:
                            key = (subject, set_name, str(clip))
                            if key not in multi_labels_map:
                                multi_labels_map[key] = set()
                            multi_labels_map[key].add(err_name)

    data = []
    processed_clips = set() # To avoid duplicate reps across different folders
    
    error_order = [
        "Barbell_moving_away_from_the_shins",
        "Hips_rising_before_the_barbell_leaves_the_ground",
        "Barbell_colliding_with_the_knees",
        "Lower_back_rounding"
    ]
    
    # Process DeadliftDataset
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} not found.")
        return
        
    for label_dir in os.listdir(dataset_dir):
        full_label_dir = os.path.join(dataset_dir, label_dir)
        if not os.path.isdir(full_label_dir):
            continue
            
        for subject_dir in os.listdir(full_label_dir):
            subject_path = os.path.join(full_label_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue
                
            for set_dir in os.listdir(subject_path):
                set_path = os.path.join(subject_path, set_dir)
                if not os.path.isdir(set_path):
                    continue
                
                angle_3d_dir = os.path.join(set_path, "Angle", "3D")
                bar_dir = os.path.join(set_path, "Coordinate", "bar")
                if os.path.exists(angle_3d_dir) and os.path.isdir(angle_3d_dir):
                    for file in os.listdir(angle_3d_dir):
                        if file.endswith(".csv"):
                            # Deduplicate based on Subject/Set/File relative path
                            dedup_id = (label_dir, subject_dir, set_dir, file)
                            if dedup_id in processed_clips:
                                continue
                            processed_clips.add(dedup_id)
                            
                            file_path = os.path.join(angle_3d_dir, file)
                            
                            clip_idx = file.replace("angle_", "").replace(".csv", "")
                            bar_file = os.path.join(bar_dir, f"bar_{clip_idx}.csv")
                            
                            print(f"Processing 3D Angle file: {file_path}")
                            try:
                                # 1. Construct the Multi-label
                                label_vec = [0, 0, 0, 0]
                                active_errors = set()
                                # Add the directory-based primary error
                                if label_dir in error_order:
                                    active_errors.add(label_dir)
                                
                                # Check multierror JSON for this specific subject/set/clip
                                # subject_dir might contain 'subject3' etc.
                                me_key = (subject_dir, set_dir, clip_idx)
                                if me_key in multi_labels_map:
                                    active_errors.update(multi_labels_map[me_key])
                                
                                for i, err in enumerate(error_order):
                                    if err in active_errors:
                                        label_vec[i] = 1
                                        
                                # 2. Start extracting and merging features
                                df_3d = pd.read_csv(file_path, header=None)
                                
                                # Drop body length (col 5). Keep joints: 1: left knee angle, 2: left hip angle, 3: right knee angle, 4: right hip angle, 6: left arm-torso angle, 7: right arm-torso angle
                                # Note: index 0 is frame, so we skip it.
                                df_3d_filtered = df_3d.iloc[:, [1, 2, 3, 4, 6, 7]]
                                
                                # Add bar_x and bar_y from bar file
                                if os.path.exists(bar_file):
                                    df_bar = pd.read_csv(bar_file, header=None)
                                    features_bar_arr = df_bar.iloc[:, [1, 2]].values
                                else:
                                    features_bar_arr = np.zeros((len(df_3d_filtered), 2))
                                
                                # Merge frame by frame
                                # Make sure they have the same length
                                min_len = min(len(df_3d_filtered), len(features_bar_arr))
                                merged_features = np.concatenate([df_3d_filtered.values[:min_len], features_bar_arr[:min_len]], axis=1)
                                
                                from dataset.tools.Deadlift_tool.utils import interpolate_features
                                from dataset.tools.Deadlift_tool.data_split import process_delta_ratio
                                
                                # Data Augmentation (Placeholder)
                                merged_features = apply_augmentation(merged_features)

                                interpolated_raw = interpolate_features(merged_features, 110)
                                
                                # 1. Standardize (Zero-Centering & Fixed Scaling)
                                base_seq = standardize_features(interpolated_raw)
                                
                                # 2. Calculate derivatives on standardized features
                                # Velocity (diff)
                                delta = np.vstack([np.zeros(8), np.diff(base_seq, axis=0)])
                                # Acceleration (diff of diff)
                                delta2 = np.vstack([np.zeros(8), np.diff(delta, axis=0)])
                                
                                # Ratio (using old function on raw values to avoid div by zero near zero-centered values)
                                filtered_interpolated = {"0": interpolated_raw}
                                delta_ratio_feature = process_delta_ratio(filtered_interpolated)["0"]
                                # But we need to clip the ratio to reasonable range
                                delta_ratio_feature = np.clip(delta_ratio_feature, -1.0, 1.0)
                                
                                # 3. Map to fn, fdn, fzn, fdsn, fd2n
                                fn = base_seq.tolist()
                                # Multiply velocity and accel by constants to make them network-friendly
                                fdn = np.clip(delta * 20.0, -1.0, 1.0).tolist()
                                fdsn = np.clip(delta2 * 400.0, -1.0, 1.0).tolist()
                                fzn = base_seq.tolist()  # Replace sequence-level Z-score with standardized base
                                fd2n = delta_ratio_feature.tolist()
                                
                                # Combine all 5 normalizations into [110, 40] array
                                all_feat = np.concatenate([
                                    np.array(fn), 
                                    np.array(fdn), 
                                    np.array(fd2n), 
                                    np.array(fzn), 
                                    np.array(fdsn)
                                ], axis=-1)
                                
                                import re
                                raw_set_val = int(re.search(r'\d+', os.path.basename(set_dir)).group())
                                set_val = f"{label_dir}_{raw_set_val}"
                                clip_val = int(re.search(r'\d+', os.path.basename(file)).group())
                                
                                sub_match = re.search(r'subject?\d+', os.path.basename(subject_dir))
                                sub_name = sub_match.group() if sub_match else os.path.basename(subject_dir)
                                
                                data.append({
                                    "subject": sub_name,
                                    "set": set_val,
                                    "clip": clip_val,
                                    "features": str(all_feat.tolist()),
                                    "label": str(label_vec)
                                })
                            except Exception as e:
                                print(f"Error processing {file_path}: {e}")
                                
                else:
                    print(f"Skipping {set_path} (No 3D Angle Data found)")
                    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

if __name__ == "__main__":
    import os
    os.makedirs("./data", exist_ok=True)
    generate_csv(r"E:\DeadliftDataset_0408", "./data/deadlift_dataset.csv")
