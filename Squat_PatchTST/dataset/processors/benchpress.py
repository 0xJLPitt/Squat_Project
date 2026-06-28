import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataset.tools.interpolate import run_interpolation
from dataset.tools.Benchpress_tool.hampel import run_hampel_bar, run_hampel_yolo_ske_rear, run_hampel_yolo_ske_top
from dataset.tools.Benchpress_tool.torso_angle_produce import run_torso_angle_produce
from dataset.tools.Benchpress_tool.autocutting import run_autocutting


def pre_process(video_path: str):
    import time
    memo = {}
    
    def run_step(name, func, args, kwargs={}):
        t0 = time.time()
        res = func(*args, **kwargs)
        memo[name] = res
        print(f"[BenchpressProcessor] {name} time : {time.time() - t0:.2f}s")
        return res

    run_step("Interpolation", run_interpolation, [video_path])
    bar_dict = run_step("Hampel Bar", run_hampel_bar, [video_path], {"sport": 'benchpress'})
    rear_ske_dict = run_step("Hampel Rear", run_hampel_yolo_ske_rear, [video_path])
    top_ske_dict = run_step("Hampel Top", run_hampel_yolo_ske_top, [video_path])
    run_step("Angle Data", run_torso_angle_produce, [video_path], {"skeleton_dict": top_ske_dict})
    split_info = run_step("Autocutting", run_autocutting, [video_path], {"bar_dict": bar_dict, "rear_ske_dict": rear_ske_dict})
    return memo

# deadliftdim=8
# bar x,y
# knee, hip, torso-arm

# benchpress dim=12
# bar x,y
# shoulder(y, angle), torso-arm, elbow, distance(wrist to shoulder line)

def apply_augmentation(df):
    """
    Placeholder for data augmentation (e.g., jittering, scaling, time-warping).
    """
    # Example: return df + np.random.normal(0, 0.01, df.shape)
    return df

def generate_csv(dataset_dir, output_csv):
    import os
    import pandas as pd
    import json
    import numpy as np
    
    data = []
    
    # Process BenchpressDataset
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} not found.")
        return
        
    for label_dir in os.listdir(dataset_dir):
        if label_dir == "wrist_bending_backward":
            continue
        full_label_dir = os.path.join(dataset_dir, label_dir)
        if not os.path.isdir(full_label_dir):
            continue
            
        for subject_dir in os.listdir(full_label_dir):
            subject_path = os.path.join(full_label_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue
                
            coordinate_base = os.path.join(subject_path, "coordinate_dataset")
            angle_base = os.path.join(subject_path, "angle_dataset")
            if not os.path.exists(coordinate_base):
                continue
                
            lat_dir = os.path.join(coordinate_base, "lateral_view")
            rear_dir = os.path.join(coordinate_base, "rear_view")
            top_dir = os.path.join(coordinate_base, "top_view")
            
            if not (os.path.exists(lat_dir) and os.path.exists(rear_dir) and os.path.exists(top_dir)):
                continue
                
            for file in os.listdir(lat_dir):
                if file.endswith(".txt"):
                    try:
                        lat_path = os.path.join(lat_dir, file)
                        rear_path = os.path.join(rear_dir, file)
                        top_path = os.path.join(top_dir, file)
                        
                        if not (os.path.exists(rear_path) and os.path.exists(top_path)):
                            continue
                            
                        # Load pre-calculated angles from angle_dataset
                        def load_angle_file(view, feature, filename):
                            p = os.path.join(angle_base, view, feature, filename)
                            res = {}
                            if os.path.exists(p):
                                try:
                                    with open(p, 'r') as f_ang:
                                        for l_ang in f_ang:
                                            p_ang = l_ang.strip().split(',')
                                            if len(p_ang) >= 2:
                                                res[int(p_ang[0])] = float(p_ang[1])
                                except Exception as e:
                                    print(f"Error loading angle file {p}: {e}")
                            return res

                        angle_dicts = {
                            "left_elbow": load_angle_file("rear_view", "left_elbow", file),
                            "right_elbow": load_angle_file("rear_view", "right_elbow", file),
                            "left_shoulder": load_angle_file("rear_view", "left_shoulder", file),
                            "right_shoulder": load_angle_file("rear_view", "right_shoulder", file),
                            "left_torso-arm": load_angle_file("top_view", "left_torso-arm", file),
                            "right_torso-arm": load_angle_file("top_view", "right_torso-arm", file),
                        }

                        bar_dict = {}
                        with open(lat_path, 'r') as f_lat:
                            for line in f_lat:
                                vals = [float(v) for v in line.strip().split(',')]
                                if len(vals) >= 3:
                                    bar_dict[int(vals[0])] = [vals[1], vals[2]]
                                    
                        rear_ske_dict = {}
                        with open(rear_path, 'r') as f_rear:
                            import re
                            for line in f_rear:
                                line = line.strip()
                                if not line: continue
                                if line.startswith("Frame"):
                                    parts = line.split(":")
                                    frame_idx = int(parts[0].replace("Frame", "").strip())
                                    vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", parts[1])]
                                    if len(vals) >= 12:
                                        rear_ske_dict[frame_idx] = vals[:12]
                                else:
                                    vals = [float(v) for v in line.split(',')]
                                    if len(vals) >= 13:
                                        rear_ske_dict[int(vals[0])] = vals[1:13]
                                        
                        top_ske_dict = {}
                        with open(top_path, 'r') as f_top:
                            import re
                            for line in f_top:
                                line = line.strip()
                                if not line: continue
                                if line.startswith("Frame"):
                                    parts = line.split(":")
                                    frame_idx = int(parts[0].replace("Frame", "").strip())
                                    vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", parts[1])]
                                    if len(vals) >= 16:
                                        r_sho_x, r_sho_y = vals[0], vals[1]
                                        l_sho_x, l_sho_y = vals[2], vals[3]
                                        r_hip_x, r_hip_y = vals[4], vals[5]
                                        l_hip_x, l_hip_y = vals[6], vals[7]
                                        r_elb_x, r_elb_y = vals[8], vals[9]
                                        l_elb_x, l_elb_y = vals[10], vals[11]
                                        r_wri_x, r_wri_y = vals[12], vals[13]
                                        l_wri_x, l_wri_y = vals[14], vals[15]
                                        
                                        top_ske_dict[frame_idx] = [
                                            l_sho_x, l_sho_y, r_sho_x, r_sho_y,
                                            l_hip_x, l_hip_y, r_hip_x, r_hip_y,
                                            l_elb_x, l_elb_y, r_elb_x, r_elb_y,
                                            l_wri_x, l_wri_y, r_wri_x, r_wri_y
                                        ]
                                else:
                                    vals = [float(v) for v in line.split(',')]
                                    if len(vals) >= 17:
                                        r_sho_x, r_sho_y = vals[1], vals[2]
                                        l_sho_x, l_sho_y = vals[3], vals[4]
                                        r_hip_x, r_hip_y = vals[5], vals[6]
                                        l_hip_x, l_hip_y = vals[7], vals[8]
                                        r_elb_x, r_elb_y = vals[9], vals[10]
                                        l_elb_x, l_elb_y = vals[11], vals[12]
                                        r_wri_x, r_wri_y = vals[13], vals[14]
                                        l_wri_x, l_wri_y = vals[15], vals[16]
                                        
                                        top_ske_dict[int(vals[0])] = [
                                            l_sho_x, l_sho_y, r_sho_x, r_sho_y,
                                            l_hip_x, l_hip_y, r_hip_x, r_hip_y,
                                            l_elb_x, l_elb_y, r_elb_x, r_elb_y,
                                            l_wri_x, l_wri_y, r_wri_x, r_wri_y
                                        ]
                                        
                                    
                        from dataset.tools.Benchpress_tool.predict import extract_raw_features, remove_outliers_and_interpolate, variation_normalize, variation_acceleration_normalize, variation_ratio_normalize, z_score_normalize
                        from scipy.interpolate import interp1d
                        
                        # 1. Base 13 metrics matrix
                        df_raw = extract_raw_features(subject_path, bar_dict, rear_ske_dict, top_ske_dict, angle_dicts=angle_dicts)
                        if df_raw.empty or len(df_raw) < 5:
                            continue
                            
                        # Drop unwanted feature
                        if "bar_ratio" in df_raw.columns:
                            df_raw = df_raw.drop(columns=["bar_ratio"])
                            
                        feature_cols = df_raw.columns[1:] # Exclude 'frame'
                        
                        # 2. Data Augmentation (Placeholder)
                        df_raw[feature_cols] = apply_augmentation(df_raw[feature_cols])

                        for col in feature_cols:
                            df_raw[col] = remove_outliers_and_interpolate(df_raw[col].values)

                        orig_indices = np.linspace(0, 1, len(df_raw))
                        target_indices = np.linspace(0, 1, 100)
                        
                        rep_100 = np.zeros((100, 12)) 
                        for c_idx, col in enumerate(feature_cols):
                            f = interp1d(orig_indices, df_raw[col].values, kind='linear', fill_value='extrapolate')
                            rep_100[:, c_idx] = f(target_indices)

                        # Apply interleaved 4 normalizations
                        norm_48 = np.zeros((100, 48))
                        for c_idx in range(12):
                            col_data = rep_100[:, c_idx]
                            v1 = variation_normalize(col_data)
                            v2 = variation_acceleration_normalize(col_data)
                            vr = variation_ratio_normalize(col_data)
                            z = z_score_normalize(col_data)
                            
                            norm_48[:, c_idx*4 + 0] = v1
                            norm_48[:, c_idx*4 + 1] = v2
                            norm_48[:, c_idx*4 + 2] = vr
                            norm_48[:, c_idx*4 + 3] = z
                            
                        # Multi-label: [tilting_to_the_left, tilting_to_the_right, scapular_protraction, elbows_flaring]
                        label_vec = [0, 0, 0, 0]
                        if label_dir == "tilting_to_the_left":
                            label_vec[0] = 1
                        elif label_dir == "tilting_to_the_right":
                            label_vec[1] = 1
                        elif label_dir == "scapular_protraction":
                            label_vec[2] = 1
                        elif label_dir == "elbows_flaring":
                            label_vec[3] = 1
                            
                        import re
                        clip_val = int(re.search(r'\d+', os.path.basename(file)).group())
                        
                        sub_match = re.search(r'subject_?\d+', os.path.basename(subject_dir))
                        sub_name = sub_match.group() if sub_match else os.path.basename(subject_dir)
                        
                        data.append({
                            "subject": sub_name,
                            "clip": clip_val,
                            "features": str(norm_48.tolist()),
                            "label": str(label_vec)
                        })
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

if __name__ == "__main__":
    generate_csv("BenchpressDataset", "./data/benchpress_dataset.csv")