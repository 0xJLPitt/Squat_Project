import json
import os
import glob

def check_json(filepath):
    if os.path.exists(filepath):
        print(f"=== {filepath} exists ===")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            print(f"Keys ({len(data)}): {list(data.keys())[:10]}")
            sample_key = list(data.keys())[0]
            print(f"Sample item [{sample_key}]: {data[sample_key]}")
        elif isinstance(data, list):
            print(f"Length: {len(data)}")
            print(f"Sample item: {data[:2]}")
        return data
    else:
        print(f"=== {filepath} DOES NOT EXIST ===")
        return None

print("1. Checking Ground Truth: C:\\squat\\squat_dataset2_0706\\S83_S108.json")
gt_data = check_json(r"C:\squat\squat_dataset2_0706\S83_S108.json")

print("\n2. Checking Global Pred Segments: C:\\squat\\squat_dataset2_0706\\segments.json")
pred_data = check_json(r"C:\squat\squat_dataset2_0706\segments.json")

rec_segments = glob.glob(r"C:\squat\squat_dataset2_0706\**\segments.json", recursive=True)
print(f"\n3. Found {len(rec_segments)} subfolder segments.json files.")
