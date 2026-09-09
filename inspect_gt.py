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

print("1. Checking Ground Truth: D:\\squat_dataset\\S01_S108.json")
gt_data = check_json(r"D:\squat_dataset\S01_S108.json")

print("\n2. Checking Global Pred Segments: D:\\squat_dataset\\segments.json")
pred_data = check_json(r"D:\squat_dataset\segments.json")

rec_segments = glob.glob(r"D:\squat_dataset\**\segments.json", recursive=True)
print(f"\n3. Found {len(rec_segments)} subfolder segments.json files.")
