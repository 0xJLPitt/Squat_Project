import json
import os

gt_path = r"D:\squat_dataset\S01_S108.json"
out_path = r"C:\squat\Squat_Project\Squat_Autocutting\gt_data_structure.txt"

with open(gt_path, 'r', encoding='utf-8') as f:
    gt_raw = json.load(f)

data_obj = gt_raw.get('data', {})

with open(out_path, 'w', encoding='utf-8') as out:
    out.write(f"Type of 'data': {type(data_obj)}\n")
    if isinstance(data_obj, dict):
        out.write(f"Total keys in 'data': {len(data_obj)}\n")
        keys = list(data_obj.keys())
        out.write("First 10 keys in 'data':\n")
        for k in keys[:10]:
            out.write(f"  Key: {k}\n")
            out.write(f"  Sample item: {data_obj[k]}\n\n")
    elif isinstance(data_obj, list):
        out.write(f"Total items in 'data' list: {len(data_obj)}\n")
        if data_obj:
            out.write(f"Sample item 0: {data_obj[0]}\n")
            out.write(f"Sample item 1: {data_obj[1]}\n")

print("Done writing gt_data_structure.txt")
