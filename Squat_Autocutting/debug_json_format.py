import json
import os
import re

gt_path = r"D:\squat_dataset\S01_S108.json"
pred_path = r"D:\squat_dataset\segments.json"
out_path = r"C:\squat\Squat_Project\Squat_Autocutting\debug_output.txt"

with open(out_path, "w", encoding="utf-8") as out:
    out.write("=== GT DATA (S01_S108.json) ===\n")
    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        out.write(f"Type: {type(gt_data)}\n")
        if isinstance(gt_data, dict):
            out.write(f"Count: {len(gt_data)}\n")
            gt_keys = list(gt_data.keys())
            out.write(f"First 10 GT Keys:\n")
            for k in gt_keys[:10]:
                out.write(f"  '{k}' -> type: {type(gt_data[k])}, sample: {gt_data[k][:1] if isinstance(gt_data[k], list) else gt_data[k]}\n")
        elif isinstance(gt_data, list):
            out.write(f"List count: {len(gt_data)}\n")
            for item in gt_data[:5]:
                out.write(f"  Item: {item}\n")
    else:
        out.write("GT path does not exist!\n")

    out.write("\n=== PRED DATA (segments.json) ===\n")
    if os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)
        out.write(f"Type: {type(pred_data)}\n")
        if isinstance(pred_data, dict):
            out.write(f"Count: {len(pred_data)}\n")
            pred_keys = list(pred_data.keys())
            out.write(f"First 10 Pred Keys:\n")
            for k in pred_keys[:10]:
                out.write(f"  '{k}' -> type: {type(pred_data[k])}, sample: {pred_data[k][:1] if isinstance(pred_data[k], list) else pred_data[k]}\n")
    else:
        out.write("Pred path does not exist!\n")

print("Finished writing debug_output.txt")
