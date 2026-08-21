import json
import os

gt_path = r"C:\squat\squat_dataset2_0706\S83_S108.json"
pred_path = r"C:\squat\squat_dataset2_0706\segments.json"
out_path = r"C:\squat\Squat_Project\keys_info.txt"

with open(out_path, 'w', encoding='utf-8') as out:
    out.write("=== GT JSON KEYS ===\n")
    if os.path.exists(gt_path):
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        if isinstance(gt_data, dict):
            out.write(f"Total GT keys: {len(gt_data)}\n")
            for k in list(gt_data.keys())[:15]:
                out.write(f"  GT Key: {k}\n")
        elif isinstance(gt_data, list):
            out.write(f"GT is a list of length: {len(gt_data)}\n")
            if gt_data:
                out.write(f"Sample item: {gt_data[0]}\n")
    else:
        out.write(f"GT file does not exist: {gt_path}\n")

    out.write("\n=== PRED JSON KEYS ===\n")
    if os.path.exists(pred_path):
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
        if isinstance(pred_data, dict):
            out.write(f"Total Pred keys: {len(pred_data)}\n")
            for k in list(pred_data.keys())[:15]:
                out.write(f"  Pred Key: {k}\n")
        elif isinstance(pred_data, list):
            out.write(f"Pred is a list of length: {len(pred_data)}\n")
            if pred_data:
                out.write(f"Sample item: {pred_data[0]}\n")
    else:
        out.write(f"Pred file does not exist: {pred_path}\n")

print("Wrote keys info to keys_info.txt")
