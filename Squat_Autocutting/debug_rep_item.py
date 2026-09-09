import json
import os

gt_path = r"D:\squat_dataset\S01_S108.json"
pred_path = r"D:\squat_dataset\segments.json"
out_path = r"C:\squat\Squat_Project\Squat_Autocutting\rep_structure.txt"

with open(out_path, 'w', encoding='utf-8') as out:
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_raw = json.load(f)
    gt_data = gt_raw.get('data', gt_raw)
    
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
        
    out.write("=== GT RECORDING SAMPLE ===\n")
    if isinstance(gt_data, dict):
        first_key = list(gt_data.keys())[0]
        out.write(f"GT Key: {first_key}\n")
        out.write(f"GT Value type: {type(gt_data[first_key])}\n")
        out.write(f"GT Value sample: {gt_data[first_key]}\n")
    elif isinstance(gt_data, list) and gt_data:
        out.write(f"GT Item 0: {gt_data[0]}\n")

    out.write("\n=== PRED RECORDING SAMPLE ===\n")
    if isinstance(pred_data, dict):
        first_key = list(pred_data.keys())[0]
        out.write(f"Pred Key: {first_key}\n")
        out.write(f"Pred Value type: {type(pred_data[first_key])}\n")
        out.write(f"Pred Value sample: {pred_data[first_key]}\n")

print("Wrote rep structure to rep_structure.txt")
