import json
import os
import re
import numpy as np
import pandas as pd

gt_path = r"C:\squat\squat_dataset2_0706\S83_S108.json"
dataset_dir = r"C:\squat\squat_dataset2_0706"
pred_json_path = os.path.join(dataset_dir, "segments.json")

# Import helpers from eval_segments
from eval_segments import extract_recording_id, parse_rep_list, parse_rep_frames, load_all_predictions

with open(gt_path, 'r', encoding='utf-8') as f:
    gt_raw = json.load(f)

if isinstance(gt_raw, dict):
    for k in ['data', 'results', 'annotations', 'records', 'items']:
        if k in gt_raw and isinstance(gt_raw[k], (dict, list)):
            gt_raw = gt_raw[k]
            break

pred_full, pred_rec = load_all_predictions(dataset_dir)

EXCLUDE_RECORDINGS = [
    "recording_20260512_112434",  # S87
    "recording_20260514_162045",  # S90
    "recording_20260521_140622",  # S94
    "recording_20260601_104441",  # S103
    "recording_20260611_153424",  # S108
    "recording_20260611_153127",  # S108
    "recording_20260511_133807",  # S084
]

print(f"Total raw items in GT: {len(gt_raw)}")

# Check structure of each item in GT
all_errors = set()
recording_error_map = {}

# Analyze annotations
for item in gt_raw:
    rec_id = item.get('recording') or item.get('video') or item.get('file') or item.get('name')
    full_k, rec_name = extract_recording_id(str(rec_id))
    
    # Let's find errors
    annos = item.get('annotations', [])
    errors_found = []
    clips = []
    if isinstance(annos, list) and len(annos) > 0:
        for anno in annos:
            res = anno.get('result', {})
            errs = res.get('errors', [])
            if errs:
                for e in errs:
                    errors_found.append(e)
            if 'clips' in res:
                clips = res['clips']
    elif isinstance(item, dict) and 'result' in item:
        res = item['result']
        errs = res.get('errors', [])
        for e in errs:
            errors_found.append(e)
        if 'clips' in res:
            clips = res['clips']

    for e in errors_found:
        all_errors.add(e)

print(f"Unique error labels found in GT: {all_errors}")
