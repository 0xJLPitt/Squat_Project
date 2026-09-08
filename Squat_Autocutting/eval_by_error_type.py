import os
import sys
import json
import glob
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#將資料依照 6 大深蹲動作分類
# （正常、下蹲深度不足、骨盆後傾、髖部上升過快、下蹲時髖部主導、下蹲時膝蓋主導），
# 分別統計各類別下的深蹲次數、起訖點平均相差幀數與標準差，產出比較圖表
def extract_recording_id(path_str):
    """從路徑字串中提取識別碼"""
    cleaned = str(path_str).replace('\\', '/').strip('/')
    cleaned_lower = cleaned.lower()
    
    match = re.search(r'(s\d+)[/_](session\d+)[/_](recording_\d+_\d+)', cleaned_lower)
    if match:
        full_key = f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
        rec_name = match.group(3)
        return full_key, rec_name
        
    match_rec = re.search(r'(recording_\d+_\d+)', cleaned_lower)
    if match_rec:
        return match_rec.group(1), match_rec.group(1)
        
    parts = [p for p in cleaned_lower.split('/') if p]
    if len(parts) >= 3:
        return "/".join(parts[-3:]), parts[-1]
    elif len(parts) >= 1:
        return parts[-1], parts[-1]
        
    return cleaned_lower, cleaned_lower

def parse_rep_list(reps_obj):
    """解開 GT 與 Pred 內部深蹲片段"""
    if not reps_obj:
        return []

    if isinstance(reps_obj, list) and len(reps_obj) > 0 and isinstance(reps_obj[0], dict):
        item0 = reps_obj[0]
        if 'result' in item0 and isinstance(item0['result'], dict) and 'clips' in item0['result']:
            return item0['result']['clips']
        if 'clips' in item0 and isinstance(item0['clips'], list):
            return item0['clips']

    if isinstance(reps_obj, dict):
        if 'result' in reps_obj and isinstance(reps_obj['result'], dict) and 'clips' in reps_obj['result']:
            return reps_obj['result']['clips']
        if 'clips' in reps_obj and isinstance(reps_obj['clips'], list):
            return reps_obj['clips']
        for key in ['reps', 'segments', 'repetitions', 'annotations', 'events', 'labels', 'data', 'reps_with_id', 'items', 'clips']:
            if key in reps_obj and isinstance(reps_obj[key], list):
                return reps_obj[key]
                
        keys = list(reps_obj.keys())
        try:
            sorted_keys = sorted(keys, key=lambda k: int(re.search(r'\d+', str(k)).group()) if re.search(r'\d+', str(k)) else k)
            return [reps_obj[k] for k in sorted_keys]
        except Exception:
            return list(reps_obj.values())

    if isinstance(reps_obj, list):
        return reps_obj
        
    return []

def parse_rep_frames(rep_item):
    """解析單一下深蹲的 start 與 end frame"""
    start_val = None
    end_val = None
    
    if isinstance(rep_item, dict):
        for k in ['start_frame', 'start', 'start_idx', 'startFrame', 'begin', 'onset', 'start_time', 's', 'start_pos', 'from']:
            if k in rep_item and rep_item[k] is not None:
                start_val = rep_item[k]
                break
        for k in ['end_frame', 'end', 'end_idx', 'endFrame', 'stop', 'offset', 'end_time', 'e', 'end_pos', 'to']:
            if k in rep_item and rep_item[k] is not None:
                end_val = rep_item[k]
                break
                
        if start_val is None or end_val is None:
            for k in ['segment', 'range', 'interval', 'frames']:
                if k in rep_item and isinstance(rep_item[k], (list, tuple)) and len(rep_item[k]) >= 2:
                    start_val = rep_item[k][0]
                    end_val = rep_item[k][-1]
                    break

    elif isinstance(rep_item, (list, tuple)):
        if len(rep_item) >= 2:
            start_val = rep_item[0]
            end_val = rep_item[-1]
            
    if start_val is not None and end_val is not None:
        try:
            return int(start_val), int(end_val)
        except (ValueError, TypeError):
            return None, None
            
    return None, None

def load_gt_records_with_errors(json_path):
    """
    讀取 GT JSON，同時提取每個 recording 的 clips 以及 errors 標籤
    """
    if not os.path.exists(json_path):
        print(f"[ERROR] 找不到 Ground Truth 檔案: {json_path}")
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        for wrapper_key in ['data', 'results', 'annotations', 'records', 'items']:
            if wrapper_key in data and isinstance(data[wrapper_key], (dict, list)):
                data = data[wrapper_key]
                break

    records = {}

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            rec_id = (item.get('recording') or item.get('video') or item.get('file') or 
                      item.get('name') or item.get('id') or item.get('file_name') or item.get('path'))
            if not rec_id:
                continue

            full_k, rec_name = extract_recording_id(str(rec_id))

            # 抓取 errors 與 clips
            errors_list = []
            clips_list = []

            # 情況 1: annotations 結構
            annos = item.get('annotations', [])
            if isinstance(annos, list) and len(annos) > 0:
                for anno in annos:
                    res = anno.get('result', {}) if isinstance(anno, dict) else {}
                    errs = res.get('errors', [])
                    if isinstance(errs, list):
                        errors_list.extend(errs)
                    if 'clips' in res:
                        clips_list = res['clips']
            
            # 情況 2: 直接在 item['result']
            if 'result' in item and isinstance(item['result'], dict):
                res = item['result']
                errs = res.get('errors', [])
                if isinstance(errs, list):
                    errors_list.extend(errs)
                if 'clips' in res:
                    clips_list = res['clips']

            # 情況 3: segments/reps 直接在頂層
            if not clips_list:
                clips_list = (item.get('segments') or item.get('reps') or 
                              item.get('repetitions') or item.get('clips') or [])

            parsed_clips = parse_rep_list(clips_list)

            # 去除重複 errors
            unique_errors = list(dict.fromkeys(errors_list))

            record_info = {
                'full_key': full_k,
                'rec_name': rec_name,
                'raw_errors': unique_errors,
                'clips': parsed_clips
            }
            records[rec_name] = record_info
            records[full_k] = record_info

    elif isinstance(data, dict):
        for k, val in data.items():
            full_k, rec_name = extract_recording_id(k)
            parsed_clips = parse_rep_list(val)
            record_info = {
                'full_key': full_k,
                'rec_name': rec_name,
                'raw_errors': [],
                'clips': parsed_clips
            }
            records[rec_name] = record_info
            records[full_k] = record_info

    return records

def load_all_predictions(dataset_dir):
    """讀取預測標籤 segments.json"""
    global_json = os.path.join(dataset_dir, "segments.json")
    pred_full = {}
    pred_rec = {}
    
    if os.path.exists(global_json):
        print(f"[INFO] 讀取預測 JSON: {global_json}")
        with open(global_json, 'r', encoding='utf-8') as f:
            g_data = json.load(f)
        if isinstance(g_data, dict):
            for k, v in g_data.items():
                fk, rn = extract_recording_id(k)
                parsed = parse_rep_list(v)
                pred_full[fk] = parsed
                pred_rec[rn] = parsed
        
    sub_segments = glob.glob(os.path.join(dataset_dir, "**", "segments.json"), recursive=True)
    for sub_p in sub_segments:
        if sub_p == global_json:
            continue
        rel_p = os.path.relpath(os.path.dirname(sub_p), dataset_dir)
        full_k, rec_name = extract_recording_id(rel_p)
        try:
            with open(sub_p, 'r', encoding='utf-8') as f:
                s_data = json.load(f)
            if isinstance(s_data, dict):
                for k, v in s_data.items():
                    fk, rn = extract_recording_id(k)
                    parsed = parse_rep_list(v)
                    pred_full[fk] = parsed
                    pred_rec[rn] = parsed
            elif isinstance(s_data, list):
                parsed = parse_rep_list(s_data)
                pred_full[full_k] = parsed
                pred_rec[rec_name] = parsed
        except Exception:
            pass
            
    return pred_full, pred_rec

# 定義 6 個目標標準動作類別及其對應的正則/關鍵字
STANDARD_ACTIONS = {
    "正常": [r"正常", r"normal"],
    "下蹲深度不足": [r"下蹲深度不足", r"深度不足", r"insufficient depth"],
    "骨盆後傾": [r"骨盆後傾", r"屁股眨眼", r"butt wink"],
    "髖部上升過快(起身時髖部先啟動)": [r"髖部上升過快", r"臀部上升過快", r"起身時髖部先啟動", r"good morning"],
    "下蹲時髖部主導": [r"下蹲時髖部主導", r"下蹲時髖部過度主導", r"髖部主導", r"髖部過度主導", r"hip dominant"],
    "下蹲時膝蓋主導": [r"下蹲時膝蓋主導", r"下蹲時膝蓋過度主導", r"膝蓋主導", r"膝蓋過度主導", r"knee dominant"]
}

def map_errors_to_standard_categories(raw_errors):
    """將標註中的原始錯誤文字映射至 6 大標準類別"""
    matched_cats = set()
    for raw_err in raw_errors:
        raw_str = str(raw_err).strip()
        matched = False
        for cat_name, patterns in STANDARD_ACTIONS.items():
            for p in patterns:
                if re.search(p, raw_str, re.IGNORECASE):
                    matched_cats.add(cat_name)
                    matched = True
                    break
        if not matched and raw_str:
            # 其它未歸類錯誤
            matched_cats.add(raw_str)
    return list(matched_cats)

def evaluate_by_action_types(gt_records, pred_full, pred_rec, fps=30, exclude_list=None):
    """
    分別針對 6 種動作類別計算切割演算法與 Ground Truth 的平均差異與各動作總下數
    """
    if exclude_list is None:
        exclude_list = [
            "recording_20260512_112434",  # S87
            "recording_20260514_162045",  # S90
            "recording_20260521_140622",  # S94
            "recording_20260601_104441",  # S103
            "recording_20260611_153424",  # S108
            "recording_20260611_153127",  # S108
            "recording_20260511_133807",  # S084
        ]

    # 去重 GT 錄影 (避免 rec_name 與 full_key 重複計算)
    unique_gt = {}
    for k, v in gt_records.items():
        rec_name = v['rec_name']
        if rec_name not in unique_gt:
            unique_gt[rec_name] = v

    # 比對 GT 與 Pred
    matched_recordings = []
    for rec_name, gt_info in unique_gt.items():
        if any(ex in rec_name for ex in exclude_list):
            continue

        pred_reps = None
        if gt_info['full_key'] in pred_full:
            pred_reps = pred_full[gt_info['full_key']]
        elif rec_name in pred_rec:
            pred_reps = pred_rec[rec_name]

        if pred_reps is not None:
            matched_recordings.append({
                'rec_name': rec_name,
                'full_key': gt_info['full_key'],
                'raw_errors': gt_info['raw_errors'],
                'categories': map_errors_to_standard_categories(gt_info['raw_errors']),
                'gt_clips': gt_info['clips'],
                'pred_clips': pred_reps
            })

    print(f"[INFO] 成功配對 {len(matched_recordings)} 筆錄影進行動作分類評估 (已過濾排除清單)")

    # 收集每次深蹲 (rep) 的詳細比對資料
    detailed_reps = []
    for item in matched_recordings:
        gt_clips = item['gt_clips']
        pred_clips = item['pred_clips']
        categories = item['categories']
        rec_name = item['rec_name']

        max_reps = min(len(gt_clips), len(pred_clips), 10)
        for i in range(max_reps):
            gt_s, gt_e = parse_rep_frames(gt_clips[i])
            pred_s, pred_e = parse_rep_frames(pred_clips[i])

            if gt_s is None or pred_s is None:
                continue

            s_diff = abs(pred_s - gt_s)
            e_diff = abs(pred_e - gt_e)
            overall_diff = (s_diff + e_diff) / 2.0

            rep_data = {
                'recording': rec_name,
                'rep_id': i + 1,
                'categories': categories,
                'gt_start': gt_s,
                'pred_start': pred_s,
                'start_diff_abs': s_diff,
                'start_diff_sec': s_diff / fps,
                'gt_end': gt_e,
                'pred_end': pred_e,
                'end_diff_abs': e_diff,
                'end_diff_sec': e_diff / fps,
                'mean_frame_diff': overall_diff,
                'mean_sec_diff': overall_diff / fps
            }
            detailed_reps.append(rep_data)

    df_all_reps = pd.DataFrame(detailed_reps)

    # 針對 6 大標準動作類別進行分類彙整
    summary_list = []
    for target_cat in STANDARD_ACTIONS.keys():
        # 找出包含此類別的錄影與深蹲次數
        cat_recordings = [m for m in matched_recordings if target_cat in m['categories']]
        
        # 找出屬於該類別的 reps
        cat_reps = []
        for r in detailed_reps:
            if target_cat in r['categories']:
                cat_reps.append(r)
        
        df_cat = pd.DataFrame(cat_reps)
        num_recs = len(cat_recordings)
        num_reps = len(df_cat)

        if num_reps > 0:
            s_mae = df_cat['start_diff_abs'].mean()
            s_std = df_cat['start_diff_abs'].std()
            s_sec = df_cat['start_diff_sec'].mean()
            
            e_mae = df_cat['end_diff_abs'].mean()
            e_std = df_cat['end_diff_abs'].std()
            e_sec = df_cat['end_diff_sec'].mean()

            total_mae = (s_mae + e_mae) / 2.0
            total_sec = (s_sec + e_sec) / 2.0
        else:
            s_mae = s_std = s_sec = e_mae = e_std = e_sec = total_mae = total_sec = 0.0

        summary_list.append({
            '動作類別 (Action Category)': target_cat,
            '錄影數量 (Recordings)': num_recs,
            '評估深蹲總下數 (Evaluated Reps)': num_reps,
            'Start Frame 平均誤差 (幀)': round(s_mae, 2),
            'Start Frame 誤差 (秒)': round(s_sec, 3),
            'Start 誤差標準差 (幀)': round(s_std, 2),
            'End Frame 平均誤差 (幀)': round(e_mae, 2),
            'End Frame 誤差 (秒)': round(e_sec, 3),
            'End 誤差標準差 (幀)': round(e_std, 2),
            '起訖點平均相差 (幀)': round(total_mae, 2),
            '起訖點平均相差 (秒)': round(total_sec, 3)
        })

    df_summary = pd.DataFrame(summary_list)
    return df_summary, df_all_reps, matched_recordings

def plot_error_comparison_chart(df_summary, output_fig_path):
    """繪製 6 種動作類別的誤差比較長條圖 (包含各動作深蹲下數與影片數標註)"""
    plt.figure(figsize=(14, 7), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 支援繁體中文字體
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # X 軸標籤：包含動作名稱與總評估下數 (去除影片支數)
    labels = []
    for _, row in df_summary.iterrows():
        cat_name = row['動作類別 (Action Category)'].replace('(', '\n(')
        reps = row['評估深蹲總下數 (Evaluated Reps)']
        labels.append(f"{cat_name}\n({reps} 下)")

    x = np.arange(len(labels))
    width = 0.35

    bars_start = plt.bar(x - width/2, df_summary['Start Frame 平均誤差 (幀)'], width, label='Start Frame MAE (幀)', color='#e74c3c', alpha=0.85)
    bars_end = plt.bar(x + width/2, df_summary['End Frame 平均誤差 (幀)'], width, label='End Frame MAE (幀)', color='#3498db', alpha=0.85)

    # 柱狀圖上方標註數值 (幀數)
    for i, bar in enumerate(bars_start):
        val = df_summary['Start Frame 平均誤差 (幀)'].iloc[i]
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.12, f"{val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c0392b')

    for i, bar in enumerate(bars_end):
        val = df_summary['End Frame 平均誤差 (幀)'].iloc[i]
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.12, f"{val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2980b9')

    # 設定 Y 軸上限留出空間
    max_val = max(df_summary['Start Frame 平均誤差 (幀)'].max(), df_summary['End Frame 平均誤差 (幀)'].max())
    plt.ylim(0, max_val * 1.25)

    plt.xlabel('動作類別 (Squat Movement / Error Category) [附帶：總下數]', fontsize=12, fontweight='bold', labelpad=12)
    plt.ylabel('平均相差幀數 (Frames, at 30fps)', fontsize=12, fontweight='bold')
    plt.title('深蹲自動切割演算法 vs. Ground Truth 各動作類別誤差與下數比較 (S83 ~ S108)', fontsize=14, fontweight='bold', pad=15)
    
    plt.xticks(x, labels, fontsize=10.5, fontweight='medium')
    plt.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    try:
        plt.savefig(output_fig_path)
        print(f"\n[SUCCESS] 已產出各動作類別誤差與下數比較圖表: {output_fig_path}")
    except (OSError, PermissionError) as e:
        print(f"\n[WARNING] 原圖檔被檢視器鎖定無法覆寫 ({e})")
        fallback_path = output_fig_path.replace(".png", "_latest.png")
        try:
            plt.savefig(fallback_path)
            print(f"👉 已自動為您儲存至最新備用檔名: {fallback_path}")
        except Exception as ex:
            print(f"[ERROR] 儲存備用圖片時亦發生錯誤: {ex}")
    finally:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="分動作類別評估深蹲切片演算法誤差與下數統計")
    parser.add_argument("--dataset", type=str, default=r"C:\squat\squat_dataset2_0706", help="資料集路徑")
    parser.add_argument("--gt", type=str, default=r"C:\squat\squat_dataset2_0706\S83_S108.json", help="Ground Truth JSON 路徑")
    parser.add_argument("--fps", type=int, default=30, help="影片幀率")
    parser.add_argument("--out_dir", type=str, default=r"C:\squat\squat_dataset2_0706", help="輸出報告資料夾")
    args = parser.parse_args()

    print("=" * 80)
    print(" S83~S108 各動作類別 (正常/錯誤) 切割演算法誤差與下數統計評估")
    print(f" GT JSON : {args.gt}")
    print(f" Dataset : {args.dataset}")
    print("=" * 80)

    gt_records = load_gt_records_with_errors(args.gt)
    pred_full, pred_rec = load_all_predictions(args.dataset)

    if not gt_records:
        print("[ERROR] 無法讀取 Ground Truth 資料。")
        sys.exit(1)

    df_summary, df_details, matched_recs = evaluate_by_action_types(gt_records, pred_full, pred_rec, fps=args.fps)

    print("\n" + "=" * 80)
    print(" 📊 各動作類別評估結果彙整表 (Action Category Evaluation Summary)")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    # 輸出 CSV
    csv_summary_path = os.path.join(args.out_dir, "eval_error_types_summary.csv")
    csv_details_path = os.path.join(args.out_dir, "eval_error_types_details.csv")
    df_summary.to_csv(csv_summary_path, index=False, encoding='utf-8-sig')
    df_details.to_csv(csv_details_path, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] 彙整統計已儲存至: {csv_summary_path}")
    print(f"[INFO] 逐下詳細數據已儲存至: {csv_details_path}")

    # 輸出圖表
    chart_path = os.path.join(args.out_dir, "eval_error_types_comparison.png")
    plot_error_comparison_chart(df_summary, chart_path)

if __name__ == "__main__":
    main()
