import os
import sys
import json
import glob
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#用來計算深蹲切割演算法整體效能評估
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

def load_json_records(json_path):
    """讀取 JSON 並整理字典"""
    if not os.path.exists(json_path):
        print(f"[ERROR] 找不到 JSON 檔案: {json_path}")
        return {}, {}
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if isinstance(data, dict):
        for wrapper_key in ['data', 'results', 'annotations', 'records', 'items']:
            if wrapper_key in data and isinstance(data[wrapper_key], (dict, list)):
                print(f"[INFO] 自動解開 '{os.path.basename(json_path)}' 的頂層包裝 Key: '{wrapper_key}'")
                data = data[wrapper_key]
                break

    full_dict = {}
    rec_dict = {}
    
    if isinstance(data, dict):
        for k, reps in data.items():
            full_k, rec_name = extract_recording_id(k)
            parsed_reps = parse_rep_list(reps)
            full_dict[full_k] = parsed_reps
            rec_dict[rec_name] = parsed_reps
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                rec_id = (item.get('recording') or item.get('video') or item.get('file') or 
                          item.get('name') or item.get('id') or item.get('file_name') or item.get('path'))
                reps = (item.get('segments') or item.get('reps') or item.get('repetitions') or 
                        item.get('labels') or item.get('annotations') or item.get('results') or item)
                
                if rec_id:
                    full_k, rec_name = extract_recording_id(str(rec_id))
                    parsed_reps = parse_rep_list(reps)
                    full_dict[full_k] = parsed_reps
                    rec_dict[rec_name] = parsed_reps

    return full_dict, rec_dict

def load_all_predictions(dataset_dir):
    """讀取預測標籤"""
    global_json = os.path.join(dataset_dir, "segments.json")
    pred_full = {}
    pred_rec = {}
    
    if os.path.exists(global_json):
        print(f"[INFO] 讀取預測 JSON: {global_json}")
        f_dict, r_dict = load_json_records(global_json)
        pred_full.update(f_dict)
        pred_rec.update(r_dict)
        
    sub_segments = glob.glob(os.path.join(dataset_dir, "**", "segments.json"), recursive=True)
    for sub_p in sub_segments:
        if sub_p == global_json:
            continue
        rel_p = os.path.relpath(os.path.dirname(sub_p), dataset_dir)
        full_k, rec_name = extract_recording_id(rel_p)
        try:
            f_dict, r_dict = load_json_records(sub_p)
            if f_dict or r_dict:
                pred_full.update(f_dict)
                pred_rec.update(r_dict)
        except Exception:
            pass
            
    return pred_full, pred_rec

def evaluate_segmentation(gt_full, gt_rec, pred_full, pred_rec, fps=30):
    """計算每個錄影 S83~S108 包含的 10 次 start_frame 平均與 10 次 end_frame 平均相差"""
    detailed_records = []
    recording_summaries = []
    matched_pairs = []
    
    all_gt_keys = sorted(list(set(gt_full.keys())))
    for fk in all_gt_keys:
        if fk in pred_full:
            matched_pairs.append((fk, gt_full[fk], pred_full[fk]))
            
    if len(matched_pairs) < len(gt_rec):
        already_matched_rec = {p[0].split('/')[-1] for p in matched_pairs}
        for r_name, gt_reps in sorted(gt_rec.items()):
            if r_name in pred_rec and r_name not in already_matched_rec:
                matched_pairs.append((r_name, gt_reps, pred_rec[r_name]))

    EXCLUDE_RECORDINGS = [
        "recording_20260512_112434",  # S87
        "recording_20260514_162045",  # S90
        "recording_20260521_140622",  # S94
        "recording_20260601_104441",  # S103
        "recording_20260611_153424",  # S108
        "recording_20260611_153127",  # S108
        "recording_20260511_133807",  # S084
    ]

    # 過濾排除指定之錄影
    filtered_pairs = []
    for pair in matched_pairs:
        rec_name = pair[0]
        if any(ex in rec_name for ex in EXCLUDE_RECORDINGS):
            print(f"[EXCLUDE] 評估中排除: {rec_name}")
        else:
            filtered_pairs.append(pair)
            
    matched_pairs = filtered_pairs
    print(f"[INFO] 成功匹配並保留 {len(matched_pairs)} 個錄影片段進行評估 (已排除 5 個指定錄影)")
    
    for display_name, gt_reps, pred_reps in matched_pairs:
        gt_reps_list = parse_rep_list(gt_reps)
        pred_reps_list = parse_rep_list(pred_reps)
        
        start_diffs = []
        end_diffs = []
        
        max_reps = min(len(gt_reps_list), len(pred_reps_list), 10)
        
        for i in range(max_reps):
            gt_start, gt_end = parse_rep_frames(gt_reps_list[i])
            pred_start, pred_end = parse_rep_frames(pred_reps_list[i])
            
            if gt_start is None or pred_start is None:
                continue
                
            start_diff = abs(pred_start - gt_start)
            end_diff = abs(pred_end - gt_end)
            start_signed = pred_start - gt_start
            end_signed = pred_end - gt_end
            
            start_diffs.append(start_diff)
            end_diffs.append(end_diff)
            
            detailed_records.append({
                'recording': display_name,
                'rep_id': i + 1,
                'gt_start': gt_start,
                'pred_start': pred_start,
                'start_diff_abs': start_diff,
                'start_diff_signed': start_signed,
                'start_diff_sec': start_diff / fps,
                'gt_end': gt_end,
                'pred_end': pred_end,
                'end_diff_abs': end_diff,
                'end_diff_signed': end_signed,
                'end_diff_sec': end_diff / fps,
            })
            
        rep_count_diff = abs(len(gt_reps_list) - len(pred_reps_list))
        if start_diffs and end_diffs:
            recording_summaries.append({
                'recording': display_name,
                'gt_rep_count': len(gt_reps_list),
                'pred_rep_count': len(pred_reps_list),
                'rep_count_diff': rep_count_diff,
                'num_reps_evaluated': len(start_diffs),
                'mean_start_diff_abs': np.mean(start_diffs),
                'mean_start_diff_sec': np.mean(start_diffs) / fps,
                'std_start_diff': np.std(start_diffs),
                'mean_end_diff_abs': np.mean(end_diffs),
                'mean_end_diff_sec': np.mean(end_diffs) / fps,
                'std_end_diff': np.std(end_diffs)
            })
            
    df_details = pd.DataFrame(detailed_records)
    df_summary = pd.DataFrame(recording_summaries)
    
    return df_details, df_summary

def plot_single_chart_all_recordings(df_summary, output_fig_path):
    """
    將 S83 到 S108 所有錄影 (218 個錄影片段) 獨立計算出來的 10 個 start_frame 相差平均
    與 10 個 end_frame 相差平均，呈現為一張寬幅清晰圖表。
    """
    if df_summary.empty:
        print("[WARNING] 無可用數據繪製圖表")
        return
        
    num_recs = len(df_summary)
    
    # 根據錄影筆數動態決定圖表寬度，確保所有 218 個錄影都很清晰
    fig_width = max(24, int(num_recs * 0.25))
    plt.figure(figsize=(fig_width, 10), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    x = np.arange(num_recs)
    width = 0.4
    
    # 繪製柱狀圖
    bars_start = plt.bar(x - width/2, df_summary['mean_start_diff_abs'], width, label='10-Rep Mean Start Frame Diff (Frames)', color='#e74c3c', alpha=0.85)
    bars_end = plt.bar(x + width/2, df_summary['mean_end_diff_abs'], width, label='10-Rep Mean End Frame Diff (Frames)', color='#3498db', alpha=0.85)
    
    # 繪製全域平均基準線
    global_start_mean = df_summary['mean_start_diff_abs'].mean()
    global_end_mean = df_summary['mean_end_diff_abs'].mean()
    
    plt.axhline(global_start_mean, color='#c0392b', linestyle='--', linewidth=2, label=f'Overall Start Mean ({global_start_mean:.2f} frames)')
    plt.axhline(global_end_mean, color='#2980b9', linestyle='--', linewidth=2, label=f'Overall End Mean ({global_end_mean:.2f} frames)')
    
    # 設定標籤與標題
    plt.xlabel('Recording Session (S83 ~ S108)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Mean Frame Difference (Frames)', fontsize=12, fontweight='bold')
    plt.title(f'Squat Autocutting Performance: 10-Rep Start & End Frame Mean Error across All {num_recs} Recordings (S83 ~ S108)', fontsize=14, fontweight='bold', pad=15)
    
    # 設定 X 軸刻度標籤
    labels = df_summary['recording'].tolist()
    plt.xticks(x, labels, rotation=90, ha='center', fontsize=6)
    
    plt.legend(loc='upper right', fontsize=11, frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    try:
        plt.savefig(output_fig_path)
        print(f"\n[SUCCESS] 已為您產出涵蓋 S83~S108 所有 {num_recs} 個錄影的獨立平均相差圖表:")
        print(f"👉 {output_fig_path}")
    except (OSError, PermissionError) as e:
        print(f"\n[WARNING] 無法寫入至 {output_fig_path} ({e})")
        print("💡 原因：該圖檔目前可能正被 Windows 相片檢視器、圖檔預覽器或其他軟體開啟中，導致檔案被鎖定無法覆寫。")
        fallback_path = output_fig_path.replace(".png", "_new.png")
        try:
            plt.savefig(fallback_path)
            print(f"👉 已為您自動儲存至備用檔名: {fallback_path}")
        except Exception as ex:
            print(f"[ERROR] 儲存備用圖片時亦發生錯誤: {ex}")
    finally:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Squat Segmentation Evaluation Tool")
    parser.add_argument("--dataset", type=str, default=r"C:\squat\squat_dataset2_0706", help="資料集路徑")
    parser.add_argument("--gt", type=str, default=r"C:\squat\squat_dataset2_0706\S83_S108.json", help="Ground Truth JSON 路徑")
    parser.add_argument("--fps", type=int, default=30, help="影片 Frame Rate")
    args = parser.parse_args()

    print("=" * 70)
    print(" 深蹲切割演算法效能評估與圖表繪製")
    print(f" Dataset Path: {args.dataset}")
    print(f" GT JSON Path: {args.gt}")
    print("=" * 70)

    gt_full, gt_rec = load_json_records(args.gt)
    pred_full, pred_rec = load_all_predictions(args.dataset)

    if not gt_full and not gt_rec:
        print("[ERROR] 讀取 Ground Truth 失敗，請確認檔案路徑是否正確。")
        sys.exit(1)
        
    df_details, df_summary = evaluate_segmentation(gt_full, gt_rec, pred_full, pred_rec, fps=args.fps)

    if df_details.empty or df_summary.empty:
        print("\n[WARNING] 比對結果為空。請確認資料結構。")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" 總體評估結果摘要 (Overall Summary)")
    print("=" * 70)
    print(f"評估錄影筆數 (Recordings): {len(df_summary)}")
    print(f"評估深蹲總次數 (Total Reps): {len(df_details)}")
    print(f"Start Frame 10-Rep 平均誤差 (MAE): {df_details['start_diff_abs'].mean():.2f} 幀 ({df_details['start_diff_sec'].mean():.3f} 秒)")
    print(f"Start Frame 誤差標準差 (Std Dev): {df_details['start_diff_abs'].std():.2f} 幀")
    print(f"End Frame 10-Rep 平均誤差 (MAE):   {df_details['end_diff_abs'].mean():.2f} 幀 ({df_details['end_diff_sec'].mean():.3f} 秒)")
    print(f"End Frame 誤差標準差 (Std Dev):   {df_details['end_diff_abs'].std():.2f} 幀")

    # 下數準確率統計
    print("\n" + "=" * 70)
    print(" 下數計算準確率 (Rep Count Accuracy)")
    print("=" * 70)
    exact_match = (df_summary['rep_count_diff'] == 0).sum()
    total_recs = len(df_summary)
    mismatch_df = df_summary[df_summary['rep_count_diff'] != 0][['recording', 'gt_rep_count', 'pred_rep_count', 'rep_count_diff']]
    print(f"下數完全正確的錄影數: {exact_match} / {total_recs}  ({exact_match / total_recs * 100:.1f}%)")
    print(f"平均下數差異: {df_summary['rep_count_diff'].mean():.3f} 下")
    if not mismatch_df.empty:
        print(f"\n以下 {len(mismatch_df)} 筆錄影下數不一致:")
        print(mismatch_df.to_string(index=False))

    print("\n[Rep 1 ~ Rep 10 逐次平均誤差 (Per Rep Average)]:")
    rep_table = df_details.groupby('rep_id')[['start_diff_abs', 'end_diff_abs']].mean().reset_index()
    rep_table.columns = ['Rep ID', 'Start Frame Mean Error', 'End Frame Mean Error']
    print(rep_table.to_string(index=False))

    csv_details_path = os.path.join(args.dataset, "eval_details_per_rep.csv")
    csv_summary_path = os.path.join(args.dataset, "eval_summary_per_recording.csv")
    df_details.to_csv(csv_details_path, index=False, encoding='utf-8-sig')
    df_summary.to_csv(csv_summary_path, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] 詳細 CSV 已匯出: {csv_details_path}")
    print(f"[INFO] 各錄影平均相差 CSV 已匯出: {csv_summary_path}")

    # 專門產出使用者指定的【單一張】包含 S83~S108 所有錄影獨立平均相差的圖表
    fig_path = os.path.join(args.dataset, "all_recordings_mean_differences.png")
    plot_single_chart_all_recordings(df_summary, fig_path)

if __name__ == "__main__":
    main()
