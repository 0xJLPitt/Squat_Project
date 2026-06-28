import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import numpy as np
import os
###寫一個把三次內插整合在一起的
#第一次:把mediapipe裡面沒偵測到的先內插補植
#第二次:把槓端的沒偵測到的內插補植
#第三次:把mediapipe資料的長度對齊yolo的長度

###處理mediapipe資料的no detection
def interpolate_landmarks(input_file):
    df = pd.read_csv(input_file, header=None, names=["frame", "landmark", "x", "y"])
    df.replace('no detection', np.nan, inplace=True)
    df[['landmark', 'x', 'y']] = df[['landmark', 'x', 'y']].apply(pd.to_numeric, errors='coerce')

    # ✅ 用 pivot_table 避免重複 (frame, landmark) 報錯
    pivot_x = df.pivot_table(index='frame', columns='landmark', values='x', aggfunc='mean')
    pivot_y = df.pivot_table(index='frame', columns='landmark', values='y', aggfunc='mean')

    # 插值填補缺失
    pivot_x = pivot_x.interpolate(method='linear', limit_direction='both', axis=0)
    pivot_y = pivot_y.interpolate(method='linear', limit_direction='both', axis=0)

    # 回轉為長格式
    result_df = pivot_x.stack().reset_index()
    result_df.columns = ['frame', 'landmark', 'x']
    result_df['y'] = pivot_y.stack().reset_index(drop=True)

    return result_df[['frame', 'landmark', 'x', 'y']].to_numpy()

####處理yolo的no detection並且內插
def load_bar_data(filename):
    # 讀取 BAR 資料
    data = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                values = [float(x) for x in line.strip().split(',')]
                if len(values) == 5:  # YOLO 的資料應有 5 個欄位
                    data.append(values)
            except ValueError:
                pass
    return np.array(data)


def interpolate_missing_detections(yolo_data):
    frames = yolo_data[:, 0]
    x_center = yolo_data[:, 1]
    y_center = yolo_data[:, 2]
    widths = yolo_data[:, 3]
    heights = yolo_data[:, 4]

    # 找到有效檢測的幀索引
    valid_indices = np.where((x_center != -1) & (y_center != -1) & (widths != -1) & (heights != -1))[0]

    # 確保所有幀都有值
    all_frames = np.arange(int(frames.min()), int(frames.max()) + 1)

    # 使用有效檢測的幀進行插值
    interp_x_center = interp1d(frames[valid_indices], x_center[valid_indices], kind='linear', fill_value="extrapolate")
    interp_y_center = interp1d(frames[valid_indices], y_center[valid_indices], kind='linear', fill_value="extrapolate")
    interp_widths = interp1d(frames[valid_indices], widths[valid_indices], kind='linear', fill_value="extrapolate")
    interp_heights = interp1d(frames[valid_indices], heights[valid_indices], kind='linear', fill_value="extrapolate")

    # 對於所有幀進行插值
    interpolated_x_center = interp_x_center(all_frames)
    interpolated_y_center = interp_y_center(all_frames)
    interpolated_widths = interp_widths(all_frames)
    interpolated_heights = interp_heights(all_frames)

    # 補前面值的版本（假設要補 all_frames[0] 個平均值）
    padded_features = []
    for feature_array in [interpolated_x_center, interpolated_y_center, interpolated_widths, interpolated_heights]:
        avg = np.mean(feature_array)
        pad_len = all_frames[0]  # 補的長度
        padded_array = np.concatenate([[avg] * pad_len, feature_array])
        padded_features.append(padded_array)

    # 更新 all_frames 長度（補前面）
    all_frames = np.arange(0, all_frames[-1] + 1)

    # 組合內插後的資料
    interpolated_data = np.column_stack((all_frames, padded_features[0], padded_features[1], padded_features[2], padded_features[3]))
    return interpolated_data

def interpolate_mediapipe(yolo_frames, mediapipe_data, landmarks):
    interpolated_data = []
    # original_data = []
    
    for landmark in landmarks:
        # 獲取該 landmark 的座標
        group = mediapipe_data[mediapipe_data[:, 1] == landmark]
        frames = group[:, 0]
        x_coords = group[:, 2]
        y_coords = group[:, 3]
#         z_coords = group[:, 4]

        # original_data.append((frames, x_coords, y_coords, z_coords))


        # 將 MediaPipe 的第一幀對應到 YOLO 的第一幀，最後一幀對應到 YOLO 的最後一幀
        mediapipe_first_frame = frames[0]
        mediapipe_last_frame = frames[-1]

        # 將 YOLO 的幀範圍內的第一幀和最後一幀對應到 MediaPipe 的幀範圍
        aligned_frames = np.linspace(yolo_frames[0], yolo_frames[-1], len(frames))

        # 使用 interp1d 進行內插

        interp_x = interp1d(aligned_frames, x_coords, kind='linear')
        interp_y = interp1d(aligned_frames, y_coords, kind='linear')
#         interp_z = interp1d(aligned_frames, z_coords, kind='linear')

        
        interp_x = interp1d(aligned_frames, x_coords, kind='linear', fill_value="extrapolate")
        interp_y = interp1d(aligned_frames, y_coords, kind='linear', fill_value="extrapolate")
#         interp_z = interp1d(aligned_frames, z_coords, kind='linear', fill_value="extrapolate")
        
        # 使用 YOLO 的幀數範圍內進行內插
        interpolated_x = interp_x(yolo_frames)
        interpolated_y = interp_y(yolo_frames)
#         interpolated_z = interp_z(yolo_frames)
        
        # 將內插的資料添加到列表中
        for i, frame in enumerate(yolo_frames):
            interpolated_data.append([frame, landmark, interpolated_x[i], interpolated_y[i]])
    
    return interpolated_data

def run_interpolation(dir):
    # 載入 bar 資料
    bar_data = load_bar_data(os.path.join(dir, 'coordinates.txt'))

    # 內插 bar 中的 "no detection" 資料
    interpolated_bar_data = interpolate_missing_detections(bar_data)
    bar_interpolated_path = os.path.join(dir, 'coordinates_interpolated.txt')
    # 將結果保存到新的檔案，第一列為整數，其他為浮點數
    np.savetxt(bar_interpolated_path, interpolated_bar_data, delimiter=',', fmt='%d,%.8f,%.8f,%.8f,%.8f')

    print("BAR 'no detection' frames have been interpolated and saved to 'coordinates_interpolated.txt'.")
    if 'deadlift' in dir:
        visions = ['bar', 'left-front', 'left-back']
    elif 'benchpress' in dir:
        visions = ['rear', 'top']

    # 設定要處理的文件
    for v in visions:
        input_file = os.path.join(dir, f'skeleton_{v}.txt')  # 請替換為你的輸入檔名
        interpolated_skeleton = interpolate_landmarks(input_file)
        ####這邊先把skeleton資料內插成bar的數量，因bar資料量通常較長

        # 讀取 bar 的資料
        bar_frames = interpolated_bar_data[:, 0]  # frame numbers

        # 讀取第一份 skeleton 的資料
        landmarks = np.unique(interpolated_skeleton[:, 1])  # unique landmark numbers
        # 處理第一份 skeleton 資料
        interpolated_data = interpolate_mediapipe(bar_frames, interpolated_skeleton, landmarks)

        output_mediapipe = os.path.join(dir, f'interpolated_skeleton_{v}.txt')
        # 將內插後的資料寫入 TXT 檔案
        with open(output_mediapipe, 'w') as f:
            for entry in interpolated_data:
                f.write(f"{int(entry[0])},{int(entry[1])},{int(entry[2])},{int(entry[3])}\n")
