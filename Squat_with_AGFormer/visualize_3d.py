"""
目的: 以 3D 視覺化動畫展示 keypoints_3d.npz 的骨架動作。
執行範例:
python visualize_3d.py --npz D:\Pitt\Project\squat\demo\recording_20260415_105356\keypoints_3d.npz --save
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import sys
import os

def init_3d_pose(ax):
    """初始化 3D 骨架線段"""
    # 設定初始視角
    ax.view_init(elev=15., azim=70)

    # 左邊用藍色，右邊用紅色
    lcolor=(0, 0, 1)
    rcolor=(1, 0, 0)

    # Human3.6M 17 個關節點的連接關係
    I = np.array([0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    lines = []
    # 畫出初始的空線段
    for i in np.arange(len(I)):
        line, = ax.plot([], [], [], lw=2, color=lcolor if LR[i] else rcolor)
        lines.append(line)
    
    # 將背景的灰色網格拿掉，讓畫面更乾淨
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)
    
    return lines, I, J

def update_3d_pose(vals, ax, lines, I, J):
    """只更新線段座標，不清除畫面，這樣才能用滑鼠旋轉"""
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        lines[i].set_data(x, y)
        lines[i].set_3d_properties(z)

    # 設定顯示範圍 (避免骨架因為移動而忽大忽小)
    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])

def visualize_npz(npz_path, save_video=False):
    print(f"正在讀取檔案: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    keypoints = data['reconstruction'] # shape: (N_frames, 17, 3)
    
    # 正規化大小，為了確保畫面縮放統一
    max_value = np.max(keypoints)
    if max_value > 0:
        keypoints /= max_value
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    lines, I, J = init_3d_pose(ax)
    
    def update(frame_idx):
        update_3d_pose(keypoints[frame_idx], ax, lines, I, J)
        ax.set_title(f"Frame {frame_idx}/{len(keypoints)}")
        return lines
        
    print("建立動畫中...")
    # 產生動畫物件，設定每幀間隔 33 毫秒 (約 30 FPS)
    ani = animation.FuncAnimation(fig, update, frames=len(keypoints), interval=33)
    
    if save_video:
        out_path = npz_path.replace('.npz', '.mp4')
        print(f"正在將動畫儲存至: {out_path}")
        print("(這可能會花上一點時間，請耐心等候...)")
        try:
            # 需要系統有安裝 ffmpeg 才能輸出 mp4
            ani.save(out_path, writer='ffmpeg', fps=30)
            print("儲存成功！")
        except Exception as e:
            print("儲存失敗！可能是您的環境沒有安裝 ffmpeg。錯誤訊息:", e)
            print("您也可以改用 --show 來直接播放視窗觀看。")
    else:
        print("正在開啟預覽視窗...")
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize 3D Keypoints from .npz")
    parser.add_argument('--npz', type=str, required=True, help='指向 keypoints_3d.npz 檔案的路徑')
    parser.add_argument('--save', action='store_true', help='加上這個參數會把動畫存成 mp4 (需要 ffmpeg)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.npz):
        print(f"錯誤: 找不到檔案 {args.npz}")
        sys.exit(1)
        
    visualize_npz(args.npz, save_video=args.save)
