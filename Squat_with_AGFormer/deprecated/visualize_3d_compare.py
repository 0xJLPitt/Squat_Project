"""
目的: 同時展示並比較兩個 3D 骨架動畫的差異。
執行範例:
python visualize_3d_compare.py --npz1 D:\Pitt\Project\squat\demo\recording_20260415_105356\keypoints_3d.npz --npz2 D:\Pitt\Project\squat\demo\recording_20260415_105356\keypoints_3d_fixed.npz
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import sys
import os

def init_3d_pose(ax):
    ax.view_init(elev=15., azim=70)
    lcolor=(0, 0, 1)
    rcolor=(1, 0, 0)
    I = np.array([0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    lines = []
    for i in np.arange(len(I)):
        line, = ax.plot([], [], [], lw=2, color=lcolor if LR[i] else rcolor)
        lines.append(line)
        
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)
    return lines, I, J

def update_3d_pose(vals, ax, lines, I, J):
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        lines[i].set_data(x, y)
        lines[i].set_3d_properties(z)

    RADIUS = 0.72
    RADIUS_Z = 0.7
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])

def visualize_compare(npz1, npz2):
    print(f"載入原始資料: {npz1}")
    data1 = np.load(npz1, allow_pickle=True)['reconstruction']
    print(f"載入修正資料: {npz2}")
    data2 = np.load(npz2, allow_pickle=True)['reconstruction']
    
    max_val = max(np.max(data1), np.max(data2))
    if max_val > 0:
        data1 /= max_val
        data2 /= max_val
        
    n_frames = min(len(data1), len(data2))
    
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    ax1.set_title("Original 3D")
    ax2.set_title("Fixed Bone Lengths 3D")
    
    lines1, I1, J1 = init_3d_pose(ax1)
    lines2, I2, J2 = init_3d_pose(ax2)
    
    # 同步旋轉功能：當滑鼠在其中一個圖表上拖曳時，自動同步視角給另一個
    def on_move(event):
        if event.inaxes == ax1:
            ax2.view_init(elev=ax1.elev, azim=ax1.azim)
        elif event.inaxes == ax2:
            ax1.view_init(elev=ax2.elev, azim=ax2.azim)
            
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_release_event', on_move)
    
    def update(frame_idx):
        update_3d_pose(data1[frame_idx], ax1, lines1, I1, J1)
        update_3d_pose(data2[frame_idx], ax2, lines2, I2, J2)
        fig.suptitle(f"Frame {frame_idx}/{n_frames}", fontsize=16)
        return lines1 + lines2

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=33)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz1', type=str, required=True, help="左邊播放的原始檔案")
    parser.add_argument('--npz2', type=str, required=True, help="右邊播放的修復檔案")
    args = parser.parse_args()
    
    if not os.path.exists(args.npz1) or not os.path.exists(args.npz2):
        print("找不到指定的 .npz 檔案，請確認路徑。")
        sys.exit(1)
        
    visualize_compare(args.npz1, args.npz2)
