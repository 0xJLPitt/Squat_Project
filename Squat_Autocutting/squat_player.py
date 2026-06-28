import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import time
import ctypes

# 修正 Windows DPI 縮放導致的版型跑掉問題
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

class SquatPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Squat Analysis Pro - Ultra Performance View")
        self.root.geometry("1500x900")
        self.root.configure(bg='#1e1e1e')
        
        # 資料與播放狀態
        self.df_bar = None
        self.df_ts = None
        self.cap = None
        self.fps = 30
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        self.start_time = 0
        self.start_frame = 0
        
        # 繪圖物件
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.patch.set_facecolor('#1e1e1e')
        self.axes = None
        self.v_lines = []
        
        self._setup_ui()
        
    def _setup_ui(self):
        top_frame = tk.Frame(self.root, bg='#2d2d2d', height=50)
        top_frame.pack(fill=tk.X, side=tk.TOP)
        
        btn_open = ttk.Button(top_frame, text="Open Recording Folder", command=self.load_data)
        btn_open.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.lbl_info = tk.Label(top_frame, text="Select folder to start", bg='#2d2d2d', fg='white')
        self.lbl_info.pack(side=tk.LEFT, padx=10)

        bottom_frame = tk.Frame(self.root, bg='#2d2d2d', height=100)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.slider_var = tk.DoubleVar()
        self.slider = ttk.Scale(bottom_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                variable=self.slider_var, command=self.on_slider_move)
        self.slider.pack(fill=tk.X, padx=20, pady=10)
        
        ctrl_btns = tk.Frame(bottom_frame, bg='#2d2d2d')
        ctrl_btns.pack(pady=5)
        
        self.btn_play = ttk.Button(ctrl_btns, text="Play", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        tk.Label(ctrl_btns, text="Speed:", bg='#2d2d2d', fg='white').pack(side=tk.LEFT, padx=(20, 5))
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_menu = ttk.Combobox(ctrl_btns, textvariable=self.speed_var, values=[0.5, 1.0, 2.0, 4.0], width=5)
        self.speed_menu.pack(side=tk.LEFT, padx=5)
        
        self.lbl_frame = tk.Label(ctrl_btns, text="Frame: 0 / 0", bg='#2d2d2d', fg='white', font=('Arial', 10, 'bold'))
        self.lbl_frame.pack(side=tk.LEFT, padx=20)

        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chart_frame = tk.Frame(main_frame, bg='#1e1e1e')
        self.chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.video_frame_container = tk.Frame(main_frame, bg='black', width=500)
        self.video_frame_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=5)
        
        self.video_canvas = tk.Label(self.video_frame_container, bg='black')
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.root.bind('<space>', lambda e: self.toggle_play())

    def load_data(self):
        folder_path = filedialog.askdirectory()
        if not folder_path: return
        rec_name = os.path.basename(folder_path)
        bar_path = os.path.join(folder_path, "yolo_coordinates.txt")
        video_path = os.path.join(folder_path, "vision1.avi")
        
        try:
            df_raw = pd.read_csv(bar_path, header=None, names=['frame', 'bar_x', 'bar_y', 'w', 'h'])
            self.df_bar = df_raw.interpolate().bfill().ffill()
            self.df_bar['bar_y'] = pd.to_numeric(self.df_bar['bar_y'], errors='coerce')
            self.df_bar['v_y'] = np.gradient(self.df_bar['bar_y'].values) * 30
            self.df_bar['a_y'] = np.gradient(self.df_bar['v_y'].values) * 30
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {e}")
            return

        self.df_ts = None
        ts_path = os.path.join(folder_path, "analysis_results", f"{rec_name}_timeseries.csv")
        if os.path.exists(ts_path): self.df_ts = pd.read_csv(ts_path)

        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        self.lbl_info.config(text=f"Loaded: {rec_name}")
        self.slider.config(to=self.total_frames - 1)
        self._plot_charts()
        self.update_frame(0, force_seek=True)

    def _plot_charts(self):
        self.fig.clear()
        self.axes = self.fig.subplots(3, 1, sharex=True)
        self.fig.subplots_adjust(hspace=0.3, left=0.1, right=0.95, top=0.95, bottom=0.08)
        
        frames = self.df_bar.index
        self.axes[0].plot(frames, self.df_bar['bar_y'], color='#3498db', label='Position', linewidth=1)
        self.axes[0].invert_yaxis()
        self.axes[1].plot(frames, self.df_bar['v_y'], color='#95a5a6', linestyle='--', label='Velocity')
        self.axes[2].plot(frames, self.df_bar['a_y'], color='#9b59b6', linestyle='-.', alpha=0.6, label='Accel')
        
        reps = []
        if self.df_ts is not None:
            for rid in self.df_ts['rep_id'].unique():
                rd = self.df_ts[self.df_ts['rep_id'] == rid]
                reps.append({'s': int(rd['frame'].min()), 'e': int(rd['frame'].max())})

        self.v_lines = []
        for ax in self.axes:
            ax.set_facecolor('#1e1e1e')
            ax.grid(True, alpha=0.15, color='gray')
            ax.tick_params(colors='white', labelsize=8)
            for r in reps: ax.axvspan(r['s'], r['e'], color='#f1c40f', alpha=0.1)
            line = ax.axvline(0, color='#f1c40f', linewidth=2)
            self.v_lines.append(line)
            ax.legend(loc='upper right', fontsize=7, labelcolor='white')
            
        self.canvas.draw()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.config(text="Pause" if self.is_playing else "Play")
        if self.is_playing:
            self.start_time = time.time()
            self.start_frame = self.current_frame
            self.play_loop()

    def play_loop(self):
        if not self.is_playing: return
        
        # 核心優化：基於時間的跳幀邏輯 (Time-based frame skipping)
        elapsed = time.time() - self.start_time
        speed = self.speed_var.get()
        target_frame = int(self.start_frame + elapsed * self.fps * speed)
        
        if target_frame >= self.total_frames - 1:
            self.current_frame = self.total_frames - 1
            self.update_frame(self.current_frame)
            self.is_playing = False
            self.btn_play.config(text="Play")
            return

        # 更新當前影格
        self.current_frame = target_frame
        self.slider_var.set(self.current_frame)
        self.update_frame(self.current_frame)
        
        # 下一次循環 (盡可能快地調用，交由 update_frame 處理繪圖壓力)
        self.root.after(1, self.play_loop)

    def on_slider_move(self, event):
        if not self.is_playing:
            self.current_frame = int(self.slider_var.get())
            self.update_frame(self.current_frame, force_seek=True)

    def update_frame(self, frame_idx, force_seek=False):
        if self.cap is None: return
        
        # 1. 影片渲染優化
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if force_seek or abs(frame_idx - current_pos) > 2:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
        ret, frame = self.cap.read()
        if ret:
            # 快速縮放 (使用 INTER_NEAREST 提升速度)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            tw = 500
            th = int(h * (tw / w))
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_NEAREST)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.video_canvas.img_tk = img_tk
            self.video_canvas.config(image=img_tk)
            
        # 2. 穩定的線圖更新
        for line in self.v_lines:
            line.set_xdata([frame_idx])
        self.canvas.draw_idle()
        
        self.lbl_frame.config(text=f"Frame: {frame_idx} / {self.total_frames}")

if __name__ == "__main__":
    root = tk.Tk()
    ttk.Style().theme_use('clam')
    app = SquatPlayer(root)
    root.mainloop()
