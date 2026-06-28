"""
檔案目的: 執行手動外參校正。當自動校正找不到棋盤格時，可用此腳本手動點擊影像中的角點來計算外參。
注意: 滑鼠點擊的角點順序，必須與自動校正演算法的 1~15 點順序完全一致，否則 3D 空間會嚴重扭曲。
呼叫指令: python step4_manual_calibration.py
"""
import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QScrollArea, QMessageBox)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt5.QtCore import Qt

class StereoLabel(QLabel):
    def __init__(self, title):
        super().__init__()
        self.points = []
        self.pixmap_original = None
        self.title = title
        self.image_path = None
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setStyleSheet("border: 2px solid #ccc; background-color: #f9f9f9;")
        self.setText(f"請載入 {title}")

    def load_image(self, path):
        self.image_path = path
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: return
        h, w, c = img.shape
        qImg = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888)
        self.pixmap_original = QPixmap.fromImage(qImg)
        self.points = []
        self.update_display()
        self.setFixedSize(self.pixmap_original.size())

    def mousePressEvent(self, event):
        if self.pixmap_original is None: return
        if event.button() == Qt.LeftButton:
            self.points.append((event.pos().x(), event.pos().y()))
        elif event.button() == Qt.RightButton and self.points:
            self.points.pop()
        self.update_display()

    def update_display(self):
        if self.pixmap_original is None: return
        pix = self.pixmap_original.copy()
        painter = QPainter(pix)
        for i, p in enumerate(self.points):
            painter.setPen(QPen(QColor(255, 0, 0), 10))
            painter.drawPoint(p[0], p[1])
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.drawText(p[0] + 10, p[1] - 10, str(i + 1))
        painter.end()
        self.setPixmap(pix)

class StereoCalibWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("專業雙機手動校正工具")
        self.resize(1600, 900)
        self.target_dir = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 資料夾選擇控制列
        dir_layout = QHBoxLayout()
        self.btn_select_dir = QPushButton("選擇目標錄影資料夾")
        self.lbl_dir = QLabel("目前未選取任何資料夾 (將儲存於程式目錄)")
        self.lbl_dir.setStyleSheet("font-weight: bold; color: #444;")
        dir_layout.addWidget(self.btn_select_dir)
        dir_layout.addWidget(self.lbl_dir)
        dir_layout.addStretch()
        layout.addLayout(dir_layout)

        # 控制列
        btn_layout = QHBoxLayout()
        self.btn_a = QPushButton("載入相機 A (左)"); self.btn_b = QPushButton("載入相機 B (右)")
        self.btn_calc = QPushButton("選擇內參開始計算並儲存外參"); self.btn_calc.setStyleSheet("background: green; color: white;")
        btn_layout.addWidget(self.btn_a); btn_layout.addWidget(self.btn_b)
        btn_layout.addStretch(); btn_layout.addWidget(self.btn_calc)
        layout.addLayout(btn_layout)

        # 圖片列
        img_layout = QHBoxLayout()
        self.view_a = StereoLabel("相機 A"); self.view_b = StereoLabel("相機 B")
        self.s1 = QScrollArea(); self.s1.setWidget(self.view_a); img_layout.addWidget(self.s1)
        self.s2 = QScrollArea(); self.s2.setWidget(self.view_b); img_layout.addWidget(self.s2)
        layout.addLayout(img_layout)

        self.btn_select_dir.clicked.connect(self.select_target_dir)
        self.btn_a.clicked.connect(lambda: self.load(self.view_a))
        self.btn_b.clicked.connect(lambda: self.load(self.view_b))
        self.btn_calc.clicked.connect(self.calculate)

    def set_target_dir(self, path):
        self.target_dir = os.path.abspath(path)
        self.lbl_dir.setText(f"目前資料夾: {self.target_dir}")
        self.lbl_dir.setStyleSheet("font-weight: bold; color: green;")

    def select_target_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "選擇目標錄影資料夾", self.target_dir or "")
        if dir_path:
            self.set_target_dir(dir_path)

    def load(self, view):
        start_dir = self.target_dir or ""
        path, _ = QFileDialog.getOpenFileName(self, "選圖", start_dir, "Images (*.jpg *.png)")
        if path: view.load_image(path)

    def calculate(self):
        p1, p2 = self.view_a.points, self.view_b.points
        if len(p1) != len(p2) or len(p1) < 6:
            QMessageBox.warning(self, "警告", "兩邊點數需相同且大於 6 點")
            return

        # 尋找內參預設資料夾
        default_intrinsic_dir = ""
        if self.target_dir:
            test_path = os.path.join(self.target_dir, "intrinsics_vision")
            if os.path.exists(test_path):
                default_intrinsic_dir = test_path
            else:
                parent_dir = os.path.dirname(self.target_dir)
                test_path = os.path.join(parent_dir, "intrinsics_vision")
                if os.path.exists(test_path):
                    default_intrinsic_dir = test_path
                else:
                    default_intrinsic_dir = self.target_dir

        # 載入內參
        try:
             path1, _ = QFileDialog.getOpenFileName(self, "選取 相機 A 內參 (.npz)", default_intrinsic_dir, "NPZ Files (*.npz)")
             if not path1: return
             path2, _ = QFileDialog.getOpenFileName(self, "選取 相機 B 內參 (.npz)", default_intrinsic_dir, "NPZ Files (*.npz)")
             if not path2: return

             # 從檔名自動提取 ID (例如 intrinsics_vision2.npz -> vision2)
             cam1_id = os.path.basename(path1).replace("intrinsics_", "").replace(".npz", "")
             cam2_id = os.path.basename(path2).replace("intrinsics_", "").replace(".npz", "")

             m1 = np.load(path1)["mtx"]; d1 = np.load(path1)["dist"]
             m2 = np.load(path2)["mtx"]; d2 = np.load(path2)["dist"]
        except Exception as e:
             QMessageBox.critical(self, "錯誤", f"載入內參失敗: {str(e)}")
             return

        # 生成 3D 座標 (假設 5x3, 25mm)
        objp = np.zeros((len(p1), 3), np.float32)
        # 注意：假設使用者是按 5x3 順序點滿，如果只點幾點，這裡需要互動式設計
        # 簡單起見，我們假設你點的是完整的網格
        grid = np.zeros((5*3, 3), np.float32)
        grid[:, :2] = np.mgrid[0:5, 0:3].T.reshape(-1, 2) * 25.0
        # 如果點數剛好 15，直接使用 grid
        obj_pts = [grid[:len(p1)]] 

        ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            obj_pts, [np.array(p1, dtype=np.float32).reshape(-1,1,2)], 
            [np.array(p2, dtype=np.float32).reshape(-1,1,2)],
            m1, d1, m2, d2, (640, 480), flags=cv2.CALIB_FIX_INTRINSIC
        )

        if ret:
            # 決定輸出資料夾
            out_dir = self.target_dir if self.target_dir else "."
            out_name = f"extrinsics_{cam1_id}_to_{cam2_id}.npz"
            out = os.path.join(out_dir, out_name)
            np.savez(out, R=R, T=T)
            
            # 儲存手動校正視覺化圖片
            vis_msg = ""
            try:
                from calibration_visualizer import save_stereo_visualization
                img1 = cv2.imdecode(np.fromfile(self.view_a.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                img2 = cv2.imdecode(np.fromfile(self.view_b.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if img1 is not None and img2 is not None:
                    # 儲存在與載入圖片相同的目錄中，或者建立一個 calibration_visualized 子目錄
                    vis_dir = os.path.join(out_dir, "calibration_visualized")
                    frame_name = os.path.basename(self.view_a.image_path)
                    saved_path = save_stereo_visualization(
                        img1, img2, p1, p2, cam1_id, cam2_id, 
                        frame_name, vis_dir, pattern_size=(5, 3), is_manual=True
                    )
                    vis_msg = f"\n\n📸 視覺化圖已儲存至:\n{saved_path}"
            except Exception as vis_err:
                print(f"[WARN] 儲存手動標記視覺化影像失敗: {vis_err}")

            QMessageBox.information(
                self, "成功", 
                f"外參已儲存至 {out}\n\n"
                f"檢測誤差: {ret:.4f}{vis_msg}"
            )

if __name__ == "__main__":
    # ================= 設定路徑 =================
    # 您可以在這裡直接寫入要標註的資料夾路徑，就不需要用命令列輸入或每次在介面選取了
    TARGET_DIR = r"E:\squat\recordings_20260507\recording_20260423_190238_棋盤" 
    # ============================================

    app = QApplication(sys.argv)
    window = StereoCalibWindow()
    
    # 優先使用程式碼內設定的 TARGET_DIR；若不存在或為空，則檢查命令列參數
    if os.path.exists(TARGET_DIR) and os.path.isdir(TARGET_DIR):
        window.set_target_dir(TARGET_DIR)
    elif len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        window.set_target_dir(sys.argv[1])
        
    window.show()
    sys.exit(app.exec_())
