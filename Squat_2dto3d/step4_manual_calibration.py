"""
檔案目的: 執行手動與半自動外參校正（可拖曳微調版）。
特點:
  1. 支援滑鼠即時拖曳（Drag & Drop）修改 1~15 角點，滑鼠懸停高亮、抓取游標提示。
  2. 支援鍵盤方向鍵 (Up/Down/Left/Right) 1 pixel 精確微調（Shift 為 5 pixel）。
  3. 內建局部「放大鏡 (Magnifier HUD)」，清晰檢視棋盤格交點像素與準心。
  4. 支援滑鼠滾輪縮放 (Zoom) 與中/右鍵拖曳平移視野 (Pan)。
  5. 支援「一鍵載入 Visualized 對照圖」：從 extrinsics/visualized 選取檔案後，自動載入相機雙圖與內參，並自動辨識初始角點。
  6. 提供「次像素吸附 (SubPix Refine)」、「180度反轉點序 (Invert)」、「自動偵測角點 (Auto-Detect)」等輔助工具。
  7. 支援單張影格校正與「多影格收集合併校正」，精準計算 R, T, RMS Error 與 Baseline。
呼叫指令:
  python step4_manual_calibration.py
"""

import sys
import os
import re
import glob
import json
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QMessageBox, QGroupBox, QSplitter, QStatusBar,
    QCheckBox, QComboBox, QFrame, QToolTip
)
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor, QImage, QFont,
    QCursor, QBrush, QPainterPath, QKeySequence
)
from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF, pyqtSignal

# 嘗試載入視覺化對照工具
try:
    from calibration_visualizer import save_stereo_visualization
except ImportError:
    try:
        from Squat_2dto3d.calibration_visualizer import save_stereo_visualization
    except ImportError:
        save_stereo_visualization = None


# ==============================================================================
# 【對向相機 180 度點序反轉設定】 (Opposite Camera Configuration)
# ==============================================================================
# 當相機為對向拍攝時，棋盤格在空間中相差 180 度 (點 1 與點 15 顛倒)。
# 您可以在此直接修改設定對向相機，或者指定特定相機配對中誰需要反轉：
#
# 【設定 A】對向相機清單 (凡是此相機，載入時預設執行 180 度反轉):
# 臥推拍攝配置中，i17 與 i15、i16 位於對向，因此設定為 ["i17"]
OPPOSITE_CAMERAS = ["i17"]

# 【設定 B】特定相機配對反轉規則 (Pair-Specific Rules):
# 格式: (相機1, 相機2): "需要反轉的相機名稱" (若都不反轉則設為 None)
PAIR_FLIP_RULES = {
    ("i15", "i17"): "i17",  # i15 與 i17 配對時，i17 反轉 180 度
    ("i16", "i17"): "i17",  # i16 與 i17 配對時，i17 反轉 180 度
    ("i15", "i16"): None,   # i15 與 i16 為同側，皆不反轉
    # 實驗室深蹲配置 (vision / squat):
    ("vision2", "vision3"): None,
    ("vision3", "vision4"): "vision4",
    ("vision4", "vision5"): None,
    ("RR", "RLU"): None,
    ("RLU", "FL"): "FL",
    ("FL", "FR"): None,
}

def should_camera_flip(cam_name, partner_name=None):
    """判斷給定相機在當前配對中是否需要 180 度反轉"""
    if not cam_name:
        return False
    if partner_name:
        pair = (cam_name, partner_name)
        pair_inv = (partner_name, cam_name)
        if pair in PAIR_FLIP_RULES:
            return PAIR_FLIP_RULES[pair] == cam_name
        elif pair_inv in PAIR_FLIP_RULES:
            return PAIR_FLIP_RULES[pair_inv] == cam_name
    return cam_name in OPPOSITE_CAMERAS
# ==============================================================================


class DraggableStereoCanvas(QWidget):
    """
    可拖曳、縮放、平移並具備放大鏡輔助的互動式標註畫布
    """
    pointsChanged = pyqtSignal()
    pointSelected = pyqtSignal(int)

    def __init__(self, title="相機視角"):
        super().__init__()
        self.title = title
        self.image_path = None
        self.cam_id = ""
        self.cv_img = None           # BGR numpy 影像
        self.qimg = None             # QImage 原圖
        
        # 標註角點 (浮點數座標 [x, y]，對應原圖像素)
        self.points = []
        self.pattern_size = (5, 3)   # 5 列 3 行 (共 15 點)
        
        # 互動狀態
        self.selected_idx = None
        self.dragged_idx = None
        self.hover_idx = None
        
        # 縮放與平移
        self.scale = 1.0
        self.offset = QPointF(0.0, 0.0)
        self.is_panning = False
        self.last_mouse_pos = QPoint()
        
        # 放大鏡設定
        self.show_magnifier = True
        self.magnifier_zoom = 4.0
        self.magnifier_size = 170
        
        # 抓取感應半徑 (螢幕像素)
        self.grab_radius = 16.0
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3a3a3a;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.exists(path) and path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")):
                fname = os.path.basename(path)
                if fname.startswith("match_"):
                    win = self.window()
                    if win and hasattr(win, "quick_load_visualized") and callable(win.quick_load_visualized):
                        win.quick_load_visualized(path)
                        return
                if self.load_image(path):
                    self.auto_detect_corners()
                    self.pointsChanged.emit()

    # ================= 影像載入與重設 =================
    def load_image(self, path, cam_id=None):
        self.image_path = path
        if cam_id:
            self.cam_id = cam_id
        else:
            base = os.path.basename(path)
            for cid in ["i15", "i16", "i17", "vision2", "vision3", "vision4", "vision5", "RR", "RLU", "FL", "FR"]:
                if cid.lower() in path.lower():
                    self.cam_id = cid
                    break

        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "錯誤", f"無法讀取影像: {path}")
            return False

        self.cv_img = img
        img_c = np.ascontiguousarray(img)
        h, w, c = img_c.shape
        self.qimg = QImage(img_c.data, w, h, c * w, QImage.Format_BGR888).copy()
        
        # 自動適配縮放到當前視窗大小
        self.fit_to_view()
        self.points = []
        self.selected_idx = None
        self.dragged_idx = None
        self.hover_idx = None
        self.update()
        return True

    def fit_to_view(self):
        """自動調整縮放與置中以完整顯示圖片"""
        if self.qimg is None or self.width() <= 0 or self.height() <= 0:
            return
        img_w, img_h = self.qimg.width(), self.qimg.height()
        scale_x = (self.width() - 20) / max(1, img_w)
        scale_y = (self.height() - 20) / max(1, img_h)
        self.scale = min(scale_x, scale_y, 1.0)
        
        # 置中
        disp_w = img_w * self.scale
        disp_h = img_h * self.scale
        self.offset = QPointF((self.width() - disp_w) / 2.0, (self.height() - disp_h) / 2.0)
        self.update()

    def reset_view(self):
        """重設為 100% 原始比例"""
        if self.qimg is None: return
        self.scale = 1.0
        self.offset = QPointF(20.0, 20.0)
        self.update()

    # ================= 座標轉換 =================
    def widget_to_image(self, pt: QPointF):
        """視窗座標轉原圖像素座標"""
        x = (pt.x() - self.offset.x()) / self.scale
        y = (pt.y() - self.offset.y()) / self.scale
        return x, y

    def image_to_widget(self, x: float, y: float):
        """原圖像素座標轉視窗座標"""
        wx = x * self.scale + self.offset.x()
        wy = y * self.scale + self.offset.y()
        return QPointF(wx, wy)

    # ================= 自動辨識與角點處理 =================
    def auto_detect_corners(self, partner_cam=None, force_flip=None):
        """自動偵測棋盤格角點並填入點位"""
        if self.cv_img is None:
            QMessageBox.warning(self, "提示", "請先載入影像")
            return False

        gray = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        pattern = self.pattern_size

        # 針對 i15 特別採用中央躺板 ROI 偵測
        ret, corners = False, None
        if self.cam_id == "i15":
            roi_y1, roi_y2 = 260, 540
            roi_x1, roi_x2 = 400, 800
            crop = gray[roi_y1:roi_y2, roi_x1:roi_x2]
            ret, corners = cv2.findChessboardCornersSB(crop, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                corners[:, 0, 0] += roi_x1
                corners[:, 0, 1] += roi_y1

        if not ret:
            # 優先嘗試 SB 高精確演算法
            ret, corners = cv2.findChessboardCornersSB(
                gray, pattern, 
                cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            )

        if not ret:
            # 傳統棋盤格演算法 fallback
            ret, corners = cv2.findChessboardCorners(
                gray, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        # 判斷是否需 180 度點序翻轉 (依據使用者在檔案頂部的對向相機設定)
        do_flip = force_flip if force_flip is not None else should_camera_flip(self.cam_id, partner_cam)

        if ret and corners is not None:
            if do_flip:
                corners = corners[::-1, :, :].copy()

            self.points = [c[0].tolist() for c in corners]
            self.selected_idx = 0
            self.update()
            self.pointsChanged.emit()
            return True
        else:
            # 偵測失敗時，產生預設置中的 5x3 網格供使用者直接拖曳
            box_w = 200.0
            box_h = 120.0
            start_x = (w - box_w) / 2.0
            start_y = (h - box_h) / 2.0
            pts = []
            for r in range(pattern[1]):
                for c in range(pattern[0]):
                    px = start_x + c * (box_w / (pattern[0] - 1))
                    py = start_y + r * (box_h / (pattern[1] - 1))
                    pts.append([px, py])
            if do_flip:
                pts = pts[::-1]
            self.points = pts
            self.selected_idx = 0
            self.update()
            self.pointsChanged.emit()
            QMessageBox.information(
                self, "自動偵測未抓到完整角點", 
                "未能在畫面中自動辨識出完整角點。\n已在中央生成 15 點初始網格，請直接拖曳各點至棋盤格角點位置！"
            )
            return False

    def refine_subpix(self):
        """對當前所有角點執行次像素幾何吸附"""
        if self.cv_img is None or not self.points:
            return
        gray = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
        pts_arr = np.array(self.points, dtype=np.float32).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        try:
            refined = cv2.cornerSubPix(gray, pts_arr, (7, 7), (-1, -1), criteria)
            self.points = [p[0].tolist() for p in refined]
            self.update()
            self.pointsChanged.emit()
            QMessageBox.information(self, "吸附完成", f"已成功對 {len(self.points)} 個角點完成次像素幾何微調吸附！")
        except Exception as e:
            QMessageBox.warning(self, "吸附失敗", f"次像素微調吸附過程發生錯誤: {e}")

    def invert_order(self):
        """顛倒點序 (1 號變 15 號)"""
        if self.points:
            self.points = self.points[::-1]
            if self.selected_idx is not None:
                self.selected_idx = len(self.points) - 1 - self.selected_idx
            self.update()
            self.pointsChanged.emit()

    def clear_points(self):
        """清除所有點位"""
        self.points = []
        self.selected_idx = None
        self.dragged_idx = None
        self.hover_idx = None
        self.update()
        self.pointsChanged.emit()

    # ================= 滑鼠與鍵盤事件 =================
    def _find_near_point(self, mouse_pos: QPoint):
        """檢查滑鼠附近是否有角點"""
        best_idx = None
        min_dist = self.grab_radius
        for idx, (px, py) in enumerate(self.points):
            wpt = self.image_to_widget(px, py)
            dist = np.hypot(wpt.x() - mouse_pos.x(), wpt.y() - mouse_pos.y())
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        return best_idx

    def mousePressEvent(self, event):
        if self.qimg is None:
            return

        # 中鍵或右鍵（若無點位被點選）: 開始平移
        if event.button() == Qt.MiddleButton or (event.button() == Qt.RightButton and event.modifiers() == Qt.NoModifier):
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        # 左鍵點擊: 選擇或拖曳點位
        if event.button() == Qt.LeftButton:
            near_idx = self._find_near_point(event.pos())
            if near_idx is not None:
                self.dragged_idx = near_idx
                self.selected_idx = near_idx
                self.pointSelected.emit(near_idx)
                self.setCursor(Qt.ClosedHandCursor)
            else:
                # 點擊空白處: 若未滿 15 點則新增點，否則取消選擇
                if len(self.points) < 15:
                    ix, iy = self.widget_to_image(event.pos())
                    # 限制在圖片範圍內
                    ix = max(0.0, min(float(self.qimg.width() - 1), ix))
                    iy = max(0.0, min(float(self.qimg.height() - 1), iy))
                    self.points.append([ix, iy])
                    self.selected_idx = len(self.points) - 1
                    self.dragged_idx = self.selected_idx
                    self.pointSelected.emit(self.selected_idx)
                    self.pointsChanged.emit()
                else:
                    self.selected_idx = None
            self.update()

    def mouseMoveEvent(self, event):
        if self.qimg is None:
            return

        # 拖曳點位
        if self.dragged_idx is not None:
            ix, iy = self.widget_to_image(event.pos())
            ix = max(0.0, min(float(self.qimg.width() - 1), ix))
            iy = max(0.0, min(float(self.qimg.height() - 1), iy))
            self.points[self.dragged_idx] = [ix, iy]
            self.update()
            self.pointsChanged.emit()
            return

        # 平移畫布
        if self.is_panning:
            delta = event.pos() - self.last_mouse_pos
            self.offset += QPointF(delta.x(), delta.y())
            self.last_mouse_pos = event.pos()
            self.update()
            return

        # 懸停偵測
        near_idx = self._find_near_point(event.pos())
        if near_idx != self.hover_idx:
            self.hover_idx = near_idx
            if self.hover_idx is not None:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.dragged_idx is not None:
            self.dragged_idx = None
            if self.hover_idx is not None:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            self.update()

        if self.is_panning:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
            self.update()

    def wheelEvent(self, event):
        """滑鼠滾輪以滑鼠為中心縮放"""
        if self.qimg is None:
            return
        delta = event.angleDelta().y()
        if delta == 0: return

        old_scale = self.scale
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        new_scale = max(0.1, min(10.0, old_scale * factor))

        # 保持滑鼠懸停點在原圖中的位置不變
        mouse_pos = QPointF(event.pos())
        self.offset = mouse_pos - (mouse_pos - self.offset) * (new_scale / old_scale)
        self.scale = new_scale
        self.update()

    def keyPressEvent(self, event):
        """方向鍵微調選取中的點位"""
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.points):
            step = 5.0 if (event.modifiers() & Qt.ShiftModifier) else 1.0
            if event.modifiers() & Qt.ControlModifier:
                step = 0.2

            px, py = self.points[self.selected_idx]
            if event.key() == Qt.Key_Left:
                px -= step
            elif event.key() == Qt.Key_Right:
                px += step
            elif event.key() == Qt.Key_Up:
                py -= step
            elif event.key() == Qt.Key_Down:
                py += step
            elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                self.points.pop(self.selected_idx)
                self.selected_idx = None
                self.update()
                self.pointsChanged.emit()
                return
            else:
                super().keyPressEvent(event)
                return

            if self.qimg:
                px = max(0.0, min(float(self.qimg.width() - 1), px))
                py = max(0.0, min(float(self.qimg.height() - 1), py))
            self.points[self.selected_idx] = [px, py]
            self.update()
            self.pointsChanged.emit()
        else:
            super().keyPressEvent(event)

    # ================= 繪圖渲染 =================
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 填滿深色背景
        painter.fillRect(self.rect(), QColor(24, 24, 27))

        if self.qimg is None:
            painter.setPen(QColor(160, 160, 160))
            painter.setFont(QFont("Microsoft JhengHei", 12, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, f"請載入 {self.title}\n(支援直接拖曳照片至視窗)")
            return

        # 繪製影像
        painter.save()
        painter.translate(self.offset.x(), self.offset.y())
        painter.scale(self.scale, self.scale)
        painter.drawImage(0, 0, self.qimg)
        painter.restore()

        # 繪製角點與網格輔助線
        self._draw_grid_lines(painter)
        self._draw_points_and_labels(painter)

        # 繪製放大鏡 HUD
        if self.show_magnifier and (self.dragged_idx is not None or self.selected_idx is not None):
            active_idx = self.dragged_idx if self.dragged_idx is not None else self.selected_idx
            try:
                self._draw_magnifier_hud(painter, active_idx)
            except Exception as hud_err:
                pass

    def _draw_grid_lines(self, painter: QPainter):
        """繪製網格連線，第 1 列特別以桃紅色標記"""
        cols, rows = self.pattern_size
        n = len(self.points)
        if n < 2: return

        # 繪製水平列連線
        for r in range(rows):
            for c in range(cols - 1):
                i1 = r * cols + c
                i2 = i1 + 1
                if i2 < n:
                    p1 = self.image_to_widget(*self.points[i1])
                    p2 = self.image_to_widget(*self.points[i2])
                    if r == 0:
                        # 第 1 列使用醒目的桃紅色粗線 (與 calibration_visualizer 一致)
                        painter.setPen(QPen(QColor(255, 0, 255), 3.5))
                    else:
                        painter.setPen(QPen(QColor(0, 200, 255, 160), 1.8, Qt.DashLine))
                    painter.drawLine(p1, p2)

        # 繪製垂直行連線
        for c in range(cols):
            for r in range(rows - 1):
                i1 = r * cols + c
                i2 = (r + 1) * cols + c
                if i2 < n:
                    p1 = self.image_to_widget(*self.points[i1])
                    p2 = self.image_to_widget(*self.points[i2])
                    painter.setPen(QPen(QColor(0, 200, 255, 140), 1.5, Qt.DashLine))
                    painter.drawLine(p1, p2)

    def _draw_points_and_labels(self, painter: QPainter):
        """繪製角點圓圈、編號與選取外框"""
        font = QFont("Arial", 9, QFont.Bold)
        painter.setFont(font)

        for idx, (px, py) in enumerate(self.points):
            wpt = self.image_to_widget(px, py)
            is_selected = (idx == self.selected_idx)
            is_hover = (idx == self.hover_idx)
            is_first = (idx == 0)

            # 外圈光暈 / 選中光環
            if is_selected:
                painter.setPen(QPen(QColor(255, 165, 0, 220), 3))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(wpt, 13, 13)
                # 十字準心
                painter.setPen(QPen(QColor(255, 200, 0, 200), 1.5))
                painter.drawLine(QPointF(wpt.x() - 17, wpt.y()), QPointF(wpt.x() + 17, wpt.y()))
                painter.drawLine(QPointF(wpt.x(), wpt.y() - 17), QPointF(wpt.x(), wpt.y() + 17))
            elif is_hover:
                painter.setPen(QPen(QColor(255, 255, 255, 200), 2.5))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(wpt, 11, 11)

            # 角點實心圓
            core_color = QColor(255, 50, 50) if is_first else QColor(0, 255, 100)
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.setBrush(QBrush(core_color))
            painter.drawEllipse(wpt, 5.5, 5.5)

            # 編號標籤文字 (具黑底框增加對比)
            txt = str(idx + 1)
            tx = wpt.x() + 9
            ty = wpt.y() - 9

            # 陰影文字
            painter.setPen(QPen(QColor(0, 0, 0), 3))
            painter.drawText(QPointF(tx, ty), txt)
            # 主色文字
            painter.setPen(QPen(QColor(255, 255, 255) if not is_first else QColor(255, 220, 0)))
            painter.drawText(QPointF(tx, ty), txt)

    def _draw_magnifier_hud(self, painter: QPainter, active_idx: int):
        """在右上角繪製局部放大鏡 HUD，顯示像素交界與中心點"""
        if active_idx is None or active_idx >= len(self.points) or self.qimg is None:
            return

        px, py = self.points[active_idx]
        crop_size = int(self.magnifier_size / self.magnifier_zoom)
        half = crop_size // 2

        ix, iy = int(round(px)), int(round(py))
        w, h = self.qimg.width(), self.qimg.height()

        x1, x2 = max(0, ix - half), min(w, ix + half)
        y1, y2 = max(0, iy - half), min(h, iy + half)
        if x2 <= x1 or y2 <= y1: return

        # HUD 放置於畫布右上角
        hud_w, hud_h = self.magnifier_size, self.magnifier_size
        hud_x = self.width() - hud_w - 15
        hud_y = 15
        hud_rect = QRectF(hud_x, hud_y, hud_w, hud_h)

        painter.save()
        # 半透明圓角底框
        painter.setPen(QPen(QColor(0, 200, 255), 2))
        painter.setBrush(QBrush(QColor(10, 10, 15, 230)))
        painter.drawRoundedRect(hud_rect, 10, 10)

        # 局部放大影像 (直接從 self.qimg 繪製，無效能開銷且避免記憶體格式錯誤)
        painter.setClipRect(hud_rect)
        scale_hud = self.magnifier_zoom
        dest_x = hud_x + (hud_w / 2.0) - (px - x1) * scale_hud
        dest_y = hud_y + (hud_h / 2.0) - (py - y1) * scale_hud

        source_rect = QRectF(x1, y1, x2 - x1, y2 - y1)
        target_rect = QRectF(dest_x, dest_y, (x2 - x1) * scale_hud, (y2 - y1) * scale_hud)
        painter.drawImage(target_rect, self.qimg, source_rect)

        # 中心十字準心
        cx, cy = hud_x + hud_w / 2.0, hud_y + hud_h / 2.0
        painter.setPen(QPen(QColor(0, 0, 0, 180), 3))
        painter.drawLine(QPointF(cx - 15, cy), QPointF(cx + 15, cy))
        painter.drawLine(QPointF(cx, cy - 15), QPointF(cx, cy + 15))
        painter.setPen(QPen(QColor(255, 0, 100), 1.5))
        painter.drawLine(QPointF(cx - 15, cy), QPointF(cx + 15, cy))
        painter.drawLine(QPointF(cx, cy - 15), QPointF(cx, cy + 15))
        painter.drawEllipse(QPointF(cx, cy), 3, 3)

        # 標註資訊文字
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        info_text = f"點 {active_idx+1} ({px:.1f}, {py:.1f})"
        painter.drawText(QPointF(hud_x + 8, hud_y + hud_h - 8), info_text)
        painter.restore()


class StereoCalibWindow(QMainWindow):
    """
    立體雙機手動與半自動外參校正主視窗
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("立體相機外參校正工具 (可拖曳微調升級版)")
        self.resize(1700, 950)

        # 預設路徑與資料設定
        self.target_dir = None
        self.pair_id = ("i15", "i16")
        self.active_frame_name = ""
        self.intrinsic_path1 = ""
        self.intrinsic_path2 = ""

        # 多影格收集清單: [{"p1": pts1, "p2": pts2, "frame": name}, ...]
        self.collected_frames = []

        self.setAcceptDrops(True)
        self._init_ui()
        self.statusBar().showMessage("準備就緒。可點擊「從 Visualized 對照圖載入」或手動載入左右相機圖檔（亦可直接拖曳圖檔進視窗）。")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls: return
        paths = [
            u.toLocalFile() for u in urls 
            if os.path.exists(u.toLocalFile()) and u.toLocalFile().lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
            )
        ]
        if not paths: return

        # 優先檢查是否拖入 match_... 對照圖
        for p in paths:
            if os.path.basename(p).startswith("match_"):
                self.quick_load_visualized(p)
                return

        # 依拖入張數自動分派至畫布 A / B
        if len(paths) >= 2:
            self.canvas_a.load_image(paths[0])
            self.canvas_a.auto_detect_corners()
            self.canvas_b.load_image(paths[1])
            self.canvas_b.auto_detect_corners()
            self.update_point_counts()
        elif len(paths) == 1:
            if self.canvas_a.cv_img is None:
                self.canvas_a.load_image(paths[0])
                self.canvas_a.auto_detect_corners()
            else:
                self.canvas_b.load_image(paths[0])
                self.canvas_b.auto_detect_corners()
            self.update_point_counts()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        root_layout = QVBoxLayout(main_widget)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(8)

        # 1. 頂部快速工作列
        top_bar = QHBoxLayout()
        self.btn_load_vis = QPushButton("📂 從 Visualized 對照圖載入 (推薦)")
        self.btn_load_vis.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 7px 14px; border-radius: 4px;")
        self.btn_load_vis.setToolTip("選取 benchpress_3D/extrinsics/visualized 中的 match_... 圖片，自動載入雙機原圖、內參並辨識點位！")
        
        self.btn_select_dir = QPushButton("選擇專案目錄")
        self.lbl_dir = QLabel("未選取目錄 (預設為當前執行位置)")
        self.lbl_dir.setStyleSheet("color: #aaa; font-weight: bold;")

        top_bar.addWidget(self.btn_load_vis)
        top_bar.addWidget(self.btn_select_dir)
        top_bar.addWidget(self.lbl_dir)
        top_bar.addStretch()

        self.btn_help = QPushButton("❓ 操作說明")
        top_bar.addWidget(self.btn_help)
        root_layout.addLayout(top_bar)

        # 2. 畫布區域 (左右雙分割)
        splitter = QSplitter(Qt.Horizontal)

        # 相機 A 區塊
        box_a = QGroupBox("相機 A (左視角)")
        box_a.setStyleSheet("QGroupBox { font-weight: bold; color: #38bdf8; }")
        layout_a = QVBoxLayout(box_a)
        ctrl_a = QHBoxLayout()
        self.btn_load_a = QPushButton("載入影像 A")
        self.btn_detect_a = QPushButton("⚡ 自動辨識")
        self.btn_subpix_a = QPushButton("🧲 次像素吸附")
        self.btn_flip_a = QPushButton("🔄 翻轉點序 (180°)")
        self.btn_clear_a = QPushButton("清除")
        self.chk_flip_a = QCheckBox("對向相機 (180° 反轉)")
        self.chk_flip_a.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.chk_flip_a.setToolTip("若此相機位於棋盤格對向（面對面拍攝），勾選以自動進行 1~15 點序反轉")
        ctrl_a.addWidget(self.btn_load_a)
        ctrl_a.addWidget(self.btn_detect_a)
        ctrl_a.addWidget(self.btn_subpix_a)
        ctrl_a.addWidget(self.btn_flip_a)
        ctrl_a.addWidget(self.chk_flip_a)
        ctrl_a.addWidget(self.btn_clear_a)
        layout_a.addLayout(ctrl_a)

        self.canvas_a = DraggableStereoCanvas("相機 A")
        layout_a.addWidget(self.canvas_a)

        # 相機 A 資訊列
        info_a = QHBoxLayout()
        self.lbl_info_a = QLabel("點數: 0 / 15")
        self.btn_fit_a = QPushButton("適配大小")
        self.btn_fit_a.setMaximumWidth(70)
        info_a.addWidget(self.lbl_info_a)
        info_a.addStretch()
        info_a.addWidget(self.btn_fit_a)
        layout_a.addLayout(info_a)

        splitter.addWidget(box_a)

        # 相機 B 區塊
        box_b = QGroupBox("相機 B (右視角)")
        box_b.setStyleSheet("QGroupBox { font-weight: bold; color: #a78bfa; }")
        layout_b = QVBoxLayout(box_b)
        ctrl_b = QHBoxLayout()
        self.btn_load_b = QPushButton("載入影像 B")
        self.btn_detect_b = QPushButton("⚡ 自動辨識")
        self.btn_subpix_b = QPushButton("🧲 次像素吸附")
        self.btn_flip_b = QPushButton("🔄 翻轉點序 (180°)")
        self.btn_clear_b = QPushButton("清除")
        self.chk_flip_b = QCheckBox("對向相機 (180° 反轉)")
        self.chk_flip_b.setStyleSheet("color: #a78bfa; font-weight: bold;")
        self.chk_flip_b.setToolTip("若此相機位於棋盤格對向（面對面拍攝），勾選以自動進行 1~15 點序反轉")
        ctrl_b.addWidget(self.btn_load_b)
        ctrl_b.addWidget(self.btn_detect_b)
        ctrl_b.addWidget(self.btn_subpix_b)
        ctrl_b.addWidget(self.btn_flip_b)
        ctrl_b.addWidget(self.chk_flip_b)
        ctrl_b.addWidget(self.btn_clear_b)
        layout_b.addLayout(ctrl_b)

        self.canvas_b = DraggableStereoCanvas("相機 B")
        layout_b.addWidget(self.canvas_b)

        # 相機 B 資訊列
        info_b = QHBoxLayout()
        self.lbl_info_b = QLabel("點數: 0 / 15")
        self.btn_fit_b = QPushButton("適配大小")
        self.btn_fit_b.setMaximumWidth(70)
        info_b.addWidget(self.lbl_info_b)
        info_b.addStretch()
        info_b.addWidget(self.btn_fit_b)
        layout_b.addLayout(info_b)

        splitter.addWidget(box_b)
        root_layout.addWidget(splitter, stretch=1)

        # 3. 底部計算與多影格控制列
        bottom_frame = QFrame()
        bottom_frame.setStyleSheet("background-color: #262626; border-radius: 6px; padding: 6px;")
        bottom_layout = QHBoxLayout(bottom_frame)

        # 多影格管理
        self.lbl_collected = QLabel("已收集影格: 0 組")
        self.lbl_collected.setStyleSheet("color: #fbbf24; font-weight: bold; margin-right: 10px;")
        self.btn_add_frame = QPushButton("➕ 將目前微調影格加入清單")
        self.btn_clear_frames = QPushButton("清空清單")
        bottom_layout.addWidget(self.lbl_collected)
        bottom_layout.addWidget(self.btn_add_frame)
        bottom_layout.addWidget(self.btn_clear_frames)
        bottom_layout.addSpacing(20)

        # 輸出設定與計算
        bottom_layout.addStretch()
        self.btn_calc_single = QPushButton("🚀 單張計算並儲存外參")
        self.btn_calc_single.setStyleSheet("background-color: #15803d; color: white; font-weight: bold; font-size: 13px; padding: 8px 16px; border-radius: 4px;")
        
        self.btn_calc_multi = QPushButton("🌟 使用所有收集影格計算外參")
        self.btn_calc_multi.setStyleSheet("background-color: #047857; color: white; font-weight: bold; font-size: 13px; padding: 8px 16px; border-radius: 4px;")

        bottom_layout.addWidget(self.btn_calc_single)
        bottom_layout.addWidget(self.btn_calc_multi)
        root_layout.addWidget(bottom_frame)

        # 連接訊號
        self.btn_load_vis.clicked.connect(self.quick_load_visualized)
        self.btn_select_dir.clicked.connect(self.select_target_dir)
        self.btn_help.clicked.connect(self.show_help)

        self.btn_load_a.clicked.connect(lambda: self.manual_load_image(self.canvas_a))
        self.btn_load_b.clicked.connect(lambda: self.manual_load_image(self.canvas_b))
        self.btn_detect_a.clicked.connect(self.canvas_a.auto_detect_corners)
        self.btn_detect_b.clicked.connect(self.canvas_b.auto_detect_corners)
        self.btn_subpix_a.clicked.connect(self.canvas_a.refine_subpix)
        self.btn_subpix_b.clicked.connect(self.canvas_b.refine_subpix)
        self.btn_flip_a.clicked.connect(lambda: self.chk_flip_a.setChecked(not self.chk_flip_a.isChecked()))
        self.btn_flip_b.clicked.connect(lambda: self.chk_flip_b.setChecked(not self.chk_flip_b.isChecked()))
        self.chk_flip_a.toggled.connect(lambda checked: self.canvas_a.invert_order())
        self.chk_flip_b.toggled.connect(lambda checked: self.canvas_b.invert_order())
        self.btn_clear_a.clicked.connect(self.canvas_a.clear_points)
        self.btn_clear_b.clicked.connect(self.canvas_b.clear_points)
        self.btn_fit_a.clicked.connect(self.canvas_a.fit_to_view)
        self.btn_fit_b.clicked.connect(self.canvas_b.fit_to_view)

        self.canvas_a.pointsChanged.connect(self.update_point_counts)
        self.canvas_b.pointsChanged.connect(self.update_point_counts)

        self.btn_add_frame.clicked.connect(self.add_current_frame)
        self.btn_clear_frames.clicked.connect(self.clear_collected_frames)
        self.btn_calc_single.clicked.connect(self.calculate_single)
        self.btn_calc_multi.clicked.connect(self.calculate_multi)

    # ================= 專案目錄與設定 =================
    def set_target_dir(self, path):
        self.target_dir = os.path.abspath(path)
        self.lbl_dir.setText(f"專案目錄: {self.target_dir}")
        self.lbl_dir.setStyleSheet("color: #4ade80; font-weight: bold;")

    def select_target_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "選取專案目錄", self.target_dir or "")
        if dir_path:
            self.set_target_dir(dir_path)

    def update_point_counts(self):
        c_a = len(self.canvas_a.points)
        c_b = len(self.canvas_b.points)
        self.lbl_info_a.setText(f"點數: {c_a} / 15 {'✓' if c_a == 15 else ''}")
        self.lbl_info_b.setText(f"點數: {c_b} / 15 {'✓' if c_b == 15 else ''}")
        self.lbl_info_a.setStyleSheet("color: #4ade80;" if c_a == 15 else "color: #f87171;")
        self.lbl_info_b.setStyleSheet("color: #4ade80;" if c_b == 15 else "color: #f87171;")

    # ================= 核心亮點: 快速從 Visualized 對照圖載入 =================
    def quick_load_visualized(self, match_path=None):
        """
        選取 extrinsics/visualized 中的 match_... 圖檔，
        自動解析相機配對與影格，定位原始雙圖與內參，並自動執行初始角點辨識。
        """
        path = match_path
        if not path:
            start_dir = ""
            if self.target_dir:
                cand = os.path.join(self.target_dir, "extrinsics", "visualized")
                if os.path.exists(cand): start_dir = cand
                else: start_dir = self.target_dir
            else:
                default_vis = r"D:\Pitt\Project\Squat_Project\video\benchpress_3D\extrinsics\visualized"
                if os.path.exists(default_vis): start_dir = default_vis

            path, _ = QFileDialog.getOpenFileName(
                self, "選取 Visualized 對照圖", start_dir, "Match Images (match_*.jpg *.png)"
            )
            if not path:
                return

        filename = os.path.basename(path)
        # 解析 match_{cam1}_{cam2}_{frame_info}.jpg
        m = re.match(r"match_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(.+)", filename)
        if not m:
            QMessageBox.warning(self, "格式不符", f"無法從檔名辨識相機配對: {filename}\n標準格式應為 match_cam1_cam2_frameXXXXX.jpg")
            return

        c1, c2, rest = m.groups()
        self.pair_id = (c1, c2)
        self.canvas_a.cam_id = c1
        self.canvas_b.cam_id = c2

        # 提取 frame 編號 (例如 frame00006)
        fm = re.search(r"(frame\d+)", rest)
        ftag = fm.group(1) if fm else ""
        self.active_frame_name = rest

        # 推算專案根目錄
        vis_parent = os.path.abspath(os.path.join(os.path.dirname(path), "..", ".."))
        root_dir = self.target_dir or vis_parent
        if not os.path.exists(root_dir):
            root_dir = r"D:\Pitt\Project\Squat_Project\video\benchpress_3D"
        self.set_target_dir(root_dir)

        # 搜尋相機 1 與相機 2 的原圖
        def find_camera_frame(cam, tag):
            candidates = glob.glob(os.path.join(root_dir, cam, "**", f"*{tag}*.jpg"), recursive=True)
            if not candidates:
                candidates = glob.glob(os.path.join(root_dir, f"{cam}_jpg", "**", f"*{tag}*.jpg"), recursive=True)
            return candidates[0] if candidates else None

        img1_path = find_camera_frame(c1, ftag)
        img2_path = find_camera_frame(c2, ftag)

        if not img1_path or not img2_path:
            QMessageBox.warning(
                self, "找不到原始圖檔", 
                f"未能自動在 {root_dir} 找到 {c1} 或 {c2} 的 {ftag} 圖檔。\n請改用手動載入左右相機圖檔。"
            )
            return

        # 載入雙圖
        ok1 = self.canvas_a.load_image(img1_path, cam_id=c1)
        ok2 = self.canvas_b.load_image(img2_path, cam_id=c2)

        if ok1 and ok2:
            # 依據對向相機規則自動設定反轉核取方塊
            flip_a = should_camera_flip(c1, c2)
            flip_b = should_camera_flip(c2, c1)

            self.chk_flip_a.blockSignals(True)
            self.chk_flip_b.blockSignals(True)
            self.chk_flip_a.setChecked(flip_a)
            self.chk_flip_b.setChecked(flip_b)
            self.chk_flip_a.blockSignals(False)
            self.chk_flip_b.blockSignals(False)

            # 自動執行角點辨識 (傳入反轉參數)
            self.canvas_a.auto_detect_corners(partner_cam=c2, force_flip=flip_a)
            self.canvas_b.auto_detect_corners(partner_cam=c1, force_flip=flip_b)

            # 自動尋找內參
            self._auto_locate_intrinsics(root_dir, c1, c2)

            self.statusBar().showMessage(f"成功載入 {c1} <-> {c2} 影格: {ftag}！角點已自動標註，對向相機反轉已配置完成。")

    def _auto_locate_intrinsics(self, root_dir, c1, c2):
        """自動尋找兩相機內參"""
        cand_dirs = [
            os.path.join(root_dir, "intrinsics"),
            os.path.join(root_dir, "intrinsics_vision"),
            root_dir
        ]
        self.intrinsic_path1 = None
        self.intrinsic_path2 = None
        for d in cand_dirs:
            p1 = os.path.join(d, f"intrinsics_{c1}.npz")
            p2 = os.path.join(d, f"intrinsics_{c2}.npz")
            if os.path.exists(p1) and not self.intrinsic_path1:
                self.intrinsic_path1 = p1
            if os.path.exists(p2) and not self.intrinsic_path2:
                self.intrinsic_path2 = p2

    def manual_load_image(self, canvas: DraggableStereoCanvas):
        """手動選取單張影像"""
        start = self.target_dir or ""
        path, _ = QFileDialog.getOpenFileName(self, f"選取 {canvas.title} 影像", start, "Images (*.jpg *.png)")
        if path:
            if canvas.load_image(path):
                is_a = (canvas == self.canvas_a)
                other_cam = self.canvas_b.cam_id if is_a else self.canvas_a.cam_id
                do_flip = should_camera_flip(canvas.cam_id, other_cam)
                chk = self.chk_flip_a if is_a else self.chk_flip_b
                chk.blockSignals(True)
                chk.setChecked(do_flip)
                chk.blockSignals(False)
                canvas.auto_detect_corners(partner_cam=other_cam, force_flip=do_flip)
                self.update_point_counts()

    # ================= 多影格收集管理 =================
    def add_current_frame(self):
        p1, p2 = self.canvas_a.points, self.canvas_b.points
        if len(p1) != 15 or len(p2) != 15:
            QMessageBox.warning(self, "警告", "兩相機角點數需皆為完整的 15 點才能加入校正清單！")
            return
        
        fname = self.active_frame_name or (os.path.basename(self.canvas_a.image_path) if self.canvas_a.image_path else f"frame_{len(self.collected_frames)+1}")
        self.collected_frames.append({
            "p1": [pt.copy() for pt in p1],
            "p2": [pt.copy() for pt in p2],
            "frame": fname
        })
        self.lbl_collected.setText(f"已收集影格: {len(self.collected_frames)} 組")
        self.statusBar().showMessage(f"已加入影格 {fname} (目前累計 {len(self.collected_frames)} 組)")

    def clear_collected_frames(self):
        self.collected_frames = []
        self.lbl_collected.setText("已收集影格: 0 組")
        self.statusBar().showMessage("校正清單已清空。")

    # ================= 外參校正計算 =================
    def _prepare_intrinsics(self):
        """確認並載入兩相機內參"""
        c1 = self.canvas_a.cam_id or "A"
        c2 = self.canvas_b.cam_id or "B"

        if not self.intrinsic_path1 or not os.path.exists(self.intrinsic_path1):
            p1, _ = QFileDialog.getOpenFileName(self, f"選取相機 {c1} 內參 (.npz)", self.target_dir or "", "NPZ Files (*.npz)")
            if not p1: return None
            self.intrinsic_path1 = p1

        if not self.intrinsic_path2 or not os.path.exists(self.intrinsic_path2):
            p2, _ = QFileDialog.getOpenFileName(self, f"選取相機 {c2} 內參 (.npz)", self.target_dir or "", "NPZ Files (*.npz)")
            if not p2: return None
            self.intrinsic_path2 = p2

        try:
            d1 = np.load(self.intrinsic_path1)
            d2 = np.load(self.intrinsic_path2)
            m1, dist1 = d1["mtx"], d1["dist"]
            m2, dist2 = d2["mtx"], d2["dist"]
            return m1, dist1, m2, dist2, c1, c2
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"讀取內參失敗: {e}")
            return None

    def calculate_single(self):
        """僅使用當前畫布中的一組點位計算外參"""
        p1, p2 = self.canvas_a.points, self.canvas_b.points
        if len(p1) != len(p2) or len(p1) < 6:
            QMessageBox.warning(self, "警告", "兩邊點數需相同且至少 6 點以上 (建議點滿 15 點)")
            return

        res = self._prepare_intrinsics()
        if not res: return
        m1, dist1, m2, dist2, c1, c2 = res

        # 影像尺寸動態取得
        img_w = self.canvas_a.qimg.width() if self.canvas_a.qimg else 1280
        img_h = self.canvas_a.qimg.height() if self.canvas_a.qimg else 720

        objp = np.zeros((len(p1), 3), np.float32)
        grid = np.zeros((5 * 3, 3), np.float32)
        grid[:, :2] = np.mgrid[0:5, 0:3].T.reshape(-1, 2) * 25.0
        obj_pts = [grid[:len(p1)]]

        img_pts1 = [np.array(p1, dtype=np.float32).reshape(-1, 1, 2)]
        img_pts2 = [np.array(p2, dtype=np.float32).reshape(-1, 1, 2)]

        self._execute_stereo_calibrate(obj_pts, img_pts1, img_pts2, m1, dist1, m2, dist2, (img_w, img_h), c1, c2, num_frames=1)

    def calculate_multi(self):
        """使用所有已收集的影格計算外參"""
        if not self.collected_frames:
            QMessageBox.warning(self, "提示", "目前收集清單為空！請先微調並點擊「加入清單」，或使用「單張計算」。")
            return

        res = self._prepare_intrinsics()
        if not res: return
        m1, dist1, m2, dist2, c1, c2 = res

        img_w = self.canvas_a.qimg.width() if self.canvas_a.qimg else 1280
        img_h = self.canvas_a.qimg.height() if self.canvas_a.qimg else 720

        grid = np.zeros((15, 3), np.float32)
        grid[:, :2] = np.mgrid[0:5, 0:3].T.reshape(-1, 2) * 25.0

        obj_pts, img_pts1, img_pts2 = [], [], []
        for item in self.collected_frames:
            obj_pts.append(grid)
            img_pts1.append(np.array(item["p1"], dtype=np.float32).reshape(-1, 1, 2))
            img_pts2.append(np.array(item["p2"], dtype=np.float32).reshape(-1, 1, 2))

        self._execute_stereo_calibrate(obj_pts, img_pts1, img_pts2, m1, dist1, m2, dist2, (img_w, img_h), c1, c2, num_frames=len(self.collected_frames))

    def _execute_stereo_calibrate(self, obj_pts, img_pts1, img_pts2, m1, dist1, m2, dist2, img_size, c1, c2, num_frames):
        """執行 OpenCV 立體標定並儲存結果與新對照圖"""
        flags = cv2.CALIB_FIX_INTRINSIC
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_pts, img_pts1, img_pts2,
            m1, dist1, m2, dist2, img_size, flags=flags
        )

        baseline_mm = float(np.linalg.norm(T))

        # 決定外參存檔路徑
        out_dir = self.target_dir or "."
        ext_dir = os.path.join(out_dir, "extrinsics")
        os.makedirs(ext_dir, exist_ok=True)
        
        save_name = f"extrinsics_{c1}_to_{c2}.npz"
        out_path1 = os.path.join(ext_dir, save_name)
        out_path2 = os.path.join(out_dir, save_name)

        # 存入包含所有必要屬性的 .npz (與 step3.2 格式完全相容)
        save_dict = {
            "R": R,
            "T": T,
            "E": E,
            "F": F,
            "rms_error": ret,
            "baseline_mm": baseline_mm,
            "common_frames": num_frames
        }
        np.savez(out_path1, **save_dict)
        if out_path1 != out_path2:
            np.savez(out_path2, **save_dict)

        # 自動重新產生並覆蓋視覺化對照圖
        vis_msg = ""
        try:
            if save_stereo_visualization and self.canvas_a.cv_img is not None and self.canvas_b.cv_img is not None:
                vis_dir = os.path.join(ext_dir, "visualized")
                os.makedirs(vis_dir, exist_ok=True)
                frame_tag = self.active_frame_name or os.path.basename(self.canvas_a.image_path) or "manual_frame.jpg"
                c1_pts = np.array(self.canvas_a.points, dtype=np.float32).reshape(-1, 1, 2)
                c2_pts = np.array(self.canvas_b.points, dtype=np.float32).reshape(-1, 1, 2)
                
                saved_vis = save_stereo_visualization(
                    self.canvas_a.cv_img, self.canvas_b.cv_img,
                    c1_pts, c2_pts, c1, c2, frame_tag, vis_dir,
                    pattern_size=(5, 3), is_manual=True
                )
                vis_msg = f"\n\n📸 視覺化對照圖已更新儲存至:\n{saved_vis}"
        except Exception as e:
            print(f"[WARN] 儲存對照圖失敗: {e}")

        quality = "🌟 非常優秀 (< 0.5 px)" if ret < 0.5 else ("✓ 良好 (< 1.0 px)" if ret < 1.0 else "⚠️ 偏高 (> 1.0 px，建議再次檢查角點順序與對位)")
        QMessageBox.information(
            self, "外參校正成功！",
            f"相機配對: {c1} -> {c2}\n"
            f"使用影格組數: {num_frames} 組\n\n"
            f"重投影誤差 (RMS): {ret:.4f} pixels ({quality})\n"
            f"相機基線距離 (Baseline): {baseline_mm:.1f} mm ({baseline_mm/10:.1f} cm)\n\n"
            f"平移向量 T (mm):\n{T.ravel()}\n\n"
            f"外參檔案已儲存至:\n-> {out_path1}{vis_msg}"
        )
        self.statusBar().showMessage(f"校正完成！RMS: {ret:.4f} px | 基線: {baseline_mm:.1f} mm")

    def show_help(self):
        msg = (
            "【操作指南】\n\n"
            "1. 快速載入對照圖:\n"
            "   點擊頂部「從 Visualized 對照圖載入」，選取 match_*.jpg，系統會自動定位雙機原圖、內參並帶入初始角點。\n\n"
            "2. 角點微調 (拖曳與鍵盤):\n"
            "   - 滑鼠懸停至角點圓圈，左鍵按住即可自由拖曳移動。\n"
            "   - 左鍵點選角點後，可用鍵盤方向鍵 (Up/Down/Left/Right) 進行 1 pixel 微調（Shift 為 5 pixel）。\n"
            "   - 右上角設有局部放大鏡 HUD，顯示十字準心與交點像素。\n\n"
            "3. 畫面檢視:\n"
            "   - 滑鼠滾輪: 以游標為中心放大/縮小。\n"
            "   - 滑鼠中鍵或右鍵拖曳: 平移畫面視野。\n\n"
            "4. 實用工具:\n"
            "   - [⚡ 自動辨識]: 重新針對當前畫面偵測 15 個角點。\n"
            "   - [🧲 次像素吸附]: 自動吸附至精確像素幾何鞍點。\n"
            "   - [🔄 翻轉點序]: 180度倒轉 1~15 點順序（解決對向鏡頭倒轉問題）。\n\n"
            "5. 計算外參:\n"
            "   - [單張計算]: 直接以目前畫布微調好的點位計算立體外參。\n"
            "   - [加入清單 / 多影格計算]: 支援微調多張不同影格後合併計算更精準的外參。"
        )
        QMessageBox.information(self, "操作指南", msg)


if __name__ == "__main__":
    # ================= 預設專案路徑 =================
    DEFAULT_PROJECT_DIR = r"D:\Pitt\Project\Squat_Project\video\benchpress_3D"
    # ===============================================

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = StereoCalibWindow()
    if os.path.exists(DEFAULT_PROJECT_DIR):
        window.set_target_dir(DEFAULT_PROJECT_DIR)
    elif len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        window.set_target_dir(sys.argv[1])

    window.show()
    sys.exit(app.exec_())
