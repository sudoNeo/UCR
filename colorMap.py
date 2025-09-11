import sys
import math
import re
from collections import Counter

import numpy as np

def load_xy_from_table1(path="table1.txt"):
    # A = col 0 (Y), B = col 1 (X); skip the header row
    y, x = np.genfromtxt(
        path,
        dtype=float,
        skip_header=1,
        usecols=(0, 1),
        unpack=True,
        autostrip=True,
        invalid_raise=False,  # be tolerant if a weird row appears
    )
    return y, x

def load_ragged_numeric_matrix(path="table2.txt"):
    rows, widths = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)   # spaces/tabs
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    pass
            if vals:
                rows.append(vals)
                widths.append(len(vals))
    if not rows:
        raise ValueError("table2: no numeric rows found")
    modal_w, _ = Counter(widths).most_common(1)[0]
    good = [r for r in rows if len(r) == modal_w]
    A = np.asarray(good, dtype=np.float64)
    # Ensure (ny, nx) with ny << nx, e.g., (181, 25999)
    if A.shape[0] > A.shape[1]:
        A = A.T
    return A

# ---------------------------
# 2) UI (PyQtGraph) IMPLEMENTATION
# ---------------------------
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal
except ImportError:
    from PySide6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.Signal

import pyqtgraph as pg
import pyqtgraph.exporters  # for PNG export

class AdvancedControlPad(QtWidgets.QWidget):
    """
    Enhanced interactive pad for brightness & contrast control with visual feedback.
    """
    changeRequested = Signal(float, float)  # (contrast_factor, brightness_delta)
    
    def __init__(self, parent=None, kc=-0.01, kb=0.002, span=1.0, accent_color=None):
        super().__init__(parent)
        self.setMinimumSize(250, 250)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        
        self.kc = kc
        self.kb = kb
        self.span = span
        
        # State tracking
        self._dragging = False
        self._button = None
        self._x0 = 0
        self._y0 = 0
        
        # Position markers (0-1 normalized)
        self._u = 0.5  # horizontal (brightness)
        self._v = 0.5  # vertical (contrast)
        
        # Current absolute values for display
        self.current_contrast = 1.0
        self.current_brightness = 0.0
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        self._hover_pos = None
        self.accent = accent_color or QtGui.QColor(210, 210, 210)           # neutral gray
        self.accent_dim = QtGui.QColor(self.accent.red(), self.accent.green(), self.accent.blue(), 120)
        self.accent_faint = QtGui.QColor(self.accent.red(), self.accent.green(), self.accent.blue(), 60)
        
    def sizeHint(self):
        return QtCore.QSize(250, 250)
    
    def updateValues(self, contrast, brightness):
        """Update displayed values from external source"""
        self.current_contrast = contrast
        self.current_brightness = brightness
        
        # Update marker position based on values
        # Brightness: linear mapping
        self._u = np.clip(0.5 + brightness / max(1e-9, self.span), 0.0, 1.0)
        
        # Contrast: inverse exponential mapping
        if contrast > 0:
            # Map contrast back to vertical position
            self._v = np.clip(0.5 - math.log(contrast) / (2 * max(1e-9, abs(self.kc))), 0.0, 1.0)
        
        self.update()
    
    def mousePressEvent(self, event):
        if event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            self._dragging = True
            self._button = event.button()
            self._x0, self._y0 = event.x(), event.y()
            self._updateMarker(event.x(), event.y())
            self.update()
    
    def mouseMoveEvent(self, event):
        self._hover_pos = (event.x(), event.y())
        
        if not self._dragging:
            self.update()  # Update hover effect
            return
        
        dx = event.x() - self._x0
        dy = event.y() - self._y0
        
        contrast_factor = 1.0
        brightness_delta = 0.0
        
        if self._button == QtCore.Qt.LeftButton:
            # Left drag: both contrast (vertical) and brightness (horizontal)
            contrast_factor = math.exp(self.kc * dy)
            brightness_delta = self.kb * dx * self.span
        elif self._button == QtCore.Qt.RightButton:
            # Right drag: brightness only (vertical)
            brightness_delta = -self.kb * dy * self.span
        
        self._x0, self._y0 = event.x(), event.y()
        self._updateMarker(event.x(), event.y())
        
        # Update current values for display
        self.current_contrast *= contrast_factor
        self.current_brightness += brightness_delta
        
        self.update()
        self.changeRequested.emit(contrast_factor, brightness_delta)
    
    def mouseReleaseEvent(self, event):
        self._dragging = False
        self._button = None
        self.update()
    
    def mouseDoubleClickEvent(self, event):
        """Reset on double-click"""
        self._u = 0.5
        self._v = 0.5
        self.current_contrast = 1.0
        self.current_brightness = 0.0
        self.update()
        self.changeRequested.emit(1.0, 0.0)  # Ensure exact reset
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else (1 / 0.9)
        self.current_contrast *= factor
        self.changeRequested.emit(factor, 0.0)
        
        # Visual feedback: nudge marker
        self._v = np.clip(self._v + (-0.02 if delta > 0 else 0.02), 0.0, 1.0)
        self.update()
    
    def leaveEvent(self, event):
        self._hover_pos = None
        self.update()
    
    def _updateMarker(self, x, y):
        w = max(1, self.width())
        h = max(1, self.height())
        self._u = np.clip(x / w, 0.0, 1.0)
        self._v = np.clip(y / h, 0.0, 1.0)
    
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect()
        
        # Background gradient (subtle)
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QtGui.QColor(40, 40, 45))
        gradient.setColorAt(1, QtGui.QColor(25, 25, 30))
        p.fillRect(rect, QtGui.QBrush(gradient))
        
        w, h = rect.width(), rect.height()
        
        # Grid lines
        pen_grid = QtGui.QPen(QtGui.QColor(60, 60, 65), 1, QtCore.Qt.DotLine)
        p.setPen(pen_grid)
        for i in range(1, 10):
            if i == 5:
                continue
            x = rect.left() + i * w / 10.0
            y = rect.top() + i * h / 10.0
            p.drawLine(int(x), rect.top(), int(x), rect.bottom())
            p.drawLine(rect.left(), int(y), rect.right(), int(y))
        
        # Center lines
        pen_center = QtGui.QPen(QtGui.QColor(100, 100, 110), 2, QtCore.Qt.SolidLine)
        p.setPen(pen_center)
        cx_line = rect.left() + w / 2
        cy_line = rect.top() + h / 2
        p.drawLine(int(cx_line), rect.top(), int(cx_line), rect.bottom())
        p.drawLine(rect.left(), int(cy_line), rect.right(), int(cy_line))
        
        # Hover effect
        if self._hover_pos and not self._dragging:
            hx, hy = self._hover_pos
            pen_hover = QtGui.QPen(QtGui.QColor(150, 150, 160, 50), 1)
            p.setPen(pen_hover)
            p.drawLine(hx, rect.top(), hx, rect.bottom())
            p.drawLine(rect.left(), hy, rect.right(), hy)
        
        # Current position marker
        cx = rect.left() + int(self._u * w)
        cy = rect.top() + int(self._v * h)
        
        # Crosshair lines
        p.setPen(QtGui.QPen(self.accent_faint, 1))
        p.drawLine(cx, rect.top(), cx, rect.bottom())
        p.drawLine(rect.left(), cy, rect.right(), cy)
        
        # Marker circle
        pen_marker = QtGui.QPen(self.accent, 2)
        brush_marker = QtGui.QBrush(QtGui.QColor(self.accent.red(), self.accent.green(), self.accent.blue(), 200))
        p.setPen(pen_marker)
        p.setBrush(brush_marker)
        p.drawEllipse(QtCore.QPoint(cx, cy), 6, 6)
        
        # Labels
        p.setPen(QtGui.QColor(200, 200, 210))
        font = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
        p.setFont(font)
        p.drawText(rect.adjusted(8, 8, -8, -8),
                  QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft,
                  " CONTROL PAD")
        p.drawText(rect.adjusted(8, -25, -8, -8),
                  QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft,
                  "← Brightness →")
        p.save()
        p.translate(15, rect.center().y())
        p.rotate(-90)
        p.drawText(QtCore.QRect(-40, -10, 80, 20),
                  QtCore.Qt.AlignCenter,
                  "Contrast")
        p.restore()
        
        # Current values box
        font.setPointSize(9)
        p.setFont(font)
        value_rect = QtCore.QRect(rect.right() - 120, rect.bottom() - 45, 110, 35)
        p.fillRect(value_rect, QtGui.QColor(20, 20, 25, 200))
        p.setPen(QtGui.QColor(100, 100, 110))
        p.drawRect(value_rect)
        p.setPen(QtGui.QColor(255, 255, 255))
        p.drawText(value_rect.adjusted(5, 3, -5, -3),
                  QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
                  f"C: {self.current_contrast:.3f}\nB: {self.current_brightness:+.3f}")
        
        # Border
        pen_border = QtGui.QPen(QtGui.QColor(80, 80, 90), 2)
        p.setPen(pen_border)
        p.drawRect(rect.adjusted(1, 1, -1, -1))


class EnhancedBCWindow(QtWidgets.QMainWindow):
    """
    Enhanced brightness/contrast window with pyqtgraph, now using table1/table2 extents.
    """
    def __init__(self, data, x_range, y_range, cmap='viridis', vmin=None, vmax=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Heatmap with Brightness/Contrast (table1/table2)")
        self.resize(1200, 750)
        
        # Dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: #e0e0e0; }
            QPushButton {
                background-color: #3a3a3a; color: #e0e0e0;
                border: 1px solid #555; padding: 6px 12px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:pressed { background-color: #2a2a2a; }
        """)
        
        self.data = np.asarray(data)
        # If vmin/vmax not provided, derive from data (like matplotlib colorbar)
        if vmin is None: vmin = float(np.nanmin(self.data))
        if vmax is None: vmax = float(np.nanmax(self.data))
        self.base_vmin = float(vmin)
        self.base_vmax = float(vmax)
        self.span = max(1e-12, self.base_vmax - self.base_vmin)
        
        self.contrast = 1.0
        self.brightness = 0.0
        self.kc = -0.01
        self.kb = 0.002

        self.xmin, self.xmax = map(float, x_range)
        self.ymin, self.ymax = map(float, y_range)
        
        # Main layout
        cw = QtWidgets.QWidget(self)
        self.setCentralWidget(cw)
        main_layout = QtWidgets.QHBoxLayout(cw)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Left panel
        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setSpacing(10)
        
        title_label = QtWidgets.QLabel("Interactive Controls")
        title_font = QtGui.QFont("Arial", 12, QtGui.QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        left_panel.addWidget(title_label)
        
        self.pad = AdvancedControlPad(kc=self.kc, kb=self.kb, span=self.span)
        self.pad.changeRequested.connect(self._apply_pad_change)
        left_panel.addWidget(self.pad, 0, QtCore.Qt.AlignTop)
        
        instructions = QtWidgets.QLabel(
            "Controls:\n"
            "• Left-drag: Adjust both axes\n"
            "• Right-drag: Brightness only\n"
            "• Scroll: Fine-tune contrast\n"
            "• Double-click: Reset to default\n"
            "• R key: Reset values\n\n"
            "Plot interactions:\n"
            "• Scroll on plot: Contrast\n"
            "• Drag on plot: Pan view"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-size: 10px; color: #c0c0c0; padding: 10px;")
        left_panel.addWidget(instructions)
        
        reset_btn = QtWidgets.QPushButton("Reset All Values")
        reset_btn.clicked.connect(self.reset_values)
        left_panel.addWidget(reset_btn)
        
        self.stats_label = QtWidgets.QLabel()
        self.stats_label.setStyleSheet("font-size: 10px; color: #a0a0a0; padding: 5px;")
        self._update_stats()
        left_panel.addWidget(self.stats_label)
        
        left_panel.addStretch()
        main_layout.addLayout(left_panel)

        # Right panel with pyqtgraph
        right_panel = QtWidgets.QVBoxLayout()
        
        self.info = QtWidgets.QLabel(self._fmt_info())
        info_font = QtGui.QFont("Consolas", 11)
        self.info.setFont(info_font)
        self.info.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                color: #00ff88;
                padding: 8px;
                border: 1px solid #404040;
                border-radius: 4px;
            }
        """)
        right_panel.addWidget(self.info)
        
        self.graph = pg.GraphicsLayoutWidget()
        self.graph.setBackground('#1e1e1e')
        right_panel.addWidget(self.graph, 1)
        
        # Plot
        self.plot = self.graph.addPlot(row=0, col=0)
        self.plot.setAspectLocked(False)
        self.plot.setLabel('left', 'Y (table1 Column A)')
        self.plot.setLabel('bottom', 'X (table1 Column B)')
        
        # Image item; flip vertically to mimic matplotlib origin="lower"
        img_data = self.data
        self.img_item = pg.ImageItem(img_data, axisOrder='row-major')
        self.plot.addItem(self.img_item)

        # Map image to real-world X/Y using setRect
        rect = QtCore.QRectF(self.xmin, self.ymin, (self.xmax - self.xmin), (self.ymax - self.ymin))
        self.img_item.setRect(rect)

        # Nice initial view
        self.plot.setXRange(self.xmin, self.xmax, padding=0.0)
        self.plot.setYRange(self.ymin, self.ymax, padding=0.0)
        
        # Colormap and colorbar
        try:
            cm = pg.colormap.get(cmap)
        except Exception:
            cm = pg.colormap.get('viridis')
        self.lut = cm.getLookupTable(0.0, 1.0, 256)
        self.img_item.setLookupTable(self.lut)
        
        self.cbar = pg.ColorBarItem(
            values=(self.base_vmin, self.base_vmax),
            colorMap=cm,
            width=20
        )
        self.cbar.setImageItem(self.img_item)
        self.graph.addItem(self.cbar, row=0, col=1)
        
        # Interactions
        self.plot.scene().sigMouseClicked.connect(self._on_plot_click)
        
        # Add right panel
        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_panel)
        main_layout.addWidget(right_container, 1)
        
        self._apply_levels()
    
    # ---- helpers ----
    def _fmt_info(self):
        vmin_p = (self.base_vmin - self.brightness) / max(1e-12, self.contrast)
        vmax_p = (self.base_vmax - self.brightness) / max(1e-12, self.contrast)
        return (f"Contrast: {self.contrast:.3f}  |  "
                f"Brightness: {self.brightness:+.3f}  |  "
                f"Effective Range: [{vmin_p:.3f}, {vmax_p:.3f}]")
    
    def _update_stats(self):
        if self.data.size > 0:
            stats_text = (
                f"Data Statistics:\n"
                f"Shape: {self.data.shape}\n"
                f"Min: {np.nanmin(self.data):.6g}\n"
                f"Max: {np.nanmax(self.data):.6g}\n"
                f"Mean: {np.nanmean(self.data):.6g}\n"
                f"Std: {np.nanstd(self.data):.6g}"
            )
            self.stats_label.setText(stats_text)
    
    def _apply_levels(self):
        """Apply brightness and contrast adjustments"""
        vmin_prime = (self.base_vmin - self.brightness) / max(1e-12, self.contrast)
        vmax_prime = (self.base_vmax - self.brightness) / max(1e-12, self.contrast)
        if vmin_prime > vmax_prime:
            vmin_prime, vmax_prime = vmax_prime, vmin_prime
        
        self.img_item.setLevels((vmin_prime, vmax_prime))
        self.cbar.setLevels((vmin_prime, vmax_prime))
        self.info.setText(self._fmt_info())
    
    def _apply_pad_change(self, contrast_factor, brightness_delta):
        """Handle changes from the control pad"""
        self.contrast = max(0.01, self.contrast * float(contrast_factor))
        self.brightness += float(brightness_delta)
        self._apply_levels()
        self.pad.updateValues(self.contrast, self.brightness)
    
    def _on_plot_click(self, event):
        if event.double():
            self.reset_values()
    
    def reset_values(self):
        self.contrast = 1.0
        self.brightness = 0.0
        self.pad.updateValues(self.contrast, self.brightness)
        self._apply_levels()
    
    def wheelEvent(self, event):
        if self.graph.geometry().contains(event.pos()):
            delta = event.angleDelta().y()
            factor = 0.9 if delta > 0 else (1 / 0.9)
            self.contrast = max(0.01, self.contrast * factor)
            self.pad.updateValues(self.contrast, self.brightness)
            self._apply_levels()
            event.accept()
        else:
            super().wheelEvent(event)
    
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_R:
            self.reset_values()
            event.accept()
        elif event.key() == QtCore.Qt.Key_S:
            exporter = pg.exporters.ImageExporter(self.plot)
            exporter.export('brightness_contrast_view.png')
            self.info.setText("View saved to brightness_contrast_view.png")
            event.accept()
        else:
            super().keyPressEvent(event)


def main():
    # ---- Load data just like your matplotlib script ----
    y, x = load_xy_from_table1("table1.txt")
    Z = load_ragged_numeric_matrix("table2.txt")
    # extent = [xmin, xmax, ymin, ymax]
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    
    # Create the app/window
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        created_app = True
    
    # Initial vmin/vmax from data
    vmin = float(np.nanmin(Z))
    vmax = float(np.nanmax(Z))
    
    win = EnhancedBCWindow(
        data=Z,
        x_range=(xmin, xmax),
        y_range=(ymin, ymax),
        cmap='viridis',
        vmin=vmin,
        vmax=vmax
    )
    win.show()
    
    if created_app:
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
