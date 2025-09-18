import sys
import math
import re
import os
from collections import Counter
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading
import argparse
import numpy as np

# ---------------------------
# CONSTANTS
# ---------------------------
DEFAULT_XY_FILE = "table1.txt"
DEFAULT_Z_FILE = "table2.txt"
CONTROL_PAD_SIZE = 250
CONTROL_PAD_FPS = 60
MAIN_UPDATE_FPS = 30
DEFAULT_KC = -0.01
DEFAULT_KB = 0.002
DEFAULT_CMAP = 'viridis'

# ---------------------------
# 1) FILE I/O CLASS
# ---------------------------
class DataLoader:
    """Dedicated class for file I/O operations"""
    
    @staticmethod
    def detect_columns(path):
        """Detect the number and names of columns in a data file"""
        try:
            with open(path, 'r') as f:
                # Read header line
                header = f.readline().strip()
                
                # Try to parse header
                if header:
                    # Split by common delimiters
                    parts = re.split(r'[,\t\s]+', header)
                    # Filter out empty parts
                    parts = [p.strip() for p in parts if p.strip()]
                    
                    # Check if header contains numbers (likely not a header)
                    try:
                        float(parts[0])
                        # First line is data, not header
                        return len(parts), [f"Column {i}" for i in range(len(parts))]
                    except:
                        # First line is header
                        return len(parts), parts
                
                # If no header, count columns in first data line
                line = f.readline().strip()
                if line:
                    parts = re.split(r'[,\t\s]+', line)
                    parts = [p for p in parts if p.strip()]
                    return len(parts), [f"Column {i}" for i in range(len(parts))]
                
                return 0, []
        except:
            return 0, []
    
    @staticmethod
    def load_xy_data(path, x_col=1, y_col=0):
        """Load X/Y data with column selection"""
        try:
            # First detect the actual number of columns
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                # Skip header if present
                first_line = f.readline().strip()
                # Check header 
                try:
                    float(first_line.split()[0])
                    skiprows = 0
                except:
                    skiprows = 1
                if ',' in first_line:
                    delim = ','
                elif '\t' in first_line:
                    delim = '\t'
                else:
                    delim = None  # whitespace
            
            # Load all data
            data = np.loadtxt(path, skiprows=skiprows, dtype=np.float64, ndmin=2, delimiter=delim)
            if data.size == 0:
                raise ValueError("No data found in file")
            
            # Handle single column case
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Validate column indices
            num_cols = data.shape[1]
            if x_col >= num_cols:
                x_col = min(x_col, num_cols - 1)
            if y_col >= num_cols:
                y_col = min(y_col, num_cols - 1)
            
            return data[:, y_col], data[:, x_col]  # y, x
        except Exception as e:
            # Fallback to more flexible parsing
            try:
                y, x = np.genfromtxt(
                    path,
                    delimiter=delim,
                    dtype=np.float64,
                    skip_header=1,
                    usecols=(y_col, x_col),
                    unpack=True,
                    autostrip=True,
                    invalid_raise=False,
                )
                return y, x
            except:
                # If all else fails, try default columns
                y, x = np.genfromtxt(
                    path,
                    delimiter=delim,
                    dtype=np.float64,
                    skip_header=1,
                    usecols=(0, min(1, 0)),
                    unpack=True,
                    autostrip=True,
                    invalid_raise=False,
                )
                return y, x
    
    @staticmethod
    def load_matrix_data(path):
        """Optimized matrix loading with better memory usage"""
        # First pass: determine dimensions
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        
        if not lines:
            raise ValueError("No numeric rows found in file")
        
        first = lines[0]
        if ',' in first:
            sep = ','
            splitter = re.compile(r"\s*,\s*")
        elif '\t' in first:
            sep = '\t'
            splitter = re.compile(r"\t+")
        else:
            sep = ' '  # whitespace
            splitter = re.compile(r"\s+")

        rows = []
        for line in lines:
            try:
                row = np.fromstring(line, sep=sep, dtype=np.float64)
                if row.size:
                    rows.append(row)
                    continue
            except Exception:
                pass
            # Fallback parser (handles stray spaces/commas)
            parts = [p for p in splitter.split(line) if p]
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    pass
            if vals:
                rows.append(np.asarray(vals, dtype=np.float64))

        if not rows:
            raise ValueError("No numeric rows found in file")

        widths = np.array([r.size for r in rows], dtype=int)
        modal_w = np.bincount(widths).argmax()
        good_rows = [r for r in rows if r.size == modal_w]
        A = np.vstack(good_rows).astype(np.float64)
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
import pyqtgraph.exporters


class ColumnSelectionDialog(QtWidgets.QDialog):
    """Dialog for selecting X and Y columns from a data file"""
    
    def __init__(self, filepath, column_names, current_x=1, current_y=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Data Columns")
        self.setModal(True)
        self.resize(400, 200)
        
        # Store selections
        self.x_column = current_x
        self.y_column = current_y
        
        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # File info
        file_label = QtWidgets.QLabel(f"File: {os.path.basename(filepath)}")
        file_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(file_label)
        
        # Column selection
        grid = QtWidgets.QGridLayout()
        
        # X column
        grid.addWidget(QtWidgets.QLabel("X Column:"), 0, 0)
        self.x_combo = QtWidgets.QComboBox()
        for i, name in enumerate(column_names):
            self.x_combo.addItem(f"{i}: {name}")
        self.x_combo.setCurrentIndex(min(current_x, len(column_names) - 1))
        grid.addWidget(self.x_combo, 0, 1)
        
        # Y column
        grid.addWidget(QtWidgets.QLabel("Y Column:"), 1, 0)
        self.y_combo = QtWidgets.QComboBox()
        for i, name in enumerate(column_names):
            self.y_combo.addItem(f"{i}: {name}")
        self.y_combo.setCurrentIndex(min(current_y, len(column_names) - 1))
        grid.addWidget(self.y_combo, 1, 1)
        
        layout.addLayout(grid)
        
        # Spacer
        layout.addStretch()
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Connect signals
        self.x_combo.currentIndexChanged.connect(self._on_x_changed)
        self.y_combo.currentIndexChanged.connect(self._on_y_changed)
    
    def _on_x_changed(self, index):
        self.x_column = index
    
    def _on_y_changed(self, index):
        self.y_column = index
    
    def get_selections(self):
        """Return the selected column indices"""
        return self.x_column, self.y_column


class ControlPad(QtWidgets.QWidget):
    """
    Interactive control pad for brightness and contrast adjustment
    """
    changeRequested = Signal(float, float)  # (contrast_factor, brightness_delta)
    
    # Class constants
    MIN_MOVEMENT_THRESHOLD = 1
    WHEEL_FACTOR_UP = 0.9
    WHEEL_FACTOR_DOWN = 1.0 / 0.9
    MARKER_NUDGE = 0.02
    
    def __init__(self, parent=None, kc=DEFAULT_KC, kb=DEFAULT_KB, span=1.0, accent_color=None):
        super().__init__(parent)
        self.setMinimumSize(CONTROL_PAD_SIZE, CONTROL_PAD_SIZE)
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
        self._u = 0.5
        self._v = 0.5
        
        # Current absolute values for display
        self.current_contrast = 1.0
        self.current_brightness = 0.0
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        self._hover_pos = None
        self.accent = accent_color or QtGui.QColor(210, 210, 210)
        self.accent_dim = QtGui.QColor(self.accent.red(), self.accent.green(), self.accent.blue(), 120)
        self.accent_faint = QtGui.QColor(self.accent.red(), self.accent.green(), self.accent.blue(), 60)
        
        # Optimization: Update timer for batched redraws
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.timeout.connect(self._do_update)
        self._update_timer.setInterval(1000 // CONTROL_PAD_FPS)
        self._update_pending = False
        
        # Cache frequently used calculations
        self._cached_geometry = None
        self._cached_rects = {}
    
    def _cleanup(self):
        """Cleanup timer resources"""
        if hasattr(self, '_update_timer'):
            try:
                self._update_timer.stop()
                # Handle both PyQt5 and PySide6
                self._update_timer.deleteLater()
            except (RuntimeError, AttributeError):
                # Timer might already be deleted
                pass
    
    def _request_update(self):
        """Request an update, batching multiple requests"""
        if not self._update_pending:
            self._update_pending = True
            self._update_timer.start()
    
    def _do_update(self):
        """Perform the actual update"""
        self._update_timer.stop()
        self._update_pending = False
        self.update()
    
    def sizeHint(self):
        return QtCore.QSize(CONTROL_PAD_SIZE, CONTROL_PAD_SIZE)
    
    @lru_cache(maxsize=128)
    def _calculate_marker_position(self, brightness, contrast):
        """Cached calculation of marker position from values"""
        u = np.clip(0.5 + brightness / max(1e-9, self.span), 0.0, 1.0)
        v = 0.5
        if contrast > 0:
            v = np.clip(0.5 - math.log(contrast) / (2 * max(1e-9, abs(self.kc))), 0.0, 1.0)
        return u, v
    
    def updateValues(self, contrast, brightness):
        """Update displayed values from external source"""
        self.current_contrast = contrast
        self.current_brightness = brightness
        
        # Use cached calculation
        self._u, self._v = self._calculate_marker_position(brightness, contrast)
        self._request_update()
    
    def mousePressEvent(self, event):
        if event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            self._dragging = True
            self._button = event.button()
            self._x0, self._y0 = event.x(), event.y()
            self._updateMarker(event.x(), event.y())
            self.update()  # Immediate update on press
    
    def mouseMoveEvent(self, event):
        self._hover_pos = (event.x(), event.y())
        
        if not self._dragging:
            self._request_update()  # Batched update for hover
            return
        
        dx = event.x() - self._x0
        dy = event.y() - self._y0
        
        # Skip tiny movements
        if abs(dx) < self.MIN_MOVEMENT_THRESHOLD and abs(dy) < self.MIN_MOVEMENT_THRESHOLD:
            return
        
        contrast_factor = 1.0
        brightness_delta = 0.0
        
        if self._button == QtCore.Qt.LeftButton:
            contrast_factor = math.exp(self.kc * dy)
            brightness_delta = self.kb * dx * self.span
        elif self._button == QtCore.Qt.RightButton:
            brightness_delta = -self.kb * dy * self.span
        
        self._x0, self._y0 = event.x(), event.y()
        self._updateMarker(event.x(), event.y())
        
        # Update current values
        self.current_contrast *= contrast_factor
        self.current_brightness += brightness_delta
        
        # Clear cache when values change
        self._calculate_marker_position.cache_clear()
        
        self._request_update()
        self.changeRequested.emit(contrast_factor, brightness_delta)
    
    def mouseReleaseEvent(self, event):
        self._dragging = False
        self._button = None
        self.update()
    
    def mouseDoubleClickEvent(self, event):
        """Reset on double-click"""
        old_contrast = self.current_contrast
        old_brightness = self.current_brightness
        
        self._u = 0.5
        self._v = 0.5
        self.current_contrast = 1.0
        self.current_brightness = 0.0
        
        # Clear cache
        self._calculate_marker_position.cache_clear()
        
        self.update()
        # Emit the change needed to reset
        if old_contrast != 0:
            self.changeRequested.emit(1.0 / old_contrast, -old_brightness)
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = self.WHEEL_FACTOR_UP if delta > 0 else self.WHEEL_FACTOR_DOWN
        self.current_contrast *= factor
        self.changeRequested.emit(factor, 0.0)
        
        self._v = np.clip(self._v + (-self.MARKER_NUDGE if delta > 0 else self.MARKER_NUDGE), 0.0, 1.0)
        self._request_update()
    
    def leaveEvent(self, event):
        self._hover_pos = None
        self._request_update()
    
    def _updateMarker(self, x, y):
        w = max(1, self.width())
        h = max(1, self.height())
        self._u = np.clip(x / w, 0.0, 1.0)
        self._v = np.clip(y / h, 0.0, 1.0)
    
    def resizeEvent(self, event):
        """Cache geometry calculations on resize"""
        super().resizeEvent(event)
        self._cached_geometry = None
        self._cached_rects = {}
        # Clear position cache as span might affect calculations
        self._calculate_marker_position.cache_clear()
    
    def closeEvent(self, event):
        """Handle widget close event"""
        self._cleanup()
        super().closeEvent(event)
    
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect()
        
        # Cache geometry if needed
        if self._cached_geometry is None:
            self._cached_geometry = (rect.width(), rect.height())
            self._cached_rects = {
                'value': QtCore.QRect(rect.right() - 120, rect.bottom() - 45, 110, 35),
                'adjusted': rect.adjusted(1, 1, -1, -1),
                'text': rect.adjusted(8, 8, -8, -8),
                'bottom_text': rect.adjusted(8, -25, -8, -8)
            }
        
        w, h = self._cached_geometry
        
        # Background gradient
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QtGui.QColor(40, 40, 45))
        gradient.setColorAt(1, QtGui.QColor(25, 25, 30))
        p.fillRect(rect, QtGui.QBrush(gradient))
        
        # Grid lines (optimized loop)
        pen_grid = QtGui.QPen(QtGui.QColor(60, 60, 65), 1, QtCore.Qt.DotLine)
        p.setPen(pen_grid)
        
        # Use list comprehension for efficiency
        grid_lines = [(i, rect.left() + i * w / 10.0, rect.top() + i * h / 10.0) 
                      for i in range(1, 10) if i != 5]
        
        for i, x, y in grid_lines:
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
        brush_marker = QtGui.QBrush(QtGui.QColor(self.accent.red(), self.accent.green(), 
                                                 self.accent.blue(), 200))
        p.setPen(pen_marker)
        p.setBrush(brush_marker)
        p.drawEllipse(QtCore.QPoint(cx, cy), 6, 6)
        
        # Labels
        p.setPen(QtGui.QColor(200, 200, 210))
        font = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
        p.setFont(font)
        p.drawText(self._cached_rects['text'],
                  QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft,
                  " CONTROL PAD")
        p.drawText(self._cached_rects['bottom_text'],
                  QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft,
                  "← Brightness →")
        
        # Rotated text
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
        value_rect = self._cached_rects['value']
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
        p.drawRect(self._cached_rects['adjusted'])


class FileLoadThread(QtCore.QThread):
    """Thread for async file loading"""
    progress = Signal(int)
    finished = Signal(object)  # Will emit the loaded data
    error = Signal(str)
    
    def __init__(self, filename, load_func, parent=None, **kwargs):
        super().__init__(parent)
        self.filename = filename
        self.load_func = load_func
        self.kwargs = kwargs
    
    def run(self):
        try:
            self.progress.emit(30)
            data = self.load_func(self.filename, **self.kwargs)
            self.progress.emit(90)
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class InteractiveHeatmapWindow(QtWidgets.QMainWindow):
    """
    Interactive heatmap window with brightness/contrast controls and column selection
    """
    
    # Class constants
    DEFAULT_WIDTH = 1200
    DEFAULT_HEIGHT = 750
    LEVEL_UPDATE_INTERVAL = 1000 // MAIN_UPDATE_FPS
    PROGRESS_HIDE_DELAY = 500
    
    def __init__(self, data, x_range, y_range, cmap=DEFAULT_CMAP, vmin=None, vmax=None, 
                 xy_file=DEFAULT_XY_FILE, z_file=DEFAULT_Z_FILE, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Heatmap Viewer")
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        
        # Store file paths
        self.xy_file = xy_file
        self.z_file = z_file
        
        # Column selection
        self.x_column = 1  # Default: column 1 for X
        self.y_column = 0  # Default: column 0 for Y
        
        # Data loader
        self.data_loader = DataLoader()
        
        # Thread pool for concurrent operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Active loading threads
        self.active_threads = []
        
        # Apply dark theme
        self._apply_dark_theme()
        
        # Initialize data
        self.data = np.asarray(data, dtype=np.float64)
        # Replace any NaN values to avoid rendering issues
        self.data = np.nan_to_num(self.data, copy=False, nan=0.0)
        if vmin is None: vmin = float(np.nanmin(self.data))
        if vmax is None: vmax = float(np.nanmax(self.data))
        self.base_vmin = float(vmin)
        self.base_vmax = float(vmax)
        self.span = max(1e-12, self.base_vmax - self.base_vmin)
        
        self.contrast = 1.0
        self.brightness = 0.0
        self.kc = DEFAULT_KC
        self.kb = DEFAULT_KB

        self.xmin, self.xmax = map(float, x_range)
        self.ymin, self.ymax = map(float, y_range)
        
        # Optimization: Update timer for batched level updates
        self._level_update_timer = QtCore.QTimer(self)
        self._level_update_timer.timeout.connect(self._apply_levels_batched)
        self._level_update_timer.setInterval(self.LEVEL_UPDATE_INTERVAL)
        self._level_update_pending = False
        
        # Setup UI
        self._setup_ui()
    
    def _cleanup_resources(self):
        """Cleanup all resources safely"""
        # Stop timers
        if hasattr(self, '_level_update_timer'):
            try:
                self._level_update_timer.stop()
                # Handle both PyQt5 and PySide6
                self._level_update_timer.deleteLater()
            except (RuntimeError, AttributeError):
                # Timer might already be deleted
                pass
        
        # Stop threads
        if hasattr(self, 'active_threads'):
            threads_to_stop = list(self.active_threads)  # Make a copy
            for thread in threads_to_stop:
                try:
                    if thread.isRunning():
                        thread.quit()
                        if not thread.wait(1000):  # Wait max 1 second
                            thread.terminate()
                            thread.wait()
                except (RuntimeError, AttributeError):
                    # Thread might already be deleted
                    pass
            # Clear the list
            self.active_threads.clear()
        
        # Shutdown thread pool
        if hasattr(self, 'thread_pool'):
            try:
                self.thread_pool.shutdown(wait=True)
            except (RuntimeError, AttributeError):
                pass
    
    def _apply_dark_theme(self):
        """Apply dark theme styling"""
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: #e0e0e0; }
            QPushButton {
                background-color: #3a3a3a; color: #e0e0e0;
                border: 1px solid #555; padding: 6px 12px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:pressed { background-color: #2a2a2a; }
            QProgressBar {
                border: 1px solid #555; border-radius: 3px;
                text-align: center; background-color: #2a2a2a;
            }
            QProgressBar::chunk { background-color: #4a90e2; }
        """)
    
    def _setup_ui(self):
        """Setup UI components"""
        cw = QtWidgets.QWidget(self)
        self.setCentralWidget(cw)
        main_layout = QtWidgets.QHBoxLayout(cw)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Left panel
        left_panel = self._create_left_panel()
        main_layout.addLayout(left_panel)

        # Right panel with pyqtgraph
        right_panel = self._create_right_panel()
        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_panel)
        main_layout.addWidget(right_container, 1)
        
        self._apply_levels()
    
    def _create_left_panel(self):
        """Create left control panel"""
        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setSpacing(10)
        
        title_label = QtWidgets.QLabel("Interactive Controls")
        title_font = QtGui.QFont("Arial", 12, QtGui.QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        left_panel.addWidget(title_label)
        
        self.pad = ControlPad(kc=self.kc, kb=self.kb, span=self.span)
        self.pad.changeRequested.connect(self._apply_pad_change)
        left_panel.addWidget(self.pad, 0, QtCore.Qt.AlignTop)
        
        instructions = QtWidgets.QLabel(
            "Controls:\n"
            "• Left-drag: Adjust both axes\n"
            "• Right-drag: Brightness only\n"
            "• Scroll: Fine-tune contrast\n"
            "• Double-click: Reset to default\n"
            "• R key: Reset values\n"
            "• S key: Save image\n\n"
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
        
        # File selection buttons
        left_panel.addWidget(QtWidgets.QLabel(""))  # Spacer
        file_label = QtWidgets.QLabel("Data Files:")
        file_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #e0e0e0;")
        left_panel.addWidget(file_label)
        
        self.xy_btn = QtWidgets.QPushButton("Select X/Y File")
        self.xy_btn.setToolTip(f"Current: {self.xy_file}")
        self.xy_btn.clicked.connect(self.select_xy_file)
        left_panel.addWidget(self.xy_btn)
        
        self.z_btn = QtWidgets.QPushButton("Select Color Matrix File")
        self.z_btn.setToolTip(f"Current: {self.z_file}")
        self.z_btn.clicked.connect(self.select_z_file)
        left_panel.addWidget(self.z_btn)
        
        # Progress bar for file loading
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        left_panel.addWidget(self.progress_bar)
        
        self.stats_label = QtWidgets.QLabel()
        self.stats_label.setStyleSheet("font-size: 10px; color: #a0a0a0; padding: 5px;")
        self._update_stats()
        left_panel.addWidget(self.stats_label)
        
        left_panel.addStretch()
        return left_panel
    
    def _create_right_panel(self):
        """Create right visualization panel"""
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
        
        # Plot with optimizations
        self.plot = self.graph.addPlot(row=0, col=0)
        self.plot.setAspectLocked(False)
        self.plot.setLabel('left', 'Y (table1 Column A)')
        self.plot.setLabel('bottom', 'X (table1 Column B)')
        
        # Enable OpenGL for better performance with large datasets
        try:
            self.plot.enableGL()
        except:
            pass  # OpenGL not available
        
        # Image item with optimizations
        # Ensure no NaN values that could cause rendering issues
        clean_data = np.nan_to_num(self.data, copy=True, nan=0.0)
        self.img_item = pg.ImageItem(clean_data, axisOrder='row-major')
        self.img_item.setOpts(axisOrder='row-major')
        
        # Disable auto downsampling to avoid index errors
        self.img_item.setAutoDownsample(False)
        
        self.plot.addItem(self.img_item)

        # Map image to real-world X/Y
        rect = QtCore.QRectF(self.xmin, self.ymin, 
                            (self.xmax - self.xmin), 
                            (self.ymax - self.ymin))
        self.img_item.setRect(rect)

        # Set view
        self.plot.setXRange(self.xmin, self.xmax, padding=0.0)
        self.plot.setYRange(self.ymin, self.ymax, padding=0.0)
        
        # Colormap and colorbar
        try:
            cm = pg.colormap.get(DEFAULT_CMAP)
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
        
        return right_panel
    
    @lru_cache(maxsize=256)
    def _calculate_effective_range(self, vmin, vmax, brightness, contrast):
        """Cached calculation of effective range"""
        vmin_p = (vmin - brightness) / max(1e-12, contrast)
        vmax_p = (vmax - brightness) / max(1e-12, contrast)
        return vmin_p, vmax_p
    
    def _fmt_info(self):
        vmin_p, vmax_p = self._calculate_effective_range(
            self.base_vmin, self.base_vmax, self.brightness, self.contrast
        )
        return (f"Contrast: {self.contrast:.3f}  |  "
                f"Brightness: {self.brightness:+.3f}  |  "
                f"Effective Range: [{vmin_p:.3f}, {vmax_p:.3f}]")
    
    def _update_stats(self):
        if self.data.size > 0:
            # Use cached statistics if available
            if not hasattr(self, '_stats_cache'):
                self._stats_cache = {
                    'min': np.nanmin(self.data),
                    'max': np.nanmax(self.data),
                    'mean': np.nanmean(self.data),
                    'std': np.nanstd(self.data)
                }
            
            stats_text = (
                f"Data Statistics:\n"
                f"Shape: {self.data.shape}\n"
                f"Min: {self._stats_cache['min']:.6g}\n"
                f"Max: {self._stats_cache['max']:.6g}\n"
                f"Mean: {self._stats_cache['mean']:.6g}\n"
                f"Std: {self._stats_cache['std']:.6g}\n"
                f"Memory: {self.data.nbytes / 1024 / 1024:.1f} MB"
            )
            self.stats_label.setText(stats_text)
    
    def _request_level_update(self):
        """Request a batched level update"""
        if not self._level_update_pending:
            self._level_update_pending = True
            self._level_update_timer.start()
    
    def _apply_levels_batched(self):
        """Apply batched level updates"""
        self._level_update_timer.stop()
        self._level_update_pending = False
        self._apply_levels()
    
    def _apply_levels(self):
        """Apply brightness and contrast adjustments"""
        vmin_prime, vmax_prime = self._calculate_effective_range(
            self.base_vmin, self.base_vmax, self.brightness, self.contrast
        )
        if vmin_prime > vmax_prime:
            vmin_prime, vmax_prime = vmax_prime, vmin_prime
        
        self.img_item.setLevels((vmin_prime, vmax_prime))
        self.cbar.setLevels((vmin_prime, vmax_prime))
        self.info.setText(self._fmt_info())
    
    def _apply_pad_change(self, contrast_factor, brightness_delta):
        """Handle changes from control pad with batching"""
        self.contrast = max(0.01, self.contrast * float(contrast_factor))
        self.brightness += float(brightness_delta)
        
        # Clear cache when values change
        self._calculate_effective_range.cache_clear()
        
        self._request_level_update()
        self.pad.updateValues(self.contrast, self.brightness)
    
    def _on_plot_click(self, event):
        if event.double():
            self.reset_values()
    
    def reset_values(self):
        self.contrast = 1.0
        self.brightness = 0.0
        self.pad.updateValues(self.contrast, self.brightness)
        self._calculate_effective_range.cache_clear()
        self._apply_levels()
    
    def wheelEvent(self, event):
        if self.graph.geometry().contains(event.pos()):
            delta = event.angleDelta().y()
            factor = 0.9 if delta > 0 else (1 / 0.9)
            self.contrast = max(0.01, self.contrast * factor)
            self.pad.updateValues(self.contrast, self.brightness)
            self._calculate_effective_range.cache_clear()
            self._request_level_update()
            event.accept()
        else:
            super().wheelEvent(event)
    
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_R:
            self.reset_values()
            event.accept()
        elif event.key() == QtCore.Qt.Key_S:
            self._save_image()
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Handle window close event"""
        self._cleanup_resources()
        super().closeEvent(event)
    
    def _save_image(self):
        """Save current view with error handling"""
        try:
            base_xy = os.path.splitext(os.path.basename(self.xy_file))[0]
            base_z = os.path.splitext(os.path.basename(self.z_file))[0]
            filename = f'heatmap_{base_xy}_{base_z}.png'
            exporter = pg.exporters.ImageExporter(self.plot)
            exporter.export(filename)
            self.info.setText(f"View saved to {filename}")
        except Exception as e:
            self.info.setText(f"Save failed: {str(e)}")
    
    def select_xy_file(self):
        """Open dialog to select X/Y data file"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select X/Y Data File",
            os.path.dirname(self.xy_file) if self.xy_file else "",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        if filename:
            self._load_xy_file_async(filename)
    
    def select_z_file(self):
        """Open dialog to select color matrix file"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Color Matrix File",
            os.path.dirname(self.z_file) if self.z_file else "",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        if filename:
            self._load_z_file_async(filename)
    
    def _load_xy_file_async(self, filename):
        """Load X/Y file asynchronously with column selection"""
        # First detect columns
        num_cols, col_names = self.data_loader.detect_columns(filename)
        if num_cols == 0:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                "Could not detect columns in the selected file."
            )
            return
        
        # Show column selection dialog
        dialog = ColumnSelectionDialog(
            filename, col_names, 
            self.x_column if hasattr(self, 'x_column') else 1,
            self.y_column if hasattr(self, 'y_column') else 0,
            self
        )
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.x_column, self.y_column = dialog.get_selections()
            
            # Update plot labels
            self.plot.setLabel('bottom', f'X ({col_names[self.x_column]})')
            self.plot.setLabel('left', f'Y ({col_names[self.y_column]})')
            
            # Now load the file with selected columns
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Disable button during loading
            self.xy_btn.setEnabled(False)
            
            thread = FileLoadThread(
                filename, 
                self.data_loader.load_xy_data, 
                self,
                x_col=self.x_column,
                y_col=self.y_column
            )
            thread.progress.connect(self.progress_bar.setValue)
            thread.finished.connect(lambda data: self._on_xy_loaded(filename, data))
            thread.error.connect(self._on_xy_load_error)
            
            self.active_threads.append(thread)
            thread.finished.connect(lambda: self.active_threads.remove(thread))
            thread.start()
    
    def _on_xy_loaded(self, filename, data):
        """Handle successful X/Y data loading"""
        self.xy_file = filename
        self.xy_btn.setToolTip(f"Current: {os.path.basename(self.xy_file)}")
        self.xy_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        
        # Store loaded data temporarily
        self._pending_xy_data = data
        self.reload_data()
        
        QtCore.QTimer.singleShot(self.PROGRESS_HIDE_DELAY, 
                                lambda: self.progress_bar.setVisible(False))
    
    def _on_xy_load_error(self, error_msg):
        """Handle X/Y loading error"""
        self.xy_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QtWidgets.QMessageBox.critical(
            self,
            "Error Loading File",
            f"Failed to load X/Y data:\n{error_msg}"
        )
    
    def _load_z_file_async(self, filename):
        """Load color matrix file asynchronously"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Disable button during loading
        self.z_btn.setEnabled(False)
        
        thread = FileLoadThread(filename, self.data_loader.load_matrix_data, self)
        thread.progress.connect(self.progress_bar.setValue)
        thread.finished.connect(lambda data: self._on_z_loaded(filename, data))
        thread.error.connect(self._on_z_load_error)
        
        self.active_threads.append(thread)
        thread.finished.connect(lambda: self.active_threads.remove(thread))
        thread.start()
    
    def _on_z_loaded(self, filename, data):
        """Handle successful matrix data loading"""
        self.z_file = filename
        self.z_btn.setToolTip(f"Current: {os.path.basename(self.z_file)}")
        self.z_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        
        # Store loaded data temporarily
        self._pending_z_data = data
        self.reload_data()
        
        QtCore.QTimer.singleShot(self.PROGRESS_HIDE_DELAY, 
                                lambda: self.progress_bar.setVisible(False))
    
    def _on_z_load_error(self, error_msg):
        """Handle matrix loading error"""
        self.z_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QtWidgets.QMessageBox.critical(
            self,
            "Error Loading File",
            f"Failed to load color matrix data:\n{error_msg}"
        )
    
    def reload_data(self):
        """Reload data with optimizations"""
        try:
            # Clear old data to free memory
            if hasattr(self, 'data'):
                del self.data
            
            # Use pending data if available, otherwise reload
            if hasattr(self, '_pending_xy_data'):
                y, x = self._pending_xy_data
                del self._pending_xy_data
            else:
                y, x = self.data_loader.load_xy_data(
                    self.xy_file, 
                    x_col=self.x_column, 
                    y_col=self.y_column
                )
            
            if hasattr(self, '_pending_z_data'):
                Z = self._pending_z_data
                del self._pending_z_data
            else:
                Z = self.data_loader.load_matrix_data(self.z_file)
            
            # Update data
            self.data = Z.astype(np.float64)
            # Replace any NaN values to avoid rendering issues
            self.data = np.nan_to_num(self.data, copy=False, nan=0.0)
            
            # Clear caches
            if hasattr(self, '_stats_cache'):
                del self._stats_cache
            self._calculate_effective_range.cache_clear()
            
            # Update ranges
            xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
            ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
            self.xmin, self.xmax = xmin, xmax
            self.ymin, self.ymax = ymin, ymax
            
            # Update base values
            self.base_vmin = float(np.nanmin(self.data))
            self.base_vmax = float(np.nanmax(self.data))
            self.span = max(1e-12, self.base_vmax - self.base_vmin)
            
            # Reset brightness/contrast
            self.contrast = 1.0
            self.brightness = 0.0
            self.pad.updateValues(self.contrast, self.brightness)
            self.pad.span = self.span
            
            # Update image efficiently, ensure no NaN values
            clean_data = np.nan_to_num(self.data, copy=True, nan=0.0)
            self.img_item.setImage(clean_data, autoLevels=False)
            rect = QtCore.QRectF(self.xmin, self.ymin, 
                                (self.xmax - self.xmin), 
                                (self.ymax - self.ymin))
            self.img_item.setRect(rect)
            
            # Update colorbar
            self.cbar.setLevels((self.base_vmin, self.base_vmax))
            
            # Update plot ranges
            self.plot.setXRange(self.xmin, self.xmax, padding=0.0)
            self.plot.setYRange(self.ymin, self.ymax, padding=0.0)
            
            # Update stats and info
            self._update_stats()
            self._apply_levels()
            
            # Show success message
            self.info.setText(f"Data loaded: {os.path.basename(self.xy_file)} & {os.path.basename(self.z_file)}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error Reloading Data",
                f"Failed to reload data:\n{str(e)}"
            )


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Interactive heatmap viewer (CSV/TSV/space-delimited supported)."
    )
    p.add_argument("--xy", dest="xy_file", default=DEFAULT_XY_FILE,
                   help="Path to XY file (X,Y vectors; header optional).")
    p.add_argument("--z", dest="z_file", default=DEFAULT_Z_FILE,
                   help="Path to Z matrix file (rows form image lines).")
    p.add_argument("--x-col", type=int, default=1,
                   help="Zero-based column index to use for X (default: 1).")
    p.add_argument("--y-col", type=int, default=0,
                   help="Zero-based column index to use for Y (default: 0).")
    p.add_argument("--cmap", default=DEFAULT_CMAP,
                   help="Colormap to use (e.g., viridis, plasma, inferno, magma, cividis, gray, jet).")
    return p


def main():
    # Parse command line arguments
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Load data with data loader
    loader = DataLoader()
    
    # Use command line arguments
    xy_file = args.xy_file
    z_file = args.z_file
    x_col = args.x_col
    y_col = args.y_col
    cmap = args.cmap
    
    try:
        # Check if file exists and detect columns
        if os.path.exists(xy_file):
            num_cols, col_names = loader.detect_columns(xy_file)
            if num_cols > 0:
                # Load with specified columns
                y, x = loader.load_xy_data(xy_file, x_col=x_col, y_col=y_col)
            else:
                raise ValueError("No columns detected")
        else:
            raise FileNotFoundError(f"{xy_file} not found")
            
        Z = loader.load_matrix_data(z_file)
    except Exception as e:
        print(f"Error loading default files: {e}")
        print("Creating demo data...")
        # Create demo data if files not found
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        xx, yy = np.meshgrid(x, y)
        Z = np.sin(xx) * np.cos(yy)
    
    # extent = [xmin, xmax, ymin, ymax]
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    # Enable high DPI support
    try:
        app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    except:
        pass  # Older Qt version
    # Create the app/window
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        created_app = True
        
        
    
    # Initial vmin/vmax from data
    vmin = float(np.nanmin(Z))
    vmax = float(np.nanmax(Z))
    
    win = InteractiveHeatmapWindow(
        data=Z,
        x_range=(xmin, xmax),
        y_range=(ymin, ymax),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xy_file=xy_file,
        z_file=z_file
    )
    
    # Set initial column selections
    win.x_column = x_col
    win.y_column = y_col
    
    # Connect aboutToQuit signal for additional cleanup
    app.aboutToQuit.connect(win._cleanup_resources)
    
    win.show()
    
    if created_app:
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
