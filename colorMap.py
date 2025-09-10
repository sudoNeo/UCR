#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np

# PyQt5 fallback to PySide6 if necessary
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal
except ImportError:
    from PySide6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.Signal

import pyqtgraph as pg

class PadWidget(QtWidgets.QWidget):
    """
    Interactive pad controlling brightness & contrast:
      • Left‑drag: vertical = contrast (exponential), horizontal = brightness
      • Right‑drag: vertical = brightness
      • Wheel: contrast
      • R key resets (handled by main window)
    Emits (contrast_factor, brightness_delta) on changeRequested.
    """
    changeRequested = Signal(float, float)

    def __init__(self, parent=None, kc=-0.01, kb=0.002, span=1.0):
        super().__init__(parent)
        self.setMinimumSize(160, 160)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.kc = kc
        self.kb = kb
        self.span = span

        self._dragging = False
        self._button = None
        self._x0 = 0
        self._y0 = 0
        self._u = 0.5  # horizontal marker [0,1]
        self._v = 0.5  # vertical marker [0,1]

    def sizeHint(self):
        return QtCore.QSize(200, 200)

    def mousePressEvent(self, event):
        if event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            self._dragging = True
            self._button = event.button()
            self._x0, self._y0 = event.x(), event.y()
            self._updateMarker(event.x(), event.y())
            self.update()

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        dx = event.x() - self._x0
        dy = event.y() - self._y0

        contrast_factor = 1.0
        brightness_delta = 0.0

        if self._button == QtCore.Qt.LeftButton:
            contrast_factor = math.exp(self.kc * dy)
            brightness_delta = self.kb * dx * self.span
        elif self._button == QtCore.Qt.RightButton:
            brightness_delta = -self.kb * dy * self.span

        self._x0, self._y0 = event.x(), event.y()
        self._updateMarker(event.x(), event.y())
        self.update()
        self.changeRequested.emit(contrast_factor, brightness_delta)

    def mouseReleaseEvent(self, event):
        self._dragging = False
        self._button = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else (1 / 0.9)
        self.changeRequested.emit(factor, 0.0)
        # visually nudge marker
        self._v = np.clip(self._v + (-0.02 if delta > 0 else 0.02), 0.0, 1.0)
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
        p.fillRect(rect, self.palette().window())

        # dashed grid
        pen_grid = QtGui.QPen(self.palette().mid().color(), 1, QtCore.Qt.DashLine)
        p.setPen(pen_grid)
        w, h = rect.width(), rect.height()
        for i in range(1, 4):
            x = rect.left() + i * w / 4.0
            y = rect.top() + i * h / 4.0
            p.drawLine(int(x), rect.top(), int(x), rect.bottom())
            p.drawLine(rect.left(), int(y), rect.right(), int(y))

        # crosshair marker
        cx = rect.left() + int(self._u * w)
        cy = rect.top() + int(self._v * h)
        pen_cross = QtGui.QPen(self.palette().text().color(), 2)
        p.setPen(pen_cross)
        p.drawLine(cx - 8, cy, cx + 8, cy)
        p.drawLine(cx, cy - 8, cx, cy + 8)

        # border
        pen_border = QtGui.QPen(self.palette().mid().color())
        p.setPen(pen_border)
        p.drawRect(rect.adjusted(0, 0, -1, -1))

        # label
        p.setPen(self.palette().text().color())
        font = p.font()
        font.setPointSize(max(9, font.pointSize()))
        p.setFont(font)
        p.drawText(rect.adjusted(6, 6, -6, -6),
                   QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft,
                   "Pad\n← brightness →\n↑ contrast ↓")


class InteractiveBCWindow(QtWidgets.QMainWindow):
    """
    Main window containing:
      • left-side PadWidget for user control
      • right-side pyqtgraph image view with colorbar
    """
    def __init__(self, data, cmap='viridis', vmin=0.0, vmax=1.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Brightness/Contrast (PyQt)")
        self.resize(1100, 700)

        self.data = np.asarray(data)
        self.base_vmin = float(vmin)
        self.base_vmax = float(vmax)
        self.span = self.base_vmax - self.base_vmin

        self.contrast = 1.0
        self.brightness = 0.0

        self.kc = -0.01
        self.kb = 0.002

        cw = QtWidgets.QWidget(self)
        self.setCentralWidget(cw)
        hbox = QtWidgets.QHBoxLayout(cw)
        hbox.setContentsMargins(8, 8, 8, 8)
        hbox.setSpacing(10)

        # pad on the left
        self.pad = PadWidget(kc=self.kc, kb=self.kb, span=self.span)
        self.pad.changeRequested.connect(self._apply_pad_change)
        hbox.addWidget(self.pad, 0, QtCore.Qt.AlignTop)

        # pyqtgraph graphics layout on the right
        self.graph = pg.GraphicsLayoutWidget()
        hbox.addWidget(self.graph, 1)

        # image and colorbar
        self.plot = self.graph.addPlot(row=0, col=0)
        self.plot.setAspectLocked(False)
        self.img_item = pg.ImageItem(self.data, axisOrder='row-major')
        self.plot.addItem(self.img_item)
        self.lut = pg.colormap.get(cmap).getLookupTable(0.0, 1.0, 256)
        self.img_item.setLookupTable(self.lut)
        self.cbar = pg.ColorBarItem(values=(self.base_vmin, self.base_vmax),
                            colorMap=pg.colormap.get(cmap))
        self.graph.addItem(self.cbar, row=0, col=1)  # add to the layout explicitly
        self.cbar.setImageItem(self.img_item)        # no insert_in needed

        # top label with state info
        self.info = QtWidgets.QLabel(self._fmt_info())
        f = self.info.font()
        f.setPointSize(f.pointSize() + 1)
        self.info.setFont(f)
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.info)
        vbox.addWidget(self.graph, 1)
        container = QtWidgets.QWidget()
        container.setLayout(vbox)
        hbox.addWidget(container, 1)

        self._apply_levels()

    def _fmt_info(self):
        return (f"contrast = {self.contrast:.3f}    brightness = {self.brightness:+.3f}    "
                f"vmin'={(self.base_vmin - self.brightness)/self.contrast:.3f}   "
                f"vmax'={(self.base_vmax - self.brightness)/self.contrast:.3f}")

    def _apply_levels(self):
        # compute new vmin/vmax based on brightness & contrast
        vmin_prime = (self.base_vmin - self.brightness) / self.contrast
        vmax_prime = (self.base_vmax - self.brightness) / self.contrast
        if vmin_prime > vmax_prime:
            vmin_prime, vmax_prime = vmax_prime, vmin_prime
        self.img_item.setLevels((vmin_prime, vmax_prime))
        self.cbar.setLevels((vmin_prime, vmax_prime))
        self.info.setText(self._fmt_info())

    def _apply_pad_change(self, contrast_factor, brightness_delta):
        # update state with clamping like original script
        self.contrast = max(0.01, self.contrast * float(contrast_factor))
        self.brightness += float(brightness_delta)
        self._apply_levels()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_R:
            self.contrast = 1.0
            self.brightness = 0.0
            self._apply_levels()
            event.accept()
        else:
            super().keyPressEvent(event)

def demo_data():
    # replicate the demo pattern from your code
    y, x = np.mgrid[-1:1:300j, -1:1:400j]
    data = np.exp(-3 * (x**2 + y**2)) * (np.cos(8 * x) + np.cos(8 * y)) * 0.5 + 0.5
    return np.clip(data, 0, 1)

def main(data=None, cmap='viridis', vmin=0.0, vmax=1.0):
    app = QtWidgets.QApplication(sys.argv)
    if data is None:
        data = demo_data()
    win = InteractiveBCWindow(data=data, cmap=cmap, vmin=vmin, vmax=vmax)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
