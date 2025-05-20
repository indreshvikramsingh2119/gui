import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider,
    QPushButton, QHBoxLayout, QLabel
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SleepSensePlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dev Mode - Sleepsense Plotting")

        # Load data (adjust path)
        file_path = r"C:\Users\DELL\Documents\Workday1\sleepsense\DATA2304.TXT"
        self.data = pd.read_csv(file_path, header=None)
        self.time = self.data[0].astype(float) / 1000  # ms to s
        self.body_pos = self.data[1].astype(int)
        self.pulse = self.data[2].astype(float)
        self.spo2 = self.data[3].astype(float)
        self.flow = self.data[7].astype(float)

        # Normalize signals
        self.body_pos_n = self.normalize(self.body_pos)
        self.pulse_n = self.normalize(self.pulse)
        self.spo2_n = self.normalize(self.spo2)
        self.flow_n = self.normalize(self.flow)

        # Window settings
        self.start_time = self.time.iloc[0]
        self.end_time = self.time.iloc[-1]

        self.window_size = 10.0  # initial window size in seconds
        self.min_window_size = 1.0
        self.max_window_size = (self.end_time - self.start_time) / 2

        # Arrow directions for body positions
        self.arrow_directions = {
            0: (0, 0.5, '↑', 'Up (Supine)'),
            1: (-0.5, 0, '←', 'Left'),
            2: (0.5, 0, '→', 'Right'),
            3: (0, -0.5, '↓', 'Down (Prone)')
        }

        self.offsets = [0, 1.2, 2.4, 3.6]

        self.initUI()

    def normalize(self, series):
        return (series - series.min()) / (series.max() - series.min())

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Matplotlib Figure & Canvas
        self.fig = Figure(figsize=(14, 8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=0.2, top=0.85)

        # Slider layout
        slider_layout = QHBoxLayout()
        layout.addLayout(slider_layout)

        slider_label = QLabel("Time:")
        slider_layout.addWidget(slider_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(int((self.end_time - self.start_time - self.window_size) * 100))
        self.slider.setValue(0)
        self.slider.setTickInterval(100)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.update_plot)
        slider_layout.addWidget(self.slider)

        # Zoom buttons (+ and -)
        zoom_layout = QHBoxLayout()
        layout.addLayout(zoom_layout)

        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setFixedWidth(40)
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_btn)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(40)
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_btn)

        # Fixed window size buttons
        fixed_size_layout = QHBoxLayout()
        layout.addLayout(fixed_size_layout)

        self.window_sizes = [5, 10, 15, 30, 60, 120, 300]
        for size in self.window_sizes:
            label = f"{size//60}m" if size >= 60 else f"{size}s"
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, s=size: self.change_window_size(s))
            fixed_size_layout.addWidget(btn)

        self.plot_signals()
        self.update_plot()

    def plot_signals(self):
        self.ax.clear()

        # Plot normalized signals with offsets
        self.ax.plot(self.time, self.body_pos_n + self.offsets[0], color='black', label='Body Position')
        self.ax.plot(self.time, self.pulse_n + self.offsets[1], color='red', label='Pulse')
        self.ax.plot(self.time, self.spo2_n + self.offsets[2], color='green', label='SpO2')
        self.ax.plot(self.time, self.flow_n + self.offsets[3], color='black', label='Airflow')

        # Y-axis labels
        yticks_pos = [np.mean(sig) + offset for sig, offset in zip(
            [self.body_pos_n, self.pulse_n, self.spo2_n, self.flow_n], self.offsets)]
        yticks_labels = ['Body Position', 'Pulse (BPM)', 'SpO2 (%)', 'Airflow']
        self.ax.set_yticks(yticks_pos)
        self.ax.set_yticklabels(yticks_labels, fontsize=12)
        self.ax.set_ylim(-0.5, max(self.offsets) + 1)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_title('Dev Mode - Sleepsense Plotting with Body Position Arrows')

        # Add arrows for body position
        arrow_y = self.offsets[0] + 0.5
        interval = 5  # seconds between arrows
        dt = self.time.iloc[1] - self.time.iloc[0]
        sampling_step = max(int(interval / dt), 1)
        arrow_times = self.time[::sampling_step]

        for t in arrow_times:
            idx = (np.abs(self.time - t)).argmin()
            pos = self.body_pos.iloc[idx]
            dx, dy, arrow_char, label = self.arrow_directions.get(pos, (0, 0, '?', 'Unknown'))
            self.ax.annotate(
                arrow_char,
                xy=(self.time.iloc[idx], arrow_y),
                xytext=(self.time.iloc[idx] + dx, arrow_y + dy),
                fontsize=16,
                color='blue',
                ha='center',
                va='center',
                arrowprops=dict(arrowstyle='->', color='green')
            )

    def update_plot(self):
        slider_val = self.slider.value() / 100  # slider scaled to seconds
        # Clamp slider max to avoid window overflow
        max_start = self.end_time - self.window_size
        start = self.start_time + min(slider_val, max_start - self.start_time)
        end = start + self.window_size
        self.ax.set_xlim(start, end)
        self.canvas.draw_idle()

    def change_window_size(self, size):
        self.window_size = float(size)
        self.window_size = max(self.min_window_size, min(self.window_size, self.max_window_size))
        max_slider_val = int((self.end_time - self.start_time - self.window_size) * 100)
        self.slider.setMaximum(max_slider_val)
        if self.slider.value() > max_slider_val:
            self.slider.setValue(max_slider_val)
        else:
            self.update_plot()

    def zoom_in(self):
        new_size = self.window_size / 1.5  # zoom in by reducing window size
        if new_size < self.min_window_size:
            new_size = self.min_window_size
        self.window_size = new_size
        max_slider_val = int((self.end_time - self.start_time - self.window_size) * 100)
        self.slider.setMaximum(max_slider_val)
        self.update_plot()

    def zoom_out(self):
        new_size = self.window_size * 1.5  # zoom out by increasing window size
        if new_size > self.max_window_size:
            new_size = self.max_window_size
        self.window_size = new_size
        max_slider_val = int((self.end_time - self.start_time - self.window_size) * 100)
        self.slider.setMaximum(max_slider_val)
        self.update_plot()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SleepSensePlot()
    win.show()
    sys.exit(app.exec_())
