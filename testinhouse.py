import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider,
    QPushButton, QHBoxLayout, QGridLayout, QToolButton, QLabel
)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


class HoverLabel(QWidget):
    def __init__(self, signal_name, zoom_callback_in, zoom_callback_out):
        super().__init__()
        self.zoom_callback_in = zoom_callback_in
        self.zoom_callback_out = zoom_callback_out

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.label = QLabel(signal_name)
        self.label.setFont(QFont("Arial", 10))
        layout.addWidget(self.label)

        self.zoom_in_btn = QToolButton()
        self.zoom_in_btn.setText("+")
        self.zoom_in_btn.clicked.connect(lambda: self.zoom_callback_in(signal_name))
        self.zoom_in_btn.hide()

        self.zoom_out_btn = QToolButton()
        self.zoom_out_btn.setText("-")
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_callback_out(signal_name))
        self.zoom_out_btn.hide()

        layout.addWidget(self.zoom_in_btn)
        layout.addWidget(self.zoom_out_btn)

        self.setMouseTracking(True)
        self.installEventFilter(self)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Enter:
            self.zoom_in_btn.show()
            self.zoom_out_btn.show()
        elif event.type() == QEvent.Leave:
            self.zoom_in_btn.hide()
            self.zoom_out_btn.hide()
        return super().eventFilter(source, event)


class SleepSensePlot(QMainWindow):
    def __init__(self):
        super().__init__()   
        self.setWindowTitle("Developer Mode - Sleepsense Plotting")

        # Load data
        file_path = r"C:\Users\DELL\Downloads\testing_breath_2\DATA1623.TXT"
        self.data = pd.read_csv(file_path, header=None)
        self.time = self.data[0].astype(float) / 1000  # ms to seconds
        self.body_pos = self.data[1].astype(int)
        self.pulse = self.data[2].astype(float)
        self.spo2 = self.data[3].astype(float)
        self.flow = self.data[7].astype(float)
        
        # OSA, CSA, HSA total count
        self.csa_count, self.osa_count, self.hsa_count = self.detect_apneas()
        print(f"CSA: {self.csa_count}, OSA: {self.osa_count}, HSA: {self.hsa_count}")

        # Normalize
        self.body_pos_n = self.normalize(self.body_pos)
        self.pulse_n = self.normalize(self.pulse)
        self.spo2_n = self.normalize(self.spo2)
        self.flow_n = self.normalize(self.flow)

        # Parameters
        self.start_time = self.time.iloc[0]
        self.end_time = self.time.iloc[-1]
        self.window_size = 10.0
        self.window_start = self.start_time
        self.scales = {
            'Pulse': 1.0,
            'SpO2': 1.0,
            'Airflow': 1.0,
        }

        # Layout setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)

        # Timeframe buttons on top
        timeframe_layout = QHBoxLayout()
        for label, sec in [("5s", 5), ("10s", 10), ("30s", 30), ("1m", 60), ("5m", 300)]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, s=sec: self.set_window_size(s))
            timeframe_layout.addWidget(btn)
        main_layout.addLayout(timeframe_layout)

        self.canvas = FigureCanvas(Figure(figsize=(12, 6)))
        self.ax = self.canvas.figure.subplots()

        grid_layout = QGridLayout()
        main_layout.addLayout(grid_layout)

        # Left signal zoom buttons
        left_panel = QVBoxLayout()
        left_panel.setAlignment(Qt.AlignTop)
        for signal in ['Pulse', 'SpO2', 'Airflow']:
            label = HoverLabel(
                signal_name=signal,
                zoom_callback_in=self.zoom_in,
                zoom_callback_out=self.zoom_out
            )
            left_panel.addWidget(label)

        left_container = QWidget()
        left_container.setLayout(left_panel)
        grid_layout.addWidget(left_container, 0, 0, alignment=Qt.AlignTop)

        # Plot in center
        grid_layout.addWidget(self.canvas, 0, 1, 5, 1)

        # Time slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(int((self.end_time - self.start_time - self.window_size) * 10))
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_plot)
        main_layout.addWidget(self.slider)

        self.plot_signals()

    def normalize(self, series):
        return (series - series.min()) / (series.max() - series.min())

    def get_body_arrow(self, value):
        if value == 1:
            return "▲"  # Supine
        elif value == 2:
            return "◀"  # Left
        elif value == 3:
            return "▶"  # Right
        elif value == 5:
            return "▼"  # Prone
        else:
            return "?"  # Unknown

    def plot_signals(self):
        self.ax.clear()
        t0 = self.window_start
        t1 = t0 + self.window_size
        mask = (self.time >= t0) & (self.time <= t1)
        t = self.time[mask]

        offset = [0, 1.2, 2.4, 3.6]
        body_pos = self.body_pos_n[mask]
        pulse = self.pulse_n[mask] * self.scales['Pulse']
        spo2 = self.spo2_n[mask] * self.scales['SpO2']
        flow = self.flow_n[mask] * self.scales['Airflow']

        # Plot body position as points + arrows
        self.ax.plot(t, body_pos + offset[0], label="Body Position", color="black", linestyle='', marker='o')
        for ti, bi in zip(t, self.body_pos[mask]):
            symbol = self.get_body_arrow(bi)
            self.ax.text(ti, offset[0], symbol, fontsize=10, ha='center', va='bottom')

        # Plot other signals
        self.ax.plot(t, pulse + offset[1], label="Pulse", color="red")
        self.ax.plot(t, spo2 + offset[2], label="SpO2", color="green")
        self.ax.plot(t, flow + offset[3], label="Airflow", color="blue")

        yticks = [np.mean(sig) + off for sig, off in zip([body_pos, pulse, spo2, flow], offset)]
        self.ax.set_yticks(yticks)
        self.ax.set_yticklabels(['Body Position', 'Pulse', 'SpO2', 'Airflow'])
        
        custom_legend = [
            Line2D([0], [0], color="purple", linestyle='None', markersize=8, label=f"CSA {self.csa_count}"),
            Line2D([0], [0], color="teal", linestyle='None', markersize=8, label=f"OSA {self.osa_count}"),
            Line2D([0], [0], color="darkgreen", linestyle='None', markersize=8, label=f"HSA {self.hsa_count}"),
        ]
        self.ax.legend(handles=custom_legend, loc="upper right")

        self.ax.set_xlim(t0, t1)
        self.ax.set_ylim(-0.5, 5)
        self.ax.set_title("Sleepsense Signal Viewer")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.canvas.draw()

    def update_plot(self, value):
        self.window_start = self.start_time + value / 10.0
        self.plot_signals()

    def zoom_in(self, signal_name):
        self.scales[signal_name] *= 1.2
        self.plot_signals()

    def zoom_out(self, signal_name):
        self.scales[signal_name] /= 1.2
        self.plot_signals()

    def set_window_size(self, seconds):
        self.window_size = seconds
        max_slider = int((self.end_time - self.start_time - self.window_size) * 10)
        self.slider.setMaximum(max_slider)
        self.update_plot(self.slider.value())
        
    def detect_apneas(self):
        # use pulse as breaths_per_min
        btp = self.pulse.values
        max_btp = btp.max()
        min_btp = btp.min()

        csa_thresh = int(0.10 * max_btp + min_btp)
        osa_thresh = int(0.50 * max_btp + min_btp)
        hsa_thresh = int(0.80 * max_btp + min_btp)

        csa_count = osa_count = hsa_count = 0

        # find runs of constant BTP > 2 samples
        runs = []
        run_val = btp[0]
        run_len = 1
        for v in btp[1:]:
            if v == run_val:
                run_len += 1
            else:
                if run_len > 2:
                    runs.append((run_val, run_len))
                run_val = v
                run_len = 1
        if run_len > 2:
            runs.append((run_val, run_len))

        # classify each run
        for val, length in runs:
            if length >= 10 and min_btp < val <= csa_thresh:
                csa_count += 1
            elif csa_thresh < val <= osa_thresh:
                osa_count += 1
            elif osa_thresh < val <= hsa_thresh:
                hsa_count += 1

        return csa_count, osa_count, hsa_count


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SleepSensePlot()
    window.show()
    sys.exit(app.exec_())
