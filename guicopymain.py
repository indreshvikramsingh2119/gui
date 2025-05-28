import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider,
    QPushButton, QHBoxLayout, QLabel, QComboBox, QSizePolicy, QFileDialog
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class SleepApneaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sleep Apnea Detection Software")

        self.file_path = "DATA2245.TXT"  # Set your data file path here
        self.load_data()
        self.normalize_signals()

        self.window_size = 10  # default seconds
        self.window_start = self.time.iloc[0]
        self.init_ui()
        self.plot_signals()

    def load_data(self):
        self.data = pd.read_csv(self.file_path, header=None)
        self.time = self.data[0].astype(float) / 1000  # or just .astype(float) if not ms
        self.body_pos = self.data[1].astype(int)
        self.pulse = self.data[2].astype(float)
        self.spo2 = self.data[3].astype(float)
        self.airflow = self.data[7].astype(float)

    def normalize_signals(self):
        self.airflow_n = (self.airflow - self.airflow.min()) / (self.airflow.max() - self.airflow.min())
        self.spo2_n = (self.spo2 - self.spo2.min()) / (self.spo2.max() - self.spo2.min())
        self.pulse_n = (self.pulse - self.pulse.min()) / (self.pulse.max() - self.pulse.min())

    def detect_apnea_events(self, t, airflow, min_duration_sec=10, fs=10):
        # Example thresholds for normalized airflow
        csa_thresh = 0.2
        osa_thresh = 0.5
        hsa_thresh = 0.7
        min_samples = int(min_duration_sec * fs)
        events = {'CSA': [], 'OSA': [], 'HSA': []}
        below = airflow < csa_thresh
        csa_starts = np.where(np.diff(np.concatenate(([0], below.astype(int)))) == 1)[0]
        csa_ends = np.where(np.diff(np.concatenate((below.astype(int), [0]))) == -1)[0]
        for start, end in zip(csa_starts, csa_ends):
            if end - start >= min_samples:
                events['CSA'].append((t.iloc[start], t.iloc[end-1]))
        below = (airflow < osa_thresh) & (airflow >= csa_thresh)
        osa_starts = np.where(np.diff(np.concatenate(([0], below.astype(int)))) == 1)[0]
        osa_ends = np.where(np.diff(np.concatenate((below.astype(int), [0]))) == -1)[0]
        for start, end in zip(osa_starts, osa_ends):
            if end - start >= min_samples:
                events['OSA'].append((t.iloc[start], t.iloc[end-1]))
        below = (airflow < hsa_thresh) & (airflow >= osa_thresh)
        hsa_starts = np.where(np.diff(np.concatenate(([0], below.astype(int)))) == 1)[0]
        hsa_ends = np.where(np.diff(np.concatenate((below.astype(int), [0]))) == -1)[0]
        for start, end in zip(hsa_starts, hsa_ends):
            if end - start >= min_samples:
                events['HSA'].append((t.iloc[start], t.iloc[end-1]))
        return events

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()
        # Timeframe buttons
        timeframe_layout = QHBoxLayout()
        for label, sec in [("5s", 5), ("10s", 10), ("15s", 15), ("30s", 30), ("1m", 60), ("2m", 120)]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, s=sec: self.set_window_size(s))
            timeframe_layout.addWidget(btn)
        layout.addLayout(timeframe_layout)
        # Matplotlib canvas
        self.canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.ax = self.canvas.figure.subplots()
        layout.addWidget(self.canvas)
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.valueChanged.connect(self.update_plot)
        layout.addWidget(self.slider)
        self.central_widget.setLayout(layout)

    def set_window_size(self, seconds):
        self.window_size = seconds
        self.update_plot(self.slider.value())

    def update_plot(self, value):
        self.window_start = self.time.iloc[0] + value
        self.plot_signals()

    def plot_signals(self):
        self.ax.clear()
        t0 = self.window_start
        t1 = t0 + self.window_size
        mask = (self.time >= t0) & (self.time <= t1)
        t = self.time[mask]
        airflow = self.airflow_n[mask]
        spo2 = self.spo2_n[mask]
        pulse = self.pulse_n[mask]
        body_pos = self.body_pos[mask]
        # Plot signals
        self.ax.plot(t, airflow + 2, label="Airflow", color="blue")
        self.ax.plot(t, spo2 + 1, label="SpO2", color="green")
        self.ax.plot(t, pulse, label="Pulse", color="red")
        # Show body position as text
        pos_labels = {0: "Supine", 1: "Left", 2: "Right", 3: "Up", 4: "Down"}
        for idx, pos in zip(t, body_pos):
            self.ax.text(idx, -0.2, pos_labels.get(pos, ""), fontsize=8, rotation=90)
        # Detect and mark apnea events
        fs = 10  # Adjust to your sampling rate
        events = self.detect_apnea_events(t, airflow, min_duration_sec=10, fs=fs)

        first_csa = True
        first_osa = True
        first_hsa = True
        for start, end in events['CSA']:
            self.ax.axvspan(start, end, color='blue', alpha=0.2, label='CSA' if first_csa else "")
            first_csa = False
        for start, end in events['OSA']:
            self.ax.axvspan(start, end, color='red', alpha=0.2, label='OSA' if first_osa else "")
            first_osa = False
        for start, end in events['HSA']:
            self.ax.axvspan(start, end, color='orange', alpha=0.2, label='HSA' if first_hsa else "")
            first_hsa = False
        self.ax.legend()
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SleepApneaApp()
    window.show()
    sys.exit(app.exec_())