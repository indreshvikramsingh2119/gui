import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider,
    QPushButton, QHBoxLayout, QLabel, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DataChangeHandler(FileSystemEventHandler):
    def __init__(self, plot_window):
        self.plot_window = plot_window

    def on_modified(self, event):
        if event.src_path.endswith("DATA1623.TXT"):
            print("Data file changed, reloading...")
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, self.plot_window.reload_data)

class SleepSensePlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Developer Mode - Sleepsense Plotting")

        self.file_path = "DATA1131.TXT"
        self.load_data()
        self.normalize_signals()

        self.start_time = self.time.iloc[0]
        self.end_time = self.time.iloc[-1]
        self.window_size = 10.0
        self.window_start = self.start_time
        self.scales = {'Pulse': 1.0, 'SpO2': 1.0, 'Airflow': 1.0}
        self.summary_signal = 'Pulse'

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)

        timeframe_layout = QHBoxLayout()
        for label, sec in [("5s", 5), ("10s", 10), ("30s", 30), ("1m", 60), ("2m", 120)]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, s=sec: self.set_window_size(s))
            timeframe_layout.addWidget(btn)
        main_layout.addLayout(timeframe_layout)

        self.summary_canvas = FigureCanvas(Figure(figsize=(12, 2)))
        self.summary_ax = self.summary_canvas.figure.subplots()
        self.plot_summary_signal()
        main_layout.addWidget(self.summary_canvas)

        signal_selection_layout = QHBoxLayout()
        signal_label = QLabel("Signal selection:")
        signal_label.setFont(QFont("Arial", 10))
        signal_selection_layout.addWidget(signal_label)

        self.signal_selector = QComboBox()
        self.signal_selector.addItems(['Pulse', 'SpO2', 'Airflow'])
        self.signal_selector.currentTextChanged.connect(self.change_summary_signal)
        signal_selection_layout.addWidget(self.signal_selector)
        main_layout.addLayout(signal_selection_layout)

        self.canvas = FigureCanvas(Figure(figsize=(12, 6)))
        self.ax = self.canvas.figure.subplots()
        main_layout.addWidget(self.canvas)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(int((self.end_time - self.start_time - self.window_size) * 10))
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_plot)
        main_layout.addWidget(self.slider)

        self.start_file_watcher()
        self.plot_signals()

    def load_data(self):
        self.data = pd.read_csv(self.file_path, header=None)
        self.time = self.data[0].astype(float) / 1000
        self.body_pos = self.data[1].astype(int)
        self.pulse = self.data[2].astype(float)
        self.spo2 = self.data[3].astype(float)
        self.flow = self.data[7].astype(float)

    def normalize_signals(self):
        self.body_pos_n = self.normalize(self.body_pos)
        self.pulse_n = self.normalize(self.pulse)
        self.spo2_n = self.normalize(self.spo2)
        self.flow_n = self.normalize(self.flow)

    def normalize(self, series):
        return (series - series.min()) / (series.max() - series.min())

    def plot_summary_signal(self):
        self.summary_ax.clear()
        signal_mapping = {
            'Pulse': self.pulse_n,
            'SpO2': self.spo2_n,
            'Airflow': self.flow_n
        }
        signal = signal_mapping.get(self.summary_signal)
        if signal is None:
            return
        self.summary_ax.plot(self.time, signal, label=self.summary_signal, color="blue")

        rect = Rectangle(
            (self.window_start, 0),
            self.window_size,
            1,
            color='orange',
            alpha=0.3
        )
        self.summary_ax.add_patch(rect)

        self.summary_ax.set_title(f"{self.summary_signal} Overview")
        self.summary_ax.legend(loc="upper right")
        self.summary_canvas.draw()

    def change_summary_signal(self, signal_name):
        self.summary_signal = signal_name
        self.plot_summary_signal()

    def calculate_event_counts(self, flow_segment):
        csa_threshold = 0.2
        osa_threshold = 0.5
        hsa_threshold = 0.8

        csa_count = np.sum(flow_segment < csa_threshold)
        osa_count = np.sum((flow_segment >= csa_threshold) & (flow_segment < osa_threshold))
        hsa_count = np.sum((flow_segment >= osa_threshold) & (flow_segment < hsa_threshold))

        return csa_count, osa_count, hsa_count

    def plot_signals(self):
        self.ax.clear()
        t0 = self.window_start
        t1 = t0 + self.window_size
        mask = (self.time >= t0) & (self.time <= t1)
        t = self.time[mask]

        offset = [0, 1.2, 2.4, 3.6]
        body_pos = self.body_pos[mask]
        pulse = self.pulse_n[mask] * self.scales['Pulse']
        spo2 = self.spo2_n[mask] * self.scales['SpO2']
        flow = self.flow_n[mask] * self.scales['Airflow']

        self.ax.plot(t, pulse + offset[1], label="Pulse", color="red")
        self.ax.plot(t, spo2 + offset[2], label="SpO2", color="green")
        self.ax.plot(t, flow + offset[3], label="Airflow", color="blue")

        yticks = [np.mean(sig) + off for sig, off in zip([body_pos, pulse, spo2, flow], offset)]
        self.ax.set_yticks(yticks)
        self.ax.set_yticklabels(['Body Position', 'Pulse', 'SpO2', 'Airflow'])
        self.ax.set_xlim(t0, t1)
        self.ax.set_ylim(-0.5, 5)
        self.ax.set_title("Sleepsense Signal Viewer")
        self.ax.grid(True, linestyle='--', alpha=0.5)

        body_pos_map = {
            0: 'right.png',
            1: 'left.png',
            2: 'Uparrow.jpg',
            3: 'down.png',
            5: 'sitting.jpg'
        }

        step = 10  # Display one image for every 50 data points
        for time, pos in zip(t[::step], body_pos[::step]):
            if pos in body_pos_map:
                img_path = body_pos_map[pos]
                try:
                    img = plt.imread(img_path)
                    imagebox = OffsetImage(img, zoom=0.08)
                    ab = AnnotationBbox(imagebox, (time, offset[0]), frameon=False)
                    self.ax.add_artist(ab)
                except FileNotFoundError:
                    print(f"Image file '{img_path}' not found!")
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

        csa_count, osa_count, hsa_count = self.calculate_event_counts(flow)

        self.ax.text(0.95, 0.9, f"CSA: {csa_count}", color="black", fontsize=12, fontweight='bold', ha='right', transform=self.ax.transAxes)
        self.ax.text(0.95, 0.85, f"OSA: {osa_count}", color="black", fontsize=12, fontweight='bold', ha='right', transform=self.ax.transAxes)
        self.ax.text(0.95, 0.80, f"HSA: {hsa_count}", color="black", fontsize=12, fontweight='bold', ha='right', transform=self.ax.transAxes)

        self.canvas.draw()

        self.plot_summary_signal()

    def update_plot(self, value):
        self.window_start = self.start_time + value / 10.0
        self.plot_signals()

    def set_window_size(self, seconds):
        self.window_size = seconds
        max_slider = int((self.end_time - self.start_time - self.window_size) * 10)
        if max_slider < 0:
            max_slider = 0
        self.slider.setMaximum(max_slider)
        self.update_plot(self.slider.value())

    def reload_data(self):
        print("Reloading data...")
        self.load_data()
        self.normalize_signals()
        self.end_time = self.time.iloc[-1]

        max_slider = int((self.end_time - self.start_time - self.window_size) * 10)
        if max_slider < 0:
            max_slider = 0
        self.slider.setMaximum(max_slider)

        current_val = self.slider.value()
        if current_val > max_slider:
            self.slider.setValue(0)
            self.window_start = self.start_time
        else:
            self.window_start = self.start_time + current_val / 10.0

        self.plot_signals()

    def start_file_watcher(self):
        event_handler = DataChangeHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, ".", recursive=False)
        self.observer.start()

    def closeEvent(self, event):
        self.observer.stop()
        self.observer.join()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SleepSensePlot()
    window.show()
    sys.exit(app.exec_())
