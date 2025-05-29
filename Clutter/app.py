import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider,
    QPushButton, QHBoxLayout, QLabel, QComboBox, QCheckBox, QSizePolicy, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib import image as mpimg
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from functools import partial


class DataChangeHandler(FileSystemEventHandler):
    def __init__(self, plot_window):
        self.plot_window = plot_window

    def on_modified(self, event):
        if event.src_path.endswith(r"C:\Users\DELL\Documents\GitHub\gui\Clutter\DATA2245 copy.TXT"):
            print("Data file changed, reloading...")
            QTimer.singleShot(0, self.plot_window.reload_data)


class SleepSensePlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Developer Mode - Sleepsense Plotting")

        self.file_path = r"C:\Users\DELL\Documents\GitHub\gui\Clutter\DATA2245 copy.TXT"
        self.load_data()
        self.normalize_signals()

        self.start_time = self.time.iloc[0]
        self.end_time = self.time.iloc[-1]
        self.window_size = 10.0
        self.window_start = self.start_time
        self.scales = {'Pulse': 1.0, 'SpO2': 1.0, 'Airflow': 1.0}
        self.summary_signal = 'Pulse'

        self.visible_signals = {'Body Position': True, 'Pulse': True, 'SpO2': True, 'Airflow': True}

        self.init_ui()
        self.start_file_watcher()
        self.plot_signals()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        right_layout = QVBoxLayout()

        # Timeframe buttons layout
        timeframe_layout = QHBoxLayout()
        for label, sec in [("5s", 5), ("10s", 10), ("30s", 30), ("1m", 60), ("2m", 120), ("5m", 300)]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, s=sec: self.set_window_size(s))
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            timeframe_layout.addWidget(btn)
        right_layout.addLayout(timeframe_layout)

        # Event summary panel
        self.event_summary_label = QLabel("CSA: 0, OSA: 0, HSA: 0")
        self.event_summary_label.setFont(QFont("Arial", 12))
        self.event_summary_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.event_summary_label)

        # Summary signal plot (make it smaller)
        self.summary_canvas = FigureCanvas(Figure(figsize=(16, 2)))
        self.summary_ax = self.summary_canvas.figure.subplots()
        self.summary_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        right_layout.addWidget(self.summary_canvas, stretch=1)

        # Signal selector layout
        signal_selection_layout = QHBoxLayout()
        signal_label = QLabel(" Mahol pura wavyyyyyy")
        signal_label.setFont(QFont("Bold", 10))
        signal_selection_layout.addWidget(signal_label)

        self.signal_selector = QComboBox()
        self.signal_selector.addItems(['Pulse', 'SpO2', 'Airflow'])
        self.signal_selector.currentTextChanged.connect(self.change_summary_signal)
        self.signal_selector.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        signal_selection_layout.addWidget(self.signal_selector)
        signal_selection_layout.addStretch()
        right_layout.addLayout(signal_selection_layout)

        # Main signal plot (make it bigger)
        self.canvas = FigureCanvas(Figure(figsize=(74, 34)))  # Increased size
        self.ax = self.canvas.figure.subplots()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Enlarged and vertically aligned toggle buttons ---
        toggle_layout = QVBoxLayout()
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(0)
        self.toggle_buttons = {}

        # Define the order and vertical offsets (must match your offset dict)
        signals = [
            ('Airflow', 3.6),
            ('SpO2', 2.4),
            ('Pulse', 1.2),
            ('Body Position', 0.0)
        ]

        # Height of the plot area in offset units
        total_height = 4.8  # (from -0.5 to 5.5, but your signals are spaced by 1.2)

        for i, (signal, offset_val) in enumerate(signals):
            # Add vertical spacer proportional to the offset difference
            if i == 0:
                spacer_height = int(offset_val * 30)  # initial spacer
            else:
                prev_offset = signals[i-1][1]
                spacer_height = int((offset_val - prev_offset) * 60)  # adjust 60 for more/less space

            if spacer_height > 0:
                toggle_layout.addSpacing(spacer_height)

            row_layout = QHBoxLayout()
            btn = QPushButton(signal)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setMinimumHeight(50)
            btn.setFont(QFont("Arial", 14, QFont.Bold))
            btn.clicked.connect(lambda checked, sig=signal: self.toggle_signal(sig, checked))
            self.toggle_buttons[signal] = btn
            row_layout.addWidget(btn)

            if signal in ["Airflow", "SpO2", "Pulse"]:
                zoom_in_btn = QPushButton("+")
                zoom_in_btn.setFixedWidth(30)
                zoom_in_btn.setFont(QFont("Arial", 14, QFont.Bold))
                zoom_in_btn.clicked.connect(partial(self.change_signal_scale, signal, 0.2))
                row_layout.addWidget(zoom_in_btn)

                zoom_out_btn = QPushButton("−")
                zoom_out_btn.setFixedWidth(30)
                zoom_out_btn.setFont(QFont("Arial", 14, QFont.Bold))
                zoom_out_btn.clicked.connect(partial(self.change_signal_scale, signal, -0.2))
                row_layout.addWidget(zoom_out_btn)

            toggle_layout.addLayout(row_layout)

        toggle_layout.addStretch()

        # Horizontal layout holding toggles + main plot
        plot_and_toggle_layout = QHBoxLayout()
        plot_and_toggle_layout.addLayout(toggle_layout)
        plot_and_toggle_layout.addWidget(self.canvas, stretch=5)  # Main plot uses more space

        right_layout.addLayout(plot_and_toggle_layout, stretch=10)

        # --- Add this before adding self.slider to the layout ---
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.update_plot)

        right_layout.addWidget(self.slider)

        # --- Add Save Report Image button ---
        save_btn = QPushButton("Save Report Image")
        save_btn.setFont(QFont("Arial", 12, QFont.Bold))
        save_btn.setMinimumHeight(40)
        save_btn.clicked.connect(self.save_plot)
        right_layout.addWidget(save_btn)

        self.central_widget.setLayout(right_layout)
        self.resize(1400, 900)

    def create_signal_toggle(self, signal_name):
        checkbox = QCheckBox(f"Show {signal_name}")
        checkbox.setChecked(True)
        checkbox.toggled.connect(lambda checked, sig=signal_name: self.toggle_signal(sig, checked))
        self.toggle_checkboxes[signal_name] = checkbox
        return checkbox

    def load_data(self):
        self.data = pd.read_csv(self.file_path, header=None)
        self.time = self.data[0].astype(float) / 1000
        self.body_pos = self.data[1].astype(int)
        self.pulse = self.data[2].astype(float)
        self.spo2 = self.data[3].astype(float)
        self.flow = self.data[7].astype(float)

        # Print max and min value of airflow
        print(f"Airflow max: {self.flow.max()}, min: {self.flow.min()}")

        # Print most repeated (mode) value of airflow
        mode = self.flow.mode()
        if not mode.empty:
            print(f"Most repeated (mode) value in Airflow: {mode.iloc[0]}")
        else:
            print("No mode found in Airflow.")

    def normalize_signals(self):
        self.body_pos_n = self.normalize(self.body_pos)
        self.pulse_n = self.normalize(self.pulse)
        self.spo2_n = self.normalize(self.spo2)
        self.flow_n = self.normalize(self.flow)

    def normalize(self, series):
        range_ = series.max() - series.min()
        if range_ == 0:
            return series * 0
        return (series - series.min()) / range_

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

        # Smooth the signal with a moving average
        window_size = 10  # Adjust for more/less smoothing
        if len(signal) > window_size:
            signal = pd.Series(signal).rolling(window=window_size, min_periods=1, center=True).mean()

        # Make Airflow line extra thin and light in overview
        if self.summary_signal == "Airflow":
            lw = 0.2
            color = "#0a2be4"  # lighter blue
        else:
            lw = 0.5
            color = "blue"

        self.summary_ax.plot(self.time, signal, label=self.summary_signal, color=color, linewidth=lw)

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
        self.summary_canvas.draw_idle()

    def change_summary_signal(self, signal_name):
        self.summary_signal = signal_name
        self.plot_summary_signal()

    def toggle_signal(self, signal_name, visible):
        self.visible_signals[signal_name] = visible
        self.plot_signals()

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

        offset = {
            'Airflow': 3.6,
            'SpO2': 2.4,
            'Pulse': 1.2,
            'Body Position': 0.0,
        }

        # Plot Body Position images without distortion
        if self.visible_signals.get('Body Position', False):
            body_pos_segment = self.body_pos[mask]
            t_segment = t.reset_index(drop=True)
            # Calculate how many images to show (100 per minute)
            n_points = int(self.window_size * 100 / 60)
            n_points = max(1, n_points)
            indices = np.linspace(0, len(body_pos_segment) - 1, n_points, dtype=int) if len(body_pos_segment) > 0 else []
            for i in indices:
                pos = body_pos_segment.iloc[i]
                img_file = {
                    0: 'uparrow.jpg',
                    1: 'left.png',
                    2: 'right.png',
                    3: 'down.png',
                    5: 'sittingchair.png'
                }.get(pos)
                if img_file:
                    try:
                        img = mpimg.imread(img_file)
                        img_width = 0.30  # fixed time width
                        img_height = 0.30  # fixed vertical height (adjust as needed)
                        y_bottom = offset['Body Position']
                        extent = [t_segment.iloc[i], t_segment.iloc[i] + img_width, y_bottom, y_bottom + img_height]
                        self.ax.imshow(img, aspect='auto', extent=extent)
                    except FileNotFoundError:
                        print(f"Image file '{img_file}' not found. Please check the path.")

        window_size = 10  # Smoothing window for all signals

        if self.visible_signals.get('Pulse', False):
            pulse = self.pulse_n[mask] * self.scales['Pulse']
            if len(pulse) > window_size:
                pulse = pd.Series(pulse).rolling(window=window_size, min_periods=1, center=True).mean()
            self.ax.plot(t, pulse + offset['Pulse'], label="Pulse", color="red", linewidth=2.0)
        if self.visible_signals.get('SpO2', False):
            spo2 = self.spo2_n[mask] * self.scales['SpO2']
            if len(spo2) > window_size:
                spo2 = pd.Series(spo2).rolling(window=window_size, min_periods=1, center=True).mean()
            self.ax.plot(t, spo2 + offset['SpO2'], label="SpO2", color="green", linewidth=2.0)
        if self.visible_signals.get('Airflow', False):
            # Generate a synthetic sinusoidal wave between 1 and 4
            if len(t) > 1:
                period = (t.iloc[-1] - t.iloc[0]) if (t.iloc[-1] - t.iloc[0]) > 0 else 1
                sine_wave = 1.5 * np.sin(2 * np.pi * (t - t.iloc[0]) / period) + 2.5  # Range [1, 4]
            else:
                sine_wave = np.full_like(t, 2.5)
            self.ax.plot(t, sine_wave + offset['Airflow'], label="Airflow", color="#7ec8e3", linewidth=2.0)

            csa_count, osa_count, hsa_count = self.calculate_event_counts(sine_wave)
            event_summary = f"CSA: {csa_count}, OSA: {osa_count}, HSA: {hsa_count}"
            self.ax.text(0.02, 0.95, event_summary, transform=self.ax.transAxes, fontsize=10, color="black")

            # Update event summary label
            self.event_summary_label.setText(event_summary)

        self.ax.set_xlim(t0, t1)
        self.ax.set_ylim(-0.5,9)
        self.ax.set_title("Developer Mode ")
        self.ax.grid(True, linestyle='--', alpha=0.5)

        self.ax.legend(loc="upper right")
        self.canvas.draw_idle()
        self.plot_summary_signal()

    def update_plot(self, value):
        self.window_start = self.start_time + value / 10.0
        self.plot_signals()

    def set_window_size(self, seconds):
        self.window_size = seconds
        max_slider = int((self.end_time - self.start_time - self.window_size) * 10)
        max_slider = max(max_slider, 0)
        self.slider.setMaximum(max_slider)
        self.update_plot(self.slider.value())

    def reload_data(self):
        self.load_data()
        self.normalize_signals()
        self.end_time = self.time.iloc[-1]

        max_slider = int((self.end_time - self.start_time - self.window_size) * 10)
        max_slider = max(max_slider, 0)
        self.slider.setMaximum(max_slider)

        current_val = self.slider.value()
        if current_val > max_slider:
            self.slider.setValue(0)
            self.window_start = self.start_time
        else:
            self.window_start = self.start_time + current_val / 10.0

        self.plot_signals()

    def save_plot(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PDF Files (*.png);;All Files (*)", options=options)
        if file_path:
            import matplotlib.pyplot as plt

            # Create a new figure with two subplots (main + summary)
            fig, (ax_main, ax_summary) = plt.subplots(2, 1, figsize=(24, 16), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

            # --- Main plot ---
            t0 = self.window_start
            t1 = t0 + self.window_size
            mask = (self.time >= t0) & (self.time <= t1)
            t = self.time[mask]
            offset = {
                'Body Position': 0.0,
                'Pulse': 1.2,
                'SpO2': 2.4,
                'Airflow': 3.6,
            }

            # Plot Body Position images
            if self.visible_signals.get('Body Position', False):
                body_pos_segment = self.body_pos[mask]
                t_segment = t.reset_index(drop=True)
                n_points = int(self.window_size * 100 / 60)
                n_points = max(1, n_points)
                indices = np.linspace(0, len(body_pos_segment) - 1, n_points, dtype=int) if len(body_pos_segment) > 0 else []
                for i in indices:
                    pos = body_pos_segment.iloc[i]
                    img_file = {
                        0: 'right.png',
                        1: 'left.png',
                        2: 'Uparrow.jpg',
                        3: 'down.png',
                        5: 'sittingchair.png'
                    }.get(pos)
                    if img_file:
                        try:
                            img = mpimg.imread(img_file)
                            img_width = 0.30
                            img_height = 0.30
                            y_bottom = offset['Body Position']
                            extent = [t_segment.iloc[i], t_segment.iloc[i] + img_width, y_bottom, y_bottom + img_height]
                            ax_main.imshow(img, aspect='auto', extent=extent)
                        except FileNotFoundError:
                            pass

            window_size = 10
            # Plot signals
            if self.visible_signals.get('Pulse', False):
                pulse = self.pulse_n[mask] * self.scales['Pulse']
                if len(pulse) > window_size:
                    pulse = pd.Series(pulse).rolling(window=window_size, min_periods=1, center=True).mean()
                ax_main.plot(t, pulse + offset['Pulse'], label="Pulse", color="red", linewidth=2.0)
            if self.visible_signals.get('SpO2', False):
                spo2 = self.spo2_n[mask] * self.scales['SpO2']
                if len(spo2) > window_size:
                    spo2 = pd.Series(spo2).rolling(window=window_size, min_periods=1, center=True).mean()
                ax_main.plot(t, spo2 + offset['SpO2'], label="SpO2", color="green", linewidth=2.0)
            if self.visible_signals.get('Airflow', False):
                flow = self.flow_n[mask] * self.scales['Airflow']
                if len(flow) > window_size:
                    flow = pd.Series(flow).rolling(window=window_size, min_periods=1, center=True).mean()
                # Set values above 4 to zero
                flow = np.where(flow > 4, 0, flow)
                ax_main.plot(t, flow + offset['Airflow'], label="Airflow", color="#7ec8e3", linewidth=2.0)

                csa_count, osa_count, hsa_count = self.calculate_event_counts(flow)
                event_summary = f"CSA: {csa_count}, OSA: {osa_count}, HSA: {hsa_count}"
                ax_main.text(0.02, 0.95, event_summary, transform=ax_main.transAxes, fontsize=10, color="black")

            ax_main.set_xlim(t0, t1)
            ax_main.set_ylim(-0.5, 5.5)
            ax_main.set_title("Sleepsense Signal Viewer")
            ax_main.grid(True, linestyle='--', alpha=0.5)
            ax_main.legend(loc="upper right")

            # --- Summary plot ---
            summary_signal = self.summary_signal
            summary_mapping = {
                'Pulse': self.pulse_n,
                'SpO2': self.spo2_n,
                'Airflow': self.flow_n
            }
            summary_data = summary_mapping.get(summary_signal)
            window_size = 10
            if len(summary_data) > window_size:
                summary_data = pd.Series(summary_data).rolling(window=window_size, min_periods=1, center=True).mean()
            lw = 0.5 if summary_signal == "Airflow" else 1
            ax_summary.plot(self.time, summary_data, label=summary_signal, color="blue", linewidth=0.5)
            rect = Rectangle(
                (self.window_start, 0),
                self.window_size,
                1,
                color='orange',
                alpha=0.3
            )
            ax_summary.add_patch(rect)
            ax_summary.set_title(f"{summary_signal} Overview")
            ax_summary.legend(loc="upper right")
            ax_summary.set_xlim(self.time.iloc[0], self.time.iloc[-1])

            # Set x-label only on summary plot for clarity
            ax_summary.set_xlabel("Time (s)")

            plt.tight_layout()
            fig.savefig(file_path)
            plt.close(fig)
            print(f"Plot saved to {file_path}")

    def start_file_watcher(self):
        event_handler = DataChangeHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, ".", recursive=False)
        self.observer.start()

    def closeEvent(self, event):
        self.observer.stop()
        self.observer.join()
        event.accept()

    def change_signal_scale(self, signal_name, delta):
        # Clamp scale to reasonable range
        new_scale = max(0.2, min(5.0, self.scales[signal_name] + delta))
        self.scales[signal_name] = new_scale
        self.plot_signals()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SleepSensePlot()
    window.show()
    sys.exit(app.exec_())
