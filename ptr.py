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
from scipy.signal import butter, filtfilt


class DataChangeHandler(FileSystemEventHandler):
    def __init__(self, plot_window):
        self.plot_window = plot_window

    def on_modified(self, event):
        if event.src_path.endswith("DATA2245.TXT"):
            print("Data file changed, reloading...")
            QTimer.singleShot(0, self.plot_window.reload_data)


class SleepSensePlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Developer Mode - Sleepsense Plotting") 

        self.file_path = "DATA2245.TXT"
        self.load_data()
        self.normalize_signals()

        self.start_time = self.time.iloc[0]
        self.end_time = self.time.iloc[-1]
        self.window_size = 10.0
        self.window_start = self.start_time
        self.scales = {'Pulse': 1.0, 'SpO2': 1.0, 'Airflow': 1.0}
        self.summary_signal = 'Pulse'

        self.visible_signals = {'Body Position': True, 'Pulse': True, 'SpO2': True, 'Airflow': True}
        
        self.detected_windows = {'CSA': [], 'OSA': [], 'HSA': []}
        self.selected_event    = "None"

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

        # Summary signal plot (make it smaller)
        self.summary_canvas = FigureCanvas(Figure(figsize=(16, 2)))
        self.summary_ax = self.summary_canvas.figure.subplots()
        self.summary_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        right_layout.addWidget(self.summary_canvas, stretch=1)
        
        self.apnea_type_combo = QComboBox()
        self.apnea_type_combo.addItems(["CSA", "OSA", "HSA"])
        self.apnea_type_combo.setFixedWidth(100)
        right_layout.addWidget(QLabel("Apnea Type:"))
        right_layout.addWidget(self.apnea_type_combo)
        
        # after you build self.signal_selector, add:
        self.auto_detect_btn = QPushButton("Auto Detect")
        self.auto_detect_btn.setFont(QFont("Arial",12,QFont.Bold))
        self.auto_detect_btn.clicked.connect(self.on_auto_detect)
        right_layout.addWidget(self.auto_detect_btn)

        # Signal selector layout
        signal_selection_layout = QHBoxLayout()
        signal_label = QLabel(" Mahol pura wavyyyyyy")
        signal_label.setFont(QFont("Bold", 13))
        signal_selection_layout.addWidget(signal_label)

        self.signal_selector = QComboBox()
        self.signal_selector.addItems(['Pulse', 'SpO2', 'Airflow'])
        self.signal_selector.currentTextChanged.connect(self.change_summary_signal)
        self.signal_selector.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        signal_selection_layout.addWidget(self.signal_selector)
        signal_selection_layout.addStretch()
        right_layout.addLayout(signal_selection_layout)

        # Main signal plot (make it bigger)
        self.canvas = FigureCanvas(Figure(figsize=(98, 66)))  # Increased size not working will see baad mai
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
    
    def on_auto_detect(self):
        # read dropdown
        evt = self.apnea_type_combo.currentText()   # “CSA”, “OSA” or “HSA”
        self.selected_event = evt

        # run detector and pull out windows
        results = self.calculate_event_counts()
        self.detected_windows = {
            'CSA': results['CSA']['windows'],
            'OSA': results['OSA']['windows'],
            'HSA': results['HSA']['windows'],
        }

        # redraw with highlights
        self.plot_signals()

    def load_data(self):
        self.data = pd.read_csv(self.file_path, header=None)
        self.time = self.data[0].astype(float) / 1000
        self.body_pos = self.data[1].astype(int)
        self.pulse = self.data[2].astype(float)
        self.spo2 = self.data[3].astype(float)
        self.flow = self.data[7].astype(float)

        # Print how many times each value occurs in Airflow
        targets = [70, 60, 50, 40, 30,20, 15,14,13,12,11,10,7,5,4,3,2,1,8,6,9]
        for target in targets:
            count = np.sum(self.flow == target)
            # print(f"Airflow value {target} occurs {count} times")

    def normalize_signals(self):
        self.body_pos_n = self.normalize(self.body_pos)
        self.pulse_n = self.normalize(self.pulse)
        self.spo2_n = self.normalize(self.spo2)
        self.flow_n = self.normalize(self.flow)
        self.print_airflow_dominant_period()  # Add this line

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
        window_size = 10
        if len(signal) > window_size:
            signal = pd.Series(signal).rolling(window=window_size, min_periods=1, center=True).mean()

        # Use lighter colors and thinner lines
        color_map = {
            "Pulse": "#ffb3b3",    # light red
            "SpO2": "#b3ffb3",     # light green
            "Airflow": "#b3d1ff"   # light blue
        }
        lw = 0.7  # thinner line
        color = color_map.get(self.summary_signal, "gray")

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
        
    def calculate_event_counts(self):
        btp = self.flow.values.astype(int)
        
        np.set_printoptions(suppress=True, precision=2)
        time = self.time.values.astype(float)
        
        # max_btp = int(self.flow.max())
        # min_btp = int(self.flow.min())
        # print(f"Airflow max: {max_btp}, min: {min_btp}")
        
        # max_btp1 = int(self.flow.max() - 10)
        # max_btp2 = int(self.flow.max() - 20)
        # max_btp3 = int(self.flow.max() - 30)
        # max_btp4 = int(self.flow.max() - 40)
        # max_btp5 = int(self.flow.max() - 50)
        # max_btp_test = int(self.flow.max() - 55)
        # max_btp6 = int(self.flow.max() - 60)
        # max_btp7 = int(self.flow.max() - 65)
        
        # max_btp_count = (self.flow == max_btp).sum()
        # print(f"Pratyaksh Maximum value: {max_btp} occurred {max_btp_count} times")
        # max_btp_count = (self.flow == max_btp1).sum()
        # print(f"Pratyaksh Maximum value: {max_btp1} occurred {max_btp_count} times")
        # max_btp_count = (self.flow == max_btp2).sum()
        # print(f"Pratyaksh Maximum value: {max_btp2} occurred {max_btp_count} times")
        # max_btp_count = (self.flow == max_btp3).sum()
        # print(f"Pratyaksh Maximum value: {max_btp3} occurred {max_btp_count} times")
        # max_btp_count = (self.flow == max_btp4).sum()
        # print(f"Pratyaksh Maximum value: {max_btp4} occurred {max_btp_count} times")
        # max_btp_count = (self.flow == max_btp5).sum()
        # print(f"Pratyaksh Maximum value: {max_btp5} occurred {max_btp_count} times")
        # max_btp_count = (self.flow == max_btp_test).sum()
        # print(f"Pratyaksh Maximum value: {max_btp_test} occurred {max_btp_count} times")
        
        # max_btp_count = (self.flow == max_btp6).sum()
        # print(f"Pratyaksh Maximum value: {max_btp6} occurred {max_btp_count} times")
        # max_btp_count = (self.flow == max_btp7).sum()
        # print(f"Pratyaksh Maximum value: {max_btp7} occurred {max_btp_count} times")
        
        best_max = None
        best_min = None
        
        vc = self.flow.value_counts()
        
        max_btp = int(self.flow.max())
        min_btp = int(self.flow.min())
        max_candidates = [max_btp - i*10 for i in range(8)]
        min_candidates = [min_btp + i*10 for i in range(8)]
        max_cands = [v for v in max_candidates if v in vc.index]
        min_cands = [v for v in min_candidates if v in vc.index]
        best_max = max(max_cands, key=lambda v: vc[v]) if max_cands else max_btp
        best_min = max(min_cands, key=lambda v: vc[v]) if min_cands else min_btp
        
        print(f"PT Max BTP: {max_btp}, Min BTP: {min_btp}")
        print(f"Best Max BTP: {best_max}, Best Min BTP: {best_min}")
        
        btp_modified = np.where(btp > best_max, 0, btp)
        
        csa_thresh = int(0.10 * best_max + best_min)
        osa_thresh = int(0.50 * best_max + best_min)
        hsa_thresh = int(0.80 * best_max + best_min)
        
        print(f"CSA threshold: {csa_thresh}, OSA threshold: {osa_thresh}, HSA threshold: {hsa_thresh}")
        
        csa_count = osa_count = hsa_count = 0
        csa_blocks, osa_blocks, hsa_blocks = [], [], []
        csa_windows, osa_windows, hsa_windows = [], [], []
        
        print("BTP values modified:", btp_modified)

        total_samples = len(btp)

        # 1. CSA: 30-sample window (10s)
        csa_window = 30
        for i in range(0, total_samples, csa_window):
            block = btp_modified[i:i + csa_window]
            if len(block) < csa_window:
                break
            time_counts = [(j // 3) + 1 for j in range(len(block))]  # [1..10]
            # print("CSA block", block)
            if np.all((block > best_min) & (block <= csa_thresh)):
                csa_count += 1
                csa_blocks.append((time_counts, block.tolist()))
                start_ms = round(time[i], 2)
                end_ms   = round(time[i+29], 2)
                print(f"CSA block: {block}, start: {start_ms}, end: {end_ms}")
                csa_windows.append((start_ms, end_ms))

        # 2. OSA: 15-sample window (5s)
        osa_window = 15
        for i in range(0, total_samples, osa_window):
            block = btp_modified[i:i + osa_window]
            if len(block) < osa_window:
                break
            time_counts = [(j // 3) + 1 for j in range(len(block))]  # [1..5]
            # print("OSA block", block)
            if np.all((block > csa_thresh) & (block <= osa_thresh)):
                osa_count += 1
                osa_blocks.append((time_counts, block.tolist()))
                start_ms = round(time[i], 2)
                end_ms   = round(time[i+14], 2)
                print(f"OSA block: {block}, start: {start_ms}, end: {end_ms}")
                osa_windows.append((start_ms, end_ms))

        # 3. HSA: 15-sample window (5s)
        hsa_window = 15
        for i in range(0, total_samples, hsa_window):
            block = btp_modified[i:i + hsa_window]
            if len(block) < hsa_window:
                break
            time_counts = [(j // 3) + 1 for j in range(len(block))]  # [1..5]
            # print("HSA block", block)
            if np.all((block > osa_thresh) & (block <= hsa_thresh)):
                hsa_count += 1
                hsa_blocks.append((time_counts, block.tolist()))
                start_ms = round(time[i], 2)
                end_ms   = round(time[i+14], 2)
                print(f"HSA block: {block}, start: {start_ms}, end: {end_ms}")
                hsa_windows.append((start_ms, end_ms))
                
        print(f"CSA Windows: {csa_count}")
        print(f"OSA Windows: {osa_count}")
        print(f"HSA Windows: {hsa_count}")

        return {
            'CSA': {'count': csa_count, 'windows': csa_windows},
            'OSA': {'count': osa_count, 'windows': osa_windows},
            'HSA': {'count': hsa_count, 'windows': hsa_windows},
            }

    def bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def downsample(x, y, max_points=2000):
        if len(x) > max_points:
            idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
            return x.iloc[idx] if hasattr(x, 'iloc') else x[idx], y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
        return x, y

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
                        img_width = 0.30
                        img_height = 0.30
                        y_bottom = offset['Body Position']
                        extent = [t_segment.iloc[i], t_segment.iloc[i] + img_width, y_bottom, y_bottom + img_height]
                        self.ax.imshow(img, aspect='auto', extent=extent)
                    except FileNotFoundError:
                        print(f"Image file '{img_file}' not found. Please check the path.")

        window_size = 10

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
            flow = self.flow_n[mask] * self.scales['Airflow']
            smooth_window = 1000
            if len(flow) > smooth_window:
                flow_smooth = pd.Series(flow).rolling(window=smooth_window, min_periods=1, center=True).mean()
            else:
                flow_smooth = flow

            lower = np.percentile(flow_smooth, 1)
            upper = np.percentile(flow_smooth, 99)
            flow_smooth = np.clip(flow_smooth, lower, upper)
            
            results = self.calculate_event_counts()
            csa_count = results['CSA']['count']
            osa_count = results['OSA']['count']
            hsa_count = results['HSA']['count']
            event_summary = f"CSA: {csa_count}, OSA: {osa_count}, HSA: {hsa_count}"
            self.ax.text(0.02, 0.95, event_summary, transform=self.ax.transAxes, fontsize=10, color="black")

            flow_display = flow_smooth
            lower = np.percentile(flow_display, 1)
            upper = np.percentile(flow_display, 99)
            flow_display = np.clip(flow_display, lower, upper)

            self.ax.plot(t, flow_display + offset['Airflow'], label="Airflow", color="#7ec8e3", linewidth=2.0)

        self.ax.set_xlim(t0, t1)
        self.ax.set_ylim(-0.5, 12)
        self.ax.set_title("Developer Mode ")
        self.ax.grid(True, linestyle='--', alpha=0.5)

        self.ax.legend(loc="upper right")
        
        airflow_offset = 3.6                  # same offset you plot Airflow at
        airflow_amplitude = 1.0 * self.scales['Airflow'] 
        
        color_map = {
            'CSA': 'purple',
            'OSA': 'teal',
            'HSA': 'darkgreen',
        }
        
        if self.selected_event in self.detected_windows:
            for start_s, end_s in self.detected_windows[self.selected_event]:
                
                duration = end_s - start_s
                
                # Ensure the highlight covers the waveform
                rect = Rectangle(
                    (start_s, airflow_offset - airflow_amplitude / 2),  # align with center of waveform
                    duration,
                    airflow_amplitude,
                    color=color_map[self.selected_event],
                    alpha=0.3,
                    linewidth=0
                )
                self.ax.add_patch(rect)
                
                mid_s = (start_s + end_s) / 2
                # self.ax.axvline(mid_s, color=color_map[self.selected_event], linestyle='--', linewidth=1.2, alpha=0.8)
                
                label = f"{self.selected_event}  {duration:.1f}s"

                # Add label on top of the midpoint
                self.ax.text(
                    mid_s,
                    airflow_offset + airflow_amplitude / 2 + 0.2,  # slightly above the waveform
                    label,
                    color=color_map[self.selected_event],
                    fontsize=9,
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor=color_map[self.selected_event], boxstyle='round,pad=0.3', alpha=0.6)
                )
        
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

            window_size = 17
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
                smooth_window = 1000  # Try 1000 or higher for more smoothing
                if len(flow) > smooth_window:
                    flow_smooth = pd.Series(flow).rolling(window=smooth_window, min_periods=1, center=True).mean()
                else:
                    flow_smooth = flow

                # Clip outliers before scaling
                lower = np.percentile(flow_smooth, 1)
                upper = np.percentile(flow_smooth, 99)
                flow_smooth = np.clip(flow_smooth, lower, upper)

                # Only smooth and clip outliers, do not rescale to a fixed range
                flow_display = flow_smooth  # Already multiplied by self.scales['Airflow']
                lower = np.percentile(flow_display, 1)
                upper = np.percentile(flow_display, 99)
                flow_display = np.clip(flow_display, lower, upper)
                
                results = self.calculate_event_counts()
                csa_count = results['CSA']['count']
                osa_count = results['OSA']['count']
                hsa_count = results['HSA']['count']
                event_summary = f"CSA: {csa_count}, OSA: {osa_count}, HSA: {hsa_count}"
                ax_main.text(0.02, 0.95, event_summary, transform=ax_main.transAxes, fontsize=10, color="black")

                ax_main.plot(t, flow_display + offset['Airflow'], label="Airflow", color="#7ec8e3", linewidth=2.0)

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

    def print_airflow_dominant_period(self):
        # Use a strongly smoothed version of the normalized Airflow data
        smooth_window = 300
        flow = self.flow_n
        if len(flow) > smooth_window:
            flow_smooth = pd.Series(flow).rolling(window=smooth_window, min_periods=1, center=True).mean()
        else:
            flow_smooth = flow

        # Find zero-crossings (from negative to positive)
        zero_crossings = np.where(np.diff(np.sign(flow_smooth - np.mean(flow_smooth))) > 0)[0]
        if len(zero_crossings) > 1:
            # Calculate intervals in seconds
            intervals = np.diff(self.time.iloc[zero_crossings])
            # Find the most frequent interval (mode)
            if len(intervals) > 0:
                mode_interval = pd.Series(intervals).mode().iloc[0]
                print(f"Most frequent Airflow period (seconds): {mode_interval:.2f}")
            else:
                print("Not enough intervals to determine period.")
        else:
            print("Not enough zero-crossings to determine period.")




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SleepSensePlot()
    window.show()
    sys.exit(app.exec_())
