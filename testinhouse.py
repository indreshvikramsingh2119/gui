import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider,
    QPushButton, QHBoxLayout, QLabel, QComboBox, QCheckBox, QSizePolicy, QFileDialog, QInputDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from matplotlib import image as mpimg
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from functools import partial
from scipy.signal import butter, filtfilt


class DataChangeHandler(FileSystemEventHandler):
    def __init__(self, plot_window):
        self.plot_window = plot_window

    def on_modified(self, event):
        if event.src_path.endswith("DATA2304.TXT"):
            print("Data file changed, reloading...")  # Debug message
            QTimer.singleShot(0, self.plot_window.reload_data)


class SleepSensePlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Developer Mode - Sleepsense Plotting") 

        self.file_path = "DATA2304.TXT"
        self.load_data()

        self.start_time = self.time.iloc[0]
        self.end_time = self.time.iloc[-1]
        self.window_size = 10.0
        self.window_start = self.start_time
        self.scales = {'Pulse': 1.0, 'SpO2': 1.0, 'Airflow': 1.0}
        self.summary_signal = 'Pulse'
        self.visible_signals = {'Body Position': True, 'Pulse': True, 'SpO2': True, 'Airflow': True}
 
        self.init_ui()              # <-- call this before normalize_signals
        self.normalize_signals()    # <-- call this after init_ui
        self.start_file_watcher()
        self.plot_signals()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        right_layout = QVBoxLayout()

        # Timeframe buttons layout
        timeframe_layout = QHBoxLayout()
        for label, sec in [("5s", 5), ("10s", 10), ("30s", 30), ("1m", 60), ("2m", 120)]:
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
        
        # manual hilight for CSA, HSA, OSA events
        self.manual_events = []
        
        # remove manual detect apnea event
        self.manual_event_patches = []
        
        # remove auto detect apnea event
        self.detected_event_patches = []
        
        self.selector = RectangleSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],      # left-click only
            minspanx=0.1,    # 0.1s min
            interactive=True
        )

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
        
        # right click button press
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)

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

        # --- Event count label ---
        self.event_count_label = QLabel()
        self.event_count_label.setFont(QFont("Arial", 13, QFont.Bold))
        self.event_count_label.setStyleSheet("color: #222; background: #f7f7f7; border-radius: 6px; padding: 8px;")
        right_layout.addWidget(self.event_count_label)

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

        # Print how many times each value occurs in Airflow
        targets = [70, 60, 50, 40, 30,20, 15,14,13,12,11,10,7,5,4,3,2,1,8,6,9]
        for target in targets:
            count = np.sum(self.flow == target)
            # print(f"Airflow value {target} occurs {count} times")
    
    def on_mouse_click(self, event):
        if event.button != 3:  # Right click only
            return

        click_x = event.xdata
        if click_x is None:
            return

        # Find and remove the manual-detected event closest to the click
        for item in self.manual_event_patches:
            start, end, typ, rect, text = item
            if start <= click_x <= end:
                self.manual_event_patches.remove(item)
                if (start, end, typ) in self.manual_events:
                    self.manual_events.remove((start, end, typ))
                rect.remove()
                text.remove()
                self.plot_signals()
                return

        # Find and remove the auto-detected event closest to the click
        for item in self.detected_event_patches:
            start, end, typ, rect, text = item
            if start <= click_x <= end:
                self.detected_event_patches.remove(item)
                if hasattr(self, "detected_events") and (start, end, typ) in self.detected_events:
                    self.detected_events.remove((start, end, typ))
                rect.remove()
                text.remove()
                self.plot_signals()
                return

            
    def on_select(self, eclick, erelease):
        # eclick.xdata is the start, erelease.xdata is the end
        start, end = sorted([eclick.xdata, erelease.xdata])

        # ask the user what kind of event:
        typ, ok = QInputDialog.getItem(
            self,
            "Apnea Detection Event",
            "Event type:",
            ["CSA","OSA","HSA"],
            0,
            False
        )
        if not ok:
            return

        # save it and redraw
        self.manual_events.append((start, end, typ))
        self.plot_signals()

    def normalize_signals(self):
        self.body_pos_n = self.normalize(self.body_pos)
        self.pulse_n = self.normalize(self.pulse)
        self.spo2_n = self.normalize(self.spo2)
        self.flow_n = self.normalize(self.flow)
        self.print_airflow_dominant_period()
        self.detect_apnea_events()  # <-- Add this line to detect apnea events PTR ADD THIS LINE

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

        # --- Calculate total event counts for full data ---
        total_hsa = total_csa = total_osa = 0
        if hasattr(self, "detected_events"):
            for _, _, typ in self.detected_events:
                if typ == "HSA":
                    total_hsa += 1
                elif typ == "CSA":
                    total_csa += 1
                elif typ == "OSA":
                    total_osa += 1
                    
        if hasattr(self, "manual_events"):
            for _, _, typ in self.manual_events:
                if typ == "HSA":
                    total_hsa += 1
                elif typ == "CSA":
                    total_csa += 1
                elif typ == "OSA":
                    total_osa += 1

        t0 = self.time.iloc[0]
        t1 = self.time.iloc[-1]
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
            body_pos_segment = self.body_pos[mask].reset_index(drop=True)
            t_segment = t.reset_index(drop=True)
            y_base = offset['Body Position'] + 0.4  # vertical position for arrows

            arrow_map = {
                1: '←',  # Left
                2: '→',  # Right
                3: '↑',  # Up
                4: '↓',  # Down
                5: '⏏',  # Sitting/Other
            }
            # Plot an arrow for every 50th data point
            for i in range(0, len(body_pos_segment), 50):
                pos = body_pos_segment.iloc[i]
                arrow = arrow_map.get(pos, '?')
                self.ax.text(
                    t_segment.iloc[i], y_base, arrow,
                    fontsize=16, ha='center', va='center', color='purple', fontweight='bold', clip_on=True
                )

        window_size = 70

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
            smooth_window = 20  # much less smoothing
            if len(flow) > smooth_window:
                flow_smooth = pd.Series(flow).rolling(window=smooth_window, min_periods=1, center=True).mean()
            else:
                flow_smooth = flow

            # Optionally skip clipping for now
            # lower = np.percentile(flow_smooth, 1)
            # upper = np.percentile(flow_smooth, 99)
            # flow_smooth = np.clip(flow_smooth, lower, upper)

            self.ax.plot(t, flow_smooth + offset['Airflow'], label="Airflow", color="#7ec8e3", linewidth=2.0)
            
        airflow_base = 3.6
        bar_half = 0.3
        colors = {"CSA":"red","OSA":"purple","HSA":"darkgreen"}

        for start, end, typ in self.manual_events:
            if end < t0 or start > t1:
                continue
            xs = max(start,t0); xe = min(end,t1)
            dur = xe - xs

            # draw the box at airflow level
            rect = Rectangle(
                (xs, airflow_base - bar_half),
                xe - xs,
                2*bar_half,
                color=colors[typ],
                alpha=0.4,
                edgecolor="none"
            )
            self.ax.add_patch(rect)

            # label above it
            xm = 0.5*(xs+xe)
            ym = airflow_base + bar_half + 0.05
            
            text_obj = self.ax.text(
                xm, ym,
                f"{typ} ({dur:.1f}s)",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color=colors[typ],
                bbox=dict(facecolor="white", edgecolor=colors[typ], pad=2, alpha=0.8)
            )
            
            # remove manual apnea detect event patch
            self.manual_event_patches.append((start, end, typ, rect, text_obj))
            
            # remove auto apnea detect event patch
            self.detected_event_patches.append((start, end, typ, rect, text_obj))

        # --- Highlight detected apnea/hypopnea events ---
        if hasattr(self, "detected_events"):
            event_colors = {"CSA": "red", "OSA": "purple", "HSA": "darkgreen"}  # Red, Purple, Darkgreen
            airflow_base = 3.6  # y-offset for Airflow
            bar_half_height = 0.3  # half the thickness of the bar
            
            for start, end, typ in self.detected_events:
                if end < t0 or start > t1:
                    continue
                plot_start = max(start, t0) 
                plot_end = min(end, t1)
                duration   = plot_end - plot_start

                # 1) draw a colored rectangle at the airflow level
                rect = Rectangle(
                    (plot_start, airflow_base - bar_half_height),
                    plot_end - plot_start,
                    2 * bar_half_height,
                    color=event_colors[typ],
                    alpha=0.5,
                    edgecolor="none"
                )
                self.ax.add_patch(rect)

                # 2) put the label above the top edge of that rect
                x_mid = (plot_start + plot_end) / 2
                y_top = airflow_base + bar_half_height + 0.05  # a little above the rect

                self.ax.text(
                    x_mid, y_top,
                    f"{typ} ({duration:.1f}s)",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", pad=2)
                )
                
            # To avoid duplicate legend entries
            handles, labels = self.ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            self.ax.legend(unique.values(), unique.keys(), loc="upper right")

                                    # --- Add statistics box for HSA, CSA, OSA in current window ---
        hsa_count = csa_count = osa_count = 0
        t0 = self.window_start
        t1 = t0 + self.window_size
        if hasattr(self, "detected_events"):
            for start, end, typ in self.detected_events:
                if end < t0 or start > t1:
                    continue
                if typ == "HSA":
                    hsa_count += 1
                elif typ == "CSA":
                    csa_count += 1
                elif typ == "OSA":
                    osa_count += 1

        # stats_text = f"HSA: {hsa_count}\nCSA: {csa_count}\nOSA: {osa_count}"
        # # Always create the stats box after clearing the axes
        # self._stats_box = self.ax.text(
        #     0.01, 0.98, stats_text,
        #     transform=self.ax.transAxes,
        #     fontsize=14,
        #     verticalalignment='top',
        #     horizontalalignment='left',
        #     bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="gray", alpha=0.8)
        # )

        # --- Add total event counts box at the top left ---
        total_stats_text = (
            f"Total Events\n"
            f"HSA: {total_hsa}  CSA: {total_csa}  OSA: {total_osa}"
        )
        self.ax.text(
            0.01, 0.98, total_stats_text,
            transform=self.ax.transAxes,
            fontsize=13,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e6f7ff", edgecolor="#3399cc", alpha=0.8)
        )

        self.ax.set_xlim(t0, t1)
        self.ax.set_ylim(-0.5, 12)
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
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png);;All Files (*)", options=options)
        if file_path:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            # Create a new figure with two subplots (main + summary)
            fig, (ax_main, ax_summary) = plt.subplots(
                2, 1, figsize=(18, 10), sharex=True, 
                gridspec_kw={'height_ratios': [4, 1]}
            )

            # --- Main plot ---
            t0 = self.time.iloc[0]
            t1 = self.time.iloc[-1]
            mask = (self.time >= t0) & (self.time <= t1)
            t = self.time[mask]
            offset = {
                'Body Position': 0.0,
                'Pulse': 1.2,
                'SpO2': 2.4,
                'Airflow': 3.6,
            }

            # Plot signals
            if self.visible_signals.get('Pulse', False):
                pulse = self.pulse_n[mask] * self.scales['Pulse']
                window_size = 10
                if len(pulse) > window_size:
                    pulse = pd.Series(pulse).rolling(window=window_size, min_periods=1, center=True).mean()
                ax_main.plot(t, pulse + offset['Pulse'], label="Pulse", color="red", linewidth=2.0)
            if self.visible_signals.get('SpO2', False):
                spo2 = self.spo2_n[mask] * self.scales['SpO2']
                window_size = 10
                if len(spo2) > window_size:
                    spo2 = pd.Series(spo2).rolling(window=window_size, min_periods=1, center=True).mean()
                ax_main.plot(t, spo2 + offset['SpO2'], label="SpO2", color="green", linewidth=2.0)
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
                flow_display = flow_smooth
                ax_main.plot(t, flow_display + offset['Airflow'], label="Airflow", color="#7ec8e3", linewidth=2.0)

            # Highlight events as vertical colored regions
            if hasattr(self, "detected_events"):
                event_colors = {"CSA": "#ff4d4d", "OSA": "#4da6ff", "HSA": "#ffd966"}
                for start, end, typ in self.detected_events:
                    if end < t0 or start > t1:
                        continue
                    plot_start = max(start, t0)
                    plot_end = min(end, t1)
                    ax_main.axvspan(
                        plot_start, plot_end,
                        color=event_colors.get(typ, "#cccccc"),
                        alpha=0.3,
                        label=typ
                    )
                handles, labels = ax_main.get_legend_handles_labels()
                unique = dict(zip(labels, handles))
                ax_main.legend(unique.values(), unique.keys(), loc="upper right")

            ax_main.set_xlim(t0, t1)
            ax_main.set_ylim(-0.5, 12)
            ax_main.set_title("Main Graph")
            ax_main.grid(True, linestyle='--', alpha=0.5)

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
            # ax_summary.plot(self.time, summary_data, label=summary_signal, color="blue", linewidth=0.7)
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
                # print(f"Most frequent Airflow period (seconds): {mode_interval:.2f}")
            else:
                print("Not enough intervals to determine period.")
        else:
            print("Not enough zero-crossings to determine period.")

    def detect_apnea_events(self):
        """
        Uses block-based logic to detect CSA, OSA, HSA events.
        """
        results = self.calculate_event_counts()
        self.detected_events = []
        for typ in ['CSA', 'OSA', 'HSA']:
            for start, end in results[typ]['windows']:
                self.detected_events.append((start, end, typ))
        # Print counts for all event types, or "Not detected" if zero
        for typ in ['CSA', 'OSA', 'HSA']:
            count = results[typ]['count']
            if count == 0:
                print(f"{typ} not detected")
            else:
                print(f"Total {typ} events in data: {count}")
        # Optionally print for debug
        print("Detected Apnea/Hypopnea Events:")
        for start, end, typ in self.detected_events:
            print(f"{typ} from {start:.1f}s to {end:.1f}s (duration: {end-start:.1f}s)")

    def calculate_event_counts(self):
        btp = self.flow.values.astype(int)
        np.set_printoptions(suppress=True, precision=2)
        time = self.time.values.astype(float)

        vc = pd.Series(btp).value_counts()

        max_btp = int(np.max(btp))
        min_btp = int(np.min(btp))
        max_candidates = [max_btp - i*10 for i in range(8)]
        min_candidates = [min_btp + i*10 for i in range(8)]
        max_cands = [v for v in max_candidates if v in vc.index]
        min_cands = [v for v in min_candidates if v in vc.index]
        best_max = max(max_cands, key=lambda v: vc[v]) if max_cands else max_btp
        best_min = max(min_cands, key=lambda v: vc[v]) if min_cands else min_btp

        btp_modified = np.where(btp > best_max, 0, btp)

        csa_thresh = int(0.10 * best_max + best_min)
        osa_thresh = int(0.50 * best_max + best_min)
        hsa_thresh = int(0.80 * best_max + best_min)

        print(f"Best Max: {best_max}, Best Min: {best_min}")
        print(f"CSA Threshold: {csa_thresh}, OSA Threshold: {osa_thresh}, HSA Threshold: {hsa_thresh}")

        csa_count = osa_count = hsa_count = 0
        csa_windows, osa_windows, hsa_windows = [], [], []

        total_samples = len(btp)

        # 1. CSA: 30-sample window (10s)
        csa_window = 30
        for i in range(0, total_samples, csa_window):
            block = btp_modified[i:i + csa_window]
            # print(f"CSA Block {block}")
            if len(block) < csa_window:
                break
            if np.all((block > best_min) & (block <= csa_thresh)):
                csa_count += 1
                start_ms = round(time[i], 2)
                end_ms   = round(time[i + csa_window - 1], 2)
                csa_windows.append((start_ms, end_ms))

        # 2. OSA: 15-sample window (5s)
        osa_window = 15
        for i in range(0, total_samples, osa_window):
            block = btp_modified[i:i + osa_window]
            # print(f"OSA Block {block}")
            if len(block) < osa_window:
                break
            if np.all((block > csa_thresh) & (block <= osa_thresh)):
                osa_count += 1
                start_ms = round(time[i], 2)
                end_ms   = round(time[i + osa_window - 1], 2)
                osa_windows.append((start_ms, end_ms))

        # 3. HSA: 15-sample window (5s)
        hsa_window = 15
        for i in range(0, total_samples, hsa_window):
            block = btp_modified[i:i + hsa_window]
            # print(f"HSA Block {block}")
            if len(block) < hsa_window:
                break
            if np.all((block > osa_thresh) & (block <= hsa_thresh)):
                hsa_count += 1
                start_ms = round(time[i], 2)
                end_ms   = round(time[i + hsa_window - 1], 2)
                hsa_windows.append((start_ms, end_ms))

        return {
            'CSA': {'count': csa_count, 'windows': csa_windows},
            'OSA': {'count': osa_count, 'windows': osa_windows},
            'HSA': {'count': hsa_count, 'windows': hsa_windows},
        }

    def update_event_count_label(self):
        hsa_count = csa_count = osa_count = 0
        if hasattr(self, "detected_events"):
            for _, _, typ in self.detected_events:
                if typ == "HSA":
                    hsa_count += 1
                elif typ == "CSA":
                    csa_count += 1
                elif typ == "OSA":
                    osa_count += 1
        self.event_count_label.setText(
            f"<b>Event Counts (Full Data):</b><br>"
            f"HSA: {hsa_count}<br>CSA: {csa_count}<br>OSA: {osa_count}"
        )



def print_airflow_and_apnea_ranges(self):
    # Print counts for airflow values 1 to 10
    print("Airflow value counts (1 to 10):")
    for val in range(1, 11):
        count = np.sum(self.flow == val)
        print(f"Airflow value {val}: {count}")

    # Detect CSA, OSA, HSA based on value ranges
    csa_count = np.sum((self.flow >= 1) & (self.flow < 2))
    osa_count = np.sum((self.flow >= 2) & (self.flow < 6))
    hsa_count = np.sum((self.flow >= 6) & (self.flow < 9))
    print(f"\nCSA (1 <= airflow < 2): {csa_count}")
    print(f"OSA (2 <= airflow < 6): {osa_count}")
    print(f"HSA (6 <= airflow < 9): {hsa_count}")


def load_data(self):
    self.data = pd.read_csv(self.file_path, header=None)
    self.time = self.data[0].astype(float) / 1000
    self.body_pos = self.data[1].astype(int)
    self.pulse = self.data[2].astype(float)
    self.spo2 = self.data[3].astype(float)
    self.flow = self.data[7].astype(float)

    # Print airflow value counts and apnea detection by range
    self.print_airflow_and_apnea_ranges()


def detect_apnea_events(self):
    """
    Uses block-based logic to detect CSA, OSA, HSA events.
    """
    results = self.calculate_event_counts()
    self.detected_events = []
    for typ in ['CSA', 'OSA', 'HSA']:
        for start, end in results[typ]['windows']:
            self.detected_events.append((start, end, typ))
    # Print counts for all event types, or "Not detected" if zero
    for typ in ['CSA', 'OSA', 'HSA']:
        count = results[typ]['count']
        if count == 0:
            print(f"{typ} not detected")
        else:
            print(f"Total {typ} events in data: {count}")
    # Optionally print for debug
    print("Detected Apnea/Hypopnea Events:")
    for start, end, typ in self.detected_events:
        print(f"{typ} from {start:.1f}s to {end:.1f}s (duration: {end-start:.1f}s)")

    """
    Uses block-based logic to detect CSA, OSA, HSA events.
    """
    results = self.calculate_event_counts()
    self.detected_events = []
    for typ in ['CSA', 'OSA', 'HSA']:
        for start, end in results[typ]['windows']:
            self.detected_events.append((start, end, typ))
    # Print counts for all event types
    print(f"Total CSA events in data: {results['CSA']['count']}")
    print(f"Total OSA events in data: {results['OSA']['count']}")
    print(f"Total HSA events in data: {results['HSA']['count']}")
    # Optionally print for debug
    print("Detected Apnea/Hypopnea Events:")
    for start, end, typ in self.detected_events:
        print(f"{typ} from {start:.1f}s to {end:.1f}s (duration: {end-start:.1f}s)")

def detect_apnea_events(self):
    # ...existing code...
    for start, end, typ in self.detected_events:
        print(f"{typ} from {start:.1f}s to {end:.1f}s (duration: {end-start:.1f}s)")

    # Update left panel event count label
    self.update_left_event_count_label()

def update_left_event_count_label(self):
    hsa_count = csa_count = osa_count = 0
    if hasattr(self, "detected_events"):
        for _, _, typ in self.detected_events:
            if typ == "HSA":
                hsa_count += 1
            elif typ == "CSA":
                csa_count += 1
            elif typ == "OSA":
                osa_count += 1
    self.left_event_count_label.setText(
        f"<b>Event Counts:</b><br>"
        f"HSA: {hsa_count}<br>CSA: {csa_count}<br>OSA: {osa_count}"
    )
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SleepSensePlot()
    window.show()
    sys.exit(app.exec_())
