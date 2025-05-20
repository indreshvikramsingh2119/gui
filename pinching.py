import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle

class SleepSenseVisualizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data()
        self.normalize_signals()
        self.setup_plot_elements()
        self.init_plot()
        self.connect_events()

    def load_data(self):
        try:
            data = pd.read_csv(self.file_path, header=None)
            self.time = data[0].astype(float) / 1000
            self.body_pos = data[1].astype(int)
            self.pulse = data[2].astype(float)
            self.spo2 = data[3].astype(float)
            self.flow = data[7].astype(float)
        except Exception as e:
            print(f"Failed to load data: {e}")
            raise

    def normalize(self, series):
        return (series - series.min()) / (series.max() - series.min())

    def normalize_signals(self):
        self.body_pos_n = self.normalize(self.body_pos)
        self.pulse_n = self.normalize(self.pulse)
        self.spo2_n = self.normalize(self.spo2)
        self.flow_n = self.normalize(self.flow)

    def setup_plot_elements(self):
        self.arrow_directions = {
            0: (0, 0.5, '↑', 'Up (Supine)'),
            1: (-0.5, 0, '←', 'Left'),
            2: (0.5, 0, '→', 'Right'),
            3: (0, -0.5, '↓', 'Down (Prone)')
        }
        self.offsets = [0, 1.2, 2.4, 3.6]
        self.window_size = 10
        self.airflow_zoom_factor = 1.0

    def init_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        plt.subplots_adjust(bottom=0.15, top=0.85)

        self.line_body, = self.ax.plot(self.time, self.body_pos_n + self.offsets[0], color='black', label='Body Position')
        self.line_pulse, = self.ax.plot(self.time, self.pulse_n + self.offsets[1], color='red', label='Pulse')
        self.line_spo2, = self.ax.plot(self.time, self.spo2_n + self.offsets[2], color='green', label='SpO2')
        self.line_flow, = self.ax.plot(self.time, self.flow_n + self.offsets[3], color='blue', label='Airflow')

        yticks_pos = [np.mean(sig) + offset for sig, offset in zip(
            [self.body_pos_n, self.pulse_n, self.spo2_n, self.flow_n], self.offsets)]
        self.ax.set_yticks(yticks_pos)
        self.ax.set_yticklabels(['Body Position', 'Pulse (BPM)', 'SpO2 (%)', 'Airflow'], fontsize=12)

        self.ax.set_ylim(-0.5, max(self.offsets) + 1)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_title('Sleepsense Plotting with Body Position Arrows')
        self.ax.grid(True, linestyle='--', alpha=0.6)

        self.draw_arrows()
        self.classify_segments()
        self.add_slider_and_buttons()

    def draw_arrows(self):
        arrow_y = self.offsets[0] + 0.5
        interval = 5
        sampling_step = int(1 / (self.time[1] - self.time[0]) * interval)
        arrow_indices = self.time[::sampling_step]

        for t in arrow_indices:
            idx = (np.abs(self.time - t)).argmin()
            pos = self.body_pos.iloc[idx]
            dx, dy, arrow_char, _ = self.arrow_directions.get(pos, (0, 0, '?', 'Unknown'))
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

    def classify_segments(self):
        for i in range(100, len(self.flow), 1000):
            segment = self.flow[i:i+300]
            t = self.time[i]
            if segment.mean() < 0.2:
                self.ax.add_patch(Rectangle((t, self.offsets[3]), 5, 0.5, color='red', alpha=0.3, label="OSA"))
            elif segment.std() < 0.05:
                self.ax.add_patch(Rectangle((t, self.offsets[3]), 5, 0.5, color='orange', alpha=0.3, label="CSA"))
            elif segment.mean() > 0.7 and self.pulse[i:i+300].mean() > 100:
                self.ax.add_patch(Rectangle((t, self.offsets[3]), 5, 0.5, color='purple', alpha=0.3, label="HSA"))

    def add_slider_and_buttons(self):
        start_time = self.time.iloc[0]
        end_time = self.time.iloc[-1]
        self.ax.set_xlim(start_time, start_time + self.window_size)

        slider_ax = plt.axes([0.15, 0.07, 0.7, 0.03])
        self.slider = Slider(slider_ax, 'Time', start_time, end_time - self.window_size, valinit=start_time)
        self.slider.on_changed(self.update_slider)

        window_sizes = [5, 10, 15, 30, 60, 120, 300]
        button_width, button_height = 0.08, 0.04
        spacing, start_x, button_y = 0.01, 0.1, 0.92

        for i, size in enumerate(window_sizes):
            ax_button = plt.axes([start_x + i * (button_width + spacing), button_y, button_width, button_height])
            label = f"{size//60}m" if size >= 60 else f"{size}s"
            button = Button(ax_button, label)
            button.on_clicked(lambda event, s=size: self.change_window_size(s))

        # Zoom Buttons
        self.plus_ax = plt.axes([0.91, 0.2, 0.03, 0.04])
        self.plus_button = Button(self.plus_ax, '+')
        self.plus_button.ax.set_visible(False)

        self.minus_ax = plt.axes([0.91, 0.15, 0.03, 0.04])
        self.minus_button = Button(self.minus_ax, '-')
        self.minus_button.ax.set_visible(False)

        self.plus_button.on_clicked(self.zoom_in)
        self.minus_button.on_clicked(self.zoom_out)

    def update_slider(self, val):
        t = self.slider.val
        self.ax.set_xlim(t, t + self.window_size)
        self.fig.canvas.draw_idle()

    def change_window_size(self, size):
        self.window_size = size
        end_time = self.time.iloc[-1]
        self.slider.valmax = end_time - self.window_size
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
        if self.slider.val > self.slider.valmax:
            self.slider.set_val(self.slider.valmax)
        else:
            self.update_slider(self.slider.val)

    def zoom_in(self, event):
        self.airflow_zoom_factor *= 1.2
        self.update_airflow_plot()

    def zoom_out(self, event):
        self.airflow_zoom_factor /= 1.2
        self.update_airflow_plot()

    def update_airflow_plot(self):
        new_flow = (self.flow_n - 0.5) * self.airflow_zoom_factor + 0.5
        self.line_flow.set_ydata(new_flow + self.offsets[3])
        self.fig.canvas.draw_idle()

    def connect_events(self):
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)

    def on_hover(self, event):
        if event.inaxes == self.ax:
            cont, _ = self.line_flow.contains(event)
            self.plus_button.ax.set_visible(cont)
            self.minus_button.ax.set_visible(cont)
            self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

# --- Run it ---
file_path = r"C:\Users\DELL\Documents\Workday1\sleepsense\DATA2304.TXT"
viz = SleepSenseVisualizer(file_path)
viz.show()
