import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

# File path
file_path = r"C:\Users\DELL\Documents\Workday1\sleepsense\DATA2304.TXT"
data = pd.read_csv(file_path, header=None)

time = data[0].astype(float) / 1000  # Convert ms to seconds
body_pos = data[1].astype(int)
pulse = data[2].astype(float)
spo2 = data[3].astype(float)
flow = data[7].astype(float)

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

body_pos_n = normalize(body_pos)
pulse_n = normalize(pulse)
spo2_n = normalize(spo2)
flow_n = normalize(flow)

arrow_directions = {
    0: (0, 0.5, '↑', 'Up (Supine)'),
    1: (-0.5, 0, '←', 'Left'),
    2: (0.5, 0, '→', 'Right'),
    3: (0, -0.5, '↓', 'Down (Prone)')
}

fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(bottom=0.15, top=0.85)

offsets = [0, 1.2, 2.4, 3.6]
line_body, = ax.plot([], [], color='black', label='Body Position')
line_pulse, = ax.plot([], [], color='red', label='Pulse')
line_spo2, = ax.plot([], [], color='green', label='SpO2')
line_flow, = ax.plot([], [], color='black', label='Airflow')

yticks_pos = [offset + 0.5 for offset in offsets]
yticks_labels = ['Body Position', 'Pulse (BPM)', 'SpO2 (%)', 'Airflow']
ax.set_yticks(yticks_pos)
ax.set_yticklabels(yticks_labels, fontsize=12)
ax.set_ylim(-0.5, max(offsets) + 1)
ax.set_xlabel('Time (s)')
ax.set_title('Live Sleepsense Plotting')

# Initial plot state
window_size = 10
start_time = time.iloc[0]
end_time = time.iloc[-1]
slider_val = [start_time]
ax.set_xlim(start_time, start_time + window_size)

arrow_y = offsets[0] + 0.5
arrow_interval = 5
arrow_annotations = []

# --- Slider ---
slider_ax = plt.axes([0.15, 0.07, 0.7, 0.03])
slider = Slider(slider_ax, 'Time', start_time, end_time - window_size, valinit=start_time)

def update_slider(val):
    slider_val[0] = slider.val
slider.on_changed(update_slider)

# --- Buttons for window size ---
window_sizes = [5, 10, 15, 30, 60, 120, 300]
button_width = 0.08
button_height = 0.04
button_spacing = 0.01
start_x = 0.1
button_y = 0.92

def on_button_clicked(event, size):
    global window_size
    window_size = size
    slider.valmax = end_time - window_size
    slider.ax.set_xlim(slider.valmin, slider.valmax)
    if slider.val > slider.valmax:
        slider.set_val(slider.valmax)
    else:
        update_slider(slider.val)

for i, size in enumerate(window_sizes):
    ax_button = plt.axes([start_x + i * (button_width + button_spacing), button_y, button_width, button_height])
    label = f"{size//60}m" if size >= 60 else f"{size}s"
    button = Button(ax_button, label)
    button.on_clicked(lambda event, s=size: on_button_clicked(event, s))

# --- Playback button and state ---
is_playing = [False]
play_speed = 1.0  # seconds per update

play_ax = plt.axes([0.92, 0.92, 0.06, 0.04])
play_button = Button(play_ax, 'Play')

def toggle_play(event):
    is_playing[0] = not is_playing[0]
    play_button.label.set_text('Pause' if is_playing[0] else 'Play')

play_button.on_clicked(toggle_play)

# --- Animation function ---
def animate(frame):
    if is_playing[0]:
        next_val = slider_val[0] + play_speed
        if next_val <= end_time - window_size:
            slider.set_val(next_val)
        else:
            is_playing[0] = False
            play_button.label.set_text('Play')

    t = slider_val[0]
    mask = (time >= t) & (time <= t + window_size)

    line_body.set_data(time[mask], body_pos_n[mask] + offsets[0])
    line_pulse.set_data(time[mask], pulse_n[mask] + offsets[1])
    line_spo2.set_data(time[mask], spo2_n[mask] + offsets[2])
    line_flow.set_data(time[mask], flow_n[mask] + offsets[3])
    ax.set_xlim(t, t + window_size)

    # Clear previous arrows
    for ann in arrow_annotations:
        ann.remove()
    arrow_annotations.clear()

    arrow_step = int(1 / (time[1] - time[0]) * arrow_interval)
    for idx in range(mask.idxmax() - arrow_step, mask.idxmax(), arrow_step):
        if idx >= len(time):
            break
        pos = body_pos.iloc[idx]
        dx, dy, arrow_char, label = arrow_directions.get(pos, (0, 0, '?', 'Unknown'))
        ann = ax.annotate(
            arrow_char,
            xy=(time.iloc[idx], arrow_y),
            xytext=(time.iloc[idx] + dx, arrow_y + dy),
            fontsize=16,
            color='blue',
            ha='center',
            va='center',
            arrowprops=dict(arrowstyle='->', color='green')
        )
        arrow_annotations.append(ann)

    return line_body, line_pulse, line_spo2, line_flow, *arrow_annotations

ani = FuncAnimation(fig, animate, interval=1000)  # Update every 1000 ms
plt.show()
