import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Interactive plot mode on
# plt.ion()
fig, axs = plt.subplots(6, 1, figsize=(12, 14))

time_data, spo2_data, pulse_data, body_pos_data = [], [], [], []
pleth_data, flow_data, snore_data = [], [], []

file_path = r"C:\Users\DELL\Documents\Workday1\sleepsense\DATA1623.TXT"

# Mapping BodyPos to arrows
def bodypos_to_arrow_dir(pos):
    if pos == 10:
        return (0, 1)   # Up
    elif pos == 20:
        return (0, -1)  # Down
    elif pos == 30:
        return (-1, 0)  # Left
    elif pos == 40:
        return (1, 0)   # Right
    else:
        return (0, 0)

highlight_times = [10, 20, 30, 40, 50]

try:
    for chunk in pd.read_csv(file_path, header=None, chunksize=1):
        chunk.columns = [
            "Time", "Pleth", "SpO2", "Pulse", "Flow",
            "BodyPos", "Snore", "C4", "C5", "C6"
        ]

        # Convert and validate values
        try:
            time_val = int(chunk["Time"].values[0])
            spo2_val = float(chunk["SpO2"].values[0])
            pulse_val = float(chunk["Pulse"].values[0])
            bodypos_val = int(chunk["BodyPos"].values[0])
            pleth_val = float(chunk["Pleth"].values[0])
            flow_val = float(chunk["Flow"].values[0])
            snore_val = float(chunk["Snore"].values[0])
        except ValueError:
            continue  # Skip malformed rows

        # Apply filtering
        if not (85 <= spo2_val <= 100) or not (49 <= pulse_val <= 170) or not (5 <= bodypos_val <= 35):
            continue

        # Append filtered data
        time_data.append(time_val)
        spo2_data.append(spo2_val)
        pulse_data.append(pulse_val)
        body_pos_data.append(bodypos_val)
        pleth_data.append(pleth_val)
        flow_data.append(flow_val)
        snore_data.append(snore_val)

        # Clear axes
        for ax in axs:
            ax.cla()

        # SpO2 Plot
        axs[0].plot(time_data, spo2_data, color='blue', marker='o')
        axs[0].set_ylim(85, 100)
        axs[0].set_title("SpO2 over Time")
        axs[0].set_ylabel("SpO2 (%)")

        # Pulse Plot
        axs[1].plot(time_data, pulse_data, color='red', marker='s')
        axs[1].set_ylim(49, 170)
        axs[1].set_title("Pulse over Time")
        axs[1].set_ylabel("Pulse (BPM)")

        # Pleth Wave
        axs[2].plot(time_data, pleth_data, color='purple')
        axs[2].set_title("Pleth Waveform")
        axs[2].set_ylabel("Pleth")

        # Flow Wave
        axs[3].plot(time_data, flow_data, color='orange')
        axs[3].set_title("Flow Waveform")
        axs[3].set_ylabel("Flow")

        # Snoring Wave
        axs[4].plot(time_data, snore_data, color='brown')
        axs[4].set_title("Snoring Waveform")
        axs[4].set_ylabel("Snore")

        # Body Position with Arrow Plot
        x = np.arange(len(body_pos_data))
        y = np.array(body_pos_data)
        U, V = [], []
        for pos in body_pos_data:
            u, v = bodypos_to_arrow_dir(pos)
            U.append(u)
            V.append(v)

        axs[5].quiver(x, y, U, V, angles='xy', scale_units='xy', scale=1.5, color='green')
        axs[5].set_ylim(5, 45)
        axs[5].set_title("Body Position over Time (Arrows)")
        axs[5].set_xlabel("Index")
        axs[5].set_ylabel("BodyPos")

        # Highlight lines
        for ax in axs[:5]:
            for t in highlight_times:
                if t in time_data:
                    ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)

        if len(time_data) > 1:
            for t in highlight_times:
                if t in time_data:
                    idx = time_data.index(t)
                    axs[5].axvline(x=idx, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        # plt.pause(0.2)

except KeyboardInterrupt:
    print("Stopped by Divyansh.")

finally:
    plt.ioff()
    plt.show()
