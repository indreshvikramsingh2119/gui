import numpy as np
import matplotlib.pyplot as plt

# Data
airflow_values = [10, 7, 5, 4, 3, 2, 1, 8, 6, 9]
occurrences = [9303, 9072, 9196, 9394, 9221, 9286, 9198, 9363, 9403, 9224]

# Calculate weighted average amplitude
total_occurrences = sum(occurrences)
weighted_amplitude = sum([val * occ for val, occ in zip(airflow_values, occurrences)]) / total_occurrences

# Assume a base frequency (or use data to calculate one)
period = 0.64  # seconds (from your earlier data)
frequency = 1 / period

# Time range for plotting
time = np.linspace(0, 2, 1000)  # 2 seconds, 1000 points

# Generate sine wave
amplitude = weighted_amplitude
wave = amplitude * np.sin(2 * np.pi * frequency * time)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(time, wave, label=f"Sine Wave (A={amplitude:.2f}, f={frequency:.2f} Hz)")
plt.title("Sine Wave Derived from Airflow Data")
plt.xlabel("Time (s)")
plt.ylabel("Airflow Value")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.show()
